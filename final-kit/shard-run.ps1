#requires -Version 7
[CmdletBinding(DefaultParameterSetName='List')]
param(
    [Parameter(Position=0,Mandatory)][string]$Task,
    [Parameter(Position=1,Mandatory)][string]$Label,
    [Parameter(Position=2,Mandatory)][string]$Endpoint,
    [Parameter(Position=3,Mandatory)][string]$Shard,
    [Parameter(Position=4,Mandatory,ParameterSetName='List')]
    [Parameter(ParameterSetName='Tier')][string]$Selection,
    [Parameter(Mandatory,ParameterSetName='Tier')][string]$Tier,
    [Parameter(Position=5,ParameterSetName='List')][string]$TierName,
    [Parameter(Position=4,Mandatory,ParameterSetName='Tier')]
    [Parameter(Position=6,ParameterSetName='List')][int]$Start,
    [Parameter(Position=5,Mandatory,ParameterSetName='Tier')]
    [Parameter(Position=7,ParameterSetName='List')][int]$Count
)
$ErrorActionPreference = 'Stop'
$Kit = $PSScriptRoot
. "$Kit/lib/bundle.ps1"

function New-Request([string]$Spec, [int]$Offset, [int]$Length) {
    $profilePath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:PROFILE_FILE)
    [IO.File]::Copy($profilePath, "$Scratch/lineage.json")
    [IO.File]::Copy($Spec, "$Scratch/selection.json")
    Assert-Exportable "$Scratch/selection.json"
    Assert-Exportable "$Scratch/lineage.json"
    $null = Invoke-Jq @('-es', '-f', "$Kit/lib/lineage.jq", "$Scratch/lineage.json")
    if ($Length -eq 0) {
        $Length = [int](Invoke-Jq @('-e', '.indices | length', "$Scratch/selection.json"))
    }
    Invoke-Jq @('-n', '--arg', 'task', $Task, '--argjson', 'start', "$Offset",
        '--argjson', 'count', "$Length", '--slurpfile', 'spec', "$Scratch/selection.json",
        '--slurpfile', 'pins', "$Kit/task-pins.json", '--slurpfile', 'strata', "$Kit/tiers/strata.json",
        '-f', "$Kit/lib/select.jq") -OutputFile "$Scratch/units.json"

    $sources = @{}
    $files = @(Get-ChildItem -LiteralPath "$Kit/tasks" -Filter '*.yaml' -File) +
             @(Get-ChildItem -LiteralPath "$Kit/lib" -File) +
             @(Get-Item -LiteralPath "$Kit/backend.json","$Kit/task-pins.json","$Kit/shard-run.sh","$Kit/shard-run.ps1")
    foreach ($file in $files) {
        $relative = [IO.Path]::GetRelativePath($Kit, $file.FullName).Replace('\', '/')
        $sources[$relative] = Get-Digest $file.FullName
    }
    [IO.File]::WriteAllText("$Scratch/source-hashes.json", ($sources | ConvertTo-Json) + "`n", $Utf8)
    Invoke-Jq @('-n', '--arg', 'task', $Task, '--arg', 'label', $Label, '--arg', 'url', $Endpoint,
        '--arg', 'shard', $Shard, '--arg', 'model', $env:MODEL_ID, '--argjson', 'seed', "$Seed",
        '--arg', 'time', [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'),
        '--arg', 'selection', (Get-Digest "$Scratch/selection.json"),
        '--slurpfile', 'units', "$Scratch/units.json", '--slurpfile', 'lin', "$Scratch/lineage.json",
        '--slurpfile', 'sources', "$Scratch/source-hashes.json", '--slurpfile', 'pins', "$Kit/task-pins.json",
        '--slurpfile', 'backend', "$Kit/backend.json", '-f', "$Kit/lib/manifest.jq") -OutputFile "$Scratch/manifest.json"
}

function Start-Shard {
    Invoke-Jq @('group_by(.task) | map({key:.[0].task,value:map(.index)}) | from_entries',
        "$Bundle/units.json") -OutputFile "$Bundle/samples.json"
    Invoke-Jq @('--arg', 'model', $env:MODEL_ID, '--arg', 'url', $Endpoint, '--argjson', 'seed', "$Seed",
        '.model_args | .model=$model | .base_url=$url | .seed=$seed', "$Kit/backend.json") -OutputFile "$Bundle/model-args.json"
    Invoke-Jq @('.generation_kwargs', "$Kit/backend.json") -OutputFile "$Bundle/generation.json"
    $tasks = if ($Task -ceq 'card-math') { 'card_math' } else {
        Invoke-Jq @('-r', 'keys | map("day1_" + .) | join(",")', "$Bundle/samples.json")
    }
    $arguments = @('-m', 'lm_eval', 'run', '--model', 'local-chat-completions',
        '--model_args', [IO.File]::ReadAllText("$Bundle/model-args.json"),
        '--gen_kwargs', [IO.File]::ReadAllText("$Bundle/generation.json"), '--tasks', $tasks,
        '--include_path', "$Kit/tasks", '--apply_chat_template', '--batch_size', '1',
        '--seed', "0,1234,1234,$Seed", '--samples', 'samples.json',
        '--log_samples', '--output_path', "$Bundle/native")
    Invoke-Evaluator $Python $arguments $Bundle
}

function Write-Items {
    $rows = "$Bundle/native-rows.jsonl"
    $paths = "$Bundle/native-files.relative.txt"
    foreach ($path in $rows, $paths, "$Bundle/items.jsonl") { [IO.File]::WriteAllText($path, '') }
    foreach ($file in (Get-TreeFiles "$Bundle/native")) {
        if ($file.Name -cnotlike 'samples_*.jsonl') { continue }
        $nativeTask = $file.BaseName -creplace '^samples_','' -creplace '_[0-9]{4}-.*$',''
        $relative = [IO.Path]::GetRelativePath($Bundle, $file.FullName).Replace('\', '/')
        [IO.File]::AppendAllText($paths, "$relative`n", $Utf8)
        Invoke-Jq @('-c', '--arg', 'task', $nativeTask, '. + {native_task:$task}', $file.FullName) -OutputFile $rows -Append
    }
    Invoke-Jq @('-nc', '--slurpfile', 'units', "$Bundle/units.json", '-f', "$Kit/lib/items.jq", $rows) -OutputFile "$Bundle/items.jsonl"
    $null = Invoke-Jq @('-es', '--slurpfile', 'units', "$Bundle/units.json",
        'map([.task,.index]) | sort == ($units[0] | map([.task,.index]) | sort)', "$Bundle/items.jsonl")
}

function Complete-Shard([int]$Status) {
    try { Write-Items } catch {
        [Console]::Error.WriteLine("Incomplete native output: $_")
        if ($Status -eq 0) { $Status = 65 }
    }
    $manifest = "$Bundle/manifest.json"
    Invoke-Jq @('--arg', 'time', [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'),
        '--argjson', 'rc', "$Status", '.utc_end=$time | .exit_status=$rc | .state=(if $rc==0 then "complete-unqualified" else "failed" end)',
        $manifest) -OutputFile "$Scratch/final.json"
    [IO.File]::Move("$Scratch/final.json", $manifest, $true)
    try { Publish-Seal $Bundle } catch {
        [Console]::Error.WriteLine("Export refused: $_")
        Invoke-Jq @('.state="export-refused" | .exit_status=65', $manifest) -OutputFile "$Scratch/final.json"
        [IO.File]::Move("$Scratch/final.json", $manifest, $true)
        return 65
    }
    return $Status
}

if ($Task -cnotin 'minerva_math500','mmlu_pro','card-math') {
    if ($Task -cnotlike 'mmlu_pro_*') { throw 'Unsupported task' }
    $null = Invoke-Jq @('-e', '--arg', 'task', $Task, 'has($task)', "$Kit/task-pins.json")
}
$idPattern = '^[A-Za-z0-9][A-Za-z0-9_.-]*\z'
if ($Label -cnotmatch $idPattern -or $Shard -cnotmatch $idPattern) { throw 'Invalid label/shard' }
if ($Endpoint -cnotmatch '^http://(127\.0\.0\.1|localhost|\[::1\]):([0-9]{1,5})/v1/chat/completions\z') {
    throw 'Local full endpoint URL required'
}
$port = [int]$Matches[2]
if ($port -lt 1 -or $port -gt 65535) { throw 'Invalid endpoint port' }
if ($env:MODEL_ID -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_./:-]*\z' -or !$env:PROFILE_FILE) {
    throw 'MODEL_ID and PROFILE_FILE are required'
}
$Seed = 1234L
if ($env:DRAW_SEED) {
    if ($env:DRAW_SEED -notmatch '^[0-9]{1,10}\z' -or [long]$env:DRAW_SEED -gt 4294967295) { throw 'Invalid seed' }
    $Seed = [long]$env:DRAW_SEED
}
$Python = $env:PILOT_PYTHON
if (!$Python) {
    $Python = if ($IsWindows) { "$Kit/.venv/Scripts/python.exe" } else { "$Kit/.venv/bin/python" }
}
$Python = (Get-Command -Name $Python -CommandType Application | Select-Object -First 1).Source
$version = & $Python -m pip show lm_eval
if ($LASTEXITCODE -ne 0 -or 'Version: 0.4.13' -cnotin $version) { throw 'lm_eval==0.4.13 required' }

Initialize-Directory "$Kit/.tmp"
$Scratch = "$Kit/.tmp/shard.$([guid]::NewGuid())"
New-PrivateDirectory $Scratch
try {
    # -File binds --tier as a named switch; an argument array passes it literally.
    if ($TierName) { $Tier = $TierName }
    if ($Tier -or $Selection -ceq '--tier') {
        if (($Selection -and $Selection -cne '--tier') -or $Task -cnotin 'minerva_math500','mmlu_pro' -or
            $Tier -cnotmatch '^n[0-9]+\z' -or $Start -lt 0 -or $Count -lt 1) { throw 'Invalid tier/range' }
        Assert-Seal "$Kit/tiers" -Tiers
        New-Request "$Kit/tiers/$Task/$Tier.json" $Start $Count
    } else {
        if ($PSBoundParameters.ContainsKey('Start') -or $PSBoundParameters.ContainsKey('Count')) { throw 'Unexpected tier range' }
        New-Request ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Selection)) 0 0
    }

    foreach ($dir in "$Kit/bundles", "$Kit/bundles/$Task", "$Kit/bundles/$Task/$Label") { Initialize-Directory $dir }
    $Bundle = "$Kit/bundles/$Task/$Label/$Shard"
    if (Test-Path -LiteralPath $Bundle) {
        Assert-Seal $Bundle
        $null = Invoke-Jq @('-es', '--slurpfile', 'requested', "$Scratch/manifest.json", '-f',
            "$Kit/lib/resume.jq", "$Bundle/manifest.json")
        Write-Output 'SKIP: matching complete sealed bundle'
        exit 0
    }
    foreach ($dir in "$Kit/.cache", "$Kit/.cache/hf", "$Kit/.cache/matplotlib") { Initialize-Directory $dir }
    # Move a populated directory into place so two simultaneous reservations cannot both succeed.
    $prepared = "$Scratch/attempt"
    New-PrivateDirectory $prepared
    foreach ($name in 'units.json','source-hashes.json','manifest.json') { [IO.File]::Move("$Scratch/$name", "$prepared/$name") }
    [IO.Directory]::Move($prepared, $Bundle)
    $status = 143
    try { $status = Start-Shard } catch {
        [Console]::Error.WriteLine("Evaluator failed: $_")
        $status = 65
    } finally { $status = Complete-Shard $status }
    [Console]::Error.WriteLine("bundle=$Bundle exit=$status")
    exit $status
} finally {
    [IO.Directory]::Delete($Scratch, $true)
}
