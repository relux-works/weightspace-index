# PowerShell 7 parity candidate. UNVERIFIED: no PowerShell executable on build host.
# Same jq selection/manifest definitions, upstream CLI and bundle schema as Bash.
[CmdletBinding()]
param([Parameter(Position=0,Mandatory)][string]$Task,
 [Parameter(Position=1,Mandatory)][string]$Label,
 [Parameter(Position=2,Mandatory)][string]$Endpoint,
 [Parameter(Position=3,Mandatory)][string]$Shard,
 [Parameter(Position=4,Mandatory)][string]$Selection,
 [Parameter(Position=5)][string]$Tier,
 [Parameter(Position=6)][int]$Start=0,
 [Parameter(Position=7)][int]$Count=0)
$ErrorActionPreference='Stop'
$Kit=$PSScriptRoot
$Utf8=[Text.UTF8Encoding]::new($false)
function Write-Utf8([string]$Path,[string]$Text) { [IO.File]::WriteAllText($Path,$Text+"`n",$Utf8) }
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Jq([string[]]$Arguments) {
 $out=& jq @Arguments
 if ($LASTEXITCODE -ne 0) { throw 'REFUSED: jq validation failed' }
 return ($out -join "`n")
}
function Scan-File([string]$Path) {
 $text=[IO.File]::ReadAllText($Path)
 if ($text -cmatch '(sk-|hf_|ghp_|ghs_|github_pat_)[A-Za-z0-9_-]{16,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----') { throw 'REFUSED: export value scan' }
 if ($Path -match '\.jsonl?$') {
  $filter='all(.. | objects | to_entries[]; ((.key|ascii_downcase|gsub("-";"_"))|test("(^|_)(api_key|apikey|access_token|auth_token|refresh_token|authorization|password|passwd|secret|private_key|client_secret|token|secret_key|api_token|bearer_token|session_token|hf_token|github_token|credentials|credential|privatekey|clientsecret)$")|not) or (.value==null or .value=="" or .value==false or .value==[] or .value=={}))'
  $null=Jq -Arguments @('-e','-s',$filter,$Path)
 }
}
function Files([string]$Root) {
 $entries=@(Get-ChildItem -LiteralPath $Root -Recurse -Force)
 if (@($entries|Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) { throw 'REFUSED: link in bundle' }
 @($entries|Where-Object { -not $_.PSIsContainer }|Sort-Object FullName)
}
function Seal([string]$Root) {
 $lines=@()
 foreach ($f in (Files $Root)) { Scan-File $f.FullName; $rel=[IO.Path]::GetRelativePath($Root,$f.FullName).Replace('\','/'); $lines+=((Hash $f.FullName)+'  '+$rel) }
 if (Test-Path -LiteralPath "$Root/SHA256SUMS") { throw 'REFUSED: existing seal' }
 Write-Utf8 "$Root/SHA256SUMS" ($lines -join "`n")
}
function Verify([string]$Root) {
 $seen=@{}
 foreach ($line in [IO.File]::ReadAllLines("$Root/SHA256SUMS")) {
  if ($line -notmatch '^([a-f0-9]{64})  (.+)$') { throw 'REFUSED: invalid seal' }
  $digest=$Matches[1]; $rel=$Matches[2]
  if ($rel -match '(^/|(^|/)\.\.(/|$)|\\|:)' -or $rel -eq 'SHA256SUMS' -or $seen.ContainsKey($rel)) { throw 'REFUSED: seal path' }
  if ((Hash "$Root/$rel") -ne $digest) { throw 'REFUSED: seal hash' }; $seen[$rel]=$true
 }
 $actual=@(Files $Root|Where-Object { $_.Name -ne 'SHA256SUMS' })
 if ($actual.Count -ne $seen.Count) { throw 'REFUSED: seal inventory' }
 foreach ($f in $actual) { Scan-File $f.FullName; $rel=[IO.Path]::GetRelativePath($Root,$f.FullName).Replace('\','/'); if (-not $seen.ContainsKey($rel)) { throw 'REFUSED: unsealed file' } }
}
if ($Task -notin @('minerva_math500','mmlu_pro','card-math')) {
 $null=Jq -Arguments @('-e','--arg','t',$Task,'has($t)',"$Kit/task-pins.json")
 if (-not $Task.StartsWith('mmlu_pro_')) { throw 'REFUSED: unsupported task' }
}
if ($Label -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$' -or $Shard -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') { throw 'REFUSED: unsafe label/shard' }
if ($Endpoint -cnotmatch '^http://(127\.0\.0\.1|localhost|\[::1\]):[0-9]+/v1/chat/completions$') { throw 'REFUSED: endpoint' }
if ($env:MODEL_ID -cnotmatch '^[A-Za-z0-9][A-Za-z0-9_./:-]*$' -or -not $env:PROFILE_FILE) { throw 'REFUSED: MODEL_ID/PROFILE_FILE required' }
$Py=$env:PILOT_PYTHON
if (-not $Py) { if ($IsWindows) { $Py="$Kit/.venv/Scripts/python.exe" } else { $Py="$Kit/.venv/bin/python" } }
$info=& $Py -m pip show lm_eval 2>$null
if ($LASTEXITCODE -ne 0 -or 'Version: 0.4.13' -notin $info) { throw 'REFUSED: lm_eval==0.4.13 required' }
$Seed=1234
if ($env:DRAW_SEED) { if ($env:DRAW_SEED -notmatch '^\d+$' -or [double]$env:DRAW_SEED -gt 4294967295) { throw 'REFUSED: seed' }; $Seed=[long]$env:DRAW_SEED }
Scan-File $env:PROFILE_FILE
$lineage=[IO.File]::ReadAllText($env:PROFILE_FILE)|ConvertFrom-Json
$required=@('reference_config','base_weights_digest','patch_digest','tokenizer_digest','template_digest','quantization','conversion_tool_versions','patch_quantization_order','engine_version_observed','plugin_version_observed','machine_layout_evidence')
foreach ($k in $required) { if (-not $lineage.$k -or $lineage.$k -match 'REQUIRED') { throw 'REFUSED: incomplete lineage' } }
if ($lineage.reference_config -ne 'qwen35-0.8b-vllm-metal-bf16-2x12-v1' -or $lineage.quantization -ne 'bf16') { throw 'REFUSED: nonreference config' }
foreach ($k in @('base_weights_digest','tokenizer_digest','template_digest')) { if ($lineage.$k -cnotmatch '^[a-f0-9]{64}$') { throw 'REFUSED: missing digest' } }
if ($Selection -eq '--tier') {
 if ($Task -eq 'card-math' -or $Tier -cnotmatch '^n[0-9]+$' -or $Start -lt 0 -or $Count -lt 1) { throw 'REFUSED: tier/range' }
 foreach ($line in [IO.File]::ReadAllLines("$Kit/tiers/SHA256SUMS")) {
  if ($line -notmatch '^([a-f0-9]{64})  (.+)$' -or (Hash "$Kit/tiers/$($Matches[2])") -ne $Matches[1]) { throw 'REFUSED: frozen tier hash' }
 }
 $Spec="$Kit/tiers/$Task/$Tier.json"
} else { $Spec=$Selection; $Start=0; $Count=[int](Jq -Arguments @('.indices|length',$Spec)) }
Scan-File $Spec
$Units=Jq -Arguments @('-n','--arg','task',$Task,'--argjson','start',"$Start",'--argjson','count',"$Count",'--slurpfile','spec',$Spec,'--slurpfile','pins',"$Kit/task-pins.json",'--slurpfile','strata',"$Kit/tiers/strata.json",'-f',"$Kit/lib/select.jq")
$Protocol='lm_eval-0.4.13-qwen35-text-thinking-32k-math500-mmlupro-tiers-v2'
if ($Task -eq 'card-math') { $Protocol='qwen35-0.8b-card-replication-32k-v1' }
$Bundle="$Kit/bundles/$Task/$Label/$Shard"
if (Test-Path -LiteralPath $Bundle) {
 Verify $Bundle
 $filter='.state=="complete-unqualified" and .exit_status==0 and .sample_units==$u and .protocol_id==$p and .model_label==$l and .served_model==$model and .endpoint==$url and .request_seed==$seed and .lineage==$lin[0] and .sampling==$backend[0].generation_kwargs'
 $null=Jq -Arguments @('-e','--argjson','u',$Units,'--arg','p',$Protocol,'--arg','l',$Label,'--arg','model',$env:MODEL_ID,'--arg','url',$Endpoint,'--argjson','seed',"$Seed",'--slurpfile','lin',$env:PROFILE_FILE,'--slurpfile','backend',"$Kit/backend.json",$filter,"$Bundle/manifest.json")
 Write-Output 'SKIP: matching complete sealed bundle'; exit 0
}
$null=New-Item -ItemType Directory -Force -Path "$Kit/bundles/$Task/$Label","$Kit/.cache/hf","$Kit/.tmp"
$null=New-Item -ItemType Directory -Path $Bundle # exclusive reservation, no Force
$env:HF_HOME="$Kit/.cache/hf"; $env:XDG_CACHE_HOME="$Kit/.cache"; $env:TMPDIR="$Kit/.tmp"
$env:MPLCONFIGDIR="$Kit/.cache/matplotlib"; $env:PYTHONDONTWRITEBYTECODE='1'
$env:HF_HUB_DISABLE_IMPLICIT_TOKEN='1'; $env:HF_HUB_DISABLE_TELEMETRY='1'; $env:DO_NOT_TRACK='1'
Write-Utf8 "$Bundle/units.json" $Units
Write-Utf8 "$Bundle/samples.json" (Jq -Arguments @('group_by(.task)|map({key:.[0].task,value:map(.index)})|from_entries',"$Bundle/units.json"))
Write-Utf8 "$Bundle/model-args.json" (Jq -Arguments @('--arg','model',$env:MODEL_ID,'--arg','url',$Endpoint,'--argjson','seed',"$Seed",'.model_args|.model=$model|.base_url=$url|.seed=$seed',"$Kit/backend.json"))
Write-Utf8 "$Bundle/generation.json" (Jq -Arguments @('.generation_kwargs',"$Kit/backend.json"))
$source=@{}
$sourceFiles=@(Get-ChildItem "$Kit/tasks/*.yaml")+@(Get-ChildItem "$Kit/lib/*")+@(Get-Item "$Kit/backend.json","$Kit/task-pins.json","$Kit/shard-run.sh","$PSCommandPath")
foreach ($f in $sourceFiles) { $source[[IO.Path]::GetRelativePath($Kit,$f.FullName).Replace('\','/')]=Hash $f.FullName }
Write-Utf8 "$Bundle/source-hashes.json" ($source|ConvertTo-Json -Depth 30)
Write-Utf8 "$Bundle/manifest.json" (Jq -Arguments @('-n','--arg','task',$Task,'--arg','label',$Label,'--arg','url',$Endpoint,'--arg','shard',$Shard,'--arg','model',$env:MODEL_ID,'--arg','protocol',$Protocol,'--arg','time',([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')),'--arg','selection',(Hash $Spec),'--argjson','seed',"$Seed",'--slurpfile','units',"$Bundle/units.json",'--slurpfile','lin',$env:PROFILE_FILE,'--slurpfile','backend',"$Kit/backend.json",'--slurpfile','sources',"$Bundle/source-hashes.json",'--slurpfile','pins',"$Kit/task-pins.json",'-f',"$Kit/lib/manifest.jq"))
$Tasks=if ($Task -eq 'card-math') { 'card_math' } else { Jq -Arguments @('-r','keys|map("day1_"+.)|join(",")',"$Bundle/samples.json") }
$RunArgs=@('-m','lm_eval','run','--model','local-chat-completions','--model_args',[IO.File]::ReadAllText("$Bundle/model-args.json"),'--gen_kwargs',[IO.File]::ReadAllText("$Bundle/generation.json"),'--tasks',$Tasks,'--include_path',"$Kit/tasks",'--apply_chat_template','--batch_size','1','--seed',"0,1234,1234,$Seed",'--samples',[IO.File]::ReadAllText("$Bundle/samples.json"),'--log_samples','--output_path',"$Bundle/native")
$Process=$null; $Started=$false; $stdout=$null; $stderr=$null; $Rc=143
try {
 $psi=[Diagnostics.ProcessStartInfo]::new(); $psi.FileName=$Py; $psi.UseShellExecute=$false
 $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
 foreach ($arg in $RunArgs) { $psi.ArgumentList.Add($arg) }
 $Process=[Diagnostics.Process]::new(); $Process.StartInfo=$psi; $Started=$Process.Start()
 Write-Utf8 "$Bundle/lm-eval.pid" "$($Process.Id)"
 $stdout=$Process.StandardOutput.ReadToEndAsync(); $stderr=$Process.StandardError.ReadToEndAsync()
 while (-not $Process.WaitForExit(200)) { } # Ctrl-C must enter finally; qualify on target.
 $Rc=$Process.ExitCode
} finally {
 if ($Started -and $Process -and -not $Process.HasExited) { $Process.Kill(); $Process.WaitForExit() }
 if ($null -ne $stdout -and $null -ne $stderr) { Write-Utf8 "$Bundle/run.log" (($stdout.GetAwaiter().GetResult())+($stderr.GetAwaiter().GetResult())) }
 $rows=@(); $paths=@()
 if (Test-Path "$Bundle/native") {
  foreach ($f in (Get-ChildItem "$Bundle/native" -Recurse -Filter 'samples_*.jsonl' | Sort-Object FullName)) {
   $nt=$f.BaseName -replace '^samples_','' -replace '_[0-9]{4}-.*$',''
   $rows+=Jq -Arguments @('-c','--arg','t',$nt,'.+{native_task:$t}',$f.FullName)
   $paths+=[IO.Path]::GetRelativePath($Bundle,$f.FullName).Replace('\','/')
  }
 }
 Write-Utf8 "$Bundle/native-rows.jsonl" ($rows -join "`n")
 Write-Utf8 "$Bundle/native-files.relative.txt" ($paths -join "`n")
 Write-Utf8 "$Bundle/items.jsonl" (Jq -Arguments @('-c','--slurpfile','units',"$Bundle/units.json",'. as $r | $units[0][] | select(.task==$r.native_task and .index==$r.doc_id) | .+{prompt_sha256:$r.prompt_hash,doc_sha256:$r.doc_hash,finish_reason:null,prompt_tokens:null,completion_tokens:null,total_tokens:null}',"$Bundle/native-rows.jsonl"))
 try { $null=Jq -Arguments @('-e','-s','--slurpfile','units',"$Bundle/units.json",'(length==($units[0]|length)) and (map([.task,.index])|unique|length)==length and (all(.[];(.prompt_sha256|test("^[a-f0-9]{64}$")) and (.doc_sha256|test("^[a-f0-9]{64}$"))))',"$Bundle/items.jsonl") } catch { $Rc=65 }
 Write-Utf8 "$Bundle/manifest.json" (Jq -Arguments @('--arg','time',([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')),'--argjson','rc',"$Rc",'.utc_end=$time|.exit_status=$rc|.state=(if $rc==0 then "complete-unqualified" else "failed" end)',"$Bundle/manifest.json"))
 try { Seal $Bundle; Verify $Bundle } catch {
  Write-Utf8 "$Bundle/manifest.json" (Jq -Arguments @('.state="export-refused"|.exit_status=65',"$Bundle/manifest.json")); $Rc=65
 }
}
Write-Output "bundle=$Bundle exit=$Rc"; exit $Rc
