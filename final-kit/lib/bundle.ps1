# PowerShell bundle I/O. The entry point sets Kit, Scratch and ErrorActionPreference.
$Utf8 = [Text.UTF8Encoding]::new($false)
$JqPath = (Get-Command jq -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source

function New-ProcessInfo([string]$Program, [string[]]$Arguments) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Program
    $info.UseShellExecute = $false
    $info.WorkingDirectory = (Get-Location).ProviderPath
    $info.RedirectStandardOutput = $true
    $info.StandardOutputEncoding = $Utf8
    foreach ($argument in $Arguments) { $info.ArgumentList.Add($argument) }
    return $info
}

function Invoke-Jq([string[]]$Arguments, [string]$OutputFile, [switch]$Append) {
    # ArgumentList preserves literal JSON quotes on every PowerShell 7 platform.
    $process = [Diagnostics.Process]::Start((New-ProcessInfo $JqPath $Arguments))
    try {
        if ($OutputFile) {
            $mode = if ($Append) { 'Append' } else { 'Create' }
            $stream = [IO.File]::Open($OutputFile, $mode, 'Write')
            try { $process.StandardOutput.BaseStream.CopyTo($stream) } finally { $stream.Dispose() }
        } else {
            $output = $process.StandardOutput.ReadToEnd().TrimEnd([char[]]"`r`n")
        }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw 'jq validation or transformation failed' }
        if (!$OutputFile) { return $output }
    } finally {
        if (!$process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
}

function Get-Digest([string]$Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-Directory([string]$Path) {
    $entry = Get-Item -Force -LiteralPath $Path
    if (!$entry.PSIsContainer -or ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected a directory at $Path"
    }
}

function Initialize-Directory([string]$Path) {
    $null = [IO.Directory]::CreateDirectory($Path)
    Assert-Directory $Path
}

function New-PrivateDirectory([string]$Path) {
    # Exclusive reservation. A losing concurrent attempt must never write here.
    $null = New-Item -Path $Path -ItemType Directory -ErrorAction Stop
    if (!$IsWindows) {
        & chmod 700 $Path
        if ($LASTEXITCODE -ne 0) { throw "Cannot protect directory at $Path" }
    }
}

function Get-TreeFiles([string]$Root) {
    Assert-Directory $Root
    foreach ($entry in (Get-ChildItem -LiteralPath $Root -Recurse -Force | Sort-Object FullName)) {
        if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            $entry.Name -match '[\x00-\x1f\x7f\\]') { throw 'Unsupported bundle entry' }
        if (!$entry.PSIsContainer) {
            if (!$IsWindows -and $entry.UnixStat.ItemType -ne 'File') { throw 'Expected a regular file' }
            $entry
        }
    }
}

function Assert-Exportable([string]$Path) {
    $pattern = '(sk-|hf_|ghp_|ghs_|github_pat_)[A-Za-z0-9_-]{16,}|-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----'
    foreach ($line in [IO.File]::ReadLines($Path)) {
        if ($line -cmatch $pattern) { throw 'Export value scan failed' }
    }
    if ($Path -match '\.jsonl?$') {
        $null = Invoke-Jq @('-ne', '-f', "$Kit/lib/privacy.jq", $Path)
    }
}

function Get-Checksums([string]$Root, [switch]$Tiers) {
    foreach ($file in (Get-TreeFiles $Root)) {
        $name = [IO.Path]::GetRelativePath($Root, $file.FullName).Replace('\', '/')
        if ($Tiers) {
            if ($name -cnotlike '*.json') { continue }
        } else {
            if ($name -ceq 'SHA256SUMS') { continue }
            Assert-Exportable $file.FullName
        }
        "$(Get-Digest $file.FullName)  $name"
    }
}

function Assert-Seal([string]$Root, [switch]$Tiers) {
    # Match exact on-disk paths and hashes. Never resolve a path supplied by a seal.
    $actual = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($line in (Get-Checksums $Root -Tiers:$Tiers)) { $null = $actual.Add($line) }
    if ($actual.Count -eq 0) { throw 'Empty inventory' }
    foreach ($line in [IO.File]::ReadLines("$Root/SHA256SUMS")) {
        if (!$actual.Remove($line)) { throw 'Invalid or duplicate seal entry' }
    }
    if ($actual.Count -ne 0) { throw 'Incomplete seal' }
}

function Publish-Seal([string]$Root) {
    $lines = @(Get-Checksums $Root)
    [IO.File]::WriteAllText("$Scratch/checksums", ($lines -join "`n") + "`n", $Utf8)
    [IO.File]::Move("$Scratch/checksums", "$Root/SHA256SUMS")
}

function Invoke-Evaluator([string]$Program, [string[]]$Arguments, [string]$Root) {
    $info = New-ProcessInfo $Program $Arguments
    $info.WorkingDirectory = $Root
    $info.RedirectStandardError = $true
    $info.Environment['HF_HOME'] = "$Kit/.cache/hf"
    $info.Environment['XDG_CACHE_HOME'] = "$Kit/.cache"
    $info.Environment['TMPDIR'] = "$Kit/.tmp"
    $info.Environment['MPLCONFIGDIR'] = "$Kit/.cache/matplotlib"
    $info.Environment['PYTHONDONTWRITEBYTECODE'] = '1'
    $info.Environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    $info.Environment['HF_HUB_DISABLE_TELEMETRY'] = '1'
    $info.Environment['DO_NOT_TRACK'] = '1'
    $log = [IO.File]::Create("$Root/run.log")
    try {
        $errors = [IO.File]::Create("$Scratch/stderr")
        try {
            $process = [Diagnostics.Process]::Start($info)
            try {
                # Drain both pipes concurrently to disk, including during cancellation.
                $stdout = $process.StandardOutput.BaseStream.CopyToAsync($log)
                $stderr = $process.StandardError.BaseStream.CopyToAsync($errors)
                [IO.File]::WriteAllText("$Root/lm-eval.pid", "$($process.Id)`n", $Utf8)
                while (!$process.WaitForExit(200)) { }
                return $process.ExitCode
            } finally {
                if (!$process.HasExited) { $process.Kill(); $process.WaitForExit() }
                try {
                    $stdout.GetAwaiter().GetResult() | Out-Null
                    $stderr.GetAwaiter().GetResult() | Out-Null
                } finally { $process.Dispose() }
            }
        } finally {
            $errors.Position = 0
            try { $errors.CopyTo($log) } finally { $errors.Dispose() }
        }
    } finally { $log.Dispose() }
}
