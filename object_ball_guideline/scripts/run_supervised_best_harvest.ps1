param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WslScript = "$RepoRoot/scripts/run_supervised_best_harvest_wsl.sh"

if (-not (Test-Path $WslScript)) {
    throw "Missing script: $WslScript"
}

function Convert-ToWslArg {
    param(
        [string]$Value
    )

    if ($Value -match '^[A-Za-z]:\\') {
        $drive = $Value.Substring(0, 1).ToLowerInvariant()
        $rest = $Value.Substring(2).Replace('\', '/')
        return "/mnt/$drive$rest"
    }
    return $Value
}

$Joined = [string]::Join(' ', ($ArgsList | ForEach-Object {
    $Value = Convert-ToWslArg $_
    '"' + ($Value -replace '"', '\"') + '"'
}))

wsl bash -lc "cd /mnt/c/My_Project/Test_New_Approach && bash scripts/run_supervised_best_harvest_wsl.sh $Joined"
