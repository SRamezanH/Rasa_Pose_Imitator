<#
.SYNOPSIS
    Updates code on Ubuntu PC from local Windows repository
.DESCRIPTION
    1. Creates a bundle of local changes not on Ubuntu
    2. Copies bundle to Ubuntu via SCP
    3. Applies changes on Ubuntu via SSH
    4. Cleans up temporary files
#>

param (
    [string]$UbuntuUser = "cedra",
    [string]$UbuntuHost = "172.27.72.227",
    [string]$RemoteRepoPath = "~/psl_project/",
    [string]$LocalRepoPath = (Get-Location).Path,
    [string]$BundleName = "windows-update-$(Get-Date -Format 'yyyyMMdd-HHmmss').bundle",
    [int]$SSHTimeout = 10
)

# Error handling setup
$ErrorActionPreference = "Stop"
$success = $false

try {
    # 1. Verify local repository status
    Write-Host "`nVerifying local repository..." -ForegroundColor Cyan
    Push-Location $LocalRepoPath
    
    if (-not (Test-Path .\.git)) {
        throw "Not a Git repository: $LocalRepoPath"
    }
    
    $localChanges = git status --porcelain
    if ($localChanges) {
        Write-Host "Warning: Uncommitted changes detected in local repository:" -ForegroundColor Yellow
        $localChanges | ForEach-Object { Write-Host "  $_" }
        $response = Read-Host "Continue anyway? (y/n)"
        if ($response -ne 'y') { exit }
    }

    # 2. Create bundle of changes not on Ubuntu
    Write-Host "`nCreating Git bundle..." -ForegroundColor Cyan
    $bundlePath = Join-Path $LocalRepoPath $BundleName
    git bundle create $bundlePath HEAD main
    
    if (-not (Test-Path $bundlePath)) {
        throw "Failed to create bundle file"
    }
    $bundleSize = (Get-Item $bundlePath).Length / 1MB
    Write-Host "Created bundle: $BundleName (${bundleSize:N2} MB)" -ForegroundColor Green

    # 3. Transfer bundle
    Write-Host "`nTransferring bundle to Ubuntu..." -ForegroundColor Cyan
    $remoteBundlePath = "/tmp/$BundleName"
    
    scp $bundlePath "${UbuntuUser}@${UbuntuHost}:$remoteBundlePath"
    
    # 4. Apply changes on Ubuntu - SINGLE LINE COMMAND
    Write-Host "`n[4/5] Applying changes on Ubuntu..." -ForegroundColor Cyan
    
    # Using single line command to avoid parsing issues
    $sshCommand = "cd $RemoteRepoPath && git pull $remoteBundlePath 2>&1"
    Write-Host "Executing: $sshCommand" -ForegroundColor DarkGray
    $output = ssh ${UbuntuUser}@${UbuntuHost} $sshCommand
    
    if ($LASTEXITCODE -ne 0) {
        throw "Git pull failed:`n$output"
    }
    Write-Host "Git pull succeeded:`n$output" -ForegroundColor Green

    # 5. Cleanup
    Write-Host "`n[5/5] Cleaning up..." -ForegroundColor Cyan
    Remove-Item $bundlePath -ErrorAction SilentlyContinue
    ssh ${UbuntuUser}@${UbuntuHost} "rm $remoteBundlePath" | Out-Null

    Write-Host "`nUpdate completed successfully!`n" -ForegroundColor Green
}
catch {
    Write-Host "`nERROR: $_" -ForegroundColor Red
    Write-Host "Stack Trace: $($_.ScriptStackTrace)" -ForegroundColor DarkRed
    exit 1
}
finally {
    Pop-Location | Out-Null
}