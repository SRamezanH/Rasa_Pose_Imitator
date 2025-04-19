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
    [string]$RemoteRepoPath = "/home/cedra/psl_project",
    [string]$LocalRepoPath = (Get-Location).Path,
    [string]$BundleName = "windows-update-$(Get-Date -Format 'yyyyMMdd-HHmmss').bundle",
    [int]$SSHTimeout = 30
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
    git bundle create $bundlePath origin/main..main
    
    if (-not (Test-Path $bundlePath)) {
        throw "Failed to create bundle file"
    }
    $bundleSize = (Get-Item $bundlePath).Length / 1MB
    Write-Host "Created bundle: $BundleName (${bundleSize:N2} MB)" -ForegroundColor Green

    # 3. Copy bundle to Ubuntu
    Write-Host "`nTransferring bundle to Ubuntu..." -ForegroundColor Cyan
    $remoteBundlePath = "/tmp/$BundleName"
    
    scp -o ConnectTimeout=$SSHTimeout $bundlePath "${UbuntuUser}@${UbuntuHost}:$remoteBundlePath"
    Write-Host "Transfer completed" -ForegroundColor Green

    # 4. Apply changes on Ubuntu
    Write-Host "`nApplying changes on Ubuntu..." -ForegroundColor Cyan
    $sshCommand = @"
if [ ! -d "$RemoteRepoPath" ]; then
    echo "Error: Remote repository path not found" >&2
    exit 1
fi
cd "$RemoteRepoPath" && 
git pull "$remoteBundlePath" && 
echo "Changes applied successfully" || 
exit 1
"@
    
    $output = ssh -o ConnectTimeout=$SSHTimeout ${UbuntuUser}@${UbuntuHost} $sshCommand 2>&1
    Write-Host $output -ForegroundColor Green

    $success = $true
}
catch {
    Write-Host "`nERROR: $_" -ForegroundColor Red
    Write-Host "Stack Trace: $($_.ScriptStackTrace)" -ForegroundColor DarkRed
}
finally {
    # 5. Cleanup
    Write-Host "`nCleaning up..." -ForegroundColor Cyan
    if (Test-Path $bundlePath) {
        Remove-Item $bundlePath -ErrorAction SilentlyContinue
        Write-Host "Local bundle removed" -ForegroundColor DarkGray
    }
    
    if ($success) {
        # Only remove remote bundle if everything succeeded
        try {
            ssh -o ConnectTimeout=10 ${UbuntuUser}@${UbuntuHost} "rm $remoteBundlePath" | Out-Null
            Write-Host "Remote bundle removed" -ForegroundColor DarkGray
        }
        catch {
            Write-Host "Warning: Could not remove remote bundle" -ForegroundColor Yellow
        }
    }
    
    Pop-Location | Out-Null
    
    if ($success) {
        Write-Host "`nUpdate completed successfully!`n" -ForegroundColor Green
    }
    else {
        Write-Host "`nUpdate failed`n" -ForegroundColor Red
        exit 1
    }
}