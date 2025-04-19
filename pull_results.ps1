<#
.SYNOPSIS
    Pulls result files from Ubuntu PC to Windows laptop
.DESCRIPTION
    1. Connects to Ubuntu via SSH
    2. Creates a zip archive of specified result files
    3. Copies the zip to Windows laptop
    4. Optionally cleans up the remote zip file
#>

param (
    [string]$UbuntuUser = "cedra",
    [string]$UbuntuHost = "172.27.72.227",
    [string]$RemoteRepoPath = "/home/cedra/psl_project",
    [string]$LocalRepoPath = "D:\PhD Thesis\Codes\cedra",
    [string]$LocalDownloadPath = "D:\PhD Thesis\Codes\cedra",
    [string]$ZipName = "results-$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
)

# Error handling
$ErrorActionPreference = "Stop"

try {
    # 1. Prepare paths
    $remoteZipPath = "/tmp/$ZipName"
    $localZipPath = Join-Path $LocalDownloadPath $ZipName
    
    # 2. Create zip on Ubuntu - using single line command
    Write-Host "`n[1/3] Creating zip archive on Ubuntu..." -ForegroundColor Cyan
    
    # Using single line command to avoid line ending issues
    $sshCommand = "cd $RemoteRepoPath && zip -r $remoteZipPath model_test/*.log model_test/fig model_test/fig_4 *.log"
    
    $output = ssh ${UbuntuUser}@${UbuntuHost} $sshCommand 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Zip creation failed:`n$output"
    }
    
    # 3. Copy zip to Windows
    Write-Host "`n[2/3] Copying zip to Windows..." -ForegroundColor Cyan
    scp "${UbuntuUser}@${UbuntuHost}:$remoteZipPath" $localZipPath
    
    # Verify download
    if (-not (Test-Path $localZipPath)) {
        throw "File transfer failed - zip not found locally"
    }
    
    $fileSize = (Get-Item $localZipPath).Length / 1MB
    Write-Host "Downloaded: $($localZipPath) (${fileSize:N2} MB)" -ForegroundColor Green
    
    # 4. Clean up remote zip if requested
    Write-Host "`n[3/3] Cleaning up remote zip..." -ForegroundColor Cyan
    ssh ${UbuntuUser}@${UbuntuHost} "rm $remoteZipPath" | Out-Null
    
    Write-Host "`nResults download completed successfully!`n" -ForegroundColor Green
    Write-Host "Saved to: $localZipPath" -ForegroundColor Cyan
}
catch {
    Write-Host "`nERROR: $_" -ForegroundColor Red
    Write-Host "Stack Trace: $($_.ScriptStackTrace)" -ForegroundColor DarkRed
    exit 1
}