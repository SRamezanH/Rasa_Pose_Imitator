<#
.SYNOPSIS
    Automates Git bundle transfer from offline Ubuntu PC to local Windows repository
.DESCRIPTION
    1. Connects to Ubuntu via SSH
    2. Creates a Git bundle of changes
    3. Copies bundle to Windows
    4. Applies changes to local repository
    5. Pushes to GitHub
.NOTES
    Requires:
    - SSH access to Ubuntu PC
    - Git installed on Windows
    - PowerShell 5.1 or newer
#>

param (
    [string]$UbuntuUser = "cedra",
    [string]$UbuntuHost = "172.27.72.227",
    [string]$RemoteRepoPath = "/home/cedra/psl_project",
    [string]$LocalRepoPath = "D:\PhD Thesis\Codes\cedra",
    [string]$BundleName = "ubuntu-update.bundle"
)

# 1. Connect to Ubuntu and create bundle
Write-Host "Creating Git bundle on Ubuntu..." -ForegroundColor Cyan
$SshCommand = @"
cd $RemoteRepoPath && 
git bundle create /tmp/$BundleName --all --not --remotes=origin
"@

ssh ${UbuntuUser}@${UbuntuHost} $SshCommand

# 2. Copy bundle to Windows
Write-Host "Copying bundle to Windows..." -ForegroundColor Cyan
scp ${UbuntuUser}@${UbuntuHost}:/tmp/${BundleName} ${LocalRepoPath}

# 3. Apply changes to local repository
Write-Host "Applying changes to local repository..." -ForegroundColor Cyan
Push-Location $LocalRepoPath
git pull .\$BundleName
git push origin main
Pop-Location

# 4. Clean up
Write-Host "Cleaning up..." -ForegroundColor Cyan
ssh ${UbuntuUser}@${UbuntuHost} "rm /tmp/$BundleName"

Write-Host "Sync completed successfully!" -ForegroundColor Green