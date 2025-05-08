# Run this in PowerShell as Administrator
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\Program Files\Git\cmd",
    "Machine"
)