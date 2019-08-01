echo Download and install python

powershell -command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 ; (New-Object Net.WebClient).DownloadFile('https://www.python.org/ftp/python/3.7.2/python-3.7.2-amd64.exe', 'python.exe') }"

python.exe 

echo Set environment variable 

powershell -command "& { [Environment]::SetEnvironmentVariable('Path', $env:USERPROFILE+'\AppData\Local\Programs\Python\Python37\;'+$env:USERPROFILE+'\AppData\Local\Programs\Python\Python37\Scripts\;'+$env:Path, 'User') }"

exit