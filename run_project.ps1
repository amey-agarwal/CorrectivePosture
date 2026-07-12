$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========== CorrectivePosture ==========" -ForegroundColor Cyan

$python = "C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "Python executable:"
Write-Host $python

if (!(Test-Path $python)) {
    Write-Host ""
    Write-Host "Python not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit
}

Write-Host ""
Write-Host "Creating virtual environment..."
& $python -m venv pyvenv

Write-Host ""
Write-Host "Activating environment..."
& ".\pyvenv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "Installing packages..."
python -m pip install --upgrade pip

pip install -r requirements.txt
pip install mediapipe==0.10.21
pip install flask==3.1.3
pip install Flask-SocketIO==5.6.1
pip install eventlet==0.41.0
pip install pyaudio==0.2.14

Write-Host ""
Write-Host "Starting application..."
python app.py

Read-Host "Press Enter to close"
