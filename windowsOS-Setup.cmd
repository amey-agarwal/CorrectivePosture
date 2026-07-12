@echo off
title CorrectivePosture Setup
setlocal EnableDelayedExpansion

echo ========================================
echo CorrectivePosture Setup (Windows)
echo ========================================
echo.

REM -----------------------------------------
REM Check Python
REM -----------------------------------------

python --version >nul 2>&1

if %errorlevel%==0 (
    for /f "tokens=2" %%v in ('python --version') do (
        set VERSION=%%v
    )

    echo Found Python !VERSION!

    echo !VERSION! | findstr /b "3.12" >nul
    if errorlevel 1 (
        echo.
        echo Python 3.12 not detected.
        goto INSTALLPYTHON
    )

    goto CONTINUE

) else (
    goto INSTALLPYTHON
)

:INSTALLPYTHON

echo.
echo Python 3.12 not installed.

where winget >nul 2>&1

if %errorlevel%==0 (
    echo Installing Python 3.12 using winget...
    winget install -e --id Python.Python.3.12
) else (
    echo.
    echo Winget not available.
    echo Download Python 3.12 from:
    echo https://www.python.org/downloads/release/python-3120/
    echo.
    echo IMPORTANT:
    echo Check "Add Python to PATH" during installation.
    pause
)

echo.
echo Restarting Python detection...

python --version >nul 2>&1

if errorlevel 1 (
    echo.
    echo Python still not found.
    echo Please reopen Command Prompt after installation.
    pause
    exit /b
)

:CONTINUE

REM -----------------------------------------
REM Check pip
REM -----------------------------------------

python -m pip --version >nul 2>&1

if errorlevel 1 (
    echo Installing pip...
    curl -O https://bootstrap.pypa.io/get-pip.py
    python get-pip.py
    del get-pip.py
)

pause