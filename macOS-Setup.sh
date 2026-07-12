#!/bin/bash

set -e

echo "========================================"
echo " CorrectivePosture Setup (macOS)"
echo "========================================"
echo

############################################
# Check Homebrew
############################################

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found."
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    if [[ -d /opt/homebrew/bin ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -d /usr/local/bin ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

############################################
# Check Python
############################################

PYTHON_OK=false

if command -v python3 >/dev/null 2>&1; then
    VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

    if [[ "$VERSION" == "3.12" ]]; then
        PYTHON_OK=true
        PYTHON_CMD=python3
    fi
fi

if [ "$PYTHON_OK" = false ]; then
    echo
    echo "Python 3.12 not found."
    echo "Installing Python 3.12..."
    brew install python@3.12

    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_CMD=python3.12
    else
        PYTHON_CMD=python3
    fi
else
    echo "Python 3.12 detected."
fi

############################################
# Check pip
############################################

if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py
    rm get-pip.py
fi

############################################
# Install PortAudio
############################################

echo
echo "Installing PortAudio..."
brew install portaudio || true

############################################
# Create venv
############################################

echo
echo "Creating virtual environment..."

if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    $PYTHON_CMD -m venv venv
fi