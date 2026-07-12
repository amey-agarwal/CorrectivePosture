#!/bin/bash
source venv/bin/activate

############################################
# Upgrade pip
############################################

python -m pip install --upgrade pip # setuptools wheel

############################################
# Install dependencies
############################################

echo
echo "Installing Python packages..."

pip install -r requirements.txt
pip install opencv-python
pip install mediapipe==0.10.21
pip install flask==3.1.3
pip install Flask-SocketIO==5.6.1
pip install eventlet==0.41.0
pip install pyaudio==0.2.14
python app.py