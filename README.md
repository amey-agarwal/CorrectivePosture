# PostureAware ML Project

~ Amey Agarwal; Anna Cho

> Research study in submission to graduate course CS7771 at the Georgia Institute of Technology, Atlanta, GA


A webcam-based posture monitoring tool built with Python, MediaPipe, and Flask.  
Runs entirely locally — no data leaves your machine.

---
# Setup Instructions

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.9 – 3.11 recommended (3.12 works) |
| Webcam | Any USB or built-in camera |
| Browser | Chrome or Firefox (Safari has known WebSocket quirks) |
| Speakers | Required for audio chime alerts |

> **Python 3.12 note:** MediaPipe's official support ends at 3.11.  
> 3.12 works in practice but if you hit install errors, install Python 3.11 instead.

---

## Step 1 — Install Python

<details>
<summary><strong>macOS</strong></summary>

```bash
# Check if Python 3.9–3.11 is already installed
python3 --version

# If not, install via Homebrew (recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```
</details>

<details>
<summary><strong>Windows</strong></summary>

1. Download the Python 3.11 installer from https://www.python.org/downloads/
2. Run the installer — **tick "Add Python to PATH"** before clicking Install
3. Open **Command Prompt** and verify:
   ```cmd
   python --version
   ```
</details>

<details>
<summary><strong>Linux (Ubuntu / Debian)</strong></summary>

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y
```
</details>

---

## Step 2 — Install the PortAudio system library (for audio chimes)

This must be done **before** `pip install -r requirements.txt`.

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install portaudio
```
</details>

<details>
<summary><strong>Windows</strong></summary>

No action needed — the PyAudio wheel for Windows bundles PortAudio automatically.

</details>

<details>
<summary><strong>Linux (Ubuntu / Debian)</strong></summary>

```bash
sudo apt install portaudio19-dev libportaudio2 -y
```
</details>

---

## Step 3 — Set up the project

Run the commands for your OS in a terminal / Command Prompt opened inside the
`CorrectivePosture` project folder.

<details>
<summary><strong>macOS / Linux</strong></summary>

```bash
# 1. Navigate to the project folder (adjust path as needed)
cd ~/Downloads/CorrectivePosture

# 2. Create an isolated virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install all dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Windows (Command Prompt)</strong></summary>

```cmd
:: 1. Navigate to the project folder
cd %USERPROFILE%\Downloads\CorrectivePosture

:: 2. Create an isolated virtual environment
python -m venv venv

:: 3. Activate it
venv\Scripts\activate

:: 4. Upgrade pip
python -m pip install --upgrade pip

:: 5. Install all dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
# If you see an execution policy error on step 3, run this first (once):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 1. Navigate to the project folder
cd "$env:USERPROFILE\Downloads\CorrectivePosture"

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

---

## Step 4 — Run the system

```bash
# macOS / Linux (virtual environment must be active)
python app.py

# Windows
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

> **Do not** open the URL on a different machine — the webcam feed and audio
> alerts only work when the browser is on the **same machine** running `app.py`.

---

## Step 5 — Export participant data after a session

Run the export script to create a timestamped ZIP of all session data:

```bash
# macOS / Linux
python export_data.py

# Windows
python export_data.py
```

The ZIP file appears in the project folder and contains:
- `data/questionnaire_responses.csv` — all participant questionnaire responses (one row each)
- `data/<pid>_<condition>_alerts.csv` — per-alert timing log for each participant
- `data/<pid>_<condition>_summary.json` — full session summary including baseline

Share the ZIP file with the researcher.

---

## Troubleshooting

### "No module named mediapipe"
The virtual environment is not active. Run:
- macOS/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

### Camera not accessible in browser
1. Ensure no other application (Zoom, Teams, FaceTime) is using the camera
2. Grant camera permissions when the browser asks
3. If using Chrome, go to `chrome://settings/content/camera` and allow `localhost`

### No audio chimes / PyAudio install fails
- macOS: run `brew install portaudio` first, then `pip install pyaudio`
- Linux: run `sudo apt install portaudio19-dev` first, then `pip install pyaudio`
- Windows: try `pip install pyaudio` — if it fails, try `pip install pipwin && pipwin install pyaudio`
- The system runs without sound if PyAudio is unavailable — a warning is printed at startup

### "Address already in use" on port 5000
Another process is using port 5000. On macOS, AirPlay Receiver uses port 5000.  
Disable it: **System Settings → General → AirDrop & Handoff → AirPlay Receiver → Off**  
Or change the port in `app.py` (last line): `socketio.run(app, port=5001)`

### MediaPipe / OpenCV conflict
Do **not** install `opencv-python` or `opencv-python-headless` alongside this project.
MediaPipe installs `opencv-contrib-python` automatically. Having two OpenCV packages
causes import errors. Fix with:
```bash
pip uninstall opencv-python opencv-python-headless
pip install -r requirements.txt
```

---

## File structure reference

```
CorrectivePosture/
├── app.py                  ← main Flask server (run this)
├── audio_alerts.py         ← PyAudio chime module
├── calibration.py          ← baseline calibration logic
├── analyze_results.py      ← post-study statistical analysis
├── export_data.py          ← data export / ZIP creator
├── requirements.txt        ← Python dependencies
├── SETUP.md                ← this file
├── templates/
│   ├── index.html          ← participant registration
│   ├── calibration.html    ← pre-session calibration wizard
│   ├── monitor.html        ← live monitoring session
│   ├── questionnaire.html  ← post-session survey
│   └── results.html        ← session summary
└── data/                   ← auto-created; all session data saved here
    ├── questionnaire_responses.csv   (all participants, one row each)
    ├── P01_immediate_alerts.csv
    ├── P01_immediate_summary.json
    └── ...
```
