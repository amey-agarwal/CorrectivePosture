"""
audio_alerts.py
────────────────
Plays short audible chimes using PyAudio when posture alerts fire, when a
session ends, or when no person/posture has been detected for a long time.

Tones are synthesized on the fly (simple sine-wave beeps) so no external
.wav files are required. Playback runs in a background thread so it never
blocks the camera/Flask loop.

Requires PyAudio + the PortAudio system library:
  macOS:   brew install portaudio  &&  pip install pyaudio
  Windows: pip install pyaudio        (wheels bundle PortAudio)
  Linux:   sudo apt-get install portaudio19-dev  &&  pip install pyaudio

If PyAudio/PortAudio is not available, this module degrades gracefully:
all play_* functions become no-ops and a warning is printed once, so the
rest of the system keeps running without sound.
"""

import threading
import numpy as np

try:
    import pyaudio
    _PYAUDIO_AVAILABLE = True
except (ImportError, OSError):
    _PYAUDIO_AVAILABLE = False
    print("⚠ PyAudio not available — audio chimes are disabled. "
          "Install PortAudio + pyaudio to enable sound alerts. "
          "See audio_alerts.py header for install commands.")

SAMPLE_RATE = 44100


def _tone(freq, duration, volume=0.5, fade_ms=15):
    """Generate a single sine-wave tone as int16 PCM bytes, with a short
    fade-in/out to avoid audible clicks."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)

    fade_samples = int(SAMPLE_RATE * fade_ms / 1000)
    if fade_samples > 0 and fade_samples * 2 < len(wave):
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

    audio = (wave * volume * 32767).astype(np.int16)
    return audio.tobytes()


def _play_bytes(pcm_bytes):
    if not _PYAUDIO_AVAILABLE:
        return
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                          rate=SAMPLE_RATE, output=True)
        stream.write(pcm_bytes)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"⚠ Audio playback failed: {e}")
    finally:
        pa.terminate()


def _play_async(pcm_bytes):
    """Fire-and-forget playback on a background thread."""
    threading.Thread(target=_play_bytes, args=(pcm_bytes,), daemon=True).start()


# ─── Public chime presets ─────────────────────────────────────────────────────
def play_posture_alert():
    """Short, attention-getting two-tone chime for a poor-posture alert."""
    seq = _tone(880, 0.12) + _tone(660, 0.15)
    _play_async(seq)


def play_correction_success():
    """Light upward 'ding' confirming posture was corrected."""
    seq = _tone(523, 0.08) + _tone(784, 0.12)
    _play_async(seq)


def play_no_person_detected():
    """Low, slow tone used when no person/posture has been detected for a
    long stretch (e.g. participant left the desk or camera lost tracking)."""
    seq = _tone(220, 0.25) + _tone(196, 0.3)
    _play_async(seq)


def play_session_end():
    """Distinct three-note chime played once at the end of the 30-minute
    session, signalling the participant to stop and export their data."""
    seq = _tone(523, 0.15) + _tone(659, 0.15) + _tone(784, 0.25)
    _play_async(seq)


def play_calibration_complete():
    """Friendly confirmation chime once baseline calibration finishes."""
    seq = _tone(440, 0.1) + _tone(554, 0.1) + _tone(659, 0.15)
    _play_async(seq)


def test_chime():
    """Used by the calibration step's 'Test Chime' button so the
    participant can confirm their volume is audible before the study
    starts."""
    play_posture_alert()


def is_available():
    return _PYAUDIO_AVAILABLE