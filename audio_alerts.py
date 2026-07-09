"""
audio_alerts.py
───────────────
Plays short synthesised chimes using PyAudio.  All tones are generated
from sine waves on the fly — no external .wav files required.

Playback is always dispatched to a daemon thread so it never blocks the
camera/Flask loop.

Install requirements
--------------------
  macOS  : brew install portaudio  &&  pip install pyaudio
  Windows: pip install pyaudio          (wheels bundle PortAudio)
  Linux  : sudo apt-get install portaudio19-dev  &&  pip install pyaudio

If PyAudio/PortAudio is absent the module degrades to a no-op and prints
one warning, so the rest of the system keeps running without sound.
"""

import threading
import numpy as np

try:
    import pyaudio
    _PA = True
except (ImportError, OSError):
    _PA = False
    print(
        "⚠  PyAudio not available — audio chimes are disabled.\n"
        "   Install PortAudio + pyaudio to enable sound alerts.\n"
        "   See audio_alerts.py header for platform instructions."
    )

SAMPLE_RATE = 44100


# ── Synthesis helpers ─────────────────────────────────────────────────────────

def _sine(freq: float, duration: float, volume: float = 0.45) -> np.ndarray:
    """Return a float64 sine-wave array, faded at both ends to avoid clicks."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, False)
    wave = np.sin(2 * np.pi * freq * t) * volume

    # 10-ms cosine fade-in / fade-out
    fade = int(SAMPLE_RATE * 0.010)
    if fade > 0 and 2 * fade < n:
        ramp = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2
        wave[:fade]  *= ramp
        wave[-fade:] *= ramp[::-1]

    return wave


def _concat(*arrays: np.ndarray, gap_ms: int = 0) -> bytes:
    """Concatenate tone arrays (optionally with silence gaps) → int16 PCM bytes."""
    parts = []
    silence = np.zeros(int(SAMPLE_RATE * gap_ms / 1000))
    for i, a in enumerate(arrays):
        parts.append(a)
        if gap_ms > 0 and i < len(arrays) - 1:
            parts.append(silence)
    merged = np.concatenate(parts)
    return (merged * 32767).astype(np.int16).tobytes()


def _play(pcm: bytes) -> None:
    """Blocking playback — called only from a background thread."""
    if not _PA:
        return
    pa = pyaudio.PyAudio()
    try:
        s = pa.open(format=pyaudio.paInt16, channels=1,
                    rate=SAMPLE_RATE, output=True)
        s.write(pcm)
        s.stop_stream()
        s.close()
    except Exception as exc:
        print(f"⚠  Audio playback error: {exc}")
    finally:
        pa.terminate()


def _async(pcm: bytes) -> None:
    """Fire-and-forget."""
    threading.Thread(target=_play, args=(pcm,), daemon=True).start()


# ── Public chimes ─────────────────────────────────────────────────────────────

def play_posture_alert() -> None:
    """Two-tone descending chime — poor posture detected."""
    pcm = _concat(_sine(880, 0.12), _sine(660, 0.18), gap_ms=20)
    _async(pcm)


def play_correction_success() -> None:
    """Ascending two-note ding — posture corrected."""
    pcm = _concat(_sine(523, 0.08), _sine(784, 0.14), gap_ms=15)
    _async(pcm)


def play_no_person_detected() -> None:
    """Low slow pulse — participant out of frame for too long."""
    pcm = _concat(_sine(220, 0.28), _sine(196, 0.32), gap_ms=40)
    _async(pcm)


def play_session_end() -> None:
    """Three-note ascending chime — 30-minute session complete."""
    pcm = _concat(
        _sine(523, 0.15), _sine(659, 0.15), _sine(784, 0.28),
        gap_ms=25,
    )
    _async(pcm)


def play_calibration_start() -> None:
    """Single soft tone — calibration capture has begun."""
    pcm = _concat(_sine(440, 0.20))
    _async(pcm)


def play_calibration_complete() -> None:
    """Three-note rising chime — baseline captured successfully."""
    pcm = _concat(
        _sine(440, 0.10), _sine(554, 0.10), _sine(659, 0.18),
        gap_ms=20,
    )
    _async(pcm)


def test_chime() -> None:
    """Same as the posture-alert chime; used on the calibration page
    so the participant can confirm their speakers are working."""
    play_posture_alert()


def is_available() -> bool:
    return _PA