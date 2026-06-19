"""
calibration.py
────────────────
Runs once at the start of each participant session, BEFORE the study
timer starts. Three checks happen in sequence, all surfaced through the
web UI (no separate OpenCV window is ever opened):

  1. Lighting check       — samples webcam frames, flags under/over-exposed
                             or low-contrast conditions that would make
                             MediaPipe's pose estimation unreliable.
  2. Baseline posture capture — asks the participant to sit normally for a
                             few seconds and records their neutral
                             forward-head angle, shoulder tilt, and neck
                             tilt. These become the personalised reference
                             values that `analyze_posture()` compares
                             against, instead of fixed population
                             thresholds.
  3. Chime test            — plays the same chime used for real posture
                             alerts so the participant can confirm their
                             volume is audible.

All capture happens through the existing MJPEG video pipeline; this
module only holds state and pure-Python analysis functions, so it
introduces no extra camera windows or capture loops.
"""

import time
import numpy as np

import audio_alerts


class CalibrationSession:
    """Holds in-progress calibration state for one participant session."""

    def __init__(self, sample_seconds=5.0, required_samples=20):
        self.sample_seconds = sample_seconds
        self.required_samples = required_samples

        self.stage = 'lighting'       # 'lighting' -> 'posture' -> 'audio' -> 'done'
        self.lighting_samples = []     # list of (brightness, contrast)
        self.lighting_ok = None
        self.lighting_message = ''

        self.posture_samples = []      # list of raw metric dicts from analyze_posture
        self.posture_started_at = None
        self.baseline = None           # final computed baseline thresholds

        self.audio_tested = False

    # ── Stage 1: Lighting ──────────────────────────────────────────────
    def add_lighting_frame(self, frame_bgr):
        """Call once per incoming frame while stage == 'lighting'."""
        gray = frame_bgr.mean(axis=2)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        self.lighting_samples.append((brightness, contrast))

        if len(self.lighting_samples) >= 15:
            self._evaluate_lighting()

    def _evaluate_lighting(self):
        b_vals = [s[0] for s in self.lighting_samples]
        c_vals = [s[1] for s in self.lighting_samples]
        avg_b = float(np.mean(b_vals))
        avg_c = float(np.mean(c_vals))

        if avg_b < 50:
            self.lighting_ok = False
            self.lighting_message = 'Too dark — add a light source facing you, or move closer to a window.'
        elif avg_b > 200:
            self.lighting_ok = False
            self.lighting_message = 'Too bright / overexposed — reduce backlight or strong light behind you.'
        elif avg_c < 20:
            self.lighting_ok = False
            self.lighting_message = 'Low contrast — the camera may be struggling to see you clearly. Try better lighting.'
        else:
            self.lighting_ok = True
            self.lighting_message = 'Lighting looks good.'

        return self.lighting_ok

    def lighting_status(self):
        return {
            'sampled': len(self.lighting_samples),
            'required': 15,
            'ok': self.lighting_ok,
            'message': self.lighting_message,
        }

    # ── Stage 2: Baseline posture ──────────────────────────────────────
    def start_posture_capture(self):
        self.stage = 'posture'
        self.posture_samples = []
        self.posture_started_at = time.time()

    def add_posture_sample(self, raw_metrics):
        """raw_metrics: dict with keys fha, shoulder_tilt, torso_inclination
        (i.e. the same dict shape `analyze_posture` returns, captured with
        baseline=None so it reflects raw angles, not a good/bad verdict)."""
        if raw_metrics is None:
            return
        self.posture_samples.append(raw_metrics)

    def posture_capture_progress(self):
        if not self.posture_started_at:
            return 0.0
        elapsed = time.time() - self.posture_started_at
        return min(1.0, elapsed / self.sample_seconds)

    def posture_capture_done(self):
        return (self.posture_started_at is not None and
                time.time() - self.posture_started_at >= self.sample_seconds and
                len(self.posture_samples) >= 5)

    def finalize_baseline(self):
        """Compute median baseline values + tolerance bands from captured
        samples. Falls back to population defaults if too few/noisy
        samples were captured."""
        if len(self.posture_samples) < 5:
            self.baseline = None
            return None

        fha_vals   = [s['fha'] for s in self.posture_samples]
        sh_vals    = [s['shoulder_tilt'] for s in self.posture_samples]
        neck_vals  = [s['torso_inclination'] for s in self.posture_samples]

        # Use median for robustness against momentary tracking glitches
        baseline = {
            'fha':            round(float(np.median(fha_vals)), 1),
            'shoulder_tilt':  round(float(np.median(sh_vals)), 1),
            'neck_tilt':      round(float(np.median(neck_vals)), 1),
            # Tolerance = how far from baseline counts as "poor posture".
            # Wider tolerance for FHA since head movement is natural;
            # tighter for shoulder tilt which is usually more stable.
            'fha_tol':        12.0,
            'shoulder_tol':   6.0,
            'neck_tol':       10.0,
            'n_samples':      len(self.posture_samples),
        }
        self.baseline = baseline
        self.stage = 'audio'
        return baseline

    # ── Stage 3: Audio test ─────────────────────────────────────────────
    def run_audio_test(self):
        audio_alerts.test_chime()
        self.audio_tested = True
        self.stage = 'done'
        return audio_alerts.is_available()

    def is_complete(self):
        return self.stage == 'done' and self.baseline is not None

    def summary(self):
        return {
            'stage': self.stage,
            'lighting_ok': self.lighting_ok,
            'lighting_message': self.lighting_message,
            'baseline': self.baseline,
            'audio_tested': self.audio_tested,
            'audio_available': audio_alerts.is_available(),
        }