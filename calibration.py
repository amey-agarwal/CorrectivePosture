"""
calibration.py
──────────────
Runs once before each study session.  Three stages:

  1. lighting   — samples webcam frames; flags under/over-exposure and
                  low contrast that would impair MediaPipe tracking.
  2. posture    — captures the participant's neutral "good" sitting
                  position for CAPTURE_SECONDS seconds and derives
                  personalised baseline values for each metric.
  3. audio      — plays the alert chime so the participant can confirm
                  volume is audible before the study starts.

All logic here is pure Python / NumPy; it does not open any camera or
windows — the existing MJPEG video pipeline feeds frames in.
"""

from __future__ import annotations
import time
import numpy as np
import audio_alerts

# How many seconds to capture the neutral sitting position
CAPTURE_SECONDS: float = 8.0

# Tolerance (added on top of the captured baseline value before flagging
# a reading as "bad posture").  Tune to taste.
TOLERANCES = {
    "neck_tilt":      8.0,   # degrees
    "shoulder_tilt":  6.0,   # degrees
    "face_proximity": 0.10,  # normalised-landmark units (larger = further away)
}

# Minimum number of valid frames needed before the baseline is accepted
MIN_SAMPLES = 20


class CalibrationSession:
    """Holds in-progress calibration state for one participant session."""

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self.stage: str = "lighting"   # lighting → posture → audio → done

        # Stage 1 ── lighting
        self._light_frames: list[tuple[float, float]] = []  # (brightness, contrast)
        self.lighting_ok: bool | None = None
        self.lighting_message: str = ""

        # Stage 2 ── posture baseline
        self._posture_samples: list[dict] = []
        self._capture_start: float | None = None
        self.baseline: dict | None = None

        # Stage 3 ── audio
        self.audio_tested: bool = False

    # ── Stage 1 ─────────────────────────────────────────────────────────

    def add_lighting_frame(self, frame_bgr) -> None:
        """Feed a raw BGR frame; call each time a new frame arrives while
        stage == 'lighting'."""
        # print(len(self._light_frames))
        gray = frame_bgr.mean(axis=2)
        self._light_frames.append((float(gray.mean()), float(gray.std())))
        if self.lighting_ok is None and len(self._light_frames) >= 20:
            self._evaluate_lighting()

    def _evaluate_lighting(self) -> None:
        # print("Evaluating lighting")
        b = np.mean([f[0] for f in self._light_frames])
        c = np.mean([f[1] for f in self._light_frames])

        print(f"Brightness={b:.2f}, Contrast={c:.2f}")

        if b < 45:
            ok, msg = False, "Too dark — add a light source facing you, or move to a brighter spot."
        elif b > 205:
            ok, msg = False, "Overexposed — reduce strong backlighting or move away from a bright window."
        elif c < 18:
            ok, msg = False, "Low contrast — the camera can barely see you. Improve your lighting."
        else:
            ok, msg = True, "Lighting looks good ✓"

        print(ok, msg)

        self.lighting_ok = ok
        self.lighting_message = msg

    def lighting_status(self) -> dict:
        return {
            "sampled": len(self._light_frames),
            "required": 20,
            "ok": self.lighting_ok,
            "message": self.lighting_message,
        }

    # ── Stage 2 ─────────────────────────────────────────────────────────

    def start_posture_capture(self) -> None:
        self.stage = "posture"
        self._posture_samples = []
        self._capture_start = time.time()
        audio_alerts.play_calibration_start()

    def add_posture_sample(self, raw_metrics: dict | None) -> None:
        """raw_metrics: output of analyze_posture() with baseline=None."""
        if raw_metrics is None:
            return
        self._posture_samples.append(raw_metrics)

    def posture_capture_progress(self) -> float:
        """0.0 – 1.0"""
        if self._capture_start is None:
            return 0.0
        return min(1.0, (time.time() - self._capture_start) / CAPTURE_SECONDS)

    def posture_capture_done(self) -> bool:
        return (
            self._capture_start is not None
            and time.time() - self._capture_start >= CAPTURE_SECONDS
            and len(self._posture_samples) >= MIN_SAMPLES
        )

    def finalize_baseline(self) -> dict | None:
        """Compute median baseline from collected samples.
        Returns baseline dict, or None if too few samples."""
        if len(self._posture_samples) < MIN_SAMPLES:
            return None

        def med(key: str) -> float:
            return float(np.median([s[key] for s in self._posture_samples]))

        self.baseline = {
            "neck_tilt":           round(med("neck_tilt"), 2),
            "shoulder_tilt":       round(med("shoulder_tilt"), 2),
            "face_proximity":      round(med("face_proximity"), 4),
            "neck_tilt_tol":       TOLERANCES["neck_tilt"],
            "shoulder_tilt_tol":   TOLERANCES["shoulder_tilt"],
            # For proximity: the threshold is the baseline value MINUS the
            # tolerance, meaning the face must not come significantly CLOSER
            # than the arm's-length baseline position.
            "face_proximity_tol":  TOLERANCES["face_proximity"],
            "n_samples":           len(self._posture_samples),
        }
        audio_alerts.play_calibration_complete()
        self.stage = "audio"
        return self.baseline

    # ── Stage 3 ─────────────────────────────────────────────────────────

    def run_audio_test(self) -> bool:
        audio_alerts.test_chime()
        self.audio_tested = True
        self.stage = "done"
        return audio_alerts.is_available()

    # ── Helpers ──────────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        return self.stage == "done" and self.baseline is not None

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "lighting": self.lighting_status(),
            "lighting_ok": self.lighting_ok,
            "lighting_message": self.lighting_message,
            "baseline": self.baseline,
            "audio_tested": self.audio_tested,
            "audio_available": audio_alerts.is_available(),
            "progress": self.posture_capture_progress(),
        }