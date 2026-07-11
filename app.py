"""
Adaptive Delayed Feedback System for Posture Improvement
Flask + MediaPipe + WebSocket backend

Session flow
────────────
  / (index)  →  /calibrate/<session_id>  →  /monitor/<session_id>
                      ↑
               created by POST /api/register

Posture metrics (upper-body only — chest-and-above)
────────────────────────────────────────────────────
  1. Neck tilt        — angle of the head from the vertical (ear-to-shoulder)
  2. Shoulder tilt    — horizontal roll of the shoulder line
  3. Face proximity   — normalised distance from nose to mid-shoulder;
                        a decreasing value means the participant is leaning
                        closer to the screen (forward head / hunching).

No cv2.imshow() or cv2.waitKey() are ever called — frames only leave
this process as MJPEG bytes streamed to the browser.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit

import audio_alerts
from calibration import CalibrationSession

import eventlet

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "posture-study-2026"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
QUESTIONNAIRE_MASTER_CSV = os.path.join(DATA_DIR, "questionnaire_responses.csv")

QUESTIONNAIRE_FIELDS = [
    "participant_id", "condition", "age", "gender", "session_duration_s",
    "total_alerts", "ignored_alerts", "avg_latency_s", "median_latency_s",
    "pct_time_poor_posture", "final_delay",
    # Likert items
    "helpfulness_aware", "helpfulness_useful", "helpfulness_motivated",
    "annoyance_disruptive", "annoyance_frequency", "annoyance_ignored",
    "usability_easy", "usability_clear", "usability_comfort",
    "willingness_daily", "willingness_recommend", "willingness_timing",
    # Computed subscale averages (from JS)
    "score_helpfulness", "score_annoyance", "score_usability", "score_willingness",
    # Open-ended
    "open_helpful", "open_improve", "open_other",
    # Metadata
    "submitted_at",
]

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── MediaPipe (single global instance, not thread-safe → one pose per process)
mp_pose = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
    model_complexity=1,
)

# How long with no landmarks before the "are-you-there" chime fires
NO_PERSON_CHIME_S: float = 25.0


# ── Shared webcam singleton ──────────────────────────────────────────────────
class _Webcam:
    """
    Reference-counted wrapper around a single cv2.VideoCapture so both the
    calibration and the live-monitor MJPEG routes share one device handle.
    Never calls imshow / waitKey → no extra window ever opens.
    """

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._refs: int = 0

    def acquire(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS, 24)
        self._refs += 1
        return self._cap

    def release(self) -> None:
        self._refs = max(0, self._refs - 1)
        if self._refs == 0 and self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()


webcam = _Webcam()

# ── In-memory session stores ─────────────────────────────────────────────────
sessions: dict[str, dict]              = {}
calibrations: dict[str, CalibrationSession] = {}


# ── Session factory ───────────────────────────────────────────────────────────
def _new_session(pid: str, condition: str,
                 age=None, gender=None, baseline=None) -> dict:
    return {
        "participant_id": pid,
        "condition": condition,           # "immediate" | "adaptive"
        "age": age,
        "gender": gender,
        "start_time": time.time(),
        "end_time": None,
        "baseline": baseline,             # personalised thresholds from calibration

        # Posture state
        "posture_good": True,
        "bad_start": None,
        "total_poor_s": 0.0,

        # No-person tracking
        "last_seen": time.time(),
        "no_person_chimed": False,

        # Alert state
        "alert_active": False,
        "alert_shown_at": None,
        "last_alert_time": None,
        "correction_pending": False,

        # Adaptive delay
        "delay": 5.0,
        "delay_min": 2.0,
        "delay_max": 20.0,
        "alpha": 0.30,

        # Logging
        "alerts": [],
        "current_alert": None,
        "n_alerts": 0,
        "latencies": [],
        "n_ignored": 0,

        "active": False,
    }


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  POSTURE ANALYSIS — upper-body only (chest and above)
# ╚══════════════════════════════════════════════════════════════════════════════

def _angle(a, b, c) -> float:
    """Angle at vertex b in the triangle a-b-c, in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def analyze_posture(landmarks, w: int, h: int, baseline: dict | None = None) -> dict | None:
    """
    Extract the three study metrics from upper-body pose landmarks.

    Metrics
    -------
    neck_tilt        — angle of the mid-ear → mid-shoulder segment from
                       vertical.  0° = perfectly upright; grows as the
                       head juts forward or tilts to the side.
    shoulder_tilt    — horizontal roll angle of the shoulder line.
                       0° = level; grows as one shoulder rises/drops.
    face_proximity   — normalised distance between nose tip and
                       mid-shoulder point (using normalised landmark
                       coordinates in [0,1]).  Smaller value ≡ face
                       closer to camera / screen.

    Requires only: NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER,
                   RIGHT_SHOULDER.  No hips needed. Cause no Shakira.

    Returns None if visibility of any key landmark is < 0.55.
    """
    lm = landmarks.landmark

    KEY = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
    ]
    if min(lm[k].visibility for k in KEY) < 0.55:
        return None

    def px(k):
        return [lm[k].x * w, lm[k].y * h]

    def norm(k):
        return [lm[k].x, lm[k].y]  # normalised coords for distance metric

    nose_px  = px(mp_pose.PoseLandmark.NOSE)
    l_ear    = px(mp_pose.PoseLandmark.LEFT_EAR)
    r_ear    = px(mp_pose.PoseLandmark.RIGHT_EAR)
    l_sh     = px(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_sh     = px(mp_pose.PoseLandmark.RIGHT_SHOULDER)

    mid_ear  = [(l_ear[0] + r_ear[0]) / 2, (l_ear[1] + r_ear[1]) / 2]
    mid_sh   = [(l_sh[0]  + r_sh[0])  / 2, (l_sh[1]  + r_sh[1])  / 2]

    # ── Metric 1: Neck tilt ────────────────────────────────────────────
    # Angle between the ear-to-shoulder segment and vertical.
    vert_ref = [mid_sh[0], mid_sh[1] - 100]
    neck_tilt = _angle(mid_ear, mid_sh, vert_ref)

    # ── Metric 2: Shoulder tilt (roll) ────────────────────────────────
    dx = abs(l_sh[0] - r_sh[0])
    dy = abs(l_sh[1] - r_sh[1])
    shoulder_tilt = math.degrees(math.atan2(dy, dx + 1e-9))

    # ── Metric 3: Face proximity ───────────────────────────────────────
    # Use normalised landmark coords so the metric is independent of
    # frame resolution.  Distance decreases as person leans forward.
    nose_n  = norm(mp_pose.PoseLandmark.NOSE)
    l_sh_n  = norm(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_sh_n  = norm(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    mid_sh_n = [(l_sh_n[0] + r_sh_n[0]) / 2, (l_sh_n[1] + r_sh_n[1]) / 2]
    face_proximity = math.hypot(
        nose_n[0] - mid_sh_n[0],
        nose_n[1] - mid_sh_n[1],
    )

    # ── Thresholds ──────────────────────────────────────────────────────
    if baseline:
        neck_thresh  = baseline["neck_tilt"]      + baseline["neck_tilt_tol"]
        sh_thresh    = baseline["shoulder_tilt"]  + baseline["shoulder_tilt_tol"]
        # Proximity: bad when face is significantly CLOSER than baseline
        prox_thresh  = baseline["face_proximity"] - baseline["face_proximity_tol"]
    else:
        neck_thresh  = 22.0
        sh_thresh    = 10.0
        prox_thresh  = 0.28   # if face_proximity drops below this → too close

    neck_bad  = neck_tilt    > neck_thresh
    sh_bad    = shoulder_tilt > sh_thresh
    prox_bad  = face_proximity < prox_thresh

    n_bad = sum([neck_bad, sh_bad, prox_bad])
    good  = not (n_bad >= 2 or neck_tilt > neck_thresh + 12 or prox_bad) # number of bad posture out of 3

    # Composite score 0-100
    neck_score  = max(0, 100 - (neck_tilt  / neck_thresh)  * 50)
    sh_score    = max(0, 100 - (shoulder_tilt / sh_thresh) * 50)
    # Proximity score: 100 when at or beyond baseline distance; drops as closer
    prox_ratio  = max(0.0, face_proximity / max(prox_thresh, 0.01))
    prox_score  = min(100, prox_ratio * 70)
    score       = int(neck_score * 0.45 + sh_score * 0.30 + prox_score * 0.25)
    score       = max(0, min(100, score))

    return {
        "good":           good,
        "score":          score,
        "neck_tilt":      round(neck_tilt,      1),
        "shoulder_tilt":  round(shoulder_tilt,  1),
        "face_proximity": round(face_proximity,  3),
        "neck_bad":       neck_bad,
        "sh_bad":         sh_bad,
        "prox_bad":       prox_bad,
        "min_vis":        round(min(lm[k].visibility for k in KEY), 2),
        # Expose thresholds so the UI can draw reference lines
        "thresh_neck":    round(neck_thresh,  1),
        "thresh_sh":      round(sh_thresh,    1),
        "thresh_prox":    round(prox_thresh,  3),
    }


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  ADAPTIVE DELAY ENGINE
# ╚══════════════════════════════════════════════════════════════════════════════

def _update_delay(sess: dict, latency: float | None) -> None:
    """Exponentially weighted update of the inter-alert delay threshold."""
    a, cur = sess["alpha"], sess["delay"]
    if latency is None:               # ignored: shorten delay
        target = cur * 0.70
    elif latency < cur:               # fast response: lengthen delay
        target = cur * (1 + (1 - latency / cur) * 0.50)
    else:                             # slow response: shorten delay
        target = cur * 0.85
    sess["delay"] = round(
        max(sess["delay_min"],
            min(sess["delay_max"], a * target + (1 - a) * cur)), 2)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  FEEDBACK STATE MACHINE
# ╚══════════════════════════════════════════════════════════════════════════════

def _process_posture(session_id: str, p: dict, ts: float) -> dict:
    """
    Core per-frame decision logic.
    Returns a dict of 'actions' to push to the browser via SocketIO.
    """
    sess = sessions.get(session_id)
    if not sess or not sess["active"]:
        return {}

    sess["last_seen"]      = ts
    sess["no_person_chimed"] = False
    actions: dict = {}

    good = p["good"]

    # good→bad
    if not good and sess["posture_good"]:
        sess["bad_start"]   = ts
        sess["posture_good"] = False

    # bad→good
    elif good and not sess["posture_good"]:
        if sess["bad_start"]:
            sess["total_poor_s"] += ts - sess["bad_start"]
            sess["bad_start"] = None

        if sess["correction_pending"] and sess["alert_shown_at"]:
            lat = round(ts - sess["alert_shown_at"], 2)
            sess["latencies"].append(lat)
            if sess["current_alert"]:
                sess["current_alert"].update(
                    corrected=True, latency=lat, correction_time=ts)
                sess["alerts"].append(sess["current_alert"])
                sess["current_alert"] = None
            if sess["condition"] == "adaptive":
                _update_delay(sess, lat)
            sess["correction_pending"] = False
            sess["alert_active"]       = False
            actions["hide_alert"]      = True
            actions["latency"]         = lat
            actions["new_delay"]       = sess["delay"]
            audio_alerts.play_correction_success()

        sess["posture_good"] = True

    # Should we fire an alert?
    if not good and not sess["alert_active"]:
        bad_dur = ts - (sess["bad_start"] or ts)
        threshold = 1.0 if sess["condition"] == "immediate" else sess["delay"]

        if bad_dur >= threshold:
            gap = ts - (sess["last_alert_time"] or 0)
            if gap >= sess["delay"]:
                sess["alert_active"]   = True
                sess["alert_shown_at"] = ts
                sess["last_alert_time"] = ts
                sess["correction_pending"] = True
                sess["n_alerts"] += 1
                sess["current_alert"] = {
                    "alert_id": sess["n_alerts"],
                    "shown_at": ts,
                    "condition": sess["condition"],
                    "delay_threshold": sess["delay"],
                    "corrected": False,
                    "latency": None,
                }
                actions["show_alert"] = True
                actions["alert_id"]   = sess["n_alerts"]
                actions["delay"]      = sess["delay"]
                audio_alerts.play_posture_alert()

    # Alert timeout (10 s ignored)
    if sess["alert_active"] and sess["alert_shown_at"]:
        if ts - sess["alert_shown_at"] > 10.0 and sess["correction_pending"]:
            sess["n_ignored"] += 1
            if sess["current_alert"]:
                sess["alerts"].append(sess["current_alert"])
                sess["current_alert"] = None
            if sess["condition"] == "adaptive":
                _update_delay(sess, None)
                actions["new_delay"] = sess["delay"]
            sess["correction_pending"] = False
            sess["alert_active"]       = False
            actions["alert_expired"]   = True

    return actions


def _no_person(session_id: str, ts: float) -> None:
    sess = sessions.get(session_id)
    if not sess or not sess["active"]:
        return
    if ts - sess["last_seen"] >= NO_PERSON_CHIME_S and not sess["no_person_chimed"]:
        audio_alerts.play_no_person_detected()
        sess["no_person_chimed"] = True


def _live_metrics(session_id: str) -> dict:
    sess = sessions.get(session_id, {})
    if not sess:
        return {}
    elapsed  = time.time() - sess["start_time"]
    avg_lat  = round(sum(sess["latencies"]) / len(sess["latencies"]), 1) \
               if sess["latencies"] else None
    return {
        "elapsed":      round(elapsed, 0),
        "n_alerts":     sess["n_alerts"],
        "avg_latency":  avg_lat,
        "poor_s":       round(sess["total_poor_s"], 1),
        "delay":        round(sess["delay"], 1),
        "condition":    sess["condition"],
        "n_ignored":    sess["n_ignored"],
    }


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  MJPEG GENERATORS
# ╚══════════════════════════════════════════════════════════════════════════════

def _encode(frame) -> bytes:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            + buf.tobytes() + b"\r\n")


def _gen_calibration(session_id: str):
    """MJPEG stream used on the calibration page — feeds frames into the
    CalibrationSession and overlays status text."""
    cal = calibrations.get(session_id)
    print("Generator ",id(cal))
    cap = webcam.acquire()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Lighting stage: just sample raw frames
            if cal and cal.stage == "lighting":
                cal.add_lighting_frame(frame)
                cv2.putText(frame, "Checking lighting…", (14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)

            # Posture stage: run MediaPipe and collect samples
            elif cal and cal.stage == "posture":
                res = pose.process(rgb)
                if res.pose_landmarks:
                    metrics = analyze_posture(res.pose_landmarks, w, h)
                    cal.add_posture_sample(metrics)
                    print(len(cal._posture_samples))
                    mp_draw.draw_landmarks(
                        frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec((100, 220, 100), 2, 3),
                        mp_draw.DrawingSpec((100, 220, 100), 2),
                    )
                    pct = int(cal.posture_capture_progress() * 100)
                    cv2.putText(frame, f"Capturing baseline… {pct}%", (14, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100, 220, 100), 2)

            yield _encode(frame)
            eventlet.sleep(0.01)
    finally:
        webcam.release()


def _gen_monitor(session_id: str):
    """MJPEG stream used on the monitor page — full posture analysis loop."""
    sess = sessions.get(session_id)
    baseline = sess["baseline"] if sess else None
    cap      = webcam.acquire()
    last_emit = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = pose.process(rgb)
            now   = time.time()

            p = None
            if res.pose_landmarks:
                p = analyze_posture(res.pose_landmarks, w, h, baseline)
                if p:
                    pass
                    # colour = (0, 210, 70) if p["good"] else (50, 50, 240)
                    # mp_draw.draw_landmarks(
                    #     frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    #     mp_draw.DrawingSpec(colour, 2, 3),
                    #     mp_draw.DrawingSpec(colour, 2),
                    # )
                    # cv2.rectangle(frame, (6, 6), (180, 40), (0, 0, 0), -1)
                    # cv2.putText(frame, f"Score: {p['score']}", (12, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour, 2)

            if now - last_emit > 0.20:   # emit at ~5 Hz
                last_emit = now
                if p:
                    actions = _process_posture(session_id, p, now)
                    socketio.emit("posture_update", {
                        "posture":  p,
                        "actions":  actions,
                        "metrics":  _live_metrics(session_id),
                        "ts":       now,
                    })
                else:
                    _no_person(session_id, now)
                    socketio.emit("posture_update", {
                        "posture":  None,
                        "actions":  {},
                        "metrics":  _live_metrics(session_id),
                        "ts":       now,
                    })

            yield _encode(frame)
            eventlet.sleep(0.01)
    finally:
        webcam.release()


# ╔══════════════════════════════════════════════════════════════════════════════
# ║  DATA EXPORT
# ╚══════════════════════════════════════════════════════════════════════════════

def _save_session(session_id: str):
    """
    Write per-session objective data (alert log CSV + summary JSON).
    Called once when the session ends (api_stop) or when /results is loaded.
    Does NOT include questionnaire data — that is handled by _save_questionnaire.
    Returns (summary_dict, alert_csv_path, summary_json_path).
    """
    sess = sessions.get(session_id)
    if not sess:
        return None, None, None
 
    os.makedirs(DATA_DIR, exist_ok=True)
    pid  = sess["participant_id"]
    cond = sess["condition"]
 
    # Use a fixed filename per session (no timestamp) so repeated calls
    # to _save_session overwrite rather than creating duplicate files.
    alert_path = os.path.join(DATA_DIR, f"{pid}_{cond}_alerts.csv")
    sum_path   = os.path.join(DATA_DIR, f"{pid}_{cond}_summary.json")
 
    # ── Alert log CSV ─────────────────────────────────────────────────
    with open(alert_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "alert_id", "shown_at", "condition", "delay_threshold",
            "corrected", "latency", "correction_time",
        ])
        writer.writeheader()
        for a in sess["alerts"]:
            writer.writerow({
                "alert_id":        a.get("alert_id"),
                "shown_at":        round(a.get("shown_at", 0), 3),
                "condition":       a.get("condition"),
                "delay_threshold": a.get("delay_threshold"),
                "corrected":       a.get("corrected"),
                "latency":         a.get("latency"),
                "correction_time": round(a.get("correction_time", 0), 3)
                                   if a.get("correction_time") else None,
            })
 
    # ── Summary JSON ──────────────────────────────────────────────────
    elapsed = (sess.get("end_time") or time.time()) - sess["start_time"]
    lats    = sess["latencies"]
    summary = {
        "participant_id":     pid,
        "condition":          cond,
        "age":                sess.get("age"),
        "gender":             sess.get("gender"),
        "baseline":           sess.get("baseline"),
        "session_duration_s": round(elapsed, 1),
        "total_alerts":       sess["n_alerts"],
        "ignored_alerts":     sess["n_ignored"],
        "total_poor_s":       round(sess["total_poor_s"], 1),
        "pct_time_poor_posture":           round(sess["total_poor_s"] / elapsed * 100, 1)
                              if elapsed > 0 else 0,
        "latencies":          lats,
        "avg_latency_s":      round(sum(lats) / len(lats), 2) if lats else None,
        "median_latency_s":   round(float(np.median(lats)), 2) if lats else None,
        "final_delay":        sess["delay"],
        # questionnaire field starts empty; filled by _save_questionnaire
        "questionnaire":      sess.get("questionnaire"),
    }
 
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
 
    return summary, alert_path, sum_path

def _save_questionnaire(session_id: str, q_data: dict):
    """
    Called immediately when the participant submits the questionnaire.
 
    1. Stores q_data in sess["questionnaire"] (in-memory, for /results page).
    2. Updates the per-session summary JSON (adds "questionnaire" key).
    3. Appends one row to the master questionnaire CSV so all participants'
       responses are in one file ready for analysis (Excel / SPSS / R / Python).
    """
    sess = sessions.get(session_id)
    if not sess:
        return
 
    sess["questionnaire"] = q_data
    os.makedirs(DATA_DIR, exist_ok=True)
    pid  = sess["participant_id"]
    cond = sess["condition"]
 
    # ── 1. Update per-session summary JSON ────────────────────────────
    sum_path = os.path.join(DATA_DIR, f"{pid}_{cond}_summary.json")
    if os.path.exists(sum_path):
        with open(sum_path) as f:
            summary = json.load(f)
        summary["questionnaire"] = q_data
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        # Session file doesn't exist yet — create it now including questionnaire
        summary, _, _ = _save_session(session_id)
        if summary:
            summary["questionnaire"] = q_data
            with open(sum_path, "w") as f:
                json.dump(summary, f, indent=2)
 
    # ── 2. Append row to master questionnaire CSV ─────────────────────
    elapsed = (sess.get("end_time") or time.time()) - sess["start_time"]
    lats    = sess["latencies"]
 
    row = {
        "participant_id":     pid,
        "condition":          cond,
        "age":                sess.get("age"),
        "gender":             sess.get("gender"),
        "session_duration_s": round(elapsed, 1),
        "total_alerts":       sess["n_alerts"],
        "ignored_alerts":     sess["n_ignored"],
        "avg_latency_s":      round(sum(lats) / len(lats), 2) if lats else None,
        "median_latency_s":   round(float(np.median(lats)), 2) if lats else None,
        "pct_time_poor_posture":           round(sess["total_poor_s"] / elapsed * 100, 1)
                              if elapsed > 0 else 0,
        "final_delay":        sess["delay"],
        # Likert items (all expected to be present; default to None if missing)
        "helpfulness_aware":       q_data.get("helpfulness_aware"),
        "helpfulness_useful":      q_data.get("helpfulness_useful"),
        "helpfulness_motivated":   q_data.get("helpfulness_motivated"),
        "annoyance_disruptive":    q_data.get("annoyance_disruptive"),
        "annoyance_frequency":     q_data.get("annoyance_frequency"),
        "annoyance_ignored":       q_data.get("annoyance_ignored"),
        "usability_easy":          q_data.get("usability_easy"),
        "usability_clear":         q_data.get("usability_clear"),
        "usability_comfort":       q_data.get("usability_comfort"),
        "willingness_daily":       q_data.get("willingness_daily"),
        "willingness_recommend":   q_data.get("willingness_recommend"),
        "willingness_timing":      q_data.get("willingness_timing"),
        # Subscale averages computed by the JS before POST
        "score_helpfulness":  q_data.get("score_helpfulness"),
        "score_annoyance":    q_data.get("score_annoyance"),
        "score_usability":    q_data.get("score_usability"),
        "score_willingness":  q_data.get("score_willingness"),
        # Open-ended
        "open_helpful":  q_data.get("open_helpful", ""),
        "open_improve":  q_data.get("open_improve", ""),
        "open_other":    q_data.get("open_other",   ""),
        "submitted_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
 
    file_exists = os.path.exists(QUESTIONNAIRE_MASTER_CSV)
    with open(QUESTIONNAIRE_MASTER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUESTIONNAIRE_FIELDS)
        if not file_exists:
            writer.writeheader()   # write header only on first participant
        writer.writerow(row)

# ╔══════════════════════════════════════════════════════════════════════════════
# ║  ROUTES
# ╚══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calibrate/<session_id>")
def calibrate(session_id):
    if session_id not in calibrations:
        return "Session not found.", 404
    return render_template("calibration.html", session_id=session_id)


@app.route("/monitor/<session_id>")
def monitor(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    sess = sessions[session_id]
    return render_template("monitor.html", session=sess, session_id=session_id)


@app.route("/questionnaire/<session_id>")
def questionnaire(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    return render_template("questionnaire.html", session_id=session_id)


@app.route("/results/<session_id>")
def results(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    summary, _, _ = _save_session(session_id)
    return render_template("results.html", summary=summary, session_id=session_id)


# ── Video feeds ───────────────────────────────────────────────────────────────

@app.route("/video_feed/calibrate/<session_id>")
def video_feed_calibrate(session_id):
    print("video feed calbrate entered")
    return Response(_gen_calibration(session_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feed/monitor/<session_id>")
def video_feed_monitor(session_id):
    return Response(_gen_monitor(session_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    """Create a pending session + calibration object; redirect target is
    /calibrate/<session_id>."""
    d    = request.json
    pid  = d.get("participant_id", f"P{int(time.time())}")
    cond = d.get("condition", "immediate")
    sid  = f"{pid}_{cond}_{int(time.time())}"

    calibrations[sid] = CalibrationSession()
    # Session created but NOT yet active — activated after calibration
    sessions[sid] = _new_session(pid, cond, d.get("age"), d.get("gender"))

    return jsonify({"session_id": sid, "redirect": f"/calibrate/{sid}"})


@app.route("/api/calibration/status/<session_id>", methods=["POST"])
def api_cal_status(session_id):
    print("entered cal status")
    cal = calibrations.get(session_id)
    print("entered cal status 1")
    if not cal:
        return jsonify({"error": "not found"}), 404
    print("entered cal status 2")
    print("Status id cal",id(cal))
    print("Cal summary",cal.summary())
    return jsonify(cal.summary())


@app.route("/api/calibration/advance/<session_id>", methods=["POST"])
def api_cal_advance(session_id):
    """
    Called by the calibration page JS at each stage transition:
      stage=lighting  → begin posture capture
      stage=posture   → finalise baseline, advance to audio
      stage=audio     → play chime test
      stage=done      → activate session, redirect to monitor
    """
    cal  = calibrations.get(session_id)
    sess = sessions.get(session_id)
    if not cal or not sess:
        return jsonify({"error": "not found"}), 404

    action = request.json.get("action", "")

    if action == "start_posture":
        cal.start_posture_capture()
        return jsonify({"ok": True, "stage": cal.stage})

    if action == "finalise":
        baseline = cal.finalize_baseline()
        if baseline is None:
            return jsonify({"ok": False,
                            "error": "Not enough posture samples — please retry."}), 400
        sess["baseline"] = baseline
        return jsonify({"ok": True, "stage": cal.stage, "baseline": baseline})

    if action == "test_audio":
        cal.run_audio_test()
        return jsonify({"ok": True, "stage": cal.stage,
                        "audio_available": audio_alerts.is_available()})

    if action == "confirm_start":
        sess["active"] = True
        sess["start_time"] = time.time()
        return jsonify({"ok": True, "redirect": f"/monitor/{session_id}"})

    return jsonify({"error": f"unknown action '{action}'"}), 400


@app.route("/api/stop_session/<session_id>", methods=["POST"])
def api_stop(session_id):
    sess = sessions.get(session_id)
    if not sess:
        return jsonify({"error": "not found"}), 404
    sess["active"]   = False
    sess["end_time"] = time.time()
    audio_alerts.play_session_end()
    return jsonify({"status": "stopped", "metrics": _live_metrics(session_id)})


@app.route("/api/submit_questionnaire/<session_id>", methods=["POST"])
def api_questionnaire(session_id):
    """
    Receives the questionnaire payload from questionnaire.html and persists it.
    Returns JSON so the frontend can redirect to /results.
    """
    sess = sessions.get(session_id)
    if not sess:
        return jsonify({"error": "session not found"}), 404
 
    q_data = request.json
    if not q_data:
        return jsonify({"error": "empty payload"}), 400
 
    _save_questionnaire(session_id, q_data)
 
    return jsonify({"status": "saved", "redirect": f"/results/{session_id}"})

@app.route("/api/export/<session_id>")
def api_export(session_id):
    summary, ap, sp = _save_session(session_id)
    if summary is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({"summary": summary, "files": {"alerts": ap, "summary": sp}})


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on("connect")
def _on_connect():
    emit("connected", {"ok": True})


@socketio.on("join_session")
def _on_join(data):
    emit("session_joined", {"session_id": data.get("session_id")})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("=" * 60)
    print("  PostureAware — Adaptive Feedback Study System")
    print("  http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True) 
    # out of developomen debug=False