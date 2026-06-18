"""
Adaptive Delayed Feedback System for Posture Improvement
Flask + MediaPipe + WebSocket backend
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
import math
import os
import csv
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'posture-study-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ─── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

# ─── Session State ───────────────────────────────────────────────────────────────
sessions = {}

def create_session(participant_id, condition, age=None, gender=None):
    """Initialize a new participant session."""
    return {
        'participant_id': participant_id,
        'condition': condition,  # 'immediate' or 'adaptive'
        'age': age,
        'gender': gender,
        'start_time': time.time(),
        'end_time': None,

        # Posture tracking
        'current_posture_good': True,
        'posture_bad_start': None,
        'total_poor_posture_duration': 0.0,

        # Feedback state
        'feedback_active': False,
        'feedback_shown_at': None,
        'last_alert_time': None,
        'correction_pending': False,

        # Adaptive delay parameters
        'current_delay_threshold': 5.0,   # seconds; starts at 5s
        'min_delay': 2.0,
        'max_delay': 20.0,
        'alpha': 0.3,                     # EWM smoothing factor

        # Per-alert log
        'alerts': [],
        'current_alert': None,

        # Metrics
        'total_alerts': 0,
        'response_latencies': [],
        'ignored_alerts': 0,   # alerts where user didn't correct within 30s

        # Session metrics snapshots (every 30s)
        'metric_snapshots': [],

        'active': False,
    }


# ─── Posture Analysis ─────────────────────────────────────────────────────────
def compute_angle(a, b, c):
    """Angle at vertex b formed by segments ba and bc (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def analyze_posture(landmarks, image_w, image_h):
    """
    Extract posture metrics from MediaPipe Pose landmarks.
    Returns a dict with metric values and an overall 'good' boolean.
    """
    #print("==================Entered analyze_posture==================")
    lm = landmarks.landmark

    def pt(idx):
        return [lm[idx].x * image_w, lm[idx].y * image_h]

    # Key points
    nose       = pt(mp_pose.PoseLandmark.NOSE)
    left_ear   = pt(mp_pose.PoseLandmark.LEFT_EAR)
    right_ear  = pt(mp_pose.PoseLandmark.RIGHT_EAR)
    left_sh    = pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_sh   = pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_hip   = pt(mp_pose.PoseLandmark.LEFT_HIP)
    right_hip  = pt(mp_pose.PoseLandmark.RIGHT_HIP)

    # Visibility check
    key_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    min_vis = min(lm[i].visibility for i in key_indices)
    if min_vis < 0.5:
        print("Not enough visibility")
        return None  # Not enough visibility

    # ── Metric 1: Forward-head angle ──────────────────────────────────
    # Mid-ear → mid-shoulder vertical deviation
    mid_ear = [(left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2]
    mid_sh  = [(left_sh[0] + right_sh[0]) / 2,  (left_sh[1] + right_sh[1]) / 2]
    # Reference vertical above mid_sh
    vertical_ref = [mid_sh[0], mid_sh[1] - 100]
    fha = compute_angle(mid_ear, mid_sh, vertical_ref)
    # fha close to 0 = upright, increases as head moves forward

    # ── Metric 2: Shoulder alignment (horizontal tilt) ────────────────
    sh_dx = abs(left_sh[0] - right_sh[0])
    sh_dy = abs(left_sh[1] - right_sh[1])
    shoulder_tilt = math.degrees(math.atan2(sh_dy, sh_dx + 1e-6))
    # Near 0 = level shoulders; increases as one shoulder rises/falls

    # ── Metric 3: Torso inclination ───────────────────────────────────
    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    torso_dx = mid_sh[0] - mid_hip[0]
    torso_dy = mid_sh[1] - mid_hip[1]
    # Angle of torso from vertical (should be near 0 when sitting upright)
    torso_inclination = abs(math.degrees(math.atan2(torso_dx, -torso_dy + 1e-6)))

    # ── Classification thresholds ──────────────────────────────────────
    FHA_THRESHOLD         = 25.0   # degrees forward head
    SHOULDER_THRESHOLD    = 10.0   # degrees tilt
    TORSO_THRESHOLD       = 15.0   # degrees lean

    fha_bad       = fha > FHA_THRESHOLD
    sh_bad        = shoulder_tilt > SHOULDER_THRESHOLD
    torso_bad     = torso_inclination > TORSO_THRESHOLD

    if fha_bad:
        print("Forward head bad posture detected")
    if sh_bad:
        print("tilt bad posture detected")
    if torso_bad:
        print("lean bad psoture detected")

    # Overall: bad if any two or more metrics are bad, or FHA alone is very bad
    issues = sum([fha_bad, sh_bad, torso_bad])
    posture_good = not (issues >= 2 or fha > 35)

    # Compute a 0-100 score (100 = perfect)
    fha_score     = max(0, 100 - (fha / FHA_THRESHOLD) * 50)
    sh_score      = max(0, 100 - (shoulder_tilt / SHOULDER_THRESHOLD) * 50)
    torso_score   = max(0, 100 - (torso_inclination / TORSO_THRESHOLD) * 50)
    overall_score = int((fha_score * 0.5 + sh_score * 0.25 + torso_score * 0.25))
    overall_score = max(0, min(100, overall_score))

    return {
        'good': posture_good,
        'score': overall_score,
        'fha': round(fha, 1),
        'shoulder_tilt': round(shoulder_tilt, 1),
        'torso_inclination': round(torso_inclination, 1),
        'fha_bad': fha_bad,
        'sh_bad': sh_bad,
        'torso_bad': torso_bad,
        'min_visibility': round(min_vis, 2),
    }


# ─── Adaptive Delay Engine ────────────────────────────────────────────────────
def update_delay_threshold(session, response_latency):
    """
    Exponentially weighted moving average adjustment of delay threshold.
    - Fast responders (latency < threshold) → delay increases (fewer alerts)
    - Slow/ignored responders → delay decreases (more frequent alerts)
    """
    alpha = session['alpha']
    current = session['current_delay_threshold']

    if response_latency is None:
        # Ignored: decrease delay significantly
        target = current * 0.7
    elif response_latency < current:
        # Responded quickly relative to threshold: reward with longer delay
        ratio = response_latency / current
        target = current * (1 + (1 - ratio) * 0.5)
    else:
        # Slow response: decrease delay
        target = current * 0.85

    new_threshold = alpha * target + (1 - alpha) * current
    new_threshold = max(session['min_delay'], min(session['max_delay'], new_threshold))
    session['current_delay_threshold'] = round(new_threshold, 2)
    return new_threshold


# ─── Feedback Decision Logic ──────────────────────────────────────────────────
def process_posture_update(session_id, posture_data, timestamp):
    """
    Core state machine: decides when to fire alerts based on condition.
    Returns dict of actions to push to client.
    """
    if session_id not in sessions:
        return {}

    sess = sessions[session_id]
    if not sess['active']:
        return {}

    actions = {}
    posture_good = posture_data['good']

    # ── Transition: good → bad ────────────────────────────────────────
    if not posture_good and sess['current_posture_good']:
        sess['posture_bad_start'] = timestamp
        sess['current_posture_good'] = False

    # ── Transition: bad → good ────────────────────────────────────────
    elif posture_good and not sess['current_posture_good']:
        # Accumulate poor posture duration
        if sess['posture_bad_start']:
            duration = timestamp - sess['posture_bad_start']
            sess['total_poor_posture_duration'] += duration
            sess['posture_bad_start'] = None

        # Log correction latency if feedback was pending
        if sess['correction_pending'] and sess['feedback_shown_at']:
            latency = timestamp - sess['feedback_shown_at']
            sess['response_latencies'].append(round(latency, 2))

            if sess['current_alert']:
                sess['current_alert']['correction_time'] = timestamp
                sess['current_alert']['latency'] = round(latency, 2)
                sess['current_alert']['corrected'] = True
                sess['alerts'].append(sess['current_alert'])
                sess['current_alert'] = None

            # Update adaptive delay for adaptive condition
            if sess['condition'] == 'adaptive':
                update_delay_threshold(sess, latency)

            sess['correction_pending'] = False
            sess['feedback_active'] = False
            actions['hide_alert'] = True
            actions['latency_recorded'] = round(latency, 2)
            actions['new_delay'] = round(sess['current_delay_threshold'], 1)

        sess['current_posture_good'] = True

    # ── Check if alert should fire ─────────────────────────────────────
    if not posture_good and not sess['feedback_active']:
        bad_duration = timestamp - (sess['posture_bad_start'] or timestamp)

        if sess['condition'] == 'immediate':
            should_alert = bad_duration >= 1.0  # 1s grace period
        else:  # adaptive
            should_alert = bad_duration >= sess['current_delay_threshold']

        if should_alert:
            # Rate limit: don't re-alert within 3 seconds
            if not sess['last_alert_time'] or (timestamp - sess['last_alert_time']) >= 3.0:
                sess['feedback_active'] = True
                sess['feedback_shown_at'] = timestamp
                sess['last_alert_time'] = timestamp
                sess['correction_pending'] = True
                sess['total_alerts'] += 1

                alert_record = {
                    'alert_id': sess['total_alerts'],
                    'shown_at': timestamp,
                    'condition': sess['condition'],
                    'delay_threshold': sess['current_delay_threshold'],
                    'corrected': False,
                    'latency': None,
                }
                sess['current_alert'] = alert_record

                actions['show_alert'] = True
                actions['alert_id'] = sess['total_alerts']
                actions['current_delay'] = round(sess['current_delay_threshold'], 1)

    # ── Check for ignored alert (timeout at 30s) ───────────────────────
    if sess['feedback_active'] and sess['feedback_shown_at']:
        time_since_alert = timestamp - sess['feedback_shown_at']
        if time_since_alert > 30.0 and sess['correction_pending']:
            sess['ignored_alerts'] += 1
            if sess['current_alert']:
                sess['current_alert']['corrected'] = False
                sess['current_alert']['latency'] = None
                sess['alerts'].append(sess['current_alert'])
                sess['current_alert'] = None

            # Adaptive: reduce delay even more for ignored
            if sess['condition'] == 'adaptive':
                update_delay_threshold(sess, None)
                actions['new_delay'] = round(sess['current_delay_threshold'], 1)

            sess['correction_pending'] = False
            sess['feedback_active'] = False
            actions['alert_expired'] = True

    return actions


# ─── Video Stream ─────────────────────────────────────────────────────────────
def generate_frames(session_id):
    """Generator that yields annotated MJPEG frames."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 24)

    last_emit = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow('Video Feed', frame)

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        posture_data = None
        if results.pose_landmarks:
            posture_data = analyze_posture(results.pose_landmarks, w, h)

            if posture_data:
                # Draw skeleton overlay
                # color = (0, 220, 80) if posture_data['good'] else (50, 50, 255)
                # mp_drawing.draw_landmarks(
                #     frame,
                #     results.pose_landmarks,
                #     mp_pose.POSE_CONNECTIONS,
                #     mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                #     mp_drawing.DrawingSpec(color=color, thickness=2)
                # )

                # # Score badge
                # score_text = f"Score: {posture_data['score']}"
                # cv2.rectangle(frame, (8, 8), (175, 42), (0, 0, 0), -1)
                # cv2.putText(frame, score_text, (14, 32),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                # Emit posture update via SocketIO (throttled to 5 Hz)
                now = time.time()
                if now - last_emit > 0.2:
                    last_emit = now
                    actions = process_posture_update(session_id, posture_data, now)
                    #print("POSTURE UPDATE EMITTED****************************")
                    socketio.emit('posture_update', {
                        'posture': posture_data,
                        'actions': actions,
                        'timestamp': now,
                        'session_metrics': get_live_metrics(session_id),
                    })
                    socketio.sleep(2) #non-blocking sleep required by eventlet/gevent
                    #print(sessions[session_id])

        # _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
        #        buffer.tobytes() + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_live_metrics(session_id):
    if session_id not in sessions:
        return {}
    sess = sessions[session_id]
    elapsed = time.time() - sess['start_time'] if sess['start_time'] else 0
    avg_latency = (sum(sess['response_latencies']) / len(sess['response_latencies'])
                   if sess['response_latencies'] else None)
    return {
        'elapsed': round(elapsed, 0),
        'total_alerts': sess['total_alerts'],
        'avg_latency': round(avg_latency, 1) if avg_latency else None,
        'total_poor_duration': round(sess['total_poor_posture_duration'], 1),
        'current_delay': round(sess['current_delay_threshold'], 1),
        'condition': sess['condition'],
        'ignored': sess['ignored_alerts'],
    }


# ─── CSV Data Export ──────────────────────────────────────────────────────────
def save_session_data(session_id):
    if session_id not in sessions:
        return
    sess = sessions[session_id]

    os.makedirs('data', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    pid = sess['participant_id']

    # Alert log CSV
    alert_path = f"data/{pid}_{sess['condition']}_{ts}_alerts.csv"
    with open(alert_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'alert_id', 'shown_at', 'condition', 'delay_threshold',
            'corrected', 'latency', 'correction_time'
        ])
        writer.writeheader()
        for a in sess['alerts']:
            writer.writerow({
                'alert_id': a.get('alert_id'),
                'shown_at': round(a.get('shown_at', 0), 3),
                'condition': a.get('condition'),
                'delay_threshold': a.get('delay_threshold'),
                'corrected': a.get('corrected'),
                'latency': a.get('latency'),
                'correction_time': round(a.get('correction_time', 0), 3) if a.get('correction_time') else None,
            })

    # Summary JSON
    elapsed = (sess['end_time'] or time.time()) - sess['start_time']
    summary = {
        'participant_id': pid,
        'condition': sess['condition'],
        'age': sess['age'],
        'gender': sess['gender'],
        'session_duration_s': round(elapsed, 1),
        'total_alerts': sess['total_alerts'],
        'ignored_alerts': sess['ignored_alerts'],
        'total_poor_posture_s': round(sess['total_poor_posture_duration'], 1),
        'pct_time_poor_posture': round(
            sess['total_poor_posture_duration'] / elapsed * 100, 1) if elapsed > 0 else 0,
        'response_latencies': sess['response_latencies'],
        'avg_latency_s': round(sum(sess['response_latencies']) / len(sess['response_latencies']), 2)
            if sess['response_latencies'] else None,
        'median_latency_s': round(float(np.median(sess['response_latencies'])), 2)
            if sess['response_latencies'] else None,
        'final_delay_threshold': sess['current_delay_threshold'],
    }
    summary_path = f"data/{pid}_{sess['condition']}_{ts}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary, alert_path, summary_path


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor/<session_id>')
def monitor(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    sess = sessions[session_id]
    return render_template('monitor.html', session=sess, session_id=session_id)

@app.route('/questionnaire/<session_id>')
def questionnaire(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    return render_template('questionnaire.html', session_id=session_id)

@app.route('/results/<session_id>')
def results(session_id):
    if session_id not in sessions:
        return "Session not found.", 404
    summary, _, _ = save_session_data(session_id)
    return render_template('results.html', summary=summary, session_id=session_id)

@app.route('/video_feed/<session_id>')
def video_feed(session_id):
    return Response(generate_frames(session_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── API Endpoints ────────────────────────────────────────────────────────────
@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.json
    participant_id = data.get('participant_id', f"P{int(time.time())}")
    condition = data.get('condition', 'immediate')  # 'immediate' or 'adaptive'
    age = data.get('age')
    gender = data.get('gender')

    session_id = f"{participant_id}_{condition}_{int(time.time())}"
    sessions[session_id] = create_session(participant_id, condition, age, gender)
    sessions[session_id]['active'] = True

    return jsonify({'session_id': session_id, 'status': 'started'})

@app.route('/api/stop_session/<session_id>', methods=['POST'])
def stop_session(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    sessions[session_id]['active'] = False
    sessions[session_id]['end_time'] = time.time()
    metrics = get_live_metrics(session_id)
    return jsonify({'status': 'stopped', 'metrics': metrics})

@app.route('/api/submit_questionnaire/<session_id>', methods=['POST'])
def submit_questionnaire(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    data = request.json
    sessions[session_id]['questionnaire'] = data

    # Append questionnaire to summary on save
    summary, _, summary_path = save_session_data(session_id)
    summary['questionnaire'] = data
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return jsonify({'status': 'saved', 'redirect': f'/results/{session_id}'})

@app.route('/api/metrics/<session_id>')
def get_metrics(session_id):
    return jsonify(get_live_metrics(session_id))

@app.route('/api/export/<session_id>')
def export_data(session_id):
    if session_id not in sessions:
        return jsonify({'error': 'not found'}), 404
    summary, alert_path, summary_path = save_session_data(session_id)
    return jsonify({
        'summary': summary,
        'files': {'alerts': alert_path, 'summary': summary_path}
    })


# ─── SocketIO Events ──────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    emit('connected', {'status': 'ok'})

@socketio.on('join_session')
def on_join(data):
    session_id = data.get('session_id')
    emit('session_joined', {'session_id': session_id, 'status': 'ok'})

# @socketio.on('posture_update')
# def on_posture_update(data):
#     print("Posture update called")
#     print("Received:",data.posture," ",data.actions," ",data.timestamp," ",data.session_metrics)

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("=" * 60)
    print("  Posture Study System")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
