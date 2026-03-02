"""
Ball Detection & Tracking Pipeline
===================================
Supports:
  - Room lighting (front-lit, MOG2 background subtraction)
  - Backlit mode (simple threshold, faster & more robust)
  - Centroid-based tracker with Kalman filtering per track
  - Debug overlay: detections, tracks, IDs, velocity vectors
  - Counting: increment when ball crosses a defined Y line
  - Live OpenCV trackbars for all key parameters

Keybindings:
  ESC — quit
  B   — toggle backlit / room light mode
  R   — reset count
"""

import threading
import socket

import cv2
from picamera2 import Picamera2
from mjpeg_streamer import MjpegServer, Stream
import numpy as np
from scipy.spatial.distance import cdist

from flask import Flask, render_template

# ==============================================================
# CONFIG
# ==============================================================
FRAME_W = 640
FRAME_H = 400
BACKLIT      = False

TB_WIN = "Parameters"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send data
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html', ip=get_local_ip())


# ==============================================================
# TRACKBARS
# ==============================================================

def get_params():
    return {
        "backlit_thresh":  80,
        "mog2_sens":       40,
        "mog2_history":    200,
        "min_radius":      8,
        "max_radius":      120,
        "min_circ":        5.5,
        "max_disap":       10,
        "match_dist":      60,
        "count_line_pct":  85,
    }


# ==============================================================
# KALMAN FILTER
# ==============================================================

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                      [0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


# ==============================================================
# TRACK
# ==============================================================

class Track:
    _next_id = 1

    def __init__(self, centroid, radius):
        self.id   = Track._next_id
        Track._next_id += 1
        self.kalman = make_kalman()
        self.kalman.statePre  = np.array([[centroid[0]],[centroid[1]],[0.],[0.]], dtype=np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()
        self.centroid    = centroid
        self.radius      = radius
        self.disappeared = 0
        self.counted     = False
        self.prev_y      = centroid[1]
        self.trail       = [centroid]

    def predict(self):
        pred = self.kalman.predict()
        self.centroid = (int(pred[0][0]), int(pred[1][0]))

    def correct(self, centroid, radius):
        meas = np.array([[centroid[0]],[centroid[1]]], dtype=np.float32)
        c = self.kalman.correct(meas)
        self.centroid    = (int(c[0][0]), int(c[1][0]))
        self.radius      = radius
        self.disappeared = 0
        self.trail.append(self.centroid)
        if len(self.trail) > 20:
            self.trail.pop(0)


# ==============================================================
# TRACKER
# ==============================================================

class CentroidTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.ball_count = 0

    def update(self, detections, p):
        for t in self.tracks:
            t.predict()

        if not detections:
            for t in self.tracks:
                t.disappeared += 1
            self.tracks = [t for t in self.tracks if t.disappeared <= p["max_disap"]]
            return self.tracks

        det_xy = np.array([(d[0], d[1]) for d in detections], dtype=np.float32)

        if not self.tracks:
            for d in detections:
                self.tracks.append(Track((d[0], d[1]), d[2]))
            return self.tracks

        trk_xy = np.array([t.centroid for t in self.tracks], dtype=np.float32)
        cost   = cdist(trk_xy, det_xy)

        matched_t, matched_d = set(), set()
        while True:
            if cost.size == 0:
                break
            ti, di = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[ti, di] > p["match_dist"]:
                break
            if ti in matched_t or di in matched_d:
                cost[ti, di] = 1e9
                continue
            self.tracks[ti].correct((detections[di][0], detections[di][1]), detections[di][2])
            matched_t.add(ti)
            matched_d.add(di)
            cost[ti, di] = 1e9

        for ti, t in enumerate(self.tracks):
            if ti not in matched_t:
                t.disappeared += 1

        for di, d in enumerate(detections):
            if di not in matched_d:
                self.tracks.append(Track((d[0], d[1]), d[2]))

        self.tracks = [t for t in self.tracks if t.disappeared <= p["max_disap"]]
        return self.tracks

    def check_counting(self, tracks, line_y):
        for t in tracks:
            if t.counted:
                continue
            if t.prev_y < line_y <= t.centroid[1]:   # downward crossing
                self.ball_count += 1
                t.counted = True
            t.prev_y = t.centroid[1]


# ==============================================================
# DETECTION
# ==============================================================

def detect_backlit(gray, p):
    _, mask = cv2.threshold(gray, p["backlit_thresh"], 255, cv2.THRESH_BINARY_INV)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def detect_roomlight(gray, p):
    _, fg = cv2.threshold(gray, p["backlit_thresh"], 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fg


def extract_detections(mask, p, dbg=None):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = (4 * np.pi * area) / (perim ** 2)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), int(r)

        rejected_color = None
        if not (p["min_radius"] <= r <= p["max_radius"]):
            rejected_color = (0, 0, 80)
        elif circ < p["min_circ"]:
            rejected_color = (0, 0, 140)

        if rejected_color:
            if dbg is not None:
                cv2.circle(dbg, (cx, cy), max(r,1), rejected_color, 1)
            continue

        dets.append((cx, cy, r))
    return dets


# ==============================================================
# OVERLAY
# ==============================================================

COLORS = [
    (0,255,0),(255,128,0),(0,200,255),(255,0,200),
    (128,255,0),(0,128,255),(255,255,0),(0,255,180),
    (200,0,255),(255,80,80)
]

def draw_overlay(frame, tracks, detections, ball_count, line_y, p):
    ov = frame.copy()

    # Count line
    cv2.line(ov, (0, line_y), (FRAME_W, line_y), (0,255,255), 1)
    cv2.putText(ov, "COUNT LINE", (5, line_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # Raw detections (grey outline)
    for cx, cy, r in detections:
        cv2.circle(ov, (cx, cy), r, (180,180,180), 1)

    for t in tracks:
        col = COLORS[t.id % len(COLORS)]
        cx, cy = t.centroid
        r = max(t.radius, p["min_radius"])

        # Trail
        for i in range(1, len(t.trail)):
            a = i / len(t.trail)
            cv2.line(ov, t.trail[i-1], t.trail[i],
                     tuple(int(v*a) for v in col), 1)

        # Circle (thinner when coasting)
        cv2.circle(ov, (cx, cy), r, col, 2 if t.disappeared == 0 else 1)

        # Velocity arrow
        st = t.kalman.statePost
        vx, vy = float(st[2][0]), float(st[3][0])
        speed  = np.sqrt(vx**2 + vy**2)
        cv2.arrowedLine(ov, (cx, cy),
                        (int(cx + vx*4), int(cy + vy*4)),
                        col, 1, tipLength=0.3)

        # Labels
        lbl = f"#{t.id}" + (" ✓" if t.counted else "")
        cv2.putText(ov, lbl,         (cx-10, cy-r-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(ov, f"{speed:.1f}px/f", (cx-10, cy+r+12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        print(f"#{t.id}: pos=({cx},{cy}) r={r} v=({vx:.1f},{vy:.1f}) spd={speed:.1f} disp={t.disappeared} count={t.counted}")

    # HUD
    mode = "BACKLIT" if BACKLIT else "ROOM LIGHT (MOG2)"
    for i, line in enumerate([
        f"Mode: {mode}  [B]=toggle",
        f"Active tracks: {sum(1 for t in tracks if t.disappeared==0)}",
        f"Total counted: {ball_count}  [R]=reset",
        f"Circ>{p['min_circ']:.2f}  R:{p['min_radius']}-{p['max_radius']}px",
    ]):
        cv2.putText(ov, line, (10, 20+i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    return ov


# ==============================================================
# MAIN
# ==============================================================

def main():
    global BACKLIT

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (640, 400), "format": "RGB888"},
        controls={"FrameRate": 240}
    )
    cam.configure(config)
    cam.start()

    stream_cam = Stream("camera", size=(640, 400), quality=70, fps=240)
    stream_disp = Stream("display", size=(640, 400), quality=70, fps=240)
    stream_mask = Stream("mask", size=(640, 400), quality=70, fps=240)

    server = MjpegServer("0.0.0.0", 5000)
    server.add_stream(stream_cam)
    server.add_stream(stream_disp)
    server.add_stream(stream_mask)
    server.start()

    # create_trackbars()

    subtractor    = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    prev_history  = 200
    tracker       = CentroidTracker()

    print("Running.  ESC=quit  B=toggle mode  R=reset count")

    while True:
        frame_rgb = cam.capture_array()
        frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        stream_cam.set_frame(frame_rgb)

        p = get_params()

        # Rebuild subtractor if history changed (can't update in-place)
        if p["mog2_history"] != prev_history:
            subtractor   = cv2.createBackgroundSubtractorMOG2(
                history=p["mog2_history"], varThreshold=p["mog2_sens"], detectShadows=False)
            prev_history = p["mog2_history"]

        line_y = int(FRAME_H * p["count_line_pct"] / 100)
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask       = detect_backlit(gray, p) if BACKLIT else detect_roomlight(gray, p)
        detections = extract_detections(mask, p, dbg=frame)
        tracks     = tracker.update(detections, p)
        tracker.check_counting(tracks, line_y)

        display  = draw_overlay(frame, tracks, detections, tracker.ball_count, line_y, p)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for cx, cy, r in detections:
            cv2.circle(mask_bgr, (cx, cy), r, (0,255,0), 2)

        # cv2.imshow("Ball Tracker",    display)
        stream_disp.set_frame(display)
        stream_mask.set_frame(mask)
        # cv2.imshow("Detection Mask",  mask_bgr)

        # key = cv2.waitKey(1) & 0xFF
        # if key == 27:
        #     break
        # elif key == ord('b'):
        #     BACKLIT = not BACKLIT
        #     print(f"Mode → {'BACKLIT' if BACKLIT else 'ROOM LIGHT (MOG2)'}")
        # elif key == ord('r'):
        #     tracker.ball_count = 0
        #     for t in tracker.tracks:
        #         t.counted = False
        #     print("Count reset.")

    server.stop()
    cap.release()
    cv2.destroyAllWindows()

def server():
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    threading.Thread(target=server, daemon=True).start()
    main()
