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

import cv2
import numpy as np
from scipy.spatial.distance import cdist

# ==============================================================
# CONFIG
# ==============================================================

CAMERA_INDEX = 1
FRAME_W      = 640
FRAME_H      = 480
TARGET_FPS   = 100
BACKLIT      = False

TB_WIN = "Parameters"

# ==============================================================
# TRACKBARS
# ==============================================================

def nothing(_):
    pass


def create_trackbars():
    cv2.namedWindow(TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TB_WIN, 420, 520)

    # Detection
    cv2.createTrackbar("BL Thresh",     TB_WIN, 80,  255, nothing)
    cv2.createTrackbar("MOG2 Sens",     TB_WIN, 40,  200, nothing)
    cv2.createTrackbar("MOG2 History",  TB_WIN, 200, 500, nothing)

    # Blob filtering
    cv2.createTrackbar("Min Radius",    TB_WIN, 8,   150, nothing)
    cv2.createTrackbar("Max Radius",    TB_WIN, 80,  300, nothing)
    cv2.createTrackbar("Circularity %", TB_WIN, 55,  100, nothing)  # divide by 100

    # Tracker
    cv2.createTrackbar("Max Disap.",    TB_WIN, 10,  60,  nothing)
    cv2.createTrackbar("Match Dist px", TB_WIN, 60,  300, nothing)

    # Count line
    cv2.createTrackbar("Count Line Y%", TB_WIN, 85,  100, nothing)


def get_params():
    return {
        "backlit_thresh":  cv2.getTrackbarPos("BL Thresh",     TB_WIN),
        "mog2_sens":       cv2.getTrackbarPos("MOG2 Sens",     TB_WIN),
        "mog2_history":    max(1, cv2.getTrackbarPos("MOG2 History",  TB_WIN)),
        "min_radius":      max(1, cv2.getTrackbarPos("Min Radius",    TB_WIN)),
        "max_radius":      max(2, cv2.getTrackbarPos("Max Radius",    TB_WIN)),
        "min_circ":        cv2.getTrackbarPos("Circularity %",  TB_WIN) / 100.0,
        "max_disap":       max(1, cv2.getTrackbarPos("Max Disap.",    TB_WIN)),
        "match_dist":      max(1, cv2.getTrackbarPos("Match Dist px", TB_WIN)),
        "count_line_pct":  cv2.getTrackbarPos("Count Line Y%",  TB_WIN),
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

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    print(f"Camera FPS reported: {cap.get(cv2.CAP_PROP_FPS)}")

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read from camera.")
        return

    create_trackbars()

    subtractor    = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    prev_history  = 200
    tracker       = CentroidTracker()

    print("Running.  ESC=quit  B=toggle mode  R=reset count")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        cv2.imshow("Ball Tracker",    display)
        cv2.imshow("Detection Mask",  mask_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('b'):
            BACKLIT = not BACKLIT
            print(f"Mode → {'BACKLIT' if BACKLIT else 'ROOM LIGHT (MOG2)'}")
        elif key == ord('r'):
            tracker.ball_count = 0
            for t in tracker.tracks:
                t.counted = False
            print("Count reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()