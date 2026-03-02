#!/usr/bin/env python3
"""
CV Pipeline Monitor - Web UI Backend
Flask server with MJPEG streaming, configuration, and freeze mode
"""

import cv2
import numpy as np
import threading
import time
import json
import io
from flask import Flask, Response, render_template_string, request, jsonify
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Try importing picamera2 (falls back to webcam if not available) ───────────
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    logger.info("Picamera2 available")
except ImportError:
    PICAMERA_AVAILABLE = False
    logger.info("Picamera2 not found — using webcam fallback")

app = Flask(__name__)

# ─── Global State ──────────────────────────────────────────────────────────────
class PipelineState:
    def __init__(self):
        self.frozen = False
        self.frozen_frames = {}
        self.frozen_data = None

        self.config = {
            "blur_kernel": 5,
            "canny_low": 50,
            "canny_high": 150,
            "hsv_hue_low": 0,
            "hsv_hue_high": 180,
            "hsv_sat_low": 50,
            "hsv_sat_high": 255,
            "hsv_val_low": 50,
            "hsv_val_high": 255,
            "contour_min_area": 500,
            "brightness": 50,
            "contrast": 50,
        }

        self.data_history = {
            "fps": deque(maxlen=60),
            "contours": deque(maxlen=60),
            "mean_brightness": deque(maxlen=60),
            "timestamps": deque(maxlen=60),
        }

        self.current_data = {
            "fps": 0.0,
            "contours_detected": 0,
            "mean_brightness": 0.0,
            "frame_count": 0,
            "resolution": "0x0",
            "processing_time_ms": 0.0,
            "freeze_status": False,
        }

        self.lock = threading.Lock()
        self._frame_time = time.time()
        self._frame_count = 0

state = PipelineState()

# ─── Camera / Frame Source ─────────────────────────────────────────────────────
class FrameSource:
    """
    Camera source with automatic fallback chain:
      Picamera2  ->  /dev/video0 (V4L2 webcam)  ->  synthetic test pattern

    Picamera2 sensor-timeout / cable errors are detected via a hard per-capture
    wall-clock timeout (PICAM_READ_TIMEOUT). After PICAM_FAIL_THRESHOLD
    consecutive failures the instance is cleanly closed and the next source in
    the chain is opened — no server restart required.
    """

    PICAM_FAIL_THRESHOLD = 5   # consecutive errors before abandoning picam
    PICAM_READ_TIMEOUT   = 2.0 # seconds — covers libcamera's 1 s dequeue timer

    def __init__(self):
        self.cap   = None
        self.picam = None
        self._picam_failures = 0
        self._source_name    = "synthetic"
        self._init_camera()

    # ── Initialisation ────────────────────────────────────────────────────────
    def _init_camera(self):
        if PICAMERA_AVAILABLE:
            try:
                cam = Picamera2()
                cfg = cam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                cam.configure(cfg)
                cam.start()
                self.picam = cam
                self._picam_failures = 0
                self._source_name = "picamera2"
                logger.info("Camera source: Picamera2")
                return
            except Exception as e:
                logger.warning(f"Picamera2 init failed: {e}")
                self._try_stop_picam()

        self._open_webcam()

    def _open_webcam(self):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.cap = cap
                self._source_name = "webcam (/dev/video0)"
                logger.info("Camera source: webcam")
                return
            cap.release()
        except Exception as e:
            logger.warning(f"Webcam open failed: {e}")
        self._source_name = "synthetic"
        logger.warning("Camera source: synthetic test pattern")

    def _try_stop_picam(self):
        if self.picam is not None:
            try: self.picam.stop()
            except Exception: pass
            try: self.picam.close()
            except Exception: pass
            self.picam = None

    # ── Read ──────────────────────────────────────────────────────────────────
    def read(self):
        if self.picam is not None:
            return self._read_picam()
        if self.cap is not None:
            return self._read_webcam()
        return True, self._synthetic_frame()

    def _read_picam(self):
        """Capture with a hard wall-clock timeout via a worker thread."""
        result = [None]
        exc    = [None]

        def _capture():
            try:
                result[0] = self.picam.capture_array()
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_capture, daemon=True)
        t.start()
        t.join(timeout=self.PICAM_READ_TIMEOUT)

        if t.is_alive():
            # libcamera dequeue timer fired and the thread is hanging
            logger.warning("Picamera2 capture timed out (cable/sensor issue)")
            self._picam_failures += 1
        elif exc[0] is not None:
            logger.warning(f"Picamera2 capture error: {exc[0]}")
            self._picam_failures += 1
        else:
            self._picam_failures = 0      # successful read — reset counter
            return True, cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)

        if self._picam_failures >= self.PICAM_FAIL_THRESHOLD:
            logger.error(
                f"Picamera2 failed {self._picam_failures} consecutive times. "
                "Stopping and falling back to webcam / synthetic."
            )
            self._try_stop_picam()
            self._source_name = "picamera2 (error — reconnect cable)"
            self._open_webcam()

        # Return a visible error frame for this cycle
        return True, self._error_frame("CAMERA TIMEOUT — CHECK CABLE")

    def _read_webcam(self):
        try:
            ret, frame = self.cap.read()
        except Exception as e:
            logger.warning(f"Webcam read exception: {e}")
            ret, frame = False, None

        if not ret or frame is None:
            logger.warning("Webcam read failed — falling back to synthetic")
            try: self.cap.release()
            except Exception: pass
            self.cap = None
            self._source_name = "synthetic (webcam lost)"
            return True, self._error_frame("WEBCAM LOST")
        return True, frame

    # ── Synthetic / error frames ──────────────────────────────────────────────
    def _synthetic_frame(self):
        t = time.time()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int((np.sin(t)       * 0.5 + 0.5) * 580) + 30
        y = int((np.cos(t * 0.7) * 0.5 + 0.5) * 420) + 30
        cv2.circle(frame, (x, y), 40, (0, 200, 255), -1)
        cv2.circle(frame, (320, 240), 80, (50, 50, 180), 2)
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        for i in range(0, 640, 80):
            cv2.line(frame, (i, 0), (i, 480), (20, 20, 20), 1)
        for i in range(0, 480, 60):
            cv2.line(frame, (0, i), (640, i), (20, 20, 20), 1)
        cv2.putText(frame, "SYNTHETIC TEST PATTERN", (155, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)
        return frame

    def _error_frame(self, msg="CAMERA ERROR"):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 640 + 480, 20):          # red diagonal stripes
            cv2.line(frame, (i, 0), (i - 480, 480), (40, 0, 0), 10)
        cv2.rectangle(frame, (60, 175), (580, 310), (15, 15, 15), -1)
        cv2.rectangle(frame, (60, 175), (580, 310), (160, 0, 0), 2)
        cv2.putText(frame, msg, (80, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 80, 255), 2)
        cv2.putText(frame, "Check sensor ribbon cable / connector",
                    (80, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        cv2.putText(frame, f"Retrying... ({self._picam_failures}/{self.PICAM_FAIL_THRESHOLD})",
                    (80, 296), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        return frame

    @property
    def source_name(self):
        return self._source_name

    def release(self):
        self._try_stop_picam()
        if self.cap:
            try: self.cap.release()
            except Exception: pass

source = FrameSource()

# ─── Pipeline Processing ───────────────────────────────────────────────────────
latest_frames = {"input": None, "mask": None, "edges": None, "display": None}
frames_lock = threading.Lock()

def process_pipeline(frame, cfg):
    """Run the full CV pipeline and return all layer frames + stats."""
    t0 = time.time()

    # Brightness / contrast adjust
    alpha = cfg["contrast"] / 50.0      # 0–2
    beta = cfg["brightness"] - 50       # -50 to +50
    adjusted = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Blur
    ksize = max(1, cfg["blur_kernel"] | 1)  # ensure odd
    blurred = cv2.GaussianBlur(adjusted, (ksize, ksize), 0)

    # HSV mask
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([cfg["hsv_hue_low"], cfg["hsv_sat_low"], cfg["hsv_val_low"]])
    upper = np.array([cfg["hsv_hue_high"], cfg["hsv_sat_high"], cfg["hsv_val_high"]])
    mask = cv2.inRange(hsv, lower, upper)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Canny edges
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg["canny_low"], cfg["canny_high"])
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored = edges_bgr.copy()
    edges_colored[edges > 0] = [0, 255, 180]

    # Contour detection on mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) > cfg["contour_min_area"]]

    # Display overlay
    display = adjusted.copy()
    cv2.drawContours(display, valid, -1, (0, 255, 100), 2)
    for c in valid:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 80, 0), 1)
        area = cv2.contourArea(c)
        cv2.putText(display, f"{int(area)}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Stats
    mean_brightness = float(np.mean(gray))
    proc_ms = (time.time() - t0) * 1000

    return {
        "input": adjusted,
        "mask": mask_bgr,
        "edges": edges_colored,
        "display": display,
    }, {
        "contours_detected": len(valid),
        "mean_brightness": round(mean_brightness, 1),
        "processing_time_ms": round(proc_ms, 2),
    }

def encode_jpeg(frame, quality=80):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def pipeline_loop():
    """Background thread: capture → process → store frames + stats."""
    fps_timer = time.time()
    frame_count = 0

    while True:
        ret, raw = source.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            fps_timer = now
            frame_count = 0
        else:
            fps = state.current_data.get("fps", 0)

        with state.lock:
            cfg = dict(state.config)
            frozen = state.frozen

        if frozen:
            time.sleep(0.05)
            continue

        layers, stats = process_pipeline(raw, cfg)

        with frames_lock:
            for k, v in layers.items():
                latest_frames[k] = v.copy()

        with state.lock:
            state.current_data.update(stats)
            state.current_data["fps"] = round(fps, 1)
            state.current_data["frame_count"] = state.current_data.get("frame_count", 0) + 1
            state.current_data["resolution"] = f"{raw.shape[1]}x{raw.shape[0]}"
            state.current_data["freeze_status"] = False

            t = now
            state.data_history["fps"].append(round(fps, 1))
            state.data_history["contours"].append(stats["contours_detected"])
            state.data_history["mean_brightness"].append(stats["mean_brightness"])
            state.data_history["timestamps"].append(round(t, 2))

        time.sleep(0.01)

# Start pipeline thread
threading.Thread(target=pipeline_loop, daemon=True).start()

# ─── MJPEG Stream Generator ────────────────────────────────────────────────────
def gen_stream(layer):
    while True:
        with state.lock:
            frozen = state.frozen
            frozen_frames = state.frozen_frames

        if frozen and layer in frozen_frames:
            frame = frozen_frames[layer]
        else:
            with frames_lock:
                frame = latest_frames.get(layer)

        if frame is None:
            time.sleep(0.05)
            continue

        jpg = encode_jpeg(frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1 / 30)

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/stream/<layer>")
def stream(layer):
    valid = {"input", "mask", "edges", "display"}
    if layer not in valid:
        return "Invalid layer", 400
    return Response(gen_stream(layer),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/data")
def api_data():
    with state.lock:
        data = dict(state.current_data)
        history = {k: list(v) for k, v in state.data_history.items()}
        data["freeze_status"] = state.frozen
    data["camera_source"] = source.source_name
    data["camera_ok"] = "error" not in source.source_name and "lost" not in source.source_name
    return jsonify({"current": data, "history": history})

@app.route("/api/config", methods=["GET", "POST"])
def api_config():
    if request.method == "POST":
        body = request.json or {}
        with state.lock:
            for k, v in body.items():
                if k in state.config:
                    state.config[k] = int(v)
        return jsonify({"status": "ok"})
    with state.lock:
        return jsonify(state.config)

@app.route("/api/freeze", methods=["POST"])
def api_freeze():
    with state.lock:
        state.frozen = not state.frozen
        if state.frozen:
            with frames_lock:
                state.frozen_frames = {k: v.copy() for k, v in latest_frames.items() if v is not None}
            state.frozen_data = dict(state.current_data)
        status = state.frozen
    return jsonify({"frozen": status})

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

# ─── HTML Template ─────────────────────────────────────────────────────────────
import os
_here = os.path.dirname(os.path.abspath(__file__))
HTML_TEMPLATE = open(os.path.join(_here, "index.html")).read()

if __name__ == "__main__":
    logger.info("Starting CV Pipeline Monitor on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
