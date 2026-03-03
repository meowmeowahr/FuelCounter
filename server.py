import json
import math
from pathlib import Path
import socket
import threading
from urllib import request

from flask import Flask, render_template, request
from mjpeg_streamer import MjpegServer, Stream

import cv2
import numpy as np

from source import AbstractCamera

app = Flask(__name__)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually send data
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

from scipy.spatial.distance import cdist

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],
                                     [0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


class Track:
    _next_id = 1

    def __init__(self, centroid, radius):
        self.id = Track._next_id
        Track._next_id += 1

        self.kalman = make_kalman()
        self.kalman.statePre  = np.array([[centroid[0]],[centroid[1]],[0.],[0.]], np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()

        self.centroid = centroid
        self.radius = radius
        self.disappeared = 0
        self.counted = False
        self.prev_y = centroid[1]
        self.trail = [centroid]

    def predict(self):
        pred = self.kalman.predict()
        self.centroid = (int(pred[0][0]), int(pred[1][0]))

    def correct(self, centroid, radius):
        meas = np.array([[centroid[0]],[centroid[1]]], np.float32)
        c = self.kalman.correct(meas)
        self.centroid = (int(c[0][0]), int(c[1][0]))
        self.radius = radius
        self.disappeared = 0
        self.trail.append(self.centroid)
        if len(self.trail) > 20:
            self.trail.pop(0)


class CentroidTracker:
    def __init__(self):
        self.tracks = []
        self.ball_count = 0

    def update(self, detections, settings):
        for t in self.tracks:
            t.predict()

        if not detections:
            for t in self.tracks:
                t.disappeared += 1
            self.tracks = [t for t in self.tracks if t.disappeared <= settings["max_disap"]]
            return self.tracks

        det_xy = np.array([(d[0], d[1]) for d in detections], np.float32)

        if not self.tracks:
            for d in detections:
                self.tracks.append(Track((d[0], d[1]), d[2]))
            return self.tracks

        trk_xy = np.array([t.centroid for t in self.tracks], np.float32)
        cost = cdist(trk_xy, det_xy)

        matched_t, matched_d = set(), set()

        while True:
            if cost.size == 0:
                break
            ti, di = np.unravel_index(np.argmin(cost), cost.shape)
            if cost[ti, di] > settings["match_dist"]:
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

        self.tracks = [t for t in self.tracks if t.disappeared <= settings["max_disap"]]
        return self.tracks

    def check_counting(self, line_y):
        for t in self.tracks:
            if t.counted:
                continue
            if t.prev_y < line_y <= t.centroid[1]:
                self.ball_count += 1
                t.counted = True
            t.prev_y = t.centroid[1]

class Server:
    def __init__(self, camera: AbstractCamera, host='0.0.0.0', port=5000, video_port=5800):
        self.host = host
        self.port = port
        self.video_port = video_port

        self.camera = camera

        self.streams = {
            "raw": Stream("raw", camera.resolution, 50, camera.fps),
            "mask": Stream("mask", camera.resolution, 50, camera.fps),
            "visual": Stream("visual", camera.resolution, 50, camera.fps)
        }

        # ==========================
        # CONFIGURABLE SETTINGS
        # ==========================
        self.pipeline_settings = {

            # HSV
            'hsv_h_min': 21, 'hsv_h_max': 73,
            'hsv_s_min': 179, 'hsv_s_max': 255,
            'hsv_v_min': 80, 'hsv_v_max': 255,

            # Contours
            'min_radius': 17,
            'max_radius': 300,

            # Optical Flow
            'max_track_age': 15,
            'min_down_velocity': 2,

            # Counting
            'count_line_y': 300,

            'backlit_mode': False,
            'backlit_thresh': 80,
            'mog2_sens': 40,
            'mog2_history': 200,
            'min_circ': 0.2,
            'max_disap': 19,
            'match_dist': 155,
            'count_line_pct': 85,
        }

        # Tracking state
        self.prev_gray = None
        self.tracks = {}
        self.next_id = 0
        self.count = 0

        self.tracker = CentroidTracker()
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.pipeline_settings["mog2_history"],
            varThreshold=self.pipeline_settings["mog2_sens"],
            detectShadows=False
        )
        self.prev_history = self.pipeline_settings["mog2_history"]

    # ============================================================
    # CONFIG API
    # ============================================================

    def save_config(self):
        Path("./config/").mkdir(parents=True, exist_ok=True)
        with open(Path("./config/pipeline.json", "w")) as f:
            json.dump(self.pipeline_settings, f, indent=4)
        with open(Path("./config/camera.json", "w")) as f:
            json.dump(self.camera.dump(), f, indent=4)

    def serve(self):
        @app.route('/')
        def index():
            return render_template('index.html', ip=get_local_ip(), video_port=self.video_port)

        @app.route('/api/vision/config', methods=['POST'])
        def update_config():
            data = request.get_json()
            for key, value in data.items():
                if key in self.pipeline_settings:
                    self.pipeline_settings[key] = value

                if "mode" == key:
                    self.camera.set_mode(value)

                    for stream in self.streams.values():
                        stream.size = self.camera.resolution
                        stream.fps = self.camera.fps

                if "stream_quality" == key:
                    for stream in self.streams.values():
                        stream.quality = value

            self.save_config()
            return {"status": "success"}

        @app.route('/api/vision/config', methods=['GET'])
        def get_config():
            return [
                {
                    "section": "Camera",
                    "fields": [
                        {
                            "id": "mode",
                            "label": "Camera Mode",
                            "type": "select",
                            "options": [f"{m['res'][0]}x{m['res'][1]}@{m['fps']}" for m in self.camera.modes],
                            "value": self.camera.mode
                        }
                    ]
                },

                {
                    "section": "Stream",
                    "fields": [
                        {
                            "id": "stream_quality",
                            "label": "Stream Quality",
                            "type": "range",
                            "min": 10,
                            "max": 100,
                            "step": 10,
                            "value": self.streams["visual"].quality
                        }
                    ]
                },

                # ======================================================
                # MODE SELECTION
                # ======================================================
                {
                    "section": "Mode",
                    "fields": [
                        {
                            "id": "backlit_mode",
                            "label": "Backlit Mode",
                            "type": "checkbox",
                            "value": self.pipeline_settings["backlit_mode"]
                        }
                    ]
                },

                # ======================================================
                # COLOR / THRESHOLD DETECTION
                # ======================================================
                {
                    "section": "Color Detection (HSV)",
                    "fields": [
                        {"id": "hsv_h_min", "label": "HSV H Min", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_h_min']},
                        {"id": "hsv_h_max", "label": "HSV H Max", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_h_max']},
                        {"id": "hsv_s_min", "label": "HSV S Min", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_s_min']},
                        {"id": "hsv_s_max", "label": "HSV S Max", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_s_max']},
                        {"id": "hsv_v_min", "label": "HSV V Min", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_v_min']},
                        {"id": "hsv_v_max", "label": "HSV V Max", "type": "range", "min": 0, "max": 255, "step": 1, "value": self.pipeline_settings['hsv_v_max']}
                    ]
                },

                # ======================================================
                # BACKLIT / THRESHOLD MODE
                # ======================================================
                {
                    "section": "Backlit Threshold",
                    "fields": [
                        {
                            "id": "backlit_thresh",
                            "label": "Threshold",
                            "type": "range",
                            "min": 0,
                            "max": 255,
                            "step": 1,
                            "value": self.pipeline_settings["backlit_thresh"]
                        }
                    ]
                },

                # ======================================================
                # MOG2 (ROOM LIGHT)
                # ======================================================
                {
                    "section": "Background Subtractor (MOG2)",
                    "fields": [
                        {
                            "id": "mog2_history",
                            "label": "History",
                            "type": "range",
                            "min": 10,
                            "max": 1000,
                            "step": 10,
                            "value": self.pipeline_settings["mog2_history"]
                        },
                        {
                            "id": "mog2_sens",
                            "label": "Sensitivity (VarThreshold)",
                            "type": "range",
                            "min": 1,
                            "max": 200,
                            "step": 1,
                            "value": self.pipeline_settings["mog2_sens"]
                        }
                    ]
                },

                # ======================================================
                # SHAPE FILTERING
                # ======================================================
                {
                    "section": "Shape Filtering",
                    "fields": [
                        {
                            "id": "min_circ",
                            "label": "Min Circularity",
                            "type": "range",
                            "min": 0.1,
                            "max": 1.0,
                            "step": 0.01,
                            "value": self.pipeline_settings["min_circ"]
                        },
                        {
                            "id": "min_radius",
                            "label": "Min Radius",
                            "type": "range",
                            "min": 0,
                            "max": 200,
                            "step": 1,
                            "value": self.pipeline_settings["min_radius"]
                        },
                        {
                            "id": "max_radius",
                            "label": "Max Radius",
                            "type": "range",
                            "min": 0,
                            "max": 300,
                            "step": 1,
                            "value": self.pipeline_settings["max_radius"]
                        }
                    ]
                },

                # ======================================================
                # TRACKING
                # ======================================================
                {
                    "section": "Tracking",
                    "fields": [
                        {
                            "id": "match_dist",
                            "label": "Match Distance",
                            "type": "range",
                            "min": 5,
                            "max": 200,
                            "step": 1,
                            "value": self.pipeline_settings["match_dist"]
                        },
                        {
                            "id": "max_disap",
                            "label": "Max Disappearance Frames",
                            "type": "range",
                            "min": 1,
                            "max": 60,
                            "step": 1,
                            "value": self.pipeline_settings["max_disap"]
                        }
                    ]
                },

                # ======================================================
                # COUNTING
                # ======================================================
                {
                    "section": "Counting",
                    "fields": [
                        {
                            "id": "count_line_pct",
                            "label": "Count Line (%)",
                            "type": "range",
                            "min": 0,
                            "max": 100,
                            "step": 1,
                            "value": self.pipeline_settings["count_line_pct"]
                        }
                    ]
                }
            ]

        mjpeg_server = MjpegServer(self.host, self.video_port)
        for stream in self.streams.values():
            mjpeg_server.add_stream(stream)
        mjpeg_server.start()

        threading.Thread(target=lambda: app.run(host=self.host, port=self.port, debug=True, use_reloader=False),
                         daemon=True).start()

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def spin(self):
        frame = self.camera.get_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Rebuild MOG2 if needed
        if self.pipeline_settings["mog2_history"] != self.prev_history:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.pipeline_settings["mog2_history"],
                varThreshold=self.pipeline_settings["mog2_sens"],
                detectShadows=False
            )
            self.prev_history = self.pipeline_settings["mog2_history"]

        # ===============================
        # DETECTION
        # ===============================
        if self.pipeline_settings["backlit_mode"]:
            _, mask = cv2.threshold(
                gray,
                self.pipeline_settings["backlit_thresh"],
                255,
                cv2.THRESH_BINARY_INV
            )
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv,
                (self.pipeline_settings['hsv_h_min'], self.pipeline_settings['hsv_s_min'], self.pipeline_settings['hsv_v_min']),
                (self.pipeline_settings['hsv_h_max'], self.pipeline_settings['hsv_s_max'], self.pipeline_settings['hsv_v_max'])
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue

            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue

            circ = (4 * np.pi * area) / (perim ** 2)
            if circ < self.pipeline_settings["min_circ"]:
                continue

            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx, cy, r = int(cx), int(cy), int(r)

            if not (self.pipeline_settings["min_radius"] <= r <= self.pipeline_settings["max_radius"]):
                continue

            detections.append((cx, cy, r))

        # ===============================
        # TRACKING
        # ===============================
        tracks = self.tracker.update(detections, self.pipeline_settings)

        line_y = int(frame.shape[0] * self.pipeline_settings["count_line_pct"] / 100)
        self.tracker.check_counting(line_y)

        # ===============================
        # DRAW
        # ===============================
        visual = frame.copy()

        cv2.line(visual, (0, line_y), (frame.shape[1], line_y), (0,255,255), 2)

        for t in tracks:
            col = (0,255,0) if not t.counted else (0,0,255)
            cx, cy = t.centroid

            cv2.circle(visual, (cx,cy), max(t.radius,5), col, 2)
            cv2.putText(visual, f"#{t.id}", (cx-10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        cv2.putText(visual,
                    f"Count: {self.tracker.ball_count}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

        # ===============================
        self.streams["raw"].set_frame(frame)
        self.streams["mask"].set_frame(mask)
        self.streams["visual"].set_frame(visual)