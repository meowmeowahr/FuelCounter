import socket
import threading

from flask import Flask, render_template
from mjpeg_streamer import MjpegServer, Stream

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

    def serve(self):
        @app.route('/')
        def index():
            return render_template('index.html', ip=get_local_ip(), video_port=self.video_port)

        mjpeg_server = MjpegServer(self.host, self.video_port)
        for stream in self.streams.values():
            mjpeg_server.add_stream(stream)

        mjpeg_server.start()
        def run_flask():
            app.run(host=self.host, port=self.port, debug=True, use_reloader=False)
        flask_thread = threading.Thread(target=run_flask, daemon=True)
        flask_thread.start()

    def spin(self):
        frame = self.camera.get_frame()
        self.streams["raw"].set_frame(frame)