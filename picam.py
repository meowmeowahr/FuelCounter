import json
from pathlib import Path
import time
from typing import Any
import numpy as np
import cv2

from picamera2 import Picamera2

from server import Server
from source import AbstractCamera

if __name__ == "__main__":

    class PiCamera(AbstractCamera):
        def __init__(self) -> None:
            super().__init__()

            self.picam2 = Picamera2()

            self.camera_settings = {
                "size": [640, 480],
            }
            if Path("./config/camera.json").exists():
                with open(Path("./config/camera.json"), "r") as f:
                    self.camera_settings.update(json.load(f))
                with open(Path("./config/camera.json"), "w") as f:
                    json.dump(self.camera_settings, f, indent=4)
            else:
                Path("./config/").mkdir(exist_ok=True, parents=False)
                Path("./config/camera.json").touch()
                with open(Path("./config/camera.json"), "w") as f:
                    json.dump(self.camera_settings, f, indent=4)

            # Configure for RGB frames (OpenCV compatible)
            config = self.picam2.create_video_configuration(
                main={"size": self.camera_settings["size"], "format": "BGR888"},
                controls={"FrameRate": 200},
            )

            self._modes = self.picam2.sensor_modes
            self.picam2.stop()
            self.picam2.configure(config)
            self.picam2.start()

            print(self._modes)

            # Give sensor time to warm up
            time.sleep(0.5)

        def get_config(self) -> dict[str, Any]:
            return self.camera_settings

        def save_config(self, config: dict[str, Any]) -> None:
            self.camera_settings.update(config)
            with open(Path("./config/camera.json"), "w") as f:
                json.dump(self.camera_settings, f, indent=4)

            # Reconfigure camera with new settings
            config = self.picam2.create_video_configuration(
                main={"size": self.camera_settings["size"], "format": "BGR888"},
                controls={"FrameRate": 200},
            )
            self.picam2.stop()
            self.picam2.configure(config)
            self.picam2.start()

        def get_frame(self) -> np.ndarray:
            frame = self.picam2.capture_array()

            if frame is None:
                raise RuntimeError("Failed to capture frame")

            # Picamera2 gives RGB, OpenCV expects BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        @property
        def resolution(self) -> tuple[int, int]:
            return self.picam2.camera_configuration()["main"]["size"]

        @property
        def fps(self) -> int:
            return 90
            metadata = self.picam2.capture_metadata()
            if "FrameDuration" in metadata:
                return int(1_000_000 / metadata["FrameDuration"])
            raise RuntimeError("Failed to get frame duration from metadata")

    camera = PiCamera()
    server = Server(camera)
    server.serve()

    while True:
        server.spin()
