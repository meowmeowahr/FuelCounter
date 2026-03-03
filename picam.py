import enum
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

            # Configure for RGB frames (OpenCV compatible)
            config = self.picam2.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameRate": 30}
            )

            self.picam2.configure(config)
            self._modes = self.picam2.sensor_modes
            self.picam2.start()

            # Give sensor time to warm up
            time.sleep(0.5)

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
            metadata = self.picam2.capture_metadata()
            if "FrameDuration" in metadata:
                return int(1_000_000 / metadata["FrameDuration"])
            raise RuntimeError("Failed to get frame duration from metadata")
        
        @property
        def modes(self) -> list[dict[str, Any]]:
            return [{"res": m["size"], "fps": m["fps"]} for m in self._modes]
        
        @property
        def mode(self) -> str:
            config = self.picam2.camera_configuration()
            size = config["main"]["size"]
            fps = self.fps
            return f"{size[0]}x{size[1]}@{fps}"
        
        def set_mode(self, mode: str) -> None:
            # Parse mode string (e.g., "640x480@30")
            res_str, fps_str = mode.split("@")
            width, height = map(int, res_str.split("x"))
            fps = float(fps_str)

            config = self.picam2.create_video_configuration(
                main={"size": (width, height), "format": "BGR888"},
                controls={"FrameRate": fps}
            )
            self.picam2.stop()
            self.picam2.configure(config)
            self.picam2.start()

        def dump(self) -> dict:
            controls = self.picam2.camera_config["controls"]
            main = self.picam2.camera_config["main"]

            print(controls)
            print(main)

            data = {
                "controls": {

                },
                "main": main
            }

            return data


    camera = PiCamera()
    server = Server(camera)
    server.serve()

    while True:
        server.spin()
