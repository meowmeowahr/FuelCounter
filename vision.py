from server import Server
from source import AbstractCamera

import cv2

if __name__ == "__main__":

    class DummyCamera(AbstractCamera):
        def __init__(self) -> None:
            super().__init__()
            self.cap = cv2.VideoCapture(0)

        def get_frame(self) -> cv2.typing.MatLike:
            # return np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Failed to read from camera")
            return frame

        @property
        def fps(self) -> int:
            return int(self.cap.get(cv2.CAP_PROP_FPS))

    camera = DummyCamera()
    server = Server(camera)
    server.serve()

    while True:
        server.spin()
