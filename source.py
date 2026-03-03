from abc import ABC, abstractmethod
from typing import Any
import cv2

import warnings

class AbstractCamera(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_frame(self) -> cv2.typing.MatLike:
        pass

    @property
    def resolution(self) -> tuple[int, int]:
        warnings.warn("Default resolution getter blocks for one frame. Override this method if you want to avoid that.")
        frame = self.get_frame()
        return frame.shape[1], frame.shape[0]
    
    @property
    def fps(self) -> float:
        raise NotImplementedError("Default fps getter is not implemented. Override this method if you want to use it.")
