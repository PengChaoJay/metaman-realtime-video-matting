import unittest

import cv2
import numpy as np

from app.frame_processor import jpeg_to_transparent_png


class HalfMaskRemover:
    def mask(self, rgb: np.ndarray) -> np.ndarray:
        alpha = np.zeros(rgb.shape[:2], dtype=np.uint8)
        alpha[:, rgb.shape[1] // 2 :] = 255
        return alpha

    def close(self) -> None:
        pass


class FrameProcessorTests(unittest.TestCase):
    def test_jpeg_frame_becomes_transparent_png(self) -> None:
        source = np.full((16, 20, 3), 180, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        result = jpeg_to_transparent_png(jpeg.tobytes(), HalfMaskRemover())
        decoded = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        self.assertEqual(decoded.shape, (16, 20, 4))
        self.assertTrue(np.all(decoded[:, :10, 3] == 0))
        self.assertTrue(np.all(decoded[:, 10:, 3] == 255))

