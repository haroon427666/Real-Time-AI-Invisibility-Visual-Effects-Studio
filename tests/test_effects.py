import unittest
from unittest import mock

import cv2
import numpy as np

from effects import GhostEffectEngine, MaskRefiner, SelfieSegmenter


class TestGhostEffectEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GhostEffectEngine()
        self.frame = np.full((40, 60, 3), 200, dtype=np.uint8)
        self.background = np.full((20, 30, 3), 20, dtype=np.uint8)

    def test_alpha_composite(self):
        mask = np.ones((40, 60), dtype=np.float32)
        output = self.engine.composite(self.frame, np.zeros_like(self.frame), mask, alpha=0.5)
        self.assertTrue(np.all(output == 100))

    def test_background_resolution_mismatch_is_resized(self):
        mask = np.ones((40, 60), dtype=np.float32)
        output = self.engine.composite(self.frame, self.background, mask, alpha=0.0)
        self.assertEqual(output.shape, self.frame.shape)
        self.assertTrue(np.all(output == 20))

    def test_freeze_requires_real_mask(self):
        with self.assertRaises(ValueError):
            self.engine.capture_freeze(self.frame, None)

    def test_freeze_rejects_empty_mask(self):
        with self.assertRaises(ValueError):
            self.engine.capture_freeze(self.frame, np.zeros((40, 60), np.float32))

    def test_freeze_soft_edge_is_not_multiplied_twice(self):
        mask = np.full((40, 60), 0.5, dtype=np.float32)
        self.engine.capture_freeze(self.frame, mask)
        live = np.zeros_like(self.frame)
        output = self.engine.apply_time_freeze(live, alpha=1.0)
        self.assertTrue(np.all(output == 100))

    def test_red_hue_wraparound(self):
        hsv = np.zeros((10, 20, 3), dtype=np.uint8)
        hsv[:, :10] = (178, 255, 255)
        hsv[:, 10:] = (2, 255, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = self.engine.build_hsv_mask(
            bgr,
            np.array([165, 40, 40]),
            np.array([15, 255, 255]),
        )
        self.assertEqual(int(np.min(mask)), 255)

    def test_invalid_segmenter_result_keeps_frame(self):
        class InvalidSegmenter:
            def get_mask(self, frame):
                return None

        output, mask = self.engine.apply_ai_invisibility(
            self.frame,
            np.zeros_like(self.frame),
            InvalidSegmenter(),
        )
        self.assertIsNone(mask)
        self.assertTrue(np.array_equal(output, self.frame))


class TestMaskRefiner(unittest.TestCase):
    def test_mask_range_shape_and_dtype(self):
        refiner = MaskRefiner()
        mask = np.zeros((50, 70), dtype=np.float64)
        mask[10:40, 20:55] = 1.5
        output = refiner.refine(mask, expand=0, feather=1, temporal_stability=0)
        self.assertEqual(output.shape, (50, 70))
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_small_component_is_removed(self):
        refiner = MaskRefiner()
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        mask[2:4, 2:4] = 1.0
        output = refiner.refine(mask, expand=0, feather=1, temporal_stability=0)
        self.assertLess(output[2:4, 2:4].mean(), 0.1)
        self.assertGreater(output[30:70, 30:70].mean(), 0.9)


class TestSelfieSegmenterFailure(unittest.TestCase):
    def test_unavailable_model_returns_none(self):
        segmenter = SelfieSegmenter.__new__(SelfieSegmenter)
        segmenter.available = False
        segmenter.model = None
        self.assertIsNone(segmenter.get_mask(np.zeros((10, 10, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()

class TestGestureDebounce(unittest.TestCase):
    def test_gesture_requires_temporal_majority(self):
        import threading
        from collections import deque
        from effects import HandGestureController

        class FakeResult:
            multi_hand_landmarks = [object()]
            multi_handedness = None

        class FakeModel:
            def process(self, frame):
                return FakeResult()

        controller = HandGestureController.__new__(HandGestureController)
        controller.available = True
        controller.model = FakeModel()
        controller._lock = threading.Lock()
        controller._gesture_history = deque(maxlen=5)
        controller.classify_hand = mock.Mock(return_value=("2 Fingers", 0.4))
        frame = np.zeros((8, 8, 3), np.uint8)

        _, first, _, _ = controller.process_frame(frame)
        _, second, _, _ = controller.process_frame(frame)
        _, third, _, _ = controller.process_frame(frame)
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(third, "2 Fingers")
