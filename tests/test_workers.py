import time
import unittest
from unittest import mock

import numpy as np

from workers import CameraWorker, ProcessingJob, ProcessingWorker


class FakeCapture:
    instances = []

    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self.opened = True
        self.released = False
        self.read_count = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        return True

    def read(self):
        self.read_count += 1
        if self.read_count > 4:
            time.sleep(0.005)
        return True, np.full((12, 16, 3), self.index, np.uint8)

    def release(self):
        self.released = True


class FailedCapture(FakeCapture):
    def isOpened(self):
        return False

    def read(self):
        return False, None


class TestCameraWorker(unittest.TestCase):
    def tearDown(self):
        FakeCapture.instances.clear()

    @mock.patch("workers.cv2.VideoCapture", FakeCapture)
    def test_starting_again_stops_previous_capture(self):
        worker = CameraWorker()
        worker.start(index=0, width=16, height=12, fps=30, backend="Auto", buffer=1, exposure=-1, focus=-1)
        time.sleep(0.03)
        worker.start(index=1, width=16, height=12, fps=30, backend="Auto", buffer=1, exposure=-1, focus=-1)
        time.sleep(0.03)
        worker.stop()
        self.assertGreaterEqual(len(FakeCapture.instances), 2)
        self.assertTrue(all(instance.released for instance in FakeCapture.instances))

    @mock.patch("workers.cv2.VideoCapture", FailedCapture)
    def test_camera_disconnect_open_failure_is_reported(self):
        worker = CameraWorker()
        worker.start(index=0, width=16, height=12, fps=30, backend="Auto", buffer=1, exposure=-1, focus=-1)
        time.sleep(0.02)
        worker.stop()
        self.assertIsNotNone(worker.error)

    @mock.patch("workers.cv2.VideoCapture", FakeCapture)
    def test_camera_detection_respects_exclusion(self):
        cameras = CameraWorker.detect_cameras(3, exclude={1})
        self.assertEqual(cameras, [0, 2])


class TestProcessingWorker(unittest.TestCase):
    def test_motion_fallback_produces_target_mask(self):
        worker = ProcessingWorker()
        try:
            worker.segmenter.available = False
            background = np.zeros((40, 40, 3), np.uint8)
            frame = background.copy()
            frame[10:30, 12:28] = 255
            worker.submit(
                ProcessingJob(
                    sequence=2,
                    frame=frame,
                    background=background,
                    config={
                        "mode": 0,
                        "model_backend": "Auto",
                        "sub_sensitivity": 10,
                        "ai_threshold": 0.1,
                        "expand": 0,
                        "feather": 1,
                        "stability": 0.0,
                    },
                    add_mask=None,
                    remove_mask=None,
                    selection_mask=None,
                )
            )
            result = None
            deadline = time.time() + 1.0
            while result is None and time.time() < deadline:
                result = worker.get_result()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.mask)
            self.assertGreater(result.target_area, 0.05)
        finally:
            worker.stop()

    def test_empty_invalid_model_result_does_not_crash(self):
        worker = ProcessingWorker()
        try:
            worker.segmenter.available = False
            frame = np.zeros((20, 20, 3), np.uint8)
            worker.submit(
                ProcessingJob(
                    sequence=1,
                    frame=frame,
                    background=None,
                    config={"mode": 0, "model_backend": "Auto"},
                    add_mask=None,
                    remove_mask=None,
                    selection_mask=None,
                )
            )
            result = None
            deadline = time.time() + 1.0
            while result is None and time.time() < deadline:
                result = worker.get_result()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertIsNone(result.mask)
            self.assertEqual(result.final.shape, frame.shape)
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
