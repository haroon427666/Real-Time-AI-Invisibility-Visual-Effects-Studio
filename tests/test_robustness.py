import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import recording
import settings
from effects import GhostEffectEngine, MaskRefiner
from recording import VideoAudioRecorder, VirtualCameraOutput
from workers import (
    CameraWorker,
    ProcessingConfig,
    ProcessingJob,
    ProcessingResult,
    ProcessingWorker,
    WorkerState,
    result_matches_state,
)


class TransientCapture:
    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self.read_count += 1
        if self.read_count <= 2:
            return False, None
        time.sleep(0.002)
        return True, np.full((10, 12, 3), self.index + 1, np.uint8)

    def release(self):
        self.released = True


class PersistentReadFailureCapture(TransientCapture):
    def read(self):
        self.read_count += 1
        return False, None


class TestCameraRecovery(unittest.TestCase):
    @mock.patch("workers.cv2.VideoCapture", TransientCapture)
    def test_temporary_read_failures_recover(self):
        worker = CameraWorker()
        try:
            self.assertTrue(worker.start(index=0, max_read_failures=5))
            deadline = time.monotonic() + 1.0
            frame = None
            while frame is None and time.monotonic() < deadline:
                _, frame = worker.get_latest(-1)
                time.sleep(0.01)
            self.assertIsNotNone(frame)
            self.assertEqual(worker.read_failures, 2)
            self.assertIsNone(worker.error)
            self.assertEqual(worker.state, WorkerState.RUNNING)
        finally:
            worker.stop()

    @mock.patch("workers.cv2.VideoCapture", side_effect=RuntimeError("driver error"))
    def test_camera_constructor_failure_is_reported_without_hanging(self, _capture):
        worker = CameraWorker()
        try:
            started = worker.start(index=0, startup_timeout=0.5)
            self.assertFalse(started)
            self.assertEqual(worker.state, WorkerState.FAILED)
            self.assertIn("Unexpected camera failure", worker.error)
        finally:
            worker.stop()

    @mock.patch("workers.cv2.VideoCapture", PersistentReadFailureCapture)
    def test_persistent_read_failures_transition_to_failed(self):
        worker = CameraWorker()
        try:
            self.assertTrue(worker.start(index=0, max_read_failures=3))
            deadline = time.monotonic() + 1.0
            while worker.state is not WorkerState.FAILED and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(worker.state, WorkerState.FAILED)
            self.assertIn("valid frames", worker.error)
        finally:
            worker.stop()

    @mock.patch("workers.cv2.VideoCapture", TransientCapture)
    def test_restart_clears_previous_frame_state(self):
        worker = CameraWorker()
        try:
            self.assertTrue(worker.start(index=0, max_read_failures=5))
            deadline = time.monotonic() + 1.0
            first = None
            while first is None and time.monotonic() < deadline:
                _, first = worker.get_latest(-1)
                time.sleep(0.01)
            self.assertIsNotNone(first)
            self.assertTrue(worker.start(index=1, max_read_failures=5))
            sequence, immediate = worker.get_latest(-1)
            self.assertEqual(sequence, 0)
            self.assertIsNone(immediate)
            deadline = time.monotonic() + 1.0
            second = None
            while second is None and time.monotonic() < deadline:
                _, second = worker.get_latest(-1)
                time.sleep(0.01)
            self.assertTrue(np.all(second == 2))
        finally:
            worker.stop()


class TestVersionedProcessing(unittest.TestCase):
    def test_config_values_are_clamped(self):
        config = ProcessingConfig.from_value(
            {
                "mode": 99,
                "ai_alpha": -3,
                "ai_threshold": 4,
                "sub_sensitivity": 1000,
                "feather": 100,
                "stability": 5,
                "hsv_lower": [500, -10, 999],
            }
        )
        self.assertEqual(config.mode, 4)
        self.assertEqual(config.ai_alpha, 0.0)
        self.assertLess(config.ai_threshold, 1.0)
        self.assertEqual(config.sub_sensitivity, 255)
        self.assertLessEqual(config.feather, 31)
        self.assertEqual(config.feather % 2, 1)
        self.assertLessEqual(config.stability, 0.98)
        self.assertTrue(np.all(config.hsv_lower[1:] >= 0))

    def test_malformed_config_values_fall_back_safely(self):
        config = ProcessingConfig.from_value(
            {
                "mode": "not-a-number",
                "ai_alpha": float("nan"),
                "ai_threshold": None,
                "feather": "wide",
                "gestures": "false",
                "show_landmarks": "yes",
                "hsv_lower": "invalid",
                "hsv_upper": [float("inf"), -5, 999],
            }
        )
        self.assertEqual(config.mode, 0)
        self.assertEqual(config.ai_alpha, 0.0)
        self.assertEqual(config.ai_threshold, 0.15)
        self.assertEqual(config.feather, 11)
        self.assertFalse(config.gestures)
        self.assertTrue(config.show_landmarks)
        self.assertEqual(config.hsv_lower.tolist(), [35, 40, 40])
        self.assertEqual(config.hsv_upper.tolist(), [75, 0, 255])

    def test_result_generation_and_background_version_are_propagated(self):
        worker = ProcessingWorker()
        try:
            worker.segmenter.available = False
            frame = np.zeros((20, 20, 3), np.uint8)
            worker.submit(
                ProcessingJob(
                    sequence=1,
                    frame=frame,
                    background=None,
                    config={"mode": 2},
                    add_mask=None,
                    remove_mask=None,
                    selection_mask=None,
                    generation=7,
                    background_version=3,
                )
            )
            result = None
            deadline = time.monotonic() + 1.0
            while result is None and time.monotonic() < deadline:
                result = worker.get_result()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertTrue(result_matches_state(result, 7, 3))
            self.assertFalse(result_matches_state(result, 8, 3))
            self.assertFalse(result_matches_state(result, 7, 4))
        finally:
            self.assertTrue(worker.stop())

    def test_ghost_trail_does_not_run_person_segmentation(self):
        worker = ProcessingWorker()
        calls = []
        try:
            worker.segmenter.available = True
            worker.segmenter.get_mask = lambda frame: calls.append(frame) or np.ones(frame.shape[:2], np.float32)
            frame = np.zeros((20, 20, 3), np.uint8)
            worker.submit(
                ProcessingJob(
                    sequence=1,
                    frame=frame,
                    background=None,
                    config={"mode": 2, "gestures": False},
                    add_mask=None,
                    remove_mask=None,
                    selection_mask=None,
                )
            )
            deadline = time.monotonic() + 1.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = worker.get_result()
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertEqual(calls, [])
        finally:
            worker.stop()

    def test_clear_queues_removes_pending_results(self):
        worker = ProcessingWorker()
        try:
            worker.segmenter.available = False
            frame = np.zeros((20, 20, 3), np.uint8)
            worker.submit(
                ProcessingJob(1, frame, None, {"mode": 2}, None, None, None)
            )
            time.sleep(0.05)
            worker.clear_queues()
            self.assertIsNone(worker.get_result())
        finally:
            worker.stop()


class TestSettingsValidation(unittest.TestCase):
    def test_invalid_values_return_safe_defaults(self):
        value = settings.validate_settings(
            {
                "camera_index": -10,
                "resolution": "huge",
                "camera_fps": "fast",
                "camera_buffer": 999,
                "quality": "Ultra",
                "preview_mode": "unknown",
            }
        )
        self.assertEqual(value["camera_index"], 0)
        self.assertEqual(value["resolution"], "640x480")
        self.assertEqual(value["camera_fps"], 30)
        self.assertEqual(value["camera_buffer"], 5)
        self.assertEqual(value["quality"], "Balanced")
        self.assertEqual(value["schema_version"], settings.SETTINGS_SCHEMA_VERSION)

    def test_corrupt_settings_are_preserved_and_defaults_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{broken", encoding="utf-8")
            with mock.patch.object(settings, "SETTINGS_FILE", path):
                loaded = settings.load_settings()
            self.assertEqual(loaded["resolution"], "640x480")
            backups = list(Path(directory).glob("settings.corrupt.*.json"))
            self.assertEqual(len(backups), 1)

    def test_save_then_load_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            with mock.patch.object(settings, "SETTINGS_FILE", path):
                settings.save_settings({"camera_fps": 500, "resolution": "1280x720"})
                loaded = settings.load_settings()
            self.assertEqual(loaded["camera_fps"], 60)
            self.assertEqual(loaded["resolution"], "1280x720")
            self.assertEqual(json.loads(path.read_text())["schema_version"], settings.SETTINGS_SCHEMA_VERSION)

    def test_string_booleans_and_invalid_presets_are_sanitized(self):
        values = settings.validate_settings({"audio_recording": "false", "show_landmarks": "yes"})
        self.assertFalse(values["audio_recording"])
        self.assertTrue(values["show_landmarks"])
        presets = settings.validate_presets({
            "Bad": {"ai_alpha": "oops", "edge_feather": 100, "quality": "Ultra"}
        })
        self.assertEqual(presets["Bad"]["ai_alpha"], 0.0)
        self.assertEqual(presets["Bad"]["edge_feather"], 31)
        self.assertEqual(presets["Bad"]["quality"], "Balanced")


class FileCreatingWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self.path = Path(path)
        self.opened = True
        self.frames = 0

    def isOpened(self):
        return self.opened

    def write(self, frame):
        self.frames += 1

    def release(self):
        self.path.write_bytes(b"video-data")
        self.opened = False


class TestThreadedRecording(unittest.TestCase):
    @mock.patch("recording.cv2.VideoWriter", FileCreatingWriter)
    def test_stop_can_finalize_in_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mp4"
            recorder = VideoAudioRecorder(queue_size=3)
            recorder.start(str(path), (16, 12), fps=10, with_audio=False)
            frame = np.zeros((12, 16, 3), np.uint8)
            start = time.monotonic()
            recorder.write(frame, start)
            recorder.write(frame, start + 0.2)
            result = recorder.stop(wait=False)
            self.assertIn("finalizing", result["message"])
            deadline = time.monotonic() + 2.0
            completed = None
            while completed is None and time.monotonic() < deadline:
                completed = recorder.get_completed()
                time.sleep(0.01)
            self.assertIsNotNone(completed)
            self.assertTrue(path.exists())
            self.assertGreaterEqual(completed["frames"], 2)
            self.assertGreaterEqual(completed["duplicated_frames"], 1)


class FakeVirtualCamera:
    instances = []

    def __init__(self, width, height, fps, fmt):
        self.sent = 0
        self.slept = 0
        self.closed = False
        FakeVirtualCamera.instances.append(self)

    def send(self, frame):
        self.sent += 1

    def sleep_until_next_frame(self):
        self.slept += 1

    def close(self):
        self.closed = True


class TestVirtualCameraPacing(unittest.TestCase):
    def test_virtual_camera_uses_pacing_method(self):
        fake_module = mock.Mock()
        fake_module.Camera = FakeVirtualCamera
        fake_module.PixelFormat.BGR = object()
        with mock.patch.object(recording, "PYVIRTUALCAM_AVAILABLE", True), mock.patch.object(recording, "pyvirtualcam", fake_module):
            output = VirtualCameraOutput()
            output.start(16, 12, 30)
            output.send(np.zeros((12, 16, 3), np.uint8))
            deadline = time.monotonic() + 1.0
            while FakeVirtualCamera.instances[-1].sent == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            output.stop()
            camera = FakeVirtualCamera.instances[-1]
            self.assertGreaterEqual(camera.sent, 1)
            self.assertEqual(camera.sent, camera.slept)
            self.assertTrue(camera.closed)


class TestMaskRobustness(unittest.TestCase):
    def test_nan_and_infinite_masks_produce_finite_output(self):
        engine = GhostEffectEngine()
        frame = np.full((20, 30, 3), 100, np.uint8)
        replacement = np.zeros_like(frame)
        mask = np.zeros((20, 30), np.float32)
        mask[0, 0] = np.nan
        mask[0, 1] = np.inf
        mask[0, 2] = -np.inf
        output = engine.composite(frame, replacement, mask)
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.isfinite(output).all())

    def test_refiner_accepts_random_invalid_values(self):
        rng = np.random.default_rng(3)
        mask = rng.normal(size=(40, 50)).astype(np.float32)
        mask[0, 0] = np.nan
        mask[0, 1] = np.inf
        output = MaskRefiner().refine(mask, expand=0, feather=1, temporal_stability=0)
        self.assertEqual(output.shape, mask.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
