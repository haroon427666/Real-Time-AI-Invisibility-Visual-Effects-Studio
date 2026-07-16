import unittest
from unittest import mock

import numpy as np

import recording
from recording import VideoAudioRecorder, VirtualCameraOutput


class ClosedWriter:
    def isOpened(self):
        return False

    def release(self):
        pass


class TestRecordingFailures(unittest.TestCase):
    @mock.patch("recording.cv2.VideoWriter", return_value=ClosedWriter())
    def test_writer_initialization_failure_is_reported(self, _writer):
        recorder = VideoAudioRecorder()
        with self.assertRaises(IOError):
            recorder.start("bad.mp4", (16, 12), with_audio=False)

    def test_virtual_camera_missing_dependency_is_reported(self):
        output = VirtualCameraOutput()
        with mock.patch.object(recording, "PYVIRTUALCAM_AVAILABLE", False):
            with self.assertRaises(RuntimeError):
                output.start(16, 12, 30)
