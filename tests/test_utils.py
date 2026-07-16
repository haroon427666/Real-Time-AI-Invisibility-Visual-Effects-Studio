import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from utils import (
    background_quality,
    detect_runtime,
    map_canvas_to_frame,
    map_split_preview_to_frame,
    median_background,
    safe_imwrite,
    write_diagnostics,
)


class TestCanvasMapping(unittest.TestCase):
    def test_letterbox_coordinates(self):
        point = map_canvas_to_frame(
            150, 100,
            offset_x=50, offset_y=20,
            render_width=400, render_height=300,
            frame_width=800, frame_height=600,
        )
        self.assertEqual(point, (200, 160))

    def test_click_outside_rendered_image_is_ignored(self):
        self.assertIsNone(
            map_canvas_to_frame(10, 10, 50, 20, 400, 300, 800, 600)
        )

    def test_split_preview_maps_both_halves_to_same_source(self):
        left = map_split_preview_to_frame(50, 50, 0, 0, 204, 100, 100, 100, 4)
        right = map_split_preview_to_frame(154, 50, 0, 0, 204, 100, 100, 100, 4)
        self.assertEqual(left, (50, 50))
        self.assertEqual(right, (50, 50))

    def test_split_preview_separator_is_not_editable(self):
        self.assertIsNone(
            map_split_preview_to_frame(101, 50, 0, 0, 204, 100, 100, 100, 4)
        )


class TestBackgroundUtilities(unittest.TestCase):
    def test_temporal_median_rejects_outlier(self):
        frames = [np.full((4, 4, 3), 20, np.uint8) for _ in range(4)]
        frames.append(np.full((4, 4, 3), 250, np.uint8))
        output = median_background(frames)
        self.assertTrue(np.all(output == 20))

    def test_empty_background_raises(self):
        with self.assertRaises(ValueError):
            median_background([])

    def test_background_quality_has_expected_fields(self):
        frames = [np.zeros((20, 20, 3), np.uint8) for _ in range(3)]
        result = background_quality(frames)
        self.assertIn("score", result)
        self.assertIn("label", result)


class TestEngineeringHelpers(unittest.TestCase):
    def test_screenshot_write_failure_raises(self):
        with mock.patch("utils.cv2.imwrite", return_value=False):
            with self.assertRaises(IOError):
                safe_imwrite("bad.png", np.zeros((5, 5, 3), np.uint8))

    def test_diagnostics_report_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_diagnostics(path, {"metrics": {"fps": 30}})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["metrics"]["fps"], 30)
            self.assertIn("runtime", data)

    def test_runtime_detection_works_without_gpu(self):
        result = detect_runtime()
        self.assertIn("gpu", result)
        self.assertIn("opencv", result)


if __name__ == "__main__":
    unittest.main()
