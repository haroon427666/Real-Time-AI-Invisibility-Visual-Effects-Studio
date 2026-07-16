"""Pure utility helpers used by the UI, workers, and tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


def map_canvas_to_frame(
    x: int,
    y: int,
    offset_x: int,
    offset_y: int,
    render_width: int,
    render_height: int,
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[int, int]]:
    local_x = x - offset_x
    local_y = y - offset_y
    if (
        render_width <= 0
        or render_height <= 0
        or local_x < 0
        or local_y < 0
        or local_x >= render_width
        or local_y >= render_height
    ):
        return None
    frame_x = min(frame_width - 1, max(0, int(local_x * frame_width / render_width)))
    frame_y = min(frame_height - 1, max(0, int(local_y * frame_height / render_height)))
    return frame_x, frame_y


def map_split_preview_to_frame(
    x: int,
    y: int,
    offset_x: int,
    offset_y: int,
    render_width: int,
    render_height: int,
    source_width: int,
    source_height: int,
    separator_width: int = 4,
) -> Optional[Tuple[int, int]]:
    combined_width = source_width * 2 + separator_width
    point = map_canvas_to_frame(
        x,
        y,
        offset_x,
        offset_y,
        render_width,
        render_height,
        combined_width,
        source_height,
    )
    if point is None:
        return None
    combined_x, source_y = point
    if combined_x < source_width:
        source_x = combined_x
    elif combined_x >= source_width + separator_width:
        source_x = combined_x - source_width - separator_width
    else:
        return None
    if not (0 <= source_x < source_width and 0 <= source_y < source_height):
        return None
    return source_x, source_y


def median_background(frames: Iterable[np.ndarray]) -> np.ndarray:
    valid = [frame for frame in frames if frame is not None]
    if not valid:
        raise ValueError("No camera frames are available for background capture")
    target_h, target_w = valid[-1].shape[:2]
    aligned = [
        frame
        if frame.shape[:2] == (target_h, target_w)
        else cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in valid
    ]
    return np.median(np.stack(aligned, axis=0), axis=0).astype(np.uint8)


def background_quality(frames: Iterable[np.ndarray]) -> Dict[str, float | str]:
    valid = [frame for frame in frames if frame is not None]
    if not valid:
        return {"score": 0.0, "label": "Empty", "motion": 1.0, "sharpness": 0.0}
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in valid]
    sharpness = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in gray_frames]))
    if len(gray_frames) > 1:
        motion = float(
            np.mean(
                [
                    np.mean(cv2.absdiff(gray_frames[i - 1], gray_frames[i])) / 255.0
                    for i in range(1, len(gray_frames))
                ]
            )
        )
    else:
        motion = 0.0
    brightness = float(np.mean(gray_frames[-1]))
    sharp_score = min(1.0, sharpness / 180.0)
    motion_score = max(0.0, 1.0 - motion * 10.0)
    exposure_score = max(0.0, 1.0 - abs(brightness - 125.0) / 125.0)
    score = float(np.clip(0.45 * sharp_score + 0.4 * motion_score + 0.15 * exposure_score, 0.0, 1.0))
    label = "Excellent" if score >= 0.8 else "Good" if score >= 0.6 else "Fair" if score >= 0.4 else "Poor"
    return {"score": score, "label": label, "motion": motion, "sharpness": sharpness, "brightness": brightness}


def camera_motion_score(reference: np.ndarray, frame: np.ndarray) -> float:
    if reference is None or frame is None:
        return 0.0
    if reference.shape[:2] != frame.shape[:2]:
        reference = cv2.resize(reference, (frame.shape[1], frame.shape[0]))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.resize(ref_gray, (320, 240))
    cur_gray = cv2.resize(cur_gray, (320, 240))
    orb = cv2.ORB_create(nfeatures=400)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(cur_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return float(np.mean(cv2.absdiff(ref_gray, cur_gray)) / 255.0)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    matches = sorted(matches, key=lambda item: item.distance)[:80]
    if len(matches) < 8:
        return float(np.mean(cv2.absdiff(ref_gray, cur_gray)) / 255.0)
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if matrix is None:
        return 0.0
    translation = float(np.hypot(matrix[0, 2], matrix[1, 2])) / 320.0
    rotation_scale_error = float(abs(matrix[0, 0] - 1.0) + abs(matrix[0, 1]))
    inlier_penalty = 1.0 - float(np.mean(inliers)) if inliers is not None else 0.5
    return float(np.clip(translation * 2.5 + rotation_scale_error + inlier_penalty * 0.15, 0.0, 1.0))


def detect_runtime() -> Dict[str, str | bool | float]:
    runtime: Dict[str, str | bool | float] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "opencv_cuda_devices": 0,
        "torch_available": False,
        "gpu": "CPU / not detected",
        "gpu_memory_mb": 0.0,
        "ffmpeg": bool(shutil.which("ffmpeg")),
    }
    try:
        runtime["opencv_cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        pass
    try:
        import torch

        runtime["torch_available"] = True
        if torch.cuda.is_available():
            runtime["gpu"] = torch.cuda.get_device_name(0)
            runtime["gpu_memory_mb"] = float(torch.cuda.memory_allocated(0) / 1024**2)
    except ImportError:
        pass
    return runtime



def safe_imwrite(path: str | os.PathLike, image: np.ndarray) -> None:
    """Write an image and raise when OpenCV reports failure."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    if not cv2.imwrite(str(path), image):
        raise IOError(f"OpenCV could not write image to {path}")


def write_diagnostics(path: str | os.PathLike, payload: Dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime": detect_runtime(),
        **payload,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
