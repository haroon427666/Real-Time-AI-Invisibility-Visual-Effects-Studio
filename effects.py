"""Computer-vision models, mask refinement, gestures, and ghost effects."""

from __future__ import annotations

from collections import Counter, deque
import logging
import math
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    mp = None
    MEDIAPIPE_AVAILABLE = False


def _odd(value: int, minimum: int = 1) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


class SelfieSegmenter:
    """Safe wrapper around MediaPipe Selfie Segmentation."""

    def __init__(self, model_selection: int = 0):
        self.available = MEDIAPIPE_AVAILABLE
        self.model = None
        self._lock = threading.Lock()
        if self.available:
            try:
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.model = self.mp_selfie.SelfieSegmentation(
                    model_selection=int(model_selection)
                )
            except Exception:
                LOGGER.exception("Could not initialize MediaPipe Selfie Segmentation")
                self.available = False

    def get_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a float32 person-confidence mask in [0, 1]."""
        if not self.available or self.model is None or frame is None:
            return None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                results = self.model.process(rgb_frame)
            if results.segmentation_mask is not None:
                return np.clip(
                    results.segmentation_mask.astype(np.float32), 0.0, 1.0
                )
        except Exception:
            LOGGER.exception("MediaPipe segmentation failed")
        return None

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                LOGGER.exception("Failed to close MediaPipe segmenter")
            finally:
                self.model = None


class HandGestureController:
    """MediaPipe Hands wrapper with orientation-aware joint-angle gestures."""

    def __init__(self, history_size: int = 7):
        self.available = MEDIAPIPE_AVAILABLE
        self.model = None
        self._lock = threading.Lock()
        self._gesture_history: deque[Optional[str]] = deque(maxlen=history_size)
        if self.available:
            try:
                self.mp_hands = mp.solutions.hands
                self.mp_drawing = mp.solutions.drawing_utils
                self.model = self.mp_hands.Hands(
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                )
            except Exception:
                LOGGER.exception("Could not initialize MediaPipe Hands")
                self.available = False

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
        if denom < 1e-8:
            return 0.0
        cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def process_frame(self, frame: np.ndarray):
        """Return landmarks, stable gesture, pinch value, and handedness."""
        if not self.available or self.model is None:
            return None, None, 1.0, None
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                results = self.model.process(rgb_frame)
            if not results.multi_hand_landmarks:
                self._gesture_history.append(None)
                return None, None, 1.0, None

            landmarks = results.multi_hand_landmarks[0]
            handedness = None
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
            gesture, pinch_value = self.classify_hand(landmarks, handedness)
            self._gesture_history.append(gesture)

            non_empty = [g for g in self._gesture_history if g]
            stable = None
            if non_empty:
                candidate, votes = Counter(non_empty).most_common(1)[0]
                if votes >= max(3, len(self._gesture_history) // 2):
                    stable = candidate
            return landmarks, stable, pinch_value, handedness
        except Exception:
            LOGGER.exception("Hand processing failed")
            return None, None, 1.0, None

    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> np.ndarray:
        if not self.available or hand_landmarks is None:
            return frame
        output = frame.copy()
        self.mp_drawing.draw_landmarks(
            output,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(
                color=(181, 173, 0), thickness=2, circle_radius=2
            ),
            self.mp_drawing.DrawingSpec(
                color=(99, 46, 255), thickness=2, circle_radius=2
            ),
        )
        return output

    def classify_hand(self, landmarks, handedness: Optional[str] = None):
        points = np.array(
            [[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32
        )

        def distance(i: int, j: int) -> float:
            return float(np.linalg.norm(points[i] - points[j]))

        hand_scale = max(distance(0, 9), 1e-3)
        finger_triplets = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
        raised = []
        for mcp, pip, tip in finger_triplets:
            angle = self._angle(points[mcp], points[pip], points[tip])
            extended_from_wrist = distance(0, tip) > distance(0, pip) * 1.08
            raised.append(angle > 150.0 and extended_from_wrist)

        thumb_angle = self._angle(points[2], points[3], points[4])
        thumb_spread = distance(4, 5) / hand_scale
        lateral_extension = True
        if handedness == "Right":
            lateral_extension = points[4, 0] < points[3, 0]
        elif handedness == "Left":
            lateral_extension = points[4, 0] > points[3, 0]
        thumb_extended = thumb_angle > 145.0 and thumb_spread > 0.45 and lateral_extension

        pinch_ratio = distance(4, 8) / hand_scale
        pinch_value = float(np.clip((pinch_ratio - 0.18) / 0.7, 0.0, 1.0))

        # A real thumbs-up needs an extended thumb pointing mostly upward in image
        # coordinates, while all four fingers are folded. This remains independent
        # of left/right handedness.
        thumb_vector = points[4, :2] - points[2, :2]
        points_up = thumb_vector[1] < -abs(thumb_vector[0]) * 0.55
        if thumb_extended and points_up and not any(raised):
            return "Thumbs Up", 1.0

        if not thumb_extended and not any(raised):
            return "Fist", 0.0

        if pinch_ratio < 0.34 and any(raised[1:]):
            return "Pinch", pinch_value

        total_fingers = int(thumb_extended) + sum(bool(value) for value in raised)
        names = {
            1: "1 Finger",
            2: "2 Fingers",
            3: "3 Fingers",
            4: "4 Fingers",
            5: "5 Fingers (Palm)",
        }
        return names.get(total_fingers), pinch_value

    def close(self) -> None:
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                LOGGER.exception("Failed to close MediaPipe Hands")
            finally:
                self.model = None


class MaskRefiner:
    """Reusable spatial and temporal matte cleanup pipeline."""

    def __init__(self):
        self.previous_mask: Optional[np.ndarray] = None

    @staticmethod
    def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
        binary = (mask >= 0.35).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 1:
            return mask
        keep = np.zeros_like(binary)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
                keep[labels == label] = 1
        return mask * keep.astype(np.float32)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        binary = (mask >= 0.35).astype(np.uint8) * 255
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flood = padded.copy()
        flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
        filled = cv2.bitwise_or(binary, holes).astype(np.float32) / 255.0
        return np.maximum(mask, filled)

    def refine(
        self,
        mask: np.ndarray,
        guide_frame: Optional[np.ndarray] = None,
        expand: int = 1,
        feather: int = 11,
        temporal_stability: float = 0.65,
        minimum_area_ratio: float = 0.001,
        reset_temporal: bool = False,
    ) -> np.ndarray:
        if mask is None:
            raise ValueError("Mask cannot be None")
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.clip(mask, 0.0, 1.0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        minimum_area = max(8, int(mask.size * max(0.0, minimum_area_ratio)))
        mask = self._remove_small_components(mask, minimum_area)
        mask = self._fill_holes(mask)

        if expand > 0:
            mask = cv2.dilate(mask, kernel, iterations=int(expand))
        elif expand < 0:
            mask = cv2.erode(mask, kernel, iterations=abs(int(expand)))

        # Edge-aware filtering where OpenCV contrib is available; otherwise use
        # a bilateral filter followed by a compact Gaussian feather.
        if guide_frame is not None and hasattr(cv2, "ximgproc"):
            guide = cv2.cvtColor(guide_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            try:
                mask = cv2.ximgproc.guidedFilter(guide, mask, radius=6, eps=1e-3)
            except cv2.error:
                LOGGER.debug("Guided filter unavailable at runtime", exc_info=True)
        else:
            mask = cv2.bilateralFilter(mask, 7, 0.08, 7)

        if feather > 1:
            kernel_size = _odd(feather, 3)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        stability = float(np.clip(temporal_stability, 0.0, 0.98))
        if reset_temporal or self.previous_mask is None or self.previous_mask.shape != mask.shape:
            refined = mask
        else:
            refined = stability * self.previous_mask + (1.0 - stability) * mask
        self.previous_mask = np.clip(refined, 0.0, 1.0).astype(np.float32)
        return self.previous_mask.copy()

    def reset(self) -> None:
        self.previous_mask = None


class GhostEffectEngine:
    """All existing ghost effects with safe compositing and mask refinement."""

    def __init__(self):
        self.accumulator: Optional[np.ndarray] = None
        self.frozen_person: Optional[np.ndarray] = None
        self.frozen_mask: Optional[np.ndarray] = None
        self.refiners = {
            "ai": MaskRefiner(),
            "cloak": MaskRefiner(),
            "motion": MaskRefiner(),
        }
        self._lock = threading.RLock()

    @staticmethod
    def _prepare_background(frame: np.ndarray, background: Optional[np.ndarray]):
        if background is None:
            return None
        if background.shape[:2] != frame.shape[:2]:
            return cv2.resize(
                background,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return background

    @staticmethod
    def _apply_manual_mask(
        mask: np.ndarray,
        add_mask: Optional[np.ndarray] = None,
        remove_mask: Optional[np.ndarray] = None,
        selection_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = np.nan_to_num(
            mask.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
        )
        result = np.clip(result, 0.0, 1.0)
        size = (result.shape[1], result.shape[0])
        if selection_mask is not None:
            selected = _resize_mask(selection_mask, size)
            result *= np.clip(selected, 0.0, 1.0)
        if add_mask is not None:
            result = np.maximum(result, np.clip(_resize_mask(add_mask, size), 0.0, 1.0))
        if remove_mask is not None:
            result *= 1.0 - np.clip(_resize_mask(remove_mask, size), 0.0, 1.0)
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def composite(
        frame: np.ndarray,
        replacement: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.0,
    ) -> np.ndarray:
        if replacement.shape[:2] != frame.shape[:2]:
            replacement = cv2.resize(replacement, (frame.shape[1], frame.shape[0]))
        matte = np.nan_to_num(
            mask.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
        )
        matte = np.clip(matte, 0.0, 1.0)
        if matte.ndim == 2:
            matte = matte[..., None]
        effective = matte * (1.0 - float(np.clip(alpha, 0.0, 1.0)))
        output = replacement.astype(np.float32) * effective + frame.astype(np.float32) * (1.0 - effective)
        return np.clip(output, 0, 255).astype(np.uint8)

    def capture_freeze(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> None:
        """Capture the actual target frame and matte; never invent a dummy mask."""
        if frame is None:
            raise ValueError("A valid frame is required")
        if mask is None:
            raise ValueError("No valid person/target mask is available to freeze")
        mask = np.asarray(mask, dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != frame.shape[:2]:
            mask = _resize_mask(mask, (frame.shape[1], frame.shape[0]))
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.clip(mask, 0.0, 1.0)
        if float(np.mean(mask > 0.35)) < 0.001:
            raise ValueError("The current target mask is empty")
        with self._lock:
            # Store the original frame. The mask is applied once during compositing,
            # avoiding the old frame * mask^2 dark-edge defect.
            self.frozen_person = frame.copy()
            self.frozen_mask = mask.copy()

    def clear_freeze(self) -> None:
        with self._lock:
            self.frozen_person = None
            self.frozen_mask = None

    @staticmethod
    def build_hsv_mask(
        frame: np.ndarray, hsv_lower: np.ndarray, hsv_upper: np.ndarray
    ) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.asarray(hsv_lower, dtype=np.int32)
        upper = np.asarray(hsv_upper, dtype=np.int32)
        lower[1:] = np.clip(lower[1:], 0, 255)
        upper[1:] = np.clip(upper[1:], 0, 255)
        lower_h, upper_h = int(lower[0]) % 180, int(upper[0]) % 180
        if lower_h <= upper_h:
            return cv2.inRange(
                hsv,
                np.array([lower_h, lower[1], lower[2]], dtype=np.uint8),
                np.array([upper_h, upper[1], upper[2]], dtype=np.uint8),
            )
        # Hue wraps around red: [lower_h, 179] U [0, upper_h].
        first = cv2.inRange(
            hsv,
            np.array([lower_h, lower[1], lower[2]], dtype=np.uint8),
            np.array([179, upper[1], upper[2]], dtype=np.uint8),
        )
        second = cv2.inRange(
            hsv,
            np.array([0, lower[1], lower[2]], dtype=np.uint8),
            np.array([upper_h, upper[1], upper[2]], dtype=np.uint8),
        )
        return cv2.bitwise_or(first, second)

    def apply_color_cloak(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        hsv_lower: np.ndarray,
        hsv_upper: np.ndarray,
        alpha: float = 0.0,
        expand: int = 1,
        feather: int = 11,
        stability: float = 0.65,
        add_mask: Optional[np.ndarray] = None,
        remove_mask: Optional[np.ndarray] = None,
        selection_mask: Optional[np.ndarray] = None,
    ):
        background = self._prepare_background(frame, background)
        if background is None:
            return frame.copy(), None
        mask = self.build_hsv_mask(frame, hsv_lower, hsv_upper).astype(np.float32) / 255.0
        mask = self._apply_manual_mask(mask, add_mask, remove_mask, selection_mask)
        refined = self.refiners["cloak"].refine(
            mask, frame, expand=expand, feather=feather, temporal_stability=stability
        )
        return self.composite(frame, background, refined, alpha), refined

    def apply_ai_invisibility(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        segmenter: Optional[SelfieSegmenter],
        threshold: float = 0.1,
        alpha: float = 0.0,
        precomputed_mask: Optional[np.ndarray] = None,
        expand: int = 1,
        feather: int = 11,
        stability: float = 0.65,
        add_mask: Optional[np.ndarray] = None,
        remove_mask: Optional[np.ndarray] = None,
        selection_mask: Optional[np.ndarray] = None,
    ):
        background = self._prepare_background(frame, background)
        if background is None:
            return frame.copy(), None
        mask = precomputed_mask
        if mask is None and segmenter is not None:
            mask = segmenter.get_mask(frame)
        if mask is None:
            return frame.copy(), None
        threshold = float(np.clip(threshold, 0.0, 0.99))
        mask = np.clip((mask.astype(np.float32) - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
        mask = self._apply_manual_mask(mask, add_mask, remove_mask, selection_mask)
        refined = self.refiners["ai"].refine(
            mask, frame, expand=expand, feather=feather, temporal_stability=stability
        )
        return self.composite(frame, background, refined, alpha), refined

    def apply_ghost_trail(self, frame: np.ndarray, decay_rate: float = 0.2):
        decay_rate = float(np.clip(decay_rate, 0.01, 1.0))
        if self.accumulator is None or self.accumulator.shape != frame.shape:
            self.accumulator = frame.astype(np.float32)
            return frame.copy()
        cv2.accumulateWeighted(frame, self.accumulator, decay_rate)
        return cv2.convertScaleAbs(self.accumulator)

    def apply_time_freeze(self, frame: np.ndarray, alpha: float = 0.5):
        with self._lock:
            if self.frozen_person is None or self.frozen_mask is None:
                return frame.copy()
            frozen_frame = self.frozen_person
            frozen_mask = self.frozen_mask
        if frozen_frame.shape[:2] != frame.shape[:2]:
            frozen_frame = cv2.resize(frozen_frame, (frame.shape[1], frame.shape[0]))
            frozen_mask = _resize_mask(frozen_mask, (frame.shape[1], frame.shape[0]))
        matte = np.clip(frozen_mask * float(np.clip(alpha, 0.0, 1.0)), 0.0, 1.0)
        matte_3d = matte[..., None]
        output = frame.astype(np.float32) * (1.0 - matte_3d) + frozen_frame.astype(np.float32) * matte_3d
        return np.clip(output, 0, 255).astype(np.uint8)

    def apply_background_subtraction_invisibility(
        self,
        frame: np.ndarray,
        background: Optional[np.ndarray],
        threshold_val: int = 25,
        alpha: float = 0.0,
        expand: int = 1,
        feather: int = 11,
        stability: float = 0.65,
        add_mask: Optional[np.ndarray] = None,
        remove_mask: Optional[np.ndarray] = None,
        selection_mask: Optional[np.ndarray] = None,
    ):
        background = self._prepare_background(frame, background)
        if background is None:
            return frame.copy(), None
        # LAB reduces sensitivity to isolated channel noise compared with raw BGR.
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB)
        diff = cv2.absdiff(frame_lab, background_lab)
        score = cv2.max(diff[..., 0], cv2.max(diff[..., 1], diff[..., 2]))
        _, binary = cv2.threshold(
            score, int(np.clip(threshold_val, 1, 255)), 255, cv2.THRESH_BINARY
        )
        mask = binary.astype(np.float32) / 255.0
        mask = self._apply_manual_mask(mask, add_mask, remove_mask, selection_mask)
        refined = self.refiners["motion"].refine(
            mask, frame, expand=expand, feather=feather, temporal_stability=stability
        )
        return self.composite(frame, background, refined, alpha), refined

    def reset_ai_temporal(self) -> None:
        self.refiners["ai"].reset()

    def reset_cloak_temporal(self) -> None:
        self.refiners["cloak"].reset()

    def reset_motion_temporal(self) -> None:
        self.refiners["motion"].reset()

    def reset_trail(self) -> None:
        self.accumulator = None

    def reset_background_dependent_state(self) -> None:
        self.reset_ai_temporal()
        self.reset_cloak_temporal()
        self.reset_motion_temporal()

    def reset_for_mode_change(self, previous_mode: int, new_mode: int) -> None:
        previous_mode = int(previous_mode)
        new_mode = int(new_mode)
        if previous_mode == new_mode:
            return
        # Clear only temporal state that cannot be meaningfully reused across modes.
        if previous_mode == 0 or new_mode == 0:
            self.reset_ai_temporal()
        if previous_mode == 1 or new_mode == 1:
            self.reset_cloak_temporal()
        if previous_mode == 2 or new_mode == 2:
            self.reset_trail()
        if previous_mode == 3 or new_mode == 3:
            self.reset_motion_temporal()

    def reset(self) -> None:
        self.accumulator = None
        self.clear_freeze()
        for refiner in self.refiners.values():
            refiner.reset()
