"""Dedicated camera and inference workers with bounded latest-frame queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto
import logging
import os
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Optional, Union

import cv2
import numpy as np

from effects import GhostEffectEngine, HandGestureController, SelfieSegmenter
from utils import camera_motion_score

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(round(_safe_float(value, float(default))))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        return bool(default)
    if value is None:
        return bool(default)
    return bool(value)


def _backend_code(name: str) -> int:
    name = (name or "Auto").lower()
    if name == "dshow" and os.name == "nt":
        return cv2.CAP_DSHOW
    if name == "msmf" and os.name == "nt":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


class WorkerState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    RECONNECTING = auto()
    FAILED = auto()
    STOPPING = auto()


@dataclass(frozen=True)
class ApplicationError:
    source: str
    code: str
    message: str
    recoverable: bool
    timestamp: float = field(default_factory=time.time)


class CameraWorker:
    """Own the camera exclusively and publish only the newest valid frame."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._sequence = 0
        self._last_delivered_sequence = 0
        self._recent: deque[np.ndarray] = deque(maxlen=60)
        self._settings: dict[str, Any] = {}
        self._capture = None
        self._state = WorkerState.STOPPED
        self.capture_fps = 0.0
        self.dropped_frames = 0
        self.read_failures = 0
        self.error: Optional[str] = None
        self.last_error: Optional[ApplicationError] = None

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def running(self) -> bool:
        return self._state is WorkerState.RUNNING

    @staticmethod
    def detect_cameras(
        max_index: int = 10,
        backend: str = "Auto",
        exclude: Optional[set[int]] = None,
    ) -> list[int]:
        valid: list[int] = []
        excluded = exclude or set()
        for index in range(max(0, int(max_index))):
            if index in excluded:
                continue
            cap = None
            try:
                cap = cv2.VideoCapture(index, _backend_code(backend))
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None and frame.size:
                        valid.append(index)
            except Exception:
                LOGGER.debug("Camera probe failed for index %s", index, exc_info=True)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        LOGGER.debug("Camera probe release failed for index %s", index, exc_info=True)
        return valid

    def _clear_frame_state(self) -> None:
        with self._lock:
            self._latest = None
            self._recent.clear()
            self._sequence = 0
            self._last_delivered_sequence = 0
        self.capture_fps = 0.0
        self.dropped_frames = 0
        self.read_failures = 0

    def start(self, startup_timeout: float = 3.0, **settings) -> bool:
        if not self.stop(timeout=3.0):
            self.error = "Previous camera worker did not stop; a second capture loop was not started"
            self.last_error = ApplicationError(
                "camera", "CAMERA_PREVIOUS_WORKER_ALIVE", self.error, False
            )
            return False

        self._clear_frame_state()
        self._settings = settings.copy()
        self._stop.clear()
        self._started.clear()
        self.error = None
        self.last_error = None
        self._state = WorkerState.STARTING
        self._thread = threading.Thread(target=self._run, name="CameraWorker", daemon=True)
        self._thread.start()

        if not self._started.wait(timeout=max(0.1, float(startup_timeout))):
            self.error = "Camera startup timed out"
            self.last_error = ApplicationError(
                "camera", "CAMERA_START_TIMEOUT", self.error, True
            )
            self.stop(timeout=1.0)
            self._state = WorkerState.FAILED
            return False
        return self._state is WorkerState.RUNNING

    def stop(self, timeout: float = 3.0) -> bool:
        if self._thread is None:
            self._state = WorkerState.STOPPED
            return True
        self._state = WorkerState.STOPPING
        self._stop.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                LOGGER.exception("Could not release camera while stopping")
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(timeout)))
            if self._thread.is_alive():
                LOGGER.error("Camera worker did not stop within timeout")
                return False
        self._thread = None
        self._capture = None
        self._state = WorkerState.STOPPED
        return True

    def _set_failure(self, code: str, message: str, recoverable: bool = True) -> None:
        self.error = message
        self.last_error = ApplicationError("camera", code, message, recoverable)
        self._state = WorkerState.FAILED

    def _run(self) -> None:
        index = int(self._settings.get("index", 0))
        cap = None
        try:
            cap = cv2.VideoCapture(
                index, _backend_code(self._settings.get("backend", "Auto"))
            )
            self._capture = cap
            width = int(self._settings.get("width", 640))
            height = int(self._settings.get("height", 480))
            fps = int(self._settings.get("fps", 30))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self._settings.get("buffer", 1)))
            exposure = float(self._settings.get("exposure", -1))
            focus = float(self._settings.get("focus", -1))
            if exposure >= 0:
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            if focus >= 0:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
            if not cap.isOpened():
                self._set_failure("CAMERA_OPEN_FAILED", f"Could not open camera index {index}")
                return

            self._state = WorkerState.RUNNING
            self._started.set()
            window_start = time.perf_counter()
            window_count = 0
            consecutive_failures = 0
            maximum_failures = max(1, int(self._settings.get("max_read_failures", 10)))

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or not getattr(frame, "size", 0):
                    consecutive_failures += 1
                    self.read_failures += 1
                    if consecutive_failures >= maximum_failures:
                        self._set_failure(
                            "CAMERA_READ_FAILED",
                            "Camera stopped returning valid frames",
                            True,
                        )
                        break
                    time.sleep(0.03)
                    continue

                if frame.ndim != 3 or frame.shape[2] != 3:
                    consecutive_failures += 1
                    self.read_failures += 1
                    LOGGER.warning("Camera returned invalid frame shape: %s", frame.shape)
                    continue

                consecutive_failures = 0
                now = time.perf_counter()
                window_count += 1
                if now - window_start >= 1.0:
                    self.capture_fps = window_count / (now - window_start)
                    window_count = 0
                    window_start = now

                with self._lock:
                    if self._sequence > self._last_delivered_sequence:
                        self.dropped_frames += 1
                    self._latest = frame
                    self._sequence += 1
                    self._recent.append(frame.copy())
        except Exception:
            LOGGER.exception("Camera worker failed")
            self._set_failure(
                "CAMERA_UNEXPECTED_FAILURE",
                "Unexpected camera failure; see logs",
                True,
            )
        finally:
            self._started.set()
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    LOGGER.debug("Camera release failed during cleanup", exc_info=True)
            if self._capture is cap:
                self._capture = None
            if self._state not in (WorkerState.FAILED, WorkerState.STOPPING):
                self._state = WorkerState.STOPPED

    def get_latest(self, last_sequence: int = -1):
        with self._lock:
            if self._latest is None or self._sequence == last_sequence:
                return self._sequence, None
            self._last_delivered_sequence = self._sequence
            return self._sequence, self._latest.copy()

    def get_recent_frames(self, count: int = 15) -> list[np.ndarray]:
        with self._lock:
            return [frame.copy() for frame in list(self._recent)[-max(0, count):]]


@dataclass(frozen=True)
class ProcessingConfig:
    mode: int = 0
    model_backend: str = "Auto"
    gestures: bool = False
    show_landmarks: bool = False
    ai_alpha: float = 0.0
    ai_threshold: float = 0.15
    cloak_alpha: float = 0.0
    hsv_lower: np.ndarray = field(
        default_factory=lambda: np.array([35, 40, 40], dtype=np.int32)
    )
    hsv_upper: np.ndarray = field(
        default_factory=lambda: np.array([85, 255, 255], dtype=np.int32)
    )
    trail_decay: float = 0.2
    sub_alpha: float = 0.0
    sub_sensitivity: int = 25
    freeze_alpha: float = 0.5
    expand: int = 1
    feather: int = 11
    stability: float = 0.65
    maintain_person_mask: bool = False
    target_selection_active: bool = False

    @classmethod
    def from_value(cls, value: Union["ProcessingConfig", dict[str, Any], None]) -> "ProcessingConfig":
        if isinstance(value, cls):
            return value.validated()
        if not isinstance(value, dict):
            value = {}
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        kwargs = {key: val for key, val in value.items() if key in allowed}
        return cls(**kwargs).validated()

    @staticmethod
    def _validated_hsv(value: Any, default: list[int]) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float64).reshape(-1)[:3]
        except (TypeError, ValueError, OverflowError):
            array = np.asarray(default, dtype=np.float64)
        if array.size != 3:
            array = np.asarray(default, dtype=np.float64)
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.rint(array).astype(np.int32, copy=False)
        array = array.copy()
        array[0] %= 180
        array[1:] = np.clip(array[1:], 0, 255)
        return array

    def validated(self) -> "ProcessingConfig":
        lower = self._validated_hsv(self.hsv_lower, [35, 40, 40])
        upper = self._validated_hsv(self.hsv_upper, [85, 255, 255])
        feather = int(np.clip(_safe_int(self.feather, 11), 1, 31))
        if feather % 2 == 0:
            feather = min(feather + 1, 31)
        return replace(
            self,
            mode=int(np.clip(_safe_int(self.mode, 0), 0, 4)),
            model_backend=str(self.model_backend or "Auto"),
            ai_alpha=float(np.clip(_safe_float(self.ai_alpha, 0.0), 0.0, 1.0)),
            ai_threshold=float(
                np.clip(_safe_float(self.ai_threshold, 0.15), 0.01, 0.99)
            ),
            cloak_alpha=float(
                np.clip(_safe_float(self.cloak_alpha, 0.0), 0.0, 1.0)
            ),
            hsv_lower=lower,
            hsv_upper=upper,
            trail_decay=float(
                np.clip(_safe_float(self.trail_decay, 0.2), 0.01, 1.0)
            ),
            sub_alpha=float(
                np.clip(_safe_float(self.sub_alpha, 0.0), 0.0, 1.0)
            ),
            sub_sensitivity=int(
                np.clip(_safe_int(self.sub_sensitivity, 25), 1, 255)
            ),
            freeze_alpha=float(
                np.clip(_safe_float(self.freeze_alpha, 0.5), 0.0, 1.0)
            ),
            expand=int(np.clip(_safe_int(self.expand, 1), -3, 5)),
            feather=feather,
            stability=float(
                np.clip(_safe_float(self.stability, 0.65), 0.0, 0.98)
            ),
            gestures=_safe_bool(self.gestures),
            show_landmarks=_safe_bool(self.show_landmarks),
            maintain_person_mask=_safe_bool(self.maintain_person_mask),
            target_selection_active=_safe_bool(self.target_selection_active),
        )


@dataclass
class ProcessingJob:
    sequence: int
    frame: np.ndarray
    background: Optional[np.ndarray]
    config: Union[ProcessingConfig, dict[str, Any]]
    add_mask: Optional[np.ndarray]
    remove_mask: Optional[np.ndarray]
    selection_mask: Optional[np.ndarray]
    generation: int = 0
    background_version: int = 0

    def __post_init__(self) -> None:
        self.config = ProcessingConfig.from_value(self.config)


@dataclass
class ProcessingResult:
    sequence: int
    raw: np.ndarray
    final: np.ndarray
    mask: Optional[np.ndarray]
    alpha_preview: np.ndarray
    gesture: Optional[str]
    pinch_value: float
    handedness: Optional[str]
    landmarks: Any
    processing_ms: float
    motion_score: float
    target_area: float
    error: Optional[str] = None
    generation: int = 0
    background_version: int = 0


def result_matches_state(
    result: "ProcessingResult", generation: int, background_version: int
) -> bool:
    return (
        result.generation == int(generation)
        and result.background_version == int(background_version)
    )


_STOP_JOB = object()


class ProcessingWorker:
    """Run effects on the newest submitted frame and reject queue buildup."""

    def __init__(self):
        self.engine = GhostEffectEngine()
        self._engine_lock = threading.RLock()
        self.segmenter = SelfieSegmenter(model_selection=0)
        self.hand_controller = HandGestureController()
        self._jobs: Queue[Any] = Queue(maxsize=1)
        self._results: Queue[ProcessingResult] = Queue(maxsize=1)
        self._stop = threading.Event()
        self._state = WorkerState.STARTING
        self._thread = threading.Thread(target=self._run, name="ProcessingWorker", daemon=True)
        self._thread.start()
        self.processed_fps = 0.0
        self.dropped_jobs = 0
        self.replaced_results = 0
        self.last_error: Optional[ApplicationError] = None

    @property
    def state(self) -> WorkerState:
        return self._state

    def submit(self, job: ProcessingJob) -> None:
        if self._stop.is_set():
            return
        try:
            self._jobs.put_nowait(job)
        except Full:
            try:
                self._jobs.get_nowait()
                self.dropped_jobs += 1
            except Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except Full:
                self.dropped_jobs += 1

    def clear_queues(self) -> None:
        for queue in (self._jobs, self._results):
            while True:
                try:
                    queue.get_nowait()
                except Empty:
                    break

    def get_result(self) -> Optional[ProcessingResult]:
        latest = None
        while True:
            try:
                latest = self._results.get_nowait()
            except Empty:
                return latest

    def capture_freeze(self, frame: np.ndarray, mask: Optional[np.ndarray]) -> None:
        with self._engine_lock:
            self.engine.capture_freeze(frame, mask)

    def clear_freeze(self) -> None:
        with self._engine_lock:
            self.engine.clear_freeze()

    @property
    def has_freeze(self) -> bool:
        with self._engine_lock:
            return self.engine.frozen_person is not None

    def reset(self) -> None:
        with self._engine_lock:
            self.engine.reset()
        self.clear_queues()

    def reset_for_mode_change(self, previous_mode: int, new_mode: int) -> None:
        with self._engine_lock:
            self.engine.reset_for_mode_change(previous_mode, new_mode)
        self.clear_queues()

    def reset_background_dependent_state(self) -> None:
        with self._engine_lock:
            self.engine.reset_background_dependent_state()
        self.clear_queues()

    def _put_result(self, result: ProcessingResult) -> None:
        try:
            self._results.put_nowait(result)
        except Full:
            try:
                self._results.get_nowait()
                self.replaced_results += 1
            except Empty:
                pass
            try:
                self._results.put_nowait(result)
            except Full:
                self.replaced_results += 1

    def _run(self) -> None:
        self._state = WorkerState.RUNNING
        count = 0
        start_window = time.perf_counter()
        while not self._stop.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except Empty:
                continue
            if job is _STOP_JOB:
                break
            started = time.perf_counter()
            try:
                with self._engine_lock:
                    result = self._process(job)
            except Exception as exc:
                LOGGER.exception("Frame processing failed")
                self.last_error = ApplicationError(
                    "processing", "FRAME_PROCESSING_FAILED", str(exc), True
                )
                blank = job.frame.copy()
                result = ProcessingResult(
                    sequence=job.sequence,
                    raw=job.frame.copy(),
                    final=blank,
                    mask=None,
                    alpha_preview=np.zeros_like(blank),
                    gesture=None,
                    pinch_value=1.0,
                    handedness=None,
                    landmarks=None,
                    processing_ms=(time.perf_counter() - started) * 1000,
                    motion_score=0.0,
                    target_area=0.0,
                    error=str(exc),
                    generation=job.generation,
                    background_version=job.background_version,
                )
            self._put_result(result)
            count += 1
            now = time.perf_counter()
            if now - start_window >= 1.0:
                self.processed_fps = count / (now - start_window)
                count = 0
                start_window = now
        self._state = WorkerState.STOPPED

    def _process(self, job: ProcessingJob) -> ProcessingResult:
        started = time.perf_counter()
        config = ProcessingConfig.from_value(job.config)
        frame = job.frame
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Processing received an invalid BGR frame")
        inference_frame = frame
        display_frame = frame.copy()

        landmarks = gesture = handedness = None
        pinch_value = 1.0
        if config.gestures and self.hand_controller.available:
            landmarks, gesture, pinch_value, handedness = self.hand_controller.process_frame(inference_frame)

        mode = config.mode
        needs_person_mask = (
            mode in (0, 4)
            or config.gestures
            or config.maintain_person_mask
            or config.target_selection_active
        )
        person_mask = None
        if (
            needs_person_mask
            and config.model_backend != "Motion Fallback"
            and self.segmenter.available
        ):
            person_mask = self.segmenter.get_mask(inference_frame)
        if needs_person_mask and person_mask is None and job.background is not None:
            _, person_mask = self.engine.apply_background_subtraction_invisibility(
                inference_frame,
                job.background,
                threshold_val=config.sub_sensitivity,
                alpha=1.0,
                expand=config.expand,
                feather=config.feather,
                stability=config.stability,
                add_mask=job.add_mask,
                remove_mask=job.remove_mask,
                selection_mask=job.selection_mask,
            )

        mask = None
        if mode == 0:
            if job.background is not None:
                display_frame, mask = self.engine.apply_ai_invisibility(
                    inference_frame,
                    job.background,
                    self.segmenter,
                    threshold=config.ai_threshold,
                    alpha=config.ai_alpha,
                    precomputed_mask=person_mask,
                    expand=config.expand,
                    feather=config.feather,
                    stability=config.stability,
                    add_mask=job.add_mask,
                    remove_mask=job.remove_mask,
                    selection_mask=job.selection_mask,
                )
        elif mode == 1:
            if job.background is not None:
                display_frame, mask = self.engine.apply_color_cloak(
                    inference_frame,
                    job.background,
                    config.hsv_lower,
                    config.hsv_upper,
                    alpha=config.cloak_alpha,
                    expand=config.expand,
                    feather=config.feather,
                    stability=config.stability,
                    add_mask=job.add_mask,
                    remove_mask=job.remove_mask,
                    selection_mask=job.selection_mask,
                )
        elif mode == 2:
            display_frame = self.engine.apply_ghost_trail(
                inference_frame, decay_rate=config.trail_decay
            )
        elif mode == 3:
            if job.background is not None:
                display_frame, mask = self.engine.apply_background_subtraction_invisibility(
                    inference_frame,
                    job.background,
                    threshold_val=config.sub_sensitivity,
                    alpha=config.sub_alpha,
                    expand=config.expand,
                    feather=config.feather,
                    stability=config.stability,
                    add_mask=job.add_mask,
                    remove_mask=job.remove_mask,
                    selection_mask=job.selection_mask,
                )
        elif mode == 4:
            display_frame = self.engine.apply_time_freeze(
                inference_frame, alpha=config.freeze_alpha
            )
            mask = self.engine.frozen_mask

        latest_mask = mask if mask is not None else person_mask
        if latest_mask is not None:
            latest_mask = np.nan_to_num(
                latest_mask.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
            )
            latest_mask = np.clip(latest_mask, 0.0, 1.0)
            alpha_preview = (latest_mask * 255).astype(np.uint8)
            alpha_preview = cv2.cvtColor(alpha_preview, cv2.COLOR_GRAY2BGR)
            target_area = float(np.mean(latest_mask > 0.35))
        else:
            alpha_preview = np.zeros_like(frame)
            target_area = 0.0

        if config.show_landmarks and landmarks is not None:
            display_frame = self.hand_controller.draw_landmarks(display_frame, landmarks)

        motion = 0.0
        if job.background is not None and job.sequence % 15 == 0:
            motion = camera_motion_score(job.background, inference_frame)

        return ProcessingResult(
            sequence=job.sequence,
            raw=frame.copy(),
            final=display_frame,
            mask=latest_mask.copy() if latest_mask is not None else None,
            alpha_preview=alpha_preview,
            gesture=gesture,
            pinch_value=pinch_value,
            handedness=handedness,
            landmarks=landmarks,
            processing_ms=(time.perf_counter() - started) * 1000.0,
            motion_score=motion,
            target_area=target_area,
            generation=job.generation,
            background_version=job.background_version,
        )

    def stop(self, timeout: float = 5.0) -> bool:
        if self._state is WorkerState.STOPPED:
            return True
        self._state = WorkerState.STOPPING
        self._stop.set()
        try:
            self._jobs.put_nowait(_STOP_JOB)
        except Full:
            try:
                self._jobs.get_nowait()
            except Empty:
                pass
            try:
                self._jobs.put_nowait(_STOP_JOB)
            except Full:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(timeout)))
        if self._thread.is_alive():
            LOGGER.error("Processing worker failed to stop; model resources remain open")
            return False
        self.segmenter.close()
        self.hand_controller.close()
        self._state = WorkerState.STOPPED
        self.clear_queues()
        return True
