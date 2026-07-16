"""Threaded video/audio recording and paced virtual-camera output."""

from __future__ import annotations

import logging
from pathlib import Path
from queue import Empty, Full, Queue
import shutil
import subprocess
import threading
import time
import wave
from typing import Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyvirtualcam

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    pyvirtualcam = None
    PYVIRTUALCAM_AVAILABLE = False


_STOP = object()


class VideoAudioRecorder:
    """Encode processed frames off the UI thread and finalize atomically."""

    def __init__(self, queue_size: int = 5):
        self.writer: Optional[cv2.VideoWriter] = None
        self.audio_stream = None
        self.wave_file: Optional[wave.Wave_write] = None
        self.final_path: Optional[Path] = None
        self.temp_video: Optional[Path] = None
        self.audio_path: Optional[Path] = None
        self.temp_muxed: Optional[Path] = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self.fps = 30.0
        self.frames_written = 0
        self.frames_duplicated = 0
        self.dropped_frames = 0
        self.audio_enabled = False
        self._audio_lock = threading.Lock()
        self._queue_size = max(1, int(queue_size))
        self._frames: Queue = Queue(maxsize=self._queue_size)
        self._completed: Queue[dict] = Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._recording = False
        self._finalizing = False
        self._session_id = ""

    @property
    def active(self) -> bool:
        return self._recording

    @property
    def finalizing(self) -> bool:
        return self._finalizing

    def start(
        self,
        path: str,
        frame_size: Tuple[int, int],
        fps: float = 30.0,
        with_audio: bool = True,
    ) -> dict:
        if self.active or self.finalizing:
            raise RuntimeError("Recording or finalization is already active")
        self.final_path = Path(path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = uuid4().hex
        stem = f".{self.final_path.stem}.{self._session_id}"
        self.temp_video = self.final_path.with_name(stem + ".video.mp4")
        self.audio_path = self.final_path.with_name(stem + ".audio.wav")
        self.temp_muxed = self.final_path.with_name(stem + ".muxed.mp4")
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise ValueError("Recording frame size must be positive")
        self.fps = max(1.0, min(120.0, float(fps)))
        self.frames_written = 0
        self.frames_duplicated = 0
        self.dropped_frames = 0
        self.audio_enabled = False
        self._frames = Queue(maxsize=self._queue_size)
        self._completed = Queue(maxsize=1)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.temp_video), fourcc, self.fps, self.frame_size
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            self.temp_video.unlink(missing_ok=True)
            raise IOError("OpenCV could not start the MP4 writer")

        status = {"audio": False, "message": "Video recording started"}
        if with_audio and SOUNDDEVICE_AVAILABLE:
            try:
                self.wave_file = wave.open(str(self.audio_path), "wb")
                self.wave_file.setnchannels(1)
                self.wave_file.setsampwidth(2)
                self.wave_file.setframerate(44100)

                def callback(indata, frames, time_info, status_flags):
                    del frames, time_info
                    if status_flags:
                        LOGGER.warning("Audio capture status: %s", status_flags)
                    with self._audio_lock:
                        if self.wave_file is not None:
                            self.wave_file.writeframes(bytes(indata))

                self.audio_stream = sd.RawInputStream(
                    samplerate=44100,
                    channels=1,
                    dtype="int16",
                    callback=callback,
                )
                self.audio_stream.start()
                self.audio_enabled = True
                status = {"audio": True, "message": "Video and microphone recording started"}
            except Exception:
                LOGGER.exception("Microphone recording could not start")
                self._close_audio()
                status = {
                    "audio": False,
                    "message": "Video started; microphone was unavailable",
                }
        elif with_audio:
            status = {
                "audio": False,
                "message": "Video started; install sounddevice for microphone audio",
            }

        self._recording = True
        self._finalizing = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="RecordingWorker",
            daemon=True,
        )
        self._thread.start()
        return status

    def write(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        if not self._recording or frame is None:
            return
        item = (time.monotonic() if timestamp is None else float(timestamp), frame.copy())
        try:
            self._frames.put_nowait(item)
        except Full:
            try:
                self._frames.get_nowait()
                self.dropped_frames += 1
            except Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except Full:
                self.dropped_frames += 1

    def _write_frame(self, frame: np.ndarray) -> None:
        if self.writer is None or self.frame_size is None:
            raise RuntimeError("Video writer is not available")
        output = frame
        if output.ndim != 3 or output.shape[2] != 3:
            raise ValueError("Recorder requires a three-channel BGR frame")
        if output.shape[1::-1] != self.frame_size:
            output = cv2.resize(output, self.frame_size, interpolation=cv2.INTER_LINEAR)
        self.writer.write(output)
        self.frames_written += 1

    def _writer_loop(self) -> None:
        first_timestamp: Optional[float] = None
        next_output_index = 0
        last_frame: Optional[np.ndarray] = None
        error: Optional[str] = None
        try:
            while True:
                item = self._frames.get()
                if item is _STOP:
                    break
                timestamp, frame = item
                if first_timestamp is None:
                    first_timestamp = timestamp
                elapsed = max(0.0, timestamp - first_timestamp)
                expected_index = int(round(elapsed * self.fps))
                if last_frame is not None:
                    maximum_fill = int(self.fps * 2)
                    fill_count = min(max(0, expected_index - next_output_index), maximum_fill)
                    for _ in range(fill_count):
                        self._write_frame(last_frame)
                        self.frames_duplicated += 1
                        next_output_index += 1
                if expected_index < next_output_index - 1:
                    self.dropped_frames += 1
                    continue
                self._write_frame(frame)
                last_frame = frame
                next_output_index += 1
        except Exception as exc:
            LOGGER.exception("Recording worker failed")
            error = str(exc)
        finally:
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    LOGGER.exception("Could not release video writer")
                self.writer = None
            self._close_audio()
            result = self._finalize(error)
            self._recording = False
            self._finalizing = False
            try:
                self._completed.put_nowait(result)
            except Full:
                try:
                    self._completed.get_nowait()
                except Empty:
                    pass
                self._completed.put_nowait(result)

    def _close_audio(self) -> None:
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception:
                LOGGER.exception("Could not close the audio stream")
            self.audio_stream = None
        with self._audio_lock:
            if self.wave_file is not None:
                try:
                    self.wave_file.close()
                except Exception:
                    LOGGER.exception("Could not close the audio file")
                self.wave_file = None

    def _finalize(self, recording_error: Optional[str]) -> dict:
        assert self.final_path is not None and self.temp_video is not None
        final_candidate = self.temp_video
        muxed = False
        finalization_error = recording_error

        if (
            recording_error is None
            and self.audio_enabled
            and self.audio_path is not None
            and self.audio_path.exists()
            and shutil.which("ffmpeg")
            and self.temp_muxed is not None
        ):
            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self.temp_video),
                "-i",
                str(self.audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(self.temp_muxed),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if not self.temp_muxed.exists() or self.temp_muxed.stat().st_size == 0:
                    raise IOError("FFmpeg produced an empty output file")
                final_candidate = self.temp_muxed
                muxed = True
            except subprocess.CalledProcessError as exc:
                finalization_error = f"FFmpeg failed: {exc.stderr.strip()}"
                LOGGER.error(finalization_error)
            except (OSError, subprocess.TimeoutExpired) as exc:
                finalization_error = f"Audio/video finalization failed: {exc}"
                LOGGER.exception("ffmpeg could not combine video and audio")

        saved_path: Optional[str] = None
        try:
            if final_candidate.exists() and final_candidate.stat().st_size > 0:
                final_candidate.replace(self.final_path)
                saved_path = str(self.final_path)
                if muxed:
                    self.temp_video.unlink(missing_ok=True)
                    if self.audio_path is not None:
                        self.audio_path.unlink(missing_ok=True)
            elif finalization_error is None:
                finalization_error = "The recorded video file was empty"
        except OSError as exc:
            LOGGER.exception("Could not atomically publish the recording")
            finalization_error = f"Could not save final recording: {exc}"

        message = (
            f"Recording saved to {saved_path}"
            if saved_path
            else "Recording could not be finalized; temporary files were kept"
        )
        if self.audio_enabled and not muxed and self.audio_path and self.audio_path.exists():
            message += f"; separate audio is available at {self.audio_path}"
        if finalization_error:
            message += f" ({finalization_error})"
        result = {
            "path": saved_path,
            "audio": muxed,
            "frames": self.frames_written,
            "duplicated_frames": self.frames_duplicated,
            "dropped_frames": self.dropped_frames,
            "error": finalization_error,
            "message": message,
        }
        self.audio_enabled = False
        return result

    def stop(self, wait: bool = True, timeout: float = 180.0) -> dict:
        if not self._recording and not self._finalizing:
            completed = self.get_completed()
            return completed or {
                "path": None,
                "audio": False,
                "message": "Recording was not active",
            }
        if self._recording:
            self._recording = False
            self._finalizing = True
            try:
                self._frames.put_nowait(_STOP)
            except Full:
                try:
                    self._frames.get_nowait()
                    self.dropped_frames += 1
                except Empty:
                    pass
                self._frames.put_nowait(_STOP)
        if not wait:
            return {
                "path": None,
                "audio": False,
                "message": "Recording stopped; finalizing in the background",
            }
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(timeout)))
        if self._thread and self._thread.is_alive():
            return {
                "path": None,
                "audio": False,
                "error": "Recording finalization timed out",
                "message": "Recording finalization timed out; temporary files were kept",
            }
        return self.get_completed() or {
            "path": None,
            "audio": False,
            "message": "Recording stopped",
        }

    def get_completed(self) -> Optional[dict]:
        latest = None
        while True:
            try:
                latest = self._completed.get_nowait()
            except Empty:
                return latest


class VirtualCameraOutput:
    """Publish frames from a bounded queue at the virtual camera's own cadence."""

    def __init__(self, queue_size: int = 2):
        self.camera = None
        self.frame_size: Optional[Tuple[int, int]] = None
        self._frames: Queue = Queue(maxsize=max(1, int(queue_size)))
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self.error: Optional[str] = None
        self.dropped_frames = 0

    @property
    def active(self) -> bool:
        return self._active

    def start(self, width: int, height: int, fps: float = 30.0) -> None:
        if not PYVIRTUALCAM_AVAILABLE:
            raise RuntimeError("pyvirtualcam is not installed")
        if not self.stop():
            raise RuntimeError(
                "The previous virtual-camera worker did not stop cleanly"
            )
        self.frame_size = (int(width), int(height))
        self.camera = pyvirtualcam.Camera(
            width=int(width),
            height=int(height),
            fps=max(1, int(fps)),
            fmt=pyvirtualcam.PixelFormat.BGR,
        )
        self._frames = Queue(maxsize=self._frames.maxsize)
        self.error = None
        self.dropped_frames = 0
        self._active = True
        self._thread = threading.Thread(
            target=self._run, name="VirtualCameraWorker", daemon=True
        )
        self._thread.start()

    def send(self, frame: np.ndarray) -> None:
        if not self._active:
            return
        try:
            self._frames.put_nowait(frame.copy())
        except Full:
            try:
                self._frames.get_nowait()
                self.dropped_frames += 1
            except Empty:
                pass
            try:
                self._frames.put_nowait(frame.copy())
            except Full:
                self.dropped_frames += 1

    def _run(self) -> None:
        try:
            while self._active:
                item = self._frames.get()
                if item is _STOP:
                    break
                output = item
                if self.camera is None or self.frame_size is None:
                    break
                if output.shape[1::-1] != self.frame_size:
                    output = cv2.resize(output, self.frame_size, interpolation=cv2.INTER_LINEAR)
                self.camera.send(output)
                sleeper = getattr(self.camera, "sleep_until_next_frame", None)
                if callable(sleeper):
                    sleeper()
        except Exception as exc:
            self.error = str(exc)
            LOGGER.exception("Virtual camera output failed")
        finally:
            self._active = False

    def stop(self, timeout: float = 3.0) -> bool:
        was_active = self._active
        self._active = False
        if was_active:
            try:
                self._frames.put_nowait(_STOP)
            except Full:
                try:
                    self._frames.get_nowait()
                except Empty:
                    pass
                try:
                    self._frames.put_nowait(_STOP)
                except Full:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.1, float(timeout)))
        if self._thread and self._thread.is_alive():
            LOGGER.error("Virtual-camera worker failed to stop; camera remains open")
            return False
        self._thread = None
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                LOGGER.exception("Could not close virtual camera")
            self.camera = None
        return True
