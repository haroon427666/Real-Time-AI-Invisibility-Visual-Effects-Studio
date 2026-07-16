"""Persistent application settings, validation, migration, and quality presets."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)
SETTINGS_DIR = Path.home() / ".ghost_invisibility_mode"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
PRESETS_FILE = SETTINGS_DIR / "presets.json"
SETTINGS_SCHEMA_VERSION = 2

QUALITY_PRESETS = {
    "Fast": {"expand": 0, "feather": 7, "stability": 0.5, "process_every": 2},
    "Balanced": {"expand": 1, "feather": 11, "stability": 0.65, "process_every": 1},
    "Quality": {"expand": 2, "feather": 17, "stability": 0.8, "process_every": 1},
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "schema_version": SETTINGS_SCHEMA_VERSION,
    "camera_index": 0,
    "resolution": "640x480",
    "camera_fps": 30,
    "camera_backend": "Auto",
    "camera_buffer": 1,
    "camera_scan_max": 9,
    "exposure": -1.0,
    "focus": -1.0,
    "quality": "Balanced",
    "model_backend": "Auto",
    "preview_mode": "Final",
    "audio_recording": True,
    "virtual_camera": False,
    "show_landmarks": True,
    "last_directory": "",
}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    return default


def validate_settings(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = DEFAULT_SETTINGS.copy()
    result["schema_version"] = SETTINGS_SCHEMA_VERSION
    result["camera_index"] = max(0, min(20, _safe_int(source.get("camera_index"), 0)))
    resolution = source.get("resolution")
    if resolution in {"640x480", "1280x720", "1920x1080"}:
        result["resolution"] = resolution
    result["camera_fps"] = max(10, min(60, _safe_int(source.get("camera_fps"), 30)))
    backend = source.get("camera_backend")
    if backend in {"Auto", "DSHOW", "MSMF"}:
        result["camera_backend"] = backend
    result["camera_buffer"] = max(1, min(5, _safe_int(source.get("camera_buffer"), 1)))
    result["camera_scan_max"] = max(0, min(20, _safe_int(source.get("camera_scan_max"), 9)))
    result["exposure"] = max(-1.0, min(10000.0, _safe_float(source.get("exposure"), -1.0)))
    result["focus"] = max(-1.0, min(10000.0, _safe_float(source.get("focus"), -1.0)))
    quality = source.get("quality")
    if quality in QUALITY_PRESETS:
        result["quality"] = quality
    model = source.get("model_backend")
    if model in {"Auto", "MediaPipe Selfie", "Motion Fallback"}:
        result["model_backend"] = model
    preview = source.get("preview_mode")
    if preview in {"Final", "Raw", "Mask", "Alpha", "Split"}:
        result["preview_mode"] = preview
    result["audio_recording"] = _safe_bool(source.get("audio_recording"), True)
    result["virtual_camera"] = _safe_bool(source.get("virtual_camera"), False)
    result["show_landmarks"] = _safe_bool(source.get("show_landmarks"), True)
    last_directory = source.get("last_directory", "")
    result["last_directory"] = str(last_directory) if last_directory is not None else ""
    return result


def _validate_preset(payload: Dict[str, Any]) -> Dict[str, Any]:
    quality = payload.get("quality")
    model = payload.get("model_backend")
    return {
        "quality": quality if quality in QUALITY_PRESETS else "Balanced",
        "model_backend": model if model in {"Auto", "MediaPipe Selfie", "Motion Fallback"} else "Auto",
        "ai_alpha": max(0.0, min(1.0, _safe_float(payload.get("ai_alpha"), 0.0))),
        "ai_threshold": max(0.01, min(0.99, _safe_float(payload.get("ai_threshold"), 0.15))),
        "cloak_alpha": max(0.0, min(1.0, _safe_float(payload.get("cloak_alpha"), 0.0))),
        "hue_center": _safe_int(payload.get("hue_center"), 60) % 180,
        "hue_tolerance": max(5, min(40, _safe_int(payload.get("hue_tolerance"), 15))),
        "sv_min": max(10, min(160, _safe_int(payload.get("sv_min"), 40))),
        "trail": max(10.0, min(98.0, _safe_float(payload.get("trail"), 80.0))),
        "sub_alpha": max(0.0, min(1.0, _safe_float(payload.get("sub_alpha"), 0.0))),
        "sub_sensitivity": max(5, min(100, _safe_int(payload.get("sub_sensitivity"), 25))),
        "freeze_alpha": max(0.1, min(1.0, _safe_float(payload.get("freeze_alpha"), 0.5))),
        "edge_expand": max(-3, min(5, _safe_int(payload.get("edge_expand"), 1))),
        "edge_feather": max(1, min(31, _safe_int(payload.get("edge_feather"), 11))),
        "stability": max(0.0, min(0.95, _safe_float(payload.get("stability"), 0.65))),
    }


def validate_presets(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for name, payload in value.items():
        if isinstance(name, str) and name.strip() and isinstance(payload, dict):
            output[name.strip()[:80]] = _validate_preset(payload)
    return output


def _backup_corrupt(path: Path) -> None:
    try:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination = path.with_name(f"{path.stem}.corrupt.{stamp}{path.suffix}")
        path.replace(destination)
        LOGGER.warning("Moved invalid settings file to %s", destination)
    except OSError:
        LOGGER.exception("Could not preserve corrupt settings file %s", path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default.copy() if isinstance(default, dict) else default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError):
        LOGGER.exception("Could not read %s", path)
        _backup_corrupt(path)
        return default.copy() if isinstance(default, dict) else default


def _save(path: Path, value: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
        temp.replace(path)
    except OSError:
        LOGGER.exception("Could not write %s", path)


def load_settings() -> Dict[str, Any]:
    return validate_settings(_load_json(SETTINGS_FILE, DEFAULT_SETTINGS))


def save_settings(settings: Dict[str, Any]) -> None:
    _save(SETTINGS_FILE, validate_settings(settings))


def load_presets() -> Dict[str, Dict[str, Any]]:
    return validate_presets(_load_json(PRESETS_FILE, {}))


def save_presets(presets: Dict[str, Dict[str, Any]]) -> None:
    _save(PRESETS_FILE, validate_presets(presets))
