"""Advanced UI and orchestration for Ghost / Invisibility Mode."""

from __future__ import annotations

from collections import deque
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from effects import GhostEffectEngine, MEDIAPIPE_AVAILABLE
from recording import (
    PYVIRTUALCAM_AVAILABLE,
    SOUNDDEVICE_AVAILABLE,
    VideoAudioRecorder,
    VirtualCameraOutput,
)
from settings import (
    DEFAULT_SETTINGS,
    QUALITY_PRESETS,
    load_presets,
    load_settings,
    save_presets,
    save_settings,
)
from utils import (
    background_quality,
    detect_runtime,
    map_canvas_to_frame,
    map_split_preview_to_frame,
    median_background,
    safe_imwrite,
    write_diagnostics,
)
from workers import (
    CameraWorker,
    ProcessingConfig,
    ProcessingJob,
    ProcessingResult,
    ProcessingWorker,
    WorkerState,
    result_matches_state,
)

APP_NAME = "Ghost / Invisibility Mode"
LOG_DIR = Path.home() / ".ghost_invisibility_mode" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)
if not LOGGER.handlers:
    handler = RotatingFileHandler(
        LOG_DIR / "ghost_invisibility.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    LOGGER.addHandler(handler)


class GhostInvisibilityApp:
    """Tk desktop UI backed by dedicated camera and processing workers."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_NAME} - Advanced Reliability Edition")
        self.root.geometry("1600x900")
        self.root.minsize(1280, 720)
        self.root.configure(bg="#121214")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.root.report_callback_exception = self._report_tk_exception

        self.settings = load_settings()
        self.presets = load_presets()
        self.runtime = detect_runtime()

        self.camera_worker = CameraWorker()
        self.processor = ProcessingWorker()
        self.recorder = VideoAudioRecorder()
        self.virtual_camera = VirtualCameraOutput()

        self.is_running = False
        self.closing = False
        self.camera_sequence = -1
        self.submitted_sequence = -1
        self.last_result: Optional[ProcessingResult] = None
        self.raw_frame: Optional[np.ndarray] = None
        self.processed_frame: Optional[np.ndarray] = None
        self.background_frame: Optional[np.ndarray] = None
        self.latest_person_mask: Optional[np.ndarray] = None
        self.background_quality_info: Dict[str, Any] = {
            "score": 0.0,
            "label": "Empty",
        }
        self.pipeline_generation = 0
        self.background_version = 0
        self.previous_effect_mode = 0
        self.stale_results_discarded = 0
        self.reconnect_attempts = 0
        self.maximum_automatic_reconnects = 6

        self.hue_center = 60
        self.sampled_s = 255
        self.sampled_v = 255
        self.hsv_lower = np.array([35, 40, 40], dtype=np.int32)
        self.hsv_upper = np.array([85, 255, 255], dtype=np.int32)

        self.manual_add_mask: Optional[np.ndarray] = None
        self.manual_remove_mask: Optional[np.ndarray] = None
        self.selection_mask: Optional[np.ndarray] = None
        self.selection_history: list[tuple] = []
        self.selection_redo: list[tuple] = []
        self._drawing = False

        self.render_offset_x = 0
        self.render_offset_y = 0
        self.render_width = 0
        self.render_height = 0
        self.render_frame_width = 0
        self.render_frame_height = 0

        self.update_after_id = None
        self.countdown_after_id = None
        self.camera_scan_after_id = None
        self.last_camera_retry = 0.0
        self.next_camera_retry_delay = 1.0
        self.camera_valid_since: Optional[float] = None
        self.background_countdown_active = False
        self.last_gesture_action = 0.0
        self.last_switch_tab_time = 0.0
        self.sampling_feedback_time = 0.0
        self.sampling_feedback_msg = ""
        self.metrics_history = deque(maxlen=60)
        self._camera_scan_queue: Queue[tuple[int, list[int]]] = Queue(maxsize=1)
        self.camera_scan_generation = 0
        self.camera_scan_active = False

        self._create_variables()
        self.setup_styles()
        self.create_widgets()
        self._register_pipeline_traces()
        self.bind_shortcuts()
        self.detect_cameras_async()
        self.start_webcam()
        self.update_loop()

    def _report_tk_exception(self, exc_type, exc_value, traceback_obj) -> None:
        LOGGER.error(
            "Unhandled Tkinter callback error",
            exc_info=(exc_type, exc_value, traceback_obj),
        )
        try:
            messagebox.showerror(
                "Unexpected error",
                "An unexpected interface error occurred. Details were saved to the log.",
            )
        except tk.TclError:
            pass

    def invalidate_pipeline(self, reset_results: bool = True) -> None:
        self.pipeline_generation += 1
        self.submitted_sequence = -1
        if reset_results:
            self.processor.clear_queues()
        self.latest_person_mask = None

    # ------------------------------------------------------------------
    # Settings and UI construction
    # ------------------------------------------------------------------
    def _create_variables(self) -> None:
        self.camera_idx_var = tk.StringVar(value=str(self.settings.get("camera_index", 0)))
        self.resolution_var = tk.StringVar(value=self.settings.get("resolution", "640x480"))
        self.camera_fps_var = tk.IntVar(value=int(self.settings.get("camera_fps", 30)))
        self.camera_backend_var = tk.StringVar(value=self.settings.get("camera_backend", "Auto"))
        self.camera_buffer_var = tk.IntVar(value=int(self.settings.get("camera_buffer", 1)))
        self.camera_scan_max_var = tk.IntVar(value=int(self.settings.get("camera_scan_max", 9)))
        self.exposure_var = tk.DoubleVar(value=float(self.settings.get("exposure", -1.0)))
        self.focus_var = tk.DoubleVar(value=float(self.settings.get("focus", -1.0)))

        self.enable_gestures_var = tk.BooleanVar(value=False)
        self.show_landmarks_var = tk.BooleanVar(value=bool(self.settings.get("show_landmarks", True)))
        self.preview_mode_var = tk.StringVar(value=self.settings.get("preview_mode", "Final"))
        self.quality_var = tk.StringVar(value=self.settings.get("quality", "Balanced"))
        self.model_backend_var = tk.StringVar(value=self.settings.get("model_backend", "Auto"))
        self.interaction_mode_var = tk.StringVar(value="Color Sample")
        self.brush_size_var = tk.IntVar(value=24)

        preset = QUALITY_PRESETS.get(self.quality_var.get(), QUALITY_PRESETS["Balanced"])
        self.edge_expand_var = tk.IntVar(value=int(preset["expand"]))
        self.edge_feather_var = tk.IntVar(value=int(preset["feather"]))
        self.temporal_stability_var = tk.DoubleVar(value=float(preset["stability"]))

        self.ai_alpha_val = tk.DoubleVar(value=0.0)
        self.ai_thresh_val = tk.DoubleVar(value=0.15)
        self.cloak_alpha_val = tk.DoubleVar(value=0.0)
        self.cloak_h_tol = tk.IntVar(value=15)
        self.cloak_sv_min = tk.IntVar(value=40)
        self.trail_val = tk.DoubleVar(value=80.0)
        self.sub_alpha_val = tk.DoubleVar(value=0.0)
        self.sub_sens_val = tk.IntVar(value=25)
        self.freeze_alpha_val = tk.DoubleVar(value=0.5)

        self.audio_recording_var = tk.BooleanVar(value=bool(self.settings.get("audio_recording", True)))
        self.virtual_camera_var = tk.BooleanVar(value=False)
        self.preset_name_var = tk.StringVar(value="")

    def _register_pipeline_traces(self) -> None:
        variables = [
            self.model_backend_var,
            self.enable_gestures_var,
            self.show_landmarks_var,
            self.ai_alpha_val,
            self.ai_thresh_val,
            self.cloak_alpha_val,
            self.cloak_h_tol,
            self.cloak_sv_min,
            self.trail_val,
            self.sub_alpha_val,
            self.sub_sens_val,
            self.freeze_alpha_val,
            self.edge_expand_var,
            self.edge_feather_var,
            self.temporal_stability_var,
        ]
        for variable in variables:
            variable.trace_add("write", self._on_processing_setting_changed)

    def _on_processing_setting_changed(self, *args) -> None:
        del args
        if hasattr(self, "processor"):
            self.invalidate_pipeline()

    def setup_styles(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#121214", foreground="#eeeeee", fieldbackground="#2e2e38")
        self.style.configure("Sidebar.TFrame", background="#1e1e24")
        self.style.configure("Card.TFrame", background="#2e2e38", borderwidth=1, relief="flat")
        self.style.configure("Notebook.TFrame", background="#1e1e24")
        self.style.configure("TLabel", background="#1e1e24", foreground="#eeeeee", font=("Segoe UI", 10))
        self.style.configure("Card.TLabel", background="#2e2e38", foreground="#eeeeee", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", background="#1e1e24", foreground="#00adb5", font=("Segoe UI", 16, "bold"))
        self.style.configure("Subheader.TLabel", background="#2e2e38", foreground="#00adb5", font=("Segoe UI", 11, "bold"))
        self.style.configure("Status.TLabel", background="#121214", foreground="#aaaaaa", font=("Segoe UI", 9))
        self.style.configure("Warning.TLabel", background="#2e2e38", foreground="#ff2e63", font=("Segoe UI", 10, "bold"))
        self.style.configure("TButton", background="#393e46", foreground="#eeeeee", borderwidth=0, font=("Segoe UI", 9, "bold"), padding=7)
        self.style.map("TButton", background=[("active", "#00adb5"), ("pressed", "#008c92")], foreground=[("active", "#121214")])
        self.style.configure("Accent.TButton", background="#00adb5", foreground="#121214", borderwidth=0, font=("Segoe UI", 9, "bold"), padding=7)
        self.style.map("Accent.TButton", background=[("active", "#00d1dc"), ("pressed", "#00adb5")])
        self.style.configure("Danger.TButton", background="#ff2e63", foreground="#eeeeee", borderwidth=0, font=("Segoe UI", 9, "bold"), padding=7)
        self.style.map("Danger.TButton", background=[("active", "#ff5c85"), ("pressed", "#d61f4e")])
        self.style.configure("TNotebook", background="#1e1e24", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#2e2e38", foreground="#aaaaaa", font=("Segoe UI", 9, "bold"), padding=(9, 6))
        self.style.map("TNotebook.Tab", background=[("selected", "#1e1e24"), ("active", "#393e46")], foreground=[("selected", "#00adb5"), ("active", "#eeeeee")])
        self.style.configure("Horizontal.TScale", background="#2e2e38", troughcolor="#1e1e24")

    def create_widgets(self) -> None:
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", padding=12, width=540)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="GHOST / INVISIBILITY ENGINE", style="Header.TLabel").grid(row=0, column=0, pady=(0, 10), sticky="w")

        self.control_notebook = ttk.Notebook(sidebar)
        self.control_notebook.grid(row=1, column=0, sticky="nsew")
        self.effects_page = ttk.Frame(self.control_notebook, style="Notebook.TFrame", padding=8)
        self.camera_page = ttk.Frame(self.control_notebook, style="Notebook.TFrame", padding=8)
        self.advanced_page = ttk.Frame(self.control_notebook, style="Notebook.TFrame", padding=8)
        self.control_notebook.add(self.effects_page, text="Effects")
        self.control_notebook.add(self.camera_page, text="Camera")
        self.control_notebook.add(self.advanced_page, text="Advanced")

        self._create_effects_controls()
        self._create_camera_controls()
        self._create_advanced_controls()

        action_card = ttk.Frame(sidebar, style="Card.TFrame", padding=8)
        action_card.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for column in range(3):
            action_card.columnconfigure(column, weight=1)
        ttk.Button(action_card, text="Screenshot", style="Accent.TButton", command=self.save_screenshot).grid(row=0, column=0, padx=2, sticky="ew")
        ttk.Button(action_card, text="Transparent PNG", command=self.export_transparent_target).grid(row=0, column=1, padx=2, sticky="ew")
        self.record_btn = ttk.Button(action_card, text="Start Recording", command=self.toggle_recording)
        self.record_btn.grid(row=0, column=2, padx=2, sticky="ew")
        ttk.Button(action_card, text="Reset", command=self.reset_settings).grid(row=1, column=0, padx=2, pady=(5, 0), sticky="ew")
        ttk.Button(action_card, text="Diagnostics", command=self.export_diagnostics).grid(row=1, column=1, padx=2, pady=(5, 0), sticky="ew")
        ttk.Button(action_card, text="Model Manager", command=self.open_model_manager).grid(row=1, column=2, padx=2, pady=(5, 0), sticky="ew")

        feed_panel = tk.Frame(self.root, bg="#121214", bd=0)
        feed_panel.grid(row=0, column=1, sticky="nsew", padx=12, pady=12)
        feed_panel.columnconfigure(0, weight=1)
        feed_panel.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(feed_panel, style="Card.TFrame", padding=6)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="Preview:", style="Card.TLabel").pack(side="left")
        ttk.Combobox(toolbar, textvariable=self.preview_mode_var, values=["Final", "Raw", "Mask", "Alpha", "Split"], state="readonly", width=9).pack(side="left", padx=(4, 12))
        ttk.Label(toolbar, text="Interaction:", style="Card.TLabel").pack(side="left")
        self.interaction_combo = ttk.Combobox(toolbar, textvariable=self.interaction_mode_var, values=["Color Sample", "Select Target", "Brush Add", "Brush Remove"], state="readonly", width=14)
        self.interaction_combo.pack(side="left", padx=(4, 12))
        ttk.Button(toolbar, text="Undo", command=self.undo_selection).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo_selection).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Clear Selection", command=self.clear_manual_selection).pack(side="left", padx=2)
        self.runtime_lbl = ttk.Label(toolbar, text=self._runtime_text(), style="Card.TLabel")
        self.runtime_lbl.pack(side="right")

        self.canvas = tk.Canvas(feed_panel, bg="#1a1a1e", bd=0, highlightthickness=1, highlightbackground="#333333", cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        metrics = ttk.Frame(feed_panel, style="Card.TFrame", padding=6)
        metrics.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.metrics_lbl = ttk.Label(metrics, text="Capture: 0 FPS | Processing: 0 FPS | Latency: 0 ms | Dropped: 0", style="Card.TLabel")
        self.metrics_lbl.pack(side="left")
        self.target_status_lbl = ttk.Label(metrics, text="Target: waiting", style="Card.TLabel")
        self.target_status_lbl.pack(side="right")

        self.status_bar = ttk.Label(self.root, text="Engine Status: Initializing", style="Status.TLabel", padding=5)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _create_effects_controls(self) -> None:
        self.effects_page.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self.effects_page)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        tabs = []
        for title in ["AI Invisibility", "Color Cloak", "Ghost Trail", "Motion Mask", "Time Freeze"]:
            tab = ttk.Frame(self.notebook, style="Notebook.TFrame", padding=10)
            tab.columnconfigure(0, weight=1)
            self.notebook.add(tab, text=title)
            tabs.append(tab)
        self.tab_ai, self.tab_cloak, self.tab_trail, self.tab_sub, self.tab_freeze = tabs
        self.previous_effect_mode = self.notebook.index("current")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_effect_tab_changed)

        if MEDIAPIPE_AVAILABLE:
            ttk.Label(self.tab_ai, text="MediaPipe person segmentation with fallback").grid(row=0, column=0, sticky="w", pady=(0, 8))
        else:
            ttk.Label(self.tab_ai, text="MediaPipe is not installed; Motion Fallback is available.", style="Warning.TLabel", wraplength=330).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._add_scale(self.tab_ai, "Ghost Transparency", self.ai_alpha_val, 0.0, 1.0, 1)
        self._add_scale(self.tab_ai, "Confidence Threshold", self.ai_thresh_val, 0.01, 0.9, 3)

        ttk.Label(self.tab_cloak, text="Click the preview to sample a cloak color, or use target brushes.", wraplength=330).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.color_box_frame = tk.Frame(self.tab_cloak, height=26, bg="#00ff00", bd=1, relief="sunken")
        self.color_box_frame.grid(row=1, column=0, sticky="ew", pady=4)
        self.color_lbl = ttk.Label(self.tab_cloak, text="Target HSV: [35, 40, 40] to [85, 255, 255]", wraplength=330)
        self.color_lbl.grid(row=2, column=0, sticky="w")
        self._add_scale(self.tab_cloak, "Cloak Transparency", self.cloak_alpha_val, 0.0, 1.0, 3)
        self._add_scale(self.tab_cloak, "Hue Tolerance", self.cloak_h_tol, 5, 40, 5, self.recalculate_hsv_ranges)
        self._add_scale(self.tab_cloak, "Sat/Val Minimum", self.cloak_sv_min, 10, 160, 7, self.recalculate_hsv_ranges)

        ttk.Label(self.tab_trail, text="Temporal accumulation for motion trails.").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._add_scale(self.tab_trail, "Ghost Trail Length", self.trail_val, 10, 98, 1)

        ttk.Label(self.tab_sub, text="LAB background difference with temporal refinement.", wraplength=330).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._add_scale(self.tab_sub, "Ghost Transparency", self.sub_alpha_val, 0.0, 1.0, 1)
        self._add_scale(self.tab_sub, "Motion Sensitivity", self.sub_sens_val, 5, 100, 3)

        ttk.Label(self.tab_freeze, text="Freeze the latest real person/target matte. No artificial fallback mask is used.", wraplength=330).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(self.tab_freeze, text="Freeze Current Ghost", style="Accent.TButton", command=self.capture_freeze).grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Button(self.tab_freeze, text="Clear Frozen Ghost", command=self.clear_freeze).grid(row=2, column=0, sticky="ew", pady=3)
        self._add_scale(self.tab_freeze, "Frozen Clone Transparency", self.freeze_alpha_val, 0.1, 1.0, 3)
        self.freeze_status_lbl = ttk.Label(self.tab_freeze, text="Clone Status: Not Frozen", foreground="#888888")
        self.freeze_status_lbl.grid(row=5, column=0, sticky="w", pady=4)

    def _create_camera_controls(self) -> None:
        page = self.camera_page
        page.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(page, text="Camera Setup", style="Header.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(page, text="Source:").grid(row=row, column=0, sticky="w", pady=3)
        self.cam_combo = ttk.Combobox(page, textvariable=self.camera_idx_var, values=["0"], state="readonly", width=12)
        self.cam_combo.grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Resolution:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(page, textvariable=self.resolution_var, values=["640x480", "1280x720", "1920x1080"], state="readonly").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Target FPS:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(page, from_=10, to=60, textvariable=self.camera_fps_var).grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Backend:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(page, textvariable=self.camera_backend_var, values=["Auto", "DSHOW", "MSMF"], state="readonly").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Buffer size:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(page, from_=1, to=5, textvariable=self.camera_buffer_var).grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Scan through index:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Spinbox(page, from_=0, to=20, textvariable=self.camera_scan_max_var).grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Exposure (-1 auto):").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(page, textvariable=self.exposure_var).grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Focus (-1 auto):").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(page, textvariable=self.focus_var).grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Button(page, text="Apply / Restart Camera", style="Accent.TButton", command=self.start_webcam).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 4)); row += 1
        self.camera_scan_btn = ttk.Button(page, text="Rescan Cameras", command=self.detect_cameras_async)
        self.camera_scan_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=4); row += 1
        self.cap_bg_btn = ttk.Button(page, text="Capture Background (3s)", style="Accent.TButton", command=self.capture_background)
        self.cap_bg_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 4)); row += 1
        self.bg_status_lbl = ttk.Label(page, text="Background: Empty", foreground="#ff2e63", wraplength=330)
        self.bg_status_lbl.grid(row=row, column=0, columnspan=2, sticky="w", pady=3); row += 1
        ttk.Checkbutton(page, text="Enable Hand Gestures", variable=self.enable_gestures_var, command=self.on_gesture_toggle).grid(row=row, column=0, columnspan=2, sticky="w", pady=5); row += 1
        ttk.Checkbutton(page, text="Draw hand landmarks on final preview", variable=self.show_landmarks_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=5)

    def _create_advanced_controls(self) -> None:
        page = self.advanced_page
        page.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(page, text="Processing", style="Header.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(page, text="Model backend:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(page, textvariable=self.model_backend_var, values=["Auto", "MediaPipe Selfie", "Motion Fallback"], state="readonly").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Quality preset:").grid(row=row, column=0, sticky="w", pady=3)
        quality_combo = ttk.Combobox(page, textvariable=self.quality_var, values=list(QUALITY_PRESETS), state="readonly")
        quality_combo.grid(row=row, column=1, sticky="ew", pady=3)
        quality_combo.bind("<<ComboboxSelected>>", self.apply_quality_preset); row += 1
        ttk.Label(page, text="Edge expand/shrink:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(page, from_=-3, to=5, variable=self.edge_expand_var, orient="horizontal").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Edge feather:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(page, from_=1, to=31, variable=self.edge_feather_var, orient="horizontal").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Temporal stability:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(page, from_=0.0, to=0.95, variable=self.temporal_stability_var, orient="horizontal").grid(row=row, column=1, sticky="ew", pady=3); row += 1
        ttk.Label(page, text="Brush size:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Scale(page, from_=3, to=80, variable=self.brush_size_var, orient="horizontal").grid(row=row, column=1, sticky="ew", pady=3); row += 1

        ttk.Separator(page).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(page, text="Output", style="Header.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        ttk.Checkbutton(page, text=f"Record microphone audio ({'ready' if SOUNDDEVICE_AVAILABLE else 'optional dependency missing'})", variable=self.audio_recording_var).grid(row=row, column=0, columnspan=2, sticky="w", pady=3); row += 1
        ttk.Checkbutton(page, text=f"Virtual camera ({'ready' if PYVIRTUALCAM_AVAILABLE else 'optional dependency missing'})", variable=self.virtual_camera_var, command=self.toggle_virtual_camera).grid(row=row, column=0, columnspan=2, sticky="w", pady=3); row += 1

        ttk.Separator(page).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(page, text="Presets", style="Header.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        self.preset_combo = ttk.Combobox(page, textvariable=self.preset_name_var, values=sorted(self.presets), state="readonly")
        self.preset_combo.grid(row=row, column=0, columnspan=2, sticky="ew", pady=3); row += 1
        ttk.Button(page, text="Save Current Preset", command=self.save_named_preset).grid(row=row, column=0, sticky="ew", padx=(0, 2), pady=3)
        ttk.Button(page, text="Load Preset", command=self.load_named_preset).grid(row=row, column=1, sticky="ew", padx=(2, 0), pady=3)

    def _add_scale(self, parent, text, variable, minimum, maximum, row, command=None) -> None:
        ttk.Label(parent, text=f"{text}:").grid(row=row, column=0, sticky="w", pady=(5, 2))
        ttk.Scale(parent, from_=minimum, to=maximum, variable=variable, orient="horizontal", style="Horizontal.TScale", command=command).grid(row=row + 1, column=0, sticky="ew", pady=(0, 6))

    def _on_effect_tab_changed(self, event=None) -> None:
        if not hasattr(self, "notebook"):
            return
        new_mode = self.notebook.index("current")
        previous_mode = getattr(self, "previous_effect_mode", new_mode)
        if new_mode != previous_mode:
            self.processor.reset_for_mode_change(previous_mode, new_mode)
            self.invalidate_pipeline(reset_results=False)
            self.previous_effect_mode = new_mode

    # ------------------------------------------------------------------
    # Camera and processing pipeline
    # ------------------------------------------------------------------
    def detect_cameras_async(self) -> None:
        if self.camera_scan_after_id is not None:
            try:
                self.root.after_cancel(self.camera_scan_after_id)
            except tk.TclError:
                pass
            self.camera_scan_after_id = None

        try:
            current = int(self.camera_idx_var.get())
        except ValueError:
            current = 0
        max_index = max(0, int(self.camera_scan_max_var.get()))
        backend = self.camera_backend_var.get()
        running = self.is_running
        self.camera_scan_generation += 1
        generation = self.camera_scan_generation
        self.camera_scan_active = True
        if hasattr(self, "camera_scan_btn"):
            self.camera_scan_btn.config(state="disabled")
        self.status_bar.config(text=f"Status: Scanning camera indices 0-{max_index}...")

        def scan() -> None:
            try:
                detected = CameraWorker.detect_cameras(
                    max_index + 1,
                    backend,
                    exclude={current} if running else set(),
                )
                cameras = sorted(set(([current] if running else []) + detected))
            except Exception:
                LOGGER.exception("Camera scan failed")
                cameras = [current] if running else []
            try:
                self._camera_scan_queue.put_nowait((generation, cameras))
            except Exception:
                try:
                    self._camera_scan_queue.get_nowait()
                    self._camera_scan_queue.put_nowait((generation, cameras))
                except Exception:
                    LOGGER.debug("Camera scan result could not be queued", exc_info=True)

        threading.Thread(target=scan, name="CameraScanner", daemon=True).start()
        self.camera_scan_after_id = self.root.after(100, self._poll_camera_scan)

    def _poll_camera_scan(self) -> None:
        try:
            generation, cameras = self._camera_scan_queue.get_nowait()
        except Empty:
            self.camera_scan_after_id = self.root.after(100, self._poll_camera_scan)
            return
        if generation != self.camera_scan_generation:
            self.camera_scan_after_id = self.root.after(100, self._poll_camera_scan)
            return
        self.camera_scan_active = False
        self.camera_scan_after_id = None
        if hasattr(self, "camera_scan_btn"):
            self.camera_scan_btn.config(state="normal")
        values = [str(index) for index in cameras] or ["0"]
        self.cam_combo["values"] = values
        if self.camera_idx_var.get() not in values:
            self.camera_idx_var.set(values[0])
        message = (
            f"Status: Detected cameras: {', '.join(values)}"
            if cameras
            else "Status: No camera detected; index 0 remains available"
        )
        self.status_bar.config(text=message)

    def start_webcam(self, automatic: bool = False) -> None:
        try:
            width, height = (int(value) for value in self.resolution_var.get().split("x", 1))
            camera_index = int(self.camera_idx_var.get())
        except ValueError:
            messagebox.showerror("Camera settings", "Camera index and resolution must be valid numbers.")
            return

        if not automatic:
            self.reconnect_attempts = 0
            self.next_camera_retry_delay = 1.0
        self.invalidate_pipeline()
        self.processor.reset_background_dependent_state()
        started = self.camera_worker.start(
            index=camera_index,
            width=width,
            height=height,
            fps=self.camera_fps_var.get(),
            backend=self.camera_backend_var.get(),
            buffer=self.camera_buffer_var.get(),
            exposure=self.exposure_var.get(),
            focus=self.focus_var.get(),
            max_read_failures=10,
        )
        if not started:
            self.is_running = False
            self.status_bar.config(text=f"Status: {self.camera_worker.error}")
            return
        self.camera_sequence = -1
        self.submitted_sequence = -1
        self.raw_frame = None
        self.processed_frame = None
        self.is_running = True
        self.camera_valid_since = None
        self.last_camera_retry = time.monotonic()
        self.status_bar.config(text=f"Engine Status: Camera {camera_index} running at {width}x{height}")

    def update_loop(self) -> None:
        if self.closing:
            return
        try:
            self._pipeline_iteration()
        except Exception:
            LOGGER.exception("UI update loop failed")
            self.status_bar.config(text="Status: Processing error; details written to the log")
        finally:
            if not self.closing:
                self.update_after_id = self.root.after(15, self.update_loop)

    def _pipeline_iteration(self) -> None:
        completed_recording = self.recorder.get_completed()
        if completed_recording is not None:
            self.record_btn.config(text="Start Recording", style="TButton", state="normal")
            self.status_bar.config(text=f"Status: {completed_recording['message']}")

        if self.virtual_camera.error:
            error = self.virtual_camera.error
            self.virtual_camera.error = None
            self.virtual_camera_var.set(False)
            self.virtual_camera.stop()
            self.status_bar.config(text=f"Status: Virtual camera stopped: {error}")

        if self.camera_worker.error or self.camera_worker.state is WorkerState.FAILED:
            self.is_running = False
            self.camera_valid_since = None
            message = self.camera_worker.error or "Camera is unavailable"
            self.show_placeholder(
                f"Webcam Unavailable\n{message}\nAutomatic reconnect is active"
            )
            now = time.monotonic()
            if self.reconnect_attempts >= self.maximum_automatic_reconnects:
                self.status_bar.config(
                    text=f"Status: {message}; automatic reconnect paused. Use Apply / Restart Camera."
                )
                return
            if now - self.last_camera_retry >= self.next_camera_retry_delay:
                self.last_camera_retry = now
                self.reconnect_attempts += 1
                self.next_camera_retry_delay = min(30.0, 2.0 ** min(self.reconnect_attempts, 5))
                self.status_bar.config(
                    text=(
                        f"Status: {message}; reconnect attempt "
                        f"{self.reconnect_attempts}/{self.maximum_automatic_reconnects}"
                    )
                )
                self.start_webcam(automatic=True)
            return

        sequence, frame = self.camera_worker.get_latest(self.camera_sequence)
        if frame is not None:
            now = time.monotonic()
            if self.camera_valid_since is None:
                self.camera_valid_since = now
            elif now - self.camera_valid_since >= 3.0 and self.reconnect_attempts:
                self.reconnect_attempts = 0
                self.next_camera_retry_delay = 1.0
            self.camera_sequence = sequence
            self.raw_frame = frame
            self._ensure_manual_masks(frame.shape[:2])
            process_every = int(
                QUALITY_PRESETS.get(
                    self.quality_var.get(), QUALITY_PRESETS["Balanced"]
                )["process_every"]
            )
            if sequence != self.submitted_sequence and sequence % process_every == 0:
                self.processor.submit(
                    ProcessingJob(
                        sequence=sequence,
                        frame=frame,
                        background=self.background_frame,
                        config=self._collect_processing_config(),
                        add_mask=self.manual_add_mask.copy() if self.manual_add_mask is not None else None,
                        remove_mask=self.manual_remove_mask.copy() if self.manual_remove_mask is not None else None,
                        selection_mask=self.selection_mask.copy() if self.selection_mask is not None else None,
                        generation=self.pipeline_generation,
                        background_version=self.background_version,
                    )
                )
                self.submitted_sequence = sequence

        result = self.processor.get_result()
        if result is None:
            return
        if not result_matches_state(
            result, self.pipeline_generation, self.background_version
        ):
            self.stale_results_discarded += 1
            return

        self.last_result = result
        self.raw_frame = result.raw
        self.processed_frame = result.final
        self.latest_person_mask = result.mask
        self.handle_gesture(result)

        output = self._select_preview(result)
        self._draw_status_overlays(output, result)
        self.render_to_canvas(output)

        timestamp = time.monotonic()
        if self.recorder.active:
            self.recorder.write(result.final, timestamp=timestamp)
        if self.virtual_camera.active:
            self.virtual_camera.send(result.final)

        self.metrics_history.append(result.processing_ms)
        samples = np.asarray(self.metrics_history, dtype=np.float32)
        latency = float(np.mean(samples)) if samples.size else 0.0
        p95_latency = float(np.percentile(samples, 95)) if samples.size else 0.0
        dropped = (
            self.processor.dropped_jobs
            + self.processor.replaced_results
            + self.camera_worker.dropped_frames
            + self.recorder.dropped_frames
            + self.virtual_camera.dropped_frames
            + self.stale_results_discarded
        )
        gpu_memory = float(self.runtime.get("gpu_memory_mb", 0.0))
        if result.sequence % 90 == 0:
            self.runtime = detect_runtime()
            gpu_memory = float(self.runtime.get("gpu_memory_mb", 0.0))
            self.runtime_lbl.config(text=self._runtime_text())
        self.metrics_lbl.config(
            text=(
                f"Capture: {self.camera_worker.capture_fps:.1f} FPS | "
                f"Processing: {self.processor.processed_fps:.1f} FPS | "
                f"Latency: {latency:.1f} ms (P95 {p95_latency:.1f}) | "
                f"GPU: {gpu_memory:.0f} MB | Dropped: {dropped}"
            )
        )
        target_text = "Target: tracked" if result.target_area >= 0.001 else "Target: lost / not detected"
        if result.motion_score > 0.12:
            target_text += " | Camera moved"
        self.target_status_lbl.config(text=target_text)

    def _collect_processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            mode=self.notebook.index("current"),
            model_backend=self.model_backend_var.get(),
            gestures=self.enable_gestures_var.get(),
            show_landmarks=self.show_landmarks_var.get(),
            ai_alpha=self.ai_alpha_val.get(),
            ai_threshold=self.ai_thresh_val.get(),
            cloak_alpha=self.cloak_alpha_val.get(),
            hsv_lower=self.hsv_lower.copy(),
            hsv_upper=self.hsv_upper.copy(),
            trail_decay=1.0 - self.trail_val.get() / 100.0,
            sub_alpha=self.sub_alpha_val.get(),
            sub_sensitivity=int(self.sub_sens_val.get()),
            freeze_alpha=self.freeze_alpha_val.get(),
            expand=int(round(self.edge_expand_var.get())),
            feather=int(round(self.edge_feather_var.get())),
            stability=self.temporal_stability_var.get(),
            maintain_person_mask=(self.notebook.index("current") == 4),
            target_selection_active=(self.interaction_mode_var.get() == "Select Target"),
        ).validated()

    def _select_preview(self, result: ProcessingResult) -> np.ndarray:
        mode = self.preview_mode_var.get()
        if mode == "Raw":
            return result.raw.copy()
        if mode == "Mask":
            if result.mask is None:
                return np.zeros_like(result.raw)
            mask8 = (np.clip(result.mask, 0.0, 1.0) * 255).astype(np.uint8)
            return cv2.applyColorMap(mask8, cv2.COLORMAP_TURBO)
        if mode == "Alpha":
            return result.alpha_preview.copy()
        if mode == "Split":
            raw = result.raw
            final = result.final
            separator = np.full((raw.shape[0], 4, 3), 220, dtype=np.uint8)
            return np.hstack([raw, separator, final])
        return result.final.copy()

    def _draw_status_overlays(self, frame: np.ndarray, result: ProcessingResult) -> None:
        h, w = frame.shape[:2]
        mode = self.notebook.index("current")
        requires_background = mode in (0, 1, 3)
        messages = []
        if requires_background and self.background_frame is None:
            messages.append("BACKGROUND NOT CAPTURED - use Capture Background")
        if result.motion_score > 0.12 and self.background_frame is not None:
            messages.append("CAMERA MOVED - RECALIBRATING / CAPTURE A NEW BACKGROUND")
        if mode in (0, 1, 3) and result.target_area < 0.001:
            messages.append("TARGET LOST OR NOT DETECTED")
        if result.error:
            messages.append(f"PROCESSING ERROR: {result.error}")
        if messages:
            box_height = 28 * len(messages) + 12
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (min(w - 10, 780), box_height), (20, 20, 24), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            for index, message in enumerate(messages):
                cv2.putText(frame, message, (22, 35 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (99, 46, 255), 2, cv2.LINE_AA)
        if time.monotonic() - self.sampling_feedback_time < 1.5:
            cv2.rectangle(frame, (10, h - 38), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.putText(frame, self.sampling_feedback_msg, (20, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 173, 181), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Background, freeze, selection, and gestures
    # ------------------------------------------------------------------
    def capture_background(self) -> None:
        if self.background_countdown_active:
            return
        if self.raw_frame is None:
            messagebox.showerror("Background", "No camera frame is available.")
            return
        self.background_countdown_active = True
        self._background_countdown_tick(3)

    def _background_countdown_tick(self, remaining: int) -> None:
        if remaining > 0:
            self.cap_bg_btn.config(text=f"Step out of frame... {remaining}")
            self.status_bar.config(text=f"Status: Capturing clean background in {remaining} second(s)")
            self.countdown_after_id = self.root.after(1000, lambda: self._background_countdown_tick(remaining - 1))
            return
        try:
            frames = self.camera_worker.get_recent_frames(20)
            if len(frames) < 3:
                raise ValueError("Not enough camera frames were collected")
            quality = background_quality(frames)
            self.background_frame = median_background(frames)
            self.background_version += 1
            self.processor.reset_background_dependent_state()
            self.invalidate_pipeline(reset_results=False)
            self.background_quality_info = quality
            color = "#00adb5" if quality["score"] >= 0.6 else "#ffb347"
            self.bg_status_lbl.config(text=f"Background: {quality['label']} ({quality['score'] * 100:.0f}%)", foreground=color)
            self.status_bar.config(text="Status: Temporal-median reference background captured")
        except Exception as exc:
            LOGGER.exception("Background capture failed")
            messagebox.showerror("Background", f"Failed to capture background: {exc}")
        finally:
            self.background_countdown_active = False
            self.cap_bg_btn.config(text="Capture Background (3s)")

    def capture_freeze(self) -> None:
        if self.raw_frame is None:
            messagebox.showerror("Freeze", "No camera stream is available.")
            return
        if self.latest_person_mask is None:
            messagebox.showerror("Freeze", "No valid person/target mask is available. Stand in view or capture a background for Motion Fallback.")
            return
        try:
            self.processor.capture_freeze(self.raw_frame.copy(), self.latest_person_mask.copy())
            self.invalidate_pipeline()
            self.freeze_status_lbl.config(text="Clone Status: Frozen", foreground="#00adb5")
            self.status_bar.config(text="Status: Actual target matte frozen successfully")
        except ValueError as exc:
            messagebox.showerror("Freeze", str(exc))

    def clear_freeze(self) -> None:
        self.processor.clear_freeze()
        self.invalidate_pipeline()
        self.freeze_status_lbl.config(text="Clone Status: Not Frozen", foreground="#888888")
        self.status_bar.config(text="Status: Frozen clone cleared")

    def _ensure_manual_masks(self, frame_shape) -> None:
        h, w = frame_shape
        if self.manual_add_mask is None:
            self.manual_add_mask = np.zeros((h, w), dtype=np.uint8)
            self.manual_remove_mask = np.zeros((h, w), dtype=np.uint8)
            return
        if self.manual_add_mask.shape == (h, w):
            return

        old_h, old_w = self.manual_add_mask.shape
        old_ratio = old_w / max(1, old_h)
        new_ratio = w / max(1, h)
        if abs(old_ratio - new_ratio) <= 0.02:
            self.manual_add_mask = cv2.resize(
                self.manual_add_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            self.manual_remove_mask = cv2.resize(
                self.manual_remove_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            if self.selection_mask is not None:
                self.selection_mask = cv2.resize(
                    self.selection_mask, (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)
            self.selection_history.clear()
            self.selection_redo.clear()
            self.status_bar.config(
                text="Status: Target edits were resized for the new camera resolution"
            )
        else:
            self.manual_add_mask = np.zeros((h, w), dtype=np.uint8)
            self.manual_remove_mask = np.zeros((h, w), dtype=np.uint8)
            self.selection_mask = None
            self.selection_history.clear()
            self.selection_redo.clear()
            self.status_bar.config(
                text="Status: Target edits were reset because the aspect ratio changed"
            )

    @staticmethod
    def _encode_history_mask(mask: Optional[np.ndarray]):
        if mask is None:
            return None
        image = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return (mask.shape, mask.astype(np.uint8).tobytes(), False)
        return (mask.shape, encoded.tobytes(), True)

    @staticmethod
    def _decode_history_mask(payload):
        if payload is None:
            return None
        shape, data, compressed = payload
        if compressed:
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError("Could not restore compressed selection history")
            return (decoded > 127).astype(np.uint8)
        return np.frombuffer(data, dtype=np.uint8).reshape(shape).copy()

    def _snapshot_selection(self) -> tuple:
        return (
            self._encode_history_mask(self.manual_add_mask),
            self._encode_history_mask(self.manual_remove_mask),
            self._encode_history_mask(self.selection_mask),
            self.hue_center,
            self.sampled_s,
            self.sampled_v,
        )

    def _restore_selection(self, state: tuple) -> None:
        add, remove, selected, hue, sat, val = state
        self.manual_add_mask = self._decode_history_mask(add)
        self.manual_remove_mask = self._decode_history_mask(remove)
        self.selection_mask = self._decode_history_mask(selected)
        self.hue_center, self.sampled_s, self.sampled_v = hue, sat, val
        self.recalculate_hsv_ranges()
        self.invalidate_pipeline()

    def _push_selection_history(self) -> None:
        self.selection_history.append(self._snapshot_selection())
        if len(self.selection_history) > 30:
            self.selection_history.pop(0)
        self.selection_redo.clear()

    def undo_selection(self) -> None:
        if not self.selection_history:
            return
        self.selection_redo.append(self._snapshot_selection())
        self._restore_selection(self.selection_history.pop())
        self.status_bar.config(text="Status: Undid target/color selection change")

    def redo_selection(self) -> None:
        if not self.selection_redo:
            return
        self.selection_history.append(self._snapshot_selection())
        self._restore_selection(self.selection_redo.pop())
        self.status_bar.config(text="Status: Redid target/color selection change")

    def clear_manual_selection(self) -> None:
        if self.raw_frame is None:
            return
        self._push_selection_history()
        h, w = self.raw_frame.shape[:2]
        self.manual_add_mask = np.zeros((h, w), dtype=np.uint8)
        self.manual_remove_mask = np.zeros((h, w), dtype=np.uint8)
        self.selection_mask = None
        self.invalidate_pipeline()
        self.status_bar.config(text="Status: Manual target edits cleared")

    def _event_frame_point(self, event) -> Optional[tuple[int, int]]:
        if self.raw_frame is None:
            return None
        h, w = self.raw_frame.shape[:2]
        if self.preview_mode_var.get() == "Split":
            return map_split_preview_to_frame(
                event.x,
                event.y,
                self.render_offset_x,
                self.render_offset_y,
                self.render_width,
                self.render_height,
                w,
                h,
                separator_width=4,
            )
        return map_canvas_to_frame(
            event.x,
            event.y,
            self.render_offset_x,
            self.render_offset_y,
            self.render_width,
            self.render_height,
            w,
            h,
        )

    def on_canvas_press(self, event) -> None:
        point = self._event_frame_point(event)
        if point is None:
            return
        mode = self.interaction_mode_var.get()
        self.invalidate_pipeline()
        if mode == "Color Sample":
            if self.notebook.index("current") != 1:
                self.status_bar.config(text="Status: Color sampling is available in the Color Cloak tab")
                return
            self._push_selection_history()
            self.sample_color_patch(*point)
        elif mode == "Select Target":
            self._push_selection_history()
            self.select_target_component(*point)
        else:
            self._push_selection_history()
            self._drawing = True
            self._paint_mask(*point, add=(mode == "Brush Add"))

    def on_canvas_drag(self, event) -> None:
        if not self._drawing:
            return
        point = self._event_frame_point(event)
        if point is not None:
            self._paint_mask(*point, add=(self.interaction_mode_var.get() == "Brush Add"))

    def on_canvas_release(self, event) -> None:
        if self._drawing:
            self.invalidate_pipeline()
        self._drawing = False

    def _paint_mask(self, x: int, y: int, add: bool) -> None:
        if self.manual_add_mask is None or self.manual_remove_mask is None:
            return
        radius = max(1, int(self.brush_size_var.get()))
        target = self.manual_add_mask if add else self.manual_remove_mask
        other = self.manual_remove_mask if add else self.manual_add_mask
        cv2.circle(target, (x, y), radius, 1, -1)
        cv2.circle(other, (x, y), radius, 0, -1)
        self.sampling_feedback_time = time.monotonic()
        self.sampling_feedback_msg = "Added target area" if add else "Removed target area"

    def select_target_component(self, x: int, y: int) -> None:
        if self.latest_person_mask is None:
            self.status_bar.config(text="Status: No current mask is available for click-to-select")
            return
        binary = (self.latest_person_mask >= 0.35).astype(np.uint8)
        if y >= binary.shape[0] or x >= binary.shape[1] or binary[y, x] == 0:
            self.status_bar.config(text="Status: Click inside the detected target mask")
            return
        count, labels = cv2.connectedComponents(binary)
        label = labels[y, x]
        if label <= 0:
            return
        self.selection_mask = (labels == label).astype(np.uint8)
        self.status_bar.config(text="Status: Selected the clicked connected target")

    def sample_color_patch(self, frame_x: int, frame_y: int) -> None:
        assert self.raw_frame is not None
        h, w = self.raw_frame.shape[:2]
        radius = 5
        x0, x1 = max(0, frame_x - radius), min(w, frame_x + radius + 1)
        y0, y1 = max(0, frame_y - radius), min(h, frame_y + radius + 1)
        patch = self.raw_frame[y0:y1, x0:x1]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        bgr_median = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
        self.hue_center, self.sampled_s, self.sampled_v = [int(value) for value in np.median(hsv_patch, axis=0)]
        self.recalculate_hsv_ranges()
        rgb = (int(bgr_median[2]), int(bgr_median[1]), int(bgr_median[0]))
        self.color_box_frame.configure(bg="#%02x%02x%02x" % rgb)
        self.sampling_feedback_time = time.monotonic()
        self.sampling_feedback_msg = f"Sampled median HSV: ({self.hue_center}, {self.sampled_s}, {self.sampled_v})"
        self.status_bar.config(text=self.sampling_feedback_msg)

    def recalculate_hsv_ranges(self, event=None) -> None:
        tolerance = int(self.cloak_h_tol.get())
        minimum = int(self.cloak_sv_min.get())
        lower_h = (self.hue_center - tolerance) % 180
        upper_h = (self.hue_center + tolerance) % 180
        self.hsv_lower = np.array([lower_h, max(minimum, self.sampled_s - 70), max(minimum, self.sampled_v - 70)], dtype=np.int32)
        self.hsv_upper = np.array([upper_h, 255, 255], dtype=np.int32)
        if hasattr(self, "color_lbl"):
            wrap = " (red hue wrap)" if lower_h > upper_h else ""
            self.color_lbl.config(text=f"Target HSV: {self.hsv_lower.tolist()} to {self.hsv_upper.tolist()}{wrap}")

    def on_gesture_toggle(self) -> None:
        if self.enable_gestures_var.get() and not MEDIAPIPE_AVAILABLE:
            messagebox.showwarning("Gestures", "MediaPipe is not available, so hand gestures cannot be enabled.")
            self.enable_gestures_var.set(False)

    def handle_gesture(self, result: ProcessingResult) -> None:
        gesture = result.gesture
        if not self.enable_gestures_var.get() or not gesture:
            return
        now = time.monotonic()
        selected_tab = self.notebook.index("current")
        if gesture == "Pinch":
            if selected_tab == 0:
                self.ai_alpha_val.set(result.pinch_value)
            elif selected_tab == 1:
                self.cloak_alpha_val.set(result.pinch_value)
            elif selected_tab == 3:
                self.sub_alpha_val.set(result.pinch_value)
            elif selected_tab == 4:
                self.freeze_alpha_val.set(max(0.1, result.pinch_value))
            return

        tab_mapping = {"1 Finger": 0, "2 Fingers": 1, "3 Fingers": 2, "4 Fingers": 3, "5 Fingers (Palm)": 4}
        if gesture in tab_mapping and now - self.last_switch_tab_time > 1.5:
            self.notebook.select(tab_mapping[gesture])
            self.last_switch_tab_time = now
            self.status_bar.config(text=f"Status: Switched effect via {gesture}")
            return
        if now - self.last_gesture_action < 2.0:
            return
        if gesture == "Thumbs Up":
            self.capture_background()
            self.last_gesture_action = now
        elif gesture == "Fist" and selected_tab == 4:
            self.capture_freeze()
            self.last_gesture_action = now

    # ------------------------------------------------------------------
    # Output, presets, diagnostics, and lifecycle
    # ------------------------------------------------------------------
    def save_screenshot(self) -> None:
        if self.processed_frame is None:
            messagebox.showerror("Screenshot", "No processed frame is available.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg")], initialdir=self.settings.get("last_directory") or None, title="Save Ghost Effect Screenshot")
        if not path:
            return
        try:
            safe_imwrite(path, self.processed_frame)
            self.settings["last_directory"] = str(Path(path).parent)
            self.status_bar.config(text=f"Status: Screenshot saved to {path}")
        except Exception as exc:
            LOGGER.exception("Screenshot save failed")
            messagebox.showerror("Screenshot", f"Failed to save image: {exc}")

    def export_transparent_target(self) -> None:
        if self.raw_frame is None or self.latest_person_mask is None:
            messagebox.showerror("Transparent export", "A frame and target alpha mask are required.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Transparent PNG", "*.png")], title="Export Target With Alpha")
        if not path:
            return
        mask = self.latest_person_mask
        if mask.shape != self.raw_frame.shape[:2]:
            mask = cv2.resize(mask, (self.raw_frame.shape[1], self.raw_frame.shape[0]))
        bgra = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2BGRA)
        bgra[..., 3] = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        try:
            safe_imwrite(path, bgra)
        except Exception as exc:
            messagebox.showerror("Transparent export", str(exc))
            return
        self.status_bar.config(text=f"Status: Transparent target exported to {path}")

    def toggle_recording(self) -> None:
        if self.recorder.finalizing:
            self.status_bar.config(text="Status: Recording is still being finalized")
            return
        if self.recorder.active:
            result = self.recorder.stop(wait=False)
            self.record_btn.config(text="Finalizing...", style="TButton", state="disabled")
            self.status_bar.config(text=f"Status: {result['message']}")
            return
        if self.processed_frame is None:
            messagebox.showerror("Recording", "No processed frame is available.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            title="Save Processed Recording",
        )
        if not path:
            return
        h, w = self.processed_frame.shape[:2]
        try:
            result = self.recorder.start(
                path,
                (w, h),
                fps=max(1.0, self.processor.processed_fps or self.camera_fps_var.get()),
                with_audio=self.audio_recording_var.get(),
            )
            self.record_btn.config(text="Stop Recording", style="Danger.TButton", state="normal")
            self.status_bar.config(text=f"Status: {result['message']}")
        except Exception as exc:
            LOGGER.exception("Recording could not start")
            messagebox.showerror("Recording", str(exc))

    def toggle_virtual_camera(self) -> None:
        if not self.virtual_camera_var.get():
            self.virtual_camera.stop()
            self.status_bar.config(text="Status: Virtual camera disabled")
            return
        if self.processed_frame is None:
            self.virtual_camera_var.set(False)
            messagebox.showerror("Virtual camera", "No processed frame is available yet.")
            return
        h, w = self.processed_frame.shape[:2]
        try:
            self.virtual_camera.start(w, h, self.camera_fps_var.get())
            self.status_bar.config(text="Status: Virtual camera output enabled")
        except Exception as exc:
            self.virtual_camera_var.set(False)
            messagebox.showerror("Virtual camera", f"Could not start virtual camera: {exc}")

    def apply_quality_preset(self, event=None) -> None:
        preset = QUALITY_PRESETS.get(self.quality_var.get(), QUALITY_PRESETS["Balanced"])
        self.edge_expand_var.set(preset["expand"])
        self.edge_feather_var.set(preset["feather"])
        self.temporal_stability_var.set(preset["stability"])
        self.invalidate_pipeline()
        self.status_bar.config(text=f"Status: Applied {self.quality_var.get()} quality preset")

    def _current_preset_payload(self) -> Dict[str, Any]:
        return {
            "quality": self.quality_var.get(),
            "model_backend": self.model_backend_var.get(),
            "ai_alpha": self.ai_alpha_val.get(),
            "ai_threshold": self.ai_thresh_val.get(),
            "cloak_alpha": self.cloak_alpha_val.get(),
            "hue_center": self.hue_center,
            "hue_tolerance": self.cloak_h_tol.get(),
            "sv_min": self.cloak_sv_min.get(),
            "trail": self.trail_val.get(),
            "sub_alpha": self.sub_alpha_val.get(),
            "sub_sensitivity": self.sub_sens_val.get(),
            "freeze_alpha": self.freeze_alpha_val.get(),
            "edge_expand": self.edge_expand_var.get(),
            "edge_feather": self.edge_feather_var.get(),
            "stability": self.temporal_stability_var.get(),
        }

    def save_named_preset(self) -> None:
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        self.presets[name] = self._current_preset_payload()
        save_presets(self.presets)
        self.preset_combo["values"] = sorted(self.presets)
        self.preset_name_var.set(name)
        self.status_bar.config(text=f"Status: Saved preset '{name}'")

    def load_named_preset(self) -> None:
        name = self.preset_name_var.get()
        payload = self.presets.get(name)
        if not payload:
            return
        self.quality_var.set(payload.get("quality", "Balanced"))
        self.model_backend_var.set(payload.get("model_backend", "Auto"))
        self.ai_alpha_val.set(payload.get("ai_alpha", 0.0))
        self.ai_thresh_val.set(payload.get("ai_threshold", 0.15))
        self.cloak_alpha_val.set(payload.get("cloak_alpha", 0.0))
        self.hue_center = int(payload.get("hue_center", 60))
        self.cloak_h_tol.set(payload.get("hue_tolerance", 15))
        self.cloak_sv_min.set(payload.get("sv_min", 40))
        self.trail_val.set(payload.get("trail", 80.0))
        self.sub_alpha_val.set(payload.get("sub_alpha", 0.0))
        self.sub_sens_val.set(payload.get("sub_sensitivity", 25))
        self.freeze_alpha_val.set(payload.get("freeze_alpha", 0.5))
        self.edge_expand_var.set(payload.get("edge_expand", 1))
        self.edge_feather_var.set(payload.get("edge_feather", 11))
        self.temporal_stability_var.set(payload.get("stability", 0.65))
        self.recalculate_hsv_ranges()
        self.processor.reset()
        self.invalidate_pipeline(reset_results=False)
        self.status_bar.config(text=f"Status: Loaded preset '{name}'")

    def export_diagnostics(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON report", "*.json")], title="Export Diagnostic Report")
        if not path:
            return
        payload = {
            "settings": self._settings_payload(),
            "metrics": {
                "capture_fps": self.camera_worker.capture_fps,
                "processing_fps": self.processor.processed_fps,
                "average_latency_ms": float(np.mean(self.metrics_history)) if self.metrics_history else 0.0,
                "p95_latency_ms": float(np.percentile(self.metrics_history, 95)) if self.metrics_history else 0.0,
                "processing_jobs_replaced": self.processor.dropped_jobs,
                "processing_results_replaced": self.processor.replaced_results,
                "camera_frames_overwritten": self.camera_worker.dropped_frames,
                "camera_read_failures": self.camera_worker.read_failures,
                "recording_frames_dropped": self.recorder.dropped_frames,
                "recording_frames_duplicated": self.recorder.frames_duplicated,
                "virtual_camera_frames_dropped": self.virtual_camera.dropped_frames,
                "stale_results_discarded": self.stale_results_discarded,
                "pipeline_generation": self.pipeline_generation,
                "background_version": self.background_version,
            },
            "background_quality": self.background_quality_info,
            "target_area": self.last_result.target_area if self.last_result else 0.0,
            "model": {
                "selected_backend": self.model_backend_var.get(),
                "mediapipe_available": MEDIAPIPE_AVAILABLE,
            },
            "optional_features": {
                "sounddevice_available": SOUNDDEVICE_AVAILABLE,
                "pyvirtualcam_available": PYVIRTUALCAM_AVAILABLE,
            },
            "log_file": str(LOG_DIR / "ghost_invisibility.log"),
        }
        write_diagnostics(path, payload)
        self.status_bar.config(text=f"Status: Diagnostic report saved to {path}")

    def open_model_manager(self) -> None:
        window = tk.Toplevel(self.root)
        window.title("Model & Extension Manager")
        window.geometry("620x430")
        window.configure(bg="#1e1e24")
        text = tk.Text(window, bg="#16161a", fg="#eeeeee", insertbackground="#eeeeee", wrap="word", padx=12, pady=12)
        text.pack(fill="both", expand=True, padx=10, pady=10)
        report = [
            "Current model backends\n",
            f"• MediaPipe Selfie Segmentation: {'Available' if MEDIAPIPE_AVAILABLE else 'Unavailable'}\n",
            "  The model is bundled/managed by the MediaPipe package; no separate weight download is required.\n",
            "• Motion Fallback: Available (OpenCV only)\n\n",
            "Optional output extensions\n",
            f"• Microphone audio: {'Available' if SOUNDDEVICE_AVAILABLE else 'Install sounddevice'}\n",
            f"• Virtual camera: {'Available' if PYVIRTUALCAM_AVAILABLE else 'Install pyvirtualcam and a supported virtual-camera driver'}\n",
            f"• FFmpeg audio/video muxing: {'Available' if self.runtime.get('ffmpeg') else 'Install FFmpeg and add it to PATH'}\n\n",
            "This manager reports and validates real installed backends only; it does not display a model as active when initialization failed.",
        ]
        text.insert("1.0", "".join(report))
        text.config(state="disabled")
        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 10))

    def reset_settings(self) -> None:
        self.ai_alpha_val.set(0.0)
        self.ai_thresh_val.set(0.15)
        self.cloak_alpha_val.set(0.0)
        self.cloak_h_tol.set(15)
        self.cloak_sv_min.set(40)
        self.trail_val.set(80.0)
        self.sub_alpha_val.set(0.0)
        self.sub_sens_val.set(25)
        self.freeze_alpha_val.set(0.5)
        self.hue_center, self.sampled_s, self.sampled_v = 60, 255, 255
        self.recalculate_hsv_ranges()
        self.color_box_frame.configure(bg="#00ff00")
        self.background_frame = None
        self.background_version += 1
        self.background_quality_info = {"score": 0.0, "label": "Empty"}
        self.bg_status_lbl.config(text="Background: Empty", foreground="#ff2e63")
        self.processor.reset()
        self.invalidate_pipeline(reset_results=False)
        self.freeze_status_lbl.config(text="Clone Status: Not Frozen", foreground="#888888")
        self.clear_manual_selection()
        self.quality_var.set("Balanced")
        self.apply_quality_preset()
        self.status_bar.config(text="Status: Effect settings and captured state reset")

    def bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda event: self.save_screenshot())
        self.root.bind("<Control-z>", lambda event: self.undo_selection())
        self.root.bind("<Control-y>", lambda event: self.redo_selection())
        self.root.bind("<space>", lambda event: self.capture_background())
        self.root.bind("<Key-f>", lambda event: self.capture_freeze())
        self.root.bind("<Key-r>", lambda event: self.toggle_recording())
        self.root.bind("<Key-d>", lambda event: self.export_diagnostics())
        for index in range(5):
            self.root.bind(str(index + 1), lambda event, tab=index: self.notebook.select(tab))

    def _runtime_text(self) -> str:
        gpu = str(self.runtime.get("gpu", "CPU"))
        return f"Model: {'MediaPipe' if MEDIAPIPE_AVAILABLE else 'Fallback'} | Device: {gpu[:36]}"

    def _settings_payload(self) -> Dict[str, Any]:
        return {
            "camera_index": int(self.camera_idx_var.get() or 0),
            "resolution": self.resolution_var.get(),
            "camera_fps": self.camera_fps_var.get(),
            "camera_backend": self.camera_backend_var.get(),
            "camera_buffer": self.camera_buffer_var.get(),
            "camera_scan_max": self.camera_scan_max_var.get(),
            "exposure": self.exposure_var.get(),
            "focus": self.focus_var.get(),
            "quality": self.quality_var.get(),
            "model_backend": self.model_backend_var.get(),
            "preview_mode": self.preview_mode_var.get(),
            "audio_recording": self.audio_recording_var.get(),
            "show_landmarks": self.show_landmarks_var.get(),
            "last_directory": self.settings.get("last_directory", ""),
        }

    def render_to_canvas(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())
        img_h, img_w = rgb.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(resized))
        dx = (canvas_w - new_w) // 2
        dy = (canvas_h - new_h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(dx, dy, anchor=tk.NW, image=self.img_tk)
        self.render_offset_x, self.render_offset_y = dx, dy
        self.render_width, self.render_height = new_w, new_h
        self.render_frame_width, self.render_frame_height = img_w, img_h

    def show_placeholder(self, message: str) -> None:
        self.canvas.delete("all")
        width = max(640, self.canvas.winfo_width())
        height = max(480, self.canvas.winfo_height())
        self.canvas.create_rectangle(0, 0, width, height, fill="#1a1a1e", outline="")
        self.canvas.create_text(width // 2, height // 2, text=message, fill="#888888", font=("Segoe UI", 14, "bold"), justify="center")

    def cleanup(self) -> None:
        self.closing = True
        try:
            if self.update_after_id is not None:
                self.root.after_cancel(self.update_after_id)
            if self.countdown_after_id is not None:
                self.root.after_cancel(self.countdown_after_id)
            if self.camera_scan_after_id is not None:
                self.root.after_cancel(self.camera_scan_after_id)
        except tk.TclError:
            pass
        try:
            if self.recorder.active or self.recorder.finalizing:
                self.recorder.stop(wait=True)
        except Exception:
            LOGGER.exception("Could not finalize recording during shutdown")
        self.virtual_camera.stop()
        self.camera_worker.stop()
        self.processor.stop()
        self.settings.update(self._settings_payload())
        save_settings(self.settings)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = GhostInvisibilityApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()


if __name__ == "__main__":
    main()
