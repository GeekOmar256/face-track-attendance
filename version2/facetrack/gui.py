"""Desktop window for the Face Track pipeline, built on Tkinter.

This is the local alternative to the browser interface in webapp.py. Both drive
the same PipelineSession, so they behave identically; the difference is only
how the frames reach a screen.

Prefer the web interface on the Raspberry Pi. This window needs a display, and
on a headless board that means X11 forwarding over SSH, which is noticeably
slower than JPEG frames over HTTP.

Frames reach the canvas as raw PPM bytes through tk.PhotoImage, so Pillow is
not required: the only extra piece is python3-tk.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


import cv2
import numpy as np

from . import config
from .session import PipelineSession
from .utils import write_csv

VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
IMAGE_TYPES = [("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]

STAT_FIELDS = [
    ("frames", "Frames"),
    ("total_faces", "Faces"),
    ("detection_rate", "Detection rate"),
    ("throughput_fps", "Throughput"),
    ("mean_detect_ms", "Detection"),
    ("mean_pipeline_ms", "Pipeline mean"),
    ("max_pipeline_ms", "Pipeline max"),
    ("unknown", "Unknown faces"),
]
UNITS = {"detection_rate": "%", "throughput_fps": " fps", "mean_detect_ms": " ms",
         "mean_pipeline_ms": " ms", "max_pipeline_ms": " ms"}


class FaceTrackGUI:
    def __init__(self, root: tk.Tk, args=None) -> None:
        self.root = root
        self.root.title("Face Track - detection and recognition")
        self.root.minsize(1040, 700)

        self.session = PipelineSession()
        self._photo = None                    # PhotoImage must stay referenced
        self._last_frame_id = -1

        self._build_variables(args)
        self._build_layout()
        self._tick()

    # ------------------------------------------------------------ variables
    def _build_variables(self, args) -> None:
        self.source_kind = tk.StringVar(value="camera")
        self.source_path = tk.StringVar(value="")
        self.camera_index = tk.IntVar(value=config.CAMERA_INDEX)
        self.loop_source = tk.BooleanVar(value=False)

        self.detector_name = tk.StringVar(value=getattr(args, "detector", None) or "haar")
        self.scale_factor = tk.DoubleVar(value=config.HAAR_SCALE_FACTOR)
        self.min_neighbors = tk.IntVar(value=config.HAAR_MIN_NEIGHBORS)
        self.min_size = tk.IntVar(value=config.HAAR_MIN_SIZE[0])
        self.score_threshold = tk.DoubleVar(value=config.YUNET_SCORE_THRESHOLD)

        self.recognizer_name = tk.StringVar(value="none")
        self.recog_threshold = tk.StringVar(value="")
        self.status = tk.StringVar(value="Ready. Choose a source and press Start.")

    # --------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)

        preview = ttk.LabelFrame(top, text="Preview", padding=4)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(preview, bg="#11141a", highlightthickness=0,
                                width=660, height=500)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(top)
        controls.grid(row=0, column=1, sticky="ns")
        self._build_source_box(controls)
        self._build_detector_box(controls)
        self._build_recognizer_box(controls)
        self._build_buttons(controls)

        self._build_stats_box(outer)
        ttk.Label(outer, textvariable=self.status, relief="sunken", anchor="w",
                  padding=4).grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_source_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Source", padding=8)
        box.pack(fill="x", pady=(0, 8))
        box.columnconfigure(0, weight=1)

        ttk.Radiobutton(box, text="Webcam / Pi camera", value="camera",
                        variable=self.source_kind, command=self._sync_source).grid(
            row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text="index").grid(row=1, column=0, sticky="w", padx=(20, 4))
        ttk.Spinbox(box, from_=0, to=8, width=4, textvariable=self.camera_index).grid(
            row=1, column=1, sticky="w")

        ttk.Radiobutton(box, text="Video file", value="video",
                        variable=self.source_kind, command=self._sync_source).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Radiobutton(box, text="Photo or folder of photos", value="photos",
                        variable=self.source_kind, command=self._sync_source).grid(
            row=3, column=0, columnspan=3, sticky="w")

        self.path_entry = ttk.Entry(box, textvariable=self.source_path, width=26)
        self.path_entry.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.browse_btn = ttk.Button(box, text="Browse", width=8, command=self._browse)
        self.browse_btn.grid(row=4, column=2, sticky="e", padx=(4, 0), pady=(6, 0))
        self.loop_check = ttk.Checkbutton(box, text="Loop when the source ends",
                                          variable=self.loop_source)
        self.loop_check.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._sync_source()

    def _build_detector_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Detection", padding=8)
        box.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(box, text="Haar Cascade (baseline)", value="haar",
                        variable=self.detector_name, command=self._sync_detector).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(box, text="YuNet (DNN)", value="yunet",
                        variable=self.detector_name, command=self._sync_detector).grid(
            row=1, column=0, columnspan=2, sticky="w")

        self.haar_widgets = []
        for i, (label, var, lo, hi, step) in enumerate([
            ("scaleFactor", self.scale_factor, 1.01, 2.0, 0.01),
            ("minNeighbors", self.min_neighbors, 1, 20, 1),
            ("min size (px)", self.min_size, 20, 400, 10),
        ]):
            lab = ttk.Label(box, text=label)
            lab.grid(row=2 + i, column=0, sticky="w", padx=(20, 4), pady=1)
            spin = ttk.Spinbox(box, from_=lo, to=hi, increment=step, width=7,
                               textvariable=var)
            spin.grid(row=2 + i, column=1, sticky="w", pady=1)
            self.haar_widgets += [lab, spin]

        lab = ttk.Label(box, text="score threshold")
        lab.grid(row=5, column=0, sticky="w", padx=(20, 4), pady=1)
        spin = ttk.Spinbox(box, from_=0.05, to=1.0, increment=0.05, width=7,
                           textvariable=self.score_threshold)
        spin.grid(row=5, column=1, sticky="w", pady=1)
        self.yunet_widgets = [lab, spin]
        self._sync_detector()

    def _build_recognizer_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Recognition", padding=8)
        box.pack(fill="x", pady=(0, 8))
        for i, (text, value) in enumerate([("Off, detection only", "none"),
                                           ("LBPH (baseline)", "lbph"),
                                           ("SFace (embeddings)", "sface")]):
            ttk.Radiobutton(box, text=text, value=value, variable=self.recognizer_name,
                            command=self._sync_recognizer).grid(
                row=i, column=0, columnspan=2, sticky="w")

        self.thr_label = ttk.Label(box, text="threshold")
        self.thr_label.grid(row=3, column=0, sticky="w", padx=(20, 4), pady=(4, 0))
        self.thr_entry = ttk.Entry(box, textvariable=self.recog_threshold, width=8)
        self.thr_entry.grid(row=3, column=1, sticky="w", pady=(4, 0))
        self.thr_hint = ttk.Label(box, text="blank uses config.py", foreground="#666")
        self.thr_hint.grid(row=4, column=0, columnspan=2, sticky="w", padx=(20, 0))
        self._sync_recognizer()

    def _build_buttons(self, parent) -> None:
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(4, 0))
        self.start_btn = ttk.Button(box, text="Start", command=self.start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.stop_btn = ttk.Button(box, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=3)
        ttk.Button(box, text="Snapshot", command=self.snapshot).pack(
            side="left", expand=True, fill="x", padx=(3, 0))

    def _build_stats_box(self, parent) -> None:
        box = ttk.LabelFrame(parent, text="Statistics", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        for c in range(8):
            box.columnconfigure(c, weight=1)

        self.stat_vars = {}
        for i, (key, label) in enumerate(STAT_FIELDS):
            ttk.Label(box, text=label, foreground="#555").grid(row=0, column=i, sticky="w")
            var = tk.StringVar(value="-")
            self.stat_vars[key] = var
            lab = ttk.Label(box, textvariable=var, font=("TkDefaultFont", 11, "bold"))
            lab.grid(row=1, column=i, sticky="w")

        ttk.Separator(box, orient="horizontal").grid(row=2, column=0, columnspan=8,
                                                     sticky="ew", pady=6)
        ttk.Label(box, text="Identified", foreground="#555").grid(
            row=3, column=0, columnspan=8, sticky="w")
        self.identity_text = tk.Text(box, height=4, wrap="none", state="disabled",
                                     background="#fafafa", relief="flat")
        self.identity_text.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(2, 0))

        btns = ttk.Frame(box)
        btns.grid(row=4, column=6, columnspan=2, sticky="ne", pady=(2, 0))
        ttk.Button(btns, text="Reset statistics", command=self.reset_stats).pack(
            fill="x", pady=(0, 3))
        ttk.Button(btns, text="Export CSV", command=self.export_stats).pack(fill="x")

    # ------------------------------------------------------- widget syncing
    def _sync_source(self) -> None:
        camera = self.source_kind.get() == "camera"
        state = "disabled" if camera else "normal"
        self.path_entry.configure(state=state)
        self.browse_btn.configure(state=state)
        self.loop_check.configure(state=state)

    def _sync_detector(self) -> None:
        haar = self.detector_name.get() == "haar"
        for widget in self.haar_widgets:
            widget.configure(state="normal" if haar else "disabled")
        for widget in self.yunet_widgets:
            widget.configure(state="disabled" if haar else "normal")

    def _sync_recognizer(self) -> None:
        on = self.recognizer_name.get() != "none"
        for widget in (self.thr_label, self.thr_entry, self.thr_hint):
            widget.configure(state="normal" if on else "disabled")

    def _browse(self) -> None:
        if self.source_kind.get() == "video":
            path = filedialog.askopenfilename(title="Choose a video", filetypes=VIDEO_TYPES)
        else:
            path = filedialog.askopenfilename(title="Choose a photo", filetypes=IMAGE_TYPES)
            if not path:
                path = filedialog.askdirectory(title="Or choose a folder of photos")
        if path:
            self.source_path.set(path)

    # -------------------------------------------------------------- running
    def start(self) -> None:
        if self.session.state == "running":
            return
        camera = self.source_kind.get() == "camera"
        source = None
        if not camera:
            source = self.source_path.get().strip()
            if not source:
                messagebox.showerror("Cannot start", "Choose a file or folder first.")
                return

        threshold = self.recog_threshold.get().strip()
        try:
            threshold = float(threshold) if threshold else None
        except ValueError:
            messagebox.showerror("Cannot start", "The threshold must be a number.")
            return

        try:
            self.session.start(
                source=source,
                camera_index=self.camera_index.get(),
                detector=self.detector_name.get(),
                recognizer=self.recognizer_name.get(),
                threshold=threshold,
                loop=self.loop_source.get(),
                scale_factor=self.scale_factor.get(),
                min_neighbors=self.min_neighbors.get(),
                min_size=self.min_size.get(),
                score_threshold=self.score_threshold.get(),
            )
        except Exception as exc:
            messagebox.showerror("Cannot start", f"{exc}")
            self.status.set(str(exc))
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.set(f"Running: {self.session.detector_description}")

    def stop(self) -> None:
        self.session.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status.set("Stopped. Statistics kept, press Start to keep adding to them.")

    # --------------------------------------------------------------- ticking
    def _tick(self) -> None:
        frame_id = self.session.frame_id
        if frame_id != self._last_frame_id:
            self._last_frame_id = frame_id
            frame = self.session.latest_frame()
            if frame is not None:
                self._render(frame)

        state = self.session.state
        if state in ("finished", "error") and str(self.stop_btn["state"]) != "disabled":
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            if state == "error":
                messagebox.showerror("Pipeline error", self.session.error or "unknown error")
                self.status.set(self.session.error or "error")
            else:
                self.status.set("Source finished.")

        self._refresh_stats()
        self.root.after(30, self._tick)

    def _render(self, frame: np.ndarray) -> None:
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        h, w = frame.shape[:2]
        scale = min(width / w, height / h)
        if scale != 1.0:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)

        # tk.PhotoImage reads raw PPM, so Pillow is not needed. OpenCV writes the
        # channel order correctly from a BGR array.
        ok, buffer = cv2.imencode(".ppm", frame)
        if not ok:
            return
        self._photo = tk.PhotoImage(data=buffer.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self._photo)

    def _refresh_stats(self) -> None:
        data = self.session.stats.snapshot()
        for key, _ in STAT_FIELDS:
            self.stat_vars[key].set(f"{data[key]}{UNITS.get(key, '')}")

        lines = [f"{p['name']:<28} {p['id']:<10} {p['frames']:>5} frames  ({p['share']:.0f}%)"
                 for p in data["people"][:10]]
        if data["unknown"]:
            lines.append(f"{config.UNKNOWN_LABEL:<28} {'':<10} {data['unknown']:>5} faces")
        if not lines:
            lines = ["(recognition off, or nothing identified yet)"]

        self.identity_text.configure(state="normal")
        self.identity_text.delete("1.0", "end")
        self.identity_text.insert("1.0", "\n".join(lines))
        self.identity_text.configure(state="disabled")

    # --------------------------------------------------------------- actions
    def reset_stats(self) -> None:
        self.session.stats.reset()
        self.status.set("Statistics reset.")

    def snapshot(self) -> None:
        path = self.session.save_snapshot()
        if path is None:
            messagebox.showinfo("Snapshot", "Nothing to save yet, start a source first.")
            return
        self.status.set(f"Snapshot saved to {path}")

    def export_stats(self) -> None:
        data = self.session.stats.snapshot()
        if not data["frames"]:
            messagebox.showinfo("Export", "No statistics to export yet.")
            return
        config.ensure_dirs()
        path = filedialog.asksaveasfilename(
            title="Export statistics", defaultextension=".csv",
            initialdir=str(config.OUTPUT_DIR), initialfile="gui_session.csv",
            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        status = self.session.status()
        write_csv(Path(path), self.session.stats.export_rows(
            status["detector"], status["recognizer"], status["source"]))
        self.status.set(f"Statistics exported to {path}")

    def _on_close(self) -> None:
        self.session.stop()
        self.root.after(50, self.root.destroy)


def main(args=None) -> int:
    config.ensure_dirs()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    FaceTrackGUI(root, args)
    root.mainloop()
    return 0
