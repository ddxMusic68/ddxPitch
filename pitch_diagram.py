import queue
import threading
import tkinter as tk
from tkinter import ttk

import librosa
import matplotlib
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


class PitchDiagramApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pitch Diagram")

        self.sr = 22050
        self.block_size = 2048
        self.hop_length = 512
        self.history_seconds = 20.0
        self.max_points = int(self.history_seconds * self.sr / self.hop_length)

        self.min_note = "E2"
        self.max_note = "E4"
        self.min_hz = float(librosa.note_to_hz(self.min_note))
        self.max_hz = float(librosa.note_to_hz(self.max_note))

        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.running = False
        self.worker: threading.Thread | None = None

        self.times: list[float] = []
        self.pitches: list[float] = []
        self.t = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=8)

        self.start_btn = ttk.Button(controls, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=4)

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        ttk.Label(controls, text="Window (s):").pack(side="left", padx=(12, 4))
        self.window_var = tk.DoubleVar(value=self.history_seconds)
        self.window_scale = ttk.Scale(
            controls, from_=5.0, to=60.0, orient="horizontal", variable=self.window_var,
            command=self.on_window_change, length=180,
        )
        self.window_scale.pack(side="left", padx=4)
        self.window_label = ttk.Label(controls, text=f"{self.history_seconds:.0f}")
        self.window_label.pack(side="left", padx=4)

        note_options = []
        for midi in range(librosa.note_to_midi("C1"), librosa.note_to_midi("C7") + 1):
            note_options.append(librosa.midi_to_note(midi, octave=True))
        ttk.Label(controls, text="Min:").pack(side="left", padx=(12, 2))
        self.min_note_var = tk.StringVar(value=self.min_note)
        self.min_note_cb = ttk.Combobox(controls, textvariable=self.min_note_var, values=note_options, width=5, state="readonly")
        self.min_note_cb.pack(side="left", padx=2)
        self.min_note_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_note_range())
        ttk.Label(controls, text="Max:").pack(side="left", padx=(6, 2))
        self.max_note_var = tk.StringVar(value=self.max_note)
        self.max_note_cb = ttk.Combobox(controls, textvariable=self.max_note_var, values=note_options, width=5, state="readonly")
        self.max_note_cb.pack(side="left", padx=2)
        self.max_note_cb.bind("<<ComboboxSelected>>", lambda e: self.apply_note_range())

        self.status = ttk.Label(controls, text="Idle")
        self.status.pack(side="left", padx=12)

        fig = Figure(figsize=(9, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Pitch over time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Note")
        self.ax.grid(True, alpha=0.3)
        self.note_colors = {
            "C": "red",
            "D": "orange",
            "E": "yellow",
            "F": "green",
            "G": "blue",
            "A": "indigo",
            "B": "violet",
        }
        self.scatter = self.ax.scatter([], [], s=20)

        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)
        self._update_note_axis()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            pass
        mono = indata[:, 0].copy()
        self.audio_q.put(mono)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="Listening...")
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sr,
                blocksize=self.block_size,
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self.running = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status.config(text=f"Mic error: {e}")
            return
        self.worker = threading.Thread(target=self.process_audio, daemon=True)
        self.worker.start()
        self.root.after(100, self.update_plot)

    def stop(self) -> None:
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status.config(text="Stopped")

    def _update_note_axis(self) -> None:
        self.ax.set_ylim(self.min_hz, self.max_hz)
        min_midi = librosa.note_to_midi(self.min_note)
        max_midi = librosa.note_to_midi(self.max_note)
        if max_midi - min_midi <= 24:
            midis = range(min_midi, max_midi + 1)
        else:
            midis = range(min_midi, max_midi + 1, 2)
        ticks = [float(librosa.midi_to_hz(m)) for m in midis]
        labels = [librosa.midi_to_note(m, octave=True) for m in midis]
        self.ax.set_yticks(ticks)
        self.ax.set_yticklabels(labels)
        self.canvas.draw_idle()

    def apply_note_range(self) -> None:
        try:
            new_min = self.min_note_var.get()
            new_max = self.max_note_var.get()
            min_midi = librosa.note_to_midi(new_min)
            max_midi = librosa.note_to_midi(new_max)
            if min_midi >= max_midi:
                raise ValueError("min must be below max")
            self.min_note = new_min
            self.max_note = new_max
            self.min_hz = float(librosa.note_to_hz(self.min_note))
            self.max_hz = float(librosa.note_to_hz(self.max_note))
            self._update_note_axis()
            self.status.config(text=f"Range: {self.min_note}-{self.max_note}")
        except Exception as e:
            self.status.config(text=f"Range error: {e}")
            self.min_note_var.set(self.min_note)
            self.max_note_var.set(self.max_note)

    def on_window_change(self, _value: str = "") -> None:
        self.history_seconds = float(self.window_var.get())
        self.max_points = int(self.history_seconds * self.sr / self.hop_length)
        self.window_label.config(text=f"{self.history_seconds:.0f}")
        if len(self.times) > self.max_points:
            excess = len(self.times) - self.max_points
            del self.times[:excess]
            del self.pitches[:excess]

    def process_audio(self) -> None:
        while self.running:
            try:
                block = self.audio_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.running:
                break
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    block.astype(np.float32),
                    fmin=self.min_hz,
                    fmax=self.max_hz,
                    sr=self.sr,
                    frame_length=self.block_size,
                    hop_length=self.hop_length,
                )
            except Exception:
                continue
            for i, hz in enumerate(f0):
                dt = self.hop_length / self.sr
                self.t += dt
                val = float(hz) if np.isfinite(hz) and (voiced_flag is None or voiced_flag[i]) else np.nan
                self.times.append(self.t)
                self.pitches.append(val)
            if len(self.times) > self.max_points:
                excess = len(self.times) - self.max_points
                del self.times[:excess]
                del self.pitches[:excess]

    def update_plot(self) -> None:
        if self.times:
            x = np.array(self.times)
            y = np.array(self.pitches, dtype=float)
            mask = np.isfinite(y)
            if np.any(mask):
                xv = x[mask]
                yv = y[mask]
                offsets = np.column_stack((xv, yv))
                colors = []
                for hz in yv:
                    try:
                        midi = int(round(librosa.hz_to_midi(hz)))
                        name = librosa.midi_to_note(midi, octave=False)
                        colors.append(self.note_colors.get(name[0], "gray"))
                    except Exception:
                        colors.append("gray")
                self.scatter.set_offsets(offsets)
                self.scatter.set_facecolors(colors)
            else:
                self.scatter.set_offsets(np.empty((0, 2)))
            xmin = max(0.0, x[-1] - self.history_seconds)
            xmax = max(self.history_seconds, x[-1])
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(self.min_hz, self.max_hz)
            self.canvas.draw_idle()
        if self.running:
            self.root.after(100, self.update_plot)

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PitchDiagramApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
