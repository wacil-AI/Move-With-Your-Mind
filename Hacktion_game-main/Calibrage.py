import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
from pylsl import StreamInlet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lsl_connect import (  # noqa: E402
    TimestampRepair,
    build_stream_metadata,
    estimate_sample_rate,
    lsl_clock,
    resolve_stream,
)
from Classe_Calibrage import CalibrageBCI  # noqa: E402


class CalibrationEEGRecorder:
    def __init__(
        self,
        stream_name: Optional[str],
        stream_type: str,
        source_id: Optional[str],
        stream_timeout: float,
        output_dir: Path,
        fixed_fs_hz: Optional[float] = None,
        target_channels: int = 7,
    ) -> None:
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.source_id = source_id
        self.stream_timeout = float(stream_timeout)
        self.output_dir = Path(output_dir)
        self.fixed_fs_hz = fixed_fs_hz
        self.target_channels = int(max(1, target_channels))

        self.inlet: Optional[StreamInlet] = None
        self.metadata = None
        self.ts_repair: Optional[TimestampRepair] = None
        self.started = False
        self.recording_start_lsl: Optional[float] = None
        self.recording_start_wall: Optional[float] = None
        self.first_sample_wall: Optional[float] = None
        self.total_chunks_seen = 0
        self.total_samples_seen = 0
        self.dropped_prestart_samples = 0

        self.channel_count = self.target_channels
        self.channel_labels = [f"Ch{i + 1}" for i in range(self.channel_count)]
        self.samples: list[np.ndarray] = []
        self.timestamps: list[np.ndarray] = []
        # (marker_lsl_ts, marker_wall_ts, normalized_label)
        self.events: list[tuple[float, float, str]] = []

    def start(self) -> bool:
        try:
            stream = resolve_stream(
                name=self.stream_name,
                stream_type=self.stream_type,
                source_id=self.source_id,
                timeout_seconds=self.stream_timeout,
            )
        except Exception as exc:
            print(f"[CALIB] EEG stream not found. Calibration will run without EDF recording. ({exc})")
            return False

        self.inlet = StreamInlet(stream, max_buflen=30, max_chunklen=64)
        self.metadata = build_stream_metadata(self.inlet.info())
        self.ts_repair = TimestampRepair(float(self.metadata.nominal_srate))
        self.channel_count = min(max(1, int(self.metadata.channel_count)), self.target_channels)
        self.channel_labels = [f"Ch{i + 1}" for i in range(self.channel_count)]
        self.samples = []
        self.timestamps = []
        self.events = []
        self.recording_start_lsl = None
        self.recording_start_wall = time.perf_counter()
        self.first_sample_wall = None
        self.total_chunks_seen = 0
        self.total_samples_seen = 0
        self.dropped_prestart_samples = 0
        self.started = True
        print(
            "[CALIB] Recording EEG from stream "
            f"name='{self.metadata.name}' source_id='{self.metadata.source_id}' "
            f"channels={self.metadata.channel_count} nominal_fs={self.metadata.nominal_srate:.3f}"
        )
        return True

    def poll(self) -> None:
        if not self.started or self.inlet is None or self.ts_repair is None:
            return

        chunk, timestamps = self.inlet.pull_chunk(timeout=0.01, max_samples=256)
        if not chunk:
            return

        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.total_chunks_seen += 1
        self.total_samples_seen += int(arr.shape[0])
        if arr.shape[1] >= self.channel_count:
            arr = arr[:, : self.channel_count]
        else:
            pad = np.zeros((arr.shape[0], self.channel_count - arr.shape[1]), dtype=np.float64)
            arr = np.hstack([arr, pad])

        raw_ts = (
            np.asarray(timestamps, dtype=np.float64)
            if timestamps
            else np.empty((0,), dtype=np.float64)
        )
        fixed_ts = self.ts_repair.repair(raw_ts, arr.shape[0])
        if self.recording_start_lsl is None and fixed_ts.size:
            # Define t_start from the first real sample timestamp received.
            self.recording_start_lsl = float(fixed_ts[0])
            self.first_sample_wall = time.perf_counter()
        if self.recording_start_lsl is not None and fixed_ts.size:
            keep = fixed_ts >= float(self.recording_start_lsl)
            if not np.any(keep):
                self.dropped_prestart_samples += int(arr.shape[0])
                return
            arr = arr[keep]
            fixed_ts = fixed_ts[keep]
            if arr.size == 0:
                return
        self.samples.append(arr)
        self.timestamps.append(fixed_ts)

    def add_marker(self, label: str, timestamp: float) -> None:
        normalized = self._normalize_label(label)
        if normalized is None:
            return
        marker_lsl_ts = float(timestamp)
        marker_wall_ts = float(time.perf_counter())
        # Avoid duplicate labels emitted back-to-back (e.g. start_gauche + gauche).
        if self.events:
            _, prev_wall_ts, prev_label = self.events[-1]
            if prev_label == normalized and abs(marker_wall_ts - float(prev_wall_ts)) <= 0.05:
                return
        self.events.append((marker_lsl_ts, marker_wall_ts, normalized))

    def stop(self) -> Optional[Path]:
        if not self.started:
            return None

        for _ in range(8):
            self.poll()

        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            except Exception:
                pass

        self.started = False

        if not self.samples or not self.timestamps:
            print(
                "[CALIB] No EEG samples captured, EDF not created. "
                f"(chunks_seen={self.total_chunks_seen}, samples_seen={self.total_samples_seen}, "
                f"dropped_prestart={self.dropped_prestart_samples})"
            )
            return None

        if self.recording_start_lsl is None:
            self.recording_start_lsl = float(np.concatenate(self.timestamps)[0])

        samples = np.vstack(self.samples)
        timestamps = np.concatenate(self.timestamps)
        fs_hz = self._resolve_fs_hz(timestamps)
        data, end_rel = self._resample(samples, timestamps, fs_hz, float(self.recording_start_lsl))

        out_path = self.output_dir / f"calibrage_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.edf"
        try:
            self._write_edf(
                out_path,
                data,
                fs_hz,
                float(self.recording_start_lsl),
                float(end_rel),
            )
        except Exception as exc:
            print(f"[CALIB] Failed to write EDF: {exc}")
            return None
        duration = float(data.shape[1]) / float(fs_hz)
        print(
            f"[CALIB] EDF saved: {out_path} "
            f"(channels={data.shape[0]} samples={data.shape[1]} fs={fs_hz}Hz duration={duration:.2f}s)"
        )
        return out_path

    def _resolve_fs_hz(self, timestamps: np.ndarray) -> int:
        if self.fixed_fs_hz is not None and self.fixed_fs_hz > 0:
            return int(round(float(self.fixed_fs_hz)))
        nominal = 0.0 if self.metadata is None else float(self.metadata.nominal_srate)
        est = float(estimate_sample_rate(timestamps, nominal))
        if est <= 0:
            est = nominal if nominal > 0 else 250.0
        return max(1, int(round(est)))

    def _resample(
        self,
        samples: np.ndarray,
        timestamps: np.ndarray,
        fs_hz: int,
        start_lsl: float,
    ) -> tuple[np.ndarray, float]:
        ts = np.asarray(timestamps, dtype=np.float64)
        x = np.asarray(samples, dtype=np.float64)
        if ts.size == 0:
            return x.T, 0.0

        rel_ts = np.asarray(ts - float(start_lsl), dtype=np.float64)
        rel_ts = np.maximum(rel_ts, 0.0)
        if rel_ts.size < 2:
            return x.T, float(rel_ts[-1] if rel_ts.size else 0.0)

        rel_end = float(max(0.0, rel_ts[-1]))
        if rel_end <= 0:
            return x.T, 0.0

        # Build EDF samples on timeline starting at recording start (t=0).
        n_target = max(2, int(round(rel_end * float(fs_hz))) + 1)
        t_target = np.arange(n_target, dtype=np.float64) / float(fs_hz)
        out = np.zeros((x.shape[1], n_target), dtype=np.float64)
        for ch in range(x.shape[1]):
            y = x[:, ch]
            out[ch, :] = np.interp(t_target, rel_ts, y, left=y[0], right=y[-1])
        return out, float(t_target[-1])

    def _write_edf(
        self,
        path: Path,
        data: np.ndarray,
        fs_hz: int,
        start_lsl: float,
        end_rel: float,
    ) -> None:
        import pyedflib

        path.parent.mkdir(parents=True, exist_ok=True)
        peak = float(np.nanmax(np.abs(data))) if data.size else 0.0
        if not np.isfinite(peak) or peak <= 0:
            peak = 200.0
        phys = max(200.0, min(99999.0, peak * 1.25))

        headers = []
        for ch in range(data.shape[0]):
            headers.append(
                {
                    "label": self.channel_labels[ch] if ch < len(self.channel_labels) else f"Ch{ch + 1}",
                    "dimension": "uV",
                    "sample_frequency": int(fs_hz),
                    "physical_min": -float(phys),
                    "physical_max": float(phys),
                    "digital_min": -32768,
                    "digital_max": 32767,
                    "transducer": "",
                    "prefilter": "",
                }
            )

        writer = pyedflib.EdfWriter(
            str(path),
            n_channels=int(data.shape[0]),
            file_type=pyedflib.FILETYPE_EDFPLUS,
        )
        try:
            writer.setSignalHeaders(headers)
            writer.writeSamples(np.ascontiguousarray(data))
            writer.writeAnnotation(0.0, 0.0, "record_start")
            end_rel = max(0.0, float(end_rel))
            writer.writeAnnotation(end_rel, 0.0, "record_stop")
            written = 0
            dropped = 0
            lsl_used = 0
            wall_used = 0
            rel_tol = 1.0 / max(1.0, float(fs_hz))

            for marker_lsl_ts, marker_wall_ts, label in self.events:
                rel_lsl = float(marker_lsl_ts) - float(start_lsl)
                rel_wall = None
                if self.first_sample_wall is not None:
                    rel_wall = float(marker_wall_ts) - float(self.first_sample_wall)

                rel: Optional[float] = None
                source = "none"

                if rel_wall is not None:
                    if -1.0 <= rel_wall <= (end_rel + 1.0):
                        rel = rel_wall
                        source = "wall"

                if rel is None:
                    if -1.0 <= rel_lsl <= (end_rel + 1.0):
                        rel = rel_lsl
                        source = "lsl"

                if rel is None:
                    dropped += 1
                    continue

                rel = float(np.clip(rel, 0.0, end_rel + rel_tol))
                writer.writeAnnotation(rel, 0.0, str(label))
                written += 1
                if source == "wall":
                    wall_used += 1
                elif source == "lsl":
                    lsl_used += 1

            writer.update_header()
            print(
                "[CALIB] Marker annotations: "
                f"captured={len(self.events)} written={written} dropped={dropped} "
                f"(wall={wall_used}, lsl={lsl_used})"
            )
        finally:
            writer.close()

    @staticmethod
    def _normalize_label(label: str) -> Optional[str]:
        text = str(label).strip().lower()
        if text in {"gauche", "left", "left_hand", "start_gauche", "start_left"}:
            return "gauche"
        if text in {"droite", "right", "right_hand", "start_droite", "start_right"}:
            return "droite"
        return None


def train_model_from_edf(edf_path: Path) -> Optional[Path]:
    script = PROJECT_ROOT / "Hacktion_game-main" / "Model_simple" / "csp_lda.py"
    if not script.exists():
        print(f"[CALIB] Training script missing: {script}")
        return None

    cmd = [sys.executable, str(script), "--edf", str(edf_path)]
    print(f"[CALIB] Training model from calibration EDF: {' '.join(cmd)}")
    # Stream training logs directly so CV/train scores are visible live.
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        print("[CALIB] Model training failed.")
        return None

    expected = script.parent / f"model_{edf_path.stem}.pkl"
    if expected.exists():
        print(f"[CALIB] New model saved: {expected}")
        return expected

    print("[CALIB] Training finished but model file was not found.")
    return None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def affichage_calibrage(screen, clock, largeur, hauteur):
    stream_name = _env_str("BCI_STREAM_NAME", "") or None
    source_id = _env_str("BCI_SOURCE_ID", "") or None
    stream_type = _env_str("BCI_STREAM_TYPE", "EEG") or "EEG"
    stream_timeout = _env_float("BCI_STREAM_TIMEOUT", 15.0)
    fixed_fs = _env_float("BCI_FIXED_FS_HZ", 0.0)
    fixed_fs = None if fixed_fs <= 0 else fixed_fs
    trials_per_class = max(1, _env_int("BCI_CALIB_TRIALS_PER_CLASS", 30))

    recorder = CalibrationEEGRecorder(
        stream_name=stream_name,
        stream_type=stream_type,
        source_id=source_id,
        stream_timeout=stream_timeout,
        output_dir=PROJECT_ROOT,
        fixed_fs_hz=fixed_fs,
        target_channels=7,
    )
    recorder.start()

    print(
        "[CALIB] Protocol: "
        f"{trials_per_class} gauche + {trials_per_class} droite "
        f"(total={trials_per_class * 2} trials)"
    )
    calibrage = CalibrageBCI(
        largeur,
        hauteur,
        marker_callback=recorder.add_marker,
        trials_per_class=trials_per_class,
    )
    finished = False

    try:
        while True:
            recorder.poll()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("quit",)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "go_to_menu"

            result = calibrage.update()
            if result == "finish":
                finished = True
                break

            calibrage.draw(screen)
            pygame.display.flip()
            clock.tick(60)
    finally:
        edf_path = recorder.stop()

    model_path = None
    if finished and edf_path is not None:
        print("[CALIB] Calibration complete. Retraining model with this EDF...")
        model_path = train_model_from_edf(edf_path)

    if edf_path is not None:
        return (
            "go_to_menu",
            {
                "edf_path": str(edf_path),
                "model_path": str(model_path) if model_path is not None else None,
            },
        )
    return "go_to_menu"
