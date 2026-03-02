#!/usr/bin/env python3
"""Online LSL EEG window generator for model-ready sliding predictions.

Generates fixed-size windows as (channels, samples) with configurable
window length, overlap/stride, optional common-average reference (CAR),
and optional FFT bandpass filtering.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import signal
import socket
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np
from pylsl import StreamInlet

from lsl_connect import (
    TimestampRepair,
    build_stream_metadata,
    estimate_sample_rate,
    list_lsl_streams,
    lsl_clock,
    resolve_stream,
)


LOGGER = logging.getLogger("online_windowing")


DEFAULT_LABEL_TO_UDP = {
    "left_hand": "-1",
    "right_hand": "1",
    "left": "-1",
    "right": "1",
    "idle": "0",
}

DEFAULT_CSP_CHANNEL_ORDER = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]


class LivePredictionPlotter:
    """Optional live plot for prediction confidence and assigned class."""

    def __init__(self, history_seconds: float, refresh_hz: float) -> None:
        self.history_seconds = max(1.0, float(history_seconds))
        self.refresh_interval = 1.0 / max(1e-6, float(refresh_hz))
        self.enabled = False

        self._start = time.monotonic()
        self._last_draw = 0.0
        self._times: Deque[float] = deque()
        self._scores: Deque[float] = deque()
        self._class_ids: Deque[int] = deque()
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}

        try:
            import matplotlib.pyplot as plt

            self._plt = plt
            self._plt.ion()
            self._fig, (self._ax_score, self._ax_class) = self._plt.subplots(
                2,
                1,
                figsize=(10, 6),
                sharex=True,
            )
            self._score_line, = self._ax_score.plot([], [], color="tab:blue", linewidth=1.5, label="confidence")
            self._class_line, = self._ax_class.plot([], [], color="tab:orange", linewidth=1.2, label="class_id")

            self._ax_score.set_ylabel("Confidence")
            self._ax_score.set_ylim(-0.02, 1.02)
            self._ax_score.grid(alpha=0.3)
            self._ax_score.legend(loc="upper right")

            self._ax_class.set_xlabel("Time (s)")
            self._ax_class.set_ylabel("Class")
            self._ax_class.grid(alpha=0.3)
            self._ax_class.legend(loc="upper right")
            self._ax_class.set_yticks([])

            self._fig.suptitle("Live Prediction")
            self._fig.tight_layout()
            self.enabled = True
        except Exception as exc:
            LOGGER.warning("Live plot disabled: %s", exc)

    def update(self, label: str, confidence: Optional[float]) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        t = now - self._start

        if label not in self._label_to_id:
            idx = len(self._label_to_id)
            self._label_to_id[label] = idx
            self._id_to_label[idx] = label

        cls_id = self._label_to_id[label]
        score = float(confidence) if confidence is not None else float("nan")

        self._times.append(t)
        self._scores.append(score)
        self._class_ids.append(cls_id)

        cutoff = t - self.history_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._scores.popleft()
            self._class_ids.popleft()

        if (now - self._last_draw) < self.refresh_interval:
            return
        self._last_draw = now

        t_data = np.asarray(self._times, dtype=np.float64)
        s_data = np.asarray(self._scores, dtype=np.float64)
        c_data = np.asarray(self._class_ids, dtype=np.float64)

        self._score_line.set_data(t_data, s_data)
        self._class_line.set_data(t_data, c_data)

        x_min = max(0.0, t - self.history_seconds)
        x_max = max(self.history_seconds, t + 0.1)
        self._ax_score.set_xlim(x_min, x_max)

        if self._label_to_id:
            ids = sorted(self._id_to_label.keys())
            labels = [self._id_to_label[i] for i in ids]
            self._ax_class.set_yticks(ids)
            self._ax_class.set_yticklabels(labels)
            y_min = min(ids) - 0.5
            y_max = max(ids) + 0.5
            self._ax_class.set_ylim(y_min, y_max)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._plt.close(self._fig)
        except Exception:
            pass


class CSPLDA:
    """Compatibility wrapper for pickled CSPLDA objects.

    Some serialized models were pickled with class path `__main__.CSPLDA`.
    This class allows unpickling and provides predict/predict_proba methods.
    """

    @staticmethod
    def load(path: str) -> "CSPLDA":
        install_legacy_pickle_shims()
        with open(path, "rb") as f:
            return _CSPLDAUnpickler(f).load()

    def _label_encoder(self) -> Optional[Any]:
        le = getattr(self, "le_", None)
        if le is not None:
            return le
        return getattr(self, "_label_encoder", None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        pipe = getattr(self, "_pipe", None)
        if pipe is None:
            raise RuntimeError("CSPLDA model has no '_pipe' attribute")
        pred = np.asarray(pipe.predict(x))
        le = self._label_encoder()
        if le is not None:
            try:
                pred = np.asarray(le.inverse_transform(pred))
            except Exception:
                pass
        return pred

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pipe = getattr(self, "_pipe", None)
        if pipe is None or not hasattr(pipe, "predict_proba"):
            raise RuntimeError("CSPLDA model/pipeline does not expose predict_proba")
        return np.asarray(pipe.predict_proba(x), dtype=np.float64)


class _CSPLDAUnpickler(pickle.Unpickler):
    """Map legacy `__main__.CSPLDA` references to local compatibility class."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "CSPLDA":
            return CSPLDA
        return super().find_class(module, name)


def _make_legacy_stub(module_name: str, func_name: str) -> Any:
    def _stub(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"Legacy helper '{module_name}.{func_name}' is not available in this MNE version"
        )

    _stub.__name__ = func_name
    return _stub


def install_legacy_pickle_shims() -> None:
    """Install minimal legacy module shims only when the real module is unavailable."""
    shims = (
        ("mne.decoding._covs_ged", ("_csp_estimate", "_spoc_estimate")),
        ("mne.decoding._mod_ged", ("_csp_mod", "_spoc_mod")),
    )
    for module_name, func_names in shims:
        try:
            # Prefer real MNE modules when present (do not override them).
            importlib.import_module(module_name)
            continue
        except Exception:
            pass

        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        for func_name in func_names:
            if not hasattr(module, func_name):
                setattr(module, func_name, _make_legacy_stub(module_name, func_name))


def load_prediction_model(path: str) -> Any:
    install_legacy_pickle_shims()
    with open(path, "rb") as f:
        model = _CSPLDAUnpickler(f).load()
    if not hasattr(model, "predict"):
        if hasattr(model, "_pipe"):
            # Fall back to compatibility wrapper with loaded state.
            compat = CSPLDA()
            compat.__dict__.update(getattr(model, "__dict__", {}))
            model = compat
        else:
            raise ValueError(f"Loaded object from '{path}' has no predict()")
    return model


def infer_model_channel_count(model: Any) -> Optional[int]:
    pipe = getattr(model, "_pipe", None)
    if pipe is None:
        return None
    try:
        named = getattr(pipe, "named_steps", {})
        csp = named.get("csp")
        if csp is not None and hasattr(csp, "filters_"):
            filt = np.asarray(getattr(csp, "filters_"))
            if filt.ndim == 2 and filt.shape[1] > 0:
                return int(filt.shape[1])
    except Exception:
        return None
    return None


def parse_channel_order(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    parts = [str(x).strip() for x in str(raw).split(",")]
    out = [x for x in parts if x]
    return out or None


def _normalize_channel_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _channel_aliases(name: str) -> list[str]:
    base = _normalize_channel_name(name)
    if not base:
        return []
    aliases = [base]
    for prefix in ("EEG", "CHAN", "CHANNEL"):
        if base.startswith(prefix) and len(base) > len(prefix):
            aliases.append(base[len(prefix) :])
    # Common suffixes from acquisition software.
    for suffix in ("REF", "LE", "RE"):
        current = list(aliases)
        for item in current:
            if item.endswith(suffix) and len(item) > len(suffix):
                aliases.append(item[: -len(suffix)])
    # Deduplicate while preserving order.
    return list(dict.fromkeys([a for a in aliases if a]))


def infer_model_runtime_config(model: Any, model_channels: Optional[int]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}

    raw_cfg = getattr(model, "training_config_", None)
    if isinstance(raw_cfg, dict):
        for key in (
            "window_seconds",
            "tmin_seconds",
            "tmax_seconds",
            "bandpass_low_hz",
            "bandpass_high_hz",
            "fixed_fs_hz",
            "apply_car",
            "channel_names",
            "channel_order",
        ):
            if key in raw_cfg:
                cfg[key] = raw_cfg[key]

    # Backward-compatible attribute fallbacks.
    if "window_seconds" not in cfg and hasattr(model, "training_window_seconds_"):
        cfg["window_seconds"] = getattr(model, "training_window_seconds_")
    if "fixed_fs_hz" not in cfg and hasattr(model, "training_fixed_fs_hz_"):
        cfg["fixed_fs_hz"] = getattr(model, "training_fixed_fs_hz_")
    if "apply_car" not in cfg and hasattr(model, "training_apply_car_"):
        cfg["apply_car"] = getattr(model, "training_apply_car_")
    if "channel_order" not in cfg:
        for attr in ("expected_channel_names_", "channel_names_", "ch_names_"):
            raw = getattr(model, attr, None)
            if isinstance(raw, (list, tuple)) and raw:
                cfg["channel_order"] = [str(x) for x in raw if str(x).strip()]
                break
    if ("bandpass_low_hz" not in cfg or "bandpass_high_hz" not in cfg) and hasattr(model, "training_bandpass_hz_"):
        bp = getattr(model, "training_bandpass_hz_")
        if isinstance(bp, (list, tuple)) and len(bp) == 2:
            cfg.setdefault("bandpass_low_hz", bp[0])
            cfg.setdefault("bandpass_high_hz", bp[1])
    if "train_trial_ptp_stats" not in cfg and hasattr(model, "training_trial_ptp_stats_"):
        raw_ptp = getattr(model, "training_trial_ptp_stats_")
        if isinstance(raw_ptp, dict):
            cfg["train_trial_ptp_stats"] = dict(raw_ptp)
    if "cv_stats" not in cfg and hasattr(model, "training_cv_stats_"):
        raw_cv = getattr(model, "training_cv_stats_")
        if isinstance(raw_cv, dict):
            cfg["cv_stats"] = dict(raw_cv)

    # Heuristic fallback for legacy CSPLDA models trained via csp_lda.py.
    model_type = type(model).__name__.lower()
    if "csplda" in model_type:
        cfg.setdefault("window_seconds", 3.0)
        cfg.setdefault("bandpass_low_hz", 8.0)
        cfg.setdefault("bandpass_high_hz", 30.0)
        cfg.setdefault("apply_car", False)
        if model_channels == 7:
            cfg.setdefault("channel_order", list(DEFAULT_CSP_CHANNEL_ORDER))

    # Normalize types.
    def _as_float(v: Any) -> Optional[float]:
        try:
            out = float(v)
            return out if np.isfinite(out) else None
        except Exception:
            return None

    out: dict[str, Any] = {}
    for float_key in ("window_seconds", "tmin_seconds", "tmax_seconds", "bandpass_low_hz", "bandpass_high_hz", "fixed_fs_hz"):
        if float_key in cfg:
            val = _as_float(cfg.get(float_key))
            if val is not None:
                out[float_key] = val
    if "apply_car" in cfg:
        out["apply_car"] = bool(cfg.get("apply_car"))

    chan = cfg.get("channel_order", cfg.get("channel_names"))
    if isinstance(chan, (list, tuple)):
        names = [str(x).strip() for x in chan if str(x).strip()]
        if names:
            out["channel_order"] = names
    ptp_stats = cfg.get("train_trial_ptp_stats")
    if isinstance(ptp_stats, dict):
        norm_ptp: dict[str, float] = {}
        for k in ("median", "p95", "max", "min"):
            if k in ptp_stats:
                v = _as_float(ptp_stats.get(k))
                if v is not None:
                    norm_ptp[k] = v
        if norm_ptp:
            out["train_trial_ptp_stats"] = norm_ptp
    cv_stats = cfg.get("cv_stats")
    if isinstance(cv_stats, dict):
        norm_cv: dict[str, Any] = {}
        for k in ("mean_acc", "std_acc"):
            if k in cv_stats:
                v = _as_float(cv_stats.get(k))
                if v is not None:
                    norm_cv[k] = v
        if "n_folds" in cv_stats:
            try:
                norm_cv["n_folds"] = int(cv_stats.get("n_folds"))
            except Exception:
                pass
        if norm_cv:
            out["cv_stats"] = norm_cv
    return out


def build_channel_selector(source_names: list[str], expected_order: list[str]) -> tuple[Optional[list[int]], list[tuple[str, str, int]], list[str]]:
    if not source_names or not expected_order:
        return None, [], list(expected_order)

    src_aliases = [_channel_aliases(x) for x in source_names]
    used: set[int] = set()
    selector: list[int] = []
    mapping: list[tuple[str, str, int]] = []
    missing: list[str] = []

    for target in expected_order:
        target_aliases = _channel_aliases(target)
        if not target_aliases:
            missing.append(str(target))
            continue

        pick: Optional[int] = None
        # Pass 1: exact alias match.
        for idx, aliases in enumerate(src_aliases):
            if idx in used:
                continue
            if any(t in aliases for t in target_aliases):
                pick = idx
                break
        # Pass 2: suffix match (e.g. EEGF3REF -> F3).
        if pick is None:
            for idx, aliases in enumerate(src_aliases):
                if idx in used:
                    continue
                ok = False
                for s in aliases:
                    for t in target_aliases:
                        if len(t) >= 2 and (s.endswith(t) or t.endswith(s)):
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    pick = idx
                    break

        if pick is None:
            missing.append(str(target))
            continue

        used.add(pick)
        selector.append(pick)
        mapping.append((str(target), str(source_names[pick]), int(pick)))

    if not selector:
        return None, [], list(expected_order)
    return selector, mapping, missing


def align_channels_for_model(x: np.ndarray, target_channels: Optional[int]) -> np.ndarray:
    if target_channels is None or target_channels <= 0:
        return x
    current = int(x.shape[0])
    if current == target_channels:
        return x
    if current > target_channels:
        return x[:target_channels, :]
    pad = np.zeros((target_channels - current, x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad])


class UdpCommandSender:
    """Send label-mapped commands to the game via UDP."""

    def __init__(self, host: str, port: int, label_to_payload: dict[str, str]) -> None:
        self.host = str(host)
        self.port = int(port)
        self.address = (self.host, self.port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.label_to_payload = {str(k): str(v) for k, v in label_to_payload.items()}
        self.sent_count = 0

    def send_label(self, label: str) -> bool:
        payload = self.label_to_payload.get(str(label))
        if payload is None:
            return False
        self._sock.sendto(payload.encode("utf-8"), self.address)
        self.sent_count += 1
        return True

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def parse_label_map(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return dict(DEFAULT_LABEL_TO_UDP)
    path = Path(str(raw))
    text = path.read_text(encoding="utf-8") if path.exists() else str(raw)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Label map must be a JSON object")
    merged = dict(DEFAULT_LABEL_TO_UDP)
    merged.update({str(k): str(v) for k, v in data.items()})
    return merged


@dataclass
class OnlineWindow:
    x: np.ndarray  # shape: (channels, samples)
    t_start_lsl: float
    t_end_lsl: float
    fs_hz: float
    raw_samples: int


@dataclass
class WindowConfig:
    window_seconds: float
    stride_seconds: float
    fixed_fs_hz: Optional[float] = None
    apply_car: bool = False
    bandpass_low_hz: Optional[float] = None
    bandpass_high_hz: Optional[float] = None
    history_seconds: float = 10.0


class OnlineEEGWindowGenerator:
    """Maintain an online buffer and emit sliding, overlapping windows."""

    def __init__(self, inlet: StreamInlet, config: WindowConfig, nominal_srate: float) -> None:
        self.inlet = inlet
        self.config = config
        self.nominal_srate = float(nominal_srate)
        self.ts_repair = TimestampRepair(self.nominal_srate)

        self._samples: Deque[np.ndarray] = deque()
        self._timestamps: Deque[float] = deque()
        self._channel_count: Optional[int] = None
        self._next_window_end_lsl: Optional[float] = None
        self.last_chunk_samples: int = 0
        self.total_samples_received: int = 0
        self.total_chunks_received: int = 0

        # This is the variable you can feed directly to a model.
        self.latest_model_input: Optional[np.ndarray] = None

    def poll(self, pull_timeout: float, max_chunk_samples: int) -> list[OnlineWindow]:
        self.last_chunk_samples = 0
        chunk, timestamps = self.inlet.pull_chunk(
            timeout=float(pull_timeout),
            max_samples=int(max_chunk_samples),
        )
        if chunk:
            samples = np.asarray(chunk, dtype=np.float64)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            self.last_chunk_samples = int(samples.shape[0])
            self.total_samples_received += self.last_chunk_samples
            self.total_chunks_received += 1
            raw_ts = (
                np.asarray(timestamps, dtype=np.float64)
                if timestamps
                else np.empty((0,), dtype=np.float64)
            )
            fixed_ts = self.ts_repair.repair(raw_ts, samples.shape[0])
            self._append_chunk(samples, fixed_ts)
        return self._emit_ready_windows()

    def _append_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        if samples.ndim != 2 or timestamps.ndim != 1:
            return
        if samples.shape[0] == 0 or samples.shape[0] != timestamps.shape[0]:
            return

        if self._channel_count is None:
            self._channel_count = int(samples.shape[1])
        elif int(samples.shape[1]) != self._channel_count:
            target = self._channel_count
            if samples.shape[1] > target:
                samples = samples[:, :target]
            else:
                pad = np.zeros((samples.shape[0], target - samples.shape[1]), dtype=samples.dtype)
                samples = np.hstack([samples, pad])

        for row, ts in zip(samples, timestamps):
            self._samples.append(np.asarray(row, dtype=np.float64))
            self._timestamps.append(float(ts))

        self._drop_old()

    def _drop_old(self) -> None:
        if not self._timestamps:
            return
        newest = self._timestamps[-1]
        history = max(
            float(self.config.history_seconds),
            float(self.config.window_seconds) * 2.0,
        )
        threshold = newest - history
        while self._timestamps and self._timestamps[0] < threshold:
            self._timestamps.popleft()
            self._samples.popleft()

    def _emit_ready_windows(self) -> list[OnlineWindow]:
        if len(self._timestamps) < 2:
            return []

        if self._next_window_end_lsl is None:
            self._next_window_end_lsl = float(self._timestamps[0]) + float(self.config.window_seconds)

        out: list[OnlineWindow] = []
        latest = float(self._timestamps[-1])
        while latest >= float(self._next_window_end_lsl):
            end_ts = float(self._next_window_end_lsl)
            start_ts = end_ts - float(self.config.window_seconds)
            window = self._build_window(start_ts, end_ts)
            if window is not None:
                out.append(window)
                self.latest_model_input = window.x
            self._next_window_end_lsl = end_ts + float(self.config.stride_seconds)

        return out

    def _build_window(self, start_ts: float, end_ts: float) -> Optional[OnlineWindow]:
        ts = np.asarray(self._timestamps, dtype=np.float64)
        if ts.size < 2:
            return None
        mask = np.logical_and(ts >= start_ts, ts <= end_ts)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            return None

        all_samples = np.vstack(self._samples)
        raw = all_samples[idx, :]  # (samples, channels)
        raw_ts = ts[idx]

        fs_hz = self._resolve_output_fs(raw_ts)
        n_target = max(2, int(round(float(self.config.window_seconds) * fs_hz)))
        x = self._resample_window(raw, raw_ts, start_ts, fs_hz, n_target)  # (channels, samples)

        if self.config.apply_car:
            # CAR per time sample across channels.
            x = x - np.mean(x, axis=0, keepdims=True)

        x = self._bandpass_fft(
            x,
            fs_hz=fs_hz,
            low_hz=self.config.bandpass_low_hz,
            high_hz=self.config.bandpass_high_hz,
        )

        return OnlineWindow(
            x=x,
            t_start_lsl=start_ts,
            t_end_lsl=end_ts,
            fs_hz=fs_hz,
            raw_samples=int(raw.shape[0]),
        )

    def _resolve_output_fs(self, ts: np.ndarray) -> float:
        if self.config.fixed_fs_hz is not None and self.config.fixed_fs_hz > 0:
            return float(self.config.fixed_fs_hz)
        if self.nominal_srate > 0:
            return float(self.nominal_srate)
        est = estimate_sample_rate(ts, self.nominal_srate)
        return float(est if est > 0 else 250.0)

    @staticmethod
    def _resample_window(
        raw_samples: np.ndarray,
        raw_ts: np.ndarray,
        start_ts: float,
        fs_hz: float,
        n_target: int,
    ) -> np.ndarray:
        # raw_samples: (samples, channels) -> output: (channels, samples)
        t_rel = np.asarray(raw_ts, dtype=np.float64) - float(start_ts)
        t_target = np.arange(n_target, dtype=np.float64) / float(fs_hz)

        channels = raw_samples.shape[1]
        out = np.zeros((channels, n_target), dtype=np.float64)
        for ch in range(channels):
            y = np.asarray(raw_samples[:, ch], dtype=np.float64)
            out[ch, :] = np.interp(t_target, t_rel, y, left=y[0], right=y[-1])
        return out

    @staticmethod
    def _bandpass_fft(
        x: np.ndarray,
        fs_hz: float,
        low_hz: Optional[float],
        high_hz: Optional[float],
    ) -> np.ndarray:
        if x.size == 0:
            return x
        nyq = 0.5 * float(fs_hz)
        low = 0.0 if low_hz is None else max(0.0, float(low_hz))
        high = nyq if high_hz is None else min(float(high_hz), nyq)
        if not (high > low and high > 0):
            return x

        freqs = np.fft.rfftfreq(x.shape[1], d=1.0 / float(fs_hz))
        mask = np.logical_and(freqs >= low, freqs <= high)
        spec = np.fft.rfft(x, axis=1)
        spec *= mask[None, :]
        return np.fft.irfft(spec, n=x.shape[1], axis=1)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_stride(window_seconds: float, stride_seconds: Optional[float], overlap_seconds: Optional[float]) -> float:
    # If stride is set, it takes precedence over overlap.
    # This avoids ambiguity with parser defaults (overlap may be set by default).
    if stride_seconds is not None:
        stride = float(stride_seconds)
        if stride <= 0:
            raise ValueError("Stride must be > 0")
        return stride

    # Backward-compatible default overlap when neither stride nor overlap is provided.
    overlap = 0.8 if overlap_seconds is None else float(overlap_seconds)
    stride_seconds = float(window_seconds) - overlap
    if stride_seconds <= 0:
        raise ValueError("Stride must be > 0 (reduce overlap or increase window)")
    return float(stride_seconds)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read LSL EEG and emit model-ready sliding windows (channels x samples)."
    )
    parser.add_argument("--list-streams", action="store_true", help="List LSL streams and exit")
    parser.add_argument("--stream-name", type=str, default=None, help="LSL stream name filter (e.g. 'FakeEEG')")
    parser.add_argument("--stream-type", type=str, default="EEG", help="LSL stream type filter (default: EEG)")
    parser.add_argument("--source-id", type=str, default=None, help="LSL source_id filter (most specific)")
    parser.add_argument("--stream-timeout", type=float, default=20.0, help="LSL stream discovery timeout (s)")

    parser.add_argument("--window-seconds", type=float, default=1.0, help="Window length in seconds")
    parser.add_argument("--stride-seconds", type=float, default=None, help="Slide step in seconds")
    parser.add_argument("--overlap-seconds", type=float, default=0.8, help="Overlap in seconds (if stride not set)")
    parser.add_argument("--fixed-fs-hz", type=float, default=None, help="Optional fixed model sampling rate")
    parser.add_argument("--history-seconds", type=float, default=10.0, help="Internal buffer history length")

    parser.add_argument("--car", action="store_true", help="Apply common average reference")
    parser.add_argument("--bandpass-low-hz", type=float, default=None, help="Bandpass low cutoff (Hz)")
    parser.add_argument("--bandpass-high-hz", type=float, default=None, help="Bandpass high cutoff (Hz)")

    parser.add_argument("--max-chunk-samples", type=int, default=64, help="LSL pull max samples")
    parser.add_argument("--pull-timeout", type=float, default=0.05, help="LSL pull timeout (s)")
    parser.add_argument(
        "--health-interval-seconds",
        type=float,
        default=2.0,
        help="How often to log stream health stats (0 to disable)",
    )
    parser.add_argument(
        "--no-samples-warn-seconds",
        type=float,
        default=3.0,
        help="Warn if no LSL samples are received for this duration",
    )
    parser.add_argument("--max-windows", type=int, default=0, help="Stop after N windows (0 = infinite)")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model pickle path for online prediction")
    parser.add_argument(
        "--no-model-config",
        action="store_true",
        help="Ignore training configuration saved in the loaded model",
    )
    parser.add_argument(
        "--model-channel-order",
        type=str,
        default=None,
        help="Comma-separated model channel order (e.g. F3,F4,C3,Cz,C4,P3,P4)",
    )
    parser.add_argument(
        "--strict-model-channel-order",
        action="store_true",
        help="Fail if any requested model-channel-order channel is missing in the LSL stream",
    )
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Optional confidence threshold [0,1]")
    parser.add_argument("--live-plot", action="store_true", help="Show live plot of confidence and predicted class")
    parser.add_argument("--plot-history-seconds", type=float, default=30.0, help="Live-plot history duration")
    parser.add_argument("--plot-refresh-hz", type=float, default=10.0, help="Live-plot refresh rate")
    parser.add_argument("--udp-host", type=str, default=None, help="Optional UDP target host (e.g. 127.0.0.1)")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP target port (game listens on 5005 by default)")
    parser.add_argument(
        "--udp-label-map",
        type=str,
        default=None,
        help="JSON object or JSON file mapping prediction labels to UDP payload strings",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)

    if args.list_streams:
        list_lsl_streams(wait_time=min(max(float(args.stream_timeout), 0.1), 10.0))
        return 0

    effective_window_seconds = float(args.window_seconds)
    effective_fixed_fs_hz = args.fixed_fs_hz
    effective_apply_car = bool(args.car)
    effective_bp_low = args.bandpass_low_hz
    effective_bp_high = args.bandpass_high_hz

    predictor: Optional[Any] = None
    model_channels: Optional[int] = None
    model_runtime_cfg: dict[str, Any] = {}
    model_channel_order: Optional[list[str]] = parse_channel_order(args.model_channel_order)
    warned_channel_mismatch = False
    if args.model_path:
        model_path = str(Path(args.model_path).expanduser().resolve())
        predictor = load_prediction_model(model_path)
        model_channels = infer_model_channel_count(predictor)
        if not args.no_model_config:
            model_runtime_cfg = infer_model_runtime_config(predictor, model_channels)
            if "window_seconds" in model_runtime_cfg:
                effective_window_seconds = float(model_runtime_cfg["window_seconds"])
            if "fixed_fs_hz" in model_runtime_cfg and (effective_fixed_fs_hz is None or float(effective_fixed_fs_hz) <= 0):
                effective_fixed_fs_hz = float(model_runtime_cfg["fixed_fs_hz"])
            if "bandpass_low_hz" in model_runtime_cfg and args.bandpass_low_hz is None:
                effective_bp_low = float(model_runtime_cfg["bandpass_low_hz"])
            if "bandpass_high_hz" in model_runtime_cfg and args.bandpass_high_hz is None:
                effective_bp_high = float(model_runtime_cfg["bandpass_high_hz"])
            if "apply_car" in model_runtime_cfg:
                effective_apply_car = bool(model_runtime_cfg["apply_car"])
            if model_channel_order is None and "channel_order" in model_runtime_cfg:
                model_channel_order = [str(x) for x in model_runtime_cfg["channel_order"]]
        LOGGER.info(
            "Loaded model: %s (type=%s, expected_channels=%s)",
            model_path,
            type(predictor).__name__,
            str(model_channels) if model_channels is not None else "unknown",
        )
        if model_runtime_cfg and not args.no_model_config:
            LOGGER.info(
                "Applied model config: window=%.3fs fixed_fs=%s CAR=%s bandpass=[%s,%s] channel_order=%s",
                effective_window_seconds,
                str(effective_fixed_fs_hz),
                str(effective_apply_car),
                str(effective_bp_low),
                str(effective_bp_high),
                str(model_channel_order),
            )
            cv_stats = model_runtime_cfg.get("cv_stats", {})
            if isinstance(cv_stats, dict):
                mean_acc = cv_stats.get("mean_acc")
                std_acc = cv_stats.get("std_acc")
                n_folds = cv_stats.get("n_folds")
                if isinstance(mean_acc, (float, int)):
                    LOGGER.info(
                        "Model CV stats: mean_acc=%.1f%% std=%.1f%% folds=%s",
                        float(mean_acc) * 100.0,
                        float(std_acc) * 100.0 if isinstance(std_acc, (float, int)) else float("nan"),
                        str(n_folds) if n_folds is not None else "?",
                    )
                    if float(mean_acc) < 0.60:
                        LOGGER.warning(
                            "Model CV mean accuracy is low (%.1f%%). Live predictions may collapse or be unstable.",
                            float(mean_acc) * 100.0,
                        )

    stride_seconds = resolve_stride(
        window_seconds=float(effective_window_seconds),
        stride_seconds=args.stride_seconds,
        overlap_seconds=args.overlap_seconds,
    )

    stream = resolve_stream(
        name=args.stream_name,
        stream_type=args.stream_type,
        source_id=args.source_id,
        timeout_seconds=float(args.stream_timeout),
    )
    inlet = StreamInlet(stream, max_buflen=10, max_chunklen=max(1, int(args.max_chunk_samples)))
    metadata = build_stream_metadata(inlet.info())
    LOGGER.info(
        "Connected stream: name='%s' type='%s' source_id='%s' channels=%d nominal_srate=%.3f",
        metadata.name,
        metadata.stream_type,
        metadata.source_id,
        metadata.channel_count,
        metadata.nominal_srate,
    )
    LOGGER.info("Stream channels: %s", metadata.channel_names)
    LOGGER.info(
        "Window config: window=%.3fs stride=%.3fs overlap=%.3fs output_shape=(channels,samples) CAR=%s bandpass=[%s,%s]",
        float(effective_window_seconds),
        stride_seconds,
        float(effective_window_seconds) - stride_seconds,
        bool(effective_apply_car),
        str(effective_bp_low),
        str(effective_bp_high),
    )

    channel_selector: Optional[list[int]] = None
    if predictor is not None and model_channel_order:
        selector, mapping, missing = build_channel_selector(metadata.channel_names, model_channel_order)
        if selector is not None:
            if missing and args.strict_model_channel_order:
                raise RuntimeError(
                    "Strict channel mapping failed. Missing expected channels: "
                    f"{missing}. stream_channels={metadata.channel_names} "
                    f"expected_order={model_channel_order}"
                )
            channel_selector = selector
            LOGGER.info("Mapped stream->model channels: %s", mapping)
            if missing:
                LOGGER.warning("Missing expected channels in stream metadata: %s", missing)
        else:
            if args.strict_model_channel_order:
                raise RuntimeError(
                    "Strict channel mapping failed. Could not map model channel order from stream metadata. "
                    f"stream_channels={metadata.channel_names} expected_order={model_channel_order}"
                )
            LOGGER.warning(
                "Could not map model channel order from stream metadata. Falling back to truncate/pad by index. "
                "stream_channels=%s expected_order=%s",
                metadata.channel_names,
                model_channel_order,
            )

    plotter: Optional[LivePredictionPlotter] = None
    if args.live_plot:
        if predictor is None:
            LOGGER.warning("Live plot requested but no model is loaded; plot will remain empty.")
        plotter = LivePredictionPlotter(
            history_seconds=float(args.plot_history_seconds),
            refresh_hz=float(args.plot_refresh_hz),
        )
        if plotter.enabled:
            LOGGER.info(
                "Live plot enabled (history=%.1fs, refresh=%.1fHz)",
                float(args.plot_history_seconds),
                float(args.plot_refresh_hz),
            )

    udp_sender: Optional[UdpCommandSender] = None
    if args.udp_host:
        label_map = parse_label_map(args.udp_label_map)
        udp_sender = UdpCommandSender(host=args.udp_host, port=int(args.udp_port), label_to_payload=label_map)
        LOGGER.info(
            "UDP sender enabled: %s:%d label_map=%s",
            args.udp_host,
            int(args.udp_port),
            label_map,
        )

    generator = OnlineEEGWindowGenerator(
        inlet=inlet,
        config=WindowConfig(
            window_seconds=float(effective_window_seconds),
            stride_seconds=float(stride_seconds),
            fixed_fs_hz=effective_fixed_fs_hz,
            apply_car=bool(effective_apply_car),
            bandpass_low_hz=effective_bp_low,
            bandpass_high_hz=effective_bp_high,
            history_seconds=float(args.history_seconds),
        ),
        nominal_srate=float(metadata.nominal_srate),
    )

    should_stop = False

    def stop_handler(_sig: int, _frame: object) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, stop_handler)
        except Exception:
            pass

    produced = 0
    started_mono = time.monotonic()
    first_sample_mono: Optional[float] = None
    last_sample_mono: float = started_mono
    last_report_mono: float = started_mono
    last_report_samples = 0
    last_report_windows = 0
    last_stale_warn_mono = 0.0
    last_pred_collapse_warn_mono = 0.0
    last_preproc_mismatch_warn_mono = 0.0
    recent_labels: Deque[str] = deque(maxlen=100)
    recent_ptp: Deque[float] = deque(maxlen=100)
    recent_std: Deque[float] = deque(maxlen=100)

    LOGGER.info("Starting online window loop. Press Ctrl+C to stop.")
    try:
        while not should_stop:
            windows = generator.poll(
                pull_timeout=float(args.pull_timeout),
                max_chunk_samples=int(args.max_chunk_samples),
            )
            now_mono = time.monotonic()
            pulled = int(generator.last_chunk_samples)
            if pulled > 0:
                last_sample_mono = now_mono
                if first_sample_mono is None:
                    first_sample_mono = now_mono
                    LOGGER.info(
                        "LSL data flow confirmed: first chunk received after %.3fs (%d samples)",
                        now_mono - started_mono,
                        pulled,
                    )

            health_interval = max(0.0, float(args.health_interval_seconds))
            if health_interval > 0 and (now_mono - last_report_mono) >= health_interval:
                dt = max(1e-6, now_mono - last_report_mono)
                sample_delta = int(generator.total_samples_received - last_report_samples)
                window_delta = int(produced - last_report_windows)
                samples_per_sec = sample_delta / dt
                windows_per_sec = window_delta / dt
                no_sample_for = max(0.0, now_mono - last_sample_mono)
                status = "LIVE" if no_sample_for < float(args.no_samples_warn_seconds) else "STALE"
                LOGGER.info(
                    "stream_health status=%s samples_total=%d chunks_total=%d sps=%.1f windows_total=%d wps=%.2f no_sample_for=%.2fs",
                    status,
                    int(generator.total_samples_received),
                    int(generator.total_chunks_received),
                    samples_per_sec,
                    produced,
                    windows_per_sec,
                    no_sample_for,
                )
                if predictor is not None and recent_labels:
                    counts: dict[str, int] = {}
                    for lbl in recent_labels:
                        counts[lbl] = counts.get(lbl, 0) + 1
                    dom_label, dom_count = max(counts.items(), key=lambda kv: kv[1])
                    dom_ratio = float(dom_count) / float(len(recent_labels))
                    LOGGER.info(
                        "pred_health recent=%d dominant=%s(%.1f%%) counts=%s",
                        len(recent_labels),
                        dom_label,
                        dom_ratio * 100.0,
                        counts,
                    )
                    if (
                        len(recent_labels) >= 40
                        and dom_ratio >= 0.95
                        and (now_mono - last_pred_collapse_warn_mono) >= max(2.0, health_interval)
                    ):
                        LOGGER.warning(
                            "Prediction collapse suspected (%.1f%% '%s' over last %d windows). "
                            "Check channel mapping/order and preprocessing consistency with training.",
                            dom_ratio * 100.0,
                            dom_label,
                            len(recent_labels),
                        )
                        last_pred_collapse_warn_mono = now_mono
                if predictor is not None and recent_ptp:
                    ptp_arr = np.asarray(recent_ptp, dtype=np.float64)
                    std_arr = np.asarray(recent_std, dtype=np.float64) if recent_std else np.asarray([], dtype=np.float64)
                    ptp_med = float(np.median(ptp_arr))
                    ptp_p95 = float(np.percentile(ptp_arr, 95.0))
                    std_med = float(np.median(std_arr)) if std_arr.size else float("nan")
                    LOGGER.info(
                        "preproc_health recent=%d ptp_med=%.3e ptp_p95=%.3e std_med=%.3e",
                        len(ptp_arr),
                        ptp_med,
                        ptp_p95,
                        std_med,
                    )
                    train_ptp_stats = model_runtime_cfg.get("train_trial_ptp_stats", {})
                    train_ptp_med = float(train_ptp_stats.get("median", np.nan)) if isinstance(train_ptp_stats, dict) else float("nan")
                    if np.isfinite(train_ptp_med) and train_ptp_med > 0 and np.isfinite(ptp_med) and ptp_med > 0:
                        ratio = ptp_med / train_ptp_med
                        if (
                            (ratio > 10.0 or ratio < 0.1)
                            and (now_mono - last_preproc_mismatch_warn_mono) >= max(2.0, health_interval)
                        ):
                            LOGGER.warning(
                                "Preproc amplitude mismatch: live_ptp_median=%.3e vs train_ptp_median=%.3e (ratio=%.2f). "
                                "Potential unit/gain mismatch or wrong channels.",
                                ptp_med,
                                train_ptp_med,
                                ratio,
                            )
                            last_preproc_mismatch_warn_mono = now_mono
                last_report_mono = now_mono
                last_report_samples = int(generator.total_samples_received)
                last_report_windows = produced

            no_samples_warn_seconds = max(0.0, float(args.no_samples_warn_seconds))
            if no_samples_warn_seconds > 0:
                silent_for = now_mono - last_sample_mono
                if (
                    silent_for >= no_samples_warn_seconds
                    and (now_mono - last_stale_warn_mono) >= no_samples_warn_seconds
                ):
                    LOGGER.warning(
                        "No LSL samples received for %.2fs (check stream/headset connection and stream filters)",
                        silent_for,
                    )
                    last_stale_warn_mono = now_mono

            if not windows:
                time.sleep(0.001)
                continue
            for w in windows:
                # `w.x` is model-ready as (channels, samples).
                pred_txt = ""
                sent_txt = ""
                model_shape_txt = ""
                if predictor is not None:
                    x_model = w.x
                    if channel_selector is not None:
                        x_model = x_model[np.asarray(channel_selector, dtype=np.int64), :]
                    channels_before_align = int(x_model.shape[0])
                    x_model = align_channels_for_model(x_model, model_channels)
                    model_shape_txt = f" model_shape={tuple(x_model.shape)}"
                    if x_model.size > 0:
                        ch_ptp = np.ptp(x_model, axis=1)
                        ch_std = np.std(x_model, axis=1)
                        recent_ptp.append(float(np.median(ch_ptp)))
                        recent_std.append(float(np.median(ch_std)))
                    if (
                        not warned_channel_mismatch
                        and model_channels is not None
                        and int(x_model.shape[0]) != channels_before_align
                    ):
                        LOGGER.warning(
                            "Channel mismatch: model input has %d channels but model expects %d. Auto-aligning.",
                            channels_before_align,
                            model_channels,
                        )
                        warned_channel_mismatch = True
                    y = predictor.predict(x_model[np.newaxis, :, :])
                    label = str(y[0]) if np.asarray(y).size > 0 else "unknown"
                    recent_labels.append(label)
                    confidence = None
                    try:
                        if hasattr(predictor, "predict_proba"):
                            proba = predictor.predict_proba(x_model[np.newaxis, :, :])
                            confidence = float(np.max(proba[0]))
                    except Exception:
                        confidence = None
                    allow_send = True
                    if confidence is not None:
                        if confidence < float(args.min_confidence):
                            pred_txt = f" pred={label} conf={confidence:.3f} (<min {float(args.min_confidence):.3f})"
                            allow_send = False
                        else:
                            pred_txt = f" pred={label} conf={confidence:.3f}"
                    else:
                        pred_txt = f" pred={label}"

                    if plotter is not None:
                        plotter.update(label=label, confidence=confidence)

                    if udp_sender is not None:
                        if allow_send and udp_sender.send_label(label):
                            sent_txt = f" udp_sent={label}"
                        elif allow_send:
                            sent_txt = f" udp_skipped={label}"
                        else:
                            sent_txt = " udp_blocked=min_conf"
                LOGGER.info(
                    "window #%d shape=%s fs=%.3fHz raw_samples=%d t=[%.6f, %.6f] lsl_now=%.6f%s%s%s",
                    produced + 1,
                    tuple(w.x.shape),
                    w.fs_hz,
                    w.raw_samples,
                    w.t_start_lsl,
                    w.t_end_lsl,
                    lsl_clock(),
                    model_shape_txt,
                    pred_txt,
                    sent_txt,
                )
                produced += 1
                if args.max_windows > 0 and produced >= int(args.max_windows):
                    should_stop = True
                    break
    finally:
        try:
            inlet.close_stream()
        except Exception:
            pass
        if udp_sender is not None:
            udp_sender.close()
        if plotter is not None:
            plotter.close()
    no_sample_tail = max(0.0, time.monotonic() - last_sample_mono)
    LOGGER.info(
        "Stopped. windows=%d samples_received=%d chunks_received=%d no_sample_for=%.2fs",
        produced,
        int(generator.total_samples_received),
        int(generator.total_chunks_received),
        no_sample_tail,
    )
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        LOGGER.error("Windowing failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
