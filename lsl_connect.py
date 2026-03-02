#!/usr/bin/env python3
"""Real-time EEG bridge: LSL stream -> classifier -> game control commands."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import signal
import socket
import sys
import time
from collections import deque
import threading
import select
import csv
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

if sys.platform.startswith("win"):
    try:
        import msvcrt  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - import guard
        msvcrt = None
    tty = None
    termios = None
else:
    msvcrt = None
    try:
        import tty
        import termios
    except Exception:  # pragma: no cover - import guard
        tty = None
        termios = None

try:
    from pylsl import StreamInlet, local_clock, resolve_byprop, resolve_streams
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'pylsl'. Install with: pip install pylsl"
    ) from exc


LOGGER = logging.getLogger("bci_bridge")


def lsl_clock() -> float:
    """Best-effort current LSL clock time."""
    try:
        return float(local_clock())
    except Exception:
        return float(time.time())


@dataclass
class StreamMetadata:
    name: str
    stream_type: str
    source_id: str
    channel_count: int
    nominal_srate: float
    channel_names: list[str]


@dataclass
class Prediction:
    label: str
    confidence: float
    timestamp: float
    sample_rate: float
    n_samples: int
    meta: dict[str, Any] = field(default_factory=dict)


class SlidingWindowBuffer:
    """Timestamped sample buffer with recent-window extraction."""

    def __init__(self, max_seconds: float) -> None:
        self.max_seconds = float(max_seconds)
        self._samples: Deque[np.ndarray] = deque()
        self._timestamps: Deque[float] = deque()
        self._channel_count: Optional[int] = None
        self._warned_channel_mismatch = False

    @property
    def latest_timestamp(self) -> Optional[float]:
        return self._timestamps[-1] if self._timestamps else None

    @property
    def channel_count(self) -> Optional[int]:
        return self._channel_count

    def append_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        if samples.ndim != 2:
            raise ValueError(f"Expected 2D samples, got shape={samples.shape}")
        if timestamps.ndim != 1:
            raise ValueError(f"Expected 1D timestamps, got shape={timestamps.shape}")
        if samples.shape[0] != timestamps.shape[0]:
            raise ValueError("Sample count and timestamp count do not match")
        if samples.shape[0] == 0:
            return

        if self._channel_count is None:
            self._channel_count = int(samples.shape[1])
        elif samples.shape[1] != self._channel_count:
            samples = self._align_channels(samples)

        for row, ts in zip(samples, timestamps):
            self._samples.append(np.asarray(row, dtype=np.float64))
            self._timestamps.append(float(ts))

        self._drop_old_samples()

    def get_recent_window(self, seconds: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self._timestamps:
            n_channels = self._channel_count or 0
            return np.empty((0, n_channels), dtype=np.float64), np.empty((0,), dtype=np.float64)

        latest = self._timestamps[-1]
        threshold = latest - float(seconds)

        start_idx = 0
        for i, ts in enumerate(self._timestamps):
            if ts >= threshold:
                start_idx = i
                break

        samples = list(self._samples)[start_idx:]
        timestamps = list(self._timestamps)[start_idx:]
        if not samples:
            n_channels = self._channel_count or 0
            return np.empty((0, n_channels), dtype=np.float64), np.empty((0,), dtype=np.float64)

        return np.vstack(samples), np.asarray(timestamps, dtype=np.float64)

    def _drop_old_samples(self) -> None:
        if not self._timestamps:
            return

        newest = self._timestamps[-1]
        threshold = newest - self.max_seconds
        while self._timestamps and self._timestamps[0] < threshold:
            self._timestamps.popleft()
            self._samples.popleft()

    def _align_channels(self, samples: np.ndarray) -> np.ndarray:
        assert self._channel_count is not None
        current = int(samples.shape[1])
        target = int(self._channel_count)

        if not self._warned_channel_mismatch:
            LOGGER.warning(
                "Incoming chunk has %s channels, expected %s. Auto-aligning with truncate/pad.",
                current,
                target,
            )
            self._warned_channel_mismatch = True

        if current > target:
            return samples[:, :target]

        pad_width = target - current
        pad = np.zeros((samples.shape[0], pad_width), dtype=samples.dtype)
        return np.hstack([samples, pad])


class TimestampRepair:
    """Repair/synthesize timestamps when inlet timestamps are missing or invalid."""

    def __init__(self, nominal_srate: float) -> None:
        self.nominal_srate = float(nominal_srate)
        self._last_ts: Optional[float] = None

    def repair(self, timestamps: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty((0,), dtype=np.float64)

        step = self._default_step(timestamps)
        valid = (
            timestamps.ndim == 1
            and timestamps.size == count
            and np.all(np.isfinite(timestamps))
        )

        if not valid:
            fixed = self._synthetic(count, step)
            self._last_ts = float(fixed[-1])
            return fixed

        fixed = np.asarray(timestamps, dtype=np.float64).copy()

        if self._last_ts is not None and fixed[0] <= self._last_ts:
            fixed[0] = self._last_ts + step

        for i in range(1, count):
            if fixed[i] <= fixed[i - 1]:
                fixed[i] = fixed[i - 1] + step

        self._last_ts = float(fixed[-1])
        return fixed

    def _synthetic(self, count: int, step: float) -> np.ndarray:
        end_ts = lsl_clock()
        if self._last_ts is not None and end_ts <= self._last_ts:
            end_ts = self._last_ts + step * count
        start_ts = end_ts - step * (count - 1)
        return np.linspace(start_ts, end_ts, num=count, dtype=np.float64)

    def _default_step(self, timestamps: np.ndarray) -> float:
        if self.nominal_srate > 0:
            return 1.0 / self.nominal_srate

        if timestamps.ndim == 1 and timestamps.size > 1:
            diffs = np.diff(timestamps)
            diffs = diffs[np.isfinite(diffs)]
            diffs = diffs[diffs > 1e-9]
            if diffs.size > 0:
                return float(np.median(diffs))

        return 1e-3


class BaseClassifier:
    def predict(self, window: np.ndarray, sample_rate: float, channel_names: Sequence[str]) -> Prediction:
        raise NotImplementedError


class HeuristicClassifier(BaseClassifier):
    """Fallback classifier based on hemisphere energy difference."""

    def __init__(self, deadband: float = 0.15) -> None:
        self.deadband = float(deadband)

    def predict(self, window: np.ndarray, sample_rate: float, channel_names: Sequence[str]) -> Prediction:
        if window.size == 0:
            return Prediction("idle", 0.0, lsl_clock(), sample_rate, 0)

        rms = np.sqrt(np.mean(np.square(window), axis=0))
        split = max(1, len(rms) // 2)
        left_energy = float(np.mean(rms[:split]))
        right_energy = float(np.mean(rms[split:])) if len(rms[split:]) else left_energy

        baseline = max(1e-8, left_energy + right_energy)
        score = (left_energy - right_energy) / baseline

        if abs(score) < self.deadband:
            label = "idle"
            confidence = max(0.0, 1.0 - abs(score) / max(1e-8, self.deadband))
        elif score > 0:
            label = "left"
            confidence = min(1.0, abs(score))
        else:
            label = "right"
            confidence = min(1.0, abs(score))

        return Prediction(
            label=label,
            confidence=float(confidence),
            timestamp=lsl_clock(),
            sample_rate=float(sample_rate),
            n_samples=int(window.shape[0]),
            meta={
                "left_energy": left_energy,
                "right_energy": right_energy,
                "score": score,
            },
        )


class SklearnClassifier(BaseClassifier):
    """Sklearn model classifier over handcrafted time/frequency features."""

    def __init__(self, model_path: str) -> None:
        try:
            import joblib
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Missing dependency 'joblib'. Install with: pip install joblib") from exc

        model_obj = joblib.load(model_path)
        if isinstance(model_obj, dict) and "model" in model_obj:
            self.model = model_obj["model"]
            self.labels = model_obj.get("labels")
        else:
            self.model = model_obj
            self.labels = None

    def predict(self, window: np.ndarray, sample_rate: float, channel_names: Sequence[str]) -> Prediction:
        features = extract_features(window, sample_rate)
        x = adapt_feature_size(features.reshape(1, -1), self.model)
        pred_raw = self.model.predict(x)[0]

        label = self._decode_label(pred_raw)
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x), dtype=np.float64)
            if probs.ndim == 2 and probs.shape[1] > 0:
                confidence = float(np.max(probs[0]))

        return Prediction(
            label=label,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            timestamp=lsl_clock(),
            sample_rate=float(sample_rate),
            n_samples=int(window.shape[0]),
            meta={"feature_count": int(x.shape[1])},
        )

    def _decode_label(self, raw: Any) -> str:
        if self.labels is None:
            return str(raw)

        try:
            idx = int(raw)
            if 0 <= idx < len(self.labels):
                return str(self.labels[idx])
        except Exception:
            pass
        return str(raw)


class CustomCallableClassifier(BaseClassifier):
    """Classifier wrapper around a user-defined Python function."""

    def __init__(self, module_path: str, symbol: str) -> None:
        module_file = Path(module_path)
        if not module_file.exists():
            raise FileNotFoundError(f"Classifier module not found: {module_path}")

        spec = importlib.util.spec_from_file_location("bci_custom_classifier", module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import classifier module: {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn = getattr(module, symbol, None)
        if not callable(fn):
            raise RuntimeError(f"Classifier symbol '{symbol}' is not callable in {module_path}")

        self._fn = fn
        self._module_path = str(module_file)
        self._symbol = symbol

    def predict(self, window: np.ndarray, sample_rate: float, channel_names: Sequence[str]) -> Prediction:
        try:
            raw = self._fn(window=window, sample_rate=sample_rate, channel_names=list(channel_names))
        except TypeError:
            raw = self._fn(window, sample_rate, list(channel_names))

        label, confidence, meta = self._normalize_output(raw)
        return Prediction(
            label=label,
            confidence=confidence,
            timestamp=lsl_clock(),
            sample_rate=float(sample_rate),
            n_samples=int(window.shape[0]),
            meta=meta,
        )

    def _normalize_output(self, raw: Any) -> Tuple[str, float, dict[str, Any]]:
        if isinstance(raw, str):
            return raw, 1.0, {}

        if isinstance(raw, (tuple, list)):
            if not raw:
                raise ValueError("Custom classifier returned empty tuple/list")
            label = str(raw[0])
            confidence = float(raw[1]) if len(raw) > 1 else 1.0
            return label, float(np.clip(confidence, 0.0, 1.0)), {}

        if isinstance(raw, dict):
            if "label" not in raw:
                raise ValueError("Custom classifier dict output must include 'label'")
            label = str(raw.get("label"))
            confidence = float(raw.get("confidence", 1.0))
            meta = raw.get("meta", {})
            if not isinstance(meta, dict):
                meta = {"meta": meta}
            return label, float(np.clip(confidence, 0.0, 1.0)), meta

        raise ValueError(
            "Unsupported custom classifier output. Use str, (label, confidence), or dict(label=..., confidence=..., meta=...)."
        )


def extract_features(window: np.ndarray, sample_rate: float) -> np.ndarray:
    """Channel-wise features: basic stats + broad-band powers."""
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    var = np.var(window, axis=0)
    rms = np.sqrt(np.mean(np.square(window), axis=0))
    ptp = np.ptp(window, axis=0)

    feature_blocks = [mean, std, var, rms, ptp]

    if sample_rate > 0 and window.shape[0] > 4:
        freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / sample_rate)
        spectrum = np.abs(np.fft.rfft(window, axis=0)) ** 2

        bands = ((1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 45.0))
        for f_min, f_max in bands:
            mask = (freqs >= f_min) & (freqs < f_max)
            if np.any(mask):
                band_power = np.mean(spectrum[mask, :], axis=0)
            else:
                band_power = np.zeros(window.shape[1], dtype=np.float64)
            feature_blocks.append(band_power)
    else:
        feature_blocks.extend([np.zeros(window.shape[1], dtype=np.float64) for _ in range(5)])

    return np.concatenate(feature_blocks, axis=0).astype(np.float64)


def adapt_feature_size(x: np.ndarray, model: object) -> np.ndarray:
    expected = getattr(model, "n_features_in_", None)
    if expected is None or int(expected) == int(x.shape[1]):
        return x

    expected = int(expected)
    if x.shape[1] > expected:
        LOGGER.warning(
            "Feature vector has %s values but model expects %s. Truncating.",
            x.shape[1],
            expected,
        )
        return x[:, :expected]

    LOGGER.warning(
        "Feature vector has %s values but model expects %s. Zero-padding.",
        x.shape[1],
        expected,
    )
    pad = np.zeros((x.shape[0], expected - x.shape[1]), dtype=x.dtype)
    return np.hstack([x, pad])


class Transport(Protocol):
    def send(self, payload: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class UdpTransport:
    def __init__(self, host: str, port: int) -> None:
        self.address = (host, int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._socket.sendto(body, self.address)

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass


class TcpTransport:
    """Persistent TCP JSON-lines sender, inspired by Cirus-Alpha ControlClient."""

    def __init__(self, host: str, port: int) -> None:
        self.host = str(host)
        self.port = int(port)
        self._socket: Optional[socket.socket] = None

    def send(self, payload: dict[str, Any]) -> None:
        sock = self._ensure_socket()
        if sock is None:
            return

        try:
            body = (json.dumps(payload) + "\n").encode("utf-8")
            sock.sendall(body)
        except OSError:
            self.close()

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        except OSError:
            pass
        self._socket = None

    def _ensure_socket(self) -> Optional[socket.socket]:
        if self._socket is not None:
            return self._socket

        try:
            self._socket = socket.create_connection((self.host, self.port), timeout=1.0)
        except OSError:
            self._socket = None
        return self._socket


class LslMarkerTransport:
    """Publish predicted action labels as an LSL marker stream."""

    def __init__(self, stream_name: str, source_id: Optional[str]) -> None:
        from pylsl import StreamInfo, StreamOutlet

        resolved_source_id = source_id or f"{stream_name}_source_{int(time.time() * 1000)}"
        info = StreamInfo(stream_name, "Markers", 1, 0, "string", resolved_source_id)
        self._outlet = StreamOutlet(info)

    def send(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action", ""))
        ts = float(payload.get("timestamp", lsl_clock()))
        try:
            self._outlet.push_sample([action], ts)
        except Exception:
            pass

    def close(self) -> None:
        # pylsl outlets do not require explicit close.
        return


class PredictionPublisher:
    def __init__(
        self,
        transports: list[Transport],
        action_map: Dict[str, str],
        min_confidence: float,
        cooldown_seconds: float,
        idle_label: str,
        emit_idle: bool,
    ) -> None:
        self.transports = transports
        self.action_map = action_map
        self.min_confidence = float(min_confidence)
        self.cooldown_seconds = float(cooldown_seconds)
        self.idle_label = str(idle_label)
        self.emit_idle = bool(emit_idle)

        self._last_sent_at = 0.0
        self._last_action: Optional[str] = None
        self._run_start = lsl_clock()

    def publish(self, prediction: Prediction, stream: StreamMetadata) -> None:
        if prediction.confidence < self.min_confidence:
            return

        action = self.action_map.get(prediction.label, prediction.label)
        if not self.emit_idle:
            if prediction.label.lower() == self.idle_label.lower() or action.lower() == self.idle_label.lower():
                return

        now = lsl_clock()
        if self.cooldown_seconds > 0:
            if action == self._last_action and (now - self._last_sent_at) < self.cooldown_seconds:
                return

        payload: dict[str, Any] = {
            "type": "bci_command",
            "event": "prediction",
            "action": action,
            "label": prediction.label,
            "confidence": round(float(prediction.confidence), 6),
            "timestamp": round(float(prediction.timestamp), 6),
            "t": round(float(prediction.timestamp - self._run_start), 6),
            "sample_rate": round(float(prediction.sample_rate), 6),
            "window_samples": int(prediction.n_samples),
            "stream": {
                "name": stream.name,
                "type": stream.stream_type,
                "source_id": stream.source_id,
                "channel_count": stream.channel_count,
                "nominal_srate": stream.nominal_srate,
            },
        }
        if prediction.meta:
            payload["meta"] = make_json_safe(prediction.meta)

        for transport in self.transports:
            try:
                transport.send(payload)
            except Exception as exc:
                LOGGER.warning("Failed to send command through transport: %s", exc)

        self._last_sent_at = now
        self._last_action = action

        LOGGER.info(
            "Sent action=%s label=%s confidence=%.3f",
            action,
            prediction.label,
            prediction.confidence,
        )

    def close(self) -> None:
        for transport in self.transports:
            try:
                transport.close()
            except Exception:
                pass


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def resolve_stream(
    name: Optional[str],
    stream_type: str,
    source_id: Optional[str],
    timeout_seconds: float,
):
    deadline = lsl_clock() + float(timeout_seconds)
    while lsl_clock() < deadline:
        if source_id:
            streams = resolve_byprop("source_id", source_id, timeout=1)
        elif name:
            streams = resolve_byprop("name", name, timeout=1)
        else:
            streams = resolve_byprop("type", stream_type, timeout=1)

        if streams:
            return streams[0]

    query = (
        f"source_id={source_id}" if source_id else f"name={name}" if name else f"type={stream_type}"
    )
    raise RuntimeError(f"No LSL stream found ({query}) within {timeout_seconds:.1f}s")


def list_lsl_streams(wait_time: float) -> None:
    streams = resolve_streams(wait_time=float(wait_time))
    if not streams:
        print("No LSL streams found.")
        return

    print("Available LSL streams:")
    for idx, info in enumerate(streams, start=1):
        try:
            name = info.name()
        except Exception:
            name = ""
        try:
            stream_type = info.type()
        except Exception:
            stream_type = ""
        try:
            source_id = info.source_id()
        except Exception:
            source_id = ""
        try:
            channels = int(info.channel_count())
        except Exception:
            channels = -1
        try:
            srate = float(info.nominal_srate())
        except Exception:
            srate = -1.0

        print(
            f"[{idx}] name='{name}' type='{stream_type}' source_id='{source_id}' "
            f"channels={channels} nominal_srate={srate:.3f}"
        )


def extract_channel_names(info: Any, channel_count: int) -> list[str]:
    names: list[str] = []
    try:
        desc = info.desc()
        channels_elem = desc.child("channels")
        channel_elem = channels_elem.child("channel") if channels_elem else None
        if (not channel_elem) and channels_elem:
            channel_elem = channels_elem.child(0)

        idx = 0
        while channel_elem and idx < channel_count:
            label = str(channel_elem.child_value("label") or "").strip()
            names.append(label or f"Ch{idx + 1}")
            channel_elem = channel_elem.next_sibling()
            idx += 1
    except Exception:
        names = []

    while len(names) < channel_count:
        names.append(f"Ch{len(names) + 1}")

    return names


def build_stream_metadata(info: Any) -> StreamMetadata:
    try:
        name = str(info.name())
    except Exception:
        name = ""

    try:
        stream_type = str(info.type())
    except Exception:
        stream_type = ""

    try:
        source_id = str(info.source_id())
    except Exception:
        source_id = ""

    try:
        channel_count = int(info.channel_count())
    except Exception:
        channel_count = 0

    try:
        nominal_srate = float(info.nominal_srate())
    except Exception:
        nominal_srate = 0.0

    channel_names = extract_channel_names(info, channel_count)
    return StreamMetadata(
        name=name,
        stream_type=stream_type,
        source_id=source_id,
        channel_count=channel_count,
        nominal_srate=nominal_srate,
        channel_names=channel_names,
    )


def parse_action_map(raw: Optional[str]) -> Dict[str, str]:
    default_map = {"left": "move_left", "right": "move_right", "idle": "idle"}
    if not raw:
        return default_map

    path = Path(raw)
    text = path.read_text(encoding="utf-8") if path.exists() else raw
    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("Action map must be a JSON object")

    mapped = {str(k): str(v) for k, v in data.items()}
    return {**default_map, **mapped}


def estimate_sample_rate(timestamps: np.ndarray, nominal_srate: float) -> float:
    if timestamps.size < 3:
        return float(nominal_srate) if nominal_srate > 0 else 0.0

    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 1e-9]
    if diffs.size < 2:
        return float(nominal_srate) if nominal_srate > 0 else 0.0

    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return float(nominal_srate) if nominal_srate > 0 else 0.0

    fs_est = 1.0 / median_dt

    span = float(timestamps[-1] - timestamps[0])
    if span > 0:
        fs_span = (len(timestamps) - 1) / span
        if fs_span > 0 and abs(fs_span - fs_est) / fs_span > 0.2:
            fs_est = fs_span

    return float(fs_est)


def build_classifier(args: argparse.Namespace) -> BaseClassifier:
    if args.custom_classifier_module and args.model_path:
        raise ValueError("Use either --model-path or --custom-classifier-module, not both")

    if args.custom_classifier_module:
        return CustomCallableClassifier(
            module_path=args.custom_classifier_module,
            symbol=args.custom_classifier_symbol,
        )

    if args.model_path:
        return SklearnClassifier(args.model_path)

    return HeuristicClassifier(deadband=args.heuristic_deadband)


def build_transports(args: argparse.Namespace, stream: StreamMetadata) -> list[Transport]:
    protocols = [p.strip().lower() for p in args.game_protocol.split(",") if p.strip()]
    if not protocols:
        raise ValueError("--game-protocol must contain at least one protocol")

    transports: list[Transport] = []
    for proto in protocols:
        if proto == "udp":
            transports.append(UdpTransport(args.game_host, args.game_port))
        elif proto == "tcp":
            transports.append(TcpTransport(args.game_host, args.game_port))
        elif proto == "lsl":
            marker_name = args.lsl_marker_stream_name or f"{stream.name or 'EEG'}_BCICommands"
            transports.append(LslMarkerTransport(marker_name, args.lsl_marker_source_id))
        else:
            raise ValueError(f"Unsupported game protocol: '{proto}' (use udp,tcp,lsl)")

    return transports


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read EEG from LSL, run classification, and send game control commands."
    )

    parser.add_argument("--list-streams", action="store_true", help="List available LSL streams and exit")
    parser.add_argument("--stream-name", type=str, default=None, help="LSL stream name filter")
    parser.add_argument("--stream-type", type=str, default="EEG", help="LSL stream type filter")
    parser.add_argument("--source-id", type=str, default=None, help="LSL source_id filter")
    parser.add_argument("--stream-timeout", type=float, default=20.0, help="Stream discovery timeout (s)")

    parser.add_argument("--window-seconds", type=float, default=1.0, help="Classification window size (s)")
    parser.add_argument("--stride-seconds", type=float, default=0.2, help="Time between predictions (s)")
    parser.add_argument("--min-window-samples", type=int, default=16, help="Min samples for one prediction")
    parser.add_argument("--history-seconds", type=float, default=5.0, help="Ring buffer length (s)")
    parser.add_argument("--max-chunk-samples", type=int, default=64, help="LSL max samples per pull")
    parser.add_argument("--pull-timeout", type=float, default=0.0, help="LSL pull timeout (s)")

    parser.add_argument("--model-path", type=str, default=None, help="Path to sklearn model (.joblib)")
    parser.add_argument(
        "--custom-classifier-module",
        type=str,
        default=None,
        help="Path to Python module implementing custom classifier function",
    )
    parser.add_argument(
        "--custom-classifier-symbol",
        type=str,
        default="predict_window",
        help="Function name in custom classifier module",
    )
    parser.add_argument(
        "--heuristic-deadband",
        type=float,
        default=0.15,
        help="Heuristic classifier deadband around idle",
    )

    parser.add_argument(
        "--game-protocol",
        type=str,
        default="udp",
        help="Comma-separated protocols: udp,tcp,lsl",
    )
    parser.add_argument("--game-host", type=str, default="127.0.0.1", help="Game host (udp/tcp)")
    parser.add_argument("--game-port", type=int, default=9000, help="Game port (udp/tcp)")
    parser.add_argument(
        "--lsl-marker-stream-name",
        type=str,
        default=None,
        help="LSL marker stream name (when --game-protocol includes lsl)",
    )
    parser.add_argument(
        "--lsl-marker-source-id",
        type=str,
        default=None,
        help="LSL marker stream source_id (when --game-protocol includes lsl)",
    )
    parser.add_argument(
        "--action-map",
        type=str,
        default=None,
        help="Action map JSON object or path to JSON file",
    )
    parser.add_argument("--idle-label", type=str, default="idle", help="Idle label name")
    parser.add_argument("--emit-idle", action="store_true", help="Also send idle commands")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Do not send commands below this confidence",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.15,
        help="Suppress repeated identical actions during cooldown",
    )

    parser.add_argument(
        "--record-output",
        type=str,
        default=None,
        help="Path to CSV file where raw samples will be recorded when the record key is pressed",
    )
    parser.add_argument(
        "--record-key",
        type=str,
        default="r",
        help="Single character key used to toggle recording mode",
    )
    parser.add_argument(
        "--auto-record",
        action="store_true",
        help="Start recording immediately without waiting for key press",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    # stimulus marker stream settings (optional)
    parser.add_argument(
        "--stim-stream-name",
        type=str,
        default=None,
        help="LSL stream name for stimulus markers",
    )
    parser.add_argument(
        "--stim-stream-type",
        type=str,
        default="Markers",
        help="LSL stream type for stimulus markers",
    )
    parser.add_argument(
        "--stim-source-id",
        type=str,
        default=None,
        help="LSL source_id for stimulus markers",
    )
    parser.add_argument("--log-predictions", action="store_true", help="Log every prediction")

    return parser


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run(args: argparse.Namespace) -> int:
    setup_logging(args.log_level)

    if args.record_key and len(args.record_key) != 1:
        raise ValueError("--record-key must be a single character")

    if args.list_streams:
        list_lsl_streams(wait_time=min(max(args.stream_timeout, 0.1), 10.0))
        return 0

    stream = resolve_stream(
        name=args.stream_name,
        stream_type=args.stream_type,
        source_id=args.source_id,
        timeout_seconds=args.stream_timeout,
    )

    # ensure numeric types for pylsl
    # pylsl expects integer buffer lengths (c_int), float values trigger ctypes errors
    try:
        max_buflen = int(max(5.0, float(args.history_seconds) * 2.0))
    except Exception:
        max_buflen = 5
    try:
        max_chunklen = int(max(1, int(args.max_chunk_samples)))
    except Exception:
        max_chunklen = 1
    LOGGER.debug("Creating StreamInlet with max_buflen=%d max_chunklen=%d", max_buflen, max_chunklen)
    inlet = StreamInlet(
        stream,
        max_buflen=max_buflen,
        max_chunklen=max_chunklen,
    )

    metadata = build_stream_metadata(inlet.info())
    LOGGER.info(
        "Connected LSL stream: name='%s' type='%s' source_id='%s' channels=%d nominal_srate=%.3f",
        metadata.name,
        metadata.stream_type,
        metadata.source_id,
        metadata.channel_count,
        metadata.nominal_srate,
    )

    if metadata.nominal_srate <= 0:
        LOGGER.warning("Stream nominal sampling rate is unavailable (<= 0). Using timestamp-based estimation.")

    # LSL reference clock for this recorder run.
    # All persisted timestamps are relative to this value: t_rel = t_lsl - experiment_start_lsl.
    experiment_start_lsl = lsl_clock()
    LOGGER.info(
        "Experiment start (LSL clock)=%.6f; all stored timestamps are relative to this origin",
        experiment_start_lsl,
    )

    # if a stim stream is specified, open a second inlet to receive markers
    stim_inlet = None
    if args.stim_stream_name or args.stim_source_id:
        try:
            stim_info = resolve_stream(
                name=args.stim_stream_name,
                stream_type=args.stim_stream_type,
                source_id=args.stim_source_id,
                timeout_seconds=args.stream_timeout,
            )
            stim_inlet = StreamInlet(stim_info)
            LOGGER.info("Connected stimulus stream: name=%s type=%s source_id=%s",
                        args.stim_stream_name, args.stim_stream_type, args.stim_source_id)
        except Exception as exc:
            LOGGER.warning("Unable to resolve stimulus stream: %s", exc)
            stim_inlet = None

    should_stop = False
    stop_event = threading.Event()

    def stop_handler(_sig: int, _frame: object) -> None:
        nonlocal should_stop
        should_stop = True
        stop_event.set()

    # recording state
    recording = False
    recording_start_rel = 0.0
    recording_stop_rel: Optional[float] = None
    record_file = None  # type: Optional[Any]
    record_writer: Optional[csv.writer] = None
    edf_writer = None  # type: Optional[Any]
    edf_pending_samples = np.empty((0, 7), dtype=np.float64)
    record_sample_rate_hz: Optional[int] = None
    recorded_samples = 0  # count of samples written during current recording
    prestart_dropped_samples = 0  # samples discarded because they predate record_start
    poststop_dropped_samples = 0  # samples discarded because they are after record_stop
    first_written_rel_ts: Optional[float] = None
    last_written_rel_ts: Optional[float] = None
    last_header_update = 0.0

    command_queue: "queue.Queue[Tuple[str, Optional[float]]]" = queue.Queue()
    worker_errors: "queue.Queue[tuple[str, Exception]]" = queue.Queue()
    raw_eeg_queue: "queue.Queue[object]" = queue.Queue()
    timed_eeg_queue: "queue.Queue[object]" = queue.Queue()
    raw_sentinel = object()
    timed_sentinel = object()
    observed_ts: Deque[float] = deque(maxlen=4096)
    observed_lock = threading.Lock()
    observed_sample_count = 0
    observed_clock_start: Optional[float] = None
    observed_clock_last: Optional[float] = None
    observed_peak_abs_uV = 0.0

    def resolve_recording_sample_rate() -> int:
        """Infer a robust recording rate for EDF header metadata."""
        nominal = float(metadata.nominal_srate) if metadata.nominal_srate > 0 else 0.0

        with observed_lock:
            ts_copy = np.asarray(observed_ts, dtype=np.float64) if observed_ts else np.empty((0,), dtype=np.float64)
            sample_count = int(observed_sample_count)
            clock_start = float(observed_clock_start) if observed_clock_start is not None else None
            clock_last = float(observed_clock_last) if observed_clock_last is not None else None

        ts_rate = estimate_sample_rate(ts_copy, nominal) if ts_copy.size >= 3 else 0.0
        wall_rate = 0.0
        if clock_start is not None and clock_last is not None and sample_count > 2:
            span = max(clock_last - clock_start, 0.0)
            if span > 0.5:
                wall_rate = float(sample_count - 1) / span

        def _valid(rate: float) -> bool:
            # Some devices expose high-rate EEG streams (>5 kHz). Keep a guardrail
            # against absurd values but do not clamp valid high sampling rates.
            return np.isfinite(rate) and 20.0 <= float(rate) <= 20000.0

        candidates: list[float] = []
        if _valid(ts_rate):
            candidates.append(float(ts_rate))
        if _valid(nominal):
            candidates.append(float(nominal))
        if _valid(wall_rate):
            candidates.append(float(wall_rate))

        chosen = 250.0
        if candidates:
            if _valid(ts_rate):
                chosen = float(ts_rate)
            elif _valid(nominal):
                chosen = float(nominal)
            elif _valid(wall_rate):
                chosen = float(wall_rate)

        # If nominal is clearly implausible, prefer measured rates.
        if _valid(nominal):
            measured = float(ts_rate if _valid(ts_rate) else (wall_rate if _valid(wall_rate) else nominal))
            if measured > 0:
                rel_diff = abs(measured - nominal) / max(nominal, 1e-9)
                if rel_diff > 0.20:
                    LOGGER.warning(
                        "Sampling-rate mismatch: nominal=%.3fHz measured=%.3fHz. Using measured value for EDF header.",
                        nominal,
                        measured,
                    )
                    chosen = measured
        else:
            LOGGER.warning(
                "Nominal sampling rate %.3fHz is invalid; using measured/default EDF rate %.3fHz.",
                nominal,
                chosen,
            )

        chosen_int = max(1, int(round(chosen)))
        LOGGER.info(
            "Resolved EDF sample rate: %dHz (nominal=%.3f, ts_est=%.3f, wall_est=%.3f, ts_samples=%d, wall_samples=%d)",
            chosen_int,
            nominal,
            ts_rate,
            wall_rate,
            int(ts_copy.size),
            sample_count,
        )
        return chosen_int

    def start_recording(start_rel_ts: Optional[float] = None) -> None:
        nonlocal recording, recording_start_rel, recording_stop_rel
        nonlocal record_file, record_writer, edf_writer, edf_pending_samples, record_sample_rate_hz
        nonlocal recorded_samples, prestart_dropped_samples, poststop_dropped_samples
        nonlocal first_written_rel_ts, last_written_rel_ts
        if not args.record_output:
            return

        # Make sure any previous recording is stopped and flushed first
        if edf_writer is not None or record_file is not None:
            stop_recording()

        # Recording start in LSL-relative seconds (0 == experiment start).
        if start_rel_ts is not None and np.isfinite(start_rel_ts):
            recording_start_rel = max(0.0, float(start_rel_ts))
        else:
            recording_start_rel = max(0.0, float(lsl_clock() - experiment_start_lsl))
        recording_stop_rel = None
        # reset counters for the new recording window
        recorded_samples = 0
        prestart_dropped_samples = 0
        poststop_dropped_samples = 0
        first_written_rel_ts = None
        last_written_rel_ts = None
        edf_pending_samples = np.empty((0, 7), dtype=np.float64)

        path = Path(args.record_output)
        if path.suffix.lower() == ".edf":
            # EDF recording requires pyedflib and fixed 7 electrode labels
            try:
                import pyedflib
            except Exception:
                LOGGER.error("pyedflib is required for EDF recording. Install with: pip install pyedflib")
                return
            # prepare header for 7 fixed channels
            ch_labels = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
            nominal_sr = resolve_recording_sample_rate()
            record_sample_rate_hz = int(nominal_sr)
            with observed_lock:
                peak_abs = float(observed_peak_abs_uV)
            if not np.isfinite(peak_abs) or peak_abs <= 0:
                peak_abs = 50.0
            # Keep EDF scaling practical for viewers while preserving headroom.
            # This avoids huge y-axis spacing from overly wide physical ranges.
            phys_abs_uV = min(20000.0, max(500.0, peak_abs * 10.0))
            phys_abs_uV = float(round(phys_abs_uV, 1))
            LOGGER.info(
                "EDF physical range set to +/-%.1f uV (observed peak=%.3f uV)",
                phys_abs_uV,
                peak_abs,
            )
            headers = []
            for lbl in ch_labels:
                headers.append(
                    {
                        "label": lbl,
                        "dimension": "uV",
                        "sample_frequency": nominal_sr,
                        # EDF numeric fields are limited to 8 chars.
                        "physical_min": -phys_abs_uV,
                        "physical_max": phys_abs_uV,
                        "digital_min": -32768,
                        "digital_max": 32767,
                    }
                )
            try:
                edf_writer = pyedflib.EdfWriter(str(path), n_channels=len(ch_labels), file_type=pyedflib.FILETYPE_EDFPLUS)
                edf_writer.setSignalHeaders(headers)
                edf_writer.update_header()  # flush header to disk
                try:
                    edf_writer.writeAnnotation(0.0, 0.0, "exp_start")
                    edf_writer.writeAnnotation(0.0, 0.0, "record_start")
                except Exception:
                    pass
                recording = True
                LOGGER.info(
                    "EDF recording started -> %s (header flushed, fs=%dHz, record_start_rel=%.6f s)",
                    args.record_output,
                    nominal_sr,
                    recording_start_rel,
                )
            except Exception as exc:
                LOGGER.error("Failed to create EDF file: %s", exc)
                recording = False
                edf_writer = None
                record_sample_rate_hz = None
        else:
            try:
                record_file = open(args.record_output, "w", newline="")
                record_writer = csv.writer(record_file)
                header = ["timestamp_lsl_rel"] + metadata.channel_names
                record_writer.writerow(header)
                record_file.flush()  # flush header to disk immediately
                if metadata.nominal_srate > 0:
                    record_sample_rate_hz = int(max(1, round(metadata.nominal_srate)))
                else:
                    record_sample_rate_hz = None
                recording = True
                LOGGER.info(
                    "Recording started -> %s (header flushed, record_start_rel=%.6f s)",
                    args.record_output,
                    recording_start_rel,
                )
            except Exception as exc:
                LOGGER.error("Failed to open record file: %s", exc)
                recording = False
                record_writer = None
                record_sample_rate_hz = None

    def stop_recording(stop_rel_ts: Optional[float] = None) -> None:
        nonlocal recording, recording_start_rel, recording_stop_rel
        nonlocal record_file, record_writer, edf_writer, edf_pending_samples, record_sample_rate_hz
        nonlocal recorded_samples, prestart_dropped_samples, poststop_dropped_samples
        nonlocal first_written_rel_ts, last_written_rel_ts
        if not recording and record_file is None and edf_writer is None:
            return
        if record_file is not None:
            try:
                record_file.flush()  # flush CSV data to disk
                record_file.close()
            except Exception:
                pass
        if edf_writer is not None:
            try:
                # Flush remaining partial EDF samples once at stop to avoid
                # per-chunk record padding inflation.
                if edf_pending_samples.shape[0] > 0:
                    edf_writer.writeSamples(np.ascontiguousarray(edf_pending_samples.T))
                    recorded_samples += int(edf_pending_samples.shape[0])
                    edf_pending_samples = np.empty((0, 7), dtype=np.float64)
                if recording:
                    if stop_rel_ts is not None and np.isfinite(stop_rel_ts):
                        rel_now = max(recording_start_rel, float(stop_rel_ts))
                    else:
                        rel_now = max(0.0, float(lsl_clock() - experiment_start_lsl))
                    rel_now_file = max(0.0, float(rel_now - recording_start_rel))
                    try:
                        edf_writer.writeAnnotation(rel_now_file, 0.0, "record_stop")
                    except Exception:
                        pass
                edf_writer.update_header()  # update header with correct number of datarecords
                edf_writer.close()  # close() flushes EDF data to disk
                LOGGER.info("EDF file closed and flushed to disk")
            except Exception as exc:
                LOGGER.warning("Error closing EDF file: %s", exc)
        if prestart_dropped_samples:
            LOGGER.info(
                "Dropped %d pre-start buffered samples that arrived before record_start",
                prestart_dropped_samples,
            )
        if poststop_dropped_samples:
            LOGGER.info(
                "Dropped %d post-stop samples that arrived after record_stop",
                poststop_dropped_samples,
            )
        # report how many samples were saved
        if recorded_samples:
            try:
                sr = int(record_sample_rate_hz) if record_sample_rate_hz and record_sample_rate_hz > 0 else None
                if sr:
                    LOGGER.info("Wrote %d samples (~%.2f sec at %d Hz)",
                                recorded_samples, recorded_samples / sr, sr)
                    if first_written_rel_ts is not None and last_written_rel_ts is not None:
                        ts_span = max(0.0, float(last_written_rel_ts - first_written_rel_ts))
                        LOGGER.info(
                            "Recorded timestamp span: %.6f s (first=%.6f, last=%.6f)",
                            ts_span,
                            first_written_rel_ts,
                            last_written_rel_ts,
                        )
                else:
                    LOGGER.info("Wrote %d samples", recorded_samples)
            except Exception:
                LOGGER.info("Wrote %d samples", recorded_samples)
        recording = False
        recording_start_rel = 0.0
        recording_stop_rel = None
        record_file = None
        record_writer = None
        edf_writer = None
        edf_pending_samples = np.empty((0, 7), dtype=np.float64)
        record_sample_rate_hz = None
        recorded_samples = 0
        prestart_dropped_samples = 0
        poststop_dropped_samples = 0
        first_written_rel_ts = None
        last_written_rel_ts = None
        LOGGER.info("Recording stopped")

    def write_record_chunk(samples: np.ndarray, rel_ts: np.ndarray) -> None:
        nonlocal recorded_samples, prestart_dropped_samples, poststop_dropped_samples
        nonlocal first_written_rel_ts, last_written_rel_ts, last_header_update, edf_pending_samples
        if not recording:
            return

        # Keep only samples inside the active recording window.
        if (
            rel_ts.ndim == 1
            and rel_ts.size == samples.shape[0]
            and samples.shape[0] > 0
        ):
            keep = rel_ts >= float(recording_start_rel)
            if recording_stop_rel is not None:
                keep = np.logical_and(keep, rel_ts <= float(recording_stop_rel))
                if np.any(rel_ts > float(recording_stop_rel)):
                    poststop_dropped_samples += int(np.count_nonzero(rel_ts > float(recording_stop_rel)))
            if not np.any(keep):
                prestart_dropped_samples += int(np.count_nonzero(rel_ts < float(recording_start_rel)))
                return
            if not np.all(keep):
                prestart_dropped_samples += int(np.count_nonzero(rel_ts < float(recording_start_rel)))
                samples = samples[keep]
                rel_ts = rel_ts[keep]
            if rel_ts.size > 0:
                first = float(rel_ts[0])
                last = float(rel_ts[-1])
                if first_written_rel_ts is None:
                    first_written_rel_ts = first
                last_written_rel_ts = last

        if edf_writer is not None:
            try:
                n_ch = samples.shape[1]
                if n_ch >= 7:
                    arr = samples[:, :7]
                else:
                    pad = np.zeros((samples.shape[0], 7 - n_ch), dtype=samples.dtype)
                    arr = np.hstack([samples, pad])
                if edf_pending_samples.size == 0:
                    edf_pending_samples = np.asarray(arr, dtype=np.float64)
                else:
                    edf_pending_samples = np.vstack([edf_pending_samples, np.asarray(arr, dtype=np.float64)])

                nper = int(record_sample_rate_hz) if record_sample_rate_hz and record_sample_rate_hz > 0 else 0
                if nper <= 0:
                    nper = 1
                full = int(edf_pending_samples.shape[0] // nper)
                if full > 0:
                    n_write = full * nper
                    to_write = edf_pending_samples[:n_write]
                    edf_writer.writeSamples(np.ascontiguousarray(to_write.T))
                    recorded_samples += int(n_write)
                    edf_pending_samples = edf_pending_samples[n_write:]

                now = lsl_clock()
                if now - last_header_update > 1.0:
                    try:
                        edf_writer.update_header()
                    except Exception:
                        pass
                    last_header_update = now
            except Exception as exc:
                LOGGER.warning("EDF write error (%s), stopping recording", exc)
                stop_recording()
            return

        if record_writer is not None:
            try:
                for row_s, row_rel_ts in zip(samples, rel_ts):
                    record_writer.writerow([float(row_rel_ts)] + list(row_s))
                recorded_samples += samples.shape[0]
            except Exception as exc:
                LOGGER.warning("Error writing to record file (%s), stopping recording", exc)
                stop_recording()

    def process_command(cmd: str, cmd_rel_ts: Optional[float] = None) -> None:
        nonlocal recording_stop_rel
        if cmd == "toggle_record":
            if recording:
                stop_recording()
            else:
                start_recording()
            return
        if cmd == "record_start":
            if not recording:
                start_recording(start_rel_ts=cmd_rel_ts)
            return
        if cmd == "record_stop":
            if recording:
                if cmd_rel_ts is not None and np.isfinite(cmd_rel_ts):
                    recording_stop_rel = max(recording_start_rel, float(cmd_rel_ts))
                else:
                    recording_stop_rel = max(0.0, float(lsl_clock() - experiment_start_lsl))
                # Flush currently queued timestamped chunks before closing writer.
                # This minimizes loss around record_stop when queue backlog exists.
                deadline = time.perf_counter() + 0.120
                while time.perf_counter() < deadline:
                    drained_any = False
                    while True:
                        try:
                            item = timed_eeg_queue.get_nowait()
                        except queue.Empty:
                            break
                        if item is timed_sentinel:
                            timed_eeg_queue.put(timed_sentinel)
                            break
                        samples, rel_ts = item
                        write_record_chunk(samples, rel_ts)
                        drained_any = True
                    if not drained_any:
                        time.sleep(0.003)
                stop_recording(stop_rel_ts=recording_stop_rel)
            return

    def keyboard_listener_posix() -> None:
        if tty is None or termios is None:
            LOGGER.warning("Keyboard listener: termios/tty unavailable; keyboard input disabled")
            return

        # Switch stdin to cbreak mode so we get characters immediately.
        fd = sys.stdin.fileno()
        old_settings = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            LOGGER.info("Keyboard listener: stdin switched to cbreak mode")
        except Exception as exc:
            LOGGER.warning("Keyboard listener: failed to set cbreak mode: %s", exc)
            LOGGER.info("Keyboard input may not work; try running with stdin connected (not redirected)")
        try:
            LOGGER.debug("Keyboard listener thread started; press '%s' to toggle recording", args.record_key)
            while not should_stop:
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        ch = sys.stdin.read(1)
                        if not ch:
                            continue
                        LOGGER.debug("Keyboard listener: received char '%s'", repr(ch))
                        # toggle recording on matching key
                        if ch == args.record_key:
                            command_queue.put(("toggle_record", None))
                except Exception as exc:
                    LOGGER.debug("Keyboard listener: select/read error: %s", exc)
        finally:
            if old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    LOGGER.debug("Keyboard listener: restored terminal settings")
                except Exception:
                    pass

    def keyboard_listener_windows() -> None:
        if msvcrt is None:
            LOGGER.warning("Keyboard listener: msvcrt unavailable; keyboard input disabled")
            return

        LOGGER.info("Keyboard listener: using Windows console input")
        LOGGER.debug("Keyboard listener thread started; press '%s' to toggle recording", args.record_key)
        while not should_stop:
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue

                ch = msvcrt.getwch()
                # Special/function keys are emitted as a prefix + payload.
                if ch in ("\x00", "\xe0"):
                    _ = msvcrt.getwch()
                    continue
                LOGGER.debug("Keyboard listener: received char '%s'", repr(ch))
                if ch == args.record_key:
                    command_queue.put(("toggle_record", None))
            except Exception as exc:
                LOGGER.debug("Keyboard listener: read error: %s", exc)
                time.sleep(0.1)

    def keyboard_listener() -> None:
        if sys.platform.startswith("win"):
            keyboard_listener_windows()
        else:
            keyboard_listener_posix()

    def eeg_listener_worker() -> None:
        nonlocal observed_sample_count, observed_clock_start, observed_clock_last, observed_peak_abs_uV
        try:
            while not stop_event.is_set():
                chunk, timestamps = inlet.pull_chunk(
                    timeout=float(args.pull_timeout),
                    max_samples=int(args.max_chunk_samples),
                )
                if not chunk:
                    continue

                samples = np.asarray(chunk, dtype=np.float64)
                if samples.ndim == 1:
                    samples = samples.reshape(1, -1)
                if samples.shape[1] == 0:
                    continue
                if np.any(~np.isfinite(samples)):
                    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

                raw_ts = np.asarray(timestamps, dtype=np.float64) if timestamps else np.empty((0,), dtype=np.float64)
                now_clock = lsl_clock()
                peak_abs = 0.0
                if samples.size > 0:
                    n_obs = min(samples.shape[1], 7)
                    if n_obs > 0:
                        peak_abs = float(np.max(np.abs(samples[:, :n_obs])))
                with observed_lock:
                    observed_sample_count += int(samples.shape[0])
                    if observed_clock_start is None:
                        observed_clock_start = now_clock
                    observed_clock_last = now_clock
                    if np.isfinite(peak_abs) and peak_abs > observed_peak_abs_uV:
                        observed_peak_abs_uV = peak_abs
                raw_eeg_queue.put((samples, raw_ts))
        except Exception as exc:
            worker_errors.put(("eeg_listener_worker", exc))
        finally:
            raw_eeg_queue.put(raw_sentinel)

    def timestamp_worker() -> None:
        ts_repair = TimestampRepair(metadata.nominal_srate)
        try:
            while True:
                if stop_event.is_set() and raw_eeg_queue.empty():
                    break
                try:
                    item = raw_eeg_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is raw_sentinel:
                    break

                samples, raw_ts = item
                fixed_ts = ts_repair.repair(raw_ts, samples.shape[0])
                with observed_lock:
                    observed_ts.extend(float(x) for x in fixed_ts)
                rel_ts = fixed_ts - experiment_start_lsl
                timed_eeg_queue.put((samples, rel_ts))
        except Exception as exc:
            worker_errors.put(("timestamp_worker", exc))
        finally:
            timed_eeg_queue.put(timed_sentinel)

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, stop_handler)
        except Exception as exc:
            LOGGER.debug("Unable to register SIGTERM handler: %s", exc)

    eeg_thread = threading.Thread(target=eeg_listener_worker, daemon=True, name="eeg_listener_worker")
    ts_thread = threading.Thread(target=timestamp_worker, daemon=True, name="timestamp_worker")
    eeg_thread.start()
    ts_thread.start()
    LOGGER.info("Workers started: eeg_listener_worker (acquisition), timestamp_worker (LSL sync)")

    # start keyboard thread if recording support is requested
    listener_thread: Optional[threading.Thread] = None
    if args.record_output:
        # if auto-record is enabled, start recording immediately
        if args.auto_record:
            command_queue.put(("record_start", None))
        listener_thread = threading.Thread(target=keyboard_listener, daemon=True, name="keyboard_listener")
        listener_thread.start()

    LOGGER.info("Starting control loop. Press Ctrl+C to stop.")

    timestamp_worker_done = False
    markers: list = []
    mts: list = []

    try:
        while not timestamp_worker_done:
            if should_stop:
                stop_event.set()

            # surface worker failures to caller
            try:
                worker_name, worker_exc = worker_errors.get_nowait()
                raise RuntimeError(f"{worker_name} failed: {worker_exc}") from worker_exc
            except queue.Empty:
                pass

            # pull any available stimulus markers (non-blocking)
            if stim_inlet is not None:
                try:
                    markers, mts = stim_inlet.pull_chunk(
                        timeout=0.0,
                        max_samples=int(args.max_chunk_samples),
                    )
                except Exception:
                    markers, mts = [], []
            else:
                markers, mts = [], []

            # process any stimulus markers, even when not currently recording
            for m, ts in zip(markers or [], mts or []):
                LOGGER.debug("stimulus marker chunk: %s @%s", m, ts)
                # allow external commands via markers
                try:
                    label = m[0] if isinstance(m, (list, tuple)) and m else m
                    text_label = str(label)
                    marker_rel_ts = float(ts) - experiment_start_lsl
                    if text_label in ("record_start", "record_stop"):
                        LOGGER.info("received command marker: %s", text_label)
                    if text_label == "record_start":
                        command_queue.put(("record_start", marker_rel_ts))
                        # after starting, skip annotation logic for this marker
                        continue
                    if text_label == "record_stop":
                        command_queue.put(("record_stop", marker_rel_ts))
                        continue
                except Exception:
                    text_label = None

                # if we're recording, also treat normal markers as annotations
                if recording and edf_writer is not None:
                    try:
                        # regular stimulus annotation
                        label = m[0] if isinstance(m, (list, tuple)) and m else m
                        marker_rel_ts = float(ts) - experiment_start_lsl
                        marker_file_rel_ts = max(0.0, float(marker_rel_ts - recording_start_rel))
                        edf_writer.writeAnnotation(marker_file_rel_ts, 0.0, str(label))
                    except Exception:
                        pass

            # apply queued commands in-order on the main thread
            while True:
                try:
                    cmd_item = command_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(cmd_item, tuple):
                    cmd, cmd_rel_ts = cmd_item
                else:
                    cmd, cmd_rel_ts = str(cmd_item), None
                process_command(cmd, cmd_rel_ts)

            consumed_chunk = False
            while True:
                try:
                    item = timed_eeg_queue.get_nowait()
                except queue.Empty:
                    break

                consumed_chunk = True
                if item is timed_sentinel:
                    timestamp_worker_done = True
                    break

                samples, rel_ts = item
                write_record_chunk(samples, rel_ts)

            if not consumed_chunk:
                time.sleep(0.001)
    finally:
        stop_event.set()

        # let workers terminate and emit sentinels
        if eeg_thread.is_alive():
            eeg_thread.join(timeout=2.0)
        if ts_thread.is_alive():
            ts_thread.join(timeout=2.0)

        # drain remaining timestamped chunks so we don't lose buffered data
        while True:
            try:
                item = timed_eeg_queue.get_nowait()
            except queue.Empty:
                break
            if item is timed_sentinel:
                continue
            samples, rel_ts = item
            write_record_chunk(samples, rel_ts)

        stop_recording()

        try:
            inlet.close_stream()
        except Exception:
            pass
        if stim_inlet is not None:
            try:
                stim_inlet.close()
            except Exception:
                pass

    LOGGER.info("Stopping EEG listener.")
    return 0


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        LOGGER.error("Bridge failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
