#!/usr/bin/env python3
"""Publish a synthetic LSL EEG stream for local testing."""

from __future__ import annotations

import argparse
import math
import signal
import time
from typing import List

import numpy as np
from pylsl import StreamInfo, StreamOutlet


DEFAULT_LABELS = [
    "F3",
    "F4",
    "C3",
    "Cz",
    "C4",
    "P3",
    "P4",
    "Oz",
    "Fp1",
    "Fp2",
    "T7",
    "T8",
    "O1",
    "O2",
    "Pz",
    "Fz",
]


def build_labels(channel_count: int) -> List[str]:
    labels = DEFAULT_LABELS[:channel_count]
    while len(labels) < channel_count:
        labels.append(f"Ch{len(labels) + 1}")
    return labels


def build_stream_info(
    name: str,
    stream_type: str,
    source_id: str,
    channel_count: int,
    sample_rate: float,
) -> StreamInfo:
    info = StreamInfo(
        name=name,
        type=stream_type,
        channel_count=channel_count,
        nominal_srate=sample_rate,
        channel_format="float32",
        source_id=source_id,
    )
    channels = info.desc().append_child("channels")
    for label in build_labels(channel_count):
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "uV")
        ch.append_child_value("type", "EEG")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic LSL EEG generator")
    parser.add_argument("--stream-name", type=str, default="FakeEEG", help="LSL stream name")
    parser.add_argument("--stream-type", type=str, default="EEG", help="LSL stream type")
    parser.add_argument("--source-id", type=str, default="fake_eeg_001", help="LSL source_id")
    parser.add_argument("--channels", type=int, default=8, help="Number of EEG channels")
    parser.add_argument("--sample-rate", type=float, default=250.0, help="Sampling rate in Hz")
    parser.add_argument("--chunk-size", type=int, default=16, help="Samples per pushed chunk")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=4.0,
        help="Gaussian noise std in microvolts",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 means run until Ctrl+C)",
    )
    args = parser.parse_args()

    if args.channels <= 0:
        raise ValueError("--channels must be > 0")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    info = build_stream_info(
        name=args.stream_name,
        stream_type=args.stream_type,
        source_id=args.source_id,
        channel_count=int(args.channels),
        sample_rate=float(args.sample_rate),
    )
    outlet = StreamOutlet(info, chunk_size=int(args.chunk_size), max_buffered=360)

    print(
        "Started fake LSL stream: "
        f"name='{args.stream_name}' type='{args.stream_type}' source_id='{args.source_id}' "
        f"channels={args.channels} sample_rate={args.sample_rate:.1f}Hz"
    )
    print("Press Ctrl+C to stop.")

    running = True

    def stop_handler(_sig: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, stop_handler)
        except Exception:
            pass

    rng = np.random.default_rng()
    freqs = np.linspace(8.0, 14.0, int(args.channels), dtype=np.float64)
    phase_offsets = np.linspace(0.0, math.pi, int(args.channels), dtype=np.float64)
    t0 = time.perf_counter()
    next_tick = t0
    chunk_dt = float(args.chunk_size) / float(args.sample_rate)

    while running:
        elapsed = time.perf_counter() - t0
        if args.duration_seconds > 0 and elapsed >= args.duration_seconds:
            break

        t = elapsed + np.arange(int(args.chunk_size), dtype=np.float64) / float(args.sample_rate)
        chunk = np.zeros((int(args.chunk_size), int(args.channels)), dtype=np.float32)

        # Synthetic EEG-like mixture: alpha rhythm + slow drift + noise.
        slow = 12.0 * np.sin(2.0 * math.pi * 0.5 * t)[:, None]
        for ch in range(int(args.channels)):
            alpha = 25.0 * np.sin(2.0 * math.pi * freqs[ch] * t + phase_offsets[ch])
            noise = rng.normal(0.0, float(args.noise_std), size=t.shape[0])
            chunk[:, ch] = (alpha + slow[:, 0] + noise).astype(np.float32)

        outlet.push_chunk(chunk.tolist())

        next_tick += chunk_dt
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

    print("Fake LSL stream stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
