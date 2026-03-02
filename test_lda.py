#!/usr/bin/env python3
"""Simple smoke test for the online LDA/CSP model pickle."""

from __future__ import annotations

import argparse
import numpy as np

from online_windowing import (
    align_channels_for_model,
    infer_model_channel_count,
    load_prediction_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for model_Data28_BIS.pkl")
    parser.add_argument("--model-path", type=str, default="model_Data28_BIS.pkl", help="Model pickle path")
    parser.add_argument("--channels", type=int, default=8, help="Input channel count before model alignment")
    parser.add_argument("--samples", type=int, default=256, help="Samples per window")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rng = np.random.default_rng(int(args.seed))

    model = load_prediction_model(args.model_path)
    model_channels = infer_model_channel_count(model)
    print(f"Loaded model: {type(model).__name__}, expected_channels={model_channels}")

    x = rng.standard_normal((int(args.batch), int(args.channels), int(args.samples)))
    x_aligned = np.empty(
        (
            int(args.batch),
            int(model_channels or int(args.channels)),
            int(args.samples),
        ),
        dtype=np.float64,
    )
    for i in range(int(args.batch)):
        x_aligned[i] = align_channels_for_model(x[i], model_channels)

    y = model.predict(x_aligned)
    print("Predictions:", y)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x_aligned)
        print("Probabilities shape:", p.shape)
        print("Max confidence per sample:", np.max(p, axis=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
