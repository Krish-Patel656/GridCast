#!/usr/bin/env python3
"""Walk-forward backtest.

Usage (from the project root):
    python -m ml.backtest
    python -m ml.backtest --windows 5

A single hold-out split of ~40 races is small enough that one lucky weekend
moves the numbers by several points. This retrains the model from scratch for
several consecutive windows of races, always training only on races that came
earlier, and averages the result. The summary it writes is what the app shows
as the model's accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import MODELS_DIR, RANDOM_SEED, TRAINING_CSV
from ml.features import validate_dataset
from ml.train import fit

BACKTEST_JSON = MODELS_DIR / "backtest.json"


def race_keys(df: pd.DataFrame) -> np.ndarray:
    return (
        df[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .to_numpy()
    )


def slice_races(df: pd.DataFrame, keys) -> pd.DataFrame:
    wanted = {tuple(k) for k in keys}
    mask = df.apply(lambda row: (row["year"], row["round"]) in wanted, axis=1)
    return df[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the F1 model")
    parser.add_argument("--data", type=Path, default=TRAINING_CSV)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Training data not found: {args.data}. Run: python -m ml.collect_data")
        sys.exit(1)

    df = validate_dataset(pd.read_csv(args.data))
    keys = race_keys(df)
    total_eval = args.windows * args.window_size
    if len(keys) <= total_eval + 40:
        print(f"Not enough races ({len(keys)}) for {args.windows} windows of {args.window_size}.")
        sys.exit(1)

    print(f"Backtest: {args.windows} windows of {args.window_size} races "
          f"({len(keys)} races available)\n")

    folds = []
    for w in range(args.windows):
        end = len(keys) - (args.windows - 1 - w) * args.window_size
        start = end - args.window_size
        train_keys, val_keys = keys[:start], keys[start:end]

        train_df = slice_races(df, train_keys)
        val_df = slice_races(df, val_keys)
        label = f"{val_keys[0][0]} R{val_keys[0][1]} to {val_keys[-1][0]} R{val_keys[-1][1]}"

        result = fit(train_df, val_df, seed=args.seed + w, verbose=False)
        folds.append({
            "window": label,
            "train_races": len(train_keys),
            "val_races": len(val_keys),
            "post_quali": result.post_quali,
            "pre_quali": result.pre_quali,
            "grid_baseline": result.grid_baseline,
            "pace_prior_baseline": result.pace_prior_baseline,
        })

        print(f"{label:<26} "
              f"after quali {result.post_quali.get('order_accuracy', 0):.1%} "
              f"(grid {result.grid_baseline.get('order_accuracy', 0):.1%}) | "
              f"before quali {result.pre_quali.get('order_accuracy', 0):.1%} "
              f"(car strength {result.pace_prior_baseline.get('order_accuracy', 0):.1%})")

    def mean_of(section: str) -> dict:
        keys_ = folds[0][section].keys()
        return {k: round(float(np.mean([f[section][k] for f in folds])), 4) for k in keys_}

    summary = {
        "windows": args.windows,
        "window_size": args.window_size,
        "races_evaluated": total_eval,
        "post_quali": mean_of("post_quali"),
        "pre_quali": mean_of("pre_quali"),
        "grid_baseline": mean_of("grid_baseline"),
        "pace_prior_baseline": mean_of("pace_prior_baseline"),
        "folds": folds,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_JSON.write_text(json.dumps(summary, indent=2))

    def show(label: str, metrics: dict) -> None:
        print(f"  {label:<19}: order accuracy {metrics['order_accuracy']:.1%} | "
              f"rho {metrics['spearman']:.3f} | podium hit {metrics['top3_overlap']:.1%} | "
              f"winner {metrics['winner_hit']:.1%}")

    print(f"\nAveraged over {total_eval} races")
    show("after qualifying", summary["post_quali"])
    show("grid order", summary["grid_baseline"])
    show("before qualifying", summary["pre_quali"])
    show("car-strength order", summary["pace_prior_baseline"])
    print(f"\nSaved {BACKTEST_JSON}")
    print("Re-run python -m ml.train to fold this into the model metadata.")


if __name__ == "__main__":
    main()
