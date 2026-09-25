#!/usr/bin/env python3
"""Train the finishing-order model.

Usage (from the project root, after collect_data):
    python -m ml.train
    python -m ml.train --epochs 150

Training optimises two things at once: how close a score is to the real
finishing position, and whether each pair of drivers is put in the right order.
Half of the training races are duplicated with the qualifying grid hidden, so
the model stays calibrated for Grands Prix that have not qualified yet.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GRID_MASK_FRACTION,
    LEARNING_RATE,
    METADATA_JSON,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    TRAINING_CSV,
    VAL_FRACTION,
    WEIGHT_DECAY,
)
from ml.dataset import RaceGroups, chronological_split
from ml.encoders import FeatureEncoders
from ml.features import mask_grid, validate_dataset
from ml.metrics import race_metrics, summarise
from ml.model import RaceOutcomeModel


@dataclass
class TrainedModel:
    model: RaceOutcomeModel
    encoders: FeatureEncoders
    best_epoch: int
    post_quali: dict
    pre_quali: dict
    grid_baseline: dict
    pace_prior_baseline: dict
    history: list[dict]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Penalise every pair of entries that is ordered the wrong way round.

    For a pair where i really finished ahead of j, the score for i must come
    out lower, so the penalty grows as scores[i] - scores[j] rises.
    """
    score_gap = scores.unsqueeze(1) - scores.unsqueeze(0)
    target_gap = targets.unsqueeze(1) - targets.unsqueeze(0)
    mask = target_gap < 0
    if not mask.any():
        return scores.sum() * 0.0
    return torch.nn.functional.softplus(score_gap[mask]).mean()


def augment_with_masked_grids(train_df: pd.DataFrame, mask_fraction: float, seed: int) -> pd.DataFrame:
    """Original rows plus a grid-hidden copy of a sample of races."""
    if mask_fraction <= 0:
        return train_df

    rng = np.random.default_rng(seed)
    races = train_df[["year", "round"]].drop_duplicates().to_numpy()
    n_masked = int(len(races) * mask_fraction)
    if n_masked == 0:
        return train_df

    chosen = {tuple(r) for r in races[rng.permutation(len(races))[:n_masked]]}
    subset = train_df[train_df.apply(lambda r: (r["year"], r["round"]) in chosen, axis=1)]
    masked = mask_grid(subset)
    # Shift the copies into their own round numbers so a single race is never
    # split across both regimes inside one batch.
    masked["round"] = masked["round"] + 1000
    return pd.concat([train_df, masked], ignore_index=True)


def _score(model, group, device) -> torch.Tensor:
    return model(
        group["driver_idx"].to(device),
        group["team_idx"].to(device),
        group["circuit_idx"].to(device),
        group["numeric"].to(device),
        group["anchor"].to(device),
    )


@torch.no_grad()
def evaluate(model, groups: RaceGroups, device) -> dict:
    model.eval()
    return summarise([
        race_metrics(_score(model, g, device).cpu().numpy(), g["target"].numpy())
        for g in groups
    ])


def baseline_metrics(groups: RaceGroups) -> dict:
    """How well the starting order alone predicts the finish."""
    return summarise([
        race_metrics(g["anchor"].numpy(), g["target"].numpy()) for g in groups
    ])


def train_one_epoch(model, groups, optimizer, device, rng, rank_weight) -> float:
    model.train()
    regression = nn.SmoothL1Loss()
    total = 0.0
    n = 0

    for group in groups.shuffled(rng):
        target = group["target"].to(device)
        optimizer.zero_grad()
        scores = _score(model, group, device)
        loss = regression(scores, target) + rank_weight * pairwise_ranking_loss(scores, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        n += 1

    return total / max(n, 1)


def fit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    rank_weight: float = 1.0,
    mask_fraction: float = GRID_MASK_FRACTION,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> TrainedModel:
    """Fit one model and keep the epoch that ranked the validation races best."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    augmented = augment_with_masked_grids(train_df, mask_fraction, seed)
    encoders = FeatureEncoders().fit(augmented)

    train_groups = RaceGroups(augmented, encoders)
    val_post = RaceGroups(val_df, encoders)
    val_pre = RaceGroups(mask_grid(val_df), encoders)

    model = RaceOutcomeModel(
        num_drivers=encoders.num_drivers,
        num_teams=encoders.num_teams,
        num_circuits=encoders.num_circuits,
        num_numeric=encoders.num_numeric,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_score = -np.inf
    best_state = None
    best: dict = {}
    history: list[dict] = []
    waited = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_groups, optimizer, device, rng, rank_weight)
        post = evaluate(model, val_post, device)
        pre = evaluate(model, val_pre, device)

        # A model is only useful if it ranks well both after and before
        # qualifying, and podium overlap is steadier than winner hit on a
        # validation set this small.
        score = (post["spearman"] + pre["spearman"]) / 2 + 0.25 * (
            post["top3_overlap"] + pre["top3_overlap"]
        ) / 2
        scheduler.step(score)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "post_quali_spearman": post["spearman"],
            "pre_quali_spearman": pre["spearman"],
            "post_quali_top3": post["top3_overlap"],
            "pre_quali_top3": pre["top3_overlap"],
        })

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(
                f"epoch {epoch:3d} | loss {train_loss:.4f} | "
                f"after quali rho {post['spearman']:.3f} podium {post['top3_overlap']:.0%} | "
                f"before quali rho {pre['spearman']:.3f} podium {pre['top3_overlap']:.0%}"
            )

        if score > best_score:
            best_score = score
            best = {"epoch": epoch, "post": post, "pre": pre}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainedModel(
        model=model,
        encoders=encoders,
        best_epoch=best.get("epoch", 0),
        post_quali=best.get("post", {}),
        pre_quali=best.get("pre", {}),
        grid_baseline=baseline_metrics(val_post),
        pace_prior_baseline=baseline_metrics(val_pre),
        history=history,
    )


def _report(label: str, metrics: dict) -> str:
    return (
        f"  {label:<19}: order accuracy {metrics.get('order_accuracy', 0):.1%} | "
        f"rho {metrics.get('spearman', 0):.3f} | "
        f"podium hit {metrics.get('top3_overlap', 0):.1%} | "
        f"winner {metrics.get('winner_hit', 0):.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the F1 finishing-order model")
    parser.add_argument("--data", type=Path, default=TRAINING_CSV)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--mask-fraction", type=float, default=GRID_MASK_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Training data not found: {args.data}")
        print("Run first:  python -m ml.collect_data")
        sys.exit(1)

    df = validate_dataset(pd.read_csv(args.data))
    train_df, val_df = chronological_split(df, args.val_fraction)

    print(f"Races: {df.groupby(['year','round']).ngroups} "
          f"(train {train_df.groupby(['year','round']).ngroups}, "
          f"val {val_df.groupby(['year','round']).ngroups})")
    print(f"Rows: {len(train_df)} training rows before augmentation\n")

    result = fit(
        train_df,
        val_df,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        rank_weight=args.rank_weight,
        mask_fraction=args.mask_fraction,
        seed=args.seed,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    result.encoders.save()
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "epoch": result.best_epoch,
            "num_drivers": result.encoders.num_drivers,
            "num_teams": result.encoders.num_teams,
            "num_circuits": result.encoders.num_circuits,
            "num_numeric": result.encoders.num_numeric,
        },
        MODEL_PATH,
    )

    backtest = {}
    backtest_path = MODELS_DIR / "backtest.json"
    if backtest_path.exists():
        try:
            backtest = json.loads(backtest_path.read_text())
        except json.JSONDecodeError:
            backtest = {}

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": sorted(int(y) for y in df["year"].unique()),
        "train_races": int(train_df.groupby(["year", "round"]).ngroups),
        "val_races": int(val_df.groupby(["year", "round"]).ngroups),
        "best_epoch": result.best_epoch,
        "post_quali": result.post_quali,
        "pre_quali": result.pre_quali,
        "grid_baseline": result.grid_baseline,
        "pace_prior_baseline": result.pace_prior_baseline,
        "backtest": backtest,
        "history": result.history,
        "model_path": str(MODEL_PATH),
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved {MODEL_PATH}")
    print("Validation (the most recent races, never seen in training)")
    print(_report("after qualifying", result.post_quali))
    print(_report("before qualifying", result.pre_quali))
    print("Baselines")
    print(_report("grid order", result.grid_baseline))
    print(_report("car-strength order", result.pace_prior_baseline))
    print("\nFor a multi-window check run:  python -m ml.backtest")


if __name__ == "__main__":
    main()
