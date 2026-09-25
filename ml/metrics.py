"""Ranking metrics for a predicted finishing order.

Exact position accuracy is a poor way to judge an F1 prediction: safety cars,
strategy, and retirements move drivers around. What matters is whether the
order is right, so these metrics score the ordering itself.
"""

from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 3:
        return 0.0
    pr, ar = _ranks(pred), _ranks(actual)
    if pr.std() == 0 or ar.std() == 0:
        return 0.0
    return float(np.corrcoef(pr, ar)[0, 1])


def race_metrics(pred_scores: np.ndarray, actual_positions: np.ndarray) -> dict:
    """Metrics for a single race.

    order_accuracy is the share of driver pairs put in the correct relative
    order, which is the most intuitive read on "how accurate is this ranking".
    """
    pred_scores = np.asarray(pred_scores, dtype=float)
    actual_positions = np.asarray(actual_positions, dtype=float)
    n = len(pred_scores)
    if n < 2:
        return {}

    pred_rank = _ranks(pred_scores)
    actual_rank = _ranks(actual_positions)

    pred_diff = pred_rank[:, None] - pred_rank[None, :]
    actual_diff = actual_rank[:, None] - actual_rank[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    concordant = ((pred_diff * actual_diff) > 0) & upper
    comparable = (actual_diff != 0) & upper

    top_k = min(3, n)
    pred_top = set(np.argsort(pred_scores, kind="stable")[:top_k])
    actual_top = set(np.argsort(actual_positions, kind="stable")[:top_k])

    top10 = min(10, n)
    pred_points = set(np.argsort(pred_scores, kind="stable")[:top10])
    actual_points = set(np.argsort(actual_positions, kind="stable")[:top10])

    return {
        "order_accuracy": float(concordant.sum() / max(comparable.sum(), 1)),
        "spearman": spearman(pred_scores, actual_positions),
        "top3_overlap": len(pred_top & actual_top) / top_k,
        "top10_overlap": len(pred_points & actual_points) / top10,
        "winner_hit": float(
            np.argsort(pred_scores, kind="stable")[0]
            == np.argsort(actual_positions, kind="stable")[0]
        ),
        "mean_rank_error": float(np.mean(np.abs(pred_rank - actual_rank))),
    }


def summarise(per_race: list[dict]) -> dict:
    """Average each metric across races, skipping races too small to score."""
    valid = [m for m in per_race if m]
    if not valid:
        return {}
    keys = valid[0].keys()
    return {k: round(float(np.mean([m[k] for m in valid])), 4) for k in keys}
