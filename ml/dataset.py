"""Race-grouped tensors.

The model is scored per race rather than per driver row, so batches are whole
races. That lets training optimise the *order* within a race instead of only
the absolute position of each driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ml.config import TARGET_COLUMN
from ml.encoders import FeatureEncoders


class RaceGroups:
    """Encoded tensors for every race in a frame."""

    def __init__(self, df: pd.DataFrame, encoders: FeatureEncoders) -> None:
        self.df = df.reset_index(drop=True)
        d_idx, t_idx, c_idx, numeric = encoders.transform(self.df)
        targets = self.df[TARGET_COLUMN].astype(np.float32).to_numpy()

        self.groups: list[dict[str, torch.Tensor]] = []
        for _, grp in self.df.groupby(["year", "round"], sort=True):
            idx = grp.index.to_numpy()
            if len(idx) < 2:
                continue
            self.groups.append({
                "driver_idx": torch.from_numpy(d_idx[idx]),
                "team_idx": torch.from_numpy(t_idx[idx]),
                "circuit_idx": torch.from_numpy(c_idx[idx]),
                "numeric": torch.from_numpy(numeric[idx]),
                "target": torch.from_numpy(targets[idx]),
                # Unscaled starting slot the model corrects from.
                "anchor": torch.from_numpy(
                    self.df.loc[idx, "grid_position"].to_numpy(dtype=np.float32)
                ),
            })

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def shuffled(self, rng: np.random.Generator):
        order = rng.permutation(len(self.groups))
        for i in order:
            yield self.groups[i]


def chronological_split(df: pd.DataFrame, val_fraction: float):
    """Hold out the most recent races so validation never sees the future."""
    races = (
        df[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .to_numpy()
    )
    n_val = max(1, int(len(races) * val_fraction))
    val_races = {tuple(r) for r in races[-n_val:]}
    mask = df.apply(lambda row: (row["year"], row["round"]) in val_races, axis=1)
    return df[~mask].copy(), df[mask].copy()
