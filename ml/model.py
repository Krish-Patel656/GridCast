"""Scoring network for race finishing order."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.config import DROPOUT, EMBEDDING_DIM, HIDDEN_DIMS, MAX_POSITION_SHIFT


class RaceOutcomeModel(nn.Module):
    """Scores each entry; a lower score means a better predicted finish.

    Starting position already explains most of a race result, so the network
    predicts a *correction* to it rather than the result from scratch. The
    anchor is the real grid slot after qualifying, or the car-strength prior
    before it. The output layer starts at zero, so an untrained model reproduces
    the grid order and training can only improve on it.
    """

    def __init__(
        self,
        num_drivers: int,
        num_teams: int,
        num_circuits: int,
        num_numeric: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dims: list[int] | None = None,
        dropout: float = DROPOUT,
        max_shift: float = MAX_POSITION_SHIFT,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or HIDDEN_DIMS
        self.max_shift = max_shift

        self.driver_emb = nn.Embedding(num_drivers, embedding_dim, padding_idx=0)
        self.team_emb = nn.Embedding(num_teams, embedding_dim, padding_idx=0)
        self.circuit_emb = nn.Embedding(num_circuits, embedding_dim, padding_idx=0)

        layers: list[nn.Module] = []
        prev = embedding_dim * 3 + num_numeric
        for hidden in hidden_dims:
            layers += [
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = hidden

        head = nn.Linear(prev, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        driver_idx: torch.Tensor,
        team_idx: torch.Tensor,
        circuit_idx: torch.Tensor,
        numeric: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            [
                self.driver_emb(driver_idx),
                self.team_emb(team_idx),
                self.circuit_emb(circuit_idx),
                numeric,
            ],
            dim=1,
        )
        shift = torch.tanh(self.mlp(features).squeeze(-1)) * self.max_shift
        return anchor + shift
