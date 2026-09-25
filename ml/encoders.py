"""Label encoders and the numeric scaler, persisted alongside the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.config import (
    CATEGORICAL_FEATURES,
    ENCODERS_JSON,
    NUMERIC_FEATURES,
    SCALER_PATH,
)


class FeatureEncoders:
    """Maps categories to embedding indices and standardises numeric columns.

    Index 0 is reserved for unseen categories (rookie drivers, new teams, new
    circuits), which is why every vocabulary is offset by one.
    """

    def __init__(self) -> None:
        self.vocabularies: dict[str, dict[str, int]] = {}
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEncoders":
        for col in CATEGORICAL_FEATURES:
            classes = sorted(df[col].astype(str).unique())
            self.vocabularies[col] = {label: i + 1 for i, label in enumerate(classes)}
        self.scaler.fit(df[NUMERIC_FEATURES].astype(float).to_numpy())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("Encoders not fitted. Call fit() or load() first.")

        encoded = []
        for col in CATEGORICAL_FEATURES:
            vocab = self.vocabularies[col]
            values = df[col].astype(str).map(lambda v: vocab.get(v, 0))
            encoded.append(values.to_numpy(dtype=np.int64))

        numeric = self.scaler.transform(df[NUMERIC_FEATURES].astype(float).to_numpy())
        return encoded[0], encoded[1], encoded[2], numeric.astype(np.float32)

    def save(self, encoders_path: Path = ENCODERS_JSON, scaler_path: Path = SCALER_PATH) -> None:
        encoders_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "numeric_features": NUMERIC_FEATURES,
            "vocabularies": self.vocabularies,
        }
        encoders_path.write_text(json.dumps(payload, indent=2))
        joblib.dump(self.scaler, scaler_path)

    def load(
        self,
        encoders_path: Path = ENCODERS_JSON,
        scaler_path: Path = SCALER_PATH,
    ) -> "FeatureEncoders":
        payload = json.loads(encoders_path.read_text())
        saved_features = payload.get("numeric_features", [])
        if saved_features and saved_features != NUMERIC_FEATURES:
            raise RuntimeError(
                "Saved encoders were built for a different feature set. "
                "Re-run: python -m ml.collect_data && python -m ml.train"
            )
        self.vocabularies = payload["vocabularies"]
        self.scaler = joblib.load(scaler_path)
        self._fitted = True
        return self

    @property
    def num_drivers(self) -> int:
        return len(self.vocabularies["driver_code"]) + 1

    @property
    def num_teams(self) -> int:
        return len(self.vocabularies["team_id"]) + 1

    @property
    def num_circuits(self) -> int:
        return len(self.vocabularies["circuit_id"]) + 1

    @property
    def num_numeric(self) -> int:
        return len(NUMERIC_FEATURES)
