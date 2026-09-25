"""Predict the finishing order of a Grand Prix from pre-race data only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from fastf1.ergast import Ergast

from ml.config import ENCODERS_JSON, METADATA_JSON, MODEL_PATH
from ml.encoders import FeatureEncoders
from ml.ergast_client import qualifying_results
from ml.features import (
    FeatureBuilder,
    SeasonLedger,
    build_race_rows,
    load_season,
)
from ml.model import RaceOutcomeModel

import f1_data

ENTRY_SOURCE_LABELS = {
    "qualifying": "Qualifying results (real starting grid)",
    "standings": "Current championship entry list (pre-qualifying)",
    "last_race": "Field from the most recent race (pre-qualifying)",
    "previous_season": "Field from the end of last season (pre-qualifying)",
}


class RacePredictor:
    """Loads the trained model and orders the field for one race."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.encoders = FeatureEncoders().load()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self._season_cache: dict[int, list[dict[str, Any]]] = {}

    def _load_model(self) -> RaceOutcomeModel:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Run: python -m ml.train"
            )
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if checkpoint.get("num_numeric") != self.encoders.num_numeric:
            raise RuntimeError(
                "The saved model was trained on a different feature set. "
                "Re-run: python -m ml.collect_data && python -m ml.train"
            )
        model = RaceOutcomeModel(
            num_drivers=checkpoint["num_drivers"],
            num_teams=checkpoint["num_teams"],
            num_circuits=checkpoint["num_circuits"],
            num_numeric=checkpoint["num_numeric"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def _season(self, ergast: Ergast, year: int) -> list[dict[str, Any]]:
        if year not in self._season_cache:
            try:
                self._season_cache[year] = load_season(ergast, year)
            except Exception:
                self._season_cache[year] = []
        return self._season_cache[year]

    def build_race_features(self, year: int, round_num: int) -> tuple[pd.DataFrame, str]:
        race_info = f1_data.get_race_info(year, round_num)
        if not race_info:
            raise ValueError(f"No race found for {year} round {round_num}")

        ergast = Ergast()
        season = self._season(ergast, year)
        completed = [r for r in season if r["round"] < round_num]

        # Rebuild the season exactly the way collection did, so form, circuit
        # history, and the championship table all line up with training.
        builder = FeatureBuilder()
        ledger = SeasonLedger(self._prior_constructor_positions(ergast, year))
        for race in completed:
            builder.update_after_race(race["circuit_id"], race["entries"])
            ledger.add_race(race)

        driver_map, team_map = ledger.standings()
        entries, source = self._resolve_entry_list(
            ergast, year, round_num, driver_map, team_map, completed
        )
        if not entries:
            raise ValueError(
                "No pre-race entry list is available for this Grand Prix yet. It needs "
                "published qualifying, a completed race this season, or last season's field."
            )

        rows = build_race_rows(
            round_num=round_num,
            total_rounds=self._total_rounds(year, season, round_num),
            circuit_id=self._circuit_id(race_info, season, round_num),
            entries=entries,
            builder=builder,
            driver_map=driver_map,
            team_map=team_map,
        )
        by_driver = {e["driver_code"]: e for e in entries}
        for row in rows:
            row["listed_grid"] = by_driver[row["driver_code"]].get("grid_position")

        return pd.DataFrame(rows), source

    def _prior_constructor_positions(self, ergast: Ergast, year: int) -> dict[str, int]:
        """Last season's constructor order, used before this season has data."""
        previous = self._season(ergast, year - 1)
        if not previous:
            return {}
        ledger = SeasonLedger()
        for race in previous:
            ledger.add_race(race)
        return ledger.final_constructor_positions()

    def _total_rounds(self, year: int, season: list[dict], round_num: int) -> int:
        scheduled = len(f1_data.get_races_for_year(year))
        completed = max((r["round"] for r in season), default=0)
        return max(scheduled, completed, round_num)

    def _circuit_id(self, race_info: dict, season: list[dict], round_num: int) -> str:
        """Match the circuit key used during training (the Ergast locality)."""
        for race in season:
            if race["round"] == round_num:
                return race["circuit_id"]
        return str(race_info.get("location") or race_info.get("country") or "unknown")

    def _resolve_entry_list(self, ergast, year, round_num, driver_map, team_map, completed):
        """Best available pre-race field, preferring the most informative source.

        The last race's field beats the championship table as a line-up: the
        table also lists drivers who have since been replaced.
        """
        entries = self._from_qualifying(ergast, year, round_num)
        if entries:
            field, _ = self._pre_race_field(ergast, year, driver_map, team_map, completed)
            return self._fill_empty_seats(entries, field), "qualifying"

        return self._pre_race_field(ergast, year, driver_map, team_map, completed)

    def _pre_race_field(self, ergast, year, driver_map, team_map, completed):
        """Line-up to fall back on when qualifying has not been published."""
        if completed:
            return self._from_race(completed[-1]), "last_race"

        entries = self._from_standings(driver_map, team_map)
        if entries:
            return entries, "standings"

        previous = self._season(ergast, year - 1)
        if previous:
            return self._from_race(previous[-1]), "previous_season"

        return [], "none"

    @staticmethod
    def _fill_empty_seats(
        entries: list[dict[str, Any]], fallback: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add drivers the qualifying sheet is missing, two cars per team at most.

        Ergast occasionally publishes an incomplete qualifying classification —
        a driver who set no time is simply absent. Capping at two entries per
        constructor restores those cars without also re-adding a driver whose
        seat has since been taken by someone who did qualify.
        """
        listed = {e["driver_code"] for e in entries}
        seats: dict[str, int] = {}
        for entry in entries:
            seats[entry["team_id"]] = seats.get(entry["team_id"], 0) + 1

        filled = list(entries)
        for entry in fallback:
            if entry["driver_code"] in listed or seats.get(entry["team_id"], 0) >= 2:
                continue
            filled.append({**entry, "grid_position": None})
            listed.add(entry["driver_code"])
            seats[entry["team_id"]] = seats.get(entry["team_id"], 0) + 1
        return filled

    def _from_qualifying(self, ergast, year, round_num) -> list[dict[str, Any]]:
        """Qualifying happens before the race, so its order is fair to use."""
        df = qualifying_results(ergast, year, round_num)
        if df is None or df.empty:
            return []

        entries = []
        for _, row in df.iterrows():
            code = str(row.get("driverCode", "")).strip()
            team_id = str(row.get("constructorId", "")).strip()
            if not code or not team_id:
                continue
            entries.append({
                "driver_code": code,
                "driver_name": f"{row.get('givenName', '')} {row.get('familyName', '')}".strip() or code,
                "team_id": team_id,
                "team_name": str(row.get("constructorName", team_id)),
                "grid_position": _safe_int(row.get("position")),
            })
        return entries

    def _from_standings(self, driver_map, team_map) -> list[dict[str, Any]]:
        entries = []
        for code, info in driver_map.items():
            team_id = info.get("constructor_id", "")
            if not code or not team_id:
                continue
            entries.append({
                "driver_code": code,
                "driver_name": info.get("driver_name") or code,
                "team_id": team_id,
                "team_name": info.get("constructor_name")
                or team_map.get(team_id, {}).get("name")
                or team_id,
                "grid_position": None,
            })
        return entries

    def _from_race(self, race: dict) -> list[dict[str, Any]]:
        return [
            {
                "driver_code": e["driver_code"],
                "driver_name": e["driver_name"],
                "team_id": e["team_id"],
                "team_name": e["team_name"],
                "grid_position": None,
            }
            for e in race["entries"]
        ]

    def predict_race(self, year: int, round_num: int) -> dict[str, Any]:
        df, source = self.build_race_features(year, round_num)
        if df.empty:
            return {"predictions": [], "grid_known": False, "source": source, "source_label": ""}

        d_idx, t_idx, c_idx, numeric = self.encoders.transform(df)
        anchor = df["grid_position"].to_numpy(dtype=np.float32)
        with torch.no_grad():
            scores = self.model(
                torch.from_numpy(d_idx).to(self.device),
                torch.from_numpy(t_idx).to(self.device),
                torch.from_numpy(c_idx).to(self.device),
                torch.from_numpy(numeric).to(self.device),
                torch.from_numpy(anchor).to(self.device),
            ).cpu().numpy()

        order = np.argsort(scores, kind="stable")
        podium_odds = _podium_probability(scores)
        grid_known = source == "qualifying"

        predictions = []
        for rank, idx in enumerate(order, start=1):
            row = df.iloc[idx]
            grid = _safe_int(row.get("listed_grid")) if grid_known else None
            predictions.append({
                "predicted_position": rank,
                "expected_position": round(float(scores[idx]), 2),
                "podium_chance": round(float(podium_odds[idx]) * 100, 1),
                "driver_code": row["driver_code"],
                "driver_name": row["driver_name"],
                "team_name": f1_data.team_display_name(row["team_id"], row["team_name"]),
                "team_id": row["team_id"],
                "team_color": f1_data.team_color(row["team_id"], year),
                "team_logo": f1_data.team_logo_url(row["team_id"], year),
                "grid_position": grid,
                "grid_delta": (grid - rank) if grid else None,
            })

        return {
            "predictions": predictions,
            "grid_known": grid_known,
            "source": source,
            "source_label": ENTRY_SOURCE_LABELS.get(source, source),
        }


def model_status() -> dict[str, Any]:
    ready = MODEL_PATH.exists() and ENCODERS_JSON.exists()
    meta: dict[str, Any] = {}
    if METADATA_JSON.exists():
        try:
            meta = json.loads(METADATA_JSON.read_text())
        except json.JSONDecodeError:
            meta = {}
    return {"ready": ready, "model_path": str(MODEL_PATH), "metadata": meta}


def get_predictor() -> Optional[RacePredictor]:
    try:
        return RacePredictor()
    except (FileNotFoundError, RuntimeError, OSError):
        return None


def _podium_probability(scores: np.ndarray, temperature: float = 1.5) -> np.ndarray:
    """Soft read of how confident the ordering is at the front of the field.

    A score is an expected finishing position, so the gaps between the leading
    scores say how clear-cut the model thinks the front runners are.
    """
    scaled = -np.asarray(scores, dtype=float) / temperature
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum()


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
