"""Pre-race feature engineering.

Every feature is built from information that exists before lights out: the
championship as it stood after the previous round, prior finishes, circuit
history, and the qualifying grid when it has already been published. Nothing
from the race being predicted is ever read.

Championship tables are accumulated from race and sprint results rather than
requested per round, which keeps a full rebuild inside the Ergast rate limit
and guarantees the table matches the races the feature history has seen.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import pandas as pd

from ml.config import NUMERIC_FEATURES
from ml.ergast_client import season_results, season_sprint_results


def is_dnf(status: str) -> bool:
    """True when a classified result was not a running finish."""
    s = (status or "").strip().lower()
    if not s:
        return False
    return not (s.startswith("finished") or s.startswith("+"))


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_season(ergast, year: int) -> list[dict[str, Any]]:
    """Completed races for a season, oldest first, with points scored."""
    frames, desc = season_results(ergast, year)
    if not frames:
        return []

    sprint_frames, sprint_desc = season_sprint_results(ergast, year)
    sprint_points: dict[int, dict[str, float]] = {}
    for i, frame in enumerate(sprint_frames):
        round_num = int(sprint_desc.iloc[i]["round"])
        per_driver = {}
        for _, row in frame.iterrows():
            code = str(row.get("driverCode", "")).strip()
            if code:
                per_driver[code] = _safe_float(row.get("points"))
        sprint_points[round_num] = per_driver

    races: list[dict[str, Any]] = []
    for i, frame in enumerate(frames):
        meta = desc.iloc[i]
        round_num = int(meta["round"])
        sprints = sprint_points.get(round_num, {})

        entries = []
        for _, row in frame.iterrows():
            code = str(row.get("driverCode", "")).strip()
            team_id = str(row.get("constructorId", "")).strip()
            position = _safe_int(row.get("position"))
            if not code or not team_id or position is None:
                continue
            status = str(row.get("status", ""))
            entries.append({
                "driver_code": code,
                "driver_name": f"{row.get('givenName', '')} {row.get('familyName', '')}".strip() or code,
                "team_id": team_id,
                "team_name": str(row.get("constructorName", team_id)),
                "finish_position": position,
                "grid_position": _safe_int(row.get("grid")),
                "points": _safe_float(row.get("points")) + sprints.get(code, 0.0),
                "status": status,
                "dnf": is_dnf(status),
            })

        if entries:
            races.append({
                "year": int(meta["season"]),
                "round": round_num,
                "event_name": str(meta.get("raceName", f"Round {round_num}")),
                "country": str(meta.get("country", "")),
                "circuit_id": str(meta.get("locality") or meta.get("circuitId") or "unknown"),
                "entries": entries,
            })

    races.sort(key=lambda r: r["round"])
    return races


class SeasonLedger:
    """Running championship tables, advanced one race at a time.

    Round 1 has no current-season table, so the previous season's final
    constructor order seeds the car-strength prior — a legitimate pre-race
    signal, since last year's pecking order is public knowledge.
    """

    def __init__(self, prior_constructor_positions: Optional[dict[str, int]] = None) -> None:
        self.driver_points: dict[str, float] = defaultdict(float)
        self.driver_wins: dict[str, int] = defaultdict(int)
        self.driver_team: dict[str, str] = {}
        self.driver_names: dict[str, str] = {}
        self.team_points: dict[str, float] = defaultdict(float)
        self.team_names: dict[str, str] = {}
        self.prior_constructor_positions = prior_constructor_positions or {}
        self.races_seen = 0

    def standings(self) -> tuple[dict[str, dict], dict[str, dict]]:
        driver_map: dict[str, dict] = {}
        team_map: dict[str, dict] = {}

        ranked_drivers = sorted(
            self.driver_points.items(),
            key=lambda kv: (-kv[1], -self.driver_wins.get(kv[0], 0), kv[0]),
        )
        for position, (code, points) in enumerate(ranked_drivers, start=1):
            driver_map[code] = {
                "points": points,
                "position": float(position),
                "wins": self.driver_wins.get(code, 0),
                "driver_name": self.driver_names.get(code, code),
                "constructor_id": self.driver_team.get(code, ""),
                "constructor_name": self.team_names.get(self.driver_team.get(code, ""), ""),
            }

        ranked_teams = sorted(self.team_points.items(), key=lambda kv: (-kv[1], kv[0]))
        for position, (team_id, points) in enumerate(ranked_teams, start=1):
            team_map[team_id] = {
                "points": points,
                "position": float(position),
                "name": self.team_names.get(team_id, team_id),
            }

        if self.races_seen == 0:
            for team_id, position in self.prior_constructor_positions.items():
                team_map[team_id] = {
                    "points": 0.0,
                    "position": float(position),
                    "name": self.team_names.get(team_id, team_id),
                }

        return driver_map, team_map

    def add_race(self, race: dict[str, Any]) -> None:
        for entry in race["entries"]:
            code = entry["driver_code"]
            team_id = entry["team_id"]
            self.driver_points[code] += entry.get("points", 0.0)
            self.team_points[team_id] += entry.get("points", 0.0)
            self.driver_team[code] = team_id
            self.driver_names[code] = entry.get("driver_name", code)
            self.team_names[team_id] = entry.get("team_name", team_id)
            if entry["finish_position"] == 1:
                self.driver_wins[code] += 1
        self.races_seen += 1

    def final_constructor_positions(self) -> dict[str, int]:
        _, team_map = self.standings()
        return {team_id: int(info["position"]) for team_id, info in team_map.items()}


class FeatureBuilder:
    """Rolling form history, updated only after a race has been consumed."""

    def __init__(self) -> None:
        self.driver_finishes: dict[str, list[int]] = defaultdict(list)
        self.driver_dnfs: dict[str, list[int]] = defaultdict(list)
        self.team_dnfs: dict[str, list[int]] = defaultdict(list)
        self.team_finishes: dict[str, list[int]] = defaultdict(list)
        self.circuit_driver_finishes: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.circuit_team_finishes: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

    @staticmethod
    def _avg(values: list[float], default: float) -> float:
        if not values:
            return default
        return sum(values) / len(values)

    def build_features(
        self,
        *,
        round_num: int,
        total_rounds: int,
        driver_code: str,
        team_id: str,
        circuit_id: str,
        grid_position: Optional[int],
        driver_points_before: float,
        driver_position_before: float,
        constructor_points_before: float,
        constructor_position_before: float,
    ) -> dict[str, Any]:
        # Car quality is the strongest pre-race signal, so when qualifying has
        # not happened yet the constructor's championship rank stands in for an
        # expected starting slot (two cars per team).
        constructor_pos = float(constructor_position_before or 10.0)
        pace_prior = min(constructor_pos * 2.0 - 0.5, 20.0)

        has_grid = grid_position is not None and grid_position > 0
        grid = float(grid_position) if has_grid else pace_prior

        last_3 = self.driver_finishes[driver_code][-3:]
        last_5 = self.driver_finishes[driver_code][-5:]
        dnf_5 = self.driver_dnfs[driver_code][-5:]
        team_dnf_5 = self.team_dnfs[team_id][-10:]
        team_last_3 = self.team_finishes[team_id][-3:]
        circuit_hist = self.circuit_driver_finishes[circuit_id][driver_code][-3:]
        team_circuit_hist = self.circuit_team_finishes[circuit_id][team_id][-3:]

        # With no history yet (season openers, rookies) the grid is the best
        # available guess for where a car belongs.
        fallback = grid

        return {
            "grid_position": grid,
            "has_grid": 1.0 if has_grid else 0.0,
            "pace_prior": pace_prior,
            "driver_points_before": float(driver_points_before or 0.0),
            "driver_position_before": float(driver_position_before or 20.0),
            "constructor_points_before": float(constructor_points_before or 0.0),
            "constructor_position_before": constructor_pos,
            "avg_finish_last_3": self._avg(last_3, fallback),
            "avg_finish_last_5": self._avg(last_5, fallback),
            "team_avg_finish_last_3": self._avg(team_last_3, fallback),
            "circuit_avg_finish": self._avg(circuit_hist, fallback),
            "team_circuit_avg_finish": self._avg(team_circuit_hist, fallback),
            "dnf_rate_last_5": self._avg(dnf_5, 0.1),
            "team_dnf_rate_last_5": self._avg(team_dnf_5, 0.1),
            "season_progress": round_num / max(total_rounds, 1),
        }

    def update_after_race(self, circuit_id: str, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            driver = entry["driver_code"]
            team = entry["team_id"]
            pos = int(entry["finish_position"])
            dnf = 1 if entry.get("dnf") else 0
            self.driver_finishes[driver].append(pos)
            self.team_finishes[team].append(pos)
            self.circuit_driver_finishes[circuit_id][driver].append(pos)
            self.circuit_team_finishes[circuit_id][team].append(pos)
            self.driver_dnfs[driver].append(dnf)
            self.team_dnfs[team].append(dnf)


def build_race_rows(
    *,
    round_num: int,
    total_rounds: int,
    circuit_id: str,
    entries: list[dict[str, Any]],
    builder: FeatureBuilder,
    driver_map: dict[str, dict],
    team_map: dict[str, dict],
) -> list[dict[str, Any]]:
    """Feature rows for one race. Shared by collection and prediction so the
    two can never drift apart."""
    rows = []
    for entry in entries:
        driver_code = entry["driver_code"]
        team_id = entry["team_id"]
        if not driver_code or not team_id:
            continue
        drv_st = driver_map.get(driver_code, {})
        team_st = team_map.get(drv_st.get("constructor_id") or team_id) or team_map.get(team_id, {})

        numeric = builder.build_features(
            round_num=round_num,
            total_rounds=total_rounds,
            driver_code=driver_code,
            team_id=team_id,
            circuit_id=circuit_id,
            grid_position=entry.get("grid_position"),
            driver_points_before=drv_st.get("points", 0.0),
            driver_position_before=drv_st.get("position", 20.0),
            constructor_points_before=team_st.get("points", 0.0),
            constructor_position_before=team_st.get("position", 10.0),
        )
        rows.append({
            "driver_code": driver_code,
            "driver_name": entry.get("driver_name") or driver_code,
            "team_id": team_id,
            "team_name": entry.get("team_name") or team_id,
            "circuit_id": circuit_id,
            **numeric,
        })
    return add_field_relative_features(rows)


FIELD_FEATURES = (
    "driver_points_share",
    "constructor_points_share",
    "grid_percentile",
    "grid_vs_teammate",
)


def field_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features that only mean something relative to the rest of this field.

    Championship totals grow through a season and reset every year, and grids
    have held 20 to 22 cars across the seasons collected, so both are expressed
    relative to the field. Grid against a team-mate isolates a car that is out
    of position from a car that is simply slow.
    """
    max_driver = frame["driver_points_before"].max()
    max_team = frame["constructor_points_before"].max()
    field_size = max(len(frame) - 1, 1)

    out = pd.DataFrame(index=frame.index)
    out["driver_points_share"] = (
        frame["driver_points_before"] / max_driver if max_driver else 0.0
    )
    out["constructor_points_share"] = (
        frame["constructor_points_before"] / max_team if max_team else 0.0
    )
    out["grid_percentile"] = (frame["grid_position"] - 1.0) / field_size
    out["grid_vs_teammate"] = frame["grid_position"] - frame.groupby("team_id")[
        "grid_position"
    ].transform("mean")
    return out


def add_field_relative_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the field-relative features to one race's rows."""
    if not rows:
        return rows
    computed = field_features(pd.DataFrame(rows))
    for i, row in enumerate(rows):
        for col in FIELD_FEATURES:
            row[col] = float(computed.iloc[i][col])
    return rows


FORM_COLUMNS = (
    "avg_finish_last_3",
    "avg_finish_last_5",
    "team_avg_finish_last_3",
    "circuit_avg_finish",
    "team_circuit_avg_finish",
)


def mask_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the rows as they would look before qualifying has run."""
    masked = df.copy()
    for col in FORM_COLUMNS:
        # Rows with no history fell back to the real grid; move that fallback
        # onto the pace prior so the masked copy stays self-consistent.
        fell_back = df[col] == df["grid_position"]
        masked.loc[fell_back, col] = masked.loc[fell_back, "pace_prior"]
    masked["grid_position"] = masked["pace_prior"]
    masked["has_grid"] = 0.0

    # Anything derived from the grid has to be rebuilt off the pace prior.
    grid_derived = ["grid_percentile", "grid_vs_teammate"]
    if "round" in masked.columns:
        for _, group in masked.groupby(["year", "round"]):
            recomputed = field_features(group)
            masked.loc[group.index, grid_derived] = recomputed[grid_derived]
    else:
        masked[grid_derived] = field_features(masked)[grid_derived]
    return masked


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = (
        ["year", "round", "driver_code", "team_id", "circuit_id", "finish_position"]
        + NUMERIC_FEATURES
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}. Re-run: python -m ml.collect_data"
        )

    df = df.copy()
    df = df.dropna(subset=["finish_position", "driver_code", "team_id", "circuit_id"])
    df["finish_position"] = df["finish_position"].astype(int)
    df = df[(df["finish_position"] >= 1) & (df["finish_position"] <= 24)]
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES)
    return df.reset_index(drop=True)
