#!/usr/bin/env python3
"""Build the training set of pre-race rows.

Usage (from the project root):
    python -m ml.collect_data
    python -m ml.collect_data --seasons 2022 2023 2024 2025

Results come from the Ergast archive, which carries finishing position, grid,
team, and status for every race, so a full rebuild takes about a minute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from fastf1.ergast import Ergast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.config import (
    COLLECT_SEASONS,
    DATASET_META_JSON,
    PROCESSED_DIR,
    RAW_DIR,
    TRAINING_CSV,
)
from ml.features import (
    FeatureBuilder,
    SeasonLedger,
    build_race_rows,
    load_season,
    validate_dataset,
)


def collect_season(
    year: int,
    ergast: Ergast,
    prior_constructor_positions: dict[str, int],
    verbose: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """One row per driver per race, plus the season's final constructor order.

    Form history is kept per season: a driver's "last three races" resets in
    January, which is how a real pre-season preview would read it.
    """
    races = load_season(ergast, year)
    if not races:
        if verbose:
            print("  no results available")
        return [], prior_constructor_positions

    total_rounds = max(r["round"] for r in races)
    builder = FeatureBuilder()
    ledger = SeasonLedger(prior_constructor_positions)
    rows: list[dict] = []

    for race in races:
        driver_map, team_map = ledger.standings()
        race_rows = build_race_rows(
            round_num=race["round"],
            total_rounds=total_rounds,
            circuit_id=race["circuit_id"],
            entries=race["entries"],
            builder=builder,
            driver_map=driver_map,
            team_map=team_map,
        )

        by_driver = {e["driver_code"]: e for e in race["entries"]}
        for row in race_rows:
            entry = by_driver[row["driver_code"]]
            row.update({
                "year": year,
                "round": race["round"],
                "event_name": race["event_name"],
                "country": race["country"],
                "finish_position": entry["finish_position"],
                "status": entry["status"],
                "dnf": bool(entry["dnf"]),
            })

        rows.extend(race_rows)
        builder.update_after_race(race["circuit_id"], race["entries"])
        ledger.add_race(race)

        if verbose:
            print(f"  R{race['round']:>2}  {race['event_name'][:38]:<38} {len(race_rows):>2} entries")

    return rows, ledger.final_constructor_positions()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the F1 pre-race training set")
    parser.add_argument("--seasons", nargs="+", type=int, default=COLLECT_SEASONS)
    parser.add_argument("--output", type=Path, default=TRAINING_CSV)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    verbose = not args.quiet
    print("F1 ML — data collection")
    print(f"Seasons: {args.seasons}\n")

    ergast = Ergast()
    all_rows: list[dict] = []
    prior_positions: dict[str, int] = {}

    for year in sorted(args.seasons):
        print(f"{year}")
        try:
            season_rows, prior_positions = collect_season(
                year, ergast, prior_positions, verbose=verbose
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        all_rows.extend(season_rows)
        print(f"  {len(season_rows)} rows\n")

    if not all_rows:
        print("No data collected. Check your network connection.")
        sys.exit(1)

    df = validate_dataset(pd.DataFrame(all_rows))
    # Ordered by driver, never by result: rows sorted by finishing position let
    # a tie-break in any later ranking quietly leak the answer.
    df = df.sort_values(["year", "round", "driver_code"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    meta = {
        "rows": len(df),
        "seasons": sorted(int(y) for y in df["year"].unique()),
        "races": int(df.groupby(["year", "round"]).ngroups),
        "drivers": int(df["driver_code"].nunique()),
        "teams": int(df["team_id"].nunique()),
        "circuits": int(df["circuit_id"].nunique()),
        "grid_coverage": round(float(df["has_grid"].mean()), 4),
        "columns": list(df.columns),
    }
    DATASET_META_JSON.write_text(json.dumps(meta, indent=2))

    print(f"Saved {len(df)} rows across {meta['races']} races -> {args.output}")
    print(f"Seasons: {meta['seasons']}")


if __name__ == "__main__":
    main()
