"""Turn a predicted finishing order into a lap-by-lap race.

The ranking model says where it expects each driver to finish. That is a single
ordering, which is not a race: a race has a start, tyres going off, pit stops,
cars stuck behind cars they are quicker than, and a result that is usually but
not always what was expected.

This module runs one plausible race consistent with the prediction. Driver pace
comes from the model's expected position, then every lap is perturbed, tyres
wear, strategies play out and overtakes have to actually be completed. Re-run it
with a different seed and the race is different; run it many times and the
average lands back on the prediction.

Nothing here feeds the model. It is a presentation layer over its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Pace, as a fraction of the reference lap time.
#
# A second per lap covers most of an F1 grid, so one position of predicted
# ranking is worth a little over a tenth on a 90-second lap.
PACE_PER_POSITION = 0.0013

# How much a driver's race-day form can differ from their predicted level. This
# is what lets the simulation disagree with the model.
FORM_SPREAD = 0.0035

# Lap-to-lap inconsistency for a single driver.
LAP_NOISE = 0.0022

# Cars get lighter as fuel burns off, so lap times fall through the race.
FUEL_GAIN_PER_LAP = 0.00042

# Starting slot costs time on lap one: traffic, dirty air, and the run to the
# first corner.
FIRST_LAP_PER_POSITION = 0.0018

COMPOUNDS = {
    # offset: grip relative to the reference lap; wear: lost per lap of age
    "Soft": {"offset": -0.0045, "wear": 0.00095, "life": 22},
    "Medium": {"offset": 0.0, "wear": 0.00062, "life": 32},
    "Hard": {"offset": 0.0038, "wear": 0.00038, "life": 45},
}

PIT_LOSS_SECONDS = 21.5

# A quicker car still has to get past. Without this, the field would sort
# itself into pace order within a handful of laps and stay there.
BASE_PASS_CHANCE = 0.32
PASS_CHANCE_PER_PACE = 75.0
DIRTY_AIR_GAP = (0.3, 0.9)


@dataclass
class SimDriver:
    code: str
    name: str
    team: str
    color: str
    grid: Optional[int]
    predicted_position: int
    pace: float
    form: float
    stints: list[dict[str, Any]]
    cumulative: list[float] = field(default_factory=list)
    total: float = 0.0
    tyre_age: int = 0
    stint_index: int = 0
    pit_laps: list[int] = field(default_factory=list)

    @property
    def compound(self) -> str:
        return self.stints[min(self.stint_index, len(self.stints) - 1)]["compound"]


def _plan_stints(rng: np.random.Generator, laps: int) -> list[dict[str, Any]]:
    """A one- or two-stop plan covering the full race distance."""
    stops = 2 if laps >= 58 and rng.random() < 0.55 else 1
    compounds = ["Medium", "Hard"] if stops == 1 else ["Soft", "Medium", "Hard"]
    if stops == 1 and rng.random() < 0.3:
        compounds = ["Soft", "Medium"]

    # Split the distance into stints, nudged so the field is not on an
    # identical strategy.
    ideal = laps / (stops + 1)
    lengths = []
    for i in range(stops + 1):
        jitter = rng.integers(-4, 5) if i < stops else 0
        lengths.append(max(6, int(round(ideal)) + int(jitter)))
    lengths[-1] = max(1, laps - sum(lengths[:-1]))

    stints = []
    start = 1
    for compound, length in zip(compounds, lengths):
        stints.append({"compound": compound, "start_lap": start, "laps": length})
        start += length
    return stints


def _lap_time(driver: SimDriver, base: float, lap: int, rng: np.random.Generator) -> float:
    tyre = COMPOUNDS[driver.compound]
    fraction = (
        driver.pace
        + driver.form
        + tyre["offset"]
        + tyre["wear"] * driver.tyre_age
        - FUEL_GAIN_PER_LAP * (lap - 1)
        + rng.normal(0.0, LAP_NOISE)
    )
    if lap == 1 and driver.grid:
        fraction += FIRST_LAP_PER_POSITION * (driver.grid - 1)
    return base * (1.0 + fraction)


def _pass_chance(behind: SimDriver, ahead: SimDriver) -> float:
    """Odds the quicker car actually completes the move this lap."""
    advantage = (ahead.pace + ahead.form) - (behind.pace + behind.form)
    return float(np.clip(BASE_PASS_CHANCE + PASS_CHANCE_PER_PACE * advantage, 0.04, 0.96))


def simulate_race(
    predictions: list[dict[str, Any]],
    *,
    laps: int,
    base_lap: float,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """One race, lap by lap, consistent with the predicted order."""
    if not predictions:
        return {"laps": 0, "drivers": [], "seed": seed}

    rng = np.random.default_rng(seed)
    laps = max(1, int(laps))

    expected = [float(p.get("expected_position") or p["predicted_position"]) for p in predictions]
    reference = min(expected)

    drivers: list[SimDriver] = []
    for prediction, exp in zip(predictions, expected):
        drivers.append(
            SimDriver(
                code=prediction["driver_code"],
                name=prediction.get("driver_name") or prediction["driver_code"],
                team=prediction.get("team_name", ""),
                color=prediction.get("team_color") or "#cccccc",
                grid=prediction.get("grid_position") or prediction["predicted_position"],
                predicted_position=prediction["predicted_position"],
                pace=(exp - reference) * PACE_PER_POSITION,
                form=float(rng.normal(0.0, FORM_SPREAD)),
                stints=_plan_stints(rng, laps),
            )
        )

    order = sorted(drivers, key=lambda d: d.grid or d.predicted_position)

    for lap in range(1, laps + 1):
        for driver in drivers:
            driver.tyre_age += 1
            driver.total += _lap_time(driver, base_lap, lap, rng)

            stint = driver.stints[min(driver.stint_index, len(driver.stints) - 1)]
            last_lap_of_stint = lap >= stint["start_lap"] + stint["laps"] - 1
            if last_lap_of_stint and driver.stint_index < len(driver.stints) - 1:
                driver.total += PIT_LOSS_SECONDS + float(rng.normal(0.0, 0.9))
                driver.stint_index += 1
                driver.tyre_age = 0
                driver.pit_laps.append(lap)

        # Resolve position changes in running order, so a car can only be
        # blocked by the car directly ahead of it.
        for i in range(1, len(order)):
            ahead, behind = order[i - 1], order[i]
            if behind.total >= ahead.total:
                continue
            just_pitted = lap in ahead.pit_laps or lap in behind.pit_laps
            if just_pitted or rng.random() < _pass_chance(behind, ahead):
                continue
            behind.total = ahead.total + float(rng.uniform(*DIRTY_AIR_GAP))

        order = sorted(drivers, key=lambda d: d.total)
        for driver in drivers:
            driver.cumulative.append(round(driver.total, 3))

    winner_time = order[0].cumulative[-1]
    results = []
    for position, driver in enumerate(order, start=1):
        results.append({
            "position": position,
            "driver_code": driver.code,
            "driver_name": driver.name,
            "team_name": driver.team,
            "team_color": driver.color,
            "grid": driver.grid,
            "predicted_position": driver.predicted_position,
            "gap": round(driver.cumulative[-1] - winner_time, 3),
            "places_gained": (driver.grid - position) if driver.grid else None,
            "vs_prediction": driver.predicted_position - position,
            "pit_laps": driver.pit_laps,
            "stints": driver.stints,
            "cumulative": driver.cumulative,
        })

    return {
        "laps": laps,
        "base_lap": round(base_lap, 3),
        "seed": seed,
        "race_time": round(winner_time, 1),
        "race_time_label": _format_duration(winner_time),
        "results": results,
    }


def _format_duration(seconds: float) -> str:
    """Winning time the way a classification shows it: h:mm:ss.sss."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:06.3f}"
    return f"{int(minutes)}:{secs:06.3f}"


def order_agreement(simulated: list[dict[str, Any]]) -> dict[str, Any]:
    """How closely this particular race matched the prediction."""
    if not simulated:
        return {}
    exact = sum(1 for r in simulated if r["position"] == r["predicted_position"])
    predicted_top3 = {r["driver_code"] for r in simulated if r["predicted_position"] <= 3}
    simulated_top3 = {r["driver_code"] for r in simulated if r["position"] <= 3}
    winner = next(r for r in simulated if r["position"] == 1)
    return {
        "exact_positions": exact,
        "field_size": len(simulated),
        "podium_overlap": len(predicted_top3 & simulated_top3),
        "winner_as_predicted": winner["predicted_position"] == 1,
        "mean_shift": round(
            sum(abs(r["vs_prediction"]) for r in simulated) / len(simulated), 2
        ),
    }


__all__ = ["simulate_race", "order_agreement", "SimDriver"]
