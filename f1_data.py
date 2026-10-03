"""FastF1 data access layer for the F1 Analytics Hub."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import fastf1
import fastf1.plotting as fplt
import numpy as np
import pandas as pd
from fastf1.ergast import Ergast

FIRST_SEASON = 2020
CURRENT_YEAR = datetime.now().year

# Extracted circuit outlines are cached here: tracing one costs a full
# telemetry session load, and a layout does not change between seasons.
TRACK_CACHE_DIR = Path("data/processed/tracks")

# Seasons to search for a circuit's shape, newest first.
GEOMETRY_SEASONS = range(CURRENT_YEAR, 2017, -1)

TRACK_POINTS = 400

# Published outlines (github.com/bacinger/f1-circuits, MIT) for venues with no
# position telemetry: circuits last raced before 2018, and debut circuits.
CIRCUIT_OUTLINE_DIR = Path("data/circuits")
CIRCUIT_OUTLINES = {
    "Kuala Lumpur": "my-1999",
    "Sepang": "my-1999",
    "Madrid": "es-2026",
}

# Grand Prix distance and a typical green-flag average speed, for estimating
# race length and lap time from a circuit's length alone.
RACE_DISTANCE_M = 305_000
TYPICAL_RACE_SPEED_KMH = 205.0

# Ergast constructorId / display name -> F1 media logo slug
CONSTRUCTOR_LOGO_SLUGS: dict[str, str] = {
    "red_bull": "red-bull",
    "mclaren": "mclaren",
    "ferrari": "ferrari",
    "mercedes": "mercedes",
    "alpine": "alpine",
    "aston_martin": "aston-martin",
    "williams": "williams",
    "haas": "haas",
    "rb": "rb",
    "alphatauri": "alphatauri",
    "alpha_tauri": "alphatauri",
    "toro_rosso": "toro-rosso",
    "sauber": "kick-sauber",
    "kick_sauber": "kick-sauber",
    "alfa": "alfa-romeo",
    "alfa_romeo": "alfa-romeo",
    "renault": "renault",
    "racing_point": "racing-point",
    "force_india": "force-india",
}


def get_available_years() -> list[int]:
    """Seasons available in the app (2020 through current year)."""
    return list(range(FIRST_SEASON, CURRENT_YEAR + 1))


def _logo_year(year: int) -> int:
    """F1 media team logos are published per season; clamp to supported range."""
    return max(FIRST_SEASON, min(year, CURRENT_YEAR))


def team_logo_url(constructor_id: str, year: int) -> Optional[str]:
    slug = CONSTRUCTOR_LOGO_SLUGS.get(
        constructor_id.lower().replace(" ", "_").replace("-", "_")
    )
    if not slug:
        slug = constructor_id.lower().replace("_", "-").replace(" ", "-")
    y = _logo_year(year)
    return f"https://media.formula1.com/content/dam/fom-website/teams/{y}/{slug}-logo.png"


CONSTRUCTOR_COLORS: dict[str, str] = {
    "mercedes": "#27F4D2",
    "ferrari": "#E8002D",
    "red_bull": "#3671C6",
    "mclaren": "#FF8000",
    "aston_martin": "#229971",
    "alpine": "#0093CC",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "alphatauri": "#5E8FAA",
    "alpha_tauri": "#5E8FAA",
    "toro_rosso": "#469BFF",
    "kick_sauber": "#52E252",
    "sauber": "#52E252",
    "alfa": "#C92D4B",
    "alfa_romeo": "#C92D4B",
    "haas": "#B6BABD",
    # Audi's titanium-and-green livery is too dark to read on a dark table
    "audi": "#12A594",
    "cadillac": "#B69C6D",
    "racing_point": "#F596C8",
    "force_india": "#F596C8",
    "renault": "#FFF500",
    "lotus_f1": "#FFB800",
    "manor": "#D2D4D6",
    "marussia": "#6E0000",
    "caterham": "#0B5C28",
}


# Ergast constructor names carry sponsor and legal suffixes that FastF1 drops.
CONSTRUCTOR_DISPLAY_NAMES: dict[str, str] = {
    "red_bull": "Red Bull Racing",
    "rb": "Racing Bulls",
    "alpha_tauri": "AlphaTauri",
    "alphatauri": "AlphaTauri",
    "toro_rosso": "Toro Rosso",
    "kick_sauber": "Kick Sauber",
    "sauber": "Sauber",
    "alfa": "Alfa Romeo",
    "alfa_romeo": "Alfa Romeo",
    "haas": "Haas",
    "alpine": "Alpine",
    "aston_martin": "Aston Martin",
    "force_india": "Force India",
    "racing_point": "Racing Point",
    "lotus_f1": "Lotus",
}


@lru_cache(maxsize=512)
def team_display_name(constructor_id: str, fallback: str = "") -> str:
    """Short team label, consistent between Ergast and FastF1 sources."""
    key = (constructor_id or "").lower().replace(" ", "_").replace("-", "_")
    if key in CONSTRUCTOR_DISPLAY_NAMES:
        return CONSTRUCTOR_DISPLAY_NAMES[key]

    name = (fallback or constructor_id or "").strip()
    for suffix in (" Formula 1 Team", " F1 Team", " Formula One Team", " Racing Team"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or constructor_id


@lru_cache(maxsize=512)
def team_color(team_key: str, year: int) -> str:
    """Team colour from a constructor id or display name."""
    key = (team_key or "").lower().replace(" ", "_").replace("-", "_")
    if key in CONSTRUCTOR_COLORS:
        return CONSTRUCTOR_COLORS[key]
    for candidate, color in CONSTRUCTOR_COLORS.items():
        if candidate.replace("_", "") in key.replace("_", ""):
            return color
    try:
        return _normalize_hex(fplt.get_team_color(team_key, year))
    except Exception:
        return "#888888"


@lru_cache(maxsize=16)
def get_event_schedule(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    # Drop pre-season / testing (round 0) and duplicate round rows
    races = schedule[schedule["RoundNumber"] > 0].copy()
    races = races.drop_duplicates(subset=["RoundNumber"], keep="first")
    return races.sort_values("RoundNumber")


@lru_cache(maxsize=16)
def _races_for_year_cached(year: int) -> tuple[dict[str, Any], ...]:
    races = []
    for _, row in get_event_schedule(year).iterrows():
        races.append({
            "round": int(row["RoundNumber"]),
            "event_name": str(row["EventName"]),
            "official_name": str(row.get("OfficialEventName", row["EventName"])),
            "country": str(row["Country"]),
            "location": str(row.get("Location", "")),
            "date": _format_date(row.get("EventDate")),
        })
    return tuple(races)


def get_races_for_year(year: int) -> list[dict[str, Any]]:
    """Grand Prix list with round number and display names."""
    return [dict(race) for race in _races_for_year_cached(year)]


def _format_date(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if hasattr(value, "strftime"):
            return value.strftime("%d %b %Y")
        return str(value)[:10]
    except Exception:
        return None


def get_race_info(year: int, round_num: int) -> Optional[dict[str, Any]]:
    for race in get_races_for_year(year):
        if race["round"] == round_num:
            return race
    return None


def _race_start_utc(row: Any) -> Optional[datetime]:
    """UTC start time of the race session for a schedule row."""
    for i in range(1, 6):
        name = str(row.get(f"Session{i}", ""))
        if name.strip().lower() != "race":
            continue
        for key in (f"Session{i}DateUtc", f"Session{i}Date"):
            value = row.get(key)
            if value is None or pd.isna(value):
                continue
            stamp = pd.Timestamp(value)
            return (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).to_pydatetime()
    event_date = row.get("EventDate")
    if event_date is not None and not pd.isna(event_date):
        stamp = pd.Timestamp(event_date)
        return (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).to_pydatetime()
    return None


def get_next_race() -> Optional[dict[str, Any]]:
    """The next Grand Prix that has not started yet, for the countdown."""
    now = datetime.now(timezone.utc)
    for year in (CURRENT_YEAR, CURRENT_YEAR + 1):
        try:
            schedule = get_event_schedule(year)
        except Exception:
            continue
        for _, row in schedule.iterrows():
            start = _race_start_utc(row)
            if start is None or start <= now:
                continue
            return {
                "year": year,
                "round": int(row["RoundNumber"]),
                "event_name": str(row["EventName"]),
                "official_name": str(row.get("OfficialEventName", row["EventName"])),
                "country": str(row.get("Country", "")),
                "location": str(row.get("Location", "")),
                "date": _format_date(row.get("EventDate")),
                "starts_at": start.isoformat(),
            }
    return None


def _last_completed_round(year: int) -> Optional[int]:
    """Most recent GP round that has already taken place."""
    now = datetime.now(timezone.utc)
    schedule = get_event_schedule(year)
    completed = []
    for _, row in schedule.iterrows():
        event_date = row.get("EventDate")
        if event_date is None or pd.isna(event_date):
            continue
        if hasattr(event_date, "tzinfo") and event_date.tzinfo is None:
            event_date = event_date.replace(tzinfo=timezone.utc)
        if event_date <= now:
            completed.append(int(row["RoundNumber"]))
    return max(completed) if completed else None


@lru_cache(maxsize=16)
def load_driver_media_map(year: int) -> dict[str, dict[str, str]]:
    """
    Map driverCode -> headshot URL and team colors from a recent session.
    Used for standings pages when ergast has no images. Cached because it loads
    a full session and the artwork rarely changes mid-season.
    """
    media: dict[str, dict[str, str]] = {}
    round_num = _last_completed_round(year) or 1
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        for _, row in session.results.iterrows():
            code = str(row.get("Abbreviation", ""))
            if not code:
                continue
            media[code] = {
                "headshot_url": str(row.get("HeadshotUrl", "")),
                "team_color": str(row.get("TeamColor", "")),
                "team_id": str(row.get("TeamId", "")),
            }
    except Exception:
        pass
    return media


def get_race_results(year: int, round_num: int) -> dict[str, Any]:
    """Load full race classification for a Grand Prix."""
    race_info = get_race_info(year, round_num)
    if not race_info:
        raise ValueError(f"No race found for {year} round {round_num}")

    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    positions = []
    for _, row in session.results.iterrows():
        pos = row.get("Position")
        if pd.isna(pos):
            continue
        team_name = str(row.get("TeamName", ""))
        team_id = str(row.get("TeamId", ""))
        positions.append({
            "position": int(pos),
            "driver_code": str(row.get("Abbreviation", "")),
            "driver_name": str(row.get("FullName", "")),
            "team": team_name,
            "team_id": team_id,
            "team_color": _normalize_hex(str(row.get("TeamColor", "")))
            if str(row.get("TeamColor", "")).strip()
            else team_color(team_id or team_name, year),
            "team_logo": team_logo_url(team_id, year),
            "headshot_url": str(row.get("HeadshotUrl", "")),
            "grid_position": _safe_int(row.get("GridPosition")),
            "points": _safe_float(row.get("Points")),
            "laps": _safe_int(row.get("Laps")),
            "status": str(row.get("Status", "")),
        })

    return {
        "race": race_info,
        "positions": positions,
        "session_name": session.name,
        "circuit": race_info.get("location", ""),
    }


def get_driver_standings(year: int) -> list[dict[str, Any]]:
    ergast = Ergast()
    response = ergast.get_driver_standings(season=year)
    if not response.content:
        return []

    df = response.content[0]
    media = load_driver_media_map(year)
    standings = []

    for _, row in df.iterrows():
        code = str(row.get("driverCode", ""))
        constructor_ids = row.get("constructorIds") or []
        constructor_names = row.get("constructorNames") or []
        constructor_id = constructor_ids[0] if len(constructor_ids) else ""
        team_name = constructor_names[0] if len(constructor_names) else ""

        driver_media = media.get(code, {})
        standings.append({
            "position": int(row["position"]),
            "points": float(row["points"]),
            "wins": int(row.get("wins", 0)),
            "driver_code": code,
            "driver_name": f"{row.get('givenName', '')} {row.get('familyName', '')}".strip(),
            "team": team_display_name(constructor_id, team_name),
            "constructor_id": constructor_id,
            "team_logo": team_logo_url(constructor_id, year) if constructor_id else None,
            "team_color": _normalize_hex(driver_media.get("team_color", ""))
            if driver_media.get("team_color")
            else team_color(constructor_id or team_name, year),
            "headshot_url": driver_media.get("headshot_url", ""),
            "nationality": str(row.get("driverNationality", "")),
        })

    return standings


def get_constructor_standings(year: int) -> list[dict[str, Any]]:
    ergast = Ergast()
    response = ergast.get_constructor_standings(season=year)
    if not response.content:
        return []

    df = response.content[0]
    standings = []

    for _, row in df.iterrows():
        constructor_id = str(row.get("constructorId", ""))
        team_name = str(row.get("constructorName", ""))
        standings.append({
            "position": int(row["position"]),
            "points": float(row["points"]),
            "wins": int(row.get("wins", 0)),
            "team": team_display_name(constructor_id, team_name),
            "constructor_id": constructor_id,
            "team_logo": team_logo_url(constructor_id, year),
            "team_color": team_color(constructor_id or team_name, year),
            "nationality": str(row.get("constructorNationality", "")),
        })

    return standings


def get_season_schedule(year: int) -> list[dict[str, Any]]:
    """Full calendar with session types for the schedule page."""
    schedule = fastf1.get_event_schedule(year)
    events = []

    for _, row in schedule.iterrows():
        round_num = int(row["RoundNumber"])
        if round_num == 0:
            event_label = str(row.get("EventName", "Pre-Season Testing"))
        else:
            event_label = str(row["EventName"])

        sessions = []
        for i in range(1, 6):
            name = row.get(f"Session{i}")
            # FastF1 fills unused session slots with the literal string "None"
            if name is None or pd.isna(name) or str(name).strip() in ("", "None"):
                continue
            sessions.append({
                "name": str(name),
                "date": _format_date(row.get(f"Session{i}Date")),
            })

        events.append({
            "round": round_num,
            "event_name": event_label,
            "official_name": str(row.get("OfficialEventName", event_label)),
            "country": str(row.get("Country", "")),
            "location": str(row.get("Location", "")),
            "date": _format_date(row.get("EventDate")),
            "format": str(row.get("EventFormat", "")).replace("_", " "),
            "sessions": sessions,
            "is_testing": round_num == 0,
        })

    return events


def _format_lap_time(value: Any) -> Optional[str]:
    """Timedelta to m:ss.mmm, the way lap times are always written."""
    if value is None or pd.isna(value):
        return None
    total = pd.Timedelta(value).total_seconds()
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def _format_sector(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return f"{pd.Timedelta(value).total_seconds():.3f}"


def get_telemetry_preview(year: int, round_num: int, driver_code: str) -> dict[str, Any]:
    """Fastest-lap and stint summary for one driver in a race."""
    session = fastf1.get_session(year, round_num, "R")
    session.load(telemetry=True, weather=False, messages=False)
    race_info = get_race_info(year, round_num)

    driver_laps = session.laps.pick_drivers(driver_code)
    if driver_laps.empty:
        raise ValueError(f"No laps found for driver {driver_code}")

    fastest = driver_laps.pick_fastest()
    timed = driver_laps["LapTime"].dropna()

    # Car telemetry is not archived for every session; the lap table always is.
    speed_trace: list[dict[str, float]] = []
    avg_speed = None
    try:
        tel = fastest.get_car_data().add_distance()
        if not tel.empty and "Speed" in tel.columns:
            avg_speed = round(float(tel["Speed"].mean()), 1)
            step = max(1, len(tel) // 240)
            for _, point in tel.iloc[::step].iterrows():
                speed_trace.append({
                    "distance": round(float(point["Distance"]), 1),
                    "speed": round(float(point["Speed"]), 1),
                })
    except Exception:
        pass

    stints = []
    if "Stint" in driver_laps.columns:
        for stint_num, stint_laps in driver_laps.groupby("Stint", dropna=True):
            stint_times = stint_laps["LapTime"].dropna()
            compound = stint_laps["Compound"].dropna()
            stints.append({
                "stint": _safe_int(stint_num),
                "compound": str(compound.iloc[0]).title() if len(compound) else "Unknown",
                "laps": int(len(stint_laps)),
                "best": _format_lap_time(stint_times.min()) if len(stint_times) else None,
                "average": _format_lap_time(stint_times.mean()) if len(stint_times) else None,
            })

    lap_chart = [
        {
            "lap": _safe_int(row.get("LapNumber")),
            "seconds": round(pd.Timedelta(row["LapTime"]).total_seconds(), 3),
            "compound": str(row.get("Compound") or "").title(),
        }
        for _, row in driver_laps.iterrows()
        if pd.notna(row.get("LapTime"))
    ]

    return {
        "race": race_info,
        "driver_code": driver_code,
        "driver_name": _driver_name_for_code(session, driver_code),
        "team_color": _team_color_for_code(session, driver_code, year),
        "fastest_lap": _format_lap_time(fastest.get("LapTime")),
        "lap_number": _safe_int(fastest.get("LapNumber")),
        "median_lap": _format_lap_time(timed.median()) if len(timed) else None,
        "sectors": [
            _format_sector(fastest.get("Sector1Time")),
            _format_sector(fastest.get("Sector2Time")),
            _format_sector(fastest.get("Sector3Time")),
        ],
        "speed_trap_kmh": _safe_float(fastest.get("SpeedST")),
        "avg_speed_kmh": avg_speed,
        "compound": str(fastest.get("Compound") or "").title() or None,
        "tyre_life": _safe_int(fastest.get("TyreLife")),
        "total_laps": len(driver_laps),
        "stints": stints,
        "lap_chart": lap_chart,
        "speed_trace": speed_trace,
    }


def _driver_name_for_code(session: Any, driver_code: str) -> str:
    try:
        row = session.results[session.results["Abbreviation"] == driver_code]
        if len(row):
            return str(row.iloc[0]["FullName"])
    except Exception:
        pass
    return driver_code


def _team_color_for_code(session: Any, driver_code: str, year: int) -> str:
    try:
        row = session.results[session.results["Abbreviation"] == driver_code]
        if len(row):
            return team_color(str(row.iloc[0]["TeamName"]), year)
    except Exception:
        pass
    return "#FF1801"


def get_drivers_for_race(year: int, round_num: int) -> list[str]:
    """Driver codes that participated in a race session."""
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return sorted(session.results["Abbreviation"].dropna().astype(str).unique().tolist())


# --------------------------------------------------------------------------
# Circuit geometry, for the race simulation
# --------------------------------------------------------------------------

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-") or "unknown"


@lru_cache(maxsize=16)
def _circuit_ids(year: int) -> dict[int, str]:
    """Round number to Ergast circuit id for one season."""
    try:
        schedule = Ergast().get_race_schedule(year)
    except Exception:
        return {}
    return {int(row["round"]): str(row["circuitId"]) for _, row in schedule.iterrows()}


def _editions_of_circuit(year: int, round_num: int, location: str) -> list[tuple[int, int]]:
    """Every (year, round) held at the same circuit, most recent first.

    Matched on Ergast's circuit id, because FastF1's location names drift
    between seasons (Yas Marina and Yas Island, Monaco and Monte Carlo).
    Falls back to the location name when the id is unavailable.
    """
    circuit = _circuit_ids(year).get(round_num)
    editions = []
    for season in GEOMETRY_SEASONS:
        if circuit:
            editions += [(season, rnd) for rnd, cid in _circuit_ids(season).items() if cid == circuit]
            continue
        try:
            schedule = get_event_schedule(season)
        except Exception:
            continue
        for _, row in schedule.iterrows():
            if str(row.get("Location", "")) == location:
                editions.append((season, int(row["RoundNumber"])))
    return editions


def _trace_lap(year: int, round_num: int) -> Optional[dict[str, Any]]:
    """Outline of one lap, resampled onto an even time grid.

    Even spacing in *time* is what makes an animated car look right: the
    samples bunch up through corners and spread out down the straights, so a
    dot stepping through them at a constant rate slows and accelerates the way
    a car does.
    """
    try:
        session = fastf1.get_session(year, round_num, "R")
        session.load(telemetry=True, weather=False, messages=False)
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()
        if pos is None or len(pos) < 50:
            return None
    except Exception:
        return None

    try:
        rotation = float(session.get_circuit_info().rotation or 0.0)
    except Exception:
        rotation = 0.0

    elapsed = (pos["SessionTime"] - pos["SessionTime"].iloc[0]).dt.total_seconds().to_numpy()
    grid = np.linspace(elapsed[0], elapsed[-1], TRACK_POINTS)
    x = np.interp(grid, elapsed, pos["X"].to_numpy(dtype=float))
    y = np.interp(grid, elapsed, pos["Y"].to_numpy(dtype=float))

    theta = math.radians(rotation)
    rx = x * math.cos(theta) - y * math.sin(theta)
    ry = x * math.sin(theta) + y * math.cos(theta)
    return {"x": rx, "y": ry, "source_year": year}


def _outline_feature(location: str) -> Optional[dict[str, Any]]:
    name = CIRCUIT_OUTLINES.get(location)
    if not name:
        return None
    try:
        return json.loads((CIRCUIT_OUTLINE_DIR / f"{name}.geojson").read_text())["features"][0]
    except (OSError, KeyError, IndexError, json.JSONDecodeError):
        return None


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    wrapped = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(wrapped, np.ones(window) / window, mode="same")[pad:-pad]


def _published_outline(location: str) -> Optional[dict[str, Any]]:
    """A circuit's surveyed layout, resampled onto an estimated time grid.

    The outline is only a line, with no timing, so corner speeds come from its
    shape: a car can take a bend no faster than its grip allows for that
    radius. Resampling on the resulting time grid makes cars brake into corners
    the same way they do on the telemetry-traced circuits.
    """
    feature = _outline_feature(location)
    if feature is None:
        return None

    coords = np.asarray(feature["geometry"]["coordinates"], dtype=float)
    lon, lat = coords[:, 0], coords[:, 1]
    # An equirectangular projection is exact enough across a few kilometres.
    metres_per_degree = 111_320.0
    x = (lon - lon.mean()) * math.cos(math.radians(lat.mean())) * metres_per_degree
    y = (lat - lat.mean()) * metres_per_degree

    distance = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    samples = TRACK_POINTS * 4
    even = np.linspace(0.0, distance[-1], samples, endpoint=False)
    fx = _circular_smooth(np.interp(even, distance, x), 9)
    fy = _circular_smooth(np.interp(even, distance, y), 9)

    step = distance[-1] / samples
    heading = np.unwrap(np.arctan2(np.gradient(fy), np.gradient(fx)))
    curvature = _circular_smooth(np.abs(np.gradient(heading)) / step, 15)

    # v = sqrt(lateral grip / curvature), capped at top speed.
    top_speed, lateral_grip = 90.0, 40.0
    speed = np.minimum(top_speed, np.sqrt(lateral_grip / np.maximum(curvature, 1e-6)))
    speed = _circular_smooth(speed, 25)

    elapsed = np.r_[0.0, np.cumsum(step / speed)]
    loop_x, loop_y = np.r_[fx, fx[0]], np.r_[fy, fy[0]]
    grid = np.linspace(0.0, elapsed[-1], TRACK_POINTS)
    return {
        "x": np.interp(grid, elapsed, loop_x),
        "y": np.interp(grid, elapsed, loop_y),
        "source_year": None,
        "name": feature["properties"].get("Name") or location,
    }


def _corner_outline(editions: list[tuple[int, int]]) -> Optional[dict[str, Any]]:
    """Rough loop through the corner apexes, when no lap can be traced."""
    for year, round_num in editions:
        try:
            session = fastf1.get_session(year, round_num, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            info = session.get_circuit_info()
            corners = info.corners
            if corners is None or len(corners) < 4:
                continue
        except Exception:
            continue

        theta = math.radians(float(info.rotation or 0.0))
        cx = corners["X"].to_numpy(dtype=float)
        cy = corners["Y"].to_numpy(dtype=float)
        rx = cx * math.cos(theta) - cy * math.sin(theta)
        ry = cx * math.sin(theta) + cy * math.cos(theta)

        # Close the loop, then smooth it so the apexes read as a circuit
        # rather than a polygon.
        loop_x = np.append(rx, rx[0])
        loop_y = np.append(ry, ry[0])
        t = np.linspace(0, 1, len(loop_x))
        fine = np.linspace(0, 1, TRACK_POINTS)
        sx = np.interp(fine, t, loop_x)
        sy = np.interp(fine, t, loop_y)
        for _ in range(28):
            sx = np.convolve(np.r_[sx[-3:], sx, sx[:3]], np.ones(7) / 7, mode="same")[3:-3]
            sy = np.convolve(np.r_[sy[-3:], sy, sy[:3]], np.ones(7) / 7, mode="same")[3:-3]
        return {"x": sx, "y": sy, "source_year": year}
    return None


def _schematic_outline(location: str) -> dict[str, Any]:
    """A plausible closed circuit for a venue with no published geometry.

    Debut circuits have neither telemetry nor corner data, and an empty canvas
    is worse than an honest placeholder, so the shape is built from a few
    harmonics seeded by the venue name: stable for a given circuit, clearly
    labelled in the UI, and never passed off as the real layout.
    """
    rng = np.random.default_rng(abs(hash(location)) % (2**32))
    angle = np.linspace(0, 2 * math.pi, TRACK_POINTS)
    radius = np.ones_like(angle)
    for harmonic in (2, 3, 4, 5, 7):
        radius += rng.uniform(0.04, 0.17) * np.sin(harmonic * angle + rng.uniform(0, 2 * math.pi))
    return {
        "x": radius * np.cos(angle),
        "y": radius * np.sin(angle) * 0.72,
        "source_year": None,
    }


def _normalised_points(x: np.ndarray, y: np.ndarray) -> list[list[float]]:
    """Fit the outline into a 1000-unit box, keeping its proportions."""
    width = max(float(x.max() - x.min()), 1.0)
    height = max(float(y.max() - y.min()), 1.0)
    scale = 1000.0 / max(width, height)
    off_x = (1000.0 - width * scale) / 2.0
    off_y = (1000.0 - height * scale) / 2.0
    return [
        [
            round((float(px) - float(x.min())) * scale + off_x, 1),
            # SVG y grows downward, so flip to keep the real orientation.
            round(1000.0 - ((float(py) - float(y.min())) * scale + off_y), 1),
        ]
        for px, py in zip(x, y)
    ]


@lru_cache(maxsize=32)
def get_circuit_path(year: int, round_num: int) -> dict[str, Any]:
    """Normalised outline of a circuit for the simulation canvas.

    Tries the race itself first, then earlier visits to the same location: the
    2026 sessions have no position telemetry archived, but the circuits are
    unchanged, so a lap from a previous season draws the same track.
    """
    race = get_race_info(year, round_num) or {}
    location = str(race.get("location") or race.get("country") or f"{year}-{round_num}")
    cache_file = TRACK_CACHE_DIR / f"{_slug(location)}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    editions = _editions_of_circuit(year, round_num, location)
    traced = _trace_lap(year, round_num)
    kind = "lap"
    if traced is None:
        for alt_year, alt_round in editions:
            if (alt_year, alt_round) == (year, round_num):
                continue
            traced = _trace_lap(alt_year, alt_round)
            if traced is not None:
                break
    if traced is None:
        traced = _published_outline(location)
        kind = "outline"
    if traced is None:
        traced = _corner_outline(editions)
        kind = "corners"
    if traced is None:
        traced = _schematic_outline(location)
        kind = "schematic"

    notes = {
        "lap": f"Traced from a real lap at {location} in {traced['source_year']}.",
        "outline": (
            f"Real layout of {traced.get('name', location)}. No timed lap exists "
            "here yet, so corner speeds are estimated from the shape of the track."
        ),
        "corners": f"Drawn from the surveyed corner positions at {location}.",
        "schematic": (
            f"No layout has been published for {location} yet, so this is a "
            "placeholder shape — the race simulation itself is unaffected."
        ),
    }

    result = {
        "location": location,
        "points": _normalised_points(traced["x"], traced["y"]),
        "kind": kind,
        "source_year": traced["source_year"],
        "note": notes[kind],
    }

    # A placeholder is not cached, so real geometry is picked up once it exists.
    if kind != "schematic":
        try:
            TRACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(result))
        except OSError:
            pass
    return result


@lru_cache(maxsize=64)
def get_pace_reference(year: int, round_num: int) -> dict[str, Any]:
    """Race length and representative lap time, for scaling the simulation.

    Uses the race itself when it has run, otherwise the most recent earlier
    visit to the same circuit.
    """
    race = get_race_info(year, round_num) or {}
    location = str(race.get("location") or "")

    candidates = [(year, round_num)]
    candidates += [e for e in _editions_of_circuit(year, round_num, location) if e != (year, round_num)]

    for cand_year, cand_round in candidates[:4]:
        try:
            session = fastf1.get_session(cand_year, cand_round, "R")
            session.load(telemetry=False, weather=False, messages=False)
            times = session.laps["LapTime"].dropna()
            total_laps = _safe_int(session.laps["LapNumber"].max())
            if not len(times) or not total_laps:
                continue
            # The 20th percentile approximates green-flag race pace: quick
            # enough to exclude pit and safety car laps, slow enough not to be
            # one driver's single best lap.
            base = float(pd.Timedelta(times.quantile(0.2)).total_seconds())
            return {
                "laps": int(total_laps),
                "base_lap": round(base, 3),
                "source_year": cand_year,
                "is_estimate": (cand_year, cand_round) != (year, round_num),
                "source_label": f"Pace reference from {cand_year}",
            }
        except Exception:
            continue

    feature = _outline_feature(location)
    length = (feature or {}).get("properties", {}).get("length")
    if length:
        return {
            "laps": math.ceil(RACE_DISTANCE_M / length),
            "base_lap": round(length / (TYPICAL_RACE_SPEED_KMH / 3.6), 3),
            "source_year": None,
            "is_estimate": True,
            "source_label": f"Pace estimated from the {length / 1000:.3f} km lap",
        }

    return {
        "laps": 57,
        "base_lap": 92.0,
        "source_year": None,
        "is_estimate": True,
        "source_label": "Generic race pace",
    }


def _normalize_hex(color: str) -> str:
    color = (color or "").strip().lstrip("#")
    if len(color) in (3, 6):
        return f"#{color}"
    return "#888888"


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None