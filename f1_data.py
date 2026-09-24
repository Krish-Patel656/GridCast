"""FastF1 data access layer for the F1 Analytics Hub."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Optional

import fastf1
import fastf1.plotting as fplt
import pandas as pd
from fastf1.ergast import Ergast

FIRST_SEASON = 2020
CURRENT_YEAR = datetime.now().year

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