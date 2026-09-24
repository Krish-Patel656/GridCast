"""F1 Analytics Hub — FastAPI app.

Run with:
    uvicorn main:app --reload

Handlers are plain `def` on purpose: FastF1 and Ergast calls are blocking, so
FastAPI runs them on its worker threadpool instead of stalling the event loop.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import f1_data

os.makedirs("cache", exist_ok=True)
import fastf1  # noqa: E402

fastf1.Cache.enable_cache("cache")

app = FastAPI(title="F1 Analytics Hub")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

_predictor = None
_predictor_lock = threading.Lock()


def _years_context() -> dict[str, Any]:
    return {"years": f1_data.get_available_years()}


def _parse_year(value: Optional[str], years: list[int]) -> int:
    default = years[-1]
    if not value:
        return default
    try:
        year = int(value)
    except ValueError:
        return default
    return year if year in years else default


# ----- Race results -----

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    years = f1_data.get_available_years()
    selected_year = _parse_year(request.query_params.get("year"), years)

    selected_round = None
    round_param = request.query_params.get("round")
    if round_param:
        try:
            selected_round = int(round_param)
        except ValueError:
            selected_round = None

    races = []
    next_race = None
    try:
        races = f1_data.get_races_for_year(selected_year)
        next_race = f1_data.get_next_race()
    except Exception:
        pass

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "years": years,
            "year": selected_year,
            "races": races,
            "selected_round": selected_round,
            "next_race": next_race,
        },
    )


@app.get("/results", response_class=HTMLResponse)
def show_results(request: Request, year: int = Query(...), round: int = Query(...)):
    error_msg = None
    race = None
    positions = []
    try:
        data = f1_data.get_race_results(year, round)
        race = data["race"]
        positions = data["positions"]
    except Exception as exc:
        error_msg = str(exc)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "year": year,
            "round": round,
            "race": race,
            "positions": positions,
            "error_msg": error_msg,
        },
    )


@app.get("/get_races")
def get_races(year: int = Query(...)):
    try:
        races = f1_data.get_races_for_year(year)
        return JSONResponse(content={"races": races, "rounds": races})
    except Exception as exc:
        return JSONResponse(content={"races": [], "rounds": [], "error": str(exc)})


# ----- Standings -----

@app.get("/standings/drivers", response_class=HTMLResponse)
def driver_standings_page(request: Request, year: int = Query(default=f1_data.CURRENT_YEAR)):
    error_msg = None
    standings = []
    try:
        standings = f1_data.get_driver_standings(year)
    except Exception as exc:
        error_msg = str(exc)

    return templates.TemplateResponse(
        "standings_drivers.html",
        {
            "request": request,
            "year": year,
            "standings": standings,
            "error_msg": error_msg,
            **_years_context(),
        },
    )


@app.get("/standings/constructors", response_class=HTMLResponse)
def constructor_standings_page(request: Request, year: int = Query(default=f1_data.CURRENT_YEAR)):
    error_msg = None
    standings = []
    try:
        standings = f1_data.get_constructor_standings(year)
    except Exception as exc:
        error_msg = str(exc)

    return templates.TemplateResponse(
        "standings_constructors.html",
        {
            "request": request,
            "year": year,
            "standings": standings,
            "error_msg": error_msg,
            **_years_context(),
        },
    )


# ----- Schedule -----

@app.get("/schedule", response_class=HTMLResponse)
def schedule_page(request: Request, year: int = Query(default=f1_data.CURRENT_YEAR)):
    error_msg = None
    events = []
    try:
        events = f1_data.get_season_schedule(year)
    except Exception as exc:
        error_msg = str(exc)

    return templates.TemplateResponse(
        "schedule.html",
        {
            "request": request,
            "year": year,
            "events": events,
            "error_msg": error_msg,
            **_years_context(),
        },
    )


# ----- Telemetry -----

@app.get("/telemetry", response_class=HTMLResponse)
def telemetry_page(
    request: Request,
    year: Optional[int] = Query(default=None),
    round: Optional[int] = Query(default=None),
    driver: Optional[str] = Query(default=None),
):
    error_msg = None
    preview = None

    if year and round and driver:
        try:
            preview = f1_data.get_telemetry_preview(year, round, driver)
        except Exception as exc:
            error_msg = str(exc)

    return templates.TemplateResponse(
        "telemetry.html",
        {
            "request": request,
            "preview": preview,
            "selected_year": year,
            "selected_round": round,
            "selected_driver": driver,
            "error_msg": error_msg,
            **_years_context(),
        },
    )


@app.get("/get_race_drivers")
def get_race_drivers(year: int = Query(...), round: int = Query(...)):
    try:
        drivers = f1_data.get_drivers_for_race(year, round)
        return JSONResponse(content={"drivers": drivers})
    except Exception as exc:
        return JSONResponse(content={"drivers": [], "error": str(exc)}, status_code=400)


# ----- ML predictions -----

def _ml_status() -> dict[str, Any]:
    try:
        from ml.predict import model_status
    except ImportError as exc:
        return {"ready": False, "error": f"ML dependencies not installed: {exc}"}
    return model_status()


def _get_predictor():
    """One predictor for the whole process; loading weights per request is slow."""
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            from ml.predict import RacePredictor

            _predictor = RacePredictor()
        return _predictor


def _predictions_context(year: int, round_num: Optional[int] = None) -> dict[str, Any]:
    races = []
    try:
        races = f1_data.get_races_for_year(year)
    except Exception:
        pass
    return {
        "years": f1_data.get_available_years(),
        "year": year,
        "races": races,
        "selected_round": round_num,
    }


@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(
    request: Request,
    year: Optional[int] = Query(default=None),
    round: Optional[int] = Query(default=None),
):
    """Predictions are a GET so a specific race's prediction can be linked."""
    status = _ml_status()
    years = f1_data.get_available_years()
    selected_year = _parse_year(str(year) if year else None, years)

    next_race = None
    try:
        next_race = f1_data.get_next_race()
    except Exception:
        pass

    result: dict[str, Any] = {}
    error_msg = None
    race = None

    if round is not None:
        if not status.get("ready"):
            error_msg = (
                "The model has not been trained yet. Run: "
                "python -m ml.collect_data && python -m ml.train"
            )
        else:
            try:
                result = _get_predictor().predict_race(selected_year, round)
                race = f1_data.get_race_info(selected_year, round)
            except Exception as exc:
                error_msg = str(exc)

    return templates.TemplateResponse(
        "predictions.html",
        {
            "request": request,
            "ml_ready": status.get("ready", False),
            "ml_meta": status.get("metadata", {}),
            "next_race": next_race,
            "round": round,
            "race": race,
            "predictions": result.get("predictions", []),
            "grid_known": result.get("grid_known", False),
            "entry_source": result.get("source_label"),
            "error_msg": error_msg,
            **_predictions_context(selected_year, round),
        },
    )


@app.get("/api/predictions")
def api_predictions(year: int = Query(...), round: int = Query(...)):
    status = _ml_status()
    if not status.get("ready"):
        return JSONResponse(content={"error": "Model not trained"}, status_code=503)
    try:
        return JSONResponse(content=_get_predictor().predict_race(year, round))
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
