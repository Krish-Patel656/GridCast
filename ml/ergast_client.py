"""Throttled Ergast access.

The public Ergast mirror caps responses at 100 rows and rejects bursts with
"Too Many Requests", so every call goes through a small rate limiter with
exponential backoff, and whole seasons are fetched in as few requests as
possible.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import pandas as pd
from fastf1.ergast import Ergast

PAGE_SIZE = 100
MIN_INTERVAL = 0.3
MAX_RETRIES = 5

_last_call = 0.0


def _throttled(fn: Callable, *args, **kwargs):
    global _last_call

    delay = 2.0
    for attempt in range(MAX_RETRIES):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            result = fn(*args, **kwargs)
            _last_call = time.monotonic()
            return result
        except Exception as exc:
            _last_call = time.monotonic()
            if "Too Many Requests" not in str(exc) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def _paged(fetch: Callable[[int], Any]) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Walk an endpoint's pages, stitching rounds that straddle a page break.

    A page boundary lands in the middle of a race, so the same round shows up
    at the end of one page and the start of the next with different rows. Both
    halves have to be concatenated or the race silently loses drivers.
    """
    rows_by_round: dict[int, list[pd.DataFrame]] = {}
    meta_by_round: dict[int, pd.DataFrame] = {}
    offset = 0
    rows_fetched = 0

    while True:
        resp = _throttled(fetch, offset)
        if not resp.content:
            break

        desc = resp.description
        page_rows = 0
        for i, frame in enumerate(resp.content):
            round_num = int(desc.iloc[i]["round"])
            rows_by_round.setdefault(round_num, []).append(frame)
            meta_by_round.setdefault(round_num, desc.iloc[[i]])
            page_rows += len(frame)

        rows_fetched += page_rows
        total = getattr(resp, "total_results", None)
        offset += PAGE_SIZE
        if page_rows == 0 or (total is not None and rows_fetched >= int(total)):
            break

    if not meta_by_round:
        return [], pd.DataFrame()

    rounds = sorted(rows_by_round)
    frames = [
        pd.concat(rows_by_round[r], ignore_index=True).drop_duplicates() for r in rounds
    ]
    descriptions = pd.concat([meta_by_round[r] for r in rounds], ignore_index=True)
    return frames, descriptions


def season_results(ergast: Ergast, year: int) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    return _paged(
        lambda offset: ergast.get_race_results(season=year, limit=PAGE_SIZE, offset=offset)
    )


def season_sprint_results(ergast: Ergast, year: int) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Sprint races award championship points and only exist from 2021 on."""
    if year < 2021:
        return [], pd.DataFrame()
    try:
        return _paged(
            lambda offset: ergast.get_sprint_results(season=year, limit=PAGE_SIZE, offset=offset)
        )
    except Exception:
        return [], pd.DataFrame()


def qualifying_results(ergast: Ergast, year: int, round_num: int) -> Optional[pd.DataFrame]:
    try:
        resp = _throttled(ergast.get_qualifying_results, season=year, round=round_num)
    except Exception:
        return None
    return resp.content[0] if resp.content else None
