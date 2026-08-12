#!/usr/bin/env python3
"""Decide whether a scheduled run should fire, and which kind.

Exit 0 = run (stdout line is `mode|gw|window|reason`), exit 78 = skip, anything
else = this gate broke. The caller treats every other code as "fail open and
run anyway" -- a gate that cannot decide must never silence the routine. 78
rather than 1 because Python exits 1 on any uncaught exception, including an
import error raised before the handler below exists.

Cadence is deadline-relative. Per deadline there is one `decision` run
(4-30h out) and one `teamnews` run (0.5-4h out); outside those windows a
`scan` run fires at most once every SCAN_EVERY_H hours -- so a three-week
preseason gets a research pass every few days, not one in total.

Dedup is RECENCY-SCOPED, never forever: decision/teamnews match ledger rows
for the same (gw, window) started within the last WINDOW_LOOKBACK_H hours, so
last season's GW1 rows can never mask this season's, and a stray gw=0 row
can't wedge a fresh install. Failures retry at later slots, at most
MAX_ATTEMPTS per window, then the window is given up.

A deadline in the past is treated as a stale state file, not a reason to stop:
it classifies as `scan` so the pipeline runs and refreshes the state to the
next gameweek. The gate must never be able to wait for an update that only a
run it refuses to allow can produce.

Mode: `gw1` (build the initial squad) until squad.yaml has a team, `weekly`
after. Stdlib only -- runs under bare system python3 with no uv and no venv.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

PROJECT = Path(__file__).resolve().parent.parent
STATE = PROJECT / "memory" / "state.json"
LEDGER = PROJECT / "routine" / "logs" / "runs.jsonl"

# (name, lower-exclusive, upper-inclusive) in hours to the deadline.
WINDOWS = (("teamnews", 0.5, 4.0), ("decision", 4.0, 30.0))
SCAN_EVERY_H = 72.0        # scan cadence outside the deadline windows
WINDOW_LOOKBACK_H = 36.0   # how far back a (gw, window) row still counts
MAX_ATTEMPTS = 3

SKIP, BROKEN = 78, 2


def _say_run(mode: str, gw: int, window: str, reason: str) -> NoReturn:
    print(f"{mode}|{gw}|{window}|{reason}")
    sys.exit(0)


def _say_skip(reason: str) -> NoReturn:
    print(reason)
    sys.exit(SKIP)


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _recent_runs(window: str, lookback_h: float, now: datetime,
                 gw: int | None = None) -> list[dict]:
    """Ledger rows for this window started within lookback_h, oldest first.

    `gw=None` matches any gameweek (used for scan, whose cadence is purely
    time-based). Rows without a parseable start time are treated as recent --
    counting a failure twice is cheaper than retrying forever.
    """
    try:
        lines = LEDGER.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("window") != window:
            continue
        if gw is not None and row.get("gw") != gw:
            continue
        started = _parse_ts(row.get("started") or row.get("ts"))
        if started is not None and (now - started).total_seconds() / 3600.0 > lookback_h:
            continue
        rows.append(row)
    return rows


def _decide_window(hours_left: float | None) -> tuple[str, str]:
    """(window, note). Past-deadline is stale state, never a stop."""
    if hours_left is None:
        return "scan", "no deadline known"
    if hours_left <= 0:
        return "scan", f"deadline passed {-hours_left:.0f}h ago — refreshing state"
    for name, lo, hi in WINDOWS:
        if lo < hours_left <= hi:
            return name, f"{hours_left:.0f}h to deadline"
    return "scan", f"{hours_left:.0f}h to deadline"


def main() -> NoReturn:
    state = _state()
    stage = state.get("stage") or {}
    gw = int(stage.get("next_gw") or 0)
    mode = "weekly" if stage.get("has_team") else "gw1"

    deadline = _parse_ts(stage.get("next_deadline")
                         or (state.get("last_gw_state") or {}).get("next_deadline"))
    now = datetime.now(timezone.utc)
    hours_left = ((deadline - now).total_seconds() / 3600.0) if deadline else None

    window, note = _decide_window(hours_left)

    if os.environ.get("FPL_ROUTINE_FORCE") == "1":
        _say_run(mode, gw, window, "forced (FPL_ROUTINE_FORCE=1)")

    # Inside the last half hour advice can't be acted on. This fires only for a
    # LIVE deadline — a passed one classified as scan above and falls through.
    if hours_left is not None and 0 < hours_left <= 0.5:
        _say_skip(f"deadline in {hours_left:.1f}h — too late to act on advice")

    if window == "scan":
        runs = _recent_runs("scan", SCAN_EVERY_H, now)
        if any(r.get("status") == "ok" for r in runs):
            _say_skip(f"scan done within the last {SCAN_EVERY_H:.0f}h")
    else:
        runs = _recent_runs(window, WINDOW_LOOKBACK_H, now, gw=gw)
        if any(r.get("status") == "ok" for r in runs):
            _say_skip(f"GW{gw} {window} window already done")

    failures = sum(1 for r in runs if r.get("status") != "ok")
    if failures >= MAX_ATTEMPTS:
        scope = f"GW{gw} {window}" if window != "scan" else "scan"
        _say_skip(f"{scope}: {failures} failed attempts, giving up on this window")

    retry = f", retry {failures + 1}/{MAX_ATTEMPTS}" if failures else ""
    _say_run(mode, gw, window, f"GW{gw} {window} ({note}){retry}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # never let a broken gate look like a quiet day
        print(f"gate error: {exc!r}")
        sys.exit(BROKEN)
