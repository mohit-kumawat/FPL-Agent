#!/usr/bin/env python3
"""Decide whether a scheduled run should fire, and which kind.

Exit 0 = run (stdout line 1 is `mode|gw|window|reason`), exit 78 = skip,
anything else = this gate broke. The caller treats every other code as "fail
open and run anyway" -- a gate that cannot decide must never silence the
routine. 78 rather than 1 because Python exits 1 on any uncaught exception,
including an import error raised before the handler below exists.

Cadence is deadline-relative, not clock-relative. A gameweek has exactly one
deadline, so a gameweek gets at most three runs:

    scan       >30h out (or no deadline known)   news sweep, once per GW
    decision   4-30h out                         the recommendation run
    teamnews   0.5-4h out                        lineup-leak check

launchd still wakes on fixed slots (it can't do deadline-relative), and this
gate maps each wake-up to a window or a skip. One `ok` run per (gw, window);
failures retry at later slots, at most MAX_ATTEMPTS per window.

Mode: `gw1` (build the initial squad) until squad.yaml has a team, `weekly`
(hold-by-default transfer advice) after.

Imports only fpl_agent.config, which is pathlib-only and safe under bare
system python3.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from fpl_agent import config  # noqa: E402  (path setup must precede the import)

STATE = PROJECT / "memory" / "state.json"
LEDGER = PROJECT / "routine" / "logs" / "runs.jsonl"

# (name, lower-exclusive, upper-inclusive) in hours to the deadline.
WINDOWS = (("teamnews", 0.5, 4.0), ("decision", 4.0, 30.0))
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


def _window_runs(gw: int, window: str) -> list[dict]:
    """Ledger rows for this (gw, window), oldest first."""
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
        if row.get("gw") == gw and row.get("window") == window:
            rows.append(row)
    return rows


def main() -> NoReturn:
    state = _state()
    stage = state.get("stage") or {}
    gw = int(stage.get("next_gw") or 0)
    mode = "weekly" if stage.get("has_team") else "gw1"

    deadline = _parse_ts(stage.get("next_deadline")
                         or (state.get("last_gw_state") or {}).get("next_deadline"))
    now = datetime.now(timezone.utc)
    hours_left = ((deadline - now).total_seconds() / 3600.0) if deadline else None

    if os.environ.get("FPL_ROUTINE_FORCE") == "1":
        window = "scan"
        if hours_left is not None:
            for name, lo, hi in WINDOWS:
                if lo < hours_left <= hi:
                    window = name
        _say_run(mode, gw, window, "forced (FPL_ROUTINE_FORCE=1)")

    if hours_left is not None and hours_left <= 0.5:
        _say_skip(f"deadline {'passed' if hours_left <= 0 else 'too close'} "
                  f"({hours_left:.1f}h) — too late to act on advice")

    window = "scan"
    if hours_left is not None:
        for name, lo, hi in WINDOWS:
            if lo < hours_left <= hi:
                window = name

    runs = _window_runs(gw, window)
    if any(r.get("status") == "ok" for r in runs):
        _say_skip(f"GW{gw} {window} window already done")
    failures = sum(1 for r in runs if r.get("status") != "ok")
    if failures >= MAX_ATTEMPTS:
        _say_skip(f"GW{gw} {window}: {failures} failed attempts, giving up on this window")

    left = f"{hours_left:.0f}h to deadline" if hours_left is not None else "no deadline known"
    retry = f", retry {failures + 1}/{MAX_ATTEMPTS}" if failures else ""
    _say_run(mode, gw, window, f"GW{gw} {window} window open ({left}){retry}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # never let a broken gate look like a quiet day
        print(f"gate error: {exc!r}")
        sys.exit(BROKEN)
