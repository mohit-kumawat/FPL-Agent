"""What the game's rules actually are, read from the API where possible.

`bootstrap-static` publishes the live rulebook in `game_config`: the full
scoring table (`game_config.scoring`) and every chip with its gameweek window
(`chips`). Hardcoding those means a mid-season tweak silently makes every
projection wrong, and a season rollover silently makes the replay wrong.

So the split here is:

  * the model keeps STATIC constants (models.GOAL_PTS and friends), because a
    replay of 2023/24 must score under 2023/24's rules, not today's;
  * `check_scoring()` compares those constants against the live table on every
    run and reports any drift, so the failure mode is a loud warning rather
    than a quietly wrong recommendation;
  * chip windows are read live, because they carry no historical meaning and
    change shape between seasons (2026/27 ships two full sets).

Verified against the 2026/27 API and the official rule announcements on
2026-08-12: goals GKP 10 / DEF 6 / MID 5 / FWD 4, assists 3, clean sheets
GKP+DEF 4 / MID 1, goals conceded -1 per 2 for GKP+DEF, saves 1 per 3,
penalty saved +5, penalty missed -2, yellow -1, red -3, own goal -2,
appearance 1 (<60') / 2 (60'+), defensive contribution 2 points with GKP
excluded (DEF at 10 CBIT, MID/FWD at 12 including recoveries).
"""
from __future__ import annotations

from typing import Any

# Position id -> API position key, for reading game_config.scoring tables.
POS_KEY = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Chip families and the human-facing names for each half's copy. 2026/27 gives
# two full sets — one per half of the season — so a squad file that names only
# `bboost` is describing one of two.
CHIP_FAMILIES = ("wildcard", "freehit", "bboost", "3xc")
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit",
               "bboost": "Bench Boost", "3xc": "Triple Captain"}
# Chips that change the SQUAD (structural) vs the week's POINTS (EV-scored).
STRUCTURAL_CHIPS = ("wildcard", "freehit")
SCORING_CHIPS = ("3xc", "bboost")


# --------------------------------------------------------------- scoring drift
def live_scoring(boot: dict) -> dict[str, Any]:
    """`game_config.scoring` if the payload carries it, else {}."""
    return ((boot.get("game_config") or {}).get("scoring") or {}) if boot else {}


def check_scoring(boot: dict, goal_pts: dict[int, float], cs_pts: dict[int, float],
                  dc_points: dict[int, float]) -> list[str]:
    """Compare the model's static scoring constants against the live table.

    Returns human-readable mismatches. Empty means the model is scoring the
    game the API says is being played. Absent/partial API data returns nothing
    rather than false alarms — this is a tripwire, never a blocker.
    """
    scoring = live_scoring(boot)
    if not scoring:
        return []
    problems: list[str] = []

    def compare(api_key: str, static: dict[int, float], label: str) -> None:
        table = scoring.get(api_key)
        if not isinstance(table, dict):
            return
        for pos, key in POS_KEY.items():
            if key not in table:
                continue
            live = float(table[key])
            mine = float(static.get(pos, 0.0))
            if abs(live - mine) > 1e-9:
                problems.append(f"{label} for {key}: API says {live:g}, model uses {mine:g}")

    compare("goals_scored", goal_pts, "goal points")
    compare("clean_sheets", cs_pts, "clean-sheet points")
    compare("defensive_contribution", dc_points, "defensive-contribution points")
    for api_key, mine, label in (("assists", 3.0, "assist points"),
                                 ("bonus", 1.0, "bonus per point"),
                                 ("penalties_saved", 5.0, "penalty-save points"),
                                 ("penalties_missed", -2.0, "penalty-miss points"),
                                 ("yellow_cards", -1.0, "yellow-card points"),
                                 ("red_cards", -3.0, "red-card points"),
                                 ("own_goals", -2.0, "own-goal points"),
                                 ("saves", 1.0, "points per save unit"),
                                 ("short_play", 1.0, "sub-60' appearance points"),
                                 ("long_play", 2.0, "60'+ appearance points")):
        if api_key in scoring and isinstance(scoring[api_key], (int, float)):
            live = float(scoring[api_key])
            if abs(live - mine) > 1e-9:
                problems.append(f"{label}: API says {live:g}, model uses {mine:g}")
    return problems


# ------------------------------------------------------------------ chip rules
# Fallback used only when the payload carries no `chips` block. These are the
# 2026/27 windows; going silent on chips would be a worse failure than using a
# slightly stale window, so the code degrades to this and the caller can tell
# the difference via `windows_are_live`.
DEFAULT_CHIP_WINDOWS: dict[str, list[tuple[int, int]]] = {
    "wildcard": [(2, 19), (20, 38)],
    "freehit": [(2, 19), (20, 38)],
    "bboost": [(1, 19), (20, 38)],
    "3xc": [(1, 19), (20, 38)],
}


def windows_are_live(boot: dict) -> bool:
    """True when the chip windows came from the API rather than the fallback."""
    return bool(boot.get("chips"))


def chip_windows(boot: dict) -> dict[str, list[tuple[int, int]]]:
    """family -> [(start_event, stop_event), ...] in gameweek order.

    2026/27 ships two entries per family (one per half). Read live because the
    windows are exactly what changes between seasons; falls back to
    DEFAULT_CHIP_WINDOWS when the payload omits them.
    """
    if not boot.get("chips"):
        return {k: list(v) for k, v in DEFAULT_CHIP_WINDOWS.items()}
    out: dict[str, list[tuple[int, int]]] = {}
    for c in (boot.get("chips") or []):
        name = str(c.get("name", "")).lower()
        start, stop = c.get("start_event"), c.get("stop_event")
        if not name or start is None or stop is None:
            continue
        out.setdefault(name, []).append((int(start), int(stop)))
    for name in out:
        out[name].sort()
    return out


def chip_half(family: str, gw: int, windows: dict[str, list[tuple[int, int]]]) -> int | None:
    """Which copy of `family` covers gameweek `gw` — 1-indexed, None if none do."""
    for i, (start, stop) in enumerate(windows.get(family, []), start=1):
        if start <= gw <= stop:
            return i
    return None


def canonical_chips(chips_available: list[str] | None) -> set[str]:
    """Normalise squad.yaml chip names to `family` + optional half suffix.

    Accepts `bboost`, `bboost1`, `bboost2`, `wildcard1`, `3xc`, ... An
    unsuffixed name means "a copy of this chip is unused" and matches whichever
    half we are in — the back-compatible reading of pre-2025/26 squad files.
    """
    out: set[str] = set()
    for raw in (chips_available or []):
        name = str(raw).strip().lower()
        if name in CHIP_FAMILIES or (
                name[:-1] in CHIP_FAMILIES and name[-1:] in ("1", "2")):
            out.add(name)
    return out


def playable_now(chips_available: list[str] | None, gw: int,
                 windows: dict[str, list[tuple[int, int]]]) -> list[str]:
    """Families the owner can actually play in gameweek `gw`.

    A chip counts as playable when (a) the gameweek falls inside one of the
    family's windows and (b) the owner still holds the copy for that window
    (an unsuffixed name counts for either).
    """
    held = canonical_chips(chips_available)
    out: list[str] = []
    for family in CHIP_FAMILIES:
        half = chip_half(family, gw, windows)
        if half is None:
            continue
        if family in held or f"{family}{half}" in held:
            out.append(family)
    return out


def valid_chip_names() -> set[str]:
    """Every chip name squad.yaml may legally carry."""
    return set(CHIP_FAMILIES) | {f"{f}{n}" for f in CHIP_FAMILIES for n in (1, 2)}
