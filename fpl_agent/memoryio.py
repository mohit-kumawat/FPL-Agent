"""Pipeline memory: state, decision log, prediction log, signals inbox.

Everything is plain files so a headless agent (or a human) can read and
write memory without running Python.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import config

STATE_FILE = config.MEMORY_DIR / "state.json"
DECISIONS_FILE = config.MEMORY_DIR / "decisions.jsonl"
RUNS_FILE = config.MEMORY_DIR / "runs.jsonl"
PREDICTIONS_DIR = config.MEMORY_DIR / "predictions"
SIGNALS_DIR = config.ROOT / "signals"
LEARNINGS_FILE = config.MEMORY_DIR / "learnings.md"

for _d in (PREDICTIONS_DIR, SIGNALS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------- state
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"created": _now(), "watchlist": [], "beliefs": {}}


def save_state(state: dict) -> None:
    state["updated"] = _now()
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ------------------------------------------------------------------- squad
def load_squad() -> dict:
    if config.SQUAD_FILE.exists():
        return yaml.safe_load(config.SQUAD_FILE.read_text()) or {}
    return {}


def squad_selling_prices(squad: dict, players: pd.DataFrame) -> dict[int, float]:
    """FPL selling price: profits are halved, losses are taken in full.

    Rises: purchase + floor(profit / 2) in 0.1m steps.
    Falls: the current price — there is no floor at the purchase price.
    """
    out: dict[int, float] = {}
    now = players.set_index("id")["price"]
    for p in squad.get("players", []) or []:
        pid, buy = int(p["id"]), float(p["purchase_price"])
        cur = float(now.get(pid, buy))
        if cur < buy:
            out[pid] = round(cur, 1)
        else:
            profit_tenths = round((cur - buy) * 10)
            out[pid] = round(buy + (profit_tenths // 2) / 10, 1)
    return out


# ------------------------------------------------------------ append-only logs
def _append(path: Path, record: dict) -> None:
    record = {"ts": _now(), **record}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def log_decision(decision: dict) -> None:
    _append(DECISIONS_FILE, decision)


def log_run(run: dict) -> None:
    _append(RUNS_FILE, run)


def last_decision() -> dict | None:
    if not DECISIONS_FILE.exists():
        return None
    lines = DECISIONS_FILE.read_text().strip().splitlines()
    return json.loads(lines[-1]) if lines else None


def already_ran_today(kind: str) -> bool:
    if not RUNS_FILE.exists():
        return False
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for line in RUNS_FILE.read_text().strip().splitlines():
        r = json.loads(line)
        if r.get("kind") == kind and r.get("ts", "").startswith(today):
            return True
    return False


# -------------------------------------------------------------- predictions
def save_predictions(gw: int, preds: pd.DataFrame) -> None:
    cols = ["id", "web_name", "team_short", "position", "price",
            "ep_ppg", "xmins", "ep_next", "ep_horizon"]
    preds[cols].to_csv(PREDICTIONS_DIR / f"gw{gw:02d}.csv", index=False)


def score_predictions(gw: int, actual: pd.DataFrame) -> dict | None:
    """Compare stored ep_next vs actual GW points (for calibration)."""
    path = PREDICTIONS_DIR / f"gw{gw:02d}.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    merged = pred.merge(actual[["element", "total_points"]],
                        left_on="id", right_on="element")
    if merged.empty:
        return None
    err = (merged["ep_next"] - merged["total_points"]).abs()
    top = merged.nlargest(50, "ep_next")
    return {
        "gw": gw,
        "mae_all": round(float(err.mean()), 3),
        "mae_top50": round(float((top["ep_next"] - top["total_points"]).abs().mean()), 3),
        "n": len(merged),
    }


# ------------------------------------------------------------------ signals
SIGNAL_CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.6, "low": 0.3}
SIGNAL_DEFAULT_TTL_DAYS = 14
EP_NUDGE_LIMIT = 2.0


def _signal_expiry(doc: dict) -> datetime | None:
    """Explicit `expires` wins; otherwise `date` + ttl_days (default 14)."""
    exp = doc.get("expires")
    if exp:
        try:
            return datetime.fromisoformat(str(exp)).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    d = doc.get("date")
    if not d:
        return None
    try:
        ttl = int(doc.get("ttl_days", SIGNAL_DEFAULT_TTL_DAYS))
        return datetime.fromisoformat(str(d)).replace(tzinfo=timezone.utc) + timedelta(days=ttl)
    except ValueError:
        return None


def validate_signal_doc(doc: dict, filename: str = "?") -> list[str]:
    """Structural problems that make a signal unsafe to apply."""
    problems: list[str] = []
    for i, adj in enumerate(doc.get("adjustments", []) or []):
        where = f"{filename}[{i}]"
        try:
            pid = int(adj.get("player_id", 0))
        except (TypeError, ValueError):
            problems.append(f"{where}: player_id is not an integer")
            continue
        if not pid:
            problems.append(f"{where}: missing player_id")
            continue
        lo, hi = adj.get("xmins_min"), adj.get("xmins_max")
        for name, v in (("xmins_min", lo), ("xmins_max", hi)):
            if v is not None and not 0.0 <= float(v) <= 1.0:
                problems.append(f"{where}: {name}={v} outside [0, 1]")
        if lo is not None and hi is not None and float(lo) > float(hi):
            problems.append(
                f"{where}: xmins_min={lo} > xmins_max={hi} — contradictory bounds")
        ep = adj.get("ep_per_gw")
        if ep is not None and abs(float(ep)) > EP_NUDGE_LIMIT:
            problems.append(
                f"{where}: ep_per_gw={ep} exceeds the +/-{EP_NUDGE_LIMIT} limit")
    return problems


def load_signals(now: datetime | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Read signals/*.yaml -> (per-player adjustment frame, raw notes).

    Signal file schema:
      date: 2026-08-15
      source: "press conference"
      confidence: high | medium | low      # weights ep_per_gw 1.0/0.6/0.3
      ttl_days: 14                         # or an explicit `expires: 2026-08-22`
      notes: "free text"
      adjustments:
        - player_id: 448
          ep_per_gw: -0.5     # optional EP/GW nudge, |value| <= 2 (quality news)
          xmins_min: 0.9      # optional floor on expected-minutes fraction
          xmins_max: 0.5      # optional cap ("not a predicted starter")
          reason: "why"

    Minutes news should use xmins_min/xmins_max; use ep_per_gw for quality/role
    information that minutes can't express (pen duty gained, position change).
    Minutes bounds are NOT confidence-weighted — they are facts or they should
    not be written.

    Safety: expired files and files failing validation are ignored entirely and
    reported in the notes, so a stale or malformed signal cannot silently steer
    a recommendation. An autonomous operator must never depend on a human
    remembering to delete a file.
    """
    now = now or datetime.now(timezone.utc)
    rows: dict[int, dict] = {}
    notes: list[dict] = []
    for f in sorted(SIGNALS_DIR.glob("*.y*ml")):
        note = {"file": f.name, "applied": False, "problems": []}
        try:
            doc = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError as exc:
            note["problems"] = [f"unparseable YAML: {exc}"]
            notes.append(note)
            continue
        note.update({k: doc.get(k) for k in ("date", "source", "notes", "confidence")})

        expiry = _signal_expiry(doc)
        if expiry and expiry < now:
            note["problems"] = [f"expired {expiry:%Y-%m-%d} — ignored (delete or refresh)"]
            notes.append(note)
            continue
        problems = validate_signal_doc(doc, f.name)
        if problems:
            note["problems"] = problems
            notes.append(note)
            continue

        weight = SIGNAL_CONFIDENCE_WEIGHT.get(
            str(doc.get("confidence", "medium")).lower(), 0.6)
        for adj in doc.get("adjustments", []) or []:
            pid = int(adj["player_id"])
            r = rows.setdefault(pid, {"ep_per_gw": 0.0, "xmins_min": None, "xmins_max": None})
            r["ep_per_gw"] += weight * float(adj.get("ep_per_gw", 0.0))
            if adj.get("xmins_min") is not None:
                r["xmins_min"] = max(r["xmins_min"] or 0.0, float(adj["xmins_min"]))
            if adj.get("xmins_max") is not None:
                r["xmins_max"] = min(r["xmins_max"] if r["xmins_max"] is not None else 1.0,
                                     float(adj["xmins_max"]))
        note["applied"] = True
        notes.append(note)

    # cross-file contradictions: a floor above a cap cannot both be honoured
    for pid, r in rows.items():
        if r["xmins_min"] is not None and r["xmins_max"] is not None \
                and r["xmins_min"] > r["xmins_max"]:
            r["xmins_min"] = r["xmins_max"] = None
            notes.append({"file": "(merged)", "applied": False, "problems": [
                f"player {pid}: signals from different files set a floor above a "
                "cap — both minutes bounds dropped"]})

    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame, notes
