"""Pipeline memory: state, decision log, prediction log, signals inbox.

Everything is plain files so a headless agent (or a human) can read and
write memory without running Python.
"""
from __future__ import annotations

import json
import re
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
# Append-only evidence that the digest compacts into next run's context.
CALIBRATION_FILE = config.MEMORY_DIR / "calibration.jsonl"
SIGNAL_LOG_FILE = config.MEMORY_DIR / "signal_log.jsonl"
SIGNAL_SCORES_FILE = config.MEMORY_DIR / "signal_scores.jsonl"

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


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def log_calibration(score: dict) -> None:
    """Persist a scored gameweek so the digest can show a trend, not one number."""
    if not any(r.get("gw") == score.get("gw") for r in _read_jsonl(CALIBRATION_FILE)):
        _append(CALIBRATION_FILE, score)


def calibration_history(last: int = 6) -> list[dict]:
    return sorted(_read_jsonl(CALIBRATION_FILE), key=lambda r: r.get("gw", 0))[-last:]


def log_applied_signals(gw: int, frame: pd.DataFrame) -> None:
    """Record which minutes claims were live for a gameweek, with attribution.

    Written at prediction time because that is the only moment we know what the
    recommendation was actually based on. Scored after the gameweek by
    score_signals(), which is how the agent finds out whether its research was
    any good — the model's own accuracy never measured that.
    """
    if frame is None or not len(frame):
        return
    seen = {(r.get("gw"), r.get("player_id")) for r in _read_jsonl(SIGNAL_LOG_FILE)}
    for pid, row in frame.iterrows():
        lo, hi = row.get("xmins_min"), row.get("xmins_max")
        if pd.isna(lo) and pd.isna(hi):
            continue                      # ep_per_gw-only: nothing minutes-shaped
        if (gw, int(pid)) in seen:
            continue
        _append(SIGNAL_LOG_FILE, {
            "gw": int(gw), "player_id": int(pid),
            "xmins_min": None if pd.isna(lo) else float(lo),
            "xmins_max": None if pd.isna(hi) else float(hi),
            "sources": row.get("sources") or "",
        })


def score_signals(gw: int, actual: pd.DataFrame) -> dict | None:
    """Did the gameweek's minutes claims hold? Scored only once per gameweek.

    A floor claim (`xmins_min`) predicts the player plays roughly that share of
    90; a cap (`xmins_max`) predicts he does not exceed it. Both are checked
    against real minutes with a 15-minute tolerance, because a claim of "starts"
    is not falsified by an 80th-minute substitution.
    """
    claims = [r for r in _read_jsonl(SIGNAL_LOG_FILE) if r.get("gw") == gw]
    if not claims:
        return None
    if any(r.get("gw") == gw for r in _read_jsonl(SIGNAL_SCORES_FILE)):
        return None                       # already scored; never double-count
    mins = actual.groupby("element")["minutes"].sum() if len(actual) else pd.Series(dtype=float)

    tol = 15.0
    hits, misses, by_source, unscored = 0, 0, {}, 0
    for c in claims:
        pid = int(c["player_id"])
        if pid not in mins.index:
            unscored += 1                 # not in the panel: no evidence either way
            continue
        played = float(mins.loc[pid])
        lo, hi = c.get("xmins_min"), c.get("xmins_max")
        ok = True
        if lo is not None:
            ok = ok and played >= float(lo) * 90.0 - tol
        if hi is not None:
            ok = ok and played <= float(hi) * 90.0 + tol
        hits, misses = (hits + 1, misses) if ok else (hits, misses + 1)
        for src in [s for s in str(c.get("sources", "")).split(",") if s]:
            agg = by_source.setdefault(src, {"hit": 0, "miss": 0})
            agg["hit" if ok else "miss"] += 1

    record = {"gw": int(gw), "claims": len(claims), "hits": hits, "misses": misses,
              "unscored": unscored, "by_source": by_source}
    _append(SIGNAL_SCORES_FILE, record)
    return record


def signal_scorecard() -> dict:
    """Season-to-date research accuracy, and the least reliable sources."""
    rows = _read_jsonl(SIGNAL_SCORES_FILE)
    hits = sum(r.get("hits", 0) for r in rows)
    misses = sum(r.get("misses", 0) for r in rows)
    by_source: dict[str, dict] = {}
    for r in rows:
        for src, agg in (r.get("by_source") or {}).items():
            cur = by_source.setdefault(src, {"hit": 0, "miss": 0})
            cur["hit"] += agg.get("hit", 0)
            cur["miss"] += agg.get("miss", 0)
    worst = sorted((s for s in by_source.items() if sum(s[1].values()) >= 2),
                   key=lambda kv: kv[1]["hit"] / max(1, sum(kv[1].values())))[:3]
    return {"gws_scored": len(rows), "hits": hits, "misses": misses,
            "accuracy": round(hits / (hits + misses), 2) if hits + misses else None,
            "least_reliable": [{"source": s, **a} for s, a in worst]}


# ------------------------------------------------------------------ learnings
LEARNING_TIERS = ("VALIDATED", "OBSERVED FACTS", "HYPOTHESES")
TIER_CAPS = {"VALIDATED": 12, "OBSERVED FACTS": 20, "HYPOTHESES": 15}


def validate_learnings(text: str | None = None) -> list[str]:
    """Structural problems in learnings.md.

    The three-tier split exists so a hunch cannot become a model constant, but
    prose alone does not enforce it — over 40 weeks an unchecked file either
    bloats or turns into folklore. Every entry therefore needs a date and, in
    VALIDATED, a pointer to the evidence that promoted it. Warnings only: this
    file is the agent's notebook, not a gate on recommendations.
    """
    if text is None:
        text = LEARNINGS_FILE.read_text() if LEARNINGS_FILE.exists() else ""
    if not text.strip():
        return []
    problems: list[str] = []
    counts = dict.fromkeys(LEARNING_TIERS, 0)

    # An entry is a bullet plus its indented continuation lines, so validate the
    # whole entry: a date or evidence pointer often sits on a later line.
    entries: list[tuple[int, str, str]] = []       # (line_no, tier, full text)
    tier: str | None = None
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if raw.startswith("#"):
            # a non-tier heading (e.g. "Process rules") ends the tiered section;
            # only tiered claims are held to the evidence rules
            head = line.lstrip("#").strip().upper()
            tier = next((t for t in LEARNING_TIERS if head.startswith(t)), None)
            continue
        if line.startswith(("- ", "* ")):
            if tier is not None:
                entries.append((i, tier, line))
        elif entries and line and raw[:1].isspace() and tier is not None:
            i0, t0, body = entries[-1]
            entries[-1] = (i0, t0, f"{body} {line}")

    for i, t, body in entries:
        counts[t] += 1
        if not re.search(r"\d{4}-\d{2}-\d{2}", body):
            problems.append(f"line {i} ({t}): entry has no YYYY-MM-DD date")
        if t == "VALIDATED" and not re.search(
                r"eval/|report|GW\d+|MAE|rho|coverage|CI |\d+\s*pts", body):
            problems.append(f"line {i} (VALIDATED): no evidence pointer — cite the "
                            "eval report, metric, or gameweek that promoted it")
    for t, n in counts.items():
        if n > TIER_CAPS[t]:
            problems.append(f"{t} has {n} entries (cap {TIER_CAPS[t]}) — prune or "
                            "promote before adding more")
    return problems


def recent_decisions(last: int = 6) -> list[dict]:
    """Compact decision history — the digest passes these instead of one entry."""
    return _read_jsonl(DECISIONS_FILE)[-last:]


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

# Structured minutes vocabulary: the agent writes the FACT it read, not a
# number it invented. Each role expands to the xmins bounds below and merges
# with any explicit bounds exactly as if the agent had written them (floors
# take the max, caps the min). Numbers stay legal for cases the vocabulary
# can't express, but the vocabulary is the preferred interface.
SIGNAL_ROLES = {
    "expected_starter":     {"xmins_min": 0.85},
    "rotation_risk":        {"xmins_max": 0.70},
    "managed_minutes":      {"xmins_max": 0.75},
    "bench_role":           {"xmins_max": 0.45},
    "not_in_predicted_xi":  {"xmins_max": 0.35},
    "ruled_out":            {"xmins_max": 0.05},
}


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


def adjustment_bounds(adj: dict) -> dict:
    """Minutes bounds implied by one adjustment: the role's defaults with any
    explicit xmins_min/xmins_max layered on top.

    Single source of truth for BOTH validation and application — computing the
    bounds only at apply time let a role floor contradicting an explicit cap
    slip past validate_signal_doc entirely.
    """
    bounds = dict(SIGNAL_ROLES.get(adj.get("role"), {}))
    for k in ("xmins_min", "xmins_max"):
        if adj.get(k) is not None:
            bounds[k] = float(adj[k])
    return bounds


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
        role = adj.get("role")
        if role is not None and role not in SIGNAL_ROLES:
            problems.append(f"{where}: unknown role '{role}' — valid: "
                            + ", ".join(sorted(SIGNAL_ROLES)))
        # validate the EFFECTIVE bounds, so a role default that contradicts an
        # explicit bound is rejected like any other contradiction
        eff = adjustment_bounds(adj)
        lo, hi = eff.get("xmins_min"), eff.get("xmins_max")
        for name, v in (("xmins_min", lo), ("xmins_max", hi)):
            if v is not None and not 0.0 <= float(v) <= 1.0:
                problems.append(f"{where}: {name}={v} outside [0, 1]")
        if lo is not None and hi is not None and float(lo) > float(hi):
            via = f" (role '{role}' implies " if role else " ("
            problems.append(
                f"{where}: xmins_min={lo} > xmins_max={hi} — contradictory bounds"
                + (f"{via}{SIGNAL_ROLES.get(role, {})})" if role else ")"))
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
          ep_per_gw: -0.5     # LAST RESORT: |value| <= 2 enforced here, but
                              # policy (AGENT.md) is +/-0.5 max, direct quote
                              # in `source`, role facts only
          xmins_min: 0.9      # optional floor on expected-minutes fraction
          xmins_max: 0.5      # optional cap ("not a predicted starter")
          reason: "why"

    Minutes news should use xmins_min/xmins_max — signals carry minutes FACTS,
    not quality opinions. ep_per_gw exists only for role information minutes
    can't express (pen duty gained, position change).
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
            r = rows.setdefault(pid, {"ep_per_gw": 0.0, "xmins_min": None,
                                      "xmins_max": None, "sources": ""})
            # attribution: which file(s) moved this player, so score_signals can
            # tell the agent WHICH source keeps being wrong
            srcs = [s for s in r["sources"].split(",") if s]
            if f.name not in srcs:
                srcs.append(f.name)
            r["sources"] = ",".join(srcs)
            r["ep_per_gw"] += weight * float(adj.get("ep_per_gw", 0.0))
            bounds = adjustment_bounds(adj)   # role defaults + explicit overrides
            if bounds.get("xmins_min") is not None:
                r["xmins_min"] = max(r["xmins_min"] or 0.0, float(bounds["xmins_min"]))
            if bounds.get("xmins_max") is not None:
                r["xmins_max"] = min(r["xmins_max"] if r["xmins_max"] is not None else 1.0,
                                     float(bounds["xmins_max"]))
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


def load_scenarios(now: datetime | None = None) -> list[dict]:
    """Future double/blank gameweek scenarios from signals/*.yaml.

    Schema (top-level key next to `adjustments`):
      scenarios:
        - gw: 29
          kind: double        # or blank
          prob: 0.7
          note: "cup QF weekend rearrangements"

    These are research facts for the chip EV engine — the pipeline computes
    EV *given* them, the agent supplies them. Malformed or expired entries
    are dropped silently here; load_signals already reports file problems.
    """
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    for f in sorted(SIGNALS_DIR.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(f.read_text()) or {}
        except yaml.YAMLError:
            continue
        expiry = _signal_expiry(doc)
        if expiry and expiry < now:
            continue
        for s in doc.get("scenarios", []) or []:
            try:
                gw = int(s["gw"])
                kind = str(s["kind"]).lower()
                prob = float(s.get("prob", 0.5))
            except (KeyError, TypeError, ValueError):
                continue
            if kind in ("double", "blank") and 0.0 <= prob <= 1.0 and 1 <= gw <= 38:
                out.append({"gw": gw, "kind": kind, "prob": prob,
                            "note": s.get("note", ""), "file": f.name})
    return out
