"""External data in SHADOW MODE: measured against reality, never trusted by EP.

The pipeline's only numerical inputs are official FPL data. External sources
(odds, projected lineups) might beat the fixture model — but "might" is exactly
the claim the eval discipline exists to test. So external data enters through
this adapter and goes nowhere else:

  data/external/inbox/*.json       operator drops provider payloads here
  data/external/snapshots/         timestamped, hashed, provider-versioned copies
  memory/shadow_scores.jsonl       Brier / log-loss vs official results, per GW

Nothing in models.py or optimizer.py imports this module. Promotion out of
shadow mode is a pre-registered decision (eval/ discipline, >= 20 scored GWs),
made by a human reading the ledger — never by the pipeline drifting into it.
The adapter fails closed: a malformed payload is reported and skipped, and no
failure here may abort a daily run.

Payload schema (provider-neutral):
  {"provider": "oddsfeed", "version": "1", "gw": 3,
   "probabilities": [
     {"team_h": "ARS", "team_a": "CHE",
      "p_home": 0.52, "p_draw": 0.26, "p_away": 0.22,
      "p_cs_home": 0.34, "p_cs_away": 0.18}]}       # cs_* optional
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import config

EXTERNAL_DIR = config.DATA_DIR / "external"
INBOX_DIR = EXTERNAL_DIR / "inbox"
SNAP_DIR = EXTERNAL_DIR / "snapshots"
SHADOW_FILE = config.MEMORY_DIR / "shadow_scores.jsonl"
PROB_TOL = 0.02              # p_home + p_draw + p_away must sum to 1 within this


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate(payload: dict) -> list[str]:
    problems = []
    if not payload.get("provider"):
        problems.append("missing provider")
    try:
        int(payload.get("gw"))
    except (TypeError, ValueError):
        problems.append("missing/invalid gw")
    rows = payload.get("probabilities")
    if not isinstance(rows, list) or not rows:
        problems.append("missing probabilities list")
        return problems
    for i, r in enumerate(rows):
        try:
            p = [float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])]
        except (KeyError, TypeError, ValueError):
            problems.append(f"row {i}: p_home/p_draw/p_away missing or non-numeric")
            continue
        if any(not 0.0 <= x <= 1.0 for x in p) or abs(sum(p) - 1.0) > PROB_TOL:
            problems.append(f"row {i}: probabilities invalid (sum {sum(p):.3f})")
        if not r.get("team_h") or not r.get("team_a"):
            problems.append(f"row {i}: missing team_h/team_a")
    return problems


def ingest(now: str | None = None) -> list[dict]:
    """Move valid inbox payloads into hashed snapshots. Fail closed: malformed
    files stay in the inbox (visible, actionable) and are reported, not raised.
    Returns one note per inbox file."""
    notes: list[dict] = []
    if not INBOX_DIR.is_dir():
        return notes
    for path in sorted(INBOX_DIR.glob("*.json")):
        note = {"file": path.name, "ingested": False, "problems": []}
        notes.append(note)
        try:
            raw = path.read_text()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            note["problems"].append(f"unreadable: {exc}")
            continue
        problems = validate(payload)
        if problems:
            note["problems"] = problems
            continue
        sha = hashlib.sha256(raw.encode()).hexdigest()
        meta = {"provider": payload["provider"],
                "version": str(payload.get("version", "?")),
                "gw": int(payload["gw"]), "retrieved_at": now or _now(),
                "sha256": sha, "source_file": path.name}
        SNAP_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{meta['provider']}_gw{meta['gw']:02d}_{sha[:8]}"
        (SNAP_DIR / f"{stem}.json").write_text(raw)
        (SNAP_DIR / f"{stem}.meta.json").write_text(json.dumps(meta))
        path.unlink()
        note.update(ingested=True, snapshot=f"{stem}.json")
    return notes


def snapshots(gw: int | None = None) -> list[tuple[dict, dict]]:
    """[(meta, payload)] for stored snapshots, optionally one gameweek's."""
    out = []
    if not SNAP_DIR.is_dir():
        return out
    for meta_path in sorted(SNAP_DIR.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
            if gw is not None and int(meta.get("gw", -1)) != gw:
                continue
            payload = json.loads(
                meta_path.with_name(meta_path.name.replace(".meta", "")).read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        out.append((meta, payload))
    return out


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


def _outcomes(gw: int, fixtures: list[dict], boot: dict) -> dict[tuple[str, str], dict]:
    """(home_short, away_short) -> result for finished fixtures of a gameweek."""
    teams = {t["id"]: t["short_name"] for t in (boot or {}).get("teams", [])}
    out: dict[tuple[str, str], dict] = {}
    for fx in fixtures or []:
        if fx.get("event") != gw or not fx.get("finished"):
            continue
        h, a = teams.get(fx.get("team_h")), teams.get(fx.get("team_a"))
        hs, as_ = fx.get("team_h_score"), fx.get("team_a_score")
        if h is None or a is None or hs is None or as_ is None:
            continue
        res = "home" if hs > as_ else ("away" if as_ > hs else "draw")
        out[(h, a)] = {"result": res, "cs_home": as_ == 0, "cs_away": hs == 0}
    return out


def score_gw(gw: int, fixtures: list[dict], boot: dict) -> list[dict]:
    """Score every provider's stored probabilities for one finished gameweek.

    Brier (multiclass, lower better) and log-loss vs official results. Appends
    to the shadow ledger once per (provider, gw); returns the new records.
    """
    done = {(r.get("provider"), r.get("gw")) for r in _read_jsonl(SHADOW_FILE)}
    actual = _outcomes(gw, fixtures, boot)
    records = []
    for meta, payload in snapshots(gw):
        if (meta["provider"], gw) in done or not actual:
            continue
        done.add((meta["provider"], gw))   # two snapshots, one score — the
        # ledger is per (provider, gw), never per payload revision
        briers, loglosses, cs_hits, cs_total = [], [], 0, 0
        for r in payload.get("probabilities", []):
            key = (r.get("team_h"), r.get("team_a"))
            if key not in actual:
                continue
            probs = {"home": float(r["p_home"]), "draw": float(r["p_draw"]),
                     "away": float(r["p_away"])}
            res = actual[key]["result"]
            briers.append(sum((p - (1.0 if k == res else 0.0)) ** 2
                              for k, p in probs.items()))
            loglosses.append(-math.log(max(probs[res], 1e-9)))
            for side, cs_key in (("home", "p_cs_home"), ("away", "p_cs_away")):
                if r.get(cs_key) is not None:
                    cs_total += 1
                    cs_hits += (float(r[cs_key]) >= 0.5) == actual[key][f"cs_{side}"]
        if not briers:
            continue
        rec = {"ts": _now(), "provider": meta["provider"],
               "version": meta.get("version"), "gw": int(gw),
               "fixtures_scored": len(briers),
               "brier": round(sum(briers) / len(briers), 4),
               "log_loss": round(sum(loglosses) / len(loglosses), 4),
               "cs_accuracy": (round(cs_hits / cs_total, 3) if cs_total else None)}
        with SHADOW_FILE.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        records.append(rec)
    return records


def shadow_summary() -> dict:
    """Season-to-date shadow record per provider — what a promotion decision
    would read. gws_scored counts toward the >= 20 pre-registered threshold."""
    rows = _read_jsonl(SHADOW_FILE)
    by_provider: dict[str, dict] = {}
    for r in rows:
        cur = by_provider.setdefault(r.get("provider", "?"),
                                     {"gws_scored": 0, "brier": [], "log_loss": []})
        cur["gws_scored"] += 1
        cur["brier"].append(r.get("brier"))
        cur["log_loss"].append(r.get("log_loss"))
    return {p: {"gws_scored": c["gws_scored"],
                "brier_mean": round(sum(c["brier"]) / len(c["brier"]), 4),
                "log_loss_mean": round(sum(c["log_loss"]) / len(c["log_loss"]), 4),
                "promotable": c["gws_scored"] >= 20}
            for p, c in by_provider.items() if c["brier"]}


def run_shadow(boot: dict, fixtures: list[dict]) -> dict:
    """The daily hook: ingest new payloads, score finished gameweeks. Never
    raises — shadow accounting must not break a run — and by construction never
    feeds anything back into EP."""
    out: dict = {"ingested": [], "scored": [], "errors": []}
    try:
        out["ingested"] = ingest()
        finished = sorted(int(e["id"]) for e in (boot or {}).get("events", [])
                          if e.get("finished"))
        gws_with_data = {int(m.get("gw", -1)) for m, _ in snapshots()}
        for gw in finished:
            if gw in gws_with_data:
                out["scored"].extend(score_gw(gw, fixtures, boot))
    except Exception as exc:  # noqa: BLE001 — fail closed, report, continue
        out["errors"].append(str(exc))
    return out
