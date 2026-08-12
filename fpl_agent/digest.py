"""The season digest: what one run hands to the next.

Each scheduled run is a FRESH agent session with no conversation history, so
every piece of continuity has to survive as a file. Before this module the
handoff was one decision wide — `fpl status` plus today's report — which meant
the agent in week 30 knew exactly what the agent in week 2 knew. Predictions,
scored outcomes and decision history all accumulated unread.

So the pipeline compacts them here, every run, into two artifacts:

  memory/digest.json          machine-readable, for tooling
  memory/current-context.md   the ONLY season memory the next run is handed

The rule that keeps this useful is that it is *generated and bounded*, never
appended to: raw briefs, full decision history and expired research stay on disk
addressed by id, and only the compacted view travels forward. An agent cannot
forget to maintain it, and it cannot grow without limit.
"""
from __future__ import annotations

from datetime import datetime, timezone

from . import config, data, memoryio, rules

CONTEXT_FILE = config.MEMORY_DIR / "current-context.md"
DIGEST_FILE = config.MEMORY_DIR / "digest.json"
DECISION_HISTORY = 6
CALIBRATION_HISTORY = 6
EXPIRY_WARN_GWS = 5          # how close the split must be before we nag about it


def _compact_decisions(rows: list[dict]) -> list[dict]:
    """Collapse consecutive identical decisions into one line with a count.

    Several runs a day each append a decision, so a quiet week otherwise fills
    the handoff with six copies of the same hold. The agent needs the shape of
    the history, not its volume.
    """
    out: list[dict] = []
    for r in rows:
        key = (r.get("action"), r.get("captain"))
        if out and (out[-1].get("action"), out[-1].get("captain")) == key:
            out[-1]["repeats"] = out[-1].get("repeats", 1) + 1
            out[-1]["date"] = r.get("date")       # keep the most recent date
            continue
        out.append({k: r.get(k) for k in ("date", "action", "captain", "net_gain")})
    return out


def _chip_state(boot: dict, squad: dict) -> dict:
    """Which chips remain, which are live now, and what expires when."""
    nxt = next((e for e in boot.get("events", []) if e.get("is_next")), None)
    gw = int(nxt["id"]) if nxt else 0
    windows = rules.chip_windows(boot)
    held = sorted(rules.canonical_chips(squad.get("chips_available")))
    playable = rules.playable_now(squad.get("chips_available"), gw, windows)

    # First-half copies are lost at the split. Only raise this once the split is
    # actually near — warning from GW1 would cry wolf for eighteen gameweeks and
    # train the agent to ignore the line that matters in December.
    expiring: list[str] = []
    for family in rules.CHIP_FAMILIES:
        wins = windows.get(family) or []
        if not wins:
            continue
        first_stop = wins[0][1]
        near = 0 < first_stop - gw <= EXPIRY_WARN_GWS
        if near and (f"{family}1" in held or family in held):
            expiring.append(f"{rules.CHIP_LABELS[family]} "
                            f"(by GW{first_stop}, {first_stop - gw} GWs left)")
    return {"held": held, "playable_now": playable,
            "first_half_expiring": expiring,
            "windows": {k: [list(w) for w in v] for k, v in windows.items()}}


def _season_trajectory(squad: dict) -> dict:
    """Real points and rank, if the owner's entry id is known. Never fatal."""
    entry_id = squad.get("entry_id")
    if not entry_id:
        return {"available": False, "why": "no entry_id in squad.yaml yet"}
    try:
        entry = data.fetch_entry(int(entry_id))
    except Exception as exc:  # noqa: BLE001 — trajectory is context, not a blocker
        return {"available": False, "why": f"entry fetch failed: {exc}"}
    return {"available": True,
            "total_points": entry.get("summary_overall_points"),
            "overall_rank": entry.get("summary_overall_rank"),
            "gw_points": entry.get("summary_event_points"),
            "value": (entry.get("last_deadline_value") or 0) / 10.0,
            "bank": (entry.get("last_deadline_bank") or 0) / 10.0}


def build(boot: dict, fixtures: list[dict], squad: dict, state: dict,
         freshness: dict | None = None, verification: dict | None = None) -> dict:
    """Assemble the digest. Pure read — callers persist it with `write`."""
    nxt = next((e for e in boot.get("events", []) if e.get("is_next")), None)
    cur = next((e for e in boot.get("events", []) if e.get("is_current")), None)
    cal = memoryio.calibration_history(CALIBRATION_HISTORY)
    maes = [c["mae_all"] for c in cal if c.get("mae_all") is not None]

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": config.CURRENT_SEASON,
        "gameweek": {
            "next": nxt["id"] if nxt else None,
            "deadline": nxt.get("deadline_time") if nxt else None,
            "current": cur["id"] if cur else None,
            "finished": sum(1 for e in boot.get("events", []) if e.get("finished")),
        },
        "squad": {
            "has_team": len(squad.get("players") or []) == config.SQUAD_SIZE,
            "bank": squad.get("bank"),
            "free_transfers": squad.get("free_transfers"),
            "strategy_mode": str(squad.get("strategy_mode", "safe")).lower(),
        },
        "chips": _chip_state(boot, squad),
        "trajectory": _season_trajectory(squad),
        "model_calibration": {
            "gws_scored": len(cal),
            "recent_mae": [{"gw": c["gw"], "mae_all": c["mae_all"]} for c in cal],
            "mae_trend": (round(maes[-1] - maes[0], 2)
                          if len(maes) >= 2 else None),
            "unscored_pending": sorted(
                set(range(1, (nxt["id"] if nxt else 1)))
                - {c["gw"] for c in memoryio.calibration_history(50)}
            )[:5],
        },
        "research_accuracy": memoryio.signal_scorecard(),
        "recent_decisions": _compact_decisions(
            memoryio.recent_decisions(DECISION_HISTORY * 4))[-DECISION_HISTORY:],
        "open_items": [p.get("text") for p in (state.get("pending_snapshot") or [])],
        "data_freshness": {
            "fetched_at": (freshness or {}).get("fetched_at"),
            "blockers": (verification or {}).get("blockers") or [],
            "warnings": (verification or {}).get("warnings") or [],
        },
    }


def render(d: dict) -> str:
    """The compact context file. Deliberately short: this is a prompt input, so
    every line has to earn its place against the runbook it competes with."""
    gw, sq, ch = d["gameweek"], d["squad"], d["chips"]
    cal, ra, tr = d["model_calibration"], d["research_accuracy"], d["trajectory"]
    L = [f"# Operating context — generated {d['generated_utc']}", "",
         "Generated every run; the only season memory carried forward. Full history "
         "lives in memory/ and reports/ by id — read it only if you need it.", "",
         f"- **Season/GW**: {d['season']}, next GW{gw['next']} "
         f"(deadline {gw['deadline']}); {gw['finished']} finished",
         f"- **Squad**: {'saved' if sq['has_team'] else 'NOT SAVED'}, "
         f"bank £{sq['bank']}m, {sq['free_transfers']} FT, mode `{sq['strategy_mode']}`"]

    if tr.get("available"):
        L.append(f"- **Trajectory**: {tr['total_points']} pts, overall rank "
                 f"{tr['overall_rank']}, last GW {tr['gw_points']}, "
                 f"squad value £{tr['value']}m")
    else:
        L.append(f"- **Trajectory**: unavailable ({tr.get('why')})")

    L.append(f"- **Chips held**: {', '.join(ch['held']) or 'none'}"
             + (f" | playable in GW{gw['next']}: {', '.join(ch['playable_now'])}"
                if ch["playable_now"] else " | none playable this GW"))
    if ch["first_half_expiring"]:
        L.append(f"- **Expiring at the split**: {'; '.join(ch['first_half_expiring'])}"
                 " — use-it-or-lose-it, raise under Needs you")

    if cal["recent_mae"]:
        trend = cal["mae_trend"]
        arrow = "" if trend is None else (
            f" ({'improving' if trend < 0 else 'worsening'} {abs(trend):.2f})")
        L.append("- **Model calibration** (MAE, recent GWs): "
                 + ", ".join(f"GW{c['gw']} {c['mae_all']}" for c in cal["recent_mae"])
                 + arrow)
    if cal["unscored_pending"]:
        L.append(f"- **Awaiting scoring**: GW{cal['unscored_pending']} "
                 "(deferred until points are final — expected, not a fault)")

    if ra.get("accuracy") is not None:
        line = (f"- **Your research accuracy**: {ra['accuracy']:.0%} "
                f"({ra['hits']} right / {ra['misses']} wrong minutes claims over "
                f"{ra['gws_scored']} GWs)")
        if ra.get("least_reliable"):
            line += " — least reliable: " + ", ".join(
                f"{s['source']} ({s['hit']}/{s['hit'] + s['miss']})"
                for s in ra["least_reliable"])
        L.append(line)
    else:
        L.append("- **Your research accuracy**: no scored minutes claims yet")

    if d["recent_decisions"]:
        L += ["", "## Recent decisions (newest last)"]
        for r in d["recent_decisions"]:
            bits = [str(r.get("date")), str(r.get("action"))]
            if r.get("captain"):
                bits.append(f"C: {r['captain']}")
            if r.get("net_gain") is not None:
                bits.append(f"{r['net_gain']:+} EP")
            if r.get("repeats", 1) > 1:
                bits.append(f"x{r['repeats']} runs")
            L.append("- " + " | ".join(bits))
        L.append("Explain any flip against the previous line.")

    fr = d["data_freshness"]
    if fr["blockers"] or fr["warnings"]:
        L += ["", "## Data caveats"]
        L += [f"- BLOCKER: {b}" for b in fr["blockers"]]
        L += [f"- warning: {w}" for w in fr["warnings"][:4]]
    if d["open_items"]:
        L += ["", "## Open items"] + [f"- [ ] {t}" for t in d["open_items"][:6]]
    return "\n".join(L) + "\n"


def write(d: dict) -> tuple[str, str]:
    """Persist both artifacts. Returns (json_path, md_path)."""
    import json
    DIGEST_FILE.write_text(json.dumps(d, indent=2, default=str))
    CONTEXT_FILE.write_text(render(d))
    return str(DIGEST_FILE), str(CONTEXT_FILE)
