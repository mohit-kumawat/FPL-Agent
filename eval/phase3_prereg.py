"""Phase 3 — pre-register a forward prediction, then score it after the fact.

The only uncontaminated test available: 2026-27 is past the model's knowledge
cutoff, so nothing about it can be recalled. The catch is that a forward test is
only honest if the prediction is fixed *before* the deadline and cannot be
quietly revised afterwards. That is what the hash is for.

    uv run python eval/phase3_prereg.py lock <brief.md> [window]  # after a run
    uv run python eval/phase3_prereg.py verify                    # nothing edited?
    uv run python eval/phase3_prereg.py score                     # NOT IMPLEMENTED (exits 2)

`lock` seals the artifacts the owner actually acts on -- the agent's brief and
the decision/captain/XI fields of the pipeline's report -- by READING them off
disk. It never recomputes a recommendation: a fresh `fpl build` here would (a)
seal a from-scratch squad nobody was advised to buy, and (b) be computed from
data fetched after the brief was written, which is exactly the revision the
hash exists to rule out.

Each lock appends an entry; entries are never rewritten. `verify` recomputes
the hash of every stored prediction and fails loudly on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import config  # noqa: E402

LEDGER = Path(__file__).with_name("phase3-predictions.jsonl")


def digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _latest_report() -> tuple[Path | None, dict]:
    reports = sorted(config.REPORTS_DIR.glob("*.json"),
                     key=lambda p: p.stat().st_mtime)
    if not reports:
        return None, {}
    try:
        return reports[-1], json.loads(reports[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return reports[-1], {}


def lock() -> None:
    """Seal the brief + report decision already on disk. No recomputation."""
    if len(sys.argv) < 3:
        print("usage: phase3_prereg.py lock <brief.md> [window]")
        sys.exit(1)
    brief_path = Path(sys.argv[2])
    window = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    try:
        brief = brief_path.read_text()
    except OSError as exc:
        print(f"cannot read brief: {exc}")
        sys.exit(1)

    report_path, report = _latest_report()
    stage = report.get("stage") or {}
    payload = {
        "season": config.CURRENT_SEASON,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": window,
        "gw": stage.get("next_gw"),
        "next_deadline": stage.get("next_deadline"),
        "brief_file": brief_path.name,
        "brief_sha256": hashlib.sha256(brief.encode()).hexdigest(),
        "brief": brief,
        "report_file": report_path.name if report_path else None,
        "decision": report.get("decision"),
        "captain": report.get("captain"),
        "xi": report.get("xi"),
    }
    entry = {"sha256": digest(payload), "payload": payload}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"locked GW{payload['gw']} {window} (deadline {payload['next_deadline']})")
    print(f"sha256 {entry['sha256'][:16]}…  -> {LEDGER.name}")
    if report_path is None:
        print("warning: no report JSON found — sealed the brief alone")


def verify() -> None:
    if not LEDGER.exists():
        print("no predictions locked yet")
        return
    bad = 0
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        e = json.loads(line)
        ok = digest(e["payload"]) == e["sha256"]
        bad += not ok
        p = e["payload"]
        print(f"  [{i}] GW{p.get('gw')} {p.get('window', '?')} "
              f"locked {p['locked_at_utc']} {'OK' if ok else '*** TAMPERED ***'}")
    print("all predictions intact" if not bad else f"{bad} TAMPERED — results void")
    sys.exit(1 if bad else 0)


RESULTS = Path(__file__).with_name("phase3-results.jsonl")


def _gw_actuals(gw: int) -> tuple[dict[str, float], dict[str, float]] | None:
    """(points, minutes) by web_name for a finished GW of the current season.

    Uses the pipeline's own panel cache (element-summary history) plus the
    latest bootstrap for the id -> web_name mapping. Returns None while the
    GW is not finished or the panel does not carry it yet.
    """
    from fpl_agent import data  # deferred: lock/verify must not need pandas

    boot = data.load_snapshot("bootstrap") or data.fetch_bootstrap()
    ev = next((e for e in boot["events"] if e["id"] == gw), None)
    if not ev or not ev.get("finished"):
        return None
    panel = data.build_current_panel(boot)
    if panel.empty or panel["round"].max() < gw:
        return None
    g = panel[panel["round"] == gw].groupby("element")[["total_points", "minutes"]].sum()
    names = {el["id"]: el["web_name"] for el in boot["elements"]}
    counts = Counter(names.values())
    dupes = {n for n, c in counts.items() if c > 1}
    pts: dict[str, float] = {}
    mins: dict[str, float] = {}
    for pid, row in g.iterrows():
        n = names.get(int(pid))
        if n is None or n in dupes:      # ambiguous names cannot be scored
            continue
        pts[n] = float(row["total_points"])
        mins[n] = float(row["minutes"])
    return pts, mins


def _score_entry(payload: dict, pts: dict[str, float],
                 mins: dict[str, float]) -> dict:
    """Component scores for one sealed recommendation. Names, not ids, because
    that is what the sealed report carries."""
    out: dict = {"gw": payload.get("gw"), "window": payload.get("window"),
                 "locked_at_utc": payload.get("locked_at_utc")}

    cap = (payload.get("captain") or {}).get("pick")
    xi = (payload.get("xi") or {}).get("players") or []
    if cap:
        out["captain"] = cap
        out["captain_actual"] = pts.get(cap)
        out["captain_played"] = bool(mins.get(cap, 0) > 0)
    if xi:
        known = [p for p in xi if p in pts]
        out["xi_actual_sum"] = round(sum(pts[p] for p in known), 1)
        out["xi_players_scored"] = f"{len(known)}/{len(xi)}"
        if cap and known:
            out["captain_ceiling_in_xi"] = max(pts[p] for p in known)

    dec = payload.get("decision") or {}
    if dec.get("action") and dec["action"] not in ("hold", "initial_squad_proposal"):
        ins = [p for p in dec.get("in", []) if p in pts]
        outs = [p for p in dec.get("out", []) if p in pts]
        if ins or outs:
            out["transfer_gw_delta"] = round(
                sum(pts[p] for p in ins) - sum(pts[p] for p in outs), 1)
    out["action"] = dec.get("action")
    return out


def score() -> None:
    """Score every intact locked prediction whose GW has finished.

    Appends to phase3-results.jsonl (skipping entries already scored) and
    prints a table. Never touches the predictions ledger.
    """
    if not LEDGER.exists():
        print("no predictions locked yet")
        sys.exit(0)

    done: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                done.add(json.loads(line)["prediction_sha256"])
            except (json.JSONDecodeError, KeyError):
                continue

    actuals: dict[int, tuple | None] = {}
    scored = skipped = 0
    for line in LEDGER.read_text().splitlines():
        e = json.loads(line)
        payload = e["payload"]
        if digest(payload) != e["sha256"]:
            print(f"TAMPERED entry (locked {payload.get('locked_at_utc')}) — not scored")
            continue
        if e["sha256"] in done:
            continue
        gw = payload.get("gw")
        if not gw:
            skipped += 1
            continue
        if gw not in actuals:
            actuals[gw] = _gw_actuals(int(gw))
        if actuals[gw] is None:
            skipped += 1
            continue
        result = _score_entry(payload, *actuals[gw])
        result["prediction_sha256"] = e["sha256"]
        result["scored_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(result) + "\n")
        scored += 1
        cap = result.get("captain")
        cap_s = (f"captain {cap} -> {result.get('captain_actual')} "
                 f"(ceiling {result.get('captain_ceiling_in_xi')})" if cap else "no captain")
        print(f"  GW{gw} {result['window']}: {cap_s}; "
              f"XI {result.get('xi_actual_sum')} "
              f"[{result.get('xi_players_scored', '-')}]"
              + (f"; transfer Δ {result['transfer_gw_delta']:+.1f}"
                 if "transfer_gw_delta" in result else ""))

    print(f"{scored} scored, {skipped} waiting (GW unfinished or no gw recorded)"
          + (f"; results -> {RESULTS.name}" if scored else ""))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"lock": lock, "verify": verify, "score": score}.get(cmd, verify)()
