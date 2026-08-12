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


def score() -> None:
    print("scoring is NOT implemented yet.")
    print("When the gameweeks resolve, score each locked entry's decision/captain/xi")
    print("against actuals with the benchmark arms from eval/agent_backtest.py.")
    sys.exit(2)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"lock": lock, "verify": verify, "score": score}.get(cmd, verify)()
