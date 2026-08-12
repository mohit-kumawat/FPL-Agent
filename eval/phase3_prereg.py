"""Phase 3 — pre-register a forward prediction, then score it after the fact.

The only uncontaminated test available: 2026-27 is past the model's knowledge
cutoff, so nothing about it can be recalled. The catch is that a forward test is
only honest if the prediction is fixed *before* the deadline and cannot be
quietly revised afterwards. That is what the hash is for.

    uv run python eval/phase3_prereg.py lock      # before a deadline
    uv run python eval/phase3_prereg.py verify    # confirm nothing was edited
    uv run python eval/phase3_prereg.py score     # after the GWs have resolved

Each lock appends an entry; entries are never rewritten. `verify` recomputes the
hash of every stored prediction and fails loudly on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import config  # noqa: E402

LEDGER = Path(__file__).with_name("phase3-predictions.jsonl")
ROOT = Path(__file__).resolve().parent.parent


def digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def lock() -> None:
    """Capture the pipeline's current recommendation and seal it."""
    res = subprocess.run(["uv", "run", "fpl", "build"], cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        print("build failed:\n", res.stderr[-2000:])
        sys.exit(1)

    status = subprocess.run(["uv", "run", "fpl", "status"], cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    try:
        st = json.loads(status.stdout)
    except json.JSONDecodeError:
        st = {"raw": status.stdout[-500:]}

    payload = {
        "season": config.CURRENT_SEASON,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": st.get("stage"),
        "next_gw": st.get("next_gw"),
        "next_deadline": st.get("next_deadline"),
        "recommendation": res.stdout.strip(),
    }
    entry = {"sha256": digest(payload), "payload": payload}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"locked {payload['next_deadline']} (GW{payload['next_gw']})")
    print(f"sha256 {entry['sha256'][:16]}…  -> {LEDGER.name}")


def verify() -> None:
    if not LEDGER.exists():
        print("no predictions locked yet"); return
    bad = 0
    for i, line in enumerate(LEDGER.read_text().splitlines(), 1):
        e = json.loads(line)
        ok = digest(e["payload"]) == e["sha256"]
        bad += not ok
        p = e["payload"]
        print(f"  [{i}] GW{p.get('next_gw')} locked {p['locked_at_utc']} "
              f"{'OK' if ok else '*** TAMPERED ***'}")
    print("all predictions intact" if not bad else f"{bad} TAMPERED — results void")
    sys.exit(1 if bad else 0)


def score() -> None:
    """Score locked predictions once the gameweeks have actually happened."""
    if not LEDGER.exists():
        print("nothing to score"); return
    print("Scoring needs the season's results in data/history/ "
          f"(merged_gw_{config.CURRENT_SEASON}.parquet).")
    hist = ROOT / "data" / "history" / f"merged_gw_{config.CURRENT_SEASON}.parquet"
    if not hist.exists():
        print(f"  not available yet: {hist.name}")
        print("  re-run after the gameweeks resolve and the season file exists.")
        return
    print("  season file present — extend this function with the same "
          "benchmark arms used in eval/agent_backtest.py.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"lock": lock, "verify": verify, "score": score}.get(cmd, verify)()
