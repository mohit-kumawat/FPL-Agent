"""Pre-flight verification: prove the state is correct BEFORE any analysis runs.

Recommendations built on a wrong squad, stale prices, or a miscounted free
transfer are worse than no recommendations. Every daily run starts here; blocking
problems stop the run, warnings go to the report.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, data


def verify_state(boot: dict, squad: dict, fixtures: list[dict],
                 freshness: dict | None = None) -> dict:
    """Returns {'ok': bool, 'blockers': [...], 'warnings': [...], 'checks': [...]}"""
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    # ---- 0. price freshness -------------------------------------------
    if freshness:
        fetched = freshness.get("fetched_at")
        nxt_ev = next((e for e in boot["events"] if e["is_next"]), None)
        if fetched and nxt_ev:
            age_h = (datetime.now(timezone.utc)
                     - datetime.fromisoformat(fetched)).total_seconds() / 3600
            dl = datetime.fromisoformat(nxt_ev["deadline_time"].replace("Z", "+00:00"))
            hrs_to_dl = (dl - datetime.now(timezone.utc)).total_seconds() / 3600
            if 0 < hrs_to_dl <= 24 and age_h > config.DEADLINE_FRESH_HOURS:
                blockers.append(
                    f"data is {age_h:.1f}h old with {hrs_to_dl:.0f}h to the deadline "
                    f"(max {config.DEADLINE_FRESH_HOURS}h this close) — "
                    "run `fpl refresh` before acting")

    # ---- 1. data freshness / sanity ------------------------------------
    els = boot.get("elements", [])
    if len(els) < 400:
        blockers.append(f"bootstrap has only {len(els)} players — API data looks broken")
    else:
        checks.append(f"bootstrap OK: {len(els)} players, {len(boot['teams'])} teams")
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if nxt:
        dl = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
        if dl < datetime.now(timezone.utc):
            blockers.append("next-GW deadline is in the past — events data stale")
        else:
            checks.append(f"deadline sane: GW{nxt['id']} at {nxt['deadline_time']}")
    if not fixtures:
        warnings.append("fixtures list empty")

    players = data.players_frame(boot)
    by_id = players.set_index("id")

    # ---- 2. squad.yaml internal consistency ----------------------------
    entries = squad.get("players") or []
    if not entries:
        checks.append("no squad in squad.yaml (preseason build mode) — squad checks skipped")
    else:
        ids = [int(p["id"]) for p in entries]
        if len(entries) != config.SQUAD_SIZE:
            blockers.append(f"squad.yaml has {len(entries)} players, needs {config.SQUAD_SIZE}")
        if len(set(ids)) != len(ids):
            blockers.append("duplicate player ids in squad.yaml")
        unknown = [i for i in ids if i not in by_id.index]
        if unknown:
            blockers.append(f"squad.yaml ids not in FPL data (transferred out / wrong id?): {unknown}")
        else:
            mine = by_id.loc[ids]
            counts = mine["element_type"].value_counts().to_dict()
            want = {et: n for et, (_, n, _, _) in config.POSITIONS.items()}
            if {k: counts.get(k, 0) for k in want} != want:
                blockers.append(f"squad shape {counts} != required 2-5-5-3")
            club = mine["team"].value_counts()
            if (club > config.MAX_PER_CLUB).any():
                blockers.append(f">3 players from one club: {club[club > 3].to_dict()}")
            flagged = mine[mine["play_chance"] < 0.75]
            if len(flagged):
                warnings.append("availability flags on owned players: "
                                + ", ".join(f"{r.web_name} ({int(r.play_chance*100)}%)"
                                            for r in flagged.itertuples()))
            for p in entries:
                if "purchase_price" not in p:
                    blockers.append(f"{p.get('name', p['id'])} missing purchase_price "
                                    "(selling-price math impossible)")
                    break
            checks.append("squad.yaml valid: 15 players, legal shape, purchase prices present")

        ft = int(squad.get("free_transfers", 1))
        if not 0 <= ft <= 5:
            blockers.append(f"free_transfers={ft} outside legal 0-5")
        bank = float(squad.get("bank", 0.0))
        if bank < 0:
            blockers.append(f"negative bank {bank}")
        chips = set(squad.get("chips_available", []))
        valid_chips = {"wildcard1", "wildcard2", "bboost", "3xc", "freehit"}
        if chips - valid_chips:
            warnings.append(f"unknown chips in squad.yaml: {chips - valid_chips}")

    # ---- 3. reconcile vs official account data (once entry exists) -----
    entry_id = squad.get("entry_id")
    cur = next((e for e in boot["events"] if e["is_current"]), None)
    if entry_id and cur:
        try:
            picks = data.fetch_picks(int(entry_id), cur["id"])
            official = sorted(p["element"] for p in picks.get("picks", []))
            local = sorted(int(p["id"]) for p in entries)
            if official and official != local:
                extra_official = [by_id.loc[i, "web_name"] for i in official if i not in local and i in by_id.index]
                extra_local = [by_id.loc[i, "web_name"] for i in local if i not in official and i in by_id.index]
                blockers.append(
                    "squad.yaml does NOT match the official FPL picks — "
                    f"on FPL but not in file: {extra_official}; in file but not on FPL: {extra_local}. "
                    "Update squad.yaml before trusting any recommendation.")
            else:
                checks.append(f"squad.yaml matches official GW{cur['id']} picks ({len(official)} players)")
            eh = picks.get("entry_history", {})
            if eh:
                api_bank = eh.get("bank", 0) / 10.0
                if entries and abs(api_bank - float(squad.get("bank", 0))) > 0.05:
                    blockers.append(f"bank mismatch: squad.yaml {squad.get('bank')} vs official {api_bank} — "
                                    "official is source of truth; reconcile squad.yaml first")
                if eh.get("event_transfers_cost", 0) > 0:
                    checks.append(f"official: took a {eh['event_transfers_cost']}-pt hit in GW{cur['id']}")
        except Exception as exc:  # noqa: BLE001 — network/preseason
            warnings.append(f"could not fetch official picks (entry {entry_id}): {exc}")
    elif entry_id:
        checks.append("entry_id set; official-picks reconciliation starts once GW1 is live")
    else:
        checks.append("no entry_id yet — reconciliation vs official picks unavailable (preseason)")

    return {"ok": not blockers, "blockers": blockers, "warnings": warnings, "checks": checks}
