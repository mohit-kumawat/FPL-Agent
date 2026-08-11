"""Daily orchestrator: change detection -> trigger matrix -> models -> policy -> report.

Design rule: run the cheapest sufficient analysis. A quiet day far from the
deadline produces a one-line log and touches no models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import (config, data, features, lifecycle, memoryio, models, optimizer,
               policy, rating, report, verify)


def _hours_to_deadline(boot: dict) -> float | None:
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if not nxt:
        return None
    dl = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
    return (dl - datetime.now(timezone.utc)).total_seconds() / 3600


def decide_work(changes: dict, boot: dict, squad: dict, has_signals: bool) -> dict:
    """The trigger matrix. Returns which analyses to run and why."""
    hrs = _hours_to_deadline(boot)
    squad_ids = [p["id"] for p in (squad.get("players") or [])]

    def touches_squad(items: list[dict]) -> bool:
        return any(
            any(str(pid) in str(i) or i.get("player", "") for pid in squad_ids)
            for i in items
        ) if squad_ids else bool(items)

    triggers: list[str] = []
    work = {"models": False, "optimizer": False, "full_retrain": False,
            "final_check": False, "triggers": triggers, "hours_to_deadline": hrs}

    if changes["first_run"]:
        triggers.append("first run — full baseline")
        work.update(models=True, optimizer=True)
    if changes.get("new_gw_finished"):
        triggers.append("new GW finished — retrain + calibrate")
        work.update(models=True, optimizer=True, full_retrain=True)
    if changes["status_changes"] or changes["news_changes"]:
        relevant = touches_squad(changes["status_changes"] + changes["news_changes"])
        if relevant:
            triggers.append("injury/status news — rerun minutes + optimizer")
            work.update(models=True, optimizer=True)
        else:
            triggers.append("news on non-squad players — noted, no rerun")
    if has_signals:
        triggers.append("new signals in inbox — merge + optimize")
        work.update(models=True, optimizer=True)
    if hrs is not None and hrs <= 72:
        triggers.append(f"deadline in {hrs:.0f}h — full decision run")
        work.update(models=True, optimizer=True)
        if hrs <= 24:
            work["final_check"] = True
    if changes["price_changes"] and not work["models"]:
        triggers.append("price changes only — squad value updated, no model rerun")
    if not triggers:
        triggers.append("quiet day — no changes, deadline far")
    return work


def run_daily(force: bool = False) -> dict:
    """Entry point for the `fpl daily` command."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if memoryio.already_ran_today("daily") and not force:
        return {"skipped": "already ran today (use --force to rerun)"}

    refreshed = data.refresh(force=force)
    boot, fixtures = refreshed["bootstrap"], refreshed["fixtures"]
    prev = data.latest_snapshot_before("bootstrap", today)
    changes = data.detect_changes(boot, prev[1] if prev else None)

    squad = memoryio.load_squad()
    state = memoryio.load_state()

    # step 0: verify state BEFORE any analysis — bad state means no recommendation
    ver = verify.verify_state(boot, squad, fixtures, refreshed.get("freshness"))
    if not ver["ok"]:
        ctx_blocked = {
            "date": today, "changes": changes, "work": {"triggers": ["BLOCKED by verification"],
                                                         "hours_to_deadline": None},
            "stage": lifecycle.status(boot), "verification": ver,
            "signal_notes": [], "models_ran": [],
            "findings": [f"[DATA] BLOCKER: {b}" for b in ver["blockers"]],
            "recommendation": None, "rating": None,
        }
        memoryio.log_run({"kind": "daily", "triggers": ["blocked"],
                          "blockers": ver["blockers"]})
        md_path, json_path = report.write_report(ctx_blocked)
        return {**ctx_blocked, "report_md": str(md_path), "report_json": str(json_path)}

    signal_adjust, signal_notes = memoryio.load_signals()
    new_signals = bool(len(signal_adjust)) and not state.get("signals_seen") == sorted(
        n["file"] for n in signal_notes
    )

    work = decide_work(changes, boot, squad, new_signals)
    stage = lifecycle.status(boot)

    ctx: dict[str, Any] = {
        "date": today, "changes": changes, "work": work, "stage": stage,
        "verification": ver, "freshness": refreshed.get("freshness"),
        "signal_notes": signal_notes, "models_ran": [],
        "findings": [f"[DATA] verify warning: {w}" for w in ver["warnings"]],
        "recommendation": None, "rating": None,
    }

    if work["models"]:
        players = data.players_frame(boot)
        panel = data.build_current_panel(boot) if work["full_retrain"] else (
            data.build_current_panel(boot) if changes["gw_state"]["gws_finished"] else pd.DataFrame()
        )
        enriched = features.enrich_players(players, boot, fixtures,
                                           panel if not panel.empty else None)
        ep = models.expected_points(
            enriched, panel if not panel.empty else None,
            signal_adjust if len(signal_adjust) else None,
        )
        ctx["models_ran"] = ["minutes", "prior_baseline"] + (
            ["recency_A", "ridge_B", "ensemble"] if not panel.empty else []
        ) + (["signal_merge"] if len(signal_adjust) else [])

        gw_next = changes["gw_state"]["next_gw"]
        if gw_next:
            memoryio.save_predictions(gw_next, ep)

        # calibration when a GW just finished
        if work["full_retrain"] and not panel.empty:
            last_gw = int(panel["round"].max())
            score = memoryio.score_predictions(
                last_gw, panel[panel["round"] == last_gw]
            )
            if score:
                ctx["findings"].append(
                    f"[MODEL] GW{last_gw} calibration: MAE {score['mae_all']} "
                    f"(top-50: {score['mae_top50']})"
                )

        squad_ids = [p["id"] for p in (squad.get("players") or [])]
        gws_played = int(ep["gws_played"].iloc[0]) if len(ep) else 0
        signal_ids = set(signal_adjust.index.astype(int)) if len(signal_adjust) else set()
        if work["optimizer"]:
            if len(squad_ids) == config.SQUAD_SIZE:
                sell = memoryio.squad_selling_prices(squad, ep)
                plan = optimizer.plan_transfers(
                    ep, squad_ids, sell,
                    bank=float(squad.get("bank", 0.0)),
                    free_transfers=int(squad.get("free_transfers", 1)),
                )
                decision = policy.assess_transfers(plan, int(squad.get("free_transfers", 1)),
                                                   gws_played=gws_played)
                my_squad = ep[ep["id"].isin(squad_ids)]
                xi = optimizer.pick_xi(my_squad)
                cap = policy.assess_captain(xi, ep)
                ctx["rating"] = rating.rate_squad(ep, squad_ids)
                ctx["recommendation"] = {
                    "transfers": decision, "xi": xi, "captain": cap,
                    "chips": policy.chip_check(boot, my_squad,
                                               squad.get("chips_available", [])),
                }
                ctx["uncertainty"] = policy.uncertainty_flags(
                    ep, squad_ids, signal_ids, gws_played)
            else:
                build = optimizer.build_squad(ep)
                xi = optimizer.pick_xi(build["squad"])
                cap = policy.assess_captain(xi, ep)
                ctx["recommendation"] = {
                    "initial_build": build, "xi": xi, "captain": cap,
                    "chips": policy.chip_check(boot, None,
                                               squad.get("chips_available", [])),
                }
                ctx["uncertainty"] = policy.uncertainty_flags(
                    ep, list(build["squad"]["id"]), signal_ids, gws_played)

    # persist memory
    state["signals_seen"] = sorted(n["file"] for n in signal_notes)
    state["last_gw_state"] = changes["gw_state"]
    state["stage"] = {k: stage[k] for k in
                      ("stage", "next_gw", "next_deadline", "hours_to_deadline", "has_team")}
    memoryio.save_state(state)
    if ctx["recommendation"]:
        prev_dec = memoryio.last_decision()
        ctx["prev_decision"] = prev_dec
        memoryio.log_decision(report.decision_summary(ctx))
    memoryio.log_run({"kind": "daily", "triggers": work["triggers"],
                      "models_ran": ctx["models_ran"]})

    md_path, json_path = report.write_report(ctx)
    ctx["report_md"] = str(md_path)
    ctx["report_json"] = str(json_path)
    return ctx
