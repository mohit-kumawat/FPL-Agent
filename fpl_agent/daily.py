"""Daily orchestrator: change detection -> trigger matrix -> models -> policy -> report.

Design rule: run the cheapest sufficient analysis. A quiet day far from the
deadline produces a one-line log and touches no models.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from . import (config, data, digest, features, lifecycle, memoryio, models,
               optimizer, policy, rating, report, retention, simulate, verify)


def _hours_to_deadline(boot: dict) -> float | None:
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if not nxt:
        return None
    dl = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
    return (dl - datetime.now(timezone.utc)).total_seconds() / 3600


def pending_calibration(boot: dict, fixtures: list[dict], state: dict,
                        now: datetime | None = None) -> list[int]:
    """Finished gameweeks whose points are final but which were never scored.

    `new_gw_finished` is edge-triggered on the snapshot diff, so it fires on the
    day a gameweek finishes — which is usually BEFORE the 09:00 UK lockdown makes
    the points final. Without this, deferring calibration meant never doing it.
    """
    done = {int(g) for g in (state.get("scored_gws") or [])}
    out = []
    for ev in boot.get("events", []):
        gw = int(ev.get("id", 0))
        if gw in done or not ev.get("finished"):
            continue
        if data.gw_points_final(ev, fixtures, now=now)[0]:
            out.append(gw)
    return sorted(out)


def decide_work(changes: dict, boot: dict, squad: dict, has_signals: bool,
                target_ids: list[int] | None = None,
                pending_gws: list[int] | None = None) -> dict:
    """The trigger matrix. Returns which analyses to run and why.

    `target_ids` are the players the STANDING recommendation says to buy (saved
    by the previous run). News about them matters as much as news about the
    squad: advice to transfer in a player who has since been ruled out must not
    survive untouched just because he isn't owned yet.
    """
    hrs = _hours_to_deadline(boot)
    squad_ids = [int(p["id"]) for p in (squad.get("players") or [])]
    watch = set(squad_ids) | {int(t) for t in (target_ids or [])}

    def touches_squad(items: list[dict]) -> bool:
        # match by element id — the previous string-contains heuristic was
        # truthy for ANY item with a player label, so every news day retrained
        if not watch:
            return bool(items)
        return any(i.get("id") in watch for i in items)

    def owned(items: list[dict]) -> bool:
        return any(i.get("id") in set(squad_ids) for i in items) if squad_ids else False

    triggers: list[str] = []
    work = {"models": False, "optimizer": False, "full_retrain": False,
            "final_check": False, "triggers": triggers, "hours_to_deadline": hrs}

    if changes["first_run"]:
        triggers.append("first run — full baseline")
        work.update(models=True, optimizer=True)
    if changes.get("new_gw_finished"):
        triggers.append("new GW finished — retrain + calibrate")
        work.update(models=True, optimizer=True, full_retrain=True)
    if pending_gws:
        # points went final after the finish-day run deferred them
        triggers.append(f"GW{','.join(str(g) for g in pending_gws)} points now "
                        "final — score the stored predictions")
        work.update(models=True, full_retrain=True)
    if changes["status_changes"] or changes["news_changes"]:
        news = changes["status_changes"] + changes["news_changes"]
        if touches_squad(news):
            who = "squad" if owned(news) else "a recommended transfer target"
            triggers.append(f"injury/status news on {who} — rerun minutes + optimizer")
            work.update(models=True, optimizer=True)
        else:
            triggers.append("news on unrelated players — noted, no rerun")
    if has_signals:
        triggers.append("new signals in inbox — merge + optimize")
        work.update(models=True, optimizer=True)
    if hrs is not None and hrs <= 72:
        triggers.append(f"deadline in {hrs:.0f}h — full decision run")
        work.update(models=True, optimizer=True)
        if hrs <= 24:
            work["final_check"] = True
    if changes["price_changes"] and not work["models"]:
        if touches_squad(changes["price_changes"]):
            # a rise/fall on an owned player moves selling prices and
            # affordability, which can flip a marginal transfer plan
            triggers.append("price change on owned player — selling prices moved, "
                            "rerun optimizer")
            work.update(models=True, optimizer=True)
        else:
            triggers.append("price changes only — squad value updated, no model rerun")
    if not triggers:
        triggers.append("quiet day — no changes, deadline far")
    return work


def recommended_target_ids(rec: dict) -> list[int]:
    """Element ids the standing recommendation says to BUY.

    Persisted to state so the next run treats news about them as relevant even
    though they are not owned yet (see decide_work's `target_ids`). A hold
    recommends nobody, so it clears the list rather than leaving stale targets.
    """
    plan = (rec.get("transfers") or {}).get("plan")
    if plan is not None and len(plan.get("in", [])):
        return [int(x) for x in plan["in"]["id"]]
    if "initial_build" in rec:
        return [int(x) for x in rec["initial_build"]["squad"]["id"]]
    return []


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
        # logged under a distinct kind: a blocked run must NOT mark the day
        # done, or the fix-and-retry the blocker asks for is silently skipped
        memoryio.log_run({"kind": "daily_blocked", "triggers": ["blocked"],
                          "blockers": ver["blockers"]})
        # a blocked run still hands over context — in fact this is when the next
        # run most needs to know what is wrong and that no advice was produced
        digest.write(digest.build(boot, fixtures, squad, state,
                                  freshness=refreshed.get("freshness"),
                                  verification=ver))
        md_path, json_path = report.write_report(ctx_blocked)
        return {**ctx_blocked, "report_md": str(md_path), "report_json": str(json_path)}

    signal_adjust, signal_notes = memoryio.load_signals()
    new_signals = bool(len(signal_adjust)) and not state.get("signals_seen") == sorted(
        n["file"] for n in signal_notes
    )

    work = decide_work(changes, boot, squad, new_signals,
                       target_ids=state.get("target_ids") or [],
                       pending_gws=pending_calibration(boot, fixtures, state))
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
            # record the claims this recommendation rests on, while we still know
            memoryio.log_applied_signals(gw_next, signal_adjust)

        # Calibrate every finished gameweek whose points are FINAL and which has
        # not been scored yet. 2026/27 lockdown is 09:00 UK the day after the
        # last match, so scoring earlier would bank FPL's own bonus/DC revisions
        # as model error — but a deferral has to be retried, hence scored_gws.
        if not panel.empty:
            scored_gws = {int(g) for g in (state.get("scored_gws") or [])}
            for gw_done in sorted(int(r) for r in panel["round"].unique()):
                if gw_done in scored_gws:
                    continue
                ev = next((e for e in boot["events"] if e["id"] == gw_done), None)
                final, why = data.gw_points_final(ev, fixtures)
                if not final:
                    ctx["findings"].append(
                        f"[DATA] GW{gw_done} calibration deferred — {why}")
                    continue
                gw_panel = panel[panel["round"] == gw_done]
                score = memoryio.score_predictions(gw_done, gw_panel)
                if score:
                    memoryio.log_calibration(score)     # trend, not one number
                    ctx["findings"].append(
                        f"[MODEL] GW{gw_done} calibration: MAE {score['mae_all']} "
                        f"(top-50: {score['mae_top50']})"
                    )
                # were OUR minutes claims right? The model's accuracy never
                # measured that, and it is the agent's only feedback loop.
                sig = memoryio.score_signals(gw_done, gw_panel)
                if sig and sig["hits"] + sig["misses"]:
                    ctx["findings"].append(
                        f"[SIGNAL] GW{gw_done} research: {sig['hits']} of "
                        f"{sig['hits'] + sig['misses']} minutes claims held")
                # mark scored either way: with no stored prediction (a gameweek
                # from before this squad existed) there is nothing to retry
                scored_gws.add(gw_done)
            state["scored_gws"] = sorted(scored_gws)

        squad_ids = [p["id"] for p in (squad.get("players") or [])]
        gws_played = int(ep["gws_played"].iloc[0]) if len(ep) else 0
        signal_ids = set(signal_adjust.index.astype(int)) if len(signal_adjust) else set()
        mode = str(squad.get("strategy_mode", "safe")).lower()
        scenarios = memoryio.load_scenarios()

        def _captain(xi_result):
            cand_ids = [int(xi_result["captain"]["id"])] + \
                       [int(x) for x in xi_result["xi"]["id"]]
            try:
                sim = simulate.captain_outlook(ep, cand_ids)
            except Exception:  # noqa: BLE001 — simulation is framing, never a blocker
                sim = None
            return policy.assess_captain(xi_result, ep, mode=mode, sim=sim)

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
                cap = _captain(xi)
                ctx["rating"] = rating.rate_squad(ep, squad_ids)
                ctx["recommendation"] = {
                    "transfers": decision, "xi": xi, "captain": cap,
                    "chips": policy.chip_advice(boot, xi,
                                                squad.get("chips_available", []),
                                                scenarios=scenarios),
                }
                ctx["uncertainty"] = policy.uncertainty_flags(
                    ep, squad_ids, signal_ids, gws_played)
                p = decision.get("plan")
                acted = p is not None and p.get("n_transfers", 0) > 0
                ctx["findings"].extend(policy.price_timing_notes(
                    ep,
                    in_ids=[int(x) for x in p["in"]["id"]] if acted else [],
                    out_ids=[int(x) for x in p["out"]["id"]] if acted else [],
                    owned_ids=squad_ids))
            else:
                build = optimizer.build_squad(ep)
                xi = optimizer.pick_xi(build["squad"])
                cap = _captain(xi)
                ctx["recommendation"] = {
                    "initial_build": build, "xi": xi, "captain": cap,
                    "chips": policy.chip_check(boot, None,
                                               squad.get("chips_available", [])),
                }
                ctx["uncertainty"] = policy.uncertainty_flags(
                    ep, list(build["squad"]["id"]), signal_ids, gws_played)
                ctx["findings"].extend(policy.price_timing_notes(
                    ep, in_ids=[int(x) for x in build["squad"]["id"]],
                    out_ids=[], owned_ids=[]))

    # persist memory. Only overwrite the targets when this run actually produced
    # a recommendation — a quiet day must not erase yesterday's.
    if ctx.get("recommendation"):
        state["target_ids"] = recommended_target_ids(ctx["recommendation"])
    state["signals_seen"] = sorted(n["file"] for n in signal_notes)
    state["last_gw_state"] = changes["gw_state"]
    state["stage"] = {k: stage[k] for k in
                      ("stage", "next_gw", "next_deadline", "hours_to_deadline", "has_team")}
    state["pending_snapshot"] = stage.get("pending") or []
    memoryio.save_state(state)
    if ctx["recommendation"]:
        prev_dec = memoryio.last_decision()
        ctx["prev_decision"] = prev_dec
        memoryio.log_decision(report.decision_summary(ctx))
    memoryio.log_run({"kind": "daily", "triggers": work["triggers"],
                      "models_ran": ctx["models_ran"]})

    # learnings hygiene: warn, never block — an unpruned notebook becomes folklore
    for problem in memoryio.validate_learnings()[:5]:
        ctx["findings"].append(f"[DATA] learnings.md: {problem}")

    # housekeeping before the digest, so the digest reports a pruned world
    ctx["retention"] = retention.run_all(boot)
    if ctx["retention"].get("signals_archived"):
        ctx["findings"].append(
            "[DATA] archived expired signals: "
            + ", ".join(ctx["retention"]["signals_archived"][:6])
            + " (they no longer clutter this report)")

    # the handoff: compact everything this run learned into next run's context
    dg = digest.build(boot, fixtures, squad, state,
                      freshness=refreshed.get("freshness"), verification=ver)
    digest.write(dg)
    ctx["digest"] = dg

    md_path, json_path = report.write_report(ctx)
    ctx["report_md"] = str(md_path)
    ctx["report_json"] = str(json_path)
    return ctx
