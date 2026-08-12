"""The action gate: may this recommendation be acted on, and if not, why not.

Sits between model output and the report. The model is free to propose anything
the EP arithmetic supports; the gate decides whether the proposal is backed by
evidence the owner could check — official FPL data (tier 0) or an applied
research signal (tier rules in evidence.py). Three terminal states:

  qualified     every requirement holds; the owner can act on this as written
  blocked       a requirement failed and holding is safe — do nothing, the
                report says exactly what research would unblock it
  owner_choice  the pipeline cannot decide: either evidence is incomplete on a
                close call, or acting AND holding both carry unresolved risk.
                This state exists deliberately — without it, "transfer blocked
                for lack of evidence" plus "hold blocked by an injured player"
                would leave the agent with no legal output at the exact moment
                the owner most needs one.

The gate LABELS; it never rewrites the recommendation. The model's numbers stay
visible either way, because hiding a blocked proposal would also hide what is
wrong with it.
"""
from __future__ import annotations

import pandas as pd

# availability: below this chance-of-playing a player is "flagged" and needs
# positive evidence before advice that buys or keeps him can be acted on.
# 0.75 matches verify.py's warning bar for owned players.
FLAG_CHANCE = 0.75
CAPTAIN_BAND_EP = 1.0        # candidates this close to the pick share its burden


def _availability(pid: int, players: pd.DataFrame,
                  signals: pd.DataFrame | None) -> tuple[bool, str | None]:
    """Does acting on this player rest on checkable availability evidence?

    Tier 0 is the FPL API itself: status 'a' with no degraded chance-of-playing
    IS evidence. A flagged player needs an applied signal floor — research that
    says he plays — before advice involving him can qualify.
    """
    if pid not in players.index:
        return False, "not in FPL data"
    row = players.loc[pid]
    status = str(row.get("status", "a"))
    chance = float(row.get("play_chance", 1.0))
    if status == "a" and chance >= FLAG_CHANCE:
        return True, None
    has_floor = (signals is not None and len(signals)
                 and pid in signals.index
                 and "xmins_min" in signals.columns
                 and pd.notna(signals.loc[pid, "xmins_min"]))
    if has_floor:
        return True, None
    return False, (f"{row.get('web_name', pid)} flagged (status '{status}', "
                   f"{chance:.0%} chance) with no evidenced minutes claim")


def _conflicts(signal_notes: list[dict], ids: set[int]) -> list[str]:
    """Unresolved cross-file bound conflicts touching these players."""
    out = []
    for n in signal_notes or []:
        pid = n.get("conflict_player")
        if pid is not None and int(pid) in ids and not n.get("applied"):
            out.extend(n.get("problems", []))
    return out


def _research_task(reason: str) -> str:
    return (f"resolve: {reason} — find a tier-1 (club/manager) or corroborated "
            "tier-2 source and write it to signals/ with an evidence block")


def evaluate(ctx: dict, ep: pd.DataFrame, squad: dict,
             signals: pd.DataFrame | None,
             signal_notes: list[dict] | None = None) -> dict:
    """Build the gate object for this run's recommendation.

    {status, action_type, failed_requirements, next_research,
     captain: {...}, chips: {...}}
    """
    rec = ctx.get("recommendation") or {}
    players = ep.set_index("id") if len(ep) else pd.DataFrame()
    squad_ids = [int(p["id"]) for p in (squad.get("players") or [])]

    failed: list[str] = []
    research: list[str] = []

    # ---- the primary action ------------------------------------------------
    plan = (rec.get("transfers") or {}).get("plan")
    action = (rec.get("transfers") or {}).get("action", "hold")
    acting = plan is not None and plan.get("n_transfers", 0) > 0
    if "initial_build" in rec:
        action_type = "initial_squad"
        action = "initial_squad_proposal"
        in_ids = [int(x) for x in rec["initial_build"]["squad"]["id"]]
    elif acting:
        action_type = "transfer"
        in_ids = [int(x) for x in plan["in"]["id"]]
    else:
        action_type = "hold"
        in_ids = []

    # incoming players: every one needs availability evidence
    for pid in in_ids:
        ok, why = _availability(pid, players, signals)
        if not ok:
            failed.append(f"incoming availability unverified: {why}")
            research.append(_research_task(why))

    # unresolved claim conflicts on anyone the action touches
    touched = set(in_ids) | set(squad_ids)
    for c in _conflicts(signal_notes or ctx.get("signal_notes"), touched):
        failed.append(f"unresolved claim conflict: {c}")
        research.append(_research_task(c))

    # holding is only automatically safe when nobody owned carries silent risk
    hold_risks: list[str] = []
    for pid in squad_ids:
        ok, why = _availability(pid, players, signals)
        if not ok:
            hold_risks.append(why)

    if action_type in ("transfer", "initial_squad"):
        status = "qualified" if not failed else "blocked"
        if failed and hold_risks:
            # the deadlock: acting lacks evidence AND holding carries risk.
            # Neither side can be auto-chosen — enumerate both, hand it over.
            status = "owner_choice"
            failed.append("holding is not risk-free either: "
                          + "; ".join(hold_risks[:3]))
            research.extend(_research_task(r) for r in hold_risks[:2])
    else:  # hold
        if hold_risks:
            status = "owner_choice"
            failed.extend(f"owned-player risk unresolved: {r}" for r in hold_risks[:3])
            research.extend(_research_task(r) for r in hold_risks[:2])
        else:
            status = "qualified"

    # ---- captain -----------------------------------------------------------
    captain = {"status": "qualified", "unevidenced": []}
    xi_res = rec.get("xi")
    if xi_res is not None and "xi" in xi_res:
        xi = xi_res["xi"]
        cap_row = xi_res.get("captain")
        cap_ep = float(cap_row["ep_next"]) if cap_row is not None else 0.0
        band = xi[pd.to_numeric(xi["ep_next"], errors="coerce")
                  >= cap_ep - CAPTAIN_BAND_EP]
        for pid in band["id"].astype(int):
            ok, why = _availability(pid, players, signals)
            if not ok:
                captain["unevidenced"].append(why)
        if captain["unevidenced"]:
            captain["status"] = "owner_choice"
            research.extend(_research_task(r) for r in captain["unevidenced"][:2])

    # ---- chips ---------------------------------------------------------------
    # policy.chip_advice already applies the evidence rule (a default prior
    # never fires a chip — dominance test); the gate just reads the verdict out
    # of the notes so the machine-readable object carries it too.
    chip_notes = rec.get("chips") or []
    chips = {"status": "qualified"}
    if any("OWNER CHOICE" in n for n in chip_notes):
        chips["status"] = "owner_choice"
        research.append("chip EV is scenario-dependent: research future "
                        "double/blank gameweeks and record them as `scenarios:` "
                        "in signals/")

    overall = status
    if overall == "qualified" and (captain["status"] != "qualified"
                                   or chips["status"] != "qualified"):
        overall = "owner_choice"

    # dedupe research tasks, keep order
    seen: set[str] = set()
    research = [r for r in research if not (r in seen or seen.add(r))]

    return {"status": overall, "action_type": action_type, "action": action,
            "failed_requirements": failed, "next_research": research,
            "captain": captain, "chips": chips}


def headline(gate: dict) -> str:
    """One-line verdict for the report."""
    st, at = gate["status"], gate["action_type"]
    if st == "qualified":
        return f"QUALIFIED — the {at.replace('_', ' ')} below is evidence-backed and actionable"
    if st == "blocked" and at == "transfer":
        return "BLOCKED — NO ACTIONABLE TRANSFER (holding is safe; requirements below)"
    if st == "blocked":
        return f"BLOCKED — {at.replace('_', ' ')} does not qualify (requirements below)"
    return ("OWNER CHOICE — the pipeline cannot decide this on evidence; "
            "both paths and their risks are listed below")
