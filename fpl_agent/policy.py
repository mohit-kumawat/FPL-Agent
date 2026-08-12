"""Decision policy: turns optimizer output into recommendations with thresholds.

The optimizer maximizes EP; this layer decides whether acting is *worth it*
and labels every claim by source: [DATA] / [MODEL] / [SIGNAL] / [RECOMMENDATION].
"""
from __future__ import annotations

import pandas as pd

from . import config

# base thresholds over the HORIZON_GWS window (dynamic scaling below)
FT_GAIN_MIN = 2.0        # use a free transfer only if it adds >= this EP
HIT_GAIN_MIN = 6.0       # take a -4 only if it adds >= this EP net of the hit
CAPTAIN_DIFF_MIN = 0.5   # flag a captaincy call only if clearly better
HIT_XMINS_MIN = 0.6      # never recommend a hit on a minutes-capped incoming
DIFFERENTIAL_EO = 15.0   # ownership % below which a captain counts as differential


def thresholds(gws_played: int, free_transfers: int) -> tuple[float, float]:
    """Dynamic FT/hit bars.

    Early season (< 4 GWs of data) predictions are noisy -> demand 1.5x more.
    Near the FT bank cap (>= 4 of 5) banking further wastes value -> ease 25%.
    """
    ft, hit = FT_GAIN_MIN, HIT_GAIN_MIN
    if gws_played < 4:
        ft, hit = ft * 1.5, hit * 1.5
    if free_transfers >= 4:
        ft *= 0.75
    return ft, hit


def assess_transfers(plan: dict, free_transfers: int, gws_played: int = 38) -> dict:
    """Apply thresholds to the transfer plans; pick action + reasoning."""
    ft_min, hit_min = thresholds(gws_played, free_transfers)
    plans = plan["plans"]
    hold = next((p for p in plans if p["n_transfers"] == 0), None)
    decision = {"action": "hold", "plan": hold, "reasoning": [],
                "best_alternative": None}

    best_alt = max((p for p in plans if p["n_transfers"] > 0),
                   key=lambda q: q["net_gain_vs_hold"], default=None)
    if best_alt is not None:
        decision["best_alternative"] = {
            "n_transfers": best_alt["n_transfers"],
            "net_gain_vs_hold": best_alt["net_gain_vs_hold"],
        }

    # collect every plan that clears its own bar, then take the best net gain —
    # iterating and overwriting would recommend the largest passing k instead
    passing: list[tuple[dict, float, bool, pd.DataFrame]] = []
    for p in plans:
        k = p["n_transfers"]
        if k == 0:
            continue
        gain = p["net_gain_vs_hold"]
        is_hit = k > free_transfers
        threshold = hit_min if is_hit else ft_min * k

        # hard rule: a hit must not buy a minutes-risk player, unless the gain
        # clears the bar by a wide margin even so
        capped = p["in"][p["in"]["xmins"] < HIT_XMINS_MIN] if len(p["in"]) else p["in"]
        if is_hit and len(capped) and gain < threshold * 1.25:
            continue

        if gain >= threshold:
            passing.append((p, threshold, is_hit, capped))

    if passing:
        # tie-break toward fewer transfers: keeps a banked FT when gains match
        p, threshold, is_hit, capped = max(
            passing, key=lambda t: (t[0]["net_gain_vs_hold"], -t[0]["n_transfers"]))
        k = p["n_transfers"]
        gain = p["net_gain_vs_hold"]
        decision = {
            **decision,
            "action": f"{k}_transfer{'s' if k > 1 else ''}",
            "plan": p,
            "reasoning": [
                f"[MODEL] {k} transfer(s) adds {gain:+.1f} EP over "
                f"{config.HORIZON_GWS} GWs net of hits (dynamic threshold {threshold:.1f}; "
                f"gws_played={gws_played}, FTs={free_transfers})."
            ],
        }
        if is_hit and len(capped):
            decision["reasoning"].append(
                f"[MODEL] hit includes minutes-capped {list(capped['web_name'])} "
                "but gain clears the bar by >25% — verify fitness before acting.")
    if decision["action"] == "hold":
        why = (f"best alternative: {best_alt['n_transfers']} moves "
               f"{best_alt['net_gain_vs_hold']:+.1f} EP vs bar {ft_min:.1f}"
               if best_alt else "no legal alternative found")
        decision["reasoning"].append(
            f"[MODEL] No transfer clears the dynamic threshold ({why}). "
            "[RECOMMENDATION] Bank the free transfer.")
    return decision


def assess_captain(xi_result: dict, players: pd.DataFrame) -> dict:
    """EP-max captain (primary) + ownership context + differential option."""
    cap = xi_result["captain"]
    xi = xi_result["xi"]
    alts = xi[xi["id"] != cap["id"]].nlargest(3, "ep_next")
    margin = float(cap["ep_next"] - alts.iloc[0]["ep_next"]) if len(alts) else 99.0

    own = pd.to_numeric(players.set_index("id").get("selected_by_percent"),
                        errors="coerce")
    cap_eo = float(own.get(int(cap["id"]), float("nan")))

    # differential mode (advisory): best low-ownership option within 1 EP
    diff_pool = xi[(xi["id"] != cap["id"])
                   & (xi["id"].map(own).fillna(100) < DIFFERENTIAL_EO)
                   & (xi["ep_next"] >= cap["ep_next"] - 1.0)]
    differential = None
    if len(diff_pool):
        d = diff_pool.nlargest(1, "ep_next").iloc[0]
        differential = {"pick": d["web_name"],
                        "ep_next": round(float(d["ep_next"]), 2),
                        "ownership": float(own.get(int(d["id"]), float("nan")))}

    return {
        "pick": cap["web_name"],
        "vice": xi_result["vice"]["web_name"],
        "margin": round(margin, 2),
        "confident": margin >= CAPTAIN_DIFF_MIN,
        "ownership": None if pd.isna(cap_eo) else cap_eo,
        "differential": differential,
        "alternatives": alts[["web_name", "ep_next", "ep_sd"]].round(2).to_dict("records"),
    }


def uncertainty_flags(ep: pd.DataFrame, chosen_ids: list[int],
                      signal_ids: set[int], gws_played: int) -> list[str]:
    """Cold-start discipline: players without data need a signal, and a
    scoring-rules change makes every prior-season number softer."""
    flags: list[str] = []
    if gws_played >= 3:
        return flags
    mine = ep[ep["id"].isin(chosen_ids)]

    if "prior_rules_cross" in ep.columns and bool(ep["prior_rules_cross"].any()):
        flags.append("[MODEL] scoring rules changed since last season — prior-season "
                     "PPG is downweighted (process stats lead); treat close EP "
                     "comparisons as ties until real GWs accumulate")

    risky = mine[(mine["price"] >= 7.0) & (mine["minutes"] < 900)]
    for r in risky.itertuples():
        if int(r.id) not in signal_ids:
            flags.append(f"[MODEL] {r.web_name} (£{r.price}m) has <1 season of data and "
                         "NO signal — research before trusting this pick")

    # new signings / promoted-team players at any price: no meaningful league
    # sample, so their EP is a price prior wearing a number
    fresh = mine[(mine["price"] < 7.0) & (mine["minutes"] < 450)]
    for r in fresh.itertuples():
        if int(r.id) not in signal_ids:
            flags.append(f"[MODEL] {r.web_name} (£{r.price}m) has almost no league "
                         "history (new signing / promoted) — EP is a price prior; "
                         "a minutes signal would firm this up")

    thin = mine[mine["xmins"].between(0.45, 0.7)]
    for r in thin.itertuples():
        flags.append(f"[MODEL] {r.web_name} xmins {r.xmins:.2f} — rotation-zone minutes")
    return flags


def price_timing_notes(players: pd.DataFrame, in_ids: list[int],
                       out_ids: list[int], owned_ids: list[int]) -> list[str]:
    """Move-tonight-vs-wait advice from FPL's price-change predictor.

    Timing only, never selection: these notes attach urgency (or patience) to
    transfers the model already recommends, and warn when owned value is about
    to fall. Price changes land at 00:00 UK time (config.PRICE_CHANGE_TZ).
    """
    notes: list[str] = []
    if "price_change_percent" not in players.columns:
        return notes
    p = players.set_index("id")
    pct = pd.to_numeric(p["price_change_percent"], errors="coerce").fillna(0.0)

    def row(pid: int) -> tuple[str, float] | None:
        if pid not in p.index:
            return None
        return f"{p.loc[pid, 'web_name']} (£{p.loc[pid, 'price']}m)", float(pct.get(pid, 0.0))

    for pid in in_ids:
        r = row(pid)
        if r is None:
            continue
        name, v = r
        if v >= config.PRICE_MOVE_IMMINENT:
            notes.append(f"[DATA] {name} is {v:.0f}% toward tonight's RISE — "
                         "buying before 00:00 UK saves £0.1m")
        elif v >= config.PRICE_MOVE_WATCH:
            notes.append(f"[DATA] {name} has rise momentum ({v:.0f}%) — "
                         "no urgency tonight, but do not sit on this for days")
        elif v <= -config.PRICE_MOVE_IMMINENT:
            notes.append(f"[DATA] {name} is {-v:.0f}% toward tonight's FALL — "
                         "waiting past 00:00 UK buys £0.1m cheaper")

    for pid in out_ids:
        r = row(pid)
        if r is None:
            continue
        name, v = r
        if v <= -config.PRICE_MOVE_IMMINENT:
            notes.append(f"[DATA] {name} is {-v:.0f}% toward tonight's FALL — "
                         "selling before 00:00 UK protects the selling price")

    at_risk = [pid for pid in owned_ids if pid not in set(out_ids)]
    for pid in at_risk:
        r = row(pid)
        if r is None:
            continue
        name, v = r
        if v <= -config.PRICE_MOVE_IMMINENT:
            notes.append(f"[DATA] owned value at risk: {name} likely falls tonight "
                         f"({-v:.0f}%) — no action forced, but selling later costs £0.1m")
    return notes


def chip_check(boot: dict, squad_ep: pd.DataFrame | None, chips_available: list[str]) -> list[str]:
    """Conservative chip advice: only flag structurally good windows."""
    notes = []
    events = boot["events"]
    nxt = next((e for e in events if e["is_next"]), None)
    if nxt is None:
        return notes
    gw = nxt["id"]
    if "wildcard1" in chips_available and gw >= 2:
        notes.append("[DATA] Wildcard 1 window is open (GW2-19). "
                      "[RECOMMENDATION] Hold unless squad EP falls >15% below optimal.")
    if gw <= 1:
        notes.append("[DATA] No chips playable before GW1; initial squad IS the wildcard.")
    return notes