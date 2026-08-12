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

# strategy modes: how much EP the mode may trade for rank upside. `safe` trades
# none — it is the default and MUST keep recommending the EP-max pick.
STRATEGY_MODES = ("safe", "balanced", "chase")
CHASE_EP_TOLERANCE = 1.0   # chase may prefer a differential within this EP gap

# chips: play-now must beat the modelled future hold by this factor — chips are
# one-shot options, so ties go to holding
CHIP_PLAY_MARGIN = 1.15
# when the agent has supplied no double/blank scenarios, assume a usable double
# is still coming while the season is young (they cluster in GW29-37)
DEFAULT_DOUBLE_PROB = 0.8
DEFAULT_DOUBLE_LAST_GW = 30      # past this, no-scenario means no assumed double
DOUBLE_CAPTAIN_MULT = 1.7        # a doubled premium returns ~1.7x a single GW
DOUBLE_BENCH_MULT = 1.8          # bench boost scales closer to 2x (4 players)


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


def assess_captain(xi_result: dict, players: pd.DataFrame, mode: str = "safe",
                   sim: pd.DataFrame | None = None) -> dict:
    """Captain call. EP-max is always computed and always shown; `mode` decides
    whether a close differential may take the armband:

      safe      EP-max, always (default — the invariant the tests pin)
      balanced  EP-max pick; a differential within CHASE_EP_TOLERANCE is
                surfaced prominently as a legitimate alternative
      chase     the best differential within CHASE_EP_TOLERANCE becomes the
                pick; the EP-max stays visible as `safe_pick`

    `sim` (from simulate.captain_outlook) adds haul/blank tails to the framing.
    """
    if mode not in STRATEGY_MODES:
        mode = "safe"
    cap = xi_result["captain"]
    xi = xi_result["xi"]
    alts = xi[xi["id"] != cap["id"]].nlargest(3, "ep_next")
    margin = float(cap["ep_next"] - alts.iloc[0]["ep_next"]) if len(alts) else 99.0

    own = pd.to_numeric(players.set_index("id").get("selected_by_percent"),
                        errors="coerce")
    cap_eo = float(own.get(int(cap["id"]), float("nan")))

    # best low-ownership option within the tolerance
    diff_pool = xi[(xi["id"] != cap["id"])
                   & (xi["id"].map(own).fillna(100) < DIFFERENTIAL_EO)
                   & (xi["ep_next"] >= cap["ep_next"] - CHASE_EP_TOLERANCE)]
    differential = None
    if len(diff_pool):
        d = diff_pool.nlargest(1, "ep_next").iloc[0]
        differential = {"pick": d["web_name"],
                        "ep_next": round(float(d["ep_next"]), 2),
                        "ownership": float(own.get(int(d["id"]), float("nan")))}

    out = {
        "pick": cap["web_name"],
        "vice": xi_result["vice"]["web_name"],
        "margin": round(margin, 2),
        "confident": margin >= CAPTAIN_DIFF_MIN,
        "ownership": None if pd.isna(cap_eo) else cap_eo,
        "differential": differential,
        "mode": mode,
        "alternatives": alts[["web_name", "ep_next", "ep_sd"]].round(2).to_dict("records"),
    }

    if mode == "chase" and differential is not None:
        out["safe_pick"] = cap["web_name"]
        out["pick"] = differential["pick"]
        out["reasoning"] = (f"[RECOMMENDATION] chase mode: {differential['pick']} "
                            f"({differential['ownership']:.0f}% owned) within "
                            f"{CHASE_EP_TOLERANCE} EP of {cap['web_name']} — the "
                            "differential armband buys rank variance")

    if sim is not None and len(sim):
        s = sim.set_index("web_name")
        for name_key in ("pick", "safe_pick"):
            name = out.get(name_key)
            if name and name in s.index:
                r = s.loc[name]
                out.setdefault("simulation", {})[name] = {
                    "median": float(r["sim_p50"]), "p10": float(r["sim_p10"]),
                    "p90": float(r["sim_p90"]), "p_haul": float(r["p_haul"]),
                    "p_blank": float(r["p_blank"]),
                }
    return out


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


def _future_double(scenarios: list[dict] | None, gw: int) -> tuple[float, str]:
    """(probability of a usable future double, provenance label).

    Scenario facts come from the agent's research (fixture congestion, cup
    runs) via signals; the pipeline only does the arithmetic. With no
    scenarios, a default prior applies while doubles are still plausibly
    ahead — after DEFAULT_DOUBLE_LAST_GW silence means none.
    """
    future = [s for s in (scenarios or [])
              if s.get("kind") == "double" and int(s.get("gw", 0)) > gw]
    if future:
        best = max(future, key=lambda s: float(s.get("prob", 0)))
        return (float(best.get("prob", 0)),
                f"[SIGNAL] double expected GW{best['gw']} (p={best.get('prob')})")
    if gw <= DEFAULT_DOUBLE_LAST_GW:
        return DEFAULT_DOUBLE_PROB, ("[MODEL] no scenario research yet — assuming a "
                                     f"usable double before season end (p={DEFAULT_DOUBLE_PROB})")
    return 0.0, "[MODEL] season too late for an unresearched double"


def chip_advice(boot: dict, xi_result: dict | None, chips_available: list[str],
                scenarios: list[dict] | None = None) -> list[str]:
    """Chip expected value: play now vs the modelled value of holding.

    Advisory by design — a chip fires once a season, so the bar for "now" is
    now >= hold x CHIP_PLAY_MARGIN, and every number states its assumptions.
    """
    notes: list[str] = []
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if nxt is None:
        return notes
    gw = int(nxt["id"])
    if gw <= 1:
        notes.append("[DATA] No chips playable before GW1; initial squad IS the wildcard.")
        return notes
    if xi_result is None:
        if "wildcard1" in chips_available:
            notes.append("[DATA] Wildcard 1 window is open. [RECOMMENDATION] Hold "
                         "unless squad EP falls >15% below optimal.")
        return notes

    p_double, why = _future_double(scenarios, gw)
    is_double_now = float(xi_result["captain"].get("next_n_fixtures", 1)) > 1

    if "3xc" in chips_available:
        cap_ep = float(xi_result["captain"]["ep_next"])
        now_ev = cap_ep                          # TC adds +1x captain beyond 2x
        hold_ev = cap_ep * DOUBLE_CAPTAIN_MULT * p_double
        if is_double_now and now_ev >= hold_ev * CHIP_PLAY_MARGIN:
            notes.append(f"[RECOMMENDATION] Triple Captain NOW: +{now_ev:.1f} EP on a "
                         f"double, vs {hold_ev:.1f} holding. {why}")
        else:
            notes.append(f"[MODEL] Triple Captain: now +{now_ev:.1f} EP vs "
                         f"~{hold_ev:.1f} holding for a double — "
                         f"{'play' if now_ev >= hold_ev * CHIP_PLAY_MARGIN else 'hold'}. {why}")

    if "bboost" in chips_available and "bench_order" in xi_result:
        bench_ep = float(xi_result["bench_order"]["ep_next"].sum())
        now_ev = bench_ep
        hold_ev = bench_ep * DOUBLE_BENCH_MULT * p_double
        verdict = "play" if now_ev >= hold_ev * CHIP_PLAY_MARGIN else "hold"
        notes.append(f"[MODEL] Bench Boost: now +{now_ev:.1f} EP vs ~{hold_ev:.1f} "
                     f"holding for a double — {verdict}. {why}")

    for chip, label in (("freehit", "Free Hit"), ("wildcard1", "Wildcard"),
                        ("wildcard2", "Wildcard 2")):
        if chip in chips_available:
            notes.append(f"[MODEL] {label}: structural chip — play on a blank/broken "
                         "squad, not on EV; supply blank scenarios via signals for a number.")
            break
    return notes


def chip_check(boot: dict, squad_ep: pd.DataFrame | None, chips_available: list[str]) -> list[str]:
    """Back-compat wrapper: window flags only (no XI context available)."""
    return chip_advice(boot, None, chips_available)