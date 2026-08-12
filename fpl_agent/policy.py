"""Decision policy: turns optimizer output into recommendations with thresholds.

The optimizer maximizes EP; this layer decides whether acting is *worth it*
and labels every claim by source: [DATA] / [MODEL] / [SIGNAL] / [RECOMMENDATION].
"""
from __future__ import annotations

import pandas as pd

from . import config, rules

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
    cap_id = int(cap["id"])
    alts = xi[xi["id"] != cap_id].nlargest(3, "ep_next")
    margin = float(cap["ep_next"] - alts.iloc[0]["ep_next"]) if len(alts) else 99.0

    own = pd.to_numeric(players.set_index("id").get("selected_by_percent"),
                        errors="coerce")
    cap_eo = float(own.get(cap_id, float("nan")))

    # best low-ownership option within the tolerance
    diff_pool = xi[(xi["id"] != cap_id)
                   & (xi["id"].map(own).fillna(100) < DIFFERENTIAL_EO)
                   & (xi["ep_next"] >= cap["ep_next"] - CHASE_EP_TOLERANCE)]
    differential = None
    if len(diff_pool):
        d = diff_pool.nlargest(1, "ep_next").iloc[0]
        differential = {"id": int(d["id"]), "pick": d["web_name"],
                        "ep_next": round(float(d["ep_next"]), 2),
                        "ownership": float(own.get(int(d["id"]), float("nan")))}

    pick_id, vice_id = cap_id, int(xi_result["vice"]["id"])
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
        pick_id = differential["id"]
        out["reasoning"] = (f"[RECOMMENDATION] chase mode: {differential['pick']} "
                            f"({differential['ownership']:.0f}% owned) within "
                            f"{CHASE_EP_TOLERANCE} EP of {cap['web_name']} — the "
                            "differential armband buys rank variance")
        # the armband moved, so the vice must move too: captain and vice can
        # never be the same player. Highest-EP XI player who isn't the new
        # captain — which is normally the EP-max pick we just stepped off.
        if vice_id == pick_id:
            rest = xi[xi["id"] != pick_id].nlargest(1, "ep_next")
            if len(rest):
                vice_id = int(rest.iloc[0]["id"])
                out["vice"] = rest.iloc[0]["web_name"]

    if sim is not None and len(sim) and "id" in sim.columns:
        # key by element id, never web_name: web_names are not unique in FPL
        # (two Wards, two Reids), and a duplicate label makes .loc return a
        # frame, which used to raise TypeError and abort the whole daily run
        s = sim.drop_duplicates("id").set_index("id")
        for key, pid in (("pick", pick_id), ("safe_pick", cap_id)):
            name = out.get(key)
            if name and pid in s.index:
                r = s.loc[pid]
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

    # History checks read bootstrap `minutes`, which holds PRIOR-season totals
    # only until GW1 is processed — afterwards it resets to the current season,
    # where every player looks brand new (90 minutes, not 3,000). Running these
    # past gws_played 0 flagged the entire squad as new signings, burying the
    # genuine warnings. Post-GW1 the minutes-based tests are simply not
    # answerable from this frame, so they don't run.
    if gws_played == 0:
        risky = mine[(mine["price"] >= 7.0) & (mine["minutes"] < 900)]
        for r in risky.itertuples():
            if int(r.id) not in signal_ids:
                flags.append(f"[MODEL] {r.web_name} (£{r.price}m) has <1 season of data "
                             "and NO signal — research before trusting this pick")

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


def _future_double(scenarios: list[dict] | None, gw: int) -> tuple[float, str, str]:
    """(probability of a usable future double, provenance label, provenance kind).

    Scenario facts come from the agent's research (fixture congestion, cup
    runs) via signals; the pipeline only does the arithmetic. With no
    scenarios, a default prior applies while doubles are still plausibly
    ahead — after DEFAULT_DOUBLE_LAST_GW silence means none.

    The kind ("signal" / "default" / "late") matters downstream: an ACTIONABLE
    play-now must never rest on the default prior alone (see chip_advice's
    dominance test) — a default probability is a model assumption, not evidence.
    """
    future = [s for s in (scenarios or [])
              if s.get("kind") == "double" and int(s.get("gw", 0)) > gw]
    if future:
        best = max(future, key=lambda s: float(s.get("prob", 0)))
        return (float(best.get("prob", 0)),
                f"[SIGNAL] double expected GW{best['gw']} (p={best.get('prob')})",
                "signal")
    if gw <= DEFAULT_DOUBLE_LAST_GW:
        return DEFAULT_DOUBLE_PROB, ("[MODEL] no scenario research yet — assuming a "
                                     f"usable double before season end (p={DEFAULT_DOUBLE_PROB})"), "default"
    return 0.0, "[MODEL] season too late for an unresearched double", "late"


def chip_advice(boot: dict, xi_result: dict | None, chips_available: list[str],
                scenarios: list[dict] | None = None) -> list[str]:
    """Chip expected value: play now vs the modelled value of holding.

    Chip availability comes from the API (`rules.chip_windows`), not from
    assumptions: 2026/27 ships TWO full sets — Wildcard, Free Hit, Bench Boost
    and Triple Captain in each half — with Bench Boost and Triple Captain
    playable from GW1 while Wildcard and Free Hit open at GW2. Only ONE chip may
    be played per gameweek, so when several clear their bar the best is
    recommended and the rest are named as blocked.

    Advisory by design: a chip copy fires once per half, so the bar for "now" is
    now >= hold x CHIP_PLAY_MARGIN, and every number states its assumptions.
    """
    notes: list[str] = []
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    if nxt is None:
        return notes
    gw = int(nxt["id"])
    windows = rules.chip_windows(boot)
    playable = rules.playable_now(chips_available, gw, windows)
    # never present assumed windows as if they came from the API
    assumed = ("" if rules.windows_are_live(boot)
               else " [DATA] chip windows are ASSUMED (bootstrap carried no "
                    "`chips` block) — verify the window before playing.")

    def label(family: str) -> str:
        half = rules.chip_half(family, gw, windows)
        base = rules.CHIP_LABELS[family]
        return f"{base} {half}" if half else base

    if not playable:
        held = rules.canonical_chips(chips_available)
        if held:
            notes.append(f"[DATA] No chip is playable in GW{gw} "
                         f"(held: {', '.join(sorted(held))}).")
        elif gw <= 1:
            notes.append("[DATA] No chips left; the initial squad IS the wildcard.")
        return notes

    if xi_result is None:
        # no XI context (preseason build): windows only, no EV
        for family in playable:
            notes.append(f"[DATA] {label(family)} is available in GW{gw}. "
                         "[RECOMMENDATION] Hold — chip EV needs a settled squad.")
        return notes

    p_double, why, provenance = _future_double(scenarios, gw)

    def _n_fx(row) -> float:
        """Fixture count for a row, defaulting to 1 on missing/NaN."""
        v = pd.to_numeric(pd.Series([row.get("next_n_fixtures", 1)]),
                          errors="coerce").fillna(1.0).iloc[0]
        return max(1.0, float(v))

    cap_row = xi_result["captain"]
    cap_n_fx = _n_fx(cap_row)
    is_double_now = cap_n_fx > 1
    # (family, now_ev, hold_ev) for chips whose value is this week's points
    scored: list[tuple[str, float, float]] = []

    if "3xc" in playable:
        # ep_next already SUMS this gameweek's fixtures, so on a double it is
        # roughly twice a single-gameweek figure. Scaling it again by
        # DOUBLE_CAPTAIN_MULT credited the double twice and made "play now"
        # unreachable above p_double ~= 0.5; both sides must be in the same
        # single-fixture units.
        cap_ep = float(cap_row["ep_next"])              # this GW, all fixtures
        cap_per_fx = cap_ep / cap_n_fx                  # single-fixture equivalent
        scored.append(("3xc", cap_ep,                   # TC adds +1x beyond 2x
                       cap_per_fx * DOUBLE_CAPTAIN_MULT * p_double))

    if "bboost" in playable and "bench_order" in xi_result:
        bench = xi_result["bench_order"]
        ep = pd.to_numeric(bench["ep_next"], errors="coerce").fillna(0.0)
        # DataFrame.get with a default returns the SCALAR default, not a Series,
        # so build the fallback explicitly rather than chaining off it
        if "next_n_fixtures" in bench.columns:
            n = pd.to_numeric(bench["next_n_fixtures"],
                              errors="coerce").fillna(1.0).clip(lower=1)
        else:
            n = pd.Series(1.0, index=bench.index)
        scored.append(("bboost", float(ep.sum()),
                       float((ep / n).sum()) * DOUBLE_BENCH_MULT * p_double))

    # A chip is worth playing only if it beats holding AND actually banks
    # something — with zero EP on both sides (a blank, or missing fixture data)
    # `now >= hold` is trivially true and used to read as "play".
    def clears(now_ev: float, hold_ev: float) -> bool:
        return now_ev > 0.05 and now_ev >= hold_ev * CHIP_PLAY_MARGIN

    def prior_dependent(now_ev: float, hold_ev: float) -> bool:
        """True when 'play now' holds under the DEFAULT prior but flips if a
        future double is certain. Evidence rule: a default probability may never
        produce an ACTIONABLE recommendation — the dominance test is what makes
        the default-prior path safe. Play-now survives it only when it wins even
        against a guaranteed future double (then no scenario research could
        change the answer)."""
        if provenance != "default" or p_double <= 0:
            return False
        return clears(now_ev, hold_ev) and not clears(now_ev, hold_ev / p_double)

    winner = max((s for s in scored
                  if clears(s[1], s[2]) and not prior_dependent(s[1], s[2])),
                 key=lambda s: s[1] - s[2], default=None)
    for family, now_ev, hold_ev in scored:
        if prior_dependent(now_ev, hold_ev):
            notes.append(
                f"[MODEL] {label(family)}: now +{now_ev:.1f} EP clears the bar only "
                f"under the ASSUMED double prior (p={p_double}) and holds if a double "
                "is certain — OWNER CHOICE pending scenario research: confirm or rule "
                "out a usable future double (`scenarios:` in signals/). "
                "A default probability never fires a chip.")
            continue
        verdict = "play" if clears(now_ev, hold_ev) else "hold"
        if winner is not None and family == winner[0]:
            prefix = f"[RECOMMENDATION] {label(family)} NOW"
        else:
            prefix = f"[MODEL] {label(family)}"
        blocked = ("" if winner is None or family == winner[0] or verdict == "hold"
                   else f" — blocked this GW: only one chip per gameweek, and "
                        f"{rules.CHIP_LABELS[winner[0]]} is worth more")
        notes.append(f"{prefix}: now +{now_ev:.1f} EP"
                     f"{' on a double' if family == '3xc' and is_double_now else ''} "
                     f"vs ~{hold_ev:.1f} holding for a future double — "
                     f"{verdict}{blocked}. {why}")

    if "wildcard" in playable:
        notes.append(f"[DATA] {label('wildcard')} available. [RECOMMENDATION] "
                     "Structural chip — hold unless squad EP falls >15% below "
                     "optimal (see the squad rating) or injuries have broken the squad.")
    if "freehit" in playable:
        notes.append(f"[MODEL] {label('freehit')}: structural chip — play on a blank "
                     "gameweek, not on EV; supply blank scenarios via signals for a number.")
    if winner is not None and len(scored) > 1:
        notes.append("[DATA] Only one chip may be played per gameweek.")
    if assumed and notes:
        notes.append(assumed.strip())
    return notes


def chip_check(boot: dict, squad_ep: pd.DataFrame | None, chips_available: list[str]) -> list[str]:
    """Back-compat wrapper: window flags only (no XI context available)."""
    return chip_advice(boot, None, chips_available)