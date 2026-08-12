"""Monte Carlo outcomes: turn EP components + the minutes distribution into
per-player point distributions.

FPL returns are strongly non-normal (a wall of 2s, a rare 15), so captaincy
and chip calls need tail probabilities, not a normal band around the mean.
This simulator draws each scoring process from the same per-90 rates the EP
model uses:

    appearance      plays / reaches 60' from p_start, p_bench, p_60
    goals, assists  Poisson from xG90 / xA90 x fixture attack channel
    clean sheets    Poisson goals-conceded from xGC90 x defence channel
                    (EXPLICITLY inherits the Poisson assumption — this makes
                    the existing approximation stochastic, not better; a
                    bivariate scoreline model is future work)
    saves, DC,      Poisson / Bernoulli at component rates
    bonus
    cards, own      league-ballpark Bernoulli rates (small point sources —
    goals, pens     individually minor, collectively they separate 5.8 from 5.6)

A double gameweek is treated as one aggregate exposure (rates scale with
fixture count); per-match clean sheets in doubles are therefore slightly
understated. A blank is exactly zero.

Deterministic: same frame + same seed -> same output (CI-testable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .models import CS_PTS, DC_THRESHOLD, GOAL_PTS

SEED = 20260812
N_TRIALS = 2000

# small point sources: league-frequency ballpark per 90 minutes; these are
# deliberately coarse — they widen tails, they do not decide rankings
YELLOW_P90 = 0.12
RED_P90 = 0.004
OWN_GOAL_P90 = 0.003
PEN_MISS_P90 = 0.03      # first-choice taker only
PEN_SAVE_P90 = 0.025     # goalkeepers

HAUL_PTS = 10.0
BLANK_PTS = 2.0


def _p90(df: pd.DataFrame, col: str) -> np.ndarray:
    m = df["minutes"].astype(float).clip(lower=90).to_numpy()
    if col not in df.columns:
        return np.zeros(len(df))
    v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    return v * 90.0 / m


def simulate_players(ep: pd.DataFrame, n_trials: int = N_TRIALS,
                     seed: int = SEED) -> pd.DataFrame:
    """Next-GW point distribution per player.

    Returns a frame indexed like `ep` with sim_mean, sim_p10/p50/p90,
    p_haul (>= 10 pts) and p_blank (<= 2 pts).
    """
    rng = np.random.default_rng(seed)
    P = len(ep)
    et = ep["element_type"].to_numpy()

    n_fx = ep.get("next_n_fixtures", pd.Series(1, index=ep.index)).to_numpy().astype(float)
    att_mult = ep.get("next_att_mult", ep.get("next_gw_mult", pd.Series(1.0, index=ep.index))).to_numpy()
    def_mult = ep.get("next_def_mult", ep.get("next_gw_mult", pd.Series(1.0, index=ep.index))).to_numpy()
    # channel multipliers sum over a GW's fixtures; split difficulty from count
    per_fx_att = np.divide(att_mult, n_fx, out=np.ones(P), where=n_fx > 0)
    per_fx_def = np.clip(np.divide(def_mult, n_fx, out=np.ones(P), where=n_fx > 0), 0.5, 2.0)

    p_start = ep["p_start"].to_numpy()
    p_bench = ep["p_bench"].to_numpy()
    p_60 = ep["p_60"].to_numpy()

    xg90 = _p90(ep, "expected_goals")
    xa90 = _p90(ep, "expected_assists")
    xgc90 = np.clip(_p90(ep, "expected_goals_conceded"), 0.2, 4.0)
    saves90 = _p90(ep, "saves")
    bonus90 = _p90(ep, "bonus")
    dc90 = _p90(ep, "defensive_contribution")
    dc_thr = np.array([DC_THRESHOLD[t] for t in et], dtype=float)
    p_dc = np.clip(dc90 / dc_thr, 0, 1) * 0.85

    goal_pts = np.array([GOAL_PTS[t] for t in et], dtype=float)
    cs_pts = np.array([CS_PTS[t] for t in et], dtype=float)
    is_gk = et == 1
    is_gk_def = (et == 1) | (et == 2)
    pen_taker = (pd.to_numeric(ep.get("penalties_order"), errors="coerce") == 1).to_numpy() \
        if "penalties_order" in ep.columns else np.zeros(P, dtype=bool)

    # ---- calibration ---------------------------------------------------------
    # The draws use component rates, but production ep_next blends in the
    # realized-PPG path (W_COMPONENT). Left uncalibrated, a bonus magnet or
    # penalty-share player would simulate below the EP the ranking trusts.
    # Scale the positive-rate channels so the analytic simulation mean lands
    # near ep_next; shapes stay Poisson/Bernoulli, only rates move.
    xmins = ep["xmins"].to_numpy()
    p_play_a = np.clip(p_start + p_bench, 0, 1)
    e_app = (p_play_a + p_60) * n_fx
    e_goals = xg90 * xmins * n_fx * per_fx_att * goal_pts
    e_assists = xa90 * xmins * n_fx * per_fx_att * 3.0
    e_cs = p_60 * np.exp(-xgc90 * n_fx / per_fx_def) * cs_pts
    e_gc = -np.where(is_gk_def, xgc90 * xmins * n_fx / per_fx_def / 2.0, 0.0)
    e_saves = np.where(is_gk, saves90 * xmins * n_fx / 3.0, 0.0)
    e_dc = 2.0 * n_fx * p_dc * xmins
    e_bonus = np.clip(bonus90, 0, 3) * xmins * n_fx
    e_scalable = e_goals + e_assists + e_saves + e_dc + e_bonus
    e_fixed = e_app + e_cs + e_gc
    target = ep["ep_next"].to_numpy() - ep.get("sp_bonus", pd.Series(0.0, index=ep.index)).to_numpy()
    scale = np.clip(np.divide(target - e_fixed, e_scalable,
                              out=np.ones(P), where=e_scalable > 0.05),
                    0.6, 1.8)
    xg90, xa90 = xg90 * scale, xa90 * scale
    saves90, bonus90 = saves90 * scale, bonus90 * scale
    p_dc = np.clip(p_dc * scale, 0, 1)

    # ---- appearance fork ----------------------------------------------------
    # a blank GW zeroes the whole fork — otherwise "played 60' in no fixture"
    # would still qualify for a clean sheet against zero expected goals
    has_fx = (n_fx > 0)[:, None]
    r = rng.random((P, n_trials))
    started = (r < p_start[:, None]) & has_fx
    benched = ~started & (r < (p_start + p_bench)[:, None]) & has_fx
    p60_given_start = np.divide(p_60, p_start, out=np.zeros(P), where=p_start > 0).clip(0, 1)
    sixty = started & (rng.random((P, n_trials)) < p60_given_start[:, None])
    minutes = np.where(sixty, 90.0, np.where(started, 55.0, np.where(benched, 20.0, 0.0)))
    exposure = (minutes / 90.0) * n_fx[:, None]           # blank -> 0, double -> 2x

    pts = np.zeros((P, n_trials))
    plays = minutes > 0
    pts += (plays * 1.0 + sixty * 1.0) * n_fx[:, None]    # appearance points

    # ---- attacking returns --------------------------------------------------
    pts += rng.poisson(xg90[:, None] * exposure * per_fx_att[:, None]) * goal_pts[:, None]
    pts += rng.poisson(xa90[:, None] * exposure * per_fx_att[:, None]) * 3.0

    # ---- defensive returns --------------------------------------------------
    conceded = rng.poisson(xgc90[:, None] * exposure / per_fx_def[:, None])
    pts += (sixty & (conceded == 0)) * cs_pts[:, None]
    pts -= np.where(is_gk_def[:, None], conceded // 2, 0)
    pts += np.where(is_gk[:, None],
                    rng.poisson(saves90[:, None] * exposure) // 3, 0)
    pts += 2.0 * rng.binomial(n_fx[:, None].astype(int),
                              np.clip(p_dc[:, None] * (minutes / 90.0), 0, 1))

    # ---- bonus + small point sources ---------------------------------------
    pts += np.minimum(rng.poisson(np.clip(bonus90[:, None], 0, 3) * exposure),
                      3 * n_fx[:, None])
    pts -= 1.0 * (rng.random((P, n_trials)) < YELLOW_P90 * exposure)
    pts -= 3.0 * (rng.random((P, n_trials)) < RED_P90 * exposure)
    pts -= 2.0 * (rng.random((P, n_trials)) < OWN_GOAL_P90 * exposure)
    pts -= 2.0 * (pen_taker[:, None] & (rng.random((P, n_trials)) < PEN_MISS_P90 * exposure))
    pts += 5.0 * (is_gk[:, None] & (rng.random((P, n_trials)) < PEN_SAVE_P90 * exposure))

    q = np.percentile(pts, [10, 50, 90], axis=1)
    return pd.DataFrame({
        "sim_mean": pts.mean(axis=1).round(2),
        "sim_p10": q[0].round(1),
        "sim_p50": q[1].round(1),
        "sim_p90": q[2].round(1),
        "p_haul": (pts >= HAUL_PTS).mean(axis=1).round(3),
        "p_blank": (pts <= BLANK_PTS).mean(axis=1).round(3),
    }, index=ep.index)


def captain_outlook(ep: pd.DataFrame, candidate_ids: list[int],
                    n_trials: int = N_TRIALS, seed: int = SEED) -> pd.DataFrame:
    """Simulation summary for captaincy candidates, captain points = 2x."""
    cand = ep[ep["id"].isin(candidate_ids)]
    sim = simulate_players(cand, n_trials=n_trials, seed=seed)
    out = cand[["id", "web_name", "ep_next"]].join(sim)
    for c in ("sim_mean", "sim_p10", "sim_p50", "sim_p90"):
        out[f"cap_{c[4:]}"] = (2 * out[c]).round(1)
    return out.sort_values("sim_mean", ascending=False)
