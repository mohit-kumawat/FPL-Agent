"""Expected-points models.

Tiered design that survives the season boundary:

  Tier P (prior)    — prior-season baseline: blend of realized PPG and a ridge
                      prediction from per-90 process stats (xGI, xGC, ICT).
                      Players without history fall back to a position x price prior.
  Tier S (in-season)— ensemble of Model A (recency-weighted points) and
                      Model B (ridge on rolling per-90 features).
  Blend             — w = min(1, gws_played / COLD_START_GWS) on Tier S.

  EP(next GW)  = blend_ppg * xmins * fixture_mult(next) + set_piece_bonus
  EP(horizon)  = blend_ppg * xmins * sum(fixture_mult over horizon) + bonus * n_fx
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config

MIN_PRIOR_MINUTES = 900   # below this, prior-season stats are noise


# ------------------------------------------------------------------ minutes
def expected_minutes_fraction(df: pd.DataFrame) -> pd.Series:
    """P(plays) x share of 90 expected. Uses rolling minutes when the season is
    live, prior-season start rate preseason, price as a weak prior for new players."""
    gws = df["gws_played"].iloc[0] if len(df) else 0
    if gws > 0 and df["roll_minutes"].notna().any():
        frac = (df["roll_minutes"].fillna(0) / 90.0).clip(0, 1)
        # players yet to feature this season: fall back to prior start rate
        no_data = df["roll_minutes"].isna()
    else:
        frac = pd.Series(0.0, index=df.index)
        no_data = pd.Series(True, index=df.index)

    # blend season minutes-share with start-rate: a nailed-when-fit player who
    # missed a stretch (injury/AFCON) shouldn't be priced as a rotation risk
    mins_share = (df["minutes"].astype(float) / (38 * 90)).clip(0, 1)
    start_rate = (df["starts"].astype(float) / 38).clip(0, 1)
    prior_rate = (0.55 * start_rate + 0.45 * mins_share).clip(0, 1)
    # new signings / promoted with no FPL history: price implies expected role
    price_prior = ((df["price"] - 4.0) / 10.0).clip(0.3, 0.85)
    prior_rate = prior_rate.where(df["minutes"] >= _history_threshold(df), price_prior)

    frac = frac.where(~no_data, prior_rate)
    return (frac * df["play_chance"]).clip(0, 1)


# ------------------------------------------------------------- Tier P prior
_RIDGE_FEATURES = ["xgi_p90", "xgc_p90", "ict_p90", "start_rate"]


def _prior_per90_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-90 process stats from (prior-)season aggregates carried in bootstrap."""
    m = df["minutes"].astype(float).clip(lower=1)
    out = pd.DataFrame(index=df.index)
    out["xgi_p90"] = df["expected_goal_involvements"] * 90 / m
    out["xgc_p90"] = df["expected_goals_conceded"] * 90 / m
    out["ict_p90"] = df["ict_index"] * 90 / m
    out["start_rate"] = (df["starts"].astype(float) / 38).clip(0, 1)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def _history_threshold(df: pd.DataFrame) -> float:
    """Minutes needed to count as 'real history' — self-scales early season
    (at GW9 nobody has 900 minutes; half the max observed is a fair bar)."""
    max_mins = float(df["minutes"].max()) if len(df) else 0.0
    return min(MIN_PRIOR_MINUTES, max(90.0, 0.5 * max_mins))


def prior_baseline(df: pd.DataFrame) -> pd.Series:
    """Points-per-game prior for every player (Tier P)."""
    has_prior = df["minutes"] >= _history_threshold(df)
    ppg = pd.to_numeric(df["points_per_game"], errors="coerce").fillna(0.0)

    # ridge: fit process-stats -> ppg on players with real history, per position
    X_all = _prior_per90_frame(df)
    ridge_pred = pd.Series(np.nan, index=df.index)
    for pos in df["element_type"].unique():
        mask = (df["element_type"] == pos) & has_prior & (ppg > 0)
        if mask.sum() < 10:
            continue
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_all.loc[mask, _RIDGE_FEATURES])
        model = Ridge(alpha=config.RIDGE_ALPHA)
        model.fit(Xs, ppg[mask])
        pos_mask = df["element_type"] == pos
        ridge_pred[pos_mask] = model.predict(
            scaler.transform(X_all.loc[pos_mask, _RIDGE_FEATURES])
        )

    # 50/50 realized vs process (regresses hot/cold outcomes to underlying stats)
    prior = 0.5 * ppg + 0.5 * ridge_pred.fillna(ppg)

    # fallback tier: position x price bucket median for players without history
    df2 = df.assign(_prior=prior)
    bucket = (df["price"] * 2).round() / 2
    med = (
        df2[has_prior]
        .assign(_bucket=bucket[has_prior])
        .groupby(["element_type", "_bucket"])["_prior"]
        .median()
    )
    fallback = pd.Series(
        [med.get((et, b), np.nan) for et, b in zip(df["element_type"], bucket)],
        index=df.index,
    )
    # last resort: position median scaled by price rank
    pos_med = df2[has_prior].groupby("element_type")["_prior"].median()
    fallback = fallback.fillna(df["element_type"].map(pos_med) * (df["price"] / df["price"].median()).clip(0.7, 1.4))

    out = prior.where(has_prior, fallback)
    # never return NaN — a poisoned prior nulls the whole EP chain downstream
    return out.fillna(out.median()).fillna(1.0).clip(lower=0.0)


# ---------------------------------------------------------- Tier S in-season
def model_a_recency(panel: pd.DataFrame) -> pd.Series:
    """Recency-weighted mean of last FORM_WINDOW GW points, per player."""
    recent = panel.sort_values("round").groupby("element").tail(config.FORM_WINDOW)

    def wavg(g: pd.DataFrame) -> float:
        w = np.arange(1, len(g) + 1, dtype=float)
        return float(np.average(g["total_points"], weights=w))

    return recent.groupby("element").apply(wavg, include_groups=False)


def model_b_ridge_inseason(panel: pd.DataFrame) -> pd.Series:
    """Ridge on rolling per-90 process features -> next-GW points (in-season)."""
    p = panel.copy()
    for c in ["expected_goal_involvements", "expected_goals_conceded", "ict_index"]:
        p[c] = pd.to_numeric(p.get(c, 0), errors="coerce").fillna(0.0)
    p = p.sort_values(["element", "round"])
    g = p.groupby("element")
    feats = pd.DataFrame({
        "element": p["element"],
        "round": p["round"],
        "y": p["total_points"],
        "xgi_r": g["expected_goal_involvements"].transform(lambda s: s.rolling(config.FORM_WINDOW, min_periods=1).mean().shift(1)),
        "xgc_r": g["expected_goals_conceded"].transform(lambda s: s.rolling(config.FORM_WINDOW, min_periods=1).mean().shift(1)),
        "ict_r": g["ict_index"].transform(lambda s: s.rolling(config.FORM_WINDOW, min_periods=1).mean().shift(1)),
        "min_r": g["minutes"].transform(lambda s: s.rolling(config.FORM_WINDOW, min_periods=1).mean().shift(1)),
    }).dropna()
    if len(feats) < 100:
        return pd.Series(dtype=float)
    cols = ["xgi_r", "xgc_r", "ict_r", "min_r"]
    scaler = StandardScaler()
    model = Ridge(alpha=config.RIDGE_ALPHA)
    model.fit(scaler.fit_transform(feats[cols]), feats["y"])
    latest = feats.groupby("element").tail(1)
    pred = model.predict(scaler.transform(latest[cols]))
    return pd.Series(pred, index=latest["element"].values)


# ------------------------------------------------------ component model
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}     # 2025/26+ scoring (GK goal = 10)
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
DC_THRESHOLD = {1: 12, 2: 10, 3: 12, 4: 12}   # defensive contribution (2025/26+)


def component_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-90 point-generation components from whatever aggregates df carries.

    FPL points come from distinct processes; compressing them into one PPG
    number loses information (a 7.0-PPG player at 0.55 xGI/90 != one at 0.30
    who overperformed). Channels also let fixtures act where they belong:
    attack points scale with opponent defence, clean sheets with opponent attack.
    """
    m = df["minutes"].astype(float).clip(lower=90)

    def p90(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0) * 90 / m

    gp = df["element_type"].map(GOAL_PTS)
    attack90 = p90("expected_goals") * gp + p90("expected_assists") * 3

    xgc90 = p90("expected_goals_conceded").clip(0.2, 4.0)
    p_cs = np.exp(-xgc90)                       # Poisson P(0 conceded)
    cs90 = p_cs * df["element_type"].map(CS_PTS)
    # GK/DEF lose 1 pt per 2 conceded
    gc90 = -(xgc90 / 2.0).where(df["element_type"].isin([1, 2]), 0.0)

    dc90 = p90("defensive_contribution")
    dc_pts90 = 2.0 * (dc90 / df["element_type"].map(DC_THRESHOLD)).clip(0, 1) * 0.85

    neutral90 = p90("bonus") + p90("saves") / 3.0 + dc_pts90 + gc90
    return pd.DataFrame({"attack90": attack90, "cs90": cs90, "neutral90": neutral90})


# ---------------------------------------------------------------- ensemble
def expected_points(df: pd.DataFrame, panel: pd.DataFrame | None = None,
                    signal_adjust: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach ep_* columns. The single entry point for live and replay.

    Final EP = W_COMPONENT x component path + (1-W_COMPONENT) x realized path.
      component path: appearance + per-90 channels x channel fixture mults
      realized path:  ep_ppg (prior/A/B blend) x xmins x blended fixture mult
    The realized path keeps outcome information components can't see (bonus
    magnets, penalty share embedded in xG); the component path stops
    overperformance from being extrapolated.
    """
    out = df.copy()
    prior = prior_baseline(out)

    gws = int(out["gws_played"].iloc[0]) if len(out) else 0
    w_season = min(1.0, gws / config.COLD_START_GWS)

    if panel is not None and not panel.empty and gws > 0:
        a = model_a_recency(panel)
        b = model_b_ridge_inseason(panel)
        lam = config.ENSEMBLE_LAMBDA
        season_est = out["id"].map((1 - lam) * a.reindex(a.index.union(b.index)).fillna(0)
                                   + lam * b.reindex(a.index.union(b.index)).fillna(a))
        season_est = season_est.fillna(prior)
    else:
        season_est = prior

    # NaN-safe blend: 0 * NaN would poison valid estimates on either side
    out["ep_ppg"] = ((1 - w_season) * prior.fillna(0)
                     + w_season * season_est.fillna(prior.fillna(0)))
    out["xmins"] = expected_minutes_fraction(out)

    # minutes overrides from signals (news the API doesn't know yet)
    if signal_adjust is not None and len(signal_adjust):
        if "xmins_min" in signal_adjust:
            lo = out["id"].map(signal_adjust["xmins_min"])
            out["xmins"] = out["xmins"].where(lo.isna(), out["xmins"].clip(lower=lo.fillna(0)))
        if "xmins_max" in signal_adjust:
            hi = out["id"].map(signal_adjust["xmins_max"])
            out["xmins"] = out["xmins"].where(hi.isna(), out["xmins"].clip(upper=hi.fillna(1)))

    # ---- realized path (blended single multiplier) ----------------------
    base = out["ep_ppg"] * out["xmins"]
    legacy_next = base * out["next_gw_mult"]
    legacy_hor = base * out["ep_mult"]

    # ---- component path (channel-specific multipliers) -------------------
    comp = component_frame(out)
    # appearance: 1 pt for playing + 1 pt for 60'. P(play) saturates quickly;
    # P(60+) is a logistic around a 0.63 minutes expectation — smoother and less
    # optimistic for 50-59' rotation projections than a hard clip
    p_play = (out["xmins"] / 0.60).clip(0, 1)
    p60 = 1.0 / (1.0 + np.exp(-(out["xmins"] - 0.63) / 0.09))
    app_pts = p_play + p60
    n_next = out.get("next_n_fixtures", (out["next_gw_mult"] > 0).astype(int))
    comp_next = (app_pts * n_next
                 + out["xmins"] * (comp["attack90"] * out.get("next_att_mult", out["next_gw_mult"])
                                   + comp["cs90"] * out.get("next_def_mult", out["next_gw_mult"])
                                   + comp["neutral90"] * n_next))
    disc = out.get("disc_sum", out["n_fixtures"])
    comp_hor = (app_pts * disc
                + out["xmins"] * (comp["attack90"] * out.get("att_mult", out["ep_mult"])
                                  + comp["cs90"] * out.get("def_mult", out["ep_mult"])
                                  + comp["neutral90"] * disc))

    w = config.W_COMPONENT
    out["ep_next"] = (w * comp_next + (1 - w) * legacy_next
                      + out["sp_bonus"] * (n_next > 0))
    out["ep_horizon"] = (w * comp_hor + (1 - w) * legacy_hor
                         + out["sp_bonus"] * out["n_fixtures"])
    out["ep_attack90"] = comp["attack90"]
    out["ep_cs90"] = comp["cs90"]

    if signal_adjust is not None and len(signal_adjust) and "ep_per_gw" in signal_adjust:
        adj = out["id"].map(signal_adjust["ep_per_gw"]).fillna(0.0)
        out["ep_next"] = (out["ep_next"] + adj).clip(lower=0)
        out["ep_horizon"] = (out["ep_horizon"] + adj * config.HORIZON_GWS).clip(lower=0)

    # variance: attack points are Poisson-ish (high variance), CS binary,
    # appearance near-deterministic — grounded better than a flat sqrt(ppg).
    # The 1.3 factor is empirical: uncalibrated bands covered 68% of top-50
    # actuals vs the 80% target on the 2024/25 replay (eval/run_backtests.py)
    out["ep_sd"] = 1.3 * (np.sqrt((comp["attack90"] * out["xmins"]).clip(lower=0.1)) * 1.9
                          + 0.55 * np.sqrt(comp["cs90"].clip(lower=0)) + 0.3)
    out["ep_p10"] = (out["ep_next"] - 1.28 * out["ep_sd"]).clip(lower=0)
    out["ep_p90"] = out["ep_next"] + 1.28 * out["ep_sd"]
    return out
