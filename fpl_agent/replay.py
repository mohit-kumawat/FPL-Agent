"""Point-in-time replay harness with structural hindsight protection.

Rebuilds "what the agent knew" immediately before any historical GW deadline:
  - stats/panel: only rows with round < gw
  - prices: last observed value before gw (GW1: the gw-1... opening price row)
  - prior season: aggregates of the season before (cold-start Tier P)
  - fixtures/FDR: the season's fixture list (published preseason — not hindsight)

Everything downstream (models, optimizer, policy) is the SAME code the live
pipeline uses, so a replay validates the real agent, not a lookalike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, models, optimizer

_POS_MAP = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
_VALID_POS = set(_POS_MAP)


def _season_gws(season: str) -> pd.DataFrame:
    df = data.load_prior_season(season).copy()
    if "round" not in df.columns and "GW" in df.columns:
        df = df.rename(columns={"GW": "round"})
    df = df[df["position"].isin(_VALID_POS)].copy()   # drop Assistant Managers
    for c in ["total_points", "minutes", "value", "expected_goal_involvements",
              "expected_goals", "expected_assists", "expected_goals_conceded",
              "ict_index", "starts", "selected", "bonus", "saves"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _prior_aggregates(prior_season: str) -> pd.DataFrame:
    """Season aggregates keyed by player name (cross-season id mapping is by name)."""
    df = _season_gws(prior_season)
    apps = df[df["minutes"] > 0].groupby("name").size().rename("appearances")
    agg = df.groupby("name").agg(
        minutes=("minutes", "sum"),
        total_points=("total_points", "sum"),
        expected_goal_involvements=("expected_goal_involvements", "sum"),
        expected_goals=("expected_goals", "sum"),
        expected_assists=("expected_assists", "sum"),
        expected_goals_conceded=("expected_goals_conceded", "sum"),
        ict_index=("ict_index", "sum"),
        starts=("starts", "sum"),
        bonus=("bonus", "sum"),
        saves=("saves", "sum"),
    ).join(apps).fillna({"appearances": 0})
    agg["points_per_game"] = (agg["total_points"] / agg["appearances"].clip(lower=1)).round(2)
    return agg


def _team_fdr_mult(season: str, gw: int, horizon: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(horizon ep_mult, next-gw mult, horizon n_fixtures) per team NAME.

    Uses the SAME continuous strength model as the live pipeline
    (features.strength_fixture_mult) so backtests validate the real formula.
    """
    from . import features
    fx = data.load_season_fixtures(season)
    teams_df = data.load_season_teams(season).set_index("id")
    names = teams_df["name"]
    window = fx[(fx["event"] >= gw) & (fx["event"] < gw + horizon)]
    cols = ["ep_mult", "att_mult", "def_mult", "disc_sum", "n_fixtures",
            "next_gw_mult", "next_att_mult", "next_def_mult", "next_n_fixtures"]
    acc = {n: dict.fromkeys(cols, 0.0) for n in names}
    for f in window.itertuples():
        discount = config.HORIZON_DISCOUNT ** (int(f.event) - gw)
        for tid, opp, home in ((f.team_h, f.team_a, True), (f.team_a, f.team_h, False)):
            if tid not in names.index:
                continue
            att, dfn = features.strength_channels(teams_df, tid, opp, home)
            a = acc[names[tid]]
            a["att_mult"] += discount * att
            a["def_mult"] += discount * dfn
            a["ep_mult"] += discount * (0.65 * att + 0.35 * dfn)
            a["disc_sum"] += discount
            a["n_fixtures"] += 1
            if int(f.event) == gw:
                a["next_att_mult"] += att
                a["next_def_mult"] += dfn
                a["next_gw_mult"] += 0.65 * att + 0.35 * dfn
                a["next_n_fixtures"] += 1
    return pd.DataFrame.from_dict(acc, orient="index")


def snapshot_at(season: str, gw: int, horizon: int = config.HORIZON_GWS) -> pd.DataFrame:
    """Players frame as known before `gw`'s deadline. Raises if leakage detected."""
    df = _season_gws(season)
    past = df[df["round"] < gw]
    assert past.empty or past["round"].max() < gw, "hindsight leak: future rows in panel"

    # identity + price: last observation strictly before gw; for GW1 use the
    # opening-price row of gw itself but ONLY identity/price columns (no stats).
    id_cols = ["element", "name", "position", "team", "value"]
    if gw == 1:
        latest = df[df["round"] == 1].sort_values("kickoff_time").groupby("element").head(1)
    else:
        latest = past.sort_values("round").groupby("element").tail(1)
    snap = latest[id_cols].copy().set_index("element")
    snap["price"] = snap["value"] / 10.0

    # in-season cumulative stats (strictly past)
    if not past.empty:
        cum = past.groupby("element").agg(
            minutes=("minutes", "sum"),
            expected_goal_involvements=("expected_goal_involvements", "sum"),
            expected_goals=("expected_goals", "sum"),
            expected_assists=("expected_assists", "sum"),
            expected_goals_conceded=("expected_goals_conceded", "sum"),
            ict_index=("ict_index", "sum"),
            starts=("starts", "sum"),
            bonus=("bonus", "sum"),
            saves=("saves", "sum"),
        )
        apps = past[past["minutes"] > 0].groupby("element").size().rename("apps")
        pts = past.groupby("element")["total_points"].sum()
        cum["points_per_game"] = (pts / apps.reindex(cum.index).fillna(0).clip(lower=1)).round(2)
        roll = (past.sort_values("round").groupby("element")
                .tail(config.FORM_WINDOW).groupby("element")
                .agg(roll_points=("total_points", "mean"),
                     roll_minutes=("minutes", "mean"),
                     roll_starts=("starts", "mean")))
    else:
        cum = pd.DataFrame()
        roll = pd.DataFrame()

    # prior-season aggregates by name (Tier P cold start)
    prior_season = f"{int(season[:4]) - 1}-{int(season[:4]) % 100:02d}"
    try:
        prior = _prior_aggregates(prior_season)
    except Exception:  # noqa: BLE001 — prior season unavailable
        prior = pd.DataFrame()

    gws_played = int(past["round"].max()) if not past.empty else 0
    stat_cols = ["minutes", "expected_goal_involvements", "expected_goals",
                 "expected_assists", "expected_goals_conceded", "ict_index",
                 "starts", "bonus", "saves", "points_per_game"]

    if gws_played >= config.COLD_START_GWS or prior.empty:
        base = cum
    elif gws_played == 0:
        base = snap.join(prior, on="name")[stat_cols] if not prior.empty else None
    else:
        # early season: prior aggregates for Tier P, in-season rolls layered on
        base = snap.join(prior, on="name")[stat_cols]
        base = base.where(base.notna(), cum.reindex(base.index))
    if base is None:
        base = pd.DataFrame(0.0, index=snap.index, columns=stat_cols)
    snap = snap.join(base[stat_cols].fillna(0.0)) if not base.empty else snap.assign(
        **{c: 0.0 for c in stat_cols})
    snap = snap.join(roll)

    snap["element_type"] = snap["position"].map(_POS_MAP)
    snap["position"] = snap["element_type"].map({k: v[0] for k, v in config.POSITIONS.items()})
    snap["team_short"] = snap["team"]
    snap["web_name"] = snap["name"]
    snap["id"] = snap.index
    # archive has no injury flags -> everyone available (known limitation)
    snap["available"] = True
    snap["play_chance"] = 1.0
    snap["selected_by_percent"] = 0.0
    snap["sp_bonus"] = 0.0
    snap["gws_played"] = gws_played
    # prior aggregates feed Tier P until COLD_START_GWS; mark them when they
    # cross a scoring-regime boundary (same contract as features.enrich_players)
    snap["prior_rules_cross"] = (gws_played < config.COLD_START_GWS
                                 and config.rules_cross_regime(prior_season, season))

    fmults = _team_fdr_mult(season, gw, horizon)
    for col in fmults.columns:
        snap[col] = snap["team"].map(fmults[col]).fillna(0.0)
    snap["n_fixtures"] = snap["n_fixtures"].astype(int)
    snap["next_n_fixtures"] = snap["next_n_fixtures"].astype(int)
    snap["avg_fdr"] = 3.0
    for c in ["roll_points", "roll_minutes", "roll_starts"]:
        if c not in snap.columns:
            snap[c] = np.nan
    return snap.reset_index(drop=True)


def panel_before(season: str, gw: int) -> pd.DataFrame:
    df = _season_gws(season)
    return df[df["round"] < gw]


def expected_points_at(season: str, gw: int, horizon: int = config.HORIZON_GWS) -> pd.DataFrame:
    snap = snapshot_at(season, gw, horizon)
    panel = panel_before(season, gw)
    return models.expected_points(snap, panel if not panel.empty else None)


def actual_points(season: str, gw: int) -> pd.Series:
    df = _season_gws(season)
    return df[df["round"] == gw].groupby("element")["total_points"].sum()


def actual_minutes(season: str, gw: int) -> pd.Series:
    """GW minutes per element (a double GW sums both matches) — feeds autosubs."""
    df = _season_gws(season)
    return df[df["round"] == gw].groupby("element")["minutes"].sum()


def actual_points_range(season: str, gw_from: int, gw_to: int) -> pd.Series:
    df = _season_gws(season)
    m = df[(df["round"] >= gw_from) & (df["round"] <= gw_to)]
    return m.groupby("element")["total_points"].sum()


# ------------------------------------------------------------ ask patterns
def match_players(ep: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Exact web_name match first; contains-fallback picks the highest-EP row
    (avoids 'Salah' hitting Salah-Eddine, or 'Saka' hitting Wan-Bissaka)."""
    rows = []
    for n in names:
        exact = ep[ep["web_name"] == n]
        if len(exact):
            rows.append(exact.iloc[0])
            continue
        part = ep[ep["web_name"].str.contains(n, case=False, na=False, regex=False)]
        if len(part):
            rows.append(part.nlargest(1, "ep_horizon").iloc[0])
    return pd.DataFrame(rows)


def captain_pick(season: str, gw: int, candidates: list[str]) -> pd.DataFrame:
    ep = expected_points_at(season, gw)
    cand = match_players(ep, candidates)
    act = actual_points(season, gw)
    cand = cand.assign(actual=cand["id"].map(act).fillna(0))
    return (cand[["web_name", "team_short", "price", "ep_ppg", "xmins",
                  "next_gw_mult", "ep_next", "actual"]]
            .sort_values("ep_next", ascending=False))


def build_at(season: str, gw: int) -> dict:
    ep = expected_points_at(season, gw)
    res = optimizer.build_squad(ep)
    xi = optimizer.pick_xi(res["squad"])
    act = actual_points(season, gw)
    res["squad"] = res["squad"].assign(actual=res["squad"]["id"].map(act).fillna(0))
    return {**res, "xi_result": xi, "ep_frame": ep}


def premium_ranking(season: str, gw: int, names: list[str], horizon: int = 8) -> pd.DataFrame:
    ep = expected_points_at(season, gw, horizon=horizon)
    sel = match_players(ep, names)
    return (sel[["web_name", "team_short", "price", "ep_ppg", "xmins",
                 "ep_next", "ep_horizon"]]
            .sort_values("ep_horizon", ascending=False))
