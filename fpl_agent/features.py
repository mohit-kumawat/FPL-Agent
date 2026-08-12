"""Feature layer: per-player expected-points inputs + fixture difficulty horizon."""
from __future__ import annotations

import pandas as pd

from . import config


def strength_channels(teams: pd.DataFrame, team_id: int, opp_id: int,
                      is_home: bool) -> tuple[float, float]:
    """(attack_mult, defence_mult) for one fixture from FPL strength ratings.

    attack_mult scales goal/assist expectation (my venue attack vs their venue
    defence); defence_mult scales clean-sheet probability (my defence vs their
    attack). Separate channels because a fixture that's hard for a forward is
    not equally hard for a goalkeeper. Clipped; 1.0 fallback if data missing.
    """
    lo, hi = config.STRENGTH_CLIP
    try:
        me, op = teams.loc[team_id], teams.loc[opp_id]
        if is_home:
            att = me["strength_attack_home"] / op["strength_defence_away"]
            dfn = me["strength_defence_home"] / op["strength_attack_away"]
        else:
            att = me["strength_attack_away"] / op["strength_defence_home"]
            dfn = me["strength_defence_away"] / op["strength_attack_home"]
    except (KeyError, ZeroDivisionError):
        return 1.0, 1.0
    g = config.STRENGTH_GAMMA
    return (float(min(hi, max(lo, att ** g))),
            float(min(hi, max(lo, dfn ** g))))


def strength_fixture_mult(teams: pd.DataFrame, team_id: int, opp_id: int,
                          is_home: bool) -> float:
    """Blended single multiplier (65% attack / 35% defence) — used where one
    number is needed (reporting, replay compatibility)."""
    att, dfn = strength_channels(teams, team_id, opp_id, is_home)
    return 0.65 * att + 0.35 * dfn


def fixture_multipliers(boot: dict, fixtures: list[dict], horizon: int = config.HORIZON_GWS) -> pd.DataFrame:
    """Per-team fixture outlook over the next `horizon` GWs.

    Returns one row per team with:
      n_fixtures  — matches in horizon (blanks/doubles handled naturally)
      ep_mult     — discounted sum of continuous strength multipliers
                    (a double GW sums both matches; later GWs count less)
      avg_fdr     — mean FDR faced (kept for reporting/risk flags)
    """
    events = boot["events"]
    nxt = next((e["id"] for e in events if e["is_next"]), None)
    if nxt is None:  # season over
        nxt = 39
    window = range(nxt, min(nxt + horizon, 39))
    fx = pd.DataFrame(fixtures)
    fx = fx[fx["event"].isin(window)]
    teams = pd.DataFrame(boot["teams"]).set_index("id")

    rows: list[dict] = []
    for t in boot["teams"]:
        tid = t["id"]
        att_s, def_s, disc_s, fdrs = 0.0, 0.0, 0.0, []
        for f in fx.itertuples():
            if f.team_h == tid:
                opp, home, fdr = f.team_a, True, f.team_h_difficulty
            elif f.team_a == tid:
                opp, home, fdr = f.team_h, False, f.team_a_difficulty
            else:
                continue
            discount = config.HORIZON_DISCOUNT ** (int(f.event) - nxt)
            att, dfn = strength_channels(teams, tid, opp, home)
            att_s += discount * att
            def_s += discount * dfn
            disc_s += discount
            fdrs.append(fdr)
        rows.append({
            "team": tid,
            "team_short": t["short_name"],
            "n_fixtures": len(fdrs),
            "att_mult": att_s,          # discounted sum of attack channels
            "def_mult": def_s,          # discounted sum of defence channels
            "disc_sum": disc_s,         # discounted fixture count (for neutral pts)
            "ep_mult": 0.65 * att_s + 0.35 * def_s,   # blended, for reports
            "avg_fdr": (sum(fdrs) / len(fdrs)) if fdrs else None,
        })
    return pd.DataFrame(rows).set_index("team")


def next_gw_fixture_mult(boot: dict, fixtures: list[dict]) -> pd.Series:
    """FDR multiplier for just the next GW (0 for a blank, summed for a double)."""
    out = fixture_multipliers(boot, fixtures, horizon=1)
    return out["ep_mult"]


def set_piece_bonus(players: pd.DataFrame) -> pd.Series:
    """Small role-safety bonus for set-piece duty.

    Deliberately small: an incumbent taker's penalty/set-piece output is already
    embedded in his historical xG/xA, so a large injection double-counts. This
    covers only the marginal safety of holding the duty. A player who newly
    GAINS pen duty should get an explicit signal (ep_per_gw — the sanctioned
    last-resort use; see AGENT.md's signal policy), not a table edit.
    """
    bonus = pd.Series(0.0, index=players.index)
    pens = pd.to_numeric(players.get("penalties_order"), errors="coerce")
    bonus[pens == 1] += 0.2
    bonus[pens == 2] += 0.05
    corners = pd.to_numeric(players.get("corners_and_indirect_freekicks_order"), errors="coerce")
    bonus[corners == 1] += 0.05
    return bonus


def enrich_players(players: pd.DataFrame, boot: dict, fixtures: list[dict],
                   panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach fixture outlook, set-piece bonus, and rolling form (when panel exists)."""
    df = players.copy()
    fm = fixture_multipliers(boot, fixtures)
    df = df.join(fm[["n_fixtures", "att_mult", "def_mult", "disc_sum",
                     "ep_mult", "avg_fdr"]], on="team")
    fm1 = fixture_multipliers(boot, fixtures, horizon=1)
    df["next_gw_mult"] = df["team"].map(fm1["ep_mult"]).fillna(0.0)
    df["next_att_mult"] = df["team"].map(fm1["att_mult"]).fillna(0.0)
    df["next_def_mult"] = df["team"].map(fm1["def_mult"]).fillna(0.0)
    df["next_n_fixtures"] = df["team"].map(fm1["n_fixtures"]).fillna(0).astype(int)
    df["sp_bonus"] = set_piece_bonus(df)

    gws_played = sum(1 for e in boot["events"] if e["finished"])
    df["gws_played"] = gws_played

    if panel is not None and not panel.empty:
        recent = (
            panel.sort_values("round")
            .groupby("element")
            .tail(config.FORM_WINDOW)
            .groupby("element")
            .agg(
                roll_points=("total_points", "mean"),
                roll_minutes=("minutes", "mean"),
                roll_starts=("starts", "mean") if "starts" in panel.columns else ("minutes", lambda m: (m > 60).mean()),
            )
        )
        df = df.join(recent, on="id")
    else:
        df["roll_points"] = float("nan")
        df["roll_minutes"] = float("nan")
        df["roll_starts"] = float("nan")
    return df
