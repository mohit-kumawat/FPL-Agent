"""Squad rating card: how good is the current 15 vs the optimal build?"""
from __future__ import annotations

import pandas as pd

from . import config, optimizer


def _grade(pct: float) -> str:
    for cut, g in [(97, "A+"), (92, "A"), (85, "B+"), (78, "B"), (70, "C+"), (60, "C")]:
        if pct >= cut:
            return g
    return "D"


def rate_squad(players: pd.DataFrame, squad_ids: list[int],
               ep_col: str = "ep_horizon") -> dict:
    """Score a 15-man squad against the unconstrained optimal build."""
    squad = players[players["id"].isin(squad_ids)].copy()
    if len(squad) != config.SQUAD_SIZE:
        raise ValueError(f"Expected 15 players, got {len(squad)}")

    optimal = optimizer.build_squad(players, ep_col=ep_col)
    xi_mine = optimizer.pick_xi(squad, ep_col=ep_col)
    xi_opt = optimizer.pick_xi(optimal["squad"], ep_col=ep_col)

    mine_ep = xi_mine["expected_points"]
    opt_ep = xi_opt["expected_points"]
    pct = 100.0 * mine_ep / opt_ep if opt_ep else 0.0

    # per-player grades: EP vs best same-position player within +-0.5m price
    grades = []
    for _, p in squad.iterrows():
        peers = players[
            (players["element_type"] == p["element_type"])
            & (players["price"].between(p["price"] - 0.5, p["price"] + 0.5))
            & players["available"]
        ]
        peer_best = peers[ep_col].max()
        ppct = 100.0 * p[ep_col] / peer_best if peer_best else 0.0
        grades.append({
            "player": p["web_name"], "pos": p["position"], "price": p["price"],
            "ep": round(float(p[ep_col]), 1), "vs_best_in_bracket": round(ppct),
            "grade": _grade(ppct),
        })

    risks = []
    club_counts = squad["team_short"].value_counts()
    if (club_counts >= 3).any():
        risks.append(f"[DATA] 3-player exposure: {', '.join(club_counts[club_counts >= 3].index)}")
    flagged = squad[squad["play_chance"] < 0.8]
    if len(flagged):
        risks.append(f"[DATA] Injury/availability flags: {', '.join(flagged['web_name'])}")
    bench_ep = squad.nsmallest(4, ep_col)[ep_col].sum()
    if bench_ep > 0.22 * squad[ep_col].sum():
        risks.append("[MODEL] Bench is expensive relative to XI — value trapped on bench.")
    hard_run = squad.merge(
        players[["id", "avg_fdr"]], on="id", how="left", suffixes=("", "_y")
    )["avg_fdr"].mean()
    if hard_run and hard_run > 3.2:
        risks.append(f"[DATA] Tough fixture run ahead (avg FDR {hard_run:.1f}).")

    return {
        "overall_pct": round(pct, 1),
        "overall_grade": _grade(pct),
        "my_xi_ep": mine_ep,
        "optimal_xi_ep": opt_ep,
        "ep_gap": round(opt_ep - mine_ep, 2),
        "player_grades": grades,
        "risks": risks,
        "optimal_squad": optimal["squad"][["web_name", "team_short", "position", "price"]],
    }
