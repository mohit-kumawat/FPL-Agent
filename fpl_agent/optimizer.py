"""ILP optimizers (PuLP/CBC): fresh 15-man build, XI + captain + bench order,
and a transfer planner that respects free-transfer banking and hit costs."""
from __future__ import annotations

import pandas as pd
import pulp as pl

from . import config


def _solve(prob: pl.LpProblem) -> None:
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"ILP not optimal: {pl.LpStatus[prob.status]}")


def _chosen(var: dict, players: pd.DataFrame) -> pd.DataFrame:
    idx = [i for i, v in var.items() if pl.value(v) > 0.5]
    return players.loc[idx]


# ------------------------------------------------------------ fresh 15 build
def build_squad(players: pd.DataFrame, budget: float = config.BUDGET,
                ep_col: str = "ep_horizon", locked: list[int] | None = None,
                banned: list[int] | None = None) -> dict:
    """Optimal 15-man squad from scratch under full FPL rules.

    Objective: starters count fully, bench at BENCH_WEIGHT — this stops the
    solver spending real money on bench players.
    """
    pool = players[players["available"] & (players["xmins"] > 0.05)].copy()
    if banned:
        pool = pool[~pool["id"].isin(banned)]

    x = {i: pl.LpVariable(f"sq_{i}", cat="Binary") for i in pool.index}   # in 15
    s = {i: pl.LpVariable(f"xi_{i}", cat="Binary") for i in pool.index}   # in XI
    c = {i: pl.LpVariable(f"cap_{i}", cat="Binary") for i in pool.index}  # captain

    prob = pl.LpProblem("fpl_build", pl.LpMaximize)
    prob += pl.lpSum(
        pool.loc[i, ep_col] * (s[i] + c[i] + config.BENCH_WEIGHT * (x[i] - s[i]))
        for i in pool.index
    )
    prob += pl.lpSum(x.values()) == config.SQUAD_SIZE
    prob += pl.lpSum(s.values()) == config.XI_SIZE
    prob += pl.lpSum(c.values()) == 1
    prob += pl.lpSum(pool.loc[i, "price"] * x[i] for i in pool.index) <= budget
    for i in pool.index:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]
    for et, (_, squad_n, xi_min, xi_max) in config.POSITIONS.items():
        idx = pool.index[pool["element_type"] == et]
        prob += pl.lpSum(x[i] for i in idx) == squad_n
        prob += pl.lpSum(s[i] for i in idx) >= xi_min
        prob += pl.lpSum(s[i] for i in idx) <= xi_max
    for t in pool["team"].unique():
        idx = pool.index[pool["team"] == t]
        prob += pl.lpSum(x[i] for i in idx) <= config.MAX_PER_CLUB
    if locked:
        for pid in locked:
            idx = pool.index[pool["id"] == pid]
            if len(idx):
                prob += x[idx[0]] == 1
    _solve(prob)

    squad = _chosen(x, pool)
    xi = _chosen(s, pool)
    return {
        "squad": squad,
        "xi": xi,
        "bench": squad.loc[~squad.index.isin(xi.index)],
        "cost": round(float(squad["price"].sum()), 1),
        "objective": round(pl.value(prob.objective), 2),
    }


# --------------------------------------------------- XI + captain from squad
def pick_xi(squad: pd.DataFrame, ep_col: str = "ep_next") -> dict:
    """Best XI, captain, vice, and bench order from a fixed 15."""
    s = {i: pl.LpVariable(f"xi_{i}", cat="Binary") for i in squad.index}
    c = {i: pl.LpVariable(f"cap_{i}", cat="Binary") for i in squad.index}

    prob = pl.LpProblem("fpl_xi", pl.LpMaximize)
    prob += pl.lpSum(squad.loc[i, ep_col] * (s[i] + c[i]) for i in squad.index)
    prob += pl.lpSum(s.values()) == config.XI_SIZE
    prob += pl.lpSum(c.values()) == 1
    for i in squad.index:
        prob += c[i] <= s[i]
    for et, (_, _, xi_min, xi_max) in config.POSITIONS.items():
        idx = squad.index[squad["element_type"] == et]
        prob += pl.lpSum(s[i] for i in idx) >= xi_min
        prob += pl.lpSum(s[i] for i in idx) <= xi_max
    _solve(prob)

    xi = _chosen(s, squad)
    captain = _chosen(c, squad).iloc[0]
    vice = xi[xi["id"] != captain["id"]].sort_values(ep_col, ascending=False).iloc[0]
    bench = squad.loc[~squad.index.isin(xi.index)]
    bench_gk = bench[bench["element_type"] == 1]
    bench_out = bench[bench["element_type"] != 1].sort_values(ep_col, ascending=False)
    formation = "-".join(
        str((xi["element_type"] == et).sum()) for et in (2, 3, 4)
    )
    return {
        "xi": xi.sort_values(["element_type", ep_col], ascending=[True, False]),
        "captain": captain,
        "vice": vice,
        "bench_order": pd.concat([bench_gk, bench_out]),
        "formation": formation,
        "expected_points": round(float(xi[ep_col].sum() + captain[ep_col]), 2),
    }


# ------------------------------------------------------------ transfer plan
def plan_transfers(players: pd.DataFrame, current_ids: list[int],
                   selling_prices: dict[int, float], bank: float,
                   free_transfers: int, max_transfers: int = 3,
                   ep_col: str = "ep_horizon") -> dict:
    """Evaluate 0..max_transfers moves; returns the best plan net of hits.

    Solves one ILP per transfer count k with sum(out)=k, then compares
    objective - hit_cost across k. k <= free_transfers costs nothing.
    """
    pool = players[players["available"] & (players["xmins"] > 0.05)].copy()
    cur = players[players["id"].isin(current_ids)]
    pool = pd.concat([pool, cur[~cur.index.isin(pool.index)]])  # can keep unavailable

    cur_idx = set(pool.index[pool["id"].isin(current_ids)])
    sell = {i: selling_prices.get(int(pool.loc[i, "id"]), pool.loc[i, "price"])
            for i in cur_idx}
    budget = bank + sum(sell.values())

    base_ep = None
    plans = []
    for k in range(0, max_transfers + 1):
        x = {i: pl.LpVariable(f"sq_{i}", cat="Binary") for i in pool.index}
        s = {i: pl.LpVariable(f"xi_{i}", cat="Binary") for i in pool.index}
        c = {i: pl.LpVariable(f"cap_{i}", cat="Binary") for i in pool.index}
        prob = pl.LpProblem(f"fpl_transfer_{k}", pl.LpMaximize)
        prob += pl.lpSum(
            pool.loc[i, ep_col] * (s[i] + c[i] + config.BENCH_WEIGHT * (x[i] - s[i]))
            for i in pool.index
        )
        prob += pl.lpSum(x.values()) == config.SQUAD_SIZE
        prob += pl.lpSum(s.values()) == config.XI_SIZE
        prob += pl.lpSum(c.values()) == 1
        for i in pool.index:
            prob += c[i] <= s[i]
        # cost: buys at price, keeps at selling price
        prob += pl.lpSum(
            (sell[i] if i in cur_idx else pool.loc[i, "price"]) * x[i]
            for i in pool.index
        ) <= budget
        # exactly k players leave the current squad
        prob += pl.lpSum(x[i] for i in cur_idx) == len(cur_idx) - k
        for i in pool.index:
            prob += s[i] <= x[i]
        for et, (_, squad_n, xi_min, xi_max) in config.POSITIONS.items():
            idx = pool.index[pool["element_type"] == et]
            prob += pl.lpSum(x[i] for i in idx) == squad_n
            prob += pl.lpSum(s[i] for i in idx) >= xi_min
            prob += pl.lpSum(s[i] for i in idx) <= xi_max
        for t in pool["team"].unique():
            idx = pool.index[pool["team"] == t]
            prob += pl.lpSum(x[i] for i in idx) <= config.MAX_PER_CLUB
        try:
            _solve(prob)
        except RuntimeError:
            continue
        squad = _chosen(x, pool)
        hit = max(0, k - free_transfers) * config.TRANSFER_HIT
        obj = pl.value(prob.objective)
        if k == 0:
            base_ep = obj
        plans.append({
            "n_transfers": k,
            "hit_cost": hit,
            "objective": round(obj, 2),
            "net_gain_vs_hold": round(obj - hit - (base_ep or obj), 2),
            "out": cur[~cur["id"].isin(squad["id"])],
            "in": squad[~squad["id"].isin(current_ids)],
            "squad": squad,
        })
    best = max(plans, key=lambda p: p["net_gain_vs_hold"])
    return {"plans": plans, "best": best}
