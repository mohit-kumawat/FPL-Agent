"""The one season replay loop every eval arm runs through.

There used to be three near-verbatim copies of this (strategy_sim, agent_backtest,
ablation), and they had already drifted: different free-transfer caps, different
points at which hits were charged. Since the whole point of the eval suite is
that arms are comparable, the loop lives here once and the arms differ only by
the parameters below.

Scores every gameweek two ways:
  raw      XI + doubled captain (the historically published figure)
  autosub  real matchday rules via fpl_agent.scoring — automatic substitutions
           and the vice-captain fallback

`weekly_net` charges each hit in the gameweek it was taken, so per-gameweek
paired comparisons (the ablation bootstrap) see the cost where it happened;
season totals are identical either way.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import config, memoryio, optimizer, policy, replay, scoring  # noqa: E402


# The base EP frame for a (season, gw) is identical for every arm — rebuilding
# the snapshot and refitting the ridge once per arm made the ablation five times
# slower than it needs to be. Cached here rather than in replay so the leak
# audit (which monkeypatches replay internals) is never served a stale frame.
@lru_cache(maxsize=96)
def _ep_cached(season: str, gw: int) -> pd.DataFrame:
    return replay.expected_points_at(season, gw)


def ep_at(season: str, gw: int) -> pd.DataFrame:
    """Point-in-time EP frame, cached; callers get their own copy to mutate."""
    return _ep_cached(season, gw).copy()


def clear_caches() -> None:
    """Drop cached frames. Anything that monkeypatches replay internals (the
    leak audit truncates history on purpose) MUST call this on both sides of the
    patch: serving a pre-patch frame to post-patch code would make a real
    hindsight leak look clean, which is the one failure this suite exists to
    prevent."""
    _ep_cached.cache_clear()
    replay._season_gws_cached.cache_clear()


def run_season(season: str, *, max_gw: int = 38, transfers: bool = True,
               transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
               use_policy: bool = True, max_transfers: int = 2) -> dict:
    """Play one season point-in-time. Returns per-gameweek series plus totals.

    transform:  optional per-gameweek EP rewrite (the ablation arms) — applied
                to every frame including the GW1 build, so an arm never mixes
                its own EP with production EP.
    use_policy: False takes any positive-net-gain plan (threshold-free greedy),
                which is how the ablation isolates what the policy gate earns.
    """
    def arm_ep(gw: int) -> pd.DataFrame:
        ep = ep_at(season, gw)
        return transform(ep) if transform is not None else ep

    b = optimizer.build_squad(arm_ep(1))
    squad_ids = list(b["squad"]["id"])
    purchase = dict(zip(b["squad"]["id"], b["squad"]["price"]))
    bank = round(config.BUDGET - round(float(b["squad"]["price"].sum()), 1), 1)
    fts, hits, moves, n_subs, n_vice = 1, 0, 0, 0, 0
    weekly_raw: list[float] = []
    weekly_autosub: list[float] = []
    weekly_net: list[float] = []
    squads: list[pd.DataFrame] = []

    for gw in range(1, max_gw + 1):
        ep = arm_ep(gw)
        act = replay.actual_points(season, gw)
        mine = ep[ep["id"].isin(squad_ids)]
        hit_now = 0

        if transfers and gw > 1 and len(mine) == config.SQUAD_SIZE:
            sell = memoryio.squad_selling_prices(
                {"players": [{"id": pid, "purchase_price": purchase[pid]}
                             for pid in squad_ids]}, ep)
            try:
                plan = optimizer.plan_transfers(ep, squad_ids, sell, bank=bank,
                                                free_transfers=fts,
                                                max_transfers=max_transfers)
                if use_policy:
                    dec = policy.assess_transfers(plan, fts, gws_played=gw - 1)
                else:
                    best = plan["best"]
                    dec = ({"action": f"{best['n_transfers']}_transfers", "plan": best}
                           if best["n_transfers"] > 0 and best["net_gain_vs_hold"] > 0
                           else {"action": "hold", "plan": None})
            except Exception:  # noqa: BLE001 — infeasible week: hold
                dec = {"action": "hold", "plan": None}

            if dec["action"] != "hold" and dec["plan"] is not None:
                p = dec["plan"]
                out_ids = list(p["out"]["id"])
                bank = round(bank + sum(sell[i] for i in out_ids)
                             - float(p["in"]["price"].sum()), 1)
                for i in out_ids:
                    squad_ids.remove(i)
                    purchase.pop(i, None)
                for r in p["in"].itertuples():
                    squad_ids.append(int(r.id))
                    purchase[int(r.id)] = float(r.price)
                k = p["n_transfers"]
                moves += k
                hit_now = max(0, k - fts) * config.TRANSFER_HIT
                hits += hit_now
                fts = min(config.MAX_FREE_TRANSFERS, max(0, fts - k) + 1)
            else:
                fts = min(config.MAX_FREE_TRANSFERS, fts + 1)
            mine = ep[ep["id"].isin(squad_ids)]

        if len(mine) == config.SQUAD_SIZE:
            xi = optimizer.pick_xi(mine)
            s = scoring.gw_score(xi, replay.actual_minutes(season, gw), act)
            raw, sub = s["raw"], s["autosub"]
            n_subs += s["subs"]
            n_vice += s["vice_used"]
        else:
            raw = sub = 0.0
        weekly_raw.append(raw)
        weekly_autosub.append(sub)
        weekly_net.append(sub - hit_now)
        squads.append(mine)

    return {
        "weekly_raw": weekly_raw,
        "weekly_autosub": weekly_autosub,
        "weekly_net": weekly_net,
        "squads": squads,
        "hits": hits, "moves": moves, "n_subs": n_subs, "n_vice": n_vice,
        "total_raw": round(sum(weekly_raw) - hits),
        "total_autosub": round(sum(weekly_autosub) - hits),
    }
