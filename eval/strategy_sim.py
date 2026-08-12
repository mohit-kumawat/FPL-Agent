"""Full-season strategy simulation: what would this exact system have scored?

Plays entire historical seasons GW-by-GW with point-in-time data only:
GW1 build -> weekly transfer decisions through the real policy layer ->
XI + captain -> scored against actual points. Compared against baselines.

Also runs the lambda-grid robustness check across two seasons.

Usage: uv run python eval/strategy_sim.py [season ...]   (default: 2024-25 2023-24)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import backtest, config, memoryio, optimizer, policy, replay, scoring  # noqa: E402

OUT: list[str] = []


def log(s: str = "") -> None:
    print(s)
    OUT.append(s)


def simulate(season: str, max_gw: int = 38, transfers: bool = True) -> dict:
    """Run one season. transfers=False -> hold the GW1 squad all season.

    Scores every GW two ways: `raw` (XI + doubled captain, the historically
    published number) and `autosub` (real matchday rules: automatic subs +
    vice-captain fallback, via fpl_agent.scoring)."""
    b = replay.build_at(season, 1)
    squad_ids = list(b["squad"]["id"])
    purchase = dict(zip(b["squad"]["id"], b["squad"]["price"]))
    bank = round(config.BUDGET - b["cost"], 1)
    fts, hits, moves = 1, 0, 0
    total = 0.0
    total_autosub = 0.0
    n_subs, n_vice = 0, 0
    weekly: list[float] = []

    for gw in range(1, max_gw + 1):
        ep = replay.expected_points_at(season, gw)
        act = replay.actual_points(season, gw)
        mine = ep[ep["id"].isin(squad_ids)]

        if transfers and gw > 1 and len(mine) == config.SQUAD_SIZE:
            # use the production rule, never a local copy of it
            sell = memoryio.squad_selling_prices(
                {"players": [{"id": pid, "purchase_price": purchase[pid]}
                             for pid in squad_ids]},
                ep)
            try:
                plan = optimizer.plan_transfers(ep, squad_ids, sell, bank=bank,
                                                free_transfers=fts, max_transfers=2)
                dec = policy.assess_transfers(plan, fts, gws_played=gw - 1)
            except Exception:  # noqa: BLE001 — infeasible week: hold
                dec = {"action": "hold", "plan": None}
            if dec["action"] != "hold" and dec["plan"] is not None:
                p = dec["plan"]
                out_ids = list(p["out"]["id"])
                in_rows = p["in"]
                spend = float(in_rows["price"].sum())
                recoup = sum(sell[i] for i in out_ids)
                bank = round(bank + recoup - spend, 1)
                for i in out_ids:
                    squad_ids.remove(i)
                    purchase.pop(i, None)
                for r in in_rows.itertuples():
                    squad_ids.append(int(r.id))
                    purchase[int(r.id)] = float(r.price)
                k = p["n_transfers"]
                moves += k
                hits += max(0, k - fts) * config.TRANSFER_HIT
                fts = min(5, max(0, fts - k) + 1)
            else:
                fts = min(5, fts + 1)
            mine = ep[ep["id"].isin(squad_ids)]

        if len(mine) == config.SQUAD_SIZE:
            xi = optimizer.pick_xi(mine)
            mins = replay.actual_minutes(season, gw)
            s = scoring.gw_score(xi, mins, act)
            pts, pts_sub = s["raw"], s["autosub"]
            n_subs += s["subs"]
            n_vice += s["vice_used"]
        else:
            pts = pts_sub = 0.0
        weekly.append(pts)
        total += pts
        total_autosub += pts_sub

    return {"total": round(total - hits), "raw": round(total), "hits": hits,
            "moves": moves, "weekly_mean": round(total / max_gw, 1),
            "autosub_total": round(total_autosub - hits),
            "autosub_gain": round(total_autosub - total),
            "n_subs": n_subs, "n_vice": n_vice}


def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2023-24"]

    log("# Strategy-return simulation (point-in-time, no hindsight)\n")
    for season in seasons:
        log(f"## {season}")
        full = simulate(season, transfers=True)
        hold = simulate(season, transfers=False)
        log(f"- **agent strategy**: {full['total']} pts "
            f"({full['moves']} transfers, {full['hits']} hit pts, "
            f"{full['weekly_mean']} raw pts/GW)")
        log(f"- **with real matchday rules (autosubs + vice)**: "
            f"{full['autosub_total']} pts ({full['autosub_gain']:+d} from "
            f"{full['n_subs']} autosubs, vice used {full['n_vice']}x)")
        log(f"- hold-GW1-squad baseline: {hold['total']} pts "
            f"(autosub-aware {hold['autosub_total']})")
        log(f"- transfer engine added: {full['total'] - hold['total']:+d} pts "
            f"(autosub-aware {full['autosub_total'] - hold['autosub_total']:+d})")
        log("- note: no chips in either arm; real managers gain ~30-60 "
            "pts/season from chips on top")
        log("")

    log("## Lambda-grid robustness (Model A recency vs Model B ridge)")
    for season, train in (("2024-25", 26), ("2025-26", 26)):
        log(f"\n### {season} (test GW{train + 1}-38)")
        res = backtest.run(train_gws=train, season=season)
        if not res.empty:
            cols = [c for c in res.columns if c.startswith("top11_")]
            means = res[cols].mean().sort_values(ascending=False).round(3)
            log("top-11 actual points by weight (higher=better):")
            log(means.to_string())

    Path(__file__).parent.joinpath("strategy-sim-report.md").write_text("\n".join(OUT) + "\n")
    print("\nreport -> eval/strategy-sim-report.md")


if __name__ == "__main__":
    main()
