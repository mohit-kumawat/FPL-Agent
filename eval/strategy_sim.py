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
from fpl_agent import backtest  # noqa: E402
from season_loop import run_season  # noqa: E402

OUT: list[str] = []


def log(s: str = "") -> None:
    print(s)
    OUT.append(s)


def simulate(season: str, max_gw: int = 38, transfers: bool = True) -> dict:
    """Run one season through the shared eval loop (eval/season_loop.py).

    transfers=False -> hold the GW1 squad all season. Reports both the legacy
    `raw` score (XI + doubled captain) and the autosub-aware total."""
    r = run_season(season, max_gw=max_gw, transfers=transfers)
    raw = sum(r["weekly_raw"])
    return {"total": r["total_raw"], "raw": round(raw), "hits": r["hits"],
            "moves": r["moves"], "weekly_mean": round(raw / max_gw, 1),
            "autosub_total": r["total_autosub"],
            "autosub_gain": round(sum(r["weekly_autosub"]) - raw),
            "n_subs": r["n_subs"], "n_vice": r["n_vice"]}


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
