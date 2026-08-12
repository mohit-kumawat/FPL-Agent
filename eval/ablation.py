"""Ablation ladder: which part of the model actually earns the points?

Runs the SAME season loop (GW1 build -> weekly policy-gated transfers -> XI +
captain -> autosub-aware scoring) once per arm, where each arm sees a different
expected-points column. Every arm shares the replay harness, the optimizer, the
policy layer, and the scorer, so the only difference between adjacent rungs is
the model ingredient being measured:

    ppg          points-per-game only (flat horizon)
    +fixtures    ppg x fixture multipliers
    +minutes     ppg x xmins x fixture multipliers   (the legacy realized path)
    full         the component-blended production model (W_COMPONENT)
    full-greedy  full model, thresholds off: any positive net gain is taken
                 (isolates what the policy gate is worth vs always-churn)

Season totals on one season are noise (a single different transfer compounds),
so the report leans on paired per-GW differences with a bootstrap CI over
gameweeks. Treat a rung as "earning its keep" only when the CI excludes zero
on both seasons — anything else is "consistent with noise", stated as such.

Usage: uv run python eval/ablation.py [season ...]   (default: 2024-25 2023-24)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import config, memoryio, optimizer, policy, replay, scoring  # noqa: E402

SEED = 20260812
N_BOOT = 5000
OUT: list[str] = []


def log(s: str = "") -> None:
    print(s)
    OUT.append(s)


# --------------------------------------------------------------------- arms
def _arm_ppg(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.copy()
    out["ep_next"] = out["ep_ppg"].clip(lower=0)
    out["ep_horizon"] = out["ep_ppg"].clip(lower=0) * config.HORIZON_GWS
    return out


def _arm_fixtures(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.copy()
    out["ep_next"] = (out["ep_ppg"] * out["next_gw_mult"]).clip(lower=0)
    out["ep_horizon"] = (out["ep_ppg"] * out["ep_mult"]).clip(lower=0)
    return out


def _arm_minutes(ep: pd.DataFrame) -> pd.DataFrame:
    out = ep.copy()
    out["ep_next"] = (out["ep_ppg"] * out["xmins"] * out["next_gw_mult"]).clip(lower=0)
    out["ep_horizon"] = (out["ep_ppg"] * out["xmins"] * out["ep_mult"]).clip(lower=0)
    return out


def _arm_full(ep: pd.DataFrame) -> pd.DataFrame:
    return ep


ARMS: dict[str, tuple] = {
    # name -> (ep transform, use policy thresholds)
    "ppg": (_arm_ppg, True),
    "+fixtures": (_arm_fixtures, True),
    "+minutes": (_arm_minutes, True),
    "full": (_arm_full, True),
    "full-greedy": (_arm_full, False),
}


# -------------------------------------------------------------- season loop
def run_arm(season: str, transform, use_policy: bool, max_gw: int = 38) -> dict:
    """One arm through one season; per-GW autosub-aware actual points."""
    ep1 = transform(replay.expected_points_at(season, 1))
    b = optimizer.build_squad(ep1)
    squad_ids = list(b["squad"]["id"])
    purchase = dict(zip(b["squad"]["id"], b["squad"]["price"]))
    bank = round(config.BUDGET - b["cost"], 1)
    fts, hits, moves = 1, 0, 0
    weekly: list[float] = []

    for gw in range(1, max_gw + 1):
        ep = transform(replay.expected_points_at(season, gw))
        act = replay.actual_points(season, gw)
        mine = ep[ep["id"].isin(squad_ids)]
        hit_now = 0

        if gw > 1 and len(mine) == config.SQUAD_SIZE:
            sell = memoryio.squad_selling_prices(
                {"players": [{"id": p, "purchase_price": purchase[p]} for p in squad_ids]}, ep)
            try:
                plan = optimizer.plan_transfers(ep, squad_ids, sell, bank=bank,
                                                free_transfers=fts, max_transfers=2)
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
                    squad_ids.remove(i); purchase.pop(i, None)
                for r in p["in"].itertuples():
                    squad_ids.append(int(r.id)); purchase[int(r.id)] = float(r.price)
                moves += p["n_transfers"]
                hit_now = max(0, p["n_transfers"] - fts) * config.TRANSFER_HIT
                hits += hit_now
                fts = min(config.MAX_FREE_TRANSFERS, max(0, fts - p["n_transfers"]) + 1)
            else:
                fts = min(config.MAX_FREE_TRANSFERS, fts + 1)
            mine = ep[ep["id"].isin(squad_ids)]

        if len(mine) == config.SQUAD_SIZE:
            s = scoring.gw_score(optimizer.pick_xi(mine),
                                 replay.actual_minutes(season, gw), act)
            pts = s["autosub"]
        else:
            pts = 0.0
        # hits are charged in the GW they were taken so per-GW pairing sees them
        weekly.append(pts - hit_now)

    return {"weekly": weekly, "hits": hits, "moves": moves,
            "total": round(sum(weekly))}


# ---------------------------------------------------------------- statistics
def paired_diff_ci(a: list[float], b: list[float],
                   n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean per-GW difference (b - a) with a bootstrap 90% CI over gameweeks."""
    rng = random.Random(SEED)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    )
    return (sum(diffs) / n, means[int(0.05 * n_boot)], means[int(0.95 * n_boot)])


# --------------------------------------------------------------------- main
def main() -> None:
    seasons = sys.argv[1:] or ["2024-25", "2023-24"]
    log("# Ablation ladder (autosub-aware scoring, policy-gated transfers)\n")
    log(f"Seed {SEED}; {N_BOOT} bootstrap resamples; CIs are 90% over per-GW "
        "paired differences. Season totals are noisy — the CI is the evidence.\n")

    for season in seasons:
        log(f"## {season}\n")
        results: dict[str, dict] = {}
        for name, (transform, use_policy) in ARMS.items():
            results[name] = run_arm(season, transform, use_policy)
            r = results[name]
            log(f"- `{name}`: **{r['total']}** pts ({r['moves']} moves, "
                f"{r['hits']} hit pts)")
        log("")
        log("| rung | Δtotal | per-GW Δ | 90% CI | verdict |")
        log("|---|---|---|---|---|")
        ladder = list(ARMS)
        # adjacent rungs decompose the edge; the ppg -> full row is the whole
        # model's edge, which can be significant even when every rung is noisy
        pairs = list(zip(ladder, ladder[1:])) + [("ppg", "full")]
        for lo, hi in pairs:
            a, b = results[lo], results[hi]
            mean, ci_lo, ci_hi = paired_diff_ci(a["weekly"], b["weekly"])
            dtot = b["total"] - a["total"]   # weekly series already carry hits
            verdict = ("earns its keep" if ci_lo > 0
                       else "hurts" if ci_hi < 0 else "consistent with noise")
            log(f"| {lo} → {hi} | {dtot:+d} | {mean:+.2f} | "
                f"[{ci_lo:+.2f}, {ci_hi:+.2f}] | {verdict} |")
        log("")

    dest = Path(__file__).with_name("ablation-report.md")
    dest.write_text("\n".join(OUT) + "\n")
    print(f"\nreport -> {dest.name}")


if __name__ == "__main__":
    main()
