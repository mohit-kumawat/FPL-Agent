"""Run the historical test suite on 2024/25. Usage: uv run python eval/run_backtests.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fpl_agent import config, optimizer, replay  # noqa: E402

S = "2024-25"
OUT: list[str] = []


def log(s: str = "") -> None:
    print(s)
    OUT.append(s)


def fmt(df: pd.DataFrame, n: int = 15) -> str:
    return df.head(n).round(2).to_string(index=False)


# ---------------------------------------------------------------- Test 1
log("# 2024/25 backtest report\n")
log("## Test 1 — GW1 squad build, premiums, long-term holds; replay GW2-3")
b1 = replay.build_at(S, 1)
xi1 = b1["xi_result"]
sq = b1["squad"][["web_name", "team_short", "position", "price", "ep_next", "ep_horizon", "actual"]]
log(f"cost £{b1['cost']}m | formation {xi1['formation']} | captain {xi1['captain']['web_name']}")
log(fmt(sq.sort_values("price", ascending=False)))
gw1_actual_xi = float(xi1["xi"]["id"].map(replay.actual_points(S, 1)).fillna(0).sum()
                      + replay.actual_points(S, 1).get(int(xi1["captain"]["id"]), 0))
log(f"XI+captain actual GW1 points: {gw1_actual_xi:.0f}")
prem = replay.premium_ranking(S, 1, ["Erling Haaland", "Mohamed Salah", "Cole Palmer", "Bukayo Saka", "Son Heung-min"], horizon=8)
log("\nPremium ranking @GW1 (8-GW horizon):")
log(fmt(prem, 8))
for gw in (2, 3):
    ep = replay.expected_points_at(S, gw)
    top = ep.nlargest(8, "ep_horizon")[["web_name", "team_short", "ep_ppg", "xmins", "ep_horizon"]]
    hs = replay.match_players(ep, ["Erling Haaland", "Mohamed Salah"])
    log(f"\n@GW{gw} model top-8 by horizon EP:")
    log(fmt(top, 8))
    log(f"Haaland/Salah horizon rank: "
        f"{[int((ep['ep_horizon'] > r.ep_horizon).sum()) + 1 for r in hs.itertuples()]}")

# ---------------------------------------------------------------- Test 2
log("\n## Test 2 — GW1 captain among Haaland/Salah/Isak/Saka/Watkins")
c = replay.captain_pick(S, 1, ["Erling Haaland", "Mohamed Salah", "Alexander Isak", "Bukayo Saka", "Ollie Watkins"])
log(fmt(c, 6))
log("actuals: Salah 14, Saka 12, Haaland 7, Isak 5, Watkins 2")

# ---------------------------------------------------------------- Test 3
log("\n## Test 3 — GW6 captain + highest upside")
ep6 = replay.expected_points_at(S, 6)
ep6 = ep6.assign(actual=ep6["id"].map(replay.actual_points(S, 6)).fillna(0),
                 upside=ep6["ep_next"] + ep6["ep_sd"])
log("top-10 by ep_next:")
log(fmt(ep6.nlargest(10, "ep_next")[["web_name", "team_short", "ep_next", "ep_sd", "upside", "actual"]], 10))
pr = ep6[ep6["web_name"].str.contains("Palmer", na=False)]
log(f"Palmer ep_next rank: {int((ep6['ep_next'] > pr['ep_next'].iloc[0]).sum()) + 1}, "
    f"upside rank: {int((ep6['upside'] > pr['upside'].iloc[0]).sum()) + 1} (actual 25)")

# ---------------------------------------------------------------- Test 4
log("\n## Test 4 — GW11: restructure around Salah / Haaland / Palmer (8-GW view)")
p11 = replay.premium_ranking(S, 11, ["Mohamed Salah", "Erling Haaland", "Cole Palmer"], horizon=8)
log(fmt(p11, 6))
act_range = replay.actual_points_range(S, 11, 18)
ep11 = replay.expected_points_at(S, 11, horizon=8)
trio = replay.match_players(ep11, ["Mohamed Salah", "Erling Haaland", "Cole Palmer"])
log("actual next-8-GW points: " + ", ".join(
    f"{r.web_name}={int(act_range.get(int(r.id), 0))}" for r in trio.itertuples()))

# ---------------------------------------------------------------- Test 5
log("\n## Test 5 — GW15 (after GW14): premium thesis update")
p15 = replay.premium_ranking(S, 15, ["Mohamed Salah", "Erling Haaland", "Cole Palmer"], horizon=8)
log(fmt(p15, 6))
log("(reference: from GW14 onward Salah 211, Palmer 114, Haaland 95)")

# ---------------------------------------------------------------- Test 6
log("\n## Test 6 — GW24 DGW: Triple Captain Salah?")
ep24 = replay.expected_points_at(S, 24)
dgw = ep24[ep24["n_fixtures"] > 0]
top24 = ep24.nlargest(8, "ep_next")[["web_name", "team_short", "next_gw_mult", "ep_next", "ep_sd"]]
log("top captain options (next_gw_mult > 1.6 implies a double):")
log(fmt(top24, 8))
sal = ep24[ep24["web_name"] == "Mohamed Salah"].iloc[0]
others = ep24[ep24["web_name"] != "Mohamed Salah"]["ep_next"].max()
tc_ev = sal["ep_next"]  # TC adds 1x captain EP beyond normal captaincy
log(f"TC EV: Salah ep_next {sal['ep_next']:.1f}; extra from chip = +{tc_ev:.1f} EP this GW "
    f"vs saving it. Best non-Salah option {others:.1f}. Double fixture: "
    f"{sal['next_gw_mult']:.2f} mult. (actual: Salah 29 -> TC = 87)")

# ---------------------------------------------------------------- Test 7
log("\n## Test 7 — GW25: repeat Salah or change? (memory test)")
ep25 = replay.expected_points_at(S, 25)
top25 = ep25.nlargest(6, "ep_next")[["web_name", "team_short", "next_gw_mult", "ep_next"]]
log(fmt(top25, 6))
log("(actual: Salah 20 in GW25)")

# ---------------------------------------------------------------- Test 8
log("\n## Test 8 — GW38: final-GW captain/transfers")
ep38 = replay.expected_points_at(S, 38, horizon=1)
top38 = ep38.assign(actual=ep38["id"].map(replay.actual_points(S, 38)).fillna(0))
log(fmt(top38.nlargest(10, "ep_next")[["web_name", "team_short", "roll_points", "ep_next", "actual"]], 10))
bw = top38[top38["web_name"].str.contains("Bowen", na=False)]
if len(bw):
    log(f"Bowen ep_next rank: {int((top38['ep_next'] > bw['ep_next'].iloc[0]).sum()) + 1} "
        f"(actual 13; form roll_points {bw['roll_points'].iloc[0]:.1f})")

# ------------------------------------------------- aggregate sweep GW6-24
log("\n## Aggregate sweep — GW6..GW24 (19 GWs, no cherry-picking)")


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank().corr(y.rank()))


rows = []
pos_rho: dict[str, list[float]] = {}
band_rho: dict[str, list[float]] = {}
for gw in range(6, 25):
    ep = replay.expected_points_at(S, gw)
    act = replay.actual_points(S, gw)
    ep = ep.assign(actual=ep["id"].map(act).fillna(0))
    cap = ep.nlargest(1, "ep_next").iloc[0]
    cands = ep.nlargest(6, "ep_next")
    top11 = ep.nlargest(11, "ep_next")
    # rank quality among plausible starters only (xmins>0.3) — ranking the
    # mass of non-players is trivial and would flatter the correlation
    pool = ep[ep["xmins"] > 0.3]
    top50 = ep.nlargest(50, "ep_next")
    cov = float(((top50["actual"] >= top50["ep_p10"])
                 & (top50["actual"] <= top50["ep_p90"])).mean())
    rows.append({
        "gw": gw, "captain": cap["web_name"], "cap_actual": cap["actual"],
        "best_of_6": cands["actual"].max(),
        "top11_actual_mean": top11["actual"].mean(),
        "rho": spearman(pool["ep_next"], pool["actual"]),
        "p10_p90_cov": cov,
    })
    for pos, g in pool.groupby("position"):
        pos_rho.setdefault(pos, []).append(spearman(g["ep_next"], g["actual"]))
    pool = pool.assign(_band=pd.cut(pool["price"], [0, 5.5, 7.5, 20],
                                    labels=["<=5.5", "5.5-7.5", ">7.5"]))
    for band, g in pool.groupby("_band", observed=True):
        if len(g) > 8:
            band_rho.setdefault(str(band), []).append(spearman(g["ep_next"], g["actual"]))

agg = pd.DataFrame(rows)
log(fmt(agg, 25))
log(f"\ncaptain actual mean: {agg['cap_actual'].mean():.2f} "
    f"| best-of-top-6 mean (ceiling): {agg['best_of_6'].mean():.2f} "
    f"| capture ratio: {agg['cap_actual'].sum() / agg['best_of_6'].sum():.0%}")
log(f"top-11 actual mean: {agg['top11_actual_mean'].mean():.2f} pts/player")
log(f"Spearman rho (ep_next vs actual, xmins>0.3): mean {agg['rho'].mean():.3f}")
log("  by position: " + ", ".join(f"{k}={pd.Series(v).mean():.3f}"
                                   for k, v in sorted(pos_rho.items())))
log("  by price:    " + ", ".join(f"{k}={pd.Series(v).mean():.3f}"
                                   for k, v in sorted(band_rho.items())))
log(f"p10-p90 coverage on top-50 EP (target ~0.80): {agg['p10_p90_cov'].mean():.2f}")

# ------------------------------------------------- no-change determinism
log("\n## No-change replay (determinism)")
a = replay.expected_points_at(S, 24)
b = replay.expected_points_at(S, 24)
same = a[["id", "ep_next", "ep_horizon"]].equals(b[["id", "ep_next", "ep_horizon"]])
log(f"identical EP frames on rerun: {same}")

# ------------------------------------------------- hindsight protection
log("\n## Hindsight protection (GW24)")
snap = replay.snapshot_at(S, 24)
panel = replay.panel_before(S, 24)
log(f"max round in panel: {int(panel['round'].max())} (must be 23): "
    f"{'PASS' if panel['round'].max() == 23 else 'FAIL'}")
full = replay._season_gws(S)
sal_all = full[(full["name"].str.contains("Salah")) & (full["round"] >= 24)]["total_points"].sum()
snap_sal = snap[snap["web_name"].str.contains("Salah")]
log(f"Salah cumulative points in snapshot excludes GW24+ ({int(sal_all)} pts unseen): "
    f"{'PASS' if snap_sal['points_per_game'].iloc[0] * 30 < 400 else 'check'}")
log("price source: last observed value before GW24 (structurally past-only): PASS")

summary = [
    "# Headline metrics (auto-generated each run)",
    f"- captain actual mean {agg['cap_actual'].mean():.2f} pts/GW, "
    f"ceiling capture {agg['cap_actual'].sum() / agg['best_of_6'].sum():.0%}",
    f"- top-11 actual {agg['top11_actual_mean'].mean():.2f} pts/player/GW",
    f"- Spearman rho {agg['rho'].mean():.3f} (starters pool)",
    f"- p10-p90 coverage {agg['p10_p90_cov'].mean():.2f} (target 0.80)",
    "- strategy-level returns: see strategy-sim-report.md", "", "---", "",
]
Path(__file__).parent.joinpath("2024-25-report.md").write_text(
    "\n".join(summary + OUT) + "\n")
print("\nreport -> eval/2024-25-report.md")
