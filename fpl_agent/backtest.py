"""Backtest the in-season models on the prior season (2025/26).

Walk-forward: for each test GW, fit on everything before it, predict, and
score MAE + a top-picks hit metric. Compares Model A, Model B, and the
ensemble so ENSEMBLE_LAMBDA is chosen from evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data, models


def _prep(season: str) -> pd.DataFrame:
    df = data.load_prior_season(season)
    if "round" not in df.columns and "GW" in df.columns:
        df = df.rename(columns={"GW": "round"})
    if "element" not in df.columns and "id" in df.columns:
        df["element"] = df["id"]
    for c in ["expected_goal_involvements", "expected_goals_conceded",
              "ict_index", "minutes", "total_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def run(train_gws: int = 26, season: str = config.PRIOR_SEASON) -> pd.DataFrame:
    df = _prep(season)
    max_gw = int(df["round"].max())
    results = []
    for gw in range(train_gws + 1, max_gw + 1):
        hist = df[df["round"] < gw]
        actual = df[df["round"] == gw][["element", "total_points"]]
        if actual.empty:
            continue
        a = models.model_a_recency(hist)
        b = models.model_b_ridge_inseason(hist)
        both = pd.DataFrame({"a": a, "b": b}).dropna()
        preds = {"A_recency": a, "B_ridge": b}
        for lam in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            preds[f"ens_{lam}"] = (1 - lam) * both["a"] + lam * both["b"]
        row: dict = {"gw": gw, "n": len(actual)}
        for name, p in preds.items():
            m = actual.merge(p.rename("pred"), left_on="element", right_index=True)
            if m.empty:
                continue
            row[f"mae_{name}"] = float((m["pred"] - m["total_points"]).abs().mean())
            top = m.nlargest(11, "pred")
            row[f"top11_{name}"] = float(top["total_points"].mean())
        results.append(row)
    res = pd.DataFrame(results)
    if res.empty:
        print("no backtest rows — check season data")
        return res
    summary = res.drop(columns=["gw", "n"]).mean().round(3)
    print(f"Backtest {season}, test GW{train_gws + 1}-{max_gw} "
          f"({len(res)} GWs, ~{int(res['n'].mean())} players/GW)\n")
    print("mean MAE (lower better) and mean actual points of model's top-11 (higher better):")
    print(summary.to_string())
    out = config.DATA_DIR / f"backtest_{season}.csv"
    res.to_csv(out, index=False)
    print(f"\nper-GW detail -> {out}")
    return res
