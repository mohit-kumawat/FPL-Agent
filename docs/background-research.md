# Background research

Provenance: this is the original literature review that seeded the project
(generated with Perplexity, lightly edited). It is kept for the citations —
the arXiv integer-programming paper, the Uppsala ML thesis, and the public FPL
data repos — not as a description of the current system.

**The implementation has since moved well past this note.** For what the code
actually does, read the main [README](../README.md). Notably: expected points
are component-based rather than PPG-based, fixture difficulty uses continuous
team-strength channels rather than the FDR table described here, and the
evaluation is point-in-time replay rather than a train/test split.

---
# tell me about FPL teams who were purley did well just based on data an agent, I am trying to build something similar here.

Great question—and very timely, since there are now several public examples of FPL teams and agents that performed well using purely data-driven approaches. Here are the most relevant ones you can learn from, plus the patterns that made them successful.[^1_1][^1_2][^1_3][^1_4]

## Standout data-driven FPL teams/agents

### 1) Uppsala University AI Assistant Manager (2024/25 season)

- **Approach:** Built an AI assistant that predicts player points using **linear/logistic regression and XGBoost**, then feeds those predictions into a **squad optimizer** that respects FPL constraints (budget, formation, 3-per-club).[^1_2]
- **Result:** In a simulation from GW1–21 of 2024/25, the **linear models** produced **1,293 points**, which would have placed the algorithm in the **top 12%** of all FPL managers globally.[^1_2]
- **Why it worked:**
    - Simple, well-regularized models (linear) matched or beat more complex ones on out-of-sample points.
    - Clear separation between **prediction** (points models) and **optimization** (lineup selection under constraints).
    - Added an **LLM explanation layer** so recommendations were interpretable (helps trust and iteration).[^1_2]


### 2) “Data-Driven Framework” (Ramezani \& Dinh, 2023/24 season)

- **Approach:** Formulated FPL lineup selection as a **data-driven mixed-integer linear program** that jointly picks the **starting XI, bench, and captain** under budget, formation, and club-quota constraints.[^1_1]
- **Prediction methods benchmarked:** simple/weighted averages, exponential smoothing, **ARIMA**, Monte Carlo, and a **hybrid score** (ridge regression on ICT/xG/xA/starts/etc. blended with recent points).[^1_1]
- **Result:** On 2023/24 data (train GW1–26, test GW27–38), **rolling-window ARIMA** and **recency-weighted averages** were the most consistently strong out-of-sample. Hybrid metrics helped in some objectives but weren’t uniformly superior.[^1_1]
- **Why it worked:**
    - **Recency matters:** Weighted averages and low-order ARIMA adapted better to form and minutes changes.
    - **Joint optimization:** Captaincy was optimized inside the ILP (not a separate heuristic), which is often where big points are won or lost.[^1_1]
    - Robust variants (hedging against overestimation) trimmed variance but didn’t always increase total points—useful if you care about rank stability.[^1_1]


### 3) Medium “AI agent in minutes” workflow (practical agentic pattern)

- **Approach:** A browser-based AI agent was instructed to: **analyze the user’s squad, check expert sources (e.g., The Scout), review injuries/price changes, and optimize transfers, captain, and formation** under FPL rules.[^1_4]
- **Why it’s relevant:** It shows an **agentic pattern** you can replicate: rule-check → data gather → constraint-aware optimization → actionable output (transfers/captain/formation).[^1_4]


## Common patterns behind “purely data-driven” success

If you’re building something similar, these are the levers that repeatedly show up in strong results:

- **Prediction → Optimization pipeline**
    - Predict expected points per player for the upcoming gameweek(s).
    - Feed those expectations into an **integer programming** or equivalent optimizer that enforces: £100m budget, 2–5–5–3 squad, valid formations, max 3 per club, captain ∈ XI.[^1_3][^1_2][^1_1]
- **Simple, recency-aware models often win**
    - **Recency-weighted averages** and **low-order ARIMA** consistently outperformed simple averages and complex simulations in out-of-sample tests.[^1_1]
    - Linear models (with regularization) performed as well as XGBoost on fantasy points in at least one thesis, while being easier to interpret.[^1_2]
- **Features that actually move points**
    - Across positions, **ICT Index**, **starts/minutes proxies**, and attacking metrics (**xG, xA, xGI**) correlate most with points; **xGC** matters negatively for defenders/GKs.[^1_1]
    - For attackers, **xGI** (goal involvements) is a strong, parsimonious feature; for GK/DEF, **starts + xGC** are key.[^1_1]
- **Captaincy optimization is critical**
    - Treating captaincy as a decision variable in the optimizer (doubling the chosen player’s expected points) is a major edge versus picking captain by heuristics.[^1_1]
- **Budget strategy**
    - Strong managers don’t force rigid formations or budget splits; they **follow value** and let the optimizer choose formations that maximize expected output.[^1_5][^1_3]
    - Leaving ~£16–16.5m for the bench (four cheap reserves) is a common practical constraint to avoid budget violations while optimizing the XI.[^1_1]
- **Form > position (in-season)**
    - Retrospective season reviews highlight that **short-term form windows (3–5 GWs)** and **differential ownership** often beat “premium-only” strategies.[^1_6]
    - Tracking **ownership vs top-10k** and hunting underowned, high-output players is a repeatable edge.[^1_6]


## A minimal blueprint you can replicate

If you want a clean, reproducible agent:

1) **Data layer**
    - Use the public FPL data (player–gameweek panel) plus advanced stats (ICT, xG, xA, xGI, xGC, starts, selected%).[^1_7][^1_8][^1_1]
    - Engineer recency features (last 3–5 GW rolling means), fixture difficulty (FDR), and minutes proxies.[^1_8][^1_9]
2) **Prediction models**
    - Start with:
        - **Recency-weighted average points** per player.
        - **ARIMA(0,0,1) or ARIMA(1,0,1)** on total points with a rolling window.[^1_1]
        - Optionally a **ridge regression** on ICT/xGI/starts/xGC as a hybrid score.[^1_1]
    - Validate with a train/test split (e.g., GW1–26 → GW27–38) and track cumulative out-of-sample points.[^1_2][^1_1]
3) **Optimizer**
    - Implement a **0–1 integer program**:
        - Maximize ∑(expected_points_j × x_j) + ∑(expected_points_j × y_j) for captain y_j.
        - Constraints: 11 starters, budget ≤ chosen XI budget (e.g., £83.5), formation min/max per position, ≤3 per club, captain ∈ XI.[^1_1]
    - Add a bench optimizer to fill 2–5–5–3 with remaining budget and club quotas.[^1_1]
4) **Agentic loop (weekly)**
    - Before each deadline:
        - Update injuries/suspensions/price changes.
        - Re-run predictions and optimizer.
        - Output: transfers (respecting free transfer rules), optimal formation, and captain.[^1_9][^1_4]
5) **Evaluation**
    - Track **cumulative points**, **rank percentile**, and **captain hit rate** vs a baseline (e.g., template or simple average model).[^1_9][^1_2]
    - Ablate: compare weighted avg vs ARIMA vs hybrid; measure stability across gameweeks.[^1_1]

## Useful references and code bases to study

- **ArXiv paper + GitHub:** “Data-Driven Team Selection in FPL Using Integer Programming” (Ramezani \& Dinh) — includes datasets, preprocessing, and experiment notebooks.[^1_1]
- **Uppsala thesis:** “Enhancing FPL Strategies through ML and LLMs” — end-to-end assistant with squad optimization and explainability.[^1_2]
- **Tools using similar ideas:** SmartPlay, FPLai, fpl.team, Fantasy Football Fix (algorithm teams) — good for UX patterns and feature ideas (captain picks, transfer recs, squad ratings).[^1_10][^1_11][^1_7][^1_8]

If you tell me your tech stack (e.g., Python + PuLP/OR-Tools, or a no-code agent builder), I can sketch a concrete weekly agent workflow and a minimal data schema to get your first “top-12%” prototype running.[^1_4][^1_2]
<span style="display:none">[^1_12][^1_13][^1_14][^1_15]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/pdf/2505.02170.pdf

[^1_2]: https://uu.diva-portal.org/smash/get/diva2:1972615/FULLTEXT02.pdf

[^1_3]: https://medium.com/football-applied/the-data-driven-manager-using-linear-programming-to-build-the-optimal-fpl-team-for-2025-26-1bf72b9004d6

[^1_4]: https://medium.com/@jongoodey/how-i-used-an-ai-agent-to-optimise-my-fantasy-premier-league-team-in-minutes-bfc1dc7e8dea

[^1_5]: https://www.fantasyfootballfix.com/blog-index/fpl-top-50-tips-team-setup-25-26/

[^1_6]: https://fpl360.com/2026/06/18/fpl-season-review-2025-26-lessons-learned-strategy-mistakes/

[^1_7]: https://fplai.app/

[^1_8]: https://fpl.team/

[^1_9]: https://www.gethynellis.com/2025/07/how-to-win-at-fantasy-premier-league-using-data-analytics-and-power-bi/

[^1_10]: https://smartplayfpl.com/

[^1_11]: https://www.fantasyfootballfix.com/blog-index/fpl-best-algorithm-team/

[^1_12]: https://fplai.app/fpl-team-analyzer/

[^1_13]: https://fplcentral.co.uk/articles

[^1_14]: https://www.fantasyfootballhub.co.uk/ai-team-rating

[^1_15]: https://everythingaitool.com/tools/fpl-genius


---

# Go one step deeper into these. How do I actually build this? I can combine 2 models as well and generate recommendations.

Like if a person needs to build one can you figure out the data, cited sources and sites, code and model ?

Perfect—let's go from "these papers exist" to "here's exactly how you build a working two-model FPL agent that outputs transfers, captain, and formation." I'll give you: (1) data sources and schemas, (2) a minimal but strong modeling stack (two models + ensemble), (3) the optimizer (with captaincy), and (4) a weekly agentic workflow you can run end-to-end.[^2_1][^2_2][^2_3][^2_4][^2_5]

## 1) Data: where to get it, what to pull, and how to store it

Use two layers: **FPL official API** for points, prices, fixtures; and **Understat** for advanced metrics (xG, xA, xGI, xGC). This matches the feature sets used in the strongest published frameworks.[^2_2][^2_4][^2_5][^2_1]

### Primary sources (reliable, updated)

- **FPL official API (bootstrap-static)**
    - Base URL: `https://fantasy.premierleague.com/api/bootstrap-static/`
    - Key endpoints you'll call weekly:
        - `elements` (players): `id`, `first_name`, `second_name`, `element_type` (GK/DEF/MID/FWD), `team`, `now_cost`, `total_points`, `points_per_game`, `form`, `selected_by_percent`, `minutes`, `goals_scored`, `assists`, `clean_sheets`, `bonus`, `bps`, `ict_index`, `xG`, `xA`, `xGI`, `xGC` (when available), `news` (injury/suspension text).[^2_5][^2_6]
        - `events` (gameweeks): `id`, `is_current`, `is_next`, `deadline_time`.
        - `fixtures`: `id`, `event` (GW), `team_h`, `team_a`, `team_h_difficulty`, `team_a_difficulty`.
        - `teams`: `id`, `name`, `short_name`.
- **Understat (EPL advanced stats)**
    - Use `understatapi` (Python) to fetch per-player season stats: `xG`, `xA`, `npxG`, `xGChain`, `xAChain`, shots, key passes, etc. You'll merge these to FPL players via a mapping table (FPL `id` ↔ Understat `player_id`).[^2_5]
- **Prebuilt CSV repos (if you want to skip scraping)**
    - `vaastav/Fantasy-Premier-League` (GitHub): season/player CSVs, GW-by-GW merged files; easy `pd.read_csv` from raw URLs.[^2_4][^2_7]
    - Kaggle mirrors (e.g., "Fantasy Premier League 2025–2026") updated daily from the FPL API.[^2_8][^2_9]


### Minimal schema you should maintain

Keep one **player–gameweek panel** as your core table (one row per player per GW):

- `player_id` (FPL)
- `gw` (gameweek number)
- `position` (GK/DEF/MID/FWD)
- `team`
- `value` (£m; `now_cost/10`)
- `total_points` (FPL points that GW)
- `minutes`
- `ict_index`, `xG`, `xA`, `xGI`, `xGC` (if available that GW; otherwise use season aggregates as proxies)
- `selected_by_percent`
- `is_available` (boolean; derived from `news` + minutes history)
- `opponent_team`, `was_home`, `fixture_difficulty` (from fixtures)

Store this as a Parquet or CSV in `data/panel.parquet`. You'll recompute it weekly by appending the new GW.[^2_1][^2_4]

## 2) Modeling: two models + an ensemble (the "combine 2 models" part)

You want **Model A (recency/time-series)** and **Model B (features/ridge)**, then blend them. This mirrors the hybrid approach that won in the Ramezani \& Dinh study and the Uppsala thesis.[^2_2][^2_1]

### Model A — Recency/time-series forecaster (points-only)

Goal: predict next-GW points using only the player's points history. The literature shows **recency-weighted averages** and **low-order ARIMA** are the most robust out-of-sample.[^2_1]

Implement two variants and pick the better by validation:

- **Recency-weighted average**
    - For player with GW points $p_1, \dots, p_\tau$:

$$
\hat{c}^{WA} = \sum_{t=1}^{\tau} w_t p_t,\quad w_t = \frac{t}{\sum_{i=1}^{\tau} i}
$$
    - Intuition: recent form gets more weight; simple and stable.[^2_1]
- **ARIMA(0,0,1) or ARIMA(1,0,1)** with a rolling window (e.g., last 10–15 GWs)
    - Fit per player, forecast next GW, use that as $\hat{c}^{ARIMA}$.[^2_1]

Pick one as Model A (start with weighted average; it's simpler and strong).[^2_1]

### Model B — Ridge regression on advanced features (hybrid score)

Goal: predict points using **ICT, xGI, xGC, starts/minutes, selected%** (the features that actually correlate with points by position).[^2_2][^2_1]

- **Features (per player, aggregated over training window, e.g., GW1–26):**
    - `ict_index` (mean or last 3–5 GW rolling mean)
    - `xGI` (expected goal involvements; key for attackers)
    - `xGC` (expected goals conceded; negative signal, especially for DEF/GK)
    - `starts` or `minutes` (availability proxy)
    - `selected_by_percent` (popularity/rotation proxy)
    - Optionally `form` (FPL's own form metric) and `fixture_difficulty`.[^2_3][^2_1]
- **Target:** `total_points` (per GW) or average points over the training window.
- **Model:** **Ridge regression** (per position or global with position dummies). Standardize features; Ridge handles multicollinearity among xG/xA/xGI/ICT.[^2_2][^2_1]
- Output: $\hat{c}^{Ridge}$ for each player for the next GW.

This is essentially the "hybrid score" in the arXiv paper: blend realized points with a ridge prediction on match-performance features.[^2_1]

### Ensemble: combine A and B

Use a convex combination to get your final expected points $c_j$:

$$
c_j = (1-\lambda)\,\hat{c}^{A}_j + \lambda\,\hat{c}^{B}_j,\quad \lambda \in [0,1]
$$

Start with **λ = 0.4–0.6** (give slightly more weight to the feature model if you trust xGI/ICT; or more to recency if you see high variance in advanced stats). You can tune λ on a held-out GW window (e.g., GW27–32) by maximizing cumulative team points after optimization.[^2_1]

Optional: add a small **uncertainty penalty** (robust optimization) using the standard deviation of past points as $d_j$, then optimize worst-case $c_j - \kappa d_j$ if you want rank stability.[^2_1]

## 3) Optimization: pick XI + captain under FPL constraints

This is where most "good predictions" fail to become "good teams." You need a **0–1 integer program** that jointly selects **starting XI and captain** under budget, formation, and 3-per-club constraints.[^2_3][^2_1]

### Decision variables

- $x_j \in \{0,1\}$: player $j$ in starting XI.
- $y_j \in \{0,1\}$: player $j$ is captain.
- $xb_j \in \{0,1\}$: player $j$ on bench (for the 15-man squad).[^2_1]


### Objective

Maximize expected points including captain doubling:

$$
\max \sum_j c_j x_j + \sum_j c_j y_j
$$

(If you use robust scores, replace $c_j$ with $c_j - \kappa d_j$.)[^2_1]

### Constraints (core FPL rules)

- Exactly 11 starters: $\sum_j x_j = 11$.[^2_1]
- Budget for XI: $\sum_j v_j x_j \le b$ (use $b = 83.5$ to leave ~£16.5m for bench).[^2_1]
- One captain: $\sum_j y_j = 1$, and $y_j \le x_j$ (captain must start).[^2_1]
- Positional formation: for each position $k \in \{GK, DEF, MID, FWD\}$,

$$
\text{min\_limit}_k \le \sum_{j \in \text{pos}_k} x_j \le \text{max\_limit}_k
$$

(e.g., 3–4–3, 4–4–2, 5–4–1; you can loop over a set of allowed formations and pick the best).[^2_3][^2_1]
- Max 3 per club: $\sum_{j \in \text{team}_t} x_j \le 3$ for each Premier League team $t$.[^2_1]
- Bench completes a 2–5–5–3 squad under remaining budget and club quotas.[^2_1]


### Solver stack (Python)

- **PuLP** (with CBC) or **OR-Tools CP-SAT** for the integer program. Both are free and easy.[^2_3]
- Pattern:
    - Build one model per allowed formation (e.g., 343, 442, 451, 541).
    - Solve each; keep the formation with the highest objective value.[^2_3]

This is exactly the formulation used in the arXiv framework and the Medium linear-programming article (which also shows working PuLP code).[^2_3][^2_1]

## 4) Weekly agentic workflow (how it actually runs)

Think of this as a **cron job + agent prompt** that does: data refresh → predictions → optimization → recommendations.[^2_10][^2_11]

### Step-by-step (every gameweek)

1. **Refresh data**
    - Pull `bootstrap-static` (players, fixtures, events) and latest GW results.
    - Pull Understat season stats (or update from your cached CSVs).
    - Append new GW rows to `panel.parquet`.[^2_4][^2_5]
2. **Prepare features**
    - For each player: compute recency metrics (last 3–5 GW rolling means of points, ICT, xGI).
    - Aggregate advanced stats (xGI, xGC, ict_index, starts/minutes, selected%).
    - Mark unavailable players (injuries/suspensions via `news` + 0 minutes streaks).[^2_2][^2_1]
3. **Run Model A**
    - Compute weighted-average forecast (or ARIMA) for each available player → $\hat{c}^A$.[^2_1]
4. **Run Model B**
    - Fit Ridge (per position or global with position dummies) on training window → predict next-GW points → $\hat{c}^B$.[^2_2][^2_1]
5. **Ensemble**
    - Combine: $c_j = (1-\lambda)\hat{c}^A_j + \lambda \hat{c}^B_j$.
    - Optionally compute uncertainty $d_j$ (std dev of past points) for robust scoring.[^2_1]
6. **Optimize XI + captain**
    - Run the integer program for each allowed formation; select best.
    - Output:
        - Starting XI (with formation), captain, bench.
        - Expected points total and budget usage.[^2_3][^2_1]
7. **Generate recommendations (the "agent" layer)**
    - Compare optimized XI to the user's current squad.
    - Produce:
        - **Transfers in/out** (respecting free transfer rules; optionally model hit costs).
        - **Captain pick** (and a backup if the primary has >X% injury risk).
        - **Formation change** (if different from user's usual).
    - Add short explanations: "Player X has top-3 xGI + easiest fixture (FDR=2); Model B upweights xGI for midfielders; Model A shows 3-GW form surge; ensemble expected points = 7.8."[^2_10][^2_2]
8. **Log \& evaluate**
    - Store the recommended team and expected points.
    - After the GW, record actual points and update cumulative metrics (points, rank percentile, captain hit rate).[^2_11]

## 5) Minimal code skeleton (Python)

You can drop this into a notebook or a small repo. It's intentionally close to the published setups.[^2_4][^2_3][^2_1]

### Data load (example using vaastav CSVs)

```python
import pandas as pd

# Example: merged GW file for 2025-26
url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2025-26/gws/merged_gw.csv"
df = pd.read_csv(url)

# Keep key columns
cols = ["id", "element_type", "team", "total_points", "minutes", "ict_index",
        "xG", "xA", "xGI", "xGC", "selected_by_percent", "now_cost", "opponent_team", "was_home"]
df = df[cols]
df["value"] = df["now_cost"] / 10.0
```

This gives you the player–GW panel to build features from.[^2_4]

### Model A: recency-weighted average

```python
def weighted_avg_points(group):
    t = group.index + 1  # or use actual GW numbers
    w = t / t.sum()
    return (group["total_points"] * w).sum()

# Example: last 5 GWs per player
last5 = df.groupby("id").tail(5)
model_a = last5.groupby("id").apply(weighted_avg_points).rename("c_A")
```


### Model B: Ridge on features (per position)

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

features = ["ict_index", "xGI", "xGC", "minutes", "selected_by_percent"]
X = df[features].fillna(0)
y = df["total_points"]

# Simple global ridge with position dummies
pos_dummies = pd.get_dummies(df["element_type"], prefix="pos")
X2 = pd.concat([X, pos_dummies], axis=1)

scaler = StandardScaler()
X2s = scaler.fit_transform(X2)

ridge = Ridge(alpha=1.0)
ridge.fit(X2s, y)
df["c_B"] = ridge.predict(X2s)
model_b = df.groupby("id")["c_B"].mean()
```


### Ensemble + optimizer (PuLP sketch)

```python
import pulp as pl

# Merge models
exp_pts = pd.DataFrame({"c_A": model_a, "c_B": model_b}).fillna(0)
lam = 0.5
exp_pts["c"] = (1-lam)*exp_pts["c_A"] + lam*exp_pts["c_B"]

# Merge with latest player meta (value, position, team)
meta = df.groupby("id").agg(
    position=("element_type", "first"),
    team=("team", "first"),
    value=("value", "first")
).join(exp_pts)

players = meta.reset_index()

# Example: 4-4-2 formation
formation = {"GK":1, "DEF":4, "MID":4, "FWD":2}
budget = 83.5

prob = pl.LpProblem("FPL_442", pl.LpMaximize)
x = {i: pl.LpVariable(f"x_{i}", cat="Binary") for i in players.index}
y = {i: pl.LpVariable(f"y_{i}", cat="Binary") for i in players.index}

# Objective: expected points + captain
prob += pl.lpSum(players.loc[i, "c"] * x[i] for i in players.index) + \
        pl.lpSum(players.loc[i, "c"] * y[i] for i in players.index)

# Constraints
prob += pl.lpSum(x[i] for i in players.index) == 11
prob += pl.lpSum(players.loc[i, "value"] * x[i] for i in players.index) <= budget
prob += pl.lpSum(y[i] for i in players.index) == 1
for i in players.index:
    prob += y[i] <= x[i]

# Positional
for pos, count in formation.items():
    idx = players.index[players["position"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"}) == pos]
    prob += pl.lpSum(x[i] for i in idx) == count

# Max 3 per team
for t in players["team"].unique():
    idx = players.index[players["team"] == t]
    prob += pl.lpSum(x[i] for i in idx) <= 3

prob.solve(pl.PULP_CBC_CMD(msg=False))

xi = players.loc[[i for i in players.index if pl.value(x[i]) > 0.5]]
captain = players.loc[[i for i in players.index if pl.value(y[i]) > 0.5]]
```

This is the core of the arXiv framework and the Medium LP article, adapted to your two-model ensemble.[^2_3][^2_1]

## 6) Where to host data and models (practical setup)

- **Data:**
    - Primary: FPL API + Understat (via `understatapi`), or use `vaastav` CSVs for speed.[^2_5][^2_4]
    - Store in `data/panel.parquet` and `data/meta.parquet`.
- **Models:**
    - Save Ridge coefficients and scaler (joblib) so you don't retrain from scratch weekly.
    - Keep λ (ensemble weight) as a config; tune quarterly.
- **Scheduler:**
    - Run weekly (pre-deadline) via cron/GitHub Actions: refresh → predict → optimize → write recommendations to a JSON/CSV or push to a simple UI/Slack.[^2_11][^2_10]


## 7) Cited sources and repos to clone

- **ArXiv framework (data + code):** "Data-Driven Team Selection in FPL Using Integer Programming" (Ramezani \& Dinh) — datasets, preprocessing, and experiment notebooks.[^2_1]
- **Uppsala thesis (ML + LLM assistant):** "Enhancing FPL Strategies through Machine Learning and Large Language Models" — linear/XGBoost models, squad optimization, explainability layer.[^2_2]
- **LP team builder (working PuLP code):** "The Data-Driven Manager: Using Linear Programming to Build the Optimal FPL Team" (Football Applied).[^2_3]
- **FPL data repos:**
    - `vaastav/Fantasy-Premier-League` (CSVs by season/GW).[^2_7][^2_4]
    - Kaggle "Fantasy Premier League 2025–2026" (daily-updated API dumps).[^2_9][^2_8]
- **FPL API wrapper:** `fpl` Python package (async) for direct API access.[^2_12]
- **Understat integration:** `understatapi` for xG/xA/npxG/xGChain, merged to FPL IDs.[^2_5]

If you tell me your preferred stack (e.g., "Python + PuLP, deploy on GitHub Actions, output to a simple Streamlit app"), I can sketch a repo layout and a weekly agent prompt that ties data refresh → modeling → optimization → natural-language recommendations in one flow.[^2_12][^2_10][^2_3]
<span style="display:none">[^2_13][^2_14][^2_15][^2_16][^2_17][^2_18][^2_19][^2_20][^2_21]</span>

<div align="center">⁂</div>

[^2_1]: https://arxiv.org/pdf/2505.02170.pdf

[^2_2]: https://uu.diva-portal.org/smash/get/diva2:1972615/FULLTEXT02.pdf

[^2_3]: https://medium.com/football-applied/the-data-driven-manager-using-linear-programming-to-build-the-optimal-fpl-team-for-2025-26-1bf72b9004d6

[^2_4]: https://github.com/vaastav/Fantasy-Premier-League

[^2_5]: https://stateastic.home.blog/2022/08/02/getting-data-from-fpl-and-understat-to-do-analysis/

[^2_6]: https://gist.github.com/sertalpbilal/faf3e50a41d3ae51a2f38baa061df24d

[^2_7]: https://github.com/vaastav/Fantasy-Premier-League/tree/master/data/2025-26/players

[^2_8]: https://www.kaggle.com/datasets/calvinrostanto/fantasy-premier-league-2025-2026

[^2_9]: https://www.kaggle.com/datasets/meraxes10/fantasy-premier-league-dataset-2025-2026

[^2_10]: https://medium.com/@jongoodey/how-i-used-an-ai-agent-to-optimise-my-fantasy-premier-league-team-in-minutes-bfc1dc7e8dea

[^2_11]: https://www.gethynellis.com/2025/07/how-to-win-at-fantasy-premier-league-using-data-analytics-and-power-bi/

[^2_12]: https://fpl.readthedocs.io/

[^2_13]: https://github.com/calvrost/fpl-2025-2026

[^2_14]: https://github.com/Ayanab01/FPL_Stats

[^2_15]: https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2024-25/teams.csv

[^2_16]: https://www.kaggle.com/datasets/reevebarreto/fantasy-premier-league-player-data-2016-2024

[^2_17]: https://github.com/olbauday/FPL-Core-Insights

[^2_18]: https://github.com/topics/fantasy-premier-league?o=desc\&s=forks

[^2_19]: https://www.reddit.com/r/fplAnalytics/comments/1n5rd1n/help_looking_for_a_way_to_export_202526_goal/

[^2_20]: https://www.reddit.com/r/fplAnalytics/comments/1m65kf8/my_fpl_dataset_is_back_for_2526_now_with_all/

[^2_21]: https://www.github-zh.com/topics/fantasy-premier-league

