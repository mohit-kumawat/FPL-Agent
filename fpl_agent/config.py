"""Central configuration: paths, FPL rules, model settings."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
HISTORY_DIR = DATA_DIR / "history"
REPORTS_DIR = ROOT / "reports"
MEMORY_DIR = ROOT / "memory"
SQUAD_FILE = ROOT / "squad.yaml"

for _d in (DATA_DIR, SNAPSHOT_DIR, HISTORY_DIR, REPORTS_DIR, MEMORY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- FPL rules
BUDGET = 100.0          # £m for a fresh 15-man squad
SQUAD_SIZE = 15
MAX_PER_CLUB = 3
# position id -> (name, squad count, XI min, XI max)
POSITIONS = {
    1: ("GKP", 2, 1, 1),
    2: ("DEF", 5, 3, 5),
    3: ("MID", 5, 2, 5),
    4: ("FWD", 3, 1, 3),
}
XI_SIZE = 11
TRANSFER_HIT = 4        # points per extra transfer
MAX_FREE_TRANSFERS = 5  # free transfers bank up to five
# Upper bound on the transfer-count search. Must exceed MAX_FREE_TRANSFERS so a
# hit is always evaluated (and then usually rejected by policy) rather than
# never considered.
MAX_TRANSFER_SEARCH = 6

# ------------------------------------------------------------- API endpoints
API_BASE = "https://fantasy.premierleague.com/api"
ENDPOINTS = {
    "bootstrap": f"{API_BASE}/bootstrap-static/",
    "fixtures": f"{API_BASE}/fixtures/",
    "entry": f"{API_BASE}/entry/{{entry_id}}/",
    "picks": f"{API_BASE}/entry/{{entry_id}}/event/{{gw}}/picks/",
    "live": f"{API_BASE}/event/{{gw}}/live/",
    "element_summary": f"{API_BASE}/element-summary/{{player_id}}/",
}
USER_AGENT = "Mozilla/5.0 (fpl-agent personal research)"

# Price freshness. For 2026/27 FPL applies price changes daily at 00:00 *UK
# local time* (premierleague.com/en/news/4680462). That is a local wall-clock
# time, so it lands at 23:00 UTC under BST and 00:00 UTC under GMT — a fixed
# UTC constant cannot express it and would serve pre-change prices for hours.
# See data.last_price_change().
PRICE_CHANGE_TZ = "Europe/London"
PRICE_CHANGE_LOCAL_HOUR = 0
PRICE_CHANGE_LOCAL_MINUTE = 0
PRICE_CHANGE_GRACE_MINUTES = 20  # the API takes a few minutes to settle
MAX_SNAPSHOT_AGE_HOURS = 12      # refetch anything older, even same-day
DEADLINE_FRESH_HOURS = 3         # within 24h of a deadline, demand this freshness

# Prior-season gameweek history (vaastav repo)
VAASTAV_MERGED_GW = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
VAASTAV_FIXTURES = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/fixtures.csv"
)
VAASTAV_TEAMS = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/teams.csv"
)
PRIOR_SEASON = "2025-26"
CURRENT_SEASON = "2026-27"

# ------------------------------------------------------- scoring regimes
# Seasons in the same generation score points under the same rules. Prior-season
# PPG carried across a boundary embeds points earned under rules that no longer
# apply (a defender's DC points, a keeper's old bonus profile), while process
# stats (xG, xA, xGC) are rule-independent. Tier P shifts weight accordingly.
RULES_VERSION = {
    "2021-22": 1, "2022-23": 1, "2023-24": 1, "2024-25": 1,
    "2025-26": 2,   # defensive contributions added, GK goal = 10, assist changes
    "2026-27": 3,   # BPS rework
}
# Weight on realized PPG (vs the process-stat ridge) inside the Tier P prior.
PPG_PRIOR_WEIGHT = 0.5              # same regime — validated on the 2024/25 replay
# Cross-regime pick is conservative and CANNOT be validated before GW1 of the
# new rules; revisit once the ablation ladder can measure it (roadmap item 1).
PPG_PRIOR_WEIGHT_CROSS_REGIME = 0.35


def rules_cross_regime(prior_season: str, current_season: str) -> bool:
    """True when scoring rules changed between the two seasons. A season missing
    from RULES_VERSION counts as a change — downweighting an unknown regime is
    the safe failure."""
    a, b = RULES_VERSION.get(prior_season), RULES_VERSION.get(current_season)
    return a is None or b is None or a != b


# ------------------------------------------------------------- model config
HORIZON_GWS = 5             # expected-points horizon for planning
FORM_WINDOW = 5             # rolling window for recency model
ENSEMBLE_LAMBDA = 0.6       # weight on Model B (ridge) vs Model A (recency)
# Two-season lambda grid (eval/strategy-sim-report.md): the ranking FLIPS by
# season (2024/25 favors recency, 2025/26 favors ridge) — no extreme is robust.
# 0.6 is the maximin choice (best worst-case top-11 across both: 4.14).
# Earlier 0.7 was overfit to a single window; don't repeat that mistake.
RIDGE_ALPHA = 1.0
# Cold start: weight on prior-season baseline decays as current season
# accumulates gameweeks. weight_prior = max(0, (K - gws_played) / K)
COLD_START_GWS = 8
# Robustness penalty on expected points (kappa * std of past points)
ROBUST_KAPPA = 0.0          # off by default; rating card reports variance instead

# Fixture difficulty: expected-points multiplier per FDR value
# (fallback only — continuous strength-based multipliers are primary, see
# features.strength_fixture_mult; validated on 2024/25 replay)
FDR_MULTIPLIER = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.90, 5: 0.80}
# Continuous fixture model: mult = clip((0.65*att_ratio + 0.35*def_ratio)^GAMMA)
STRENGTH_GAMMA = 1.0        # sensitivity of the strength ratio
STRENGTH_CLIP = (0.70, 1.40)
HORIZON_DISCOUNT = 0.90     # per-GW discount inside ep_horizon: near-term dominates
W_COMPONENT = 0.55          # weight on component-based EP vs realized-PPG path

# Optimizer defaults
BENCH_WEIGHT = 0.15         # bench players contribute this fraction of EP in squad objective
MIN_PRICE_GK = 4.0

# --------------------------------------------------------- price-change timing
# bootstrap's price_change_percent: signed progress toward tonight's change
# (positive -> rise, negative -> fall; observed all-zero preseason). Values are
# advisory — the field's exact internals are FPL's — so thresholds are set
# where being wrong is cheap: an "act tonight" nudge on a transfer the model
# already wants, never a transfer the model didn't ask for.
PRICE_MOVE_IMMINENT = 90.0   # |pct| >= this: treat tonight's move as likely
PRICE_MOVE_WATCH = 60.0      # |pct| >= this: mention the momentum, no urgency
