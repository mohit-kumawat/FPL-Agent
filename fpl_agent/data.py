"""Data layer: FPL API client, snapshot store with change detection, prior-season history."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from . import config


# --------------------------------------------------------------------- fetch
def _get(url: str, retries: int = 3, timeout: int = 30) -> Any:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def fetch_bootstrap() -> dict:
    return _get(config.ENDPOINTS["bootstrap"])


def fetch_fixtures() -> list[dict]:
    return _get(config.ENDPOINTS["fixtures"])


def fetch_entry(entry_id: int) -> dict:
    return _get(config.ENDPOINTS["entry"].format(entry_id=entry_id))


def fetch_picks(entry_id: int, gw: int) -> dict:
    return _get(config.ENDPOINTS["picks"].format(entry_id=entry_id, gw=gw))


def fetch_element_summary(player_id: int) -> dict:
    return _get(config.ENDPOINTS["element_summary"].format(player_id=player_id))


# ----------------------------------------------------------------- snapshots
def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _snap_path(name: str, day: str | None = None) -> Path:
    return config.SNAPSHOT_DIR / f"{name}_{day or _today()}.json"


def _meta_path(name: str, day: str | None = None) -> Path:
    return config.SNAPSHOT_DIR / f"{name}_{day or _today()}.meta.json"


def save_snapshot(name: str, payload: Any) -> Path:
    path = _snap_path(name)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    _meta_path(name).write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": config.ENDPOINTS.get(name, "unknown"),
    }))
    return path


def snapshot_fetched_at(name: str, day: str | None = None) -> datetime | None:
    """When today's snapshot was actually fetched (not just its date stamp)."""
    p = _meta_path(name, day)
    if not p.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(p.read_text())["fetched_at"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def last_price_change(now: datetime | None = None) -> datetime:
    """Most recent FPL price-change boundary (daily ~01:30 UTC)."""
    now = now or datetime.now(timezone.utc)
    boundary = now.replace(hour=config.PRICE_CHANGE_UTC_HOUR,
                           minute=config.PRICE_CHANGE_UTC_MINUTE,
                           second=0, microsecond=0)
    if boundary > now:
        boundary -= timedelta(days=1)
    return boundary


def is_stale(name: str, now: datetime | None = None,
             hours_to_deadline: float | None = None) -> tuple[bool, str]:
    """Is the cached snapshot too old to trust for prices? (stale, reason).

    Near a deadline the bar tightens to DEADLINE_FRESH_HOURS so that refresh
    satisfies what verify demands — otherwise the daily run blocks itself on
    data it refuses to replace.
    """
    now = now or datetime.now(timezone.utc)
    fetched = snapshot_fetched_at(name)
    if fetched is None:
        return True, "no snapshot for today"
    if fetched < last_price_change(now):
        return True, (f"fetched {fetched:%Y-%m-%d %H:%MZ}, before the "
                      f"{last_price_change(now):%H:%MZ} price change")
    max_age = config.MAX_SNAPSHOT_AGE_HOURS
    if hours_to_deadline is not None and 0 < hours_to_deadline <= 24:
        max_age = config.DEADLINE_FRESH_HOURS
    age_h = (now - fetched).total_seconds() / 3600
    if age_h > max_age:
        return True, f"snapshot is {age_h:.1f}h old (max {max_age}h)"
    return False, f"fresh ({age_h:.1f}h old, after last price change)"


def load_snapshot(name: str, day: str | None = None) -> Any | None:
    path = _snap_path(name, day)
    if path.exists():
        return json.loads(path.read_text())
    return None


def latest_snapshot_before(name: str, day: str) -> tuple[str, Any] | None:
    """Most recent snapshot strictly before `day` (for diffing).

    The glob must exclude the `.meta.json` sidecars — they match `{name}_*.json`
    and sort after the real payload, so an unfiltered glob hands a metadata dict
    to detect_changes.
    """
    candidates = sorted(p for p in config.SNAPSHOT_DIR.glob(f"{name}_*.json")
                        if not p.name.endswith(".meta.json"))
    prev = [p for p in candidates if p.stem.split("_")[-1] < day]
    if not prev:
        return None
    p = prev[-1]
    return p.stem.split("_")[-1], json.loads(p.read_text())


def refresh(force: bool = False) -> dict:
    """Fetch bootstrap + fixtures, snapshot them, return both.

    Price freshness: FPL prices change daily at ~01:30 UTC. A same-day cache
    is NOT sufficient — a snapshot taken at 00:30 holds yesterday's prices all
    day. We refetch whenever the cached copy predates the last price-change
    boundary or exceeds MAX_SNAPSHOT_AGE_HOURS. `freshness` is returned so the
    report can state, per run, how current the prices are.
    """
    # peek at the cached copy for the next deadline so the freshness bar can
    # tighten near it (deadlines are stable, so a cached read is safe here)
    cached = load_snapshot("bootstrap")
    hours_to_deadline = None
    if cached:
        nxt = next((e for e in cached.get("events", []) if e.get("is_next")), None)
        if nxt:
            dl = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
            hours_to_deadline = (dl - datetime.now(timezone.utc)).total_seconds() / 3600

    stale, reason = is_stale("bootstrap", hours_to_deadline=hours_to_deadline)
    need = force or stale
    boot = None if need else cached
    fixtures = None if need else load_snapshot("fixtures")
    refetched = False
    if boot is None:
        boot = fetch_bootstrap()
        save_snapshot("bootstrap", boot)
        refetched = True
    if fixtures is None:
        fixtures = fetch_fixtures()
        save_snapshot("fixtures", fixtures)
    fetched = snapshot_fetched_at("bootstrap")
    return {
        "bootstrap": boot,
        "fixtures": fixtures,
        "freshness": {
            "fetched_at": fetched.isoformat() if fetched else None,
            "refetched": refetched,
            "reason": reason if need else reason,
            "last_price_change": last_price_change().isoformat(),
            "source": config.ENDPOINTS["bootstrap"],
        },
    }


# ----------------------------------------------------------- change detection
_PLAYER_WATCH_FIELDS = [
    "now_cost", "status", "news", "chance_of_playing_next_round",
    "transfers_in_event", "transfers_out_event", "selected_by_percent",
]


def detect_changes(today_boot: dict, prev_boot: dict | None) -> dict:
    """Diff two bootstrap snapshots into a structured change report."""
    changes: dict[str, Any] = {
        "first_run": prev_boot is None,
        "price_changes": [],
        "status_changes": [],
        "news_changes": [],
        "new_players": [],
        "removed_players": [],
        "gw_state": {},
    }
    events = today_boot["events"]
    nxt = next((e for e in events if e["is_next"]), None)
    cur = next((e for e in events if e["is_current"]), None)
    changes["gw_state"] = {
        "current_gw": cur["id"] if cur else None,
        "next_gw": nxt["id"] if nxt else None,
        "next_deadline": nxt["deadline_time"] if nxt else None,
        "gws_finished": sum(1 for e in events if e["finished"]),
    }
    if prev_boot is None:
        return changes

    prev_players = {e["id"]: e for e in prev_boot["elements"]}
    cur_players = {e["id"]: e for e in today_boot["elements"]}
    teams = {t["id"]: t["short_name"] for t in today_boot["teams"]}

    for pid, p in cur_players.items():
        old = prev_players.get(pid)
        label = f"{p['web_name']} ({teams.get(p['team'], '?')})"
        if old is None:
            changes["new_players"].append(label)
            continue
        if p["now_cost"] != old["now_cost"]:
            changes["price_changes"].append({
                "player": label, "from": old["now_cost"] / 10, "to": p["now_cost"] / 10,
            })
        if p["status"] != old["status"]:
            changes["status_changes"].append({
                "player": label, "from": old["status"], "to": p["status"],
                "news": p.get("news", ""),
            })
        if (p.get("news") or "") != (old.get("news") or ""):
            changes["news_changes"].append({"player": label, "news": p.get("news", "")})
    for pid, p in prev_players.items():
        if pid not in cur_players:
            changes["removed_players"].append(p["web_name"])

    prev_events = prev_boot["events"]
    changes["new_gw_finished"] = (
        sum(1 for e in events if e["finished"])
        > sum(1 for e in prev_events if e["finished"])
    )
    return changes


def content_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------- prior history
def load_prior_season(season: str = config.PRIOR_SEASON, force: bool = False) -> pd.DataFrame:
    """Player-GW panel for a prior season (vaastav merged_gw.csv), cached locally."""
    cache = config.HISTORY_DIR / f"merged_gw_{season}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    url = config.VAASTAV_MERGED_GW.format(season=season)
    r = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=120)
    r.raise_for_status()
    import io
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    df.to_parquet(cache, index=False)
    return df


# -------------------------------------------------- current-season GW panel
def build_current_panel(boot: dict, force: bool = False) -> pd.DataFrame:
    """Player-GW panel for the current season, built from element-summary.

    Expensive (~1 call/player), so it only refetches when a new finished GW
    isn't yet in the cache. Preseason returns an empty frame.
    """
    cache = config.HISTORY_DIR / f"panel_{config.CURRENT_SEASON}.parquet"
    gws_finished = sum(1 for e in boot["events"] if e["finished"])
    if gws_finished == 0:
        return pd.DataFrame()
    if cache.exists() and not force:
        panel = pd.read_parquet(cache)
        if panel["round"].max() >= gws_finished:
            return panel
    rows: list[dict] = []
    for el in boot["elements"]:
        try:
            summ = fetch_element_summary(el["id"])
        except RuntimeError:
            continue
        for h in summ.get("history", []):
            rows.append(h)
        time.sleep(0.05)  # be polite
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel.to_parquet(cache, index=False)
    return panel


def _load_csv(url_template: str, season: str, name: str) -> pd.DataFrame:
    cache = config.HISTORY_DIR / f"{name}_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import io
    r = requests.get(url_template.format(season=season),
                     headers={"User-Agent": config.USER_AGENT}, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
    df.to_parquet(cache, index=False)
    return df


def load_season_fixtures(season: str) -> pd.DataFrame:
    return _load_csv(config.VAASTAV_FIXTURES, season, "fixtures")


def load_season_teams(season: str) -> pd.DataFrame:
    return _load_csv(config.VAASTAV_TEAMS, season, "teams")


# ------------------------------------------------------------------- frames
def players_frame(boot: dict) -> pd.DataFrame:
    """Current player table with numeric columns coerced."""
    df = pd.DataFrame(boot["elements"])
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    df["team_short"] = df["team"].map(teams)
    df["price"] = df["now_cost"] / 10.0
    df["position"] = df["element_type"].map({k: v[0] for k, v in config.POSITIONS.items()})
    num_cols = [
        "form", "points_per_game", "selected_by_percent", "ict_index",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "ep_next", "ep_this", "value_season",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["available"] = df["status"].isin(["a", "d"]) & (df.get("can_select", True))
    # chance of playing: None means fully fit
    cop = df["chance_of_playing_next_round"]
    df["play_chance"] = cop.fillna(100).astype(float) / 100.0
    df.loc[df["status"].isin(["i", "s", "u", "n"]), "play_chance"] = df.loc[
        df["status"].isin(["i", "s", "u", "n"]), "play_chance"
    ].clip(upper=0.25)
    return df


def fixtures_frame(fixtures: list[dict], boot: dict) -> pd.DataFrame:
    fx = pd.DataFrame(fixtures)
    teams = {t["id"]: t["short_name"] for t in boot["teams"]}
    fx["home"] = fx["team_h"].map(teams)
    fx["away"] = fx["team_a"].map(teams)
    return fx
