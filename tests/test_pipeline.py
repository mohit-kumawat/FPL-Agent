"""Offline unit tests — no network. Run: uv run pytest -q"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fpl_agent import config, data, features, memoryio, models, optimizer, policy


# ------------------------------------------------------------- fixtures
def _teams() -> list[dict]:
    return [{"id": i, "short_name": f"T{i}", "name": f"Team {i}",
             "strength_attack_home": 1200, "strength_attack_away": 1150,
             "strength_defence_home": 1200, "strength_defence_away": 1150}
            for i in range(1, 6)]


def _players(n_per_pos: int = 6) -> pd.DataFrame:
    rows = []
    pid = 1
    for et in (1, 2, 3, 4):
        for k in range(n_per_pos):
            rows.append({
                "id": pid, "web_name": f"P{pid}", "element_type": et,
                "team": (pid % 5) + 1, "team_short": f"T{(pid % 5) + 1}",
                "position": config.POSITIONS[et][0], "price": 4.5 + k * 0.5,
                "minutes": 2500, "starts": 30, "points_per_game": 3.0 + k * 0.3,
                "expected_goals": 4.0 if et >= 3 else 1.0,
                "expected_assists": 3.0 if et >= 3 else 1.0,
                "expected_goal_involvements": 7.0 if et >= 3 else 2.0,
                "expected_goals_conceded": 40.0, "ict_index": 100.0,
                "bonus": 10, "saves": 60 if et == 1 else 0,
                "available": True, "play_chance": 1.0, "sp_bonus": 0.0,
                "gws_played": 0, "n_fixtures": 5, "next_n_fixtures": 1,
                "ep_mult": 4.5, "att_mult": 4.5, "def_mult": 4.5, "disc_sum": 4.5,
                "next_gw_mult": 1.0, "next_att_mult": 1.0, "next_def_mult": 1.0,
                "avg_fdr": 3.0, "roll_points": float("nan"),
                "roll_minutes": float("nan"), "roll_starts": float("nan"),
                "selected_by_percent": 5.0,
            })
            pid += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------- price freshness
def test_price_boundary_tracks_uk_local_midnight_in_summer():
    """BST: 00:00 UK is 23:00 UTC the previous day (+ settle grace)."""
    now = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)
    assert data.last_price_change(now) == datetime(2026, 8, 16, 23, 20, tzinfo=timezone.utc)


def test_price_boundary_tracks_uk_local_midnight_in_winter():
    """GMT: 00:00 UK is 00:00 UTC the same day (+ settle grace)."""
    now = datetime(2026, 12, 17, 9, 0, tzinfo=timezone.utc)
    assert data.last_price_change(now) == datetime(2026, 12, 17, 0, 20, tzinfo=timezone.utc)


def test_snapshot_taken_before_uk_midnight_change_is_stale(monkeypatch):
    """The core guarantee: a cache from before the change must NOT be reused.

    Regression for a fixed 01:30 UTC constant, which served pre-change prices
    for hours during BST while reporting them as fresh.
    """
    now = datetime(2026, 8, 17, 23, 30, tzinfo=timezone.utc)     # after 00:00 UK
    monkeypatch.setattr(data, "snapshot_fetched_at",
                        lambda name, day=None: datetime(2026, 8, 17, 22, 0, tzinfo=timezone.utc))
    stale, reason = data.is_stale("bootstrap", now)
    assert stale and "price change" in reason


def test_recent_snapshot_is_fresh(monkeypatch):
    now = datetime(2026, 8, 11, 9, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(data, "snapshot_fetched_at",
                        lambda name, day=None: now - timedelta(hours=2))
    stale, _ = data.is_stale("bootstrap", now)
    assert not stale


def test_old_snapshot_exceeds_max_age(monkeypatch):
    now = datetime(2026, 8, 11, 23, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(data, "snapshot_fetched_at",
                        lambda name, day=None: now - timedelta(hours=13))
    stale, reason = data.is_stale("bootstrap", now)
    assert stale and "old" in reason


def test_near_deadline_tightens_the_freshness_bar(monkeypatch):
    """refresh must replace what verify would reject, or daily blocks forever."""
    now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(data, "snapshot_fetched_at",
                        lambda name, day=None: now - timedelta(hours=4))
    assert not data.is_stale("bootstrap", now)[0]                    # normal day
    assert data.is_stale("bootstrap", now, hours_to_deadline=5.5)[0]  # deadline near


def test_snapshot_glob_ignores_meta_sidecars(tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    (tmp_path / "bootstrap_2026-08-11.json").write_text(_json.dumps({"elements": [1]}))
    (tmp_path / "bootstrap_2026-08-11.meta.json").write_text(
        _json.dumps({"fetched_at": "x", "source": "y"}))
    day, payload = data.latest_snapshot_before("bootstrap", "2026-08-12")
    assert day == "2026-08-11" and "elements" in payload


# ------------------------------------------------------------- selling price
def test_selling_price_halves_profits_but_takes_losses_in_full():
    players = pd.DataFrame([{"id": 1, "price": 5.4}, {"id": 2, "price": 5.0},
                            {"id": 3, "price": 4.5}, {"id": 4, "price": 5.1}])
    squad = {"players": [{"id": 1, "purchase_price": 5.0},
                         {"id": 2, "purchase_price": 5.0},
                         {"id": 3, "purchase_price": 5.0},
                         {"id": 4, "purchase_price": 5.0}]}
    sell = memoryio.squad_selling_prices(squad, players)
    assert sell[1] == 5.2      # +0.4 profit -> half kept
    assert sell[2] == 5.0      # unchanged
    assert sell[3] == 4.5      # fallen: the full loss is taken, no purchase-price floor
    assert sell[4] == 5.0      # +0.1 profit -> rounds down to nothing


# ------------------------------------------------------------- fixtures model
def test_strength_channels_reward_easier_opponent():
    teams = pd.DataFrame(_teams()).set_index("id")
    teams.loc[2, "strength_defence_away"] = 800    # weak opponent defence
    strong_att, _ = features.strength_channels(teams, 1, 2, is_home=True)
    normal_att, _ = features.strength_channels(teams, 1, 3, is_home=True)
    assert strong_att > normal_att


def test_fixture_multiplier_is_clipped():
    teams = pd.DataFrame(_teams()).set_index("id")
    teams.loc[2, "strength_defence_away"] = 1
    att, _ = features.strength_channels(teams, 1, 2, is_home=True)
    assert att <= config.STRENGTH_CLIP[1]


# ------------------------------------------------------------- EP model
def test_expected_points_produces_no_nans_and_ordered_bands():
    ep = models.expected_points(_players())
    assert ep["ep_next"].notna().all()
    assert ep["ep_horizon"].notna().all()
    assert (ep["ep_p10"] <= ep["ep_next"]).all()
    assert (ep["ep_next"] <= ep["ep_p90"]).all()


def test_blank_gameweek_zeroes_next_gw_ep():
    df = _players()
    df.loc[df["team"] == 1, ["next_gw_mult", "next_att_mult",
                             "next_def_mult", "next_n_fixtures"]] = 0
    ep = models.expected_points(df)
    blanked = ep[ep["team"] == 1]
    assert (blanked["ep_next"] < 0.01).all()


def test_signal_minutes_cap_reduces_ep():
    df = _players()
    base = models.expected_points(df)
    sig = pd.DataFrame({"xmins_max": [0.2], "xmins_min": [None],
                        "ep_per_gw": [0.0]}, index=[5])
    capped = models.expected_points(df, signal_adjust=sig)
    assert capped.loc[capped["id"] == 5, "ep_next"].iloc[0] < \
        base.loc[base["id"] == 5, "ep_next"].iloc[0]


# ------------------------------------------------------------- optimizer
def test_build_squad_respects_all_fpl_rules():
    ep = models.expected_points(_players(8))
    res = optimizer.build_squad(ep)
    squad = res["squad"]
    assert len(squad) == config.SQUAD_SIZE
    assert res["cost"] <= config.BUDGET + 1e-6
    assert squad["team"].value_counts().max() <= config.MAX_PER_CLUB
    for et, (_, want, _, _) in config.POSITIONS.items():
        assert (squad["element_type"] == et).sum() == want


def test_pick_xi_produces_legal_formation_and_captain():
    ep = models.expected_points(_players(8))
    squad = optimizer.build_squad(ep)["squad"]
    xi = optimizer.pick_xi(squad)
    assert len(xi["xi"]) == config.XI_SIZE
    assert (xi["xi"]["element_type"] == 1).sum() == 1
    assert xi["captain"]["id"] in set(xi["xi"]["id"])
    assert len(xi["bench_order"]) == 4


# ------------------------------------------------------------- policy
def test_thresholds_are_stricter_early_season():
    early_ft, early_hit = policy.thresholds(gws_played=1, free_transfers=1)
    late_ft, late_hit = policy.thresholds(gws_played=20, free_transfers=1)
    assert early_ft > late_ft and early_hit > late_hit


def test_thresholds_ease_when_free_transfers_are_capped():
    normal, _ = policy.thresholds(gws_played=20, free_transfers=1)
    banked, _ = policy.thresholds(gws_played=20, free_transfers=5)
    assert banked < normal


def test_marginal_gain_is_rejected():
    plan = {"plans": [
        {"n_transfers": 0, "net_gain_vs_hold": 0.0, "hit_cost": 0,
         "out": pd.DataFrame(), "in": pd.DataFrame(), "objective": 100},
        {"n_transfers": 1, "net_gain_vs_hold": 0.4, "hit_cost": 0,
         "out": pd.DataFrame(), "in": pd.DataFrame(columns=["web_name", "xmins"]),
         "objective": 100.4},
    ]}
    assert policy.assess_transfers(plan, free_transfers=1, gws_played=20)["action"] == "hold"


def test_best_net_gain_wins_not_the_largest_transfer_count():
    empty = pd.DataFrame(columns=["web_name", "xmins"])
    plan = {"plans": [
        {"n_transfers": 0, "net_gain_vs_hold": 0.0, "hit_cost": 0,
         "out": empty, "in": empty, "objective": 100},
        {"n_transfers": 1, "net_gain_vs_hold": 5.0, "hit_cost": 0,
         "out": empty, "in": empty, "objective": 105},
        {"n_transfers": 2, "net_gain_vs_hold": 4.2, "hit_cost": 0,
         "out": empty, "in": empty, "objective": 104.2},
    ]}
    d = policy.assess_transfers(plan, free_transfers=2, gws_played=20)
    assert d["action"] == "1_transfer"
    assert d["plan"]["net_gain_vs_hold"] == 5.0


def test_hit_on_minutes_capped_player_is_blocked():
    incoming = pd.DataFrame([{"web_name": "Risky", "xmins": 0.3}])
    plan = {"plans": [
        {"n_transfers": 0, "net_gain_vs_hold": 0.0, "hit_cost": 0,
         "out": pd.DataFrame(), "in": pd.DataFrame(), "objective": 100},
        {"n_transfers": 2, "net_gain_vs_hold": 6.5, "hit_cost": 4,
         "out": pd.DataFrame(), "in": incoming, "objective": 106.5},
    ]}
    # 6.5 clears the 6.0 bar but not the 7.5 required for a minutes-capped buy
    assert policy.assess_transfers(plan, free_transfers=1, gws_played=20)["action"] == "hold"


# ------------------------------------------------------------- signals
def _write_signal(tmp_path, monkeypatch, body: str) -> tuple:
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path)
    (tmp_path / "s.yaml").write_text(body)
    return memoryio.load_signals(now=datetime(2026, 8, 20, tzinfo=timezone.utc))


def test_contradictory_minutes_bounds_are_rejected(tmp_path, monkeypatch):
    frame, notes = _write_signal(tmp_path, monkeypatch, """
date: 2026-08-19
adjustments:
  - player_id: 12
    xmins_min: 0.9
    xmins_max: 0.45
""")
    assert frame.empty
    assert any("contradictory" in p for n in notes for p in n["problems"])


def test_expired_signal_is_ignored_without_manual_deletion(tmp_path, monkeypatch):
    frame, notes = _write_signal(tmp_path, monkeypatch, """
date: 2026-08-01
ttl_days: 7
adjustments:
  - player_id: 12
    xmins_max: 0.4
""")
    assert frame.empty
    assert any("expired" in p for n in notes for p in n["problems"])


def test_oversized_ep_nudge_is_rejected(tmp_path, monkeypatch):
    frame, notes = _write_signal(tmp_path, monkeypatch, """
date: 2026-08-19
adjustments:
  - player_id: 12
    ep_per_gw: 9.0
""")
    assert frame.empty
    assert any("exceeds" in p for n in notes for p in n["problems"])


def test_valid_signal_is_applied_with_confidence_weight(tmp_path, monkeypatch):
    frame, notes = _write_signal(tmp_path, monkeypatch, """
date: 2026-08-19
confidence: low
adjustments:
  - player_id: 12
    ep_per_gw: 1.0
    xmins_min: 0.85
""")
    assert frame.loc[12, "ep_per_gw"] == pytest.approx(0.3)   # low = 0.3x
    assert frame.loc[12, "xmins_min"] == 0.85
    assert notes[0]["applied"]


# ------------------------------------------------------------- transfer depth
def test_transfer_search_covers_banked_free_transfers():
    ep = models.expected_points(_players(8))
    squad = optimizer.build_squad(ep)["squad"]
    ids = list(squad["id"])
    sell = dict(zip(squad["id"], squad["price"]))
    res = optimizer.plan_transfers(ep, ids, sell, bank=5.0, free_transfers=5)
    # must evaluate beyond the old hard cap of 3 so "not optimal" != "not tried"
    assert max(p["n_transfers"] for p in res["plans"]) >= 5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
