"""Guards on the shared eval season loop.

These are structural, not numeric: the real numbers need the historical parquet
files (network on a cold cache), so the value here is pinning the contract that
made three copies of this loop drift in the first place — every arm charges hits
the same way, and the caches can never straddle a monkeypatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
season_loop = pytest.importorskip("season_loop")

from fpl_agent import config, replay  # noqa: E402


def test_all_eval_arms_share_one_loop():
    """strategy_sim, agent_backtest and ablation must call the shared loop —
    a local copy is how the free-transfer cap and hit timing drifted before."""
    eval_dir = Path(__file__).resolve().parent.parent / "eval"
    for name in ("strategy_sim.py", "agent_backtest.py", "ablation.py"):
        src = (eval_dir / name).read_text()
        assert "run_season" in src, f"{name} does not use the shared loop"
        # the giveaway of a re-typed loop: its own transfer bookkeeping
        assert "plan_transfers" not in src, f"{name} re-implements the season loop"


def test_free_transfer_cap_comes_from_config():
    """Regression: one copy hardcoded min(5, ...) while the others used the
    constant, so raising the cap would have silently applied to some arms only."""
    src = (Path(__file__).resolve().parent.parent / "eval" / "season_loop.py").read_text()
    assert "config.MAX_FREE_TRANSFERS" in src
    assert "min(5," not in src


def test_weekly_series_carry_hits_where_they_were_taken():
    """weekly_net must equal weekly_autosub minus that GW's hit, so the per-GW
    bootstrap sees the cost in the right week while totals stay unchanged."""
    src = (Path(__file__).resolve().parent.parent / "eval" / "season_loop.py").read_text()
    assert "weekly_net.append(sub - hit_now)" in src


def test_clear_caches_resets_both_layers():
    """The leak audit depends on this: a frame cached before a monkeypatch must
    never be served to patched code, or a real hindsight leak reads as clean."""
    season_loop.clear_caches()
    assert season_loop._ep_cached.cache_info().currsize == 0
    assert replay._season_gws_cached.cache_info().currsize == 0


def test_season_gws_returns_an_independent_copy():
    """The cache hands out copies — the leak audit mutates the frame it gets."""
    if not (config.HISTORY_DIR / "merged_gw_2024-25.parquet").exists():
        pytest.skip("historical parquet not cached locally")
    a = replay._season_gws("2024-25")
    a["total_points"] = -999
    b = replay._season_gws("2024-25")
    assert (b["total_points"] != -999).any()
