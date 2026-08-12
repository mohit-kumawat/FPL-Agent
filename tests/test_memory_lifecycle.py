"""Memory over a 40-week season: the handoff, the feedback loop, retention.

These pin the properties that make a long-running unattended season work — the
context passed forward stays bounded and informative, research gets scored, and
nothing accumulates without a policy.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fpl_agent import config, digest, memoryio, retention

NOW = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Never touch the real memory/ — these tests write ledgers."""
    for name in ("CALIBRATION_FILE", "SIGNAL_LOG_FILE", "SIGNAL_SCORES_FILE",
                 "DECISIONS_FILE", "RUNS_FILE", "STATE_FILE", "LEARNINGS_FILE"):
        monkeypatch.setattr(memoryio, name, tmp_path / name.lower())
    monkeypatch.setattr(memoryio, "SIGNALS_DIR", tmp_path / "signals")
    memoryio.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(digest, "CONTEXT_FILE", tmp_path / "current-context.md")
    monkeypatch.setattr(digest, "DIGEST_FILE", tmp_path / "digest.json")
    return tmp_path


def _boot(next_gw: int = 10, finished: int = 9) -> dict:
    events = [{"id": i, "finished": i <= finished, "is_current": i == finished,
               "is_next": i == next_gw,
               "deadline_time": f"2026-09-{(i % 28) + 1:02d}T11:00:00Z"}
              for i in range(1, 39)]
    return {"events": events, "chips": [
        {"name": "wildcard", "start_event": 2, "stop_event": 19},
        {"name": "wildcard", "start_event": 20, "stop_event": 38},
        {"name": "freehit", "start_event": 2, "stop_event": 19},
        {"name": "freehit", "start_event": 20, "stop_event": 38},
        {"name": "bboost", "start_event": 1, "stop_event": 19},
        {"name": "bboost", "start_event": 20, "stop_event": 38},
        {"name": "3xc", "start_event": 1, "stop_event": 19},
        {"name": "3xc", "start_event": 20, "stop_event": 38},
    ]}


SQUAD = {"players": [{"id": i, "purchase_price": 5.0} for i in range(15)],
         "bank": 0.5, "free_transfers": 1, "strategy_mode": "safe",
         "chips_available": ["wildcard1", "freehit1", "bboost1", "3xc1"]}


# ------------------------------------------------------- the context handoff
def test_context_stays_small_after_a_full_season_of_history():
    """The handoff is a prompt input. 40 weeks of ledgers must not grow it."""
    for gw in range(1, 39):
        memoryio.log_calibration({"gw": gw, "mae_all": 2.0, "mae_top50": 3.0, "n": 500})
        memoryio.log_decision({"date": f"2026-week{gw}", "action": "hold",
                               "captain": f"Player{gw}"})
    d = digest.build(_boot(), [], SQUAD, {})
    text = digest.render(d)
    assert len(text) < 4000, f"context grew to {len(text)} bytes"
    assert len(d["recent_decisions"]) <= digest.DECISION_HISTORY
    assert len(d["model_calibration"]["recent_mae"]) <= digest.CALIBRATION_HISTORY


def test_repeated_identical_decisions_collapse():
    """Several runs a day each log a decision; the handoff needs the shape of the
    history, not fifteen copies of the same hold."""
    for _ in range(15):
        memoryio.log_decision({"date": "2026-09-01", "action": "hold", "captain": "Salah"})
    memoryio.log_decision({"date": "2026-09-08", "action": "1_transfer", "captain": "Salah"})
    rows = digest.build(_boot(), [], SQUAD, {})["recent_decisions"]
    assert len(rows) == 2
    assert rows[0]["repeats"] == 15 and rows[1].get("repeats", 1) == 1
    assert "x15 runs" in digest.render(digest.build(_boot(), [], SQUAD, {}))


def test_context_carries_calibration_trend_not_just_a_number():
    for gw, mae in ((5, 3.0), (6, 2.5), (7, 2.0)):
        memoryio.log_calibration({"gw": gw, "mae_all": mae, "mae_top50": 4.0, "n": 500})
    d = digest.build(_boot(), [], SQUAD, {})
    assert d["model_calibration"]["mae_trend"] == -1.0
    assert "improving" in digest.render(d)


def test_split_expiry_is_quiet_early_and_loud_near_the_deadline():
    """Warning from GW1 would cry wolf for eighteen gameweeks."""
    early = digest.build(_boot(next_gw=3), [], SQUAD, {})
    near = digest.build(_boot(next_gw=17), [], SQUAD, {})
    assert early["chips"]["first_half_expiring"] == []
    assert near["chips"]["first_half_expiring"], "should warn 2 GWs from the split"
    assert "use-it-or-lose-it" in digest.render(near)


def test_expiry_warning_fires_on_the_last_playable_gameweek():
    """Windows are inclusive (start <= gw <= stop), so GW19 IS the last chance —
    the one gameweek the nag must not vanish."""
    last = digest.build(_boot(next_gw=19), [], SQUAD, {})
    assert last["chips"]["first_half_expiring"], "silent on the final chance"
    assert "last chance" in digest.render(last)


def test_unscored_pending_respects_state_scored_gws():
    """A GW with no stored prediction is retired via state, never reaching
    calibration.jsonl — it must not haunt the context as 'awaiting' forever."""
    d = digest.build(_boot(next_gw=10, finished=9), [], SQUAD,
                     {"scored_gws": list(range(1, 10))})
    assert d["model_calibration"]["unscored_pending"] == []


def test_awaiting_scoring_renders_gameweeks_not_a_python_list():
    d = digest.build(_boot(next_gw=4, finished=3), [], SQUAD, {})
    line = next(l for l in digest.render(d).splitlines() if "Awaiting" in l)
    assert "GW1, GW2, GW3" in line and "[" not in line


def test_blocked_verification_still_reaches_the_next_run():
    d = digest.build(_boot(), [], SQUAD, {},
                     verification={"blockers": ["squad.yaml has 14 players"],
                                   "warnings": []})
    assert "BLOCKER" in digest.render(d)


def test_digest_write_produces_both_artifacts():
    js, md = digest.write(digest.build(_boot(), [], SQUAD, {}))
    assert json.loads(open(js).read())["season"] == config.CURRENT_SEASON
    assert "Operating context" in open(md).read()


# --------------------------------------------------- the signal feedback loop
def _panel(minutes: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame([{"element": pid, "minutes": m, "round": 5}
                         for pid, m in minutes.items()])


def test_minutes_claims_are_scored_against_reality():
    frame = pd.DataFrame(
        {"xmins_min": [0.85, None], "xmins_max": [None, 0.35],
         "ep_per_gw": [0.0, 0.0], "sources": ["press.yaml", "rumour.yaml"]},
        index=[7, 9])
    memoryio.log_applied_signals(5, frame)
    # 7 was claimed a starter and played 90 (right); 9 was capped and played 90 (wrong)
    res = memoryio.score_signals(5, _panel({7: 90.0, 9: 90.0}))
    assert res["hits"] == 1 and res["misses"] == 1
    card = memoryio.signal_scorecard()
    assert card["accuracy"] == 0.5
    assert card["least_reliable"] == [] or card["least_reliable"][0]["source"]


def test_a_starter_claim_survives_an_late_substitution():
    """Tolerance matters: "starts" is not falsified by an 80th-minute hook."""
    frame = pd.DataFrame({"xmins_min": [0.85], "xmins_max": [None],
                          "ep_per_gw": [0.0], "sources": ["press.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, frame)
    assert memoryio.score_signals(5, _panel({7: 78.0}))["hits"] == 1


def test_signals_are_scored_once_and_only_once():
    frame = pd.DataFrame({"xmins_min": [0.85], "xmins_max": [None],
                          "ep_per_gw": [0.0], "sources": ["a.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, frame)
    memoryio.log_applied_signals(5, frame)          # a second run, same gameweek
    assert memoryio.score_signals(5, _panel({7: 90.0}))["claims"] == 1
    assert memoryio.score_signals(5, _panel({7: 90.0})) is None


def test_ep_only_signals_are_not_scored_as_minutes_claims():
    frame = pd.DataFrame({"xmins_min": [None], "xmins_max": [None],
                          "ep_per_gw": [0.5], "sources": ["role.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, frame)
    assert memoryio.score_signals(5, _panel({7: 90.0})) is None


def test_a_revised_claim_supersedes_the_original_for_scoring():
    """Rotation risk on Monday, ruled out at Friday's presser: the grade must
    land on the claim the advice actually rested on, not the stale one."""
    monday = pd.DataFrame({"xmins_min": [0.85], "xmins_max": [None],
                           "ep_per_gw": [0.0], "sources": ["monday.yaml"]}, index=[7])
    friday = pd.DataFrame({"xmins_min": [None], "xmins_max": [0.05],
                           "ep_per_gw": [0.0], "sources": ["friday.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, monday)
    memoryio.log_applied_signals(5, friday)
    res = memoryio.score_signals(5, _panel({7: 0.0}))
    assert res["claims"] == 1 and res["hits"] == 1 and res["misses"] == 0
    assert res["by_source"] == {"friday.yaml": {"hit": 1, "miss": 0}}


def test_double_gameweek_minutes_do_not_falsify_a_per_match_cap():
    """120 of a possible 180 is UNDER a 0.70 rotation cap — two 60-minute
    outings must not be graded against a single 90."""
    frame = pd.DataFrame({"xmins_min": [None], "xmins_max": [0.70],
                          "ep_per_gw": [0.0], "sources": ["press.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, frame)
    dgw = pd.DataFrame([{"element": 7, "minutes": 60.0, "round": 5},
                        {"element": 7, "minutes": 60.0, "round": 5}])
    assert memoryio.score_signals(5, dgw)["hits"] == 1


def test_a_source_with_no_misses_is_never_least_reliable():
    frame = pd.DataFrame({"xmins_min": [0.85, 0.85], "xmins_max": [None, None],
                          "ep_per_gw": [0.0, 0.0],
                          "sources": ["gold.yaml", "gold.yaml"]}, index=[7, 8])
    memoryio.log_applied_signals(5, frame)
    memoryio.score_signals(5, _panel({7: 90.0, 8: 90.0}))
    assert memoryio.signal_scorecard()["least_reliable"] == []


def test_a_player_missing_from_the_panel_is_unscored_not_wrong():
    frame = pd.DataFrame({"xmins_min": [0.85], "xmins_max": [None],
                          "ep_per_gw": [0.0], "sources": ["a.yaml"]}, index=[7])
    memoryio.log_applied_signals(5, frame)
    res = memoryio.score_signals(5, _panel({99: 90.0}))
    assert res["unscored"] == 1 and res["hits"] == 0 and res["misses"] == 0


# ----------------------------------------------------------------- retention
def test_expired_signals_stop_polluting_every_report():
    """Measured before the fix: 24 dead files produced 24 IGNORED lines in every
    report, forever, while contributing nothing."""
    for i in range(6):
        (memoryio.SIGNALS_DIR / f"old{i}.yaml").write_text(
            "date: 2026-08-01\nttl_days: 7\nadjustments:\n"
            f"  - player_id: {i}\n    role: expected_starter\n")
    (memoryio.SIGNALS_DIR / "live.yaml").write_text(
        "date: 2026-11-30\nttl_days: 14\n"
        "evidence:\n  tier: 1\n  url: https://club.com/news\n"
        "  publisher: club\n  published_at: 2026-11-30T10:00:00Z\n"
        "adjustments:\n"
        "  - player_id: 99\n    role: expected_starter\n")

    _, before = memoryio.load_signals(now=NOW)
    assert sum(1 for n in before if n.get("problems")) == 6

    moved = retention.archive_expired_signals(now=NOW)
    assert len(moved) == 6
    frame, after = memoryio.load_signals(now=NOW)
    assert sum(1 for n in after if n.get("problems")) == 0
    assert 99 in frame.index, "the live signal must survive archiving"
    assert (memoryio.SIGNALS_DIR / "archive" / "old0.yaml").exists(), "kept for audit"


def test_malformed_signals_are_left_visible_for_a_human():
    (memoryio.SIGNALS_DIR / "broken.yaml").write_text("{{ not yaml")
    assert retention.archive_expired_signals(now=NOW) == []
    assert (memoryio.SIGNALS_DIR / "broken.yaml").exists()


def test_snapshots_prune_but_deadline_days_are_kept_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snaps")
    config.SNAPSHOT_DIR.mkdir()
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for i in range(120):
        day = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        (config.SNAPSHOT_DIR / f"bootstrap_{day}.json").write_text("{}")
    boot = _boot()
    now = start + timedelta(days=119)
    retention.prune_snapshots(boot, now=now)

    kept = {retention._day(p.name) for p in config.SNAPSHOT_DIR.glob("*.json")}
    cutoff = (now - timedelta(days=retention.SNAPSHOT_KEEP_DAYS)).strftime("%Y-%m-%d")
    assert all(d >= cutoff or d in retention.deadline_days(boot) for d in kept)
    assert len(kept) < 120, "nothing was pruned"
    # every deadline day that existed on disk survives — the audit trail
    for d in retention.deadline_days(boot):
        if d < cutoff and (start <= datetime.strptime(d, "%Y-%m-%d").replace(
                tzinfo=timezone.utc) <= now):
            assert d in kept, f"deadline snapshot {d} was pruned"


def test_logs_rotate_by_age(tmp_path):
    for day in ("2026-08-01", "2026-11-29"):
        (tmp_path / f"{day}.log").write_text("x")
    (tmp_path / "runs.jsonl").write_text("{}")      # the ledger is not a log
    removed = retention.rotate_logs(tmp_path, now=NOW)
    assert removed == ["2026-08-01.log"]
    assert (tmp_path / "runs.jsonl").exists()


def test_run_all_never_raises_on_a_bad_directory(monkeypatch):
    monkeypatch.setattr(config, "SNAPSHOT_DIR", config.ROOT / "does-not-exist")
    assert isinstance(retention.run_all({}), dict)


# ----------------------------------------------------- learnings discipline
def test_learnings_entries_need_a_date_and_validated_needs_evidence():
    bad = ("# L\n## VALIDATED (x)\n- a claim with no date and no evidence\n"
           "## HYPOTHESES (x)\n- an undated hunch\n")
    problems = " ".join(memoryio.validate_learnings(bad))
    assert "no YYYY-MM-DD date" in problems
    assert "no evidence pointer" in problems


def test_multi_line_entries_are_validated_as_one_entry():
    """The date or the evidence often sits on a continuation line."""
    ok = ("# L\n## VALIDATED (x)\n- **Thing** (2026-08-12). Detail continues\n"
          "  here citing eval/ablation-report.md for the numbers.\n")
    assert memoryio.validate_learnings(ok) == []


def test_process_rules_section_is_exempt():
    text = ("# L\n## VALIDATED (x)\n- ok (2026-08-12) see eval/report.md\n"
            "## Process rules\n- rerun the eval suite after model changes\n")
    assert memoryio.validate_learnings(text) == []


def test_tier_caps_force_pruning():
    entries = "\n".join(f"- claim {i} (2026-08-12) eval/report.md" for i in range(30))
    problems = " ".join(memoryio.validate_learnings(f"# L\n## VALIDATED (x)\n{entries}\n"))
    assert "cap" in problems


def test_the_repo_learnings_file_is_compliant():
    """A validator whose baseline is all-red trains everyone to ignore it."""
    real = (config.ROOT / "memory" / "learnings.md").read_text()
    assert memoryio.validate_learnings(real) == []
