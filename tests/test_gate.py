"""Tests for routine/gate.py — the scheduler gate the unattended routine trusts.

The gate runs under bare system python3 with no package context, so import it
by file path. Every test drives main() end-to-end: state file + ledger in, exit
code + stdout line out. That is the exact contract run.sh consumes.
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gate", Path(__file__).resolve().parent.parent / "routine" / "gate.py")
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def _write_state(path: Path, hours_to_deadline: float | None, gw: int = 1,
                 has_team: bool = False) -> None:
    stage: dict = {"next_gw": gw, "has_team": has_team}
    if hours_to_deadline is not None:
        dl = datetime.now(timezone.utc) + timedelta(hours=hours_to_deadline)
        stage["next_deadline"] = dl.isoformat().replace("+00:00", "Z")
    path.write_text(json.dumps({"stage": stage}))


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _run(tmp_path, monkeypatch, capsys, hours: float | None, gw: int = 1,
         has_team: bool = False, ledger: list[dict] | None = None) -> tuple[int, str]:
    monkeypatch.setattr(gate, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(gate, "LEDGER", tmp_path / "logs" / "runs.jsonl")
    monkeypatch.delenv("FPL_ROUTINE_FORCE", raising=False)
    _write_state(gate.STATE, hours, gw=gw, has_team=has_team)
    if ledger is not None:
        _write_ledger(gate.LEDGER, ledger)
    with pytest.raises(SystemExit) as exc:
        gate.main()
    return exc.value.code, capsys.readouterr().out.strip()


def _ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# --------------------------------------------------------- window classification
def test_far_deadline_classifies_as_scan(tmp_path, monkeypatch, capsys):
    code, out = _run(tmp_path, monkeypatch, capsys, hours=200)
    assert code == 0
    assert out.split("|")[2] == "scan"


def test_decision_window_between_4_and_30_hours(tmp_path, monkeypatch, capsys):
    code, out = _run(tmp_path, monkeypatch, capsys, hours=20)
    assert code == 0
    assert out.split("|")[2] == "decision"


def test_teamnews_window_inside_4_hours(tmp_path, monkeypatch, capsys):
    code, out = _run(tmp_path, monkeypatch, capsys, hours=2)
    assert code == 0
    assert out.split("|")[2] == "teamnews"


def test_last_half_hour_is_too_late_to_act(tmp_path, monkeypatch, capsys):
    code, _ = _run(tmp_path, monkeypatch, capsys, hours=0.3)
    assert code == gate.SKIP


def test_missing_state_still_runs_a_scan(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "STATE", tmp_path / "absent.json")
    monkeypatch.setattr(gate, "LEDGER", tmp_path / "logs" / "runs.jsonl")
    monkeypatch.delenv("FPL_ROUTINE_FORCE", raising=False)
    with pytest.raises(SystemExit) as exc:
        gate.main()
    out = capsys.readouterr().out.strip()
    assert exc.value.code == 0
    assert out.split("|")[2] == "scan"


def test_past_deadline_is_stale_state_not_a_stop(tmp_path, monkeypatch, capsys):
    """A deadline behind us means the state file needs refreshing; the gate must
    let a run through to do that refresh, never wait for one it blocks."""
    code, out = _run(tmp_path, monkeypatch, capsys, hours=-30)
    assert code == 0
    assert out.split("|")[2] == "scan"


def test_mode_flips_to_weekly_once_squad_exists(tmp_path, monkeypatch, capsys):
    _, out_gw1 = _run(tmp_path, monkeypatch, capsys, hours=20, has_team=False)
    _, out_weekly = _run(tmp_path, monkeypatch, capsys, hours=20, has_team=True)
    assert out_gw1.split("|")[0] == "gw1"
    assert out_weekly.split("|")[0] == "weekly"


# ----------------------------------------------------------------- dedup scoping
def test_done_window_is_not_repeated(tmp_path, monkeypatch, capsys):
    ledger = [{"gw": 1, "window": "decision", "status": "ok", "started": _ago(2)}]
    code, _ = _run(tmp_path, monkeypatch, capsys, hours=20, ledger=ledger)
    assert code == gate.SKIP


def test_old_ok_row_cannot_mask_a_new_season(tmp_path, monkeypatch, capsys):
    """Last season's GW1 decision row is months old — recency scoping must let
    this season's GW1 decision run."""
    ledger = [{"gw": 1, "window": "decision", "status": "ok",
               "started": _ago(24 * 300)}]
    code, out = _run(tmp_path, monkeypatch, capsys, hours=20, ledger=ledger)
    assert code == 0
    assert out.split("|")[2] == "decision"


def test_other_gameweek_rows_do_not_match(tmp_path, monkeypatch, capsys):
    ledger = [{"gw": 0, "window": "decision", "status": "ok", "started": _ago(1)}]
    code, _ = _run(tmp_path, monkeypatch, capsys, hours=20, gw=1, ledger=ledger)
    assert code == 0


def test_scan_cadence_suppresses_within_interval(tmp_path, monkeypatch, capsys):
    ledger = [{"gw": 1, "window": "scan", "status": "ok", "started": _ago(10)}]
    code, _ = _run(tmp_path, monkeypatch, capsys, hours=200, ledger=ledger)
    assert code == gate.SKIP


def test_scan_fires_again_after_interval(tmp_path, monkeypatch, capsys):
    ledger = [{"gw": 1, "window": "scan", "status": "ok",
               "started": _ago(gate.SCAN_EVERY_H + 1)}]
    code, out = _run(tmp_path, monkeypatch, capsys, hours=200, ledger=ledger)
    assert code == 0
    assert out.split("|")[2] == "scan"


# ------------------------------------------------------------- retries & safety
def test_failures_retry_then_give_up(tmp_path, monkeypatch, capsys):
    fails = [{"gw": 1, "window": "decision", "status": "failed", "started": _ago(i + 1)}
             for i in range(gate.MAX_ATTEMPTS - 1)]
    code, out = _run(tmp_path, monkeypatch, capsys, hours=20, ledger=fails)
    assert code == 0 and "retry" in out

    fails.append({"gw": 1, "window": "decision", "status": "failed", "started": _ago(0.5)})
    code, _ = _run(tmp_path, monkeypatch, capsys, hours=20, ledger=fails)
    assert code == gate.SKIP


def test_corrupt_ledger_lines_are_ignored(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(gate, "LEDGER", tmp_path / "logs" / "runs.jsonl")
    monkeypatch.delenv("FPL_ROUTINE_FORCE", raising=False)
    _write_state(gate.STATE, hours_to_deadline=20)
    gate.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    gate.LEDGER.write_text("this is not json\n{broken\n")
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 0


def test_force_env_overrides_everything(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(gate, "LEDGER", tmp_path / "logs" / "runs.jsonl")
    _write_state(gate.STATE, hours_to_deadline=20)
    _write_ledger(gate.LEDGER, [{"gw": 1, "window": "decision", "status": "ok",
                                 "started": _ago(1)}])
    monkeypatch.setenv("FPL_ROUTINE_FORCE", "1")
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 0
    assert "forced" in capsys.readouterr().out
