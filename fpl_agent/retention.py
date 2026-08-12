"""Retention: keep the audit trail useful without letting it rot the runs.

A 40-week season is ~280 daily runs. Nothing here was pruned before, which cost
two different things:

  * disk — a bootstrap snapshot is ~1.3MB, so a full season is ~400MB of
    near-duplicate JSON;
  * signal-to-noise — an EXPIRED signal file is still parsed on every run and
    still emits an "IGNORED" line into every report, so after three months the
    Key findings section is mostly dead notices.

The second one is the real damage, and it is why archiving beats deleting: the
evidence stays on disk for audit, it just stops shouting.

Every function here is idempotent and safe to call on each run.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from . import config, memoryio

SNAPSHOT_KEEP_DAYS = 14      # rolling window for day-to-day change detection
LOG_KEEP_DAYS = 30
ARCHIVE_DIRNAME = "archive"


def _day(name: str) -> str | None:
    """`bootstrap_2026-08-12.json` -> `2026-08-12`."""
    stem = Path(name).stem.replace(".meta", "")
    part = stem.split("_")[-1]
    try:
        datetime.strptime(part, "%Y-%m-%d")
    except ValueError:
        return None
    return part


def deadline_days(boot: dict) -> set[str]:
    """Days that must be kept forever: each gameweek's deadline date.

    A snapshot taken on a deadline day is the evidence of what was knowable when
    a decision was made. Pruning those would break the audit trail the whole
    forward test rests on.
    """
    days: set[str] = set()
    for ev in (boot or {}).get("events", []):
        dl = ev.get("deadline_time")
        if not dl:
            continue
        try:
            days.add(datetime.fromisoformat(
                str(dl).replace("Z", "+00:00")).strftime("%Y-%m-%d"))
        except ValueError:
            continue
    return days


def prune_snapshots(boot: dict | None = None, keep_days: int = SNAPSHOT_KEEP_DAYS,
                    now: datetime | None = None, dry_run: bool = False) -> list[str]:
    """Delete snapshots older than `keep_days`, except deadline-day ones.

    Returns the day stamps removed (or that would be, when dry_run).
    """
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    protected = deadline_days(boot or {})
    removed: list[str] = []
    for path in sorted(config.SNAPSHOT_DIR.glob("*.json")):
        day = _day(path.name)
        if day is None or day >= cutoff or day in protected:
            continue
        removed.append(day)
        if not dry_run:
            path.unlink(missing_ok=True)
    return sorted(set(removed))


def archive_expired_signals(now: datetime | None = None,
                            dry_run: bool = False) -> list[str]:
    """Move expired signal files into `signals/archive/`.

    Expiry already stops a signal from ACTING (memoryio.load_signals ignores it)
    but not from being parsed and reported every run. Moving it out of the glob
    is what actually stops the report noise.
    """
    now = now or datetime.now(timezone.utc)
    archive = memoryio.SIGNALS_DIR / ARCHIVE_DIRNAME
    moved: list[str] = []
    for path in sorted(memoryio.SIGNALS_DIR.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue          # malformed: leave it visible, it needs a human
        expiry = memoryio._signal_expiry(doc)
        if expiry is None or expiry >= now:
            continue
        moved.append(path.name)
        if not dry_run:
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(archive / path.name))
    return moved


def rotate_logs(log_dir: Path | None = None, keep_days: int = LOG_KEEP_DAYS,
                now: datetime | None = None, dry_run: bool = False) -> list[str]:
    """Drop routine run logs older than `keep_days`. The ledger is kept."""
    log_dir = log_dir or (config.ROOT / "routine" / "logs")
    if not log_dir.is_dir():
        return []
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y-%m-%d")
    removed: list[str] = []
    for path in sorted(log_dir.glob("*.log")):
        day = path.stem
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        if day >= cutoff:
            continue
        removed.append(path.name)
        if not dry_run:
            path.unlink(missing_ok=True)
    return removed


def run_all(boot: dict | None = None, now: datetime | None = None) -> dict:
    """Everything, once per run. Never raises: housekeeping must not break a run."""
    out: dict[str, list[str]] = {}
    for key, fn in (("snapshots_pruned", lambda: prune_snapshots(boot, now=now)),
                    ("signals_archived", lambda: archive_expired_signals(now=now)),
                    ("logs_rotated", lambda: rotate_logs(now=now))):
        try:
            out[key] = fn()
        except OSError as exc:            # disk trouble is not a reason to abort
            out[key] = []
            out.setdefault("errors", []).append(f"{key}: {exc}")
    return out
