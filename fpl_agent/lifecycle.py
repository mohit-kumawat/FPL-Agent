"""GW lifecycle stage machine + pending-items tracking.

Gives any operator (human or agent) an immediate answer to:
  "where are we in the gameweek, what is done, what is pending, what's next?"

Stages:
  PRESEASON        no team saved yet, before GW1 deadline
  PLANNING         >72h to next deadline — research window
  DEADLINE_SOON    24-72h — decide transfers/captain
  DEADLINE_IMMINENT<24h — final checks, act now
  GW_LIVE          deadline passed, matches not finished
  POST_GW          GW finished, results in — calibrate, plan next
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config, memoryio

PENDING_FILE = config.MEMORY_DIR / "pending.json"


# ---------------------------------------------------------------- stage
def current_stage(boot: dict, squad: dict) -> dict:
    events = boot["events"]
    nxt = next((e for e in events if e["is_next"]), None)
    cur = next((e for e in events if e["is_current"]), None)
    has_team = len(squad.get("players") or []) == config.SQUAD_SIZE

    now = datetime.now(timezone.utc)
    hrs = None
    if nxt:
        dl = datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
        hrs = (dl - now).total_seconds() / 3600

    if cur and not cur["finished"]:
        stage = "GW_LIVE"
    elif cur and cur["finished"] and not cur.get("data_checked"):
        stage = "POST_GW"
    elif not has_team and (nxt is None or nxt["id"] == 1):
        stage = "PRESEASON"
    elif hrs is None:
        stage = "SEASON_OVER"
    elif hrs <= 24:
        stage = "DEADLINE_IMMINENT"
    elif hrs <= 72:
        stage = "DEADLINE_SOON"
    else:
        stage = "PLANNING"

    return {
        "stage": stage,
        "next_gw": nxt["id"] if nxt else None,
        "next_deadline": nxt["deadline_time"] if nxt else None,
        "hours_to_deadline": round(hrs, 1) if hrs is not None else None,
        "current_gw": cur["id"] if cur else None,
        "has_team": has_team,
    }


# --------------------------------------------------------- stage playbook
_PLAYBOOK = {
    "PRESEASON": [
        "research: confirmed starters, new-signing roles, pen takers (web + signals/)",
        "run `fpl build` and review the proposed 15",
        "enter squad on FPL site before GW1 deadline (owner's go-ahead required)",
        "seed squad.yaml with purchase prices + entry_id after saving",
    ],
    "PLANNING": [
        "monitor price-change momentum on watchlist and owned players",
        "web research on flagged injuries; write signals/ notes",
        "review fixture swings 3-6 GWs out for transfer planning",
    ],
    "DEADLINE_SOON": [
        "decide transfers (report's recommendation + your judgment)",
        "confirm captain and bench order",
        "check chip windows (DGW/BGW ahead?)",
    ],
    "DEADLINE_IMMINENT": [
        "final team-news sweep (press conferences, lineups leaks)",
        "rerun `fpl daily --force` if any owned/target player is flagged",
        "execute confirmed moves before deadline",
    ],
    "GW_LIVE": [
        "no actions possible — observe, note surprises in memory/learnings.md",
    ],
    "POST_GW": [
        "pipeline auto-calibrates predictions once data_checked",
        "update squad.yaml if transfers/price changes settled",
        "review decisions.jsonl vs outcome; add learnings",
    ],
    "SEASON_OVER": ["archive the season; run retrospective"],
}


def stage_actions(stage: str) -> list[str]:
    return _PLAYBOOK.get(stage, [])


# ------------------------------------------------------------ pending items
def load_pending() -> list[dict]:
    if PENDING_FILE.exists():
        return json.loads(PENDING_FILE.read_text())
    return []


def save_pending(items: list[dict]) -> None:
    PENDING_FILE.write_text(json.dumps(items, indent=2))


def add_pending(text: str, due: str | None = None) -> None:
    items = load_pending()
    if any(i["text"] == text and not i.get("done") for i in items):
        return
    items.append({"text": text, "added": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                  "due": due, "done": False})
    save_pending(items)


def complete_pending(text_substr: str) -> bool:
    items = load_pending()
    hit = False
    for i in items:
        if text_substr.lower() in i["text"].lower() and not i.get("done"):
            i["done"] = True
            i["completed"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            hit = True
    save_pending(items)
    return hit


def open_pending() -> list[dict]:
    return [i for i in load_pending() if not i.get("done")]


def sync_stage_pending(stage_info: dict, squad: dict) -> None:
    """Auto-seed pending items implied by the stage/state."""
    if stage_info["stage"] == "PRESEASON":
        add_pending("Enter GW1 squad on FPL site", due=stage_info["next_deadline"])
        add_pending("Seed squad.yaml with purchase prices after saving squad")
        add_pending("Store entry_id in squad.yaml after first save")
    if squad.get("entry_id"):
        complete_pending("Store entry_id")
    if len(squad.get("players") or []) == config.SQUAD_SIZE:
        complete_pending("Enter GW1 squad")
        complete_pending("Seed squad.yaml")


def status(boot: dict) -> dict:
    """One-call status object: stage + playbook + pending + last decision."""
    squad = memoryio.load_squad()
    st = current_stage(boot, squad)
    sync_stage_pending(st, squad)
    return {
        **st,
        "stage_playbook": stage_actions(st["stage"]),
        "pending": open_pending(),
        "last_decision": memoryio.last_decision(),
    }
