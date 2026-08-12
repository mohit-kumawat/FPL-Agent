#!/bin/bash
# Scheduled FPL agent run. Invoked by launchd; safe to run by hand.
#
#   ./routine/run.sh                # respects the gate (window-based, see gate.py)
#   FPL_ROUTINE_FORCE=1 ./routine/run.sh
#
# Two-part design: until squad.yaml has a team the run uses gw1_prompt.md (build
# the initial squad); after that weekly_prompt.md (hold-by-default transfer
# advice). The gate maps each wake-up to a window -- scan / decision / teamnews
# -- so a gameweek gets at most three runs, all deadline-relative.
#
# What it does: gate -> lock -> preflight -> headless `claude` following
# AGENT.md -> hash-lock the recommendation (decision/teamnews) -> rebuild the
# HTML view. The agent writes its own brief; if the run dies first, this script
# writes a failure brief so a silent death still shows up in the view.

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# launchd hands over a near-empty PATH; everything below must be reachable.
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

TODAY="$(date +%Y-%m-%d)"
STARTED_ISO="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
LOG_DIR="$PROJECT_DIR/routine/logs"
BRIEF_DIR="$PROJECT_DIR/memory/briefs"
LOG="$LOG_DIR/$TODAY.log"
LEDGER="$LOG_DIR/runs.jsonl"
LOCK="$PROJECT_DIR/.routine.lock"
mkdir -p "$LOG_DIR" "$BRIEF_DIR"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOG"; }

# `started` is what retry logic measures against -- the run's start, not its
# finish. `note` is a failure reason and nothing else.
ledger() {  # status, seconds, note
  printf '{"ts":"%s","started":"%s","date":"%s","gw":%s,"window":"%s","brief":"%s","status":"%s","seconds":%s,"note":"%s"}\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$STARTED_ISO" "$TODAY" "$GW" "$WINDOW" \
    "$BRIEF_STEM" "$1" "$2" "${3:-}" >> "$LEDGER"
}

# Did this run produce a brief? Existence alone isn't enough -- an earlier
# window's brief may share the path, and scoring that as success would present
# stale advice as a fresh pre-deadline check.
brief_is_fresh() {  # since-epoch
  [ -f "$BRIEF" ] || return 1
  [ "$(stat -f %m "$BRIEF" 2>/dev/null || echo 0)" -ge "$1" ]
}

fail_brief() {  # reason
  # Only if the agent never got one written -- never clobber real output.
  [ -f "$BRIEF" ] && return 0
  cat > "$BRIEF" <<EOF
# $TODAY — routine failed

**Status:** blocked

## What changed
The scheduled run did not complete. No research was done and no recommendation
was produced. Treat this window as if the agent never ran.

## Needs you
$1

It retries at the next scheduled slot (up to 3 attempts per window), then gives
up on the window — a run of red briefs means something needs fixing by hand.

Full output: \`routine/logs/$TODAY.log\`
EOF
}

# Every exit after the log exists goes through here. Bailing out without
# rebuilding leaves the view showing the last good day, which is the one thing
# this layer promises never to do.
finish() {  # exit-code
  python3 "$PROJECT_DIR/routine/build_view.py" >> "$LOG" 2>&1 \
    && log "view rebuilt: routine/view.html" \
    || log "warning: view rebuild failed"
  exit "$1"
}

# --- gate ------------------------------------------------------------------
# Exit 0 -> run, stdout is `mode|gw|window|reason`. Only 78 means "deliberately
# skip". Anything else is the gate itself being broken and must fail open -- a
# gate that can't decide has to be louder than a quiet day, not quieter.
GATE_OUT="$(python3 "$PROJECT_DIR/routine/gate.py" 2>&1)"; GATE_RC=$?
if [ $GATE_RC -eq 78 ]; then
  log "skip: $GATE_OUT"
  exit 0
elif [ $GATE_RC -ne 0 ]; then
  log "gate check failed (exit $GATE_RC), running anyway: $GATE_OUT"
  GATE_OUT="weekly|0|scan|gate broken, failed open"
fi
MODE="$(printf '%s' "$GATE_OUT" | cut -d'|' -f1)"
GW="$(printf '%s' "$GATE_OUT" | cut -d'|' -f2)"
WINDOW="$(printf '%s' "$GATE_OUT" | cut -d'|' -f3)"
GATE_REASON="$(printf '%s' "$GATE_OUT" | cut -d'|' -f4-)"
case "$MODE" in gw1|weekly) ;; *) MODE="weekly"; GW=0; WINDOW="scan";; esac
case "$GW" in ''|*[!0-9]*) GW=0;; esac

PROMPT_FILE="$PROJECT_DIR/routine/${MODE}_prompt.md"
BRIEF_STEM="$TODAY-$WINDOW"
BRIEF="$BRIEF_DIR/$BRIEF_STEM.md"

# --- lock (mkdir is atomic; macOS has no flock) ----------------------------
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER="$(cat "$LOCK/pid" 2>/dev/null || echo '?')"
  if [ "$OWNER" != "?" ] && ! kill -0 "$OWNER" 2>/dev/null; then
    log "clearing stale lock from pid $OWNER"
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || { log "abort: lock contended"; exit 0; }
  else
    log "abort: another run is in progress (pid $OWNER)"
    exit 0
  fi
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM

# --- credentials -----------------------------------------------------------
# .env is gitignored. Keep the key there, never in the plist or in git.
# Read the one variable rather than sourcing the file: sourcing would let a
# stray PATH= line clobber the launchd-safe PATH set above, and would hand
# every other secret in .env to a subprocess that has WebFetch and Write.
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$PROJECT_DIR/.env" ]; then
  ANTHROPIC_API_KEY="$(sed -n 's/^[[:space:]]*ANTHROPIC_API_KEY[[:space:]]*=[[:space:]]*//p' \
    "$PROJECT_DIR/.env" | head -1 | tr -d '\r' | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
  export ANTHROPIC_API_KEY
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  log "abort: ANTHROPIC_API_KEY not set (put it in $PROJECT_DIR/.env)"
  fail_brief "No \`ANTHROPIC_API_KEY\`. Add it to \`.env\` in the project root — see \`routine/README.md\`."
  ledger "no-credentials" 0 "ANTHROPIC_API_KEY missing"
  finish 1
fi

# The prompt is piped in, and a command group's status is its *last* command's,
# so a missing prompt file would otherwise vanish: the agent would get nothing
# but the trailing path line, write some brief, and the window would be
# recorded as a success. claude validates AGENT.md itself; this file has no
# such backstop.
if [ ! -r "$PROMPT_FILE" ]; then
  log "abort: $PROMPT_FILE is missing or unreadable"
  fail_brief "\`routine/${MODE}_prompt.md\` is missing or unreadable, so there was nothing to send. Restore it (or \`git checkout routine/\`)."
  ledger "no-prompt" 0 "${MODE}_prompt.md unreadable"
  finish 1
fi

# --- preflight -------------------------------------------------------------
# One cheap probe before committing the slot. A key can be present and still be
# dead -- expired, revoked, out of credit -- and unattended there is nobody to
# notice. Costs a couple of tokens. FPL_ROUTINE_SKIP_PREFLIGHT=1 bypasses it.
# The FPL API needs no equivalent: verify.py already blocks the pipeline with
# its own message when that data is missing or stale.
if [ "${FPL_ROUTINE_SKIP_PREFLIGHT:-0}" != "1" ]; then
  if ! printf 'ok' | claude --bare --print --output-format text >/dev/null 2>>"$LOG"; then
    log "abort: credential probe failed"
    fail_brief "The Anthropic API rejected the key or could not be reached, so the run never started. Check \`ANTHROPIC_API_KEY\` in \`.env\` — if it's a personal key, confirm it's still active at console.anthropic.com."
    ledger "bad-credentials" 0 "credential probe failed"
    finish 1
  fi
fi

# --- run -------------------------------------------------------------------
log "start [$MODE gw$GW $WINDOW] ($GATE_REASON)"
START=$(date +%s)

# --bare: auth is strictly ANTHROPIC_API_KEY; keychain, hooks, plugins and
# CLAUDE.md discovery are all skipped, so the run is hermetic and independent
# of any interactive login. The prompt goes in on stdin, not as an argument:
# --allowedTools is variadic, so a trailing positional would be parsed as one
# more tool name. The brief path is appended rather than left to the agent --
# it would otherwise pick the date itself. jq is allowed because it is the only
# workable way to resolve FPL element ids out of the ~1.3MB single-line
# bootstrap snapshot; without it the agent has to guess ids.
{ cat "$PROMPT_FILE"
  printf '\nThis is a %s-window run for GW%s.\n' "$WINDOW" "$GW"
  printf 'Write the brief to exactly this path: %s\n' "$BRIEF"
} | claude \
  --bare \
  --print \
  --output-format text \
  --add-dir "$PROJECT_DIR" \
  --append-system-prompt-file "$PROJECT_DIR/AGENT.md" \
  --allowedTools \
      "Bash(uv run fpl *)" "Bash(jq *)" "Bash(ls *)" "Bash(cat *)" "Bash(head *)" \
      "Bash(tail *)" "Bash(grep *)" "Bash(date *)" \
      "Read" "Write" "Edit" "Glob" "Grep" "WebSearch" "WebFetch" \
  --permission-mode dontAsk \
  >> "$LOG" 2>&1
STATUS=$?
ELAPSED=$(( $(date +%s) - START ))

if [ $STATUS -ne 0 ]; then
  log "FAILED (exit $STATUS, ${ELAPSED}s)"
  fail_brief "The agent exited with status $STATUS. Check the log for the cause — an expired or rejected API key is the usual one."
  ledger "failed" "$ELAPSED" "exit $STATUS"
elif ! brief_is_fresh "$START"; then
  log "ran but wrote no brief (${ELAPSED}s)"
  fail_brief "The agent finished without writing a brief. Its output is in the log — worth a look, the prompt may need tightening."
  ledger "no-brief" "$ELAPSED" "completed without brief"
else
  log "ok (${ELAPSED}s)"
  ledger "ok" "$ELAPSED"
  # Seal the recommendation so the forward test can't be quietly revised.
  # Scan runs are research, not recommendations -- nothing to lock.
  if [ "$WINDOW" != "scan" ] && [ "${FPL_ROUTINE_SKIP_LOCK:-0}" != "1" ]; then
    uv run python "$PROJECT_DIR/eval/phase3_prereg.py" lock >> "$LOG" 2>&1 \
      && log "recommendation hash-locked (eval/phase3-predictions.jsonl)" \
      || log "warning: phase3 lock failed (run continues; forward test loses this point)"
  fi
fi

finish $STATUS
