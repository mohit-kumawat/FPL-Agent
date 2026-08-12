#!/bin/bash
# Install (or remove) the launchd job that runs the FPL agent on a schedule.
#
#   ./routine/install.sh              # install + load
#   ./routine/install.sh --uninstall  # unload + remove
#
# The plist is generated from wherever this repo currently lives, so re-run it
# after moving the folder. No secrets go in it -- the key stays in .env.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.fpl-agent.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [ "${1:-}" = "--uninstall" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
  exit 0
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/routine/logs"
chmod +x "$PROJECT_DIR/routine/run.sh"

# Fixed slots are just wake-ups: gate.py maps each one to a deadline-relative
# window (scan / decision / teamnews) or exits immediately at no token cost.
# 08:00 sits after FPL prices settle (~01:30 UTC); the spread of later slots
# exists so the decision and teamnews windows are hit wherever a deadline lands.
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PROJECT_DIR/routine/run.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT_DIR</string>
  <key>RunAtLoad</key><false/>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>13</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>16</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>19</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>21</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$PROJECT_DIR/routine/logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$PROJECT_DIR/routine/logs/launchd.err</string>
</dict>
</plist>
PLIST_EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "installed $LABEL"
echo "  plist:  $PLIST"
echo "  slots:  08/13/16/19/21 — gate maps each to scan/decision/teamnews or skips"
echo
launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|program" | head -3 || true
echo
echo "Test it now without waiting:  FPL_ROUTINE_FORCE=1 ./routine/run.sh"
