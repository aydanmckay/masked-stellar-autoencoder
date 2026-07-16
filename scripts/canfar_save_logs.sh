#!/usr/bin/env bash
# Save CANFAR session logs to local files before they expire.
# CANFAR retains logs for only 1 hour after job completion.
set -euo pipefail

SESSION_ID="${1:?Usage: canfar_save_logs.sh <session-id> [session-id ...]}"
mkdir -p logs

for SID in "$@"; do
  NAME=$(canfar ps -a --json 2>/dev/null \
    | python3 -c "import sys,json;print(next((s['name'] for s in json.load(sys.stdin) if s['id']=='$SID'),'unknown'))")
  OUTFILE="logs/${NAME}_${SID}_$(date +%Y%m%d_%H%M%S).log"
  canfar logs "$SID" 2>&1 > "$OUTFILE"
  LINES=$(wc -l < "$OUTFILE")
  echo "Saved $OUTFILE ($LINES lines)"
done
