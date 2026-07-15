#!/usr/bin/env bash
# Launch MSA pretrain + marimo monitor on CANFAR staging
# Run from your laptop after canfar_setup.sh has completed.
set -euo pipefail

canfar login cadc --dev || true
canfar server use staging

TRAIN_URL="https://raw.githubusercontent.com/sfabbro/masked-stellar-autoencoder/main/scripts/canfar_remote_train.sh"

# ── Training session (headless, GPU) ──
# Same zero-space Python one-liner trick as canfar_setup.sh.
echo "Launching training session (16 cores, 64 GB, 1 GPU)..."
canfar create -n msa-pretrain -c 16 -m 64 -g 1 \
  headless astroai/webterm:latest \
  -- python3 -c "__import__('sys').exit(__import__('os').system(__import__('urllib.request',fromlist=['request']).urlopen('${TRAIN_URL}').read().decode()))"

TRAIN_ID=$(canfar ps -a --json | python3 -c "import sys,json;print([s['id'] for s in json.load(sys.stdin) if s.get('name')=='msa-pretrain'][0])")

# ── Marimo monitor (contributed, no GPU) ──
echo "Launching marimo monitor..."
canfar create -n msa-monitor -c 2 -m 8 \
  contributed astroai/marimo:latest

MON_ID=$(canfar ps -a --json | python3 -c "import sys,json;print([s['id'] for s in json.load(sys.stdin) if s.get('name')=='msa-monitor'][0])")

echo ""
echo "=== MSA Pretrain on CANFAR Staging ==="
echo "Training: $TRAIN_ID"
echo "Monitor:  $MON_ID  ->  canfar open $MON_ID"
echo "Logs:     canfar logs $TRAIN_ID"
echo ""
echo "To stop both: canfar delete $TRAIN_ID $MON_ID"
