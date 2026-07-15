#!/usr/bin/env bash
# One-time CANFAR setup: install MSA environment + stage 443 GB training data
# Run from your laptop. Requires: canfar CLI authenticated to staging.
set -euo pipefail

canfar login cadc --dev || true
canfar server use staging

# The Python one-liner has ZERO spaces (including inside string literals).
# canfar joins args with spaces, skaha splits on spaces — so any space in
# the code would break.  os.system() fetches the remote script via URL and
# runs it through /bin/sh -c.
SETUP_URL="https://raw.githubusercontent.com/sfabbro/masked-stellar-autoencoder/main/scripts/canfar_remote_setup.sh"

echo "Creating setup session (CPU-only, for environment install + data staging)..."
canfar create -n msa-setup -c 8 -m 32 \
  headless astroai/webterm:latest \
  -- python3 -c "__import__('sys').exit(__import__('os').system(__import__('urllib.request',fromlist=['request']).urlopen('${SETUP_URL}').read().decode()))"

# Extract session ID via JSON (canfar ps has no -n name filter)
SETUP_ID=$(canfar ps -a --json | python3 -c "import sys,json;print([s['id'] for s in json.load(sys.stdin) if s.get('name')=='msa-setup'][0])")
echo ""
echo "Setup session: $SETUP_ID"
echo "Monitor progress: canfar logs $SETUP_ID"
echo "Wait for '=== SETUP COMPLETE ===' before running canfar_launch.sh"
