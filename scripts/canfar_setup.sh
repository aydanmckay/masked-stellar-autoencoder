#!/usr/bin/env bash
# One-time CANFAR setup: install MSA environment + stage 443 GB training data
# Run from your laptop. Requires: canfar CLI authenticated to staging.
set -euo pipefail

canfar login cadc --dev || true
canfar server use staging

echo "Creating setup session (CPU-only, for environment install + data staging)..."
canfar create -n msa-setup -c 8 -m 32 \
  headless astroai/webterm:latest \
  -- bash -c 'cd /srcdir && git clone --depth 1 https://github.com/sfabbro/masked-stellar-autoencoder.git && bash masked-stellar-autoencoder/scripts/canfar_remote_setup.sh'

SETUP_ID=$(canfar ps -q -n msa-setup | head -1)
echo ""
echo "Setup session: $SETUP_ID"
echo "Monitor progress: canfar logs $SETUP_ID"
echo "Wait for '=== SETUP COMPLETE ===' before running canfar_launch.sh"
