#!/usr/bin/env bash
# Deploy trade4 live scheduler to VPS.
# Usage: ./deploy.sh [--testnet]
set -euo pipefail

VPS="ahmet@178.254.32.125"
REMOTE_DIR="~/trade4"
SESSION="trade4_live"

echo "=== Syncing code to VPS ==="
rsync -az --exclude='.git' --exclude='.venv' --exclude='.venv-test' \
      --exclude='data/' --exclude='output/' --exclude='__pycache__' \
      ./ "${VPS}:${REMOTE_DIR}/"

echo "=== Installing dependencies on VPS ==="
ssh "${VPS}" "cd ${REMOTE_DIR} && python3 -m venv .venv && .venv/bin/pip install -q -e ."

echo "=== Copying .env ==="
scp .env "${VPS}:${REMOTE_DIR}/.env"

# Check if testnet flag passed
TESTNET_FLAG=""
if [[ "${1:-}" == "--testnet" ]]; then
    TESTNET_FLAG="--testnet"
    echo "*** TESTNET MODE ***"
fi

echo "=== Starting tmux session '${SESSION}' on VPS ==="
ssh "${VPS}" bash << REMOTE
  tmux kill-session -t ${SESSION} 2>/dev/null || true
  tmux new-session -d -s ${SESSION} \
    "cd ${REMOTE_DIR} && .venv/bin/python -m trade4.live.scheduler ${TESTNET_FLAG} 2>&1 | tee -a logs/live.log"
  echo "Session started. Attach with: tmux attach -t ${SESSION}"
REMOTE

echo "=== Done. Monitor with: ssh ${VPS} 'tmux attach -t ${SESSION}' ==="
