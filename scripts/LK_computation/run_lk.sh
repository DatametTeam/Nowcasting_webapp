#!/bin/bash
# Compute Lucas-Kanade optical flow from the latest SRI frames, then notify the webapp.
#
# Crontab:
#   */5 * * * * /home/ubuntu/projects/scripts/LK_computation/run_lk.sh >> /home/ubuntu/logs/lk_compute.log 2>&1

set -uo pipefail

# ── Only two things to configure here ────────────────────────────────────────
PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/lk_config.yaml"
# ─────────────────────────────────────────────────────────────────────────────

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LK computation"
"$PYTHON" "$SCRIPT_DIR/compute_lk.py" --config "$CONFIG" \
    && echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" \
    || echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] exited with code $?"
