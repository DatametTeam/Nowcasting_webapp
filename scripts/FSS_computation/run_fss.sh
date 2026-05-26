#!/bin/bash
# Compute FSS metrics for all models then notify the webapp.
#
# Timing logic:
#   - Target ground-truth timestamp = now - 5 min (the last complete 5-min slot).
#   - Poll for the HDF file every 10 s, up to 3 min.
#   - As soon as the file lands, run the computation and notify.
#   - If the file does not arrive within 3 min, skip this tick quietly.
#
# Crontab:
#   */5 * * * * /home/ubuntu/projects/scripts/FSS_computation/run_fss.sh >> /home/ubuntu/logs/fss_compute.log 2>&1

set -uo pipefail

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"
SCRIPT_DIR="/home/ubuntu/projects/scripts/FSS_computation"
CONFIG="$SCRIPT_DIR/fss_config.yaml"
NOTIFY_URL="http://localhost:8001/api/fss/notify"

MAX_WAIT=180    # seconds to poll before giving up
POLL_INTERVAL=10

# Target timestamp: 5 minutes ago
TARGET_DATE=$(date -d '5 minutes ago' +%d-%m-%Y)
TARGET_TIME=$(date -d '5 minutes ago' +%H:%M)
TARGET_STEM="${TARGET_DATE}-${TARGET_TIME//:/-}"   # e.g. 06-05-2026-15-50

# Resolve data folder from config
DATA_FOLDER=$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_folder'])")
HDF_FILE="${DATA_FOLDER}/${TARGET_STEM}.hdf"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] FSS run — target $TARGET_DATE $TARGET_TIME"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Polling for $HDF_FILE (timeout ${MAX_WAIT}s)"

waited=0
while [ ! -f "$HDF_FILE" ]; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SKIP] HDF not found after ${MAX_WAIT}s"
        exit 0
    fi
    sleep "$POLL_INTERVAL"
    waited=$((waited + POLL_INTERVAL))
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] File arrived (+${waited}s) — processing"

# Read model list from config
MODELS=$("$PYTHON" -c "import yaml; cfg=yaml.safe_load(open('$CONFIG')); print('\n'.join(cfg['models']))")

for MODEL in $MODELS; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]  → $MODEL"
    "$PYTHON" "$SCRIPT_DIR/fss_compute.py" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --date  "$TARGET_DATE" \
        --time  "$TARGET_TIME" \
    || echo "[$(date '+%Y-%m-%d %H:%M:%S')]  [WARNING] $MODEL exited with error $?"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')]  → Probabilistic (ensemble)"
"$PYTHON" "$SCRIPT_DIR/fss_ensemble.py" \
    --config "$CONFIG" \
    --date   "$TARGET_DATE" \
    --time   "$TARGET_TIME" \
|| echo "[$(date '+%Y-%m-%d %H:%M:%S')]  [WARNING] Probabilistic ensemble exited with error $?"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All models done — notifying webapp"
curl -sf -X POST "$NOTIFY_URL" \
    && echo "[$(date '+%Y-%m-%d %H:%M:%S')] Notified OK" \
    || echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] notify failed (webapp down?)"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
