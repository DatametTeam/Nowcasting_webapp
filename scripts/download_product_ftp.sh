#!/bin/bash
#
# Cron wrapper for the national PRD product downloader (SRI_adj, VMI, ETM, VIL,
# IR_108, AMV, SITES...). Downloads the file for a specific 5-minute slot, not
# whichever file happens to be newest.
#
# Usage: download_product_ftp.sh <Product> [TIFF|SHP|TXT] [interval_minutes]

echo "==== CRON RUN $(date -u) PRODUCT=$1 ===="

if [ -z "$1" ]; then
    echo "Usage: $0 <Product> [TIFF|SHP|TXT] [interval_minutes]"
    exit 1
fi

PRODUCT="$1"
MODE="$2"
INTERVAL="$3"

# Resolve the Python script relative to this wrapper, so the pair always moves
# together with the checkout instead of pointing at a hardcoded copy that
# silently goes stale.
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT="$SCRIPT_DIR/SERVER_download_prd_from_ftp.py"

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"

LOG_DIR="/data/$PRODUCT"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" "$PRODUCT" $MODE $INTERVAL >> "$LOG_DIR/cron.log" 2>&1
