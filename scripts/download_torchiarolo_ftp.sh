#!/bin/bash
#
# Cron wrapper for the Torchiarolo (Puglia) composite downloader.
# One product per invocation: ETM, SRI, VIL or VMI.
#
# Usage: download_torchiarolo_ftp.sh <Product> [interval_minutes]

echo "==== CRON RUN $(date -u) TORCHIAROLO PRODUCT=$1 ===="

if [ -z "$1" ]; then
    echo "Usage: $0 <Product> [interval_minutes]"
    exit 1
fi

PRODUCT="$1"
INTERVAL="$2"

# Resolve the Python script relative to this wrapper — see download_product_ftp.sh
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT="$SCRIPT_DIR/SERVER_download_torchiarolo_from_ftp.py"

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"

LOG_DIR="/data/torchiarolo"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" "$PRODUCT" $INTERVAL >> "$LOG_DIR/cron.log" 2>&1
