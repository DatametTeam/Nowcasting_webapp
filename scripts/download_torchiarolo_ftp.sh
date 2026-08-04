#!/bin/bash
#
# Cron wrapper for the Torchiarolo (Puglia) composite downloader.
# One product per invocation: ETM, SRI, VIL or VMI.

echo "==== CRON RUN $(date -u) TORCHIAROLO PRODUCT=$1 ===="

if [ -z "$1" ]; then
    echo "Usage: $0 <Product> [interval_minutes]"
    exit 1
fi

PRODUCT="$1"
INTERVAL="$2"

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"
SCRIPT="/home/ubuntu/projects/Nowcasting_webapp/scripts/SERVER_download_torchiarolo_from_ftp.py"

LOG_DIR="/data/torchiarolo"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" "$PRODUCT" $INTERVAL >> "$LOG_DIR/cron.log" 2>&1
