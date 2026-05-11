#!/bin/bash

echo "==== CRON RUN $(date) PRODUCT=$1 ===="

if [ -z "$1" ]; then
    echo "Usage: $0 <Product> [TIFF|SHP] [interval_minutes]"
    exit 1
fi

PRODUCT="$1"
MODE="$2"
INTERVAL="$3"

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"
SCRIPT="/home/ubuntu/projects/scripts/SERVER_download_prd_from_ftp.py"

LOG_DIR="/home/ubuntu/data1/$PRODUCT"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" "$PRODUCT" $MODE $INTERVAL >> "$LOG_DIR/cron.log" 2>&1
