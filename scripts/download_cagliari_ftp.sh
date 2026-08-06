#!/bin/bash
#
# Cron wrapper for the Cagliari X-band downloader.
# Takes no arguments: one run fetches every product for the current slot.

echo "==== CRON RUN $(date -u) CAGLIARI ===="

# Resolve the Python script relative to this wrapper — see download_product_ftp.sh
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
SCRIPT="$SCRIPT_DIR/SERVER_download_cagliari_from_ftp.py"

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"

LOG_DIR="/data/cagliari_xband"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" >> "$LOG_DIR/cron.log" 2>&1
