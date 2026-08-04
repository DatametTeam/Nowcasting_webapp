#!/bin/bash
#
# Cron wrapper for the Torchiarolo (Puglia) composite downloader.
# Pulls SRI, VMI, ETM and VIL for one 5-minute slot per run.

echo "==== CRON RUN $(date -u) TORCHIAROLO ===="

PYTHON="/home/ubuntu/miniconda3/envs/protezionecivile/bin/python"
SCRIPT="/home/ubuntu/projects/Nowcasting_webapp/scripts/SERVER_download_torchiarolo_from_ftp.py"

LOG_DIR="/data/torchiarolo"
mkdir -p "$LOG_DIR"

"$PYTHON" "$SCRIPT" >> "$LOG_DIR/cron.log" 2>&1