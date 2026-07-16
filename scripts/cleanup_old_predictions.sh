#!/bin/bash
# Deletes real-time prediction files older than 14 days.
# Meant to run daily via cron.
#
# Crontab entry (runs at 03:00 every day):
#   0 0 * * * /home/ubuntu/projects/Nowcasting_webapp/scripts/cleanup_old_predictions.sh >> /home/ubuntu/logs/cleanup_predictions.log 2>&1

PRED_DIR="/data/nwc_webapp/real_time_results"
DAYS=14

echo "--- $(date '+%Y-%m-%d %H:%M:%S') --- Cleanup started ---"

if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: prediction directory not found: $PRED_DIR"
    exit 1
fi

# Count and list files before deletion
count=$(find "$PRED_DIR" -name "*.npy" -mtime +$DAYS | wc -l)
echo "Found $count .npy files older than $DAYS days"

if [ "$count" -eq 0 ]; then
    echo "Nothing to delete."
    echo "--- Done ---"
    exit 0
fi

# Delete and log each removed file
find "$PRED_DIR" -name "*.npy" -mtime +$DAYS -print -delete

echo "--- Deleted $count files ---"

# Show current disk usage
echo "Disk usage after cleanup:"
du -sh "$PRED_DIR"
df -h /data | tail -1
