#!/usr/bin/env python3

import os
import time
import subprocess
import logging
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
USERNAME = "radar_rw"
PASSWORD = "@radar@2017_rw"

DEST_BASE = "/data"

TOTAL_TIMEOUT = 600       # 10 minutes max wait
FAST_PHASE_SECONDS = 150  # first 2.5 min: poll aggressively
FAST_SLEEP = 1
SLOW_SLEEP = 30
LOG_INTERVAL_FAST = 30

SHP_EXTENSIONS = ["shp", "shx", "dbf"]

TXT_EXTENSIONS = ["txt"]


# -------------------------
# HELPERS
# -------------------------
def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )


def curl_list_files(server_url):
    # --ftp-pasv: passive mode required by the server firewall
    # --list-only: returns plain filenames instead of full directory listing
    cmd = [
        "curl", "-s", "--list-only", "--ftp-pasv",
        "-u", f"{USERNAME}:{PASSWORD}",
        server_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def expected_slot(interval_minutes):
    """Return the previous slot timestamp — the one whose data should now be arriving."""
    now = datetime.now()
    slot_seconds = interval_minutes * 60
    current = (now.timestamp() // slot_seconds) * slot_seconds
    return datetime.fromtimestamp(current - slot_seconds)


def download_file(server_url, filename, dest_path):
    cmd = [
        "curl", "-s", "--ftp-pasv", "-u", f"{USERNAME}:{PASSWORD}",
        "-o", dest_path, f"{server_url}{filename}",
    ]
    return subprocess.run(cmd).returncode == 0


# -------------------------
# MAIN
# -------------------------
def main(product, mode="hdf", interval=5):
    if mode == "tiff":
        primary_ext = "tif"
        server_url = f"ftp://ftp.protezionecivile.it/PRD/{product}/TIFF/"
        extensions = ["tif"]
    elif mode == "shp":
        primary_ext = "shp"
        server_url = f"ftp://ftp.protezionecivile.it/PRD/{product}/"
        extensions = SHP_EXTENSIONS
    elif mode == "txt":
        primary_ext = "txt"
        server_url = f"ftp://ftp.protezionecivile.it/PRD/{product}/"
        extensions = ["txt"]

    else:
        primary_ext = "hdf"
        server_url = f"ftp://ftp.protezionecivile.it/PRD/{product}/"
        extensions = ["hdf"]

    destination_dir = os.path.join(DEST_BASE, product)
    os.makedirs(destination_dir, exist_ok=True)
    log_path = os.path.join(destination_dir, "download.log")
    setup_logger(log_path)

    slot = expected_slot(interval)
    base_name = slot.strftime("%d-%m-%Y-%H-%M")
    primary_file = f"{base_name}.{primary_ext}"

    # Already downloaded (e.g. by an overlapping cron instance for the same slot).
    # For SHP, all three files must exist before we consider the slot complete.
    local_files = [os.path.join(destination_dir, f"{base_name}.{e}") for e in extensions]
    if all(os.path.exists(f) for f in local_files):
        return

    logging.info(f"Waiting for slot {base_name} ({product}, {mode})")

    start_time = time.time()
    last_log_time = 0

    while True:
        elapsed = time.time() - start_time
        raw = curl_list_files(server_url)

        if primary_file in raw:
            all_ok = True
            for ext in extensions:
                fname = f"{base_name}.{ext}"
                local_path = os.path.join(destination_dir, fname)
                if os.path.exists(local_path):
                    continue
                success = download_file(server_url, fname, local_path)
                if success:
                    logging.info(f"Downloaded: {fname}")
                else:
                    logging.error(f"Download failed: {fname}")
                    all_ok = False
            if all_ok:
                print(f"Downloaded {base_name} for {product}")
            break

        if elapsed < FAST_PHASE_SECONDS:
            if time.time() - last_log_time > LOG_INTERVAL_FAST:
                logging.info(f"Waiting for {primary_file} (fast polling)...")
                last_log_time = time.time()
            time.sleep(FAST_SLEEP)
        elif elapsed < TOTAL_TIMEOUT:
            logging.info(f"Waiting for {primary_file} (slow polling)...")
            time.sleep(SLOW_SLEEP)
        else:
            logging.warning(f"Timeout: {primary_file} not available.")
            print(f"Timeout: {primary_file} not available.")
            break


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python SERVER_download.py <Product> [TIFF|SHP] [interval_minutes]")
        sys.exit(1)

    product = sys.argv[1]
    mode = "hdf"
    interval = 5

    for arg in sys.argv[2:]:
        upper = arg.upper()
        if upper == "TIFF":
            mode = "tiff"
        elif upper == "SHP":
            mode = "shp"
        elif arg.isdigit():
            interval = int(arg)
        elif upper == "TXT":
            mode = "txt"

    main(product, mode, interval)