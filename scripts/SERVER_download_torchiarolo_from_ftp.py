#!/usr/bin/env python3
"""
Download the Torchiarolo (Puglia) radar composite products from the
Protezione Civile FTP server.

Source: ftp://ftp.protezionecivile.it/PUGLIA/{ETM,SRI,VIL,VMI}/DD-MM-YYYY-HH-MM.hdf

All four products are delivered for the same 5-minute slot, so a single run
waits for the trigger product (SRI) and then pulls the whole set. This keeps
the four products in sync per timestamp, which the webapp relies on when it
stacks them as map layers on a shared timeline.

Filenames encode UTC, like the other ODIM products on this server.
"""

import os
import time
import subprocess
import logging
from datetime import datetime, timezone

# -------------------------
# CONFIG
# -------------------------
USERNAME = "radar_rw"
PASSWORD = "@radar@2017_rw"
FTP_BASE = "ftp://ftp.protezionecivile.it/PUGLIA/"

DEST_BASE = "/data/torchiarolo"

TOTAL_TIMEOUT = 600       # 10 minutes max wait
FAST_PHASE_SECONDS = 150  # first 2.5 min: poll aggressively
FAST_SLEEP = 1
SLOW_SLEEP = 30
LOG_INTERVAL_FAST = 30

# All products delivered per slot. SRI is listed first and used as the trigger:
# once it appears the others are published too.
PRODUCTS = ["SRI", "VMI", "ETM", "VIL"]
PRIMARY_PRODUCT = "SRI"


# -------------------------
# HELPERS
# -------------------------
def setup_logging():
    os.makedirs(DEST_BASE, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(DEST_BASE, "download.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )


def expected_slot(interval_minutes=5):
    """Return the previous slot timestamp in UTC — the one whose data should now be arriving.

    Filenames encode UTC, so we must work in UTC here. Publication lags the
    slot by roughly 4 minutes, so at :02 we target :55 rather than :00.
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    slot_seconds = interval_minutes * 60
    current_ts = (now_ts // slot_seconds) * slot_seconds
    return datetime.fromtimestamp(current_ts - slot_seconds, tz=timezone.utc).replace(tzinfo=None)


def product_url(product):
    return f"{FTP_BASE}{product}/"


def dest_dir(product):
    return os.path.join(DEST_BASE, product)


def curl_list(product):
    """List filenames in the FTP subdirectory for the given product."""
    cmd = ["curl", "-s", "--list-only", "--ftp-pasv",
           "-u", f"{USERNAME}:{PASSWORD}", product_url(product)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def download_file(product, filename, local_path):
    # Download to a temp name and rename, so the folder watcher in the webapp
    # never sees a half-written file.
    tmp_path = f"{local_path}.part"
    cmd = ["curl", "-s", "--ftp-pasv", "-u", f"{USERNAME}:{PASSWORD}",
           "-o", tmp_path, f"{product_url(product)}{filename}"]
    if subprocess.run(cmd).returncode != 0 or not os.path.getsize(tmp_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    os.rename(tmp_path, local_path)
    return True


# -------------------------
# MAIN
# -------------------------
def main():
    setup_logging()

    slot = expected_slot()
    base_name = slot.strftime("%d-%m-%Y-%H-%M")
    primary_file = f"{base_name}.hdf"

    # Already have every product for this slot (e.g. an overlapping cron run
    # already handled it)?
    local_paths = {p: os.path.join(dest_dir(p), primary_file) for p in PRODUCTS}
    if all(os.path.exists(path) for path in local_paths.values()):
        return

    logging.info(f"Waiting for slot {base_name}")

    start_time = time.time()
    last_log_time = 0

    while True:
        elapsed = time.time() - start_time
        listing = curl_list(PRIMARY_PRODUCT)

        if primary_file in listing:
            all_ok = True
            for product in PRODUCTS:
                local_path = local_paths[product]
                if os.path.exists(local_path):
                    continue
                os.makedirs(dest_dir(product), exist_ok=True)
                if download_file(product, primary_file, local_path):
                    logging.info(f"Downloaded: {product}/{primary_file}")
                else:
                    logging.error(f"Download failed: {product}/{primary_file}")
                    all_ok = False
            if all_ok:
                print(f"Downloaded all products for slot {base_name}")
            else:
                print(f"Downloaded slot {base_name} (some files failed — check log)")
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
            print(f"Timeout: slot {base_name} not available.")
            break


if __name__ == "__main__":
    main()