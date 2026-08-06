#!/usr/bin/env python3
"""
Download a Torchiarolo (Puglia) radar composite product from the Protezione
Civile FTP server.

Source: ftp://ftp.protezionecivile.it/PUGLIA/{product}/DD-MM-YYYY-HH-MM.hdf
Products: ETM, SRI, VIL, VMI

Mirrors SERVER_download_prd_from_ftp.py: one product per invocation, so each
cron entry polls its own directory for its own file. The four products are
published independently, so a single job triggering on one of them and then
grabbing the rest would race whichever product happens to lag.

Filenames encode UTC, like the other ODIM products on this server.

Usage: SERVER_download_torchiarolo_from_ftp.py <Product> [interval_minutes]
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

# Must stay below the 5-minute cron interval: a longer budget means a stalled
# run is still alive when the next one starts, so failures pile up runs (and
# FTP connections) instead of replacing them. That is what turned an FTP
# outage into 8 concurrent pollers hammering an already-struggling server.
TOTAL_TIMEOUT = 240       # 4 minutes max wait
FAST_PHASE_SECONDS = 150  # first 2.5 min: poll aggressively
FAST_SLEEP = 1
SLOW_SLEEP = 30
LOG_INTERVAL_FAST = 30

# This FTP server occasionally stalls a connection instead of refusing it, so
# every curl call is bounded — otherwise a single hang can outlast TOTAL_TIMEOUT,
# which is only checked between attempts.
CONNECT_TIMEOUT = 15
LIST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60


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
    """Return (listing_text, ok).

    'ok' distinguishes an FTP failure from a successful listing that simply
    does not contain our file yet. Without it the two look identical — an
    outage reads exactly like "data not published", which is how an EPSV
    failure once went unnoticed for hours.
    """
    # --ftp-pasv: passive mode required by the server firewall
    # --disable-epsv: the server can stop answering EPSV entirely; because it
    #   hangs rather than returning an error, curl never falls back to PASV on
    #   its own and every transfer times out. Force plain PASV.
    # --list-only: returns plain filenames instead of full directory listing
    cmd = [
        "curl", "-s", "--list-only", "--ftp-pasv", "--disable-epsv",
        "--connect-timeout", str(CONNECT_TIMEOUT),
        "--max-time", str(LIST_TIMEOUT),
        "-u", f"{USERNAME}:{PASSWORD}",
        server_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.returncode == 0


def expected_slot(interval_minutes):
    """Return the previous slot timestamp in UTC — the one whose data should now be arriving.

    Filenames encode UTC, so we must work in UTC here rather than relying on
    the server's timezone.
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    slot_seconds = interval_minutes * 60
    current = (now_ts // slot_seconds) * slot_seconds
    return datetime.fromtimestamp(current - slot_seconds, tz=timezone.utc).replace(tzinfo=None)


def download_file(server_url, filename, dest_path):
    # Download to a temp name and rename, so the webapp's folder watcher never
    # sees a half-written file.
    tmp_path = f"{dest_path}.part"
    cmd = [
        "curl", "-s", "--ftp-pasv", "--disable-epsv",
        "--connect-timeout", str(CONNECT_TIMEOUT),
        "--max-time", str(DOWNLOAD_TIMEOUT),
        "-u", f"{USERNAME}:{PASSWORD}",
        "-o", tmp_path, f"{server_url}{filename}",
    ]
    ok = subprocess.run(cmd).returncode == 0 and os.path.getsize(tmp_path) > 0
    if not ok:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    os.rename(tmp_path, dest_path)
    return True


# -------------------------
# MAIN
# -------------------------
def main(product, interval=5):
    server_url = f"{FTP_BASE}{product}/"

    destination_dir = os.path.join(DEST_BASE, product)
    os.makedirs(destination_dir, exist_ok=True)
    log_path = os.path.join(DEST_BASE, "download.log")
    setup_logger(log_path)

    slot = expected_slot(interval)
    base_name = slot.strftime("%d-%m-%Y-%H-%M")
    filename = f"{base_name}.hdf"
    local_path = os.path.join(destination_dir, filename)

    # Already downloaded (e.g. by an overlapping cron instance for the same slot).
    if os.path.exists(local_path):
        return

    logging.info(f"Waiting for slot {base_name} ({product})")

    start_time = time.time()
    last_log_time = 0

    while True:
        elapsed = time.time() - start_time
        raw, listing_ok = curl_list_files(server_url)
        reason = "waiting" if listing_ok else "FTP LISTING FAILED"

        if filename in raw:
            if download_file(server_url, filename, local_path):
                logging.info(f"Downloaded: {product}/{filename}")
                print(f"Downloaded {base_name} for {product}")
                break
            # Listed but the transfer failed — the server drops requests
            # intermittently, so fall through and retry rather than giving up.
            logging.warning(f"Download failed, will retry: {product}/{filename}")

        if elapsed < FAST_PHASE_SECONDS:
            if time.time() - last_log_time > LOG_INTERVAL_FAST:
                logging.info(f"{reason}: {product}/{filename} (fast polling)...")
                last_log_time = time.time()
            time.sleep(FAST_SLEEP)
        elif elapsed < TOTAL_TIMEOUT:
            logging.info(f"{reason}: {product}/{filename} (slow polling)...")
            time.sleep(SLOW_SLEEP)
        else:
            if listing_ok:
                logging.error(f"Timeout: {product}/{filename} not available.")
                print(f"Timeout: {filename} not available for {product}.")
            else:
                logging.error(
                    f"Timeout: {product}/{filename} — FTP listing kept failing, "
                    f"the server may be unreachable rather than the data missing."
                )
                print(f"Timeout: {filename} for {product} (FTP unreachable).")
            break


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: SERVER_download_torchiarolo_from_ftp.py <Product> [interval_minutes]")
        sys.exit(1)

    product = sys.argv[1]
    interval = 5

    for arg in sys.argv[2:]:
        if arg.isdigit():
            interval = int(arg)

    main(product, interval)
