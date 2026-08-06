#!/usr/bin/env python3

import os
import time
import subprocess
import logging
from datetime import datetime, timedelta

# -------------------------
# CONFIG
# -------------------------
USERNAME = "radar_rw"
PASSWORD = "@radar@2017_rw"
FTP_URL = "ftp://ftp.protezionecivile.it/cagliari/hdf5/"

DEST_BASE = "/data/cagliari_xband"

TOTAL_TIMEOUT = 600       # 10 minutes max wait
FAST_PHASE_SECONDS = 150  # first 2.5 min: poll aggressively
FAST_SLEEP = 1
SLOW_SLEEP = 30
LOG_INTERVAL_FAST = 30

# Filename format: {PP}W{YY}{DOY}{HHMM}{S}{SITE}.{IDX}.h5
# e.g. PZW2613210550L.801.h5
#   PP   = product prefix (PZ=PPI, CZ=CorrectedZ/VMI, OZ=OriginalZ/CAPPI, RR=RainRate)
#   YY   = 2-digit year          (26 → 2026)
#   DOY  = 3-digit day-of-year   (132 → May 12)
#   HHMM = hour+minute           (1055 → 10:55)
#   S    = scan number (fixed 0 for this radar)
#   SITE = site identifier char  (fixed L for this radar)
#   IDX  = 3-digit product/elevation index
SCAN_SUFFIX = "0L"

# All products delivered per slot: (prefix, idx)
EXPECTED_PRODUCTS = [
    ("PZ", "801"),   # PPI elevation 1
    ("PZ", "802"),   # PPI elevation 2
    ("PZ", "803"),   # PPI elevation 3
    ("PZ", "804"),   # PPI elevation 4
    ("PZ", "805"),   # PPI elevation 5
    ("CZ", "805"),   # Corrected Z / VMI
    ("OZ", "810"),   # Original Z / CAPPI
    ("RR", "001"),   # Rain Rate / SRI  ← used as primary trigger
]

PRIMARY_PRODUCT = ("RR", "001")


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
    """Return the CURRENT slot timestamp in UTC — the one whose data should now be arriving.

    Filenames encode UTC (ODIM standard), so we must work in UTC here.
    At :57 this returns :55 (current 5-min boundary), unlike the PRD/WR10
    scripts which return the slot before (at :57 they return :50).
    """
    from datetime import timezone
    now_ts = datetime.now(timezone.utc).timestamp()
    slot_seconds = interval_minutes * 60
    current_ts = (now_ts // slot_seconds) * slot_seconds
    return datetime.fromtimestamp(current_ts, tz=timezone.utc).replace(tzinfo=None)


def slot_to_ts_str(dt):
    """Convert a slot datetime to the 9-char YYDOYHHMM string.

    The scan digit (0) and site char (L) are in SCAN_SUFFIX and appended by
    make_filename(), so they must NOT be included here.
    """
    yy = dt.year - 2000
    doy = dt.timetuple().tm_yday
    return f"{yy:02d}{doy:03d}{dt.hour:02d}{dt.minute:02d}"


def make_filename(prefix, ts_str, idx):
    return f"{prefix}W{ts_str}{SCAN_SUFFIX}.{idx}.h5"


def dest_dir(prefix, idx):
    if prefix == "PZ":
        return os.path.join(DEST_BASE, "PPI", idx)
    elif prefix == "CZ":
        return os.path.join(DEST_BASE, "CZ")
    elif prefix == "OZ":
        return os.path.join(DEST_BASE, "OZ")
    elif prefix == "RR":
        return os.path.join(DEST_BASE, "RR")
    else:
        return os.path.join(DEST_BASE, prefix)


def _subfolder(prefix):
    """FTP subdirectory for a given 2-char file prefix (e.g. 'RR' → 'RRW/')."""
    return f"{prefix}W/"


def curl_list(prefix):
    """List files in the FTP subdirectory for the given product prefix."""
    url = f"{FTP_URL}{_subfolder(prefix)}"
    cmd = ["curl", "-s", "--list-only", "--ftp-pasv", "--disable-epsv",
           "-u", f"{USERNAME}:{PASSWORD}", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def download_file(prefix, filename, local_path):
    url = f"{FTP_URL}{_subfolder(prefix)}{filename}"
    cmd = ["curl", "-s", "--ftp-pasv", "--disable-epsv", "-u", f"{USERNAME}:{PASSWORD}",
           "-o", local_path, url]
    return subprocess.run(cmd).returncode == 0


# -------------------------
# MAIN
# -------------------------
def main():
    setup_logging()

    slot = expected_slot()
    ts_str = slot_to_ts_str(slot)
    slot_label = slot.strftime("%d-%m-%Y-%H-%M")

    primary_prefix, primary_idx = PRIMARY_PRODUCT
    primary_file = make_filename(primary_prefix, ts_str, primary_idx)

    # Already have all files for this slot?
    all_local = []
    for prefix, idx in EXPECTED_PRODUCTS:
        d = dest_dir(prefix, idx)
        all_local.append(os.path.join(d, make_filename(prefix, ts_str, idx)))
    if all(os.path.exists(f) for f in all_local):
        return

    logging.info(f"Waiting for slot {slot_label}")

    start_time = time.time()
    last_log_time = 0

    while True:
        elapsed = time.time() - start_time
        listing = curl_list(primary_prefix)

        if primary_file in listing:
            all_ok = True
            for prefix, idx in EXPECTED_PRODUCTS:
                fname = make_filename(prefix, ts_str, idx)
                d = dest_dir(prefix, idx)
                os.makedirs(d, exist_ok=True)
                local_path = os.path.join(d, fname)
                if os.path.exists(local_path):
                    continue
                if download_file(prefix, fname, local_path):
                    logging.info(f"Downloaded: {fname}")
                else:
                    logging.error(f"Download failed: {fname}")
                    all_ok = False
            if all_ok:
                print(f"Downloaded all products for slot {slot_label}")
            else:
                print(f"Downloaded slot {slot_label} (some files failed — check log)")
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
            print(f"Timeout: slot {slot_label} not available.")
            break


if __name__ == "__main__":
    main()
