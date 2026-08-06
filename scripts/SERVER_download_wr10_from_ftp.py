#!/usr/bin/env python3

import os
import re
import subprocess
import logging
from datetime import datetime, timezone, timedelta

# -------------------------
# CONFIG
# -------------------------
USERNAME = "radar_rw"
PASSWORD = "@radar@2017_rw"
FTP_URL = "ftp://ftp.protezionecivile.it/wr10/"

DEST_BASE = "/data/wr10"
MAX_AGE_MINUTES = 30  # ignore files older than this (skip historic backfill)

# HDF-{PRODUCT}-A00-{YYYYMMDDHHMI}-B-{...}-{...}-{...}-{ELEV}-{UC}.z
FILE_RE = re.compile(
    r"^HDF-(?P<product>[A-Z]+)-A\d+-(?P<ts>\d{12})-[^-]+-\d+-\d+-\d+-(?P<elev>\d+)-(?P<uc>[UC])\.z$"
)


def setup_logging():
    os.makedirs(DEST_BASE, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(DEST_BASE, "download.log"),
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )


def dest_dir(product, elev):
    if product == "PPI":
        return os.path.join(DEST_BASE, "PPI", elev)
    return os.path.join(DEST_BASE, product)


def curl_list():
    cmd = ["curl", "-s", "--list-only", "--ftp-pasv", "--disable-epsv", "--connect-timeout", "15", "--max-time", "30",
           "-u", f"{USERNAME}:{PASSWORD}", FTP_URL]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"FTP listing failed: {result.stderr.strip()}")
        return []
    return result.stdout.splitlines()


def download_file(filename, local_path):
    cmd = ["curl", "-s", "--ftp-pasv", "--disable-epsv", "--connect-timeout", "15", "--max-time", "60", "-u", f"{USERNAME}:{PASSWORD}",
           "-o", local_path, f"{FTP_URL}{filename}"]
    return subprocess.run(cmd).returncode == 0


def main():
    setup_logging()

    # FTP timestamps are in UTC; compare against UTC now
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    cutoff = now - timedelta(minutes=MAX_AGE_MINUTES)

    files = curl_list()
    if not files:
        logging.warning("No files listed — FTP may be unreachable or empty.")
        return

    logging.info(f"Listed {len(files)} files on FTP")

    downloaded = skipped_old = skipped_exists = failed = 0

    for filename in files:
        m = FILE_RE.match(filename)
        if not m:
            continue

        try:
            ts = datetime.strptime(m.group("ts"), "%Y%m%d%H%M")
        except ValueError:
            continue

        if ts < cutoff:
            skipped_old += 1
            continue

        product = m.group("product")
        elev = m.group("elev")
        d = dest_dir(product, elev)
        os.makedirs(d, exist_ok=True)
        local = os.path.join(d, filename)

        if os.path.exists(local):
            skipped_exists += 1
            continue

        if download_file(filename, local):
            logging.info(f"Downloaded: {filename}")
            downloaded += 1
        else:
            logging.error(f"Failed: {filename}")
            failed += 1

    logging.info(
        f"Done: {downloaded} downloaded, {failed} failed, "
        f"{skipped_old} too old, {skipped_exists} already present"
    )
    print(
        f"Downloaded {downloaded} | failed {failed} | "
        f"old {skipped_old} | already present {skipped_exists}"
    )


if __name__ == "__main__":
    main()
