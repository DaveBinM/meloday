#!/usr/bin/env python3
"""
Meloday Cron Manager

Updates the crontab entry for meloday.py based on the currently active
timezone — using the travel timezone if today falls within a configured trip
window, otherwise using the system's local time.

Usage:
  python utilities/meloday_cron.py            # update crontab
  python utilities/meloday_cron.py --dry-run  # preview without writing
  python utilities/meloday_cron.py --status   # show current state

To auto-switch on travel start/end, add this line to your crontab once:
  0 0 * * * /path/to/venv/python /path/to/utilities/meloday_cron.py
"""

import argparse
import os
import subprocess
import sys
import yaml
from datetime import date, datetime
from zoneinfo import ZoneInfo

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yml")

# These markers delimit the block we own inside the crontab.
MARKER_START = "# --- MELODAY PLAYLIST GENERATION ---"
MARKER_END   = "# --- END MELODAY PLAYLIST GENERATION ---"

# The legacy comment that pre-dates managed mode — stripped on first migration.
LEGACY_HEADER = "# --- EXISTING PLAYLIST GENERATION ---"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_active_timezone(config):
    """Return the IANA timezone string for the currently active trip, or None."""
    for trip in config.get("travel", []):
        try:
            tz = ZoneInfo(trip["timezone"])
            dest_today = datetime.now(tz=tz).date()
            if date.fromisoformat(trip["start"]) <= dest_today <= date.fromisoformat(trip["end"]):
                return trip["timezone"]
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Schedule computation
# ---------------------------------------------------------------------------

def compute_schedule_hours(time_periods, active_tz):
    """
    For each period take its start hour (min of its hours list), express it in
    active_tz (or local if None), then convert to the system's local clock.
    Returns a sorted list of unique integers in [0, 23].
    """
    today    = date.today()
    src_tz   = ZoneInfo(active_tz) if active_tz else None
    hours    = set()

    for details in time_periods.values():
        start_hour = min(details["hours"])

        if src_tz:
            dt = datetime(today.year, today.month, today.day,
                          start_hour, 0, 0, tzinfo=src_tz)
            system_hour = dt.astimezone().hour   # OS converts to system local time
        else:
            system_hour = start_hour

        hours.add(system_hour)

    return sorted(hours)


# ---------------------------------------------------------------------------
# Crontab I/O
# ---------------------------------------------------------------------------

def read_crontab():
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    # crontab -l exits 1 when no crontab exists; that's fine — treat as empty.
    return result.stdout if result.returncode == 0 else ""


def write_crontab(content):
    proc = subprocess.run(["crontab", "-"], input=content, text=True,
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to write crontab: {proc.stderr.strip()}")


# ---------------------------------------------------------------------------
# Block building
# ---------------------------------------------------------------------------

def build_block(hours, python_path, script_path, active_tz):
    hours_str = ",".join(str(h) for h in hours)
    tz_note   = f"  # {active_tz} periods in system time" if active_tz else ""
    return (
        f"{MARKER_START}\n"
        f"0 {hours_str} * * * {python_path} {script_path}{tz_note}\n"
        f"{MARKER_END}"
    )


# ---------------------------------------------------------------------------
# Crontab update
# ---------------------------------------------------------------------------

def update_crontab(cron_cfg, time_periods, active_tz, dry_run=False):
    python_path = (cron_cfg.get("python_path") or "").strip() or sys.executable
    script_path = (cron_cfg.get("script_path") or "").strip() or \
                  os.path.join(BASE_DIR, "meloday.py")

    hours     = compute_schedule_hours(time_periods, active_tz)
    new_block = build_block(hours, python_path, script_path, active_tz)
    current   = read_crontab()

    if MARKER_START in current:
        # ---- Managed mode: replace the existing block in-place ----
        lines        = current.splitlines()
        result_lines = []
        skip         = False
        for line in lines:
            stripped = line.strip()
            if stripped == MARKER_START:
                skip = True
                result_lines.append(new_block)
            elif stripped == MARKER_END:
                skip = False
            elif not skip:
                result_lines.append(line)
        new_crontab = "\n".join(result_lines).rstrip("\n") + "\n"

    else:
        # ---- First-time migration: find and replace the unmanaged line ----
        lines        = current.splitlines()
        result_lines = []
        replaced     = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip the old legacy section header — we'll replace it with our block.
            if stripped == LEGACY_HEADER:
                continue

            # The unmanaged meloday.py playlist line (not a utilities/ helper).
            if (not stripped.startswith("#")
                    and "meloday.py" in stripped
                    and "utilities/" not in stripped
                    and not replaced):
                result_lines.append(new_block)
                replaced = True
                continue

            result_lines.append(line)

        if not replaced:
            result_lines.extend(["", new_block])

        new_crontab = "\n".join(result_lines).rstrip("\n") + "\n"

    if dry_run:
        print("[DRY RUN] Crontab would be set to:\n")
        print(new_crontab)
        return

    write_crontab(new_crontab)
    tz_info = f"travel TZ: {active_tz}" if active_tz else "local system time"
    print(f"[CRON] Schedule updated — {tz_info}")
    print(f"[CRON] Period start hours (system time): {hours}")


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status(time_periods, active_tz, cron_cfg):
    python_path = (cron_cfg.get("python_path") or "").strip() or sys.executable
    script_path = (cron_cfg.get("script_path") or "").strip() or \
                  os.path.join(BASE_DIR, "meloday.py")
    hours = compute_schedule_hours(time_periods, active_tz)

    print(f"Active timezone : {active_tz or 'local (no trip active)'}")
    print(f"Period start hours (system time): {hours}")
    print(f"Cron entry would be:")
    print(f"  0 {','.join(str(h) for h in hours)} * * * {python_path} {script_path}")
    print()

    current = read_crontab()
    if MARKER_START in current:
        lines = current.splitlines()
        in_block = False
        print("Current managed block in crontab:")
        for line in lines:
            if line.strip() == MARKER_START:
                in_block = True
            if in_block:
                print(f"  {line}")
            if line.strip() == MARKER_END:
                break
    else:
        print("No managed block found in crontab (run without --status to install).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Meloday Cron Manager")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the crontab change without writing it")
    parser.add_argument("--status",  action="store_true",
                        help="Show what the schedule would look like without modifying crontab")
    args = parser.parse_args()

    config   = load_config()
    cron_cfg = config.get("cron", {})

    if not cron_cfg.get("enabled", False):
        print("[CRON] Cron management is disabled.")
        print("       Set  cron.enabled: true  in config.yml to enable.")
        sys.exit(0)

    time_periods = config.get("time_periods", {})
    if not time_periods:
        print("[CRON] No time_periods defined in config.yml — cannot compute schedule.")
        sys.exit(1)

    active_tz = get_active_timezone(config)

    if args.status:
        show_status(time_periods, active_tz, cron_cfg)
    else:
        update_crontab(cron_cfg, time_periods, active_tz, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
