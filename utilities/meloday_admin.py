import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sqlite3
import logging
from datetime import datetime
from plexapi.server import PlexServer
from meloday import (
    PLEX_URL, PLEX_TOKEN, MUSIC_LIBRARY, 
    ESSENTIA_CACHE_PATH, BASE_DIR
)

# --- LOGGING SETUP ---
LOG_FILE = os.path.join(BASE_DIR, "logs", "meloday_admin.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Standard FileHandler for text-only output
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("MelodayAdmin")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def log_admin(msg, level="info"):
    """
    Cleans message of NUL bytes and non-printable characters for grep compatibility,
    then prints to console and writes to the log.
    """
    if not msg:
        return
        
    # Strictly filter for printable characters and standard whitespace
    clean_msg = "".join(ch for ch in str(msg) if ch.isprintable() or ch in ("\n", "\r", "\t"))
    clean_msg = clean_msg.replace('\x00', '') # Explicitly remove NUL bytes

    print(clean_msg)
    
    if level == "error":
        logger.error(clean_msg)
    elif level == "warn":
        logger.warning(clean_msg)
    else:
        logger.info(clean_msg)
    
    # Flush immediately to ensure visibility during cron execution
    file_handler.flush()

def run_maintenance():
    """
    Performs full maintenance on the Essentia cache:
    1. Orphan Cleanup: Removes entries for tracks no longer in Plex.
    2. Integrity Check: Removes entries missing required fields.
    3. VACUUM: Reclaims freed disk space.
    """
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        log_admin(f"Cache file not found at {ESSENTIA_CACHE_PATH}", "error")
        return

    log_admin(f"=== Starting Meloday Cache Maintenance: {MUSIC_LIBRARY} ===")

    try:
        # 1. Connect to Plex (safe fail point if Plex is down)
        plex = PlexServer(PLEX_URL, PLEX_TOKEN)
        music = plex.library.section(MUSIC_LIBRARY)

        log_admin("Fetching current library state from Plex...")
        current_library_keys = {str(t.ratingKey) for t in music.search(libtype='track')}

        # Safety check for unmounted drives / empty library
        if not current_library_keys:
            log_admin("Plex returned 0 tracks. Skipping cleanup to prevent data loss.", "warn")
            return

        # 2. Open SQLite and perform maintenance
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60)
        conn.execute("PRAGMA journal_mode=WAL")

        original_count = conn.execute("SELECT COUNT(*) FROM essentia_cache").fetchone()[0]

        # Orphan cleanup: entries whose rating_key is no longer in Plex
        db_keys = {row[0] for row in conn.execute("SELECT rating_key FROM essentia_cache")}
        orphan_keys = db_keys - current_library_keys
        if orphan_keys:
            conn.executemany(
                "DELETE FROM essentia_cache WHERE rating_key = ?",
                [(k,) for k in orphan_keys]
            )
        orphans_removed = len(orphan_keys)

        # Integrity cleanup: entries with no file_path are unusable regardless of mode.
        # Acoustic fields (bpm/key/energy) may be null for metadata-only entries — that's valid.
        cursor = conn.execute(
            "DELETE FROM essentia_cache WHERE file_path IS NULL"
        )
        corrupt_removed = cursor.rowcount

        conn.commit()
        # Reclaim freed pages left by deletions
        conn.execute("VACUUM")
        conn.close()

        log_admin("Maintenance Complete")
        log_admin(f"Total Entries Scanned: {original_count}")
        log_admin(f"Orphaned Tracks Removed: {orphans_removed}")
        log_admin(f"Corrupt/Invalid Entries Removed: {corrupt_removed}")
        log_admin(f"Final Optimized Cache Size: {original_count - orphans_removed - corrupt_removed}")

    except Exception as e:
        log_admin(f"Maintenance failed: {e}", "error")

if __name__ == "__main__":
    run_maintenance()