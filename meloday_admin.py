import os
import json
import portalocker
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
    1. Database Integrity: Validates JSON structure and required keys.
    2. Orphan Cleanup: Removes entries for tracks no longer in Plex.
    """
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        log_admin(f"Cache file not found at {ESSENTIA_CACHE_PATH}", "error")
        return

    log_admin(f"=== Starting Meloday Cache Maintenance: {MUSIC_LIBRARY} ===")
    
    try:
        # 1. Connect to Plex (Safe fail point if Plex is down)
        plex = PlexServer(PLEX_URL, PLEX_TOKEN)
        music = plex.library.section(MUSIC_LIBRARY)
        
        log_admin("Fetching current library state from Plex...")
        current_library_keys = {str(t.ratingKey) for t in music.search(libtype='track')}
        
        # 2. Safety Check for unmounted drives/empty library
        if not current_library_keys:
            log_admin("Plex returned 0 tracks. Skipping cleanup to prevent data loss.", "warn")
            return

        # 3. Open and Lock for maintenance
        with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='r+', timeout=60) as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                log_admin("Cache file is corrupted and could not be read.", "error")
                return

            original_count = len(cache)
            orphans_removed = 0
            corrupt_removed = 0
            
            # Required keys for a valid entry
            required_keys = {"bpm", "key", "energy", "file_path"}
            
            keys_to_check = list(cache.keys())
            for rk in keys_to_check:
                if rk not in current_library_keys:
                    del cache[rk]
                    orphans_removed += 1
                    continue
                
                entry = cache[rk]
                if not isinstance(entry, dict) or not required_keys.issubset(entry.keys()):
                    del cache[rk]
                    corrupt_removed += 1

            # 4. Save (Minified - no indents)
            f.seek(0)
            json.dump(cache, f, separators=(',', ':')) 
            f.truncate()
            
            log_admin("Maintenance Complete")
            log_admin(f"Total Entries Scanned: {original_count}")
            log_admin(f"Orphaned Tracks Removed: {orphans_removed}")
            log_admin(f"Corrupt/Invalid Entries Removed: {corrupt_removed}")
            log_admin(f"Final Optimized Cache Size: {len(cache)}")

    except Exception as e:
        log_admin(f"Maintenance failed: {e}", "error")

if __name__ == "__main__":
    run_maintenance()