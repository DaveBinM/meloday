import os
import multiprocessing
import time
import json
import sqlite3
import concurrent.futures
import logging
from datetime import timedelta, datetime
from meloday import (
    PLEX_URL, PLEX_TOKEN, MUSIC_LIBRARY, analyze_track_essentia,
    save_essentia_cache, ESSENTIA_ENABLED,
    ESSENTIA_CACHE_PATH, _essentia_cache, PlexServer, BASE_DIR,
    get_local_path, _migrate_json_to_sqlite, get_optimal_workers
)

# --- LOGGING SETUP ---
LOG_FILE = os.path.join(BASE_DIR, "logs", "pre_analyze.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("PreAnalyze")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# --- GLOBAL FOR WORKERS ---
# This will be initialized once per worker process to avoid connection overhead
worker_plex = None

def init_worker():
    """Initializes a single Plex session per worker process."""
    global worker_plex
    worker_plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=120)

def log_msg(msg, level="info", log_only=False, **kwargs):
    """Filters for printable characters and supports standard print kwargs."""
    if not msg: return
    clean_msg = "".join(ch for ch in str(msg) if ch.isprintable() or ch in ("\n", "\r", "\t"))
    clean_msg = clean_msg.replace('\x00', '')

    if not log_only:
        print(msg, **kwargs) # This now accepts end='\r'

    if level == "error":
        logger.error(clean_msg)
    else:
        logger.info(clean_msg)

# --- 1. CACHE UTILITIES ---

def load_essentia_cache_exclusive():
    """Reads all entries from the SQLite cache into a dict."""
    _migrate_json_to_sqlite()
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        return {}
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        rows = conn.execute(
            "SELECT rating_key, bpm, key, energy, year, artist, genres, styles, moods, file_path, last_synced "
            "FROM essentia_cache"
        ).fetchall()
        conn.close()
        result = {}
        for rk, bpm, key, energy, year, artist, genres_j, styles_j, moods_j, file_path, last_synced in rows:
            result[rk] = {
                "bpm": bpm, "key": key, "energy": energy, "year": year, "artist": artist,
                "genres": json.loads(genres_j) if genres_j else [],
                "styles": json.loads(styles_j) if styles_j else [],
                "moods": json.loads(moods_j) if moods_j else [],
                "file_path": file_path, "last_synced": last_synced
            }
        return result
    except Exception:
        return {}

def upsert_essentia_cache_entries(entries):
    """Batch-upserts a dict of {rk: data} into the SQLite database."""
    if not entries:
        return
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executemany("""
            INSERT OR REPLACE INTO essentia_cache
            (rating_key, bpm, key, energy, year, artist, genres, styles, moods, file_path, last_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (rk, d.get("bpm"), d.get("key"), d.get("energy"), d.get("year"),
             d.get("artist"), json.dumps(d.get("genres") or []),
             json.dumps(d.get("styles") or []),
             json.dumps(d.get("moods") or []), d.get("file_path"), d.get("last_synced"))
            for rk, d in entries.items()
        ])
        conn.commit()
        conn.close()
    except Exception as e:
        log_msg(f"[ERROR] Cache upsert failed: {e}", level="error")

# --- 2. WORKER WRAPPER ---

def analysis_worker(track_id):
    """
    Worker function to fetch and analyze a single track.
    RatingKey is passed instead of the object to minimize pickling overhead.
    Utilizes a local Plex session to ensure stability across multiple processes.
    """
    global worker_plex
    try:
        # Breadcrumb log to identify the file if the process hard-crashes
        logger.info(f"Attempting analysis on Track ID: {track_id}")

        track = worker_plex.fetchItem(track_id)
        result = analyze_track_essentia(track)
        return str(track_id), result, None
    except Exception as e:
        return str(track_id), None, str(e)

# --- 3. CORE ANALYSIS LOGIC ---

def bulk_analyze():
    """Iterates through the entire library to perform deep acoustic analysis."""
    if not ESSENTIA_ENABLED:
        log_msg("[ERROR] Essentia is not installed or enabled. Analysis cannot proceed.")
        return

    log_msg(f"--- Starting Parallel Library Analysis & Metadata Sync: {MUSIC_LIBRARY} ---")

    # Pre-load cache to filter processing list
    current_cache = load_essentia_cache_exclusive()
    _essentia_cache.update(current_cache)

    # Fetch all tracks as a flat list
    local_plex_main = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=60)
    music_section = local_plex_main.library.section(MUSIC_LIBRARY)
    log_msg("Fetching tracks from Plex... (This may take a minute)")
    all_tracks = music_section.search(libtype='track')

    # 1. Filter the processing list to avoid redundant work
    now_ts = datetime.now().timestamp()
    to_process = []
    for t in all_tracks:
        rk = str(t.ratingKey)
        cached_data = _essentia_cache.get(rk)
        # Only process if not in cache, or if the metadata sync is older than 7 days
        if cached_data is None:
            to_process.append(t.ratingKey)
        else:
            last_sync = cached_data.get("last_synced", 0)
            # Check staleness first — stale tracks are reprocessed regardless of path,
            # so we can skip the get_local_path() call entirely for them.
            if (now_ts - last_sync) > 604800:
                to_process.append(t.ratingKey)
            elif cached_data.get("file_path") != get_local_path(t):
                to_process.append(t.ratingKey)

    num_to_process = len(to_process)
    if num_to_process == 0:
        log_msg("--- Success! All tracks are analyzed and metadata is up to date. ---")
        return

    log_msg(f"Found {len(all_tracks)} total tracks. Processing {num_to_process} for analysis/sync...")

    start_time = time.time()
    completed = 0
    # Accumulates only newly completed entries — avoids writing the entire cache
    # (which may contain hundreds of thousands of pre-existing entries) on each periodic save.
    pending_saves = {}

    # 2. Maximize workers while managing memory via sliding window submission
    workers = get_optimal_workers(task_type="cpu")
    to_process_iter = iter(to_process)
    future_to_track = {}

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        max_tasks_per_child=50, # Increased to reduce worker restart overhead
        initializer=init_worker
    ) as executor:

        # Prime the pump: Submit an initial batch (workers * 4)
        for _ in range(min(len(to_process), workers * 4)):
            tid = next(to_process_iter)
            future_to_track[executor.submit(analysis_worker, tid)] = tid

        # Process as they complete and refill the window
        while future_to_track:
            # Wait for at least one future to finish
            done, _ = concurrent.futures.wait(
                future_to_track.keys(),
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                track_id = future_to_track.pop(future)
                try:
                    # Retrieve result with existing timeout
                    rk, data, error = future.result(timeout=90)

                    if data:
                        _essentia_cache[rk] = data
                        pending_saves[rk] = data
                        completed += 1
                    elif error:
                        log_msg(f"\n[SKIP] Track {track_id} failed: {error}", level="error")
                        completed += 1

                except concurrent.futures.TimeoutError:
                    log_msg(f"\n[TIMEOUT] Track {track_id} took too long. Skipping.", level="error")
                    completed += 1
                except Exception as e:
                    log_msg(f"\n[ERROR] Unexpected error on {track_id}: {e}", level="error")
                    completed += 1

                # IMMEDIATELY SUBMIT NEXT TASK to keep workers busy
                try:
                    next_tid = next(to_process_iter)
                    future_to_track[executor.submit(analysis_worker, next_tid)] = next_tid
                except StopIteration:
                    pass

                # Periodic Save: Upsert only newly completed entries every 100 completions
                if completed % 100 == 0 and pending_saves:
                    upsert_essentia_cache_entries(pending_saves)
                    pending_saves.clear()

                # Update Progress UI (Identical to original)
                elapsed = time.time() - start_time
                avg = elapsed / max(1, completed)
                est = timedelta(seconds=int((num_to_process - completed) * avg))
                log_msg(f"Progress: [{completed}/{num_to_process}] | Est: {est} | Cache: {len(_essentia_cache)} ", end='\r')


    # Final persistent save: flush any entries not yet written by a periodic save
    if pending_saves:
        upsert_essentia_cache_entries(pending_saves)
    log_msg(f"\n--- Success! Analysis and Sync complete. ---")
    log_msg(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()
