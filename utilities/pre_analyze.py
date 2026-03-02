import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import signal
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

# Spawned worker processes re-import this module from scratch. Using mode='w' in a
# worker would truncate the log file. Main process uses 'w' (fresh log each run);
# all other processes use 'a' so worker breadcrumb entries are preserved.
_log_mode = 'w' if multiprocessing.current_process().name == 'MainProcess' else 'a'
file_handler = logging.FileHandler(LOG_FILE, mode=_log_mode, encoding='utf-8')
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
    # Raise the soft FD limit to the hard limit. Essentia's native C++ decoders can leak
    # file descriptors during audio analysis. On Linux the default soft limit (1024) can
    # be exhausted after several hundred tracks per worker, causing the result pipe write
    # to block and hanging the future with no visible error.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass
    # Pre-load the Essentia cache into this worker's memory. Spawned workers start clean
    # and do NOT inherit _essentia_cache from the parent. Without this,
    # analyze_track_essentia() always falls through to full acoustic analysis
    # (MonoLoader + RhythmExtractor2013 etc.) even for tracks that only need a fast
    # metadata backfill — turning a ~1 s tag update into a ~60 s C++ analysis.
    current_cache = load_essentia_cache_exclusive()
    _essentia_cache.update(current_cache)
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

def _sigalrm_handler(signum, frame):
    raise TimeoutError("Essentia analysis exceeded per-track time limit")

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

        # Per-track timeout: Essentia's C++ decoders and os.path.exists() on stalled
        # network mounts can block indefinitely without raising an exception. SIGALRM
        # fires in this worker's main thread and converts the hang into a catchable
        # TimeoutError, skipping just this track rather than freezing the whole pool
        # until the pool-level HANG_TIMEOUT fires. SIGALRM is Unix-only; on other
        # platforms the pool-level HANG_TIMEOUT remains the backstop.
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _sigalrm_handler)
            signal.alarm(90)
        try:
            result = analyze_track_essentia(track)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

        return str(track_id), result, None
    except Exception as e:
        return str(track_id), None, str(e)

# --- 3. CORE ANALYSIS LOGIC ---

def bulk_analyze():
    """
    Scans the full Plex library and runs Essentia acoustic analysis on every unanalysed
    or stale track. Staleness is defined as either:
      - Not present in the cache at all, or
      - last_synced is more than 7 days ago (metadata may have changed), or
      - file_path has changed (track was moved/remounted).

    Uses a ProcessPoolExecutor so Essentia's CPU-bound analysis runs in true parallel.
    Results are saved periodically (every 2 min or 500 entries) to limit data loss
    if the process is interrupted.
    """
    if not ESSENTIA_ENABLED:
        log_msg("[INFO] Essentia is disabled. Running in metadata-only mode (styles, moods, genres, year).")
        log_msg("[INFO] To enable acoustic analysis (BPM, key, energy), set essentia: enabled: true in config.yml.")

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

    # Pre-build a ratingKey → local file path map for the tracks we're about to process.
    # Used only for diagnostic logging when a hung worker is detected — avoids needing
    # Plex API calls at the point of failure.
    to_process_set = set(to_process)
    track_path_map = {t.ratingKey: get_local_path(t) for t in all_tracks if t.ratingKey in to_process_set}

    start_time = time.time()
    last_save = start_time
    completed = 0
    # Accumulates only newly completed entries — avoids writing the entire cache
    # (which may contain hundreds of thousands of pre-existing entries) on each periodic save.
    pending_saves = {}

    # 2. Submit ALL tasks upfront — the executor's internal queue keeps workers busy
    # without the overhead of a manual sliding-window refill loop.
    workers = get_optimal_workers(task_type="cpu")

    # HANG DETECTION: as_completed() blocks indefinitely if a worker is stuck in native
    # Essentia C++ code (e.g. a malformed audio file). The per-future result(timeout=90)
    # never fires because as_completed() never yields a future that hasn't completed yet.
    # Replacing with wait() + a pool-level timeout lets us detect and escape a hung pool.
    # We also avoid the `with` context manager so __exit__ doesn't block on hung workers.
    HANG_TIMEOUT = 120  # seconds without ANY completion before declaring the pool hung

    # CROSS-PLATFORM PROCESS ISOLATION: 'spawn' starts each worker as a clean Python
    # interpreter on all platforms (Windows, macOS, Linux). It avoids the fork-after-
    # threads deadlock that Linux's default 'fork' method can cause when forking from a
    # multi-threaded parent. Workers pre-load the Essentia cache in init_worker() since
    # spawned processes do not inherit the parent's in-memory state.
    mp_ctx = multiprocessing.get_context('spawn')

    # max_tasks_per_child is intentionally omitted. With uniform-duration tasks (e.g.
    # metadata backfill completing in ~1-2 s each), all N workers hit their per-child
    # limit at exactly the same time (N × limit completions), causing a simultaneous
    # restart stall that HANG_TIMEOUT misidentifies as a hang. Workers live for the
    # full run; memory growth is negligible for the fast backfill path.
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        mp_context=mp_ctx
    )
    try:
        futures = {executor.submit(analysis_worker, tid): tid for tid in to_process}

        pending = set(futures.keys())
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=HANG_TIMEOUT,
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            if not done:
                # No future completed within HANG_TIMEOUT — all remaining workers are hung.
                stuck_ids = [futures[f] for f in pending]
                log_msg(
                    f"\n[HUNG] No progress for {HANG_TIMEOUT}s. "
                    f"Skipping {len(stuck_ids)} stuck track(s). "
                    f"Check pre_analyze.log for file details.",
                    level="error"
                )
                for stuck_id in stuck_ids:
                    path = track_path_map.get(stuck_id, "path unknown")
                    logger.error(f"[HUNG] Stuck: Track ID={stuck_id} | File={path}")
                completed += len(stuck_ids)
                break

            for future in done:
                track_id = futures[future]
                try:
                    rk, data, error = future.result()  # already done — returns immediately

                    if data:
                        _essentia_cache[rk] = data
                        pending_saves[rk] = data
                        completed += 1
                    elif error:
                        log_msg(f"\n[SKIP] Track {track_id} failed: {error}", level="error")
                        completed += 1

                except Exception as e:
                    log_msg(f"\n[ERROR] Unexpected error on {track_id}: {e}", level="error")
                    completed += 1

                # Periodic Save: time-based (every 2 min) with a size cap safety valve.
                now = time.time()
                if pending_saves and (now - last_save >= 120 or len(pending_saves) >= 500):
                    upsert_essentia_cache_entries(pending_saves)
                    pending_saves.clear()
                    last_save = now

                # Update Progress UI
                elapsed = now - start_time
                avg = elapsed / max(1, completed)
                est = timedelta(seconds=int((num_to_process - completed) * avg))
                log_msg(f"Progress: [{completed}/{num_to_process}] | Est: {est} | Cache: {len(_essentia_cache)} ", end='\r')

    finally:
        # wait=False: don't block on any workers that may still be running (e.g. hung ones).
        # The worker processes are children of this process and will be cleaned up on exit.
        executor.shutdown(wait=False)


    # Final persistent save: flush any entries not yet written by a periodic save
    if pending_saves:
        upsert_essentia_cache_entries(pending_saves)
    log_msg(f"\n--- Success! Analysis and Sync complete. ---")
    log_msg(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()
