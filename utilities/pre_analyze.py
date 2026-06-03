import os
import sys
# Suppress TensorFlow's CUDA probe warnings — inherited by worker subprocesses at fork time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS",  "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import atexit
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
    get_local_path, _migrate_json_to_sqlite, get_optimal_workers, _ensure_db_schema,
    _fill_missing_acoustic, _TF_MODELS_LOADED,
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
    # Do NOT load the full cache here. The existing cache entry for each track is passed
    # as an argument by the main process (which already has _essentia_cache loaded).
    # Loading all 57k+ entries in every worker on every max_tasks_per_child respawn
    # caused all workers to simultaneously hammer SQLite, hanging the pool at N*100 tracks.
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
        _ensure_db_schema(conn)
        rows = conn.execute(
            "SELECT rating_key, bpm, key, energy, danceability, brightness, year, artist, genres, styles, moods, file_path, "
            "track_updated_at, album_updated_at, artist_updated_at, "
            "beat_confidence, integrated_loudness, onset_rate, dynamic_complexity, "
            "arousal, valence, vocal_presence "
            "FROM essentia_cache"
        ).fetchall()
        conn.close()
        result = {}
        for rk, bpm, key, energy, danceability, brightness, year, artist, \
                genres_j, styles_j, moods_j, file_path, \
                track_updated_at, album_updated_at, artist_updated_at, \
                beat_confidence, integrated_loudness, onset_rate, dynamic_complexity, \
                arousal, valence, vocal_presence in rows:
            result[rk] = {
                "bpm": bpm, "key": key, "energy": energy, "danceability": danceability, "brightness": brightness,
                "year": year, "artist": artist,
                "genres": json.loads(genres_j) if genres_j else [],
                "styles": json.loads(styles_j) if styles_j else [],
                "moods":  json.loads(moods_j) if moods_j else [],
                "file_path": file_path,
                "track_updated_at": track_updated_at,
                "album_updated_at": album_updated_at,
                "artist_updated_at": artist_updated_at,
                "beat_confidence": beat_confidence,
                "integrated_loudness": integrated_loudness,
                "onset_rate": onset_rate,
                "dynamic_complexity": dynamic_complexity,
                "arousal": arousal,
                "valence": valence,
                "vocal_presence": vocal_presence,
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
        _ensure_db_schema(conn)
        conn.executemany("""
            INSERT OR REPLACE INTO essentia_cache
            (rating_key, bpm, key, energy, year, artist, genres, styles, moods, file_path,
             track_updated_at, album_updated_at, artist_updated_at, danceability, brightness,
             beat_confidence, integrated_loudness, onset_rate, dynamic_complexity,
             arousal, valence, vocal_presence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (rk, d.get("bpm"), d.get("key"), d.get("energy"), d.get("year"),
             d.get("artist"), json.dumps(d.get("genres") or []),
             json.dumps(d.get("styles") or []),
             json.dumps(d.get("moods") or []), d.get("file_path"),
             d.get("track_updated_at"), d.get("album_updated_at"), d.get("artist_updated_at"),
             d.get("danceability"), d.get("brightness"),
             d.get("beat_confidence"), d.get("integrated_loudness"),
             d.get("onset_rate"), d.get("dynamic_complexity"),
             d.get("arousal"), d.get("valence"), d.get("vocal_presence"))
            for rk, d in entries.items()
        ])
        conn.commit()
        conn.close()
    except Exception as e:
        log_msg(f"[ERROR] Cache upsert failed: {e}", level="error")

# --- 2. WORKER WRAPPER ---

def _sigalrm_handler(signum, frame):
    raise TimeoutError("Essentia analysis exceeded per-track time limit")

def analysis_worker(track_id, plex_track_ts=None, plex_album_ts=None, plex_artist_ts=None, existing_entry=None):
    """
    Worker function to fetch and analyze a single track.
    RatingKey is passed instead of the object to minimize pickling overhead.
    Utilizes a local Plex session to ensure stability across multiple processes.
    The three Plex timestamps are passed in from the main process (built from bulk
    album/artist fetches) so workers don't need extra per-track API calls.
    existing_entry is the cached data for this track from the main process's loaded cache,
    injected into _essentia_cache so analyze_track_essentia() can take the fast metadata
    refresh path rather than always running the full Essentia pipeline.
    """
    global worker_plex
    try:
        # Inject the existing cache entry so analyze_track_essentia() can take the fast
        # metadata-refresh path (or backfill only missing acoustic fields) without loading
        # the full 57k-entry cache in every worker.
        if existing_entry is not None:
            _essentia_cache[str(track_id)] = existing_entry

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
            result = analyze_track_essentia(track, plex_track_ts, plex_album_ts, plex_artist_ts)
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

    # Fetch all tracks as a flat list.
    # Use a longer timeout here — fetching 50k+ tracks in a single response can take
    # several minutes, well beyond the 120s used for normal per-track API calls.
    try:
        local_plex_main = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=600)
        music_section = local_plex_main.library.section(MUSIC_LIBRARY)
        log_msg("Fetching tracks from Plex... (This may take a minute)")
        _raw_tracks = music_section.search(libtype='track', container_size=5000)
    except Exception as e:
        log_msg(f"[ERROR] Failed to fetch tracks from Plex: {e}", level="error")
        log_msg("[ERROR] Check that Plex is reachable and try again.", level="error")
        return

    # Extract only the fields we need from each track object, then release the full
    # PlexAPI objects. At 280k+ tracks they can consume 1GB+ of memory; the slim
    # dicts below are ~50MB for the same library size.
    track_infos = [
        {
            "ratingKey":            t.ratingKey,
            "updatedAt":            t.updatedAt.timestamp() if getattr(t, "updatedAt", None) else None,
            "parentRatingKey":      str(t.parentRatingKey),
            "grandparentRatingKey": str(t.grandparentRatingKey),
            "localPath":            get_local_path(t),
        }
        for t in _raw_tracks
    ]
    total_tracks = len(track_infos)
    del _raw_tracks  # release ~1GB of PlexAPI objects

    # Fetch albums and artists to get their updatedAt timestamps.
    # These are not available on the track object directly, so we build lookup dicts
    # keyed by ratingKey and pass the relevant timestamps into each worker.
    log_msg("Fetching album and artist timestamps from Plex...")
    try:
        _raw_albums  = music_section.search(libtype='album',  container_size=5000)
        _raw_artists = music_section.search(libtype='artist', container_size=5000)
        album_updated_map  = {
            str(a.ratingKey): a.updatedAt.timestamp() if getattr(a, "updatedAt", None) else None
            for a in _raw_albums
        }
        artist_updated_map = {
            str(a.ratingKey): a.updatedAt.timestamp() if getattr(a, "updatedAt", None) else None
            for a in _raw_artists
        }
        del _raw_albums, _raw_artists
    except Exception as e:
        log_msg(f"[WARN] Could not fetch album/artist timestamps: {e}. Album/artist staleness detection disabled.")
        album_updated_map  = {}
        artist_updated_map = {}

    # 1. Filter the processing list to avoid redundant work.
    # Re-process a track only if Plex reports a change at the track, album, or artist
    # level (via updatedAt comparison), the file path changed, or acoustic fields are null.
    to_process = []
    ts_map = {}  # ratingKey → (plex_track_ts, plex_album_ts, plex_artist_ts)

    for info in track_infos:
        rk = str(info["ratingKey"])
        cached_data = _essentia_cache.get(rk)

        plex_track_ts  = info["updatedAt"]
        plex_album_ts  = album_updated_map.get(info["parentRatingKey"])
        plex_artist_ts = artist_updated_map.get(info["grandparentRatingKey"])

        if cached_data is None:
            to_process.append(info["ratingKey"])
            ts_map[info["ratingKey"]] = (plex_track_ts, plex_album_ts, plex_artist_ts)
        else:
            timestamps_changed = (
                cached_data.get("track_updated_at")  != plex_track_ts  or
                cached_data.get("album_updated_at")  != plex_album_ts  or
                cached_data.get("artist_updated_at") != plex_artist_ts
            )
            if timestamps_changed:
                to_process.append(info["ratingKey"])
                ts_map[info["ratingKey"]] = (plex_track_ts, plex_album_ts, plex_artist_ts)
            elif cached_data.get("file_path") != info["localPath"]:
                to_process.append(info["ratingKey"])
                ts_map[info["ratingKey"]] = (plex_track_ts, plex_album_ts, plex_artist_ts)
            elif cached_data.get("energy") is not None and any(
                cached_data.get(f) is None for f in (
                    "danceability", "brightness",
                    "beat_confidence", "integrated_loudness",
                    "onset_rate", "dynamic_complexity",
                )
            ):
                # Backfill base Essentia features via parallel workers.
                # arousal/valence/vocal_presence are TF features handled in the main-process
                # post-pass below — loading TF models in every worker simultaneously causes OOM.
                # Timestamps are intentionally preserved — this is not a metadata re-sync.
                to_process.append(info["ratingKey"])
                ts_map[info["ratingKey"]] = (
                    cached_data.get("track_updated_at"),
                    cached_data.get("album_updated_at"),
                    cached_data.get("artist_updated_at"),
                )

    num_to_process = len(to_process)
    if num_to_process == 0 and not _TF_MODELS_LOADED:
        log_msg("--- Success! All tracks are analyzed and metadata is up to date. ---")
        return

    if num_to_process > 0:
        log_msg(f"Found {total_tracks} total tracks. Processing {num_to_process} for analysis/sync...")

        # Pre-build a ratingKey → local file path map for the tracks we're about to process.
        # Used only for diagnostic logging when a hung worker is detected — avoids needing
        # Plex API calls at the point of failure.
        to_process_set = set(to_process)
        track_path_map = {info["ratingKey"]: info["localPath"] for info in track_infos if info["ratingKey"] in to_process_set}

        start_time = time.time()
        last_save = start_time
        completed = 0
        # Accumulates only newly completed entries — avoids writing the entire cache
        # (which may contain hundreds of thousands of pre-existing entries) on each periodic save.
        pending_saves = {}

        workers = get_optimal_workers(task_type="cpu")

        # MEMORY-ADAPTIVE BATCH SIZING
        # Essentia's C++ audio decoders accumulate heap memory that Python's GC cannot free.
        # We reclaim it by recycling the entire executor between batches (creating a fresh pool
        # with clean workers). This avoids the Python 3.12 bug where all workers hitting
        # max_tasks_per_child simultaneously causes the management thread to deadlock.
        #
        # Initial sizing uses conservative estimates; after the first batch completes we
        # measure actual worker RSS (base + growth) and recalculate all remaining batch
        # sizes from real data, maximising how many tracks fit before workers need recycling.
        #
        # 75% of psutil.available (which already excludes Plex/OS/main process) is assigned
        # to the pool; 25% is headroom for allocation spikes.
        try:
            import psutil as _psutil
            _psutil_ok       = True
            _avail_mb        = _psutil.virtual_memory().available // (1024 * 1024)
            _pool_mb         = int(_avail_mb * 0.75)
            _per_worker_mb   = _pool_mb // max(1, workers)
            _budget_mb       = max(0, _per_worker_mb - 150)   # 150 MB assumed base RSS
            _tasks_per_worker = max(50, min(2000, int(_budget_mb / 0.8)))
        except Exception:
            _psutil_ok        = False
            _avail_mb         = 0
            _pool_mb          = 0
            _per_worker_mb    = 0
            _budget_mb        = 0
            _tasks_per_worker = 400

        def _sample_worker_rss_mb(ex):
            """Return average RSS (MB) across all living workers in executor ex."""
            if not _psutil_ok:
                return None
            samples = []
            for pid in list((getattr(ex, "_processes", None) or {})):
                try:
                    samples.append(_psutil.Process(pid).memory_info().rss / (1024 * 1024))
                except (_psutil.NoSuchProcess, _psutil.AccessDenied, ProcessLookupError):
                    pass
            return (sum(samples) / len(samples)) if samples else None

        batch_size = _tasks_per_worker * workers
        to_process_list = list(to_process)
        batches = [to_process_list[i:i + batch_size] for i in range(0, len(to_process_list), batch_size)]
        log_msg(f"[INFO] Initial batch size: ~{batch_size} tracks "
                f"({_tasks_per_worker} tasks/worker, {workers} workers) — will recalibrate after first batch",
                log_only=True)

        # HANG DETECTION constants
        HANG_TIMEOUT = 120   # seconds without any completion before declaring pool hung
        HEARTBEAT    = 10    # wait() poll interval for progress updates

        # CROSS-PLATFORM PROCESS ISOLATION: 'spawn' starts each worker as a clean Python
        # interpreter, avoiding fork-after-threads deadlocks.
        mp_ctx = multiprocessing.get_context('spawn')

        # Mutable reference so the atexit handler always kills the current executor,
        # even when it is replaced between batches.
        _active_executor = [None]

        def _kill_active_workers():
            ex = _active_executor[0]
            if ex is None:
                return
            for proc in (getattr(ex, "_processes", None) or {}).values():
                try:
                    proc.kill()
                except Exception:
                    pass

        atexit.register(_kill_active_workers)

        _CONN_ERROR_PHRASES = ("connection refused", "connection reset", "failed to establish", "connectionerror")
        CONN_FAILURE_THRESHOLD = 5
        consecutive_conn_failures = 0
        aborted = False

        # RSS calibration state — populated during batch 0, used to resize batch 1+
        _baseline_rss_mb  = None   # worker RSS before any tasks (set early in batch 0)
        _baseline_tasks   = 0      # completed count when baseline was sampled

        # Prevent TF model loading in worker processes. With 'spawn' context each worker
        # re-imports meloday.py; the TF C++ runtime + three .pb models uses ~700 MB–1 GB
        # per worker and causes OOM when N workers start simultaneously. TF features
        # (arousal/valence/vocal_presence) are filled in the main-process TF post-pass below.
        os.environ['MELODAY_SKIP_TF_MODELS'] = '1'

        batch_idx = 0
        while batch_idx < len(batches) and not aborted:
            batch = batches[batch_idx]

            if batch_idx > 0:
                log_msg(f"\n[INFO] Batch {batch_idx + 1}/{len(batches)}: "
                        f"recycling workers to reclaim C++ memory ({len(batch)} tracks)...",
                        log_only=True)

            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                mp_context=mp_ctx,
                # No max_tasks_per_child — batch sizing controls worker lifetime instead,
                # avoiding the Python 3.12 simultaneous-restart deadlock.
            )
            _active_executor[0] = executor
            completed_in_batch = 0

            try:
                futures = {
                    executor.submit(analysis_worker, tid, *ts_map[tid], existing_entry=_essentia_cache.get(str(tid))): tid
                    for tid in batch
                }

                pending = set(futures.keys())
                last_completion_time = time.monotonic()
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending, timeout=HEARTBEAT,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    if not done:
                        stall_secs = int(time.monotonic() - last_completion_time)
                        if stall_secs >= HANG_TIMEOUT:
                            stuck_ids = [futures[f] for f in pending]
                            log_msg(
                                f"\n[HUNG] No progress for {stall_secs}s. "
                                f"Skipping {len(stuck_ids)} stuck track(s). "
                                f"Check pre_analyze.log for file details.",
                                level="error"
                            )
                            for stuck_id in stuck_ids:
                                path = track_path_map.get(stuck_id, "path unknown")
                                logger.error(f"[HUNG] Stuck: Track ID={stuck_id} | File={path}")
                            completed += len(stuck_ids)
                            for proc in (getattr(executor, "_processes", None) or {}).values():
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                            aborted = True
                            break
                        continue

                    last_completion_time = time.monotonic()
                    for future in done:
                        track_id = futures.pop(future)
                        try:
                            rk, data, error = future.result()

                            if data:
                                consecutive_conn_failures = 0
                                _essentia_cache[rk] = data
                                pending_saves[rk] = data
                                completed += 1
                                completed_in_batch += 1
                            elif error:
                                log_msg(f"\n[SKIP] Track {track_id} failed: {error}", level="error")
                                completed += 1
                                completed_in_batch += 1
                                error_lower = str(error).lower()
                                if any(p in error_lower for p in _CONN_ERROR_PHRASES):
                                    consecutive_conn_failures += 1
                                    if consecutive_conn_failures >= CONN_FAILURE_THRESHOLD:
                                        log_msg(
                                            f"\n[ABORT] {CONN_FAILURE_THRESHOLD} consecutive connection failures — "
                                            f"Plex appears to be down. Stopping early.",
                                            level="error"
                                        )
                                        aborted = True
                                        break
                                else:
                                    consecutive_conn_failures = 0

                        except Exception as e:
                            log_msg(f"\n[ERROR] Unexpected error on {track_id}: {e}", level="error")
                            completed += 1
                            completed_in_batch += 1

                    if aborted:
                        break

                    # Sample baseline RSS once — after enough tasks that workers have settled
                    # but before they've accumulated much leak. Doing this mid-batch (while
                    # workers are alive) avoids the need for a separate warm-up phase.
                    if batch_idx == 0 and _baseline_rss_mb is None and completed_in_batch >= workers * 2:
                        _baseline_rss_mb = _sample_worker_rss_mb(executor)
                        _baseline_tasks  = completed_in_batch

                    # Periodic Save
                    now = time.time()
                    if pending_saves and (now - last_save >= 120 or len(pending_saves) >= 500):
                        upsert_essentia_cache_entries(pending_saves)
                        pending_saves.clear()
                        last_save = now

                    # Progress UI
                    elapsed = now - start_time
                    avg = elapsed / max(1, completed)
                    est = timedelta(seconds=int((num_to_process - completed) * avg))
                    log_msg(f"Progress: [{completed}/{num_to_process}] | Est: {est} | Cache: {len(_essentia_cache)} ", end='\r')

            finally:
                # Measure peak RSS before killing workers — used to recalibrate batch sizing.
                peak_rss_mb = _sample_worker_rss_mb(executor)

                executor.shutdown(wait=False)
                for proc in (getattr(executor, "_processes", None) or {}).values():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                _active_executor[0] = None

                # Recalibrate after every batch using real measured values.
                # Leak per task = (peak RSS − baseline RSS) / tasks completed since baseline.
                # Re-query available memory now (not startup) so we use whatever headroom
                # the system actually has at this moment, not a stale startup snapshot.
                tasks_since_baseline = max(1, completed_in_batch - _baseline_tasks)
                if (peak_rss_mb and _baseline_rss_mb and _psutil_ok
                        and completed_in_batch > workers * 4):
                    try:
                        _cur_avail_mb    = _psutil.virtual_memory().available // (1024 * 1024)
                        _cur_pool_mb     = int(_cur_avail_mb * 0.85)   # 85%: available already excludes other processes
                        _cur_per_wkr_mb  = _cur_pool_mb // max(1, workers)
                    except Exception:
                        _cur_per_wkr_mb  = _per_worker_mb              # fall back to startup value

                    actual_base_mb   = _baseline_rss_mb
                    actual_leak_mb   = max(0.01, (peak_rss_mb - _baseline_rss_mb) / tasks_since_baseline)
                    new_budget_mb    = max(0, _cur_per_wkr_mb - actual_base_mb)
                    remaining        = [tid for b in batches[batch_idx + 1:] for tid in b]
                    # Cap at remaining work — no point in a batch larger than what's left.
                    practical_max    = max(50, len(remaining) // max(1, workers) + 1)
                    new_tasks        = max(50, min(practical_max, int(new_budget_mb / actual_leak_mb)))
                    new_batch_size   = new_tasks * workers
                    _tasks_per_worker = new_tasks
                    if remaining:
                        batches[batch_idx + 1:] = [
                            remaining[i:i + new_batch_size]
                            for i in range(0, len(remaining), new_batch_size)
                        ]
                    log_msg(
                        f"\n[INFO] Recalibrated: base={actual_base_mb:.0f} MB/worker, "
                        f"leak={actual_leak_mb:.3f} MB/task, avail={_cur_avail_mb:.0f} MB → "
                        f"{new_tasks} tasks/worker, ~{new_batch_size} tracks/batch "
                        f"({len(batches) - batch_idx - 1} batch(es) remaining)",
                        log_only=True
                    )

            batch_idx += 1

        # Flush remaining parallel-phase saves
        if pending_saves:
            upsert_essentia_cache_entries(pending_saves)

    del track_infos  # free ~1 GB of PlexAPI track data regardless of whether workers ran

    # --- BASE-FEATURES POST-PASS ---
    # Catches tracks whose onset_rate / dynamic_complexity / integrated_loudness are still
    # null after the parallel phase — typically tracks from a prior cascade-failure run whose
    # timestamps matched Plex (so they went through the silent backfill path in the worker
    # and any MonoLoader error was swallowed). Running single-threaded in the main process
    # makes failures visible via logger and ensures saves commit before TF post-pass begins.
    _BASE_FIELDS = ("onset_rate", "dynamic_complexity", "integrated_loudness")
    base_todo = [
        (rk, data["file_path"])
        for rk, data in _essentia_cache.items()
        if data.get("energy") is not None
        and data.get("file_path")
        and any(data.get(f) is None for f in _BASE_FIELDS)
    ]
    if base_todo:
        log_msg(f"\n[INFO] Base-features post-pass: {len(base_todo)} tracks still need "
                f"onset_rate / dynamic_complexity / integrated_loudness.")
        log_msg(f"[INFO] Running single-threaded in main process. Re-run to resume if interrupted.")
        base_pending = {}
        base_start = time.time()
        for base_i, (rk, file_path) in enumerate(base_todo):
            data = _essentia_cache[rk]
            try:
                _fill_missing_acoustic(data, file_path)
            except Exception as e:
                logger.error(f"[BASE] _fill_missing_acoustic failed for rk={rk}: {e}")
            _essentia_cache[rk] = data
            base_pending[rk] = data
            if len(base_pending) >= 200 or base_i == len(base_todo) - 1:
                upsert_essentia_cache_entries(base_pending)
                base_pending.clear()
            base_elapsed = time.time() - base_start
            base_avg = base_elapsed / max(1, base_i + 1)
            base_eta = timedelta(seconds=int((len(base_todo) - base_i - 1) * base_avg))
            log_msg(f"Base: [{base_i + 1}/{len(base_todo)}] | {base_avg:.1f}s/track | Est: {base_eta} ", end='\r')
        log_msg(f"\n[INFO] Base-features post-pass complete.")

    # --- TF POST-PASS ---
    # arousal, valence, vocal_presence require TF inference. Loading the TF C++ runtime
    # and model weights (~700 MB–1 GB) in every parallel worker simultaneously causes OOM.
    # Instead, the main process — which loaded models at import time before
    # MELODAY_SKIP_TF_MODELS was set for workers — runs inference sequentially here.
    # Single-threaded but memory-safe. Saves every 200 tracks so it is fully interruptible:
    # Ctrl+C to pause, re-run pre_analyze.py to resume from where it left off.
    if _TF_MODELS_LOADED:
        tf_todo = [
            (rk, data["file_path"])
            for rk, data in _essentia_cache.items()
            if data.get("energy") is not None
            and data.get("file_path")
            and any(data.get(f) is None for f in ("arousal", "valence", "vocal_presence"))
        ]
        if tf_todo:
            log_msg(f"\n[INFO] TF post-pass: {len(tf_todo)} tracks need arousal/valence/vocal_presence.")
            log_msg(f"[INFO] Running single-threaded in main process. Ctrl+C pauses; re-run to resume.")
            tf_pending = {}
            tf_start = time.time()
            for tf_i, (rk, file_path) in enumerate(tf_todo):
                data = _essentia_cache[rk]
                try:
                    _fill_missing_acoustic(data, file_path)
                except Exception as e:
                    logger.error(f"[TF] Failed for {rk}: {e}")
                _essentia_cache[rk] = data
                tf_pending[rk] = data
                if len(tf_pending) >= 200 or tf_i == len(tf_todo) - 1:
                    upsert_essentia_cache_entries(tf_pending)
                    tf_pending.clear()
                tf_elapsed = time.time() - tf_start
                tf_avg = tf_elapsed / max(1, tf_i + 1)
                tf_eta = timedelta(seconds=int((len(tf_todo) - tf_i - 1) * tf_avg))
                log_msg(f"TF: [{tf_i + 1}/{len(tf_todo)}] | {tf_avg:.1f}s/track | Est: {tf_eta} ", end='\r')
            log_msg(f"\n[INFO] TF post-pass complete.")

    log_msg(f"\n--- Success! Analysis and Sync complete. ---")
    log_msg(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()
