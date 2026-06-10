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
    _mood_models, _moodtheme_model, _genre_model,
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
            "arousal, valence, vocal_presence, "
            "mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, mood_acoustic, "
            "mood_electronic, danceability_hl, moodtheme, genre_discogs, "
            "lastfm_artist_tags, lastfm_track_tags, artist_origin, lastfm_listeners, "
            "lyric_valence, lyric_themes, lyric_lang, "
            "title, release_date, lastfm_synced_at, geo_synced_at, lyrics_synced_at "
            "FROM essentia_cache"
        ).fetchall()
        conn.close()
        result = {}
        for rk, bpm, key, energy, danceability, brightness, year, artist, \
                genres_j, styles_j, moods_j, file_path, \
                track_updated_at, album_updated_at, artist_updated_at, \
                beat_confidence, integrated_loudness, onset_rate, dynamic_complexity, \
                arousal, valence, vocal_presence, \
                mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, mood_acoustic, \
                mood_electronic, danceability_hl, moodtheme_j, genre_discogs_j, \
                lastfm_artist_j, lastfm_track_j, artist_origin_j, lastfm_listeners, \
                lyric_valence, lyric_themes_j, lyric_lang, \
                title, release_date, lastfm_synced_at, geo_synced_at, lyrics_synced_at in rows:
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
                "mood_happy": mood_happy, "mood_sad": mood_sad, "mood_aggressive": mood_aggressive,
                "mood_relaxed": mood_relaxed, "mood_party": mood_party, "mood_acoustic": mood_acoustic,
                "mood_electronic": mood_electronic, "danceability_hl": danceability_hl,
                "moodtheme": json.loads(moodtheme_j) if moodtheme_j else None,
                "genre_discogs": json.loads(genre_discogs_j) if genre_discogs_j else None,
                "lastfm_artist_tags": json.loads(lastfm_artist_j) if lastfm_artist_j else None,
                "lastfm_track_tags": json.loads(lastfm_track_j) if lastfm_track_j else None,
                "artist_origin": json.loads(artist_origin_j) if artist_origin_j else None,
                "lastfm_listeners": lastfm_listeners,
                "lyric_valence": lyric_valence,
                "lyric_themes": json.loads(lyric_themes_j) if lyric_themes_j else None,
                "lyric_lang": lyric_lang,
                "title": title, "release_date": release_date,
                "lastfm_synced_at": lastfm_synced_at, "geo_synced_at": geo_synced_at,
                "lyrics_synced_at": lyrics_synced_at,
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
             arousal, valence, vocal_presence,
             mood_happy, mood_sad, mood_aggressive, mood_relaxed, mood_party, mood_acoustic,
             mood_electronic, danceability_hl, moodtheme, genre_discogs, emb_effnet, emb_musicnn,
             lastfm_artist_tags, lastfm_track_tags, artist_origin, lastfm_listeners,
             lyric_valence, lyric_themes, lyric_lang,
             title, release_date, lastfm_synced_at, geo_synced_at, lyrics_synced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
        """, [
            (rk, d.get("bpm"), d.get("key"), d.get("energy"), d.get("year"),
             d.get("artist"), json.dumps(d.get("genres") or []),
             json.dumps(d.get("styles") or []),
             json.dumps(d.get("moods") or []), d.get("file_path"),
             d.get("track_updated_at"), d.get("album_updated_at"), d.get("artist_updated_at"),
             d.get("danceability"), d.get("brightness"),
             d.get("beat_confidence"), d.get("integrated_loudness"),
             d.get("onset_rate"), d.get("dynamic_complexity"),
             d.get("arousal"), d.get("valence"), d.get("vocal_presence"),
             d.get("mood_happy"), d.get("mood_sad"), d.get("mood_aggressive"),
             d.get("mood_relaxed"), d.get("mood_party"), d.get("mood_acoustic"),
             d.get("mood_electronic"), d.get("danceability_hl"),
             json.dumps(d["moodtheme"]) if d.get("moodtheme") is not None else None,
             json.dumps(d["genre_discogs"]) if d.get("genre_discogs") is not None else None,
             d.get("emb_effnet"), d.get("emb_musicnn"),
             json.dumps(d["lastfm_artist_tags"]) if d.get("lastfm_artist_tags") is not None else None,
             json.dumps(d["lastfm_track_tags"]) if d.get("lastfm_track_tags") is not None else None,
             json.dumps(d["artist_origin"]) if d.get("artist_origin") is not None else None,
             d.get("lastfm_listeners"), d.get("lyric_valence"),
             json.dumps(d["lyric_themes"]) if d.get("lyric_themes") is not None else None,
             d.get("lyric_lang"), d.get("title"), d.get("release_date"),
             d.get("lastfm_synced_at"), d.get("geo_synced_at"), d.get("lyrics_synced_at"))
            for rk, d in entries.items()
        ])
        conn.commit()
        conn.close()
    except Exception as e:
        log_msg(f"[ERROR] Cache upsert failed: {e}", level="error")


def update_analysis_columns(entries):
    """Write ONLY the analysis-derived columns (acoustic + TF + mood + embeddings) via targeted
    UPDATE, never the metadata or sync columns. Used by the TF post-pass so it can run alongside
    the Last.fm / MusicBrainz syncs without clobbering their writes (INSERT OR REPLACE would carry
    this process's start-of-run snapshot of those columns and overwrite concurrent sync updates)."""
    if not entries:
        return
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_db_schema(conn)
        conn.executemany("""
            UPDATE essentia_cache SET
              bpm=?, key=?, energy=?, danceability=?, brightness=?,
              beat_confidence=?, integrated_loudness=?, onset_rate=?, dynamic_complexity=?,
              arousal=?, valence=?, vocal_presence=?,
              mood_happy=?, mood_sad=?, mood_aggressive=?, mood_relaxed=?, mood_party=?,
              mood_acoustic=?, mood_electronic=?, danceability_hl=?, moodtheme=?, genre_discogs=?,
              emb_effnet=?, emb_musicnn=?
            WHERE rating_key=?
        """, [
            (d.get("bpm"), d.get("key"), d.get("energy"), d.get("danceability"), d.get("brightness"),
             d.get("beat_confidence"), d.get("integrated_loudness"), d.get("onset_rate"), d.get("dynamic_complexity"),
             d.get("arousal"), d.get("valence"), d.get("vocal_presence"),
             d.get("mood_happy"), d.get("mood_sad"), d.get("mood_aggressive"), d.get("mood_relaxed"),
             d.get("mood_party"), d.get("mood_acoustic"), d.get("mood_electronic"), d.get("danceability_hl"),
             json.dumps(d["moodtheme"]) if d.get("moodtheme") is not None else None,
             json.dumps(d["genre_discogs"]) if d.get("genre_discogs") is not None else None,
             d.get("emb_effnet"), d.get("emb_musicnn"), rk)
            for rk, d in entries.items()
        ])
        conn.commit()
        conn.close()
    except Exception as e:
        log_msg(f"[ERROR] Analysis-column update failed: {e}", level="error")

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
        # A track needs the post-pass if it's missing the base TF features OR (when the
        # high-level heads are present) the mood/theme/genre fields. Mirrors needs_tf in
        # _fill_missing_acoustic so the existing library backfills the new columns.
        def _row_needs_tf(data):
            return (any(data.get(f) is None for f in ("arousal", "valence", "vocal_presence"))
                    or (bool(_mood_models) and data.get("mood_happy") is None)
                    or (_moodtheme_model is not None and data.get("moodtheme") is None)
                    or (_genre_model is not None and data.get("genre_discogs") is None))
        tf_todo = [
            (rk, data["file_path"])
            for rk, data in _essentia_cache.items()
            if data.get("energy") is not None
            and data.get("file_path")
            and _row_needs_tf(data)
        ]
        if tf_todo:
            log_msg(f"\n[INFO] TF post-pass: {len(tf_todo)} tracks need TF features "
                    f"(high-level heads loaded: {len(_mood_models)} mood/danceability"
                    f"{', moodtheme' if _moodtheme_model is not None else ''}"
                    f"{', genre400' if _genre_model is not None else ''}).")
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
                    # targeted UPDATE (analysis columns only) so a concurrent Last.fm/MB sync
                    # isn't clobbered; the post-pass only touches tracks already in the cache.
                    update_analysis_columns(tf_pending)
                    tf_pending.clear()
                tf_elapsed = time.time() - tf_start
                tf_avg = tf_elapsed / max(1, tf_i + 1)
                tf_eta = timedelta(seconds=int((len(tf_todo) - tf_i - 1) * tf_avg))
                log_msg(f"TF: [{tf_i + 1}/{len(tf_todo)}] | {tf_avg:.1f}s/track | Est: {tf_eta} ", end='\r')
            log_msg(f"\n[INFO] TF post-pass complete.")

    log_msg(f"\n--- Success! Analysis and Sync complete. ---")
    log_msg(f"Total tracks in cache: {len(_essentia_cache)}")

# --- METADATA REFRESH: cache-driven; adaptive cadence by release age + data presence ---
import urllib.parse, urllib.request

def _release_age_days(release_date, year, now):
    """Days since original release — precise from release_date (unix), else year (Jan 1)."""
    if release_date:
        return (now - release_date) / 86400.0
    if year:
        try:
            return (now - datetime(int(year), 1, 1).timestamp()) / 86400.0
        except Exception:
            pass
    return 1e9

def _refresh_due(synced_at, release_date, year, has_data, source, now):
    """Should this track's <source> data be (re)fetched? Never-synced -> yes. Otherwise the
    cadence adapts: missing data on a young release retries soon (MB/LRCLIB may not have had it
    yet); Last.fm refreshes fast for new releases (listeners/tags evolve) and slows as it ages;
    geo/lyrics settle to yearly once found (origin + lyrics are static)."""
    if synced_at is None:
        return True
    age = (now - synced_at) / 86400.0
    rel = _release_age_days(release_date, year, now)
    if not has_data:
        threshold = 21 if rel < 365 else 180
    elif source == "lastfm":
        threshold = 14 if rel < 90 else (45 if rel < 730 else 120)
    else:   # geo / lyrics
        threshold = 365
    return age >= threshold

def _ensure_meta_fields(conn):
    """Populate title + release_date for cached tracks that lack a title, via ONE bulk Plex
    search (metadata only). Runs only while such tracks exist (first run / newly-analysed
    tracks); afterwards the syncs are fully cache-driven and never touch Plex."""
    n = conn.execute("SELECT COUNT(*) FROM essentia_cache WHERE title IS NULL").fetchone()[0]
    if n == 0:
        return
    log_msg(f"[INFO] Populating title/release_date for {n} tracks (one-time, then cache-driven)...")
    null_rks = {rk for (rk,) in conn.execute("SELECT rating_key FROM essentia_cache WHERE title IS NULL")}
    music = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=600).library.section(MUSIC_LIBRARY)
    pend = []
    for t in music.search(libtype='track', container_size=5000):
        rk = str(t.ratingKey)
        if rk not in null_rks:
            continue
        oaa = getattr(t, "originallyAvailableAt", None)
        rd = oaa.timestamp() if oaa else None
        pend.append((t.title or "", rd, rk))
        if len(pend) >= 500:
            conn.executemany("UPDATE essentia_cache SET title=?, release_date=? WHERE rating_key=?", pend)
            conn.commit()
            pend = []
    if pend:
        conn.executemany("UPDATE essentia_cache SET title=?, release_date=? WHERE rating_key=?", pend)
        conn.commit()


# --- METADATA SYNC: Last.fm community tags (cache-driven; targeted UPDATEs; refresh cadence) ---

def _load_lastfm_key():
    try:
        import yaml
        cfg = yaml.safe_load(open(os.path.join(BASE_DIR, "config.yml")))
        return (cfg.get("extras") or {}).get("lastfm_api_key")
    except Exception:
        return None

import threading as _threading

class _RateLimiter:
    """Spaces calls across threads to a max average rate (req/s): each caller reserves the next
    time-slot under a lock, then sleeps to it — so N workers still respect a strict API limit."""
    def __init__(self, rate):
        self._interval = 1.0 / rate
        self._lock = _threading.Lock()
        self._next = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            slot = self._next if self._next > now else now
            self._next = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)

_LF_RL = None   # Last.fm rate limiter (set during sync_lastfm_tags)
_MB_RL = None   # MusicBrainz rate limiter (set during sync_artist_origin)


def _workers_arg(default):
    if "--workers" in sys.argv:
        try:
            return max(1, int(sys.argv[sys.argv.index("--workers") + 1]))
        except Exception:
            pass
    return default


def _run_concurrent(todo, fetch_fn, update_sql, workers, label, verbose=False):
    """Generic concurrent metadata sync: worker pool calls fetch_fn(item)->(row, disp); a single
    writer thread (own DB connection) batches `update_sql`. The rate limiter inside the HTTP wrapper
    (not the worker count) caps the request rate, so this is safe for rate-limited APIs."""
    import concurrent.futures, queue as _queue
    q = _queue.Queue(maxsize=4000); STOP = object(); cnt = {"n": 0}; start = time.time(); _last = [0.0]

    def _writer():
        wc = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60); wc.execute("PRAGMA journal_mode=WAL")
        batch = []
        while True:
            it = q.get()
            if it is STOP:
                break
            row, disp = it; batch.append(row); cnt["n"] += 1
            if len(batch) >= 100:
                wc.executemany(update_sql, batch); wc.commit(); batch = []
            t = time.time()
            if verbose:
                log_msg(f"   {disp}")
            elif t - _last[0] >= 1.0:                 # repaint ~1x/sec so the count moves smoothly
                _last[0] = t
                rate = cnt["n"] / max(0.1, t - start)
                eta = timedelta(seconds=int((len(todo) - cnt["n"]) / max(0.1, rate)))
                log_msg(f"{label}: [{cnt['n']}/{len(todo)}] {rate:.1f}/s | ETA {eta} ", end='\r')
        if batch:
            wc.executemany(update_sql, batch); wc.commit()
        wc.close()

    wt = _threading.Thread(target=_writer, daemon=True); wt.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda it: q.put(fetch_fn(it)), todo))
    finally:
        q.put(STOP); wt.join()
    return cnt["n"]


def _lf_get(params, key):
    if _LF_RL:
        _LF_RL.wait()
    url = "http://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(
        {**params, "api_key": key, "format": "json", "autocorrect": 1})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception:
            if attempt < 2:
                time.sleep(1.0)
    return {}

def _lf_parse(data):
    """{tag: count 1-100} from a Last.fm gettoptags response (top 20, dropping zero-count tags)."""
    tags = (data.get("toptags") or {}).get("tag") or []
    if isinstance(tags, dict):
        tags = [tags]
    out = {}
    for t in tags[:20]:
        name = (t.get("name") or "").lower().strip()
        cnt = int(t.get("count") or 0)
        if name and cnt > 0:
            out[name] = cnt
    return out

_lf_artist_cache = {}   # artist_lower -> tags (one fetch serves all the artist's tracks)
def _lf_artist_tags(artist, key):
    k = artist.lower().strip()
    if k not in _lf_artist_cache:
        _lf_artist_cache[k] = _lf_parse(_lf_get({"method": "artist.gettoptags", "artist": artist}, key))
    return _lf_artist_cache[k]

def sync_lastfm_tags(limit=None):
    """Refresh Last.fm artist+track tags + per-track listeners for every cached track whose data is
    missing or stale (cache-driven via _refresh_due; no Plex per run). Artist tags cached per
    artist; targeted UPDATE sets lastfm_synced_at."""
    key = _load_lastfm_key()
    if not key:
        log_msg("[ERROR] No extras.lastfm_api_key in config — cannot sync Last.fm tags.", level="error")
        return
    conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_db_schema(conn)
    _ensure_meta_fields(conn)
    now = time.time()
    # migrate: stamp pre-existing data so it refreshes on cadence, not all at once on first run
    conn.execute("UPDATE essentia_cache SET lastfm_synced_at=? "
                 "WHERE lastfm_listeners IS NOT NULL AND lastfm_synced_at IS NULL", (now,))
    conn.commit()
    rows = conn.execute(
        "SELECT rating_key, artist, title, release_date, year, lastfm_synced_at, lastfm_listeners "
        "FROM essentia_cache WHERE title IS NOT NULL").fetchall()
    todo = [(rk, art, tit) for rk, art, tit, rd, yr, syn, data in rows
            if _refresh_due(syn, rd, yr, data is not None, "lastfm", now)]
    if limit:
        todo = todo[:int(limit)]
    verbose = bool(limit) and int(limit) <= 25
    log_msg(f"[INFO] Last.fm sync: {len(todo)} due of {len(rows)} cached.")
    conn.close()   # setup done; the writer thread uses its own connection
    global _LF_RL
    _LF_RL = _RateLimiter(5.0)        # Last.fm: <=5 req/s averaged (limiter spans all workers)
    workers = _workers_arg(6)
    log_msg(f"[INFO] fetching with {workers} concurrent workers (capped at 5 req/s).")

    def _fetch(item):
        rk, artist, title = item
        artist, title = artist or "", title or ""
        at = _lf_artist_tags(artist, key) if artist else {}
        info = (_lf_get({"method": "track.getInfo", "artist": artist, "track": title}, key).get("track")
                or {}) if (artist and title) else {}
        tt = _lf_parse(info); listeners = int(info.get("listeners") or 0)
        disp = f"{artist[:18]:18} - {title[:20]:20} | listeners={listeners:>8} track:{list(tt)[:3]}"
        return (json.dumps(at), json.dumps(tt), listeners, time.time(), rk), disp

    n = _run_concurrent(todo, _fetch,
        "UPDATE essentia_cache SET lastfm_artist_tags=?, lastfm_track_tags=?, lastfm_listeners=?, "
        "lastfm_synced_at=? WHERE rating_key=?", workers, "Last.fm", verbose)
    log_msg(f"\n[INFO] Last.fm sync complete: {n} tracks.")


# --- METADATA SYNC: MusicBrainz artist origin (full place hierarchy; per-artist; rate-limited) ---
_MB_UA = "meloday/1.0 (https://github.com/meloday)"
_CITY_TYPES = {"City", "District", "Town", "Municipality", "Borough", "Village"}

def _mb_get(path):
    if _MB_RL:
        _MB_RL.wait()                 # strict <=1 req/s, spaced across workers
    req = urllib.request.Request("https://musicbrainz.org/ws/2/" + path, headers={"User-Agent": _MB_UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            if not _MB_RL:
                time.sleep(1.1)       # sequential fallback when no limiter is set
            return data
        except Exception:
            time.sleep(2.0 if attempt < 2 else 1.1)
    return {}

_mb_area_chain = {}   # area_mbid -> [(name, type, codes)] from the part-of parent walk (cached)
def _resolve_area_chain(mbid):
    if mbid in _mb_area_chain:
        return _mb_area_chain[mbid]
    chain, cur, seen = [], mbid, set()
    while cur and cur not in seen:
        seen.add(cur)
        d = _mb_get(f"area/{cur}?inc=area-rels&fmt=json")
        if not d:
            break
        chain.append((d.get("name"), d.get("type"),
                      (d.get("iso-3166-1-codes") or []) + (d.get("iso-3166-2-codes") or [])))
        par = [r for r in d.get("relations", [])
               if r.get("type") == "part of" and r.get("direction") == "backward"]
        cur = par[0]["area"]["id"] if par else None
    _mb_area_chain[mbid] = chain
    return chain

def _artist_origin(name):
    """Resolve an artist's full place hierarchy from MusicBrainz — the union of the begin-area and
    area parent chains (whichever exist), plus the country code. {} if not found."""
    q = urllib.parse.quote('artist:"%s"' % name.replace('"', ""))
    a = (_mb_get(f"artist/?query={q}&fmt=json&limit=1").get("artists") or [{}])[0]
    if not a:
        return {}
    begin, area = a.get("begin-area") or {}, a.get("area") or {}
    cc = a.get("country")
    places, city, region, country = set(), None, None, None
    for ar in (begin, area):                       # union both chains (data is inconsistent)
        if not ar.get("id"):
            continue
        for nm, ty, codes in _resolve_area_chain(ar["id"]):
            if nm:
                places.add(nm.lower())
            if ty in _CITY_TYPES and not city:
                city = nm
            elif ty == "Subdivision" and not region:
                region = nm
            elif ty == "Country":
                country = nm or country
    if not (begin.get("id") or area.get("id")) and not cc:
        return {}
    return {
        "begin_area": begin.get("name"), "area": area.get("name"),
        "city": city, "region": region, "country": country, "country_code": cc,
        "places": sorted(places), "mbid": a.get("id"),
    }

def sync_artist_origin(limit=None):
    """Refresh each artist's MusicBrainz origin for cached tracks whose origin is missing or stale
    (cache-driven; per-artist, cached). Targeted UPDATE sets geo_synced_at; unresolved artists are
    retried on a cadence (not every run) via _refresh_due."""
    conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_db_schema(conn)
    _ensure_meta_fields(conn)
    now = time.time()
    # migrate: stamp already-resolved origins so they refresh yearly, not all at once
    conn.execute("UPDATE essentia_cache SET geo_synced_at=? "
                 "WHERE artist_origin IS NOT NULL AND geo_synced_at IS NULL", (now,))
    conn.commit()
    rows = conn.execute(
        "SELECT rating_key, artist, release_date, year, geo_synced_at, artist_origin "
        "FROM essentia_cache WHERE artist IS NOT NULL").fetchall()
    todo = [(rk, art) for rk, art, rd, yr, syn, data in rows
            if _refresh_due(syn, rd, yr, data is not None, "geo", now)]
    if limit:
        todo = todo[:int(limit)]
    verbose = bool(limit) and int(limit) <= 25
    log_msg(f"[INFO] MusicBrainz origin sync: {len(todo)} due of {len(rows)} cached.")
    conn.close()   # setup done; the writer thread uses its own connection
    global _MB_RL
    _MB_RL = _RateLimiter(0.95)       # MusicBrainz: strict <=1 req/s — small margin to avoid 503s
    workers = _workers_arg(3)
    import concurrent.futures, queue as _queue
    from collections import defaultdict
    artist_tracks = defaultdict(list)
    for rk, art in todo:
        artist_tracks[art or ""].append(rk)
    artists = [a for a in artist_tracks if a]
    log_msg(f"[INFO] resolving {len(artists)} unique artists with {workers} workers (capped at 1 req/s).")
    q = _queue.Queue(maxsize=2000); STOP = object(); cnt = {"a": 0, "t": 0}; start = time.time(); _last = [0.0]
    now2 = time.time()

    def _writer():
        wc = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60); wc.execute("PRAGMA journal_mode=WAL")
        batch = []
        while True:
            it = q.get()
            if it is STOP:
                break
            artist, origin = it
            oj = json.dumps(origin) if origin else None
            for rk in artist_tracks[artist]:
                batch.append((oj, now2, rk)); cnt["t"] += 1
            cnt["a"] += 1
            if len(batch) >= 300:
                wc.executemany("UPDATE essentia_cache SET artist_origin=?, geo_synced_at=? WHERE rating_key=?", batch); wc.commit(); batch = []
            if verbose:
                log_msg(f"   {artist[:22]:22} -> city={(origin or {}).get('city')} "
                        f"region={(origin or {}).get('region')} country={(origin or {}).get('country')}")
            elif time.time() - _last[0] >= 1.0:
                _last[0] = time.time()
                rate = cnt["a"] / max(0.1, time.time() - start)
                eta = timedelta(seconds=int((len(artists) - cnt["a"]) / max(0.1, rate)))
                log_msg(f"MB: [{cnt['a']}/{len(artists)} artists] {rate:.2f}/s | ETA {eta} ", end='\r')
        if batch:
            wc.executemany("UPDATE essentia_cache SET artist_origin=?, geo_synced_at=? WHERE rating_key=?", batch); wc.commit()
        wc.close()

    wt = _threading.Thread(target=_writer, daemon=True); wt.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda a: q.put((a, _artist_origin(a))), artists))
    finally:
        q.put(STOP); wt.join()
    log_msg(f"\n[INFO] MusicBrainz origin sync complete: {cnt['a']} artists, {cnt['t']} tracks.")


# --- METADATA SYNC: lyrics (LRCLIB; sentiment + theme keywords + language) ---
_LRCLIB_UA = "meloday/1.0 (https://github.com/meloday)"
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None
try:
    from langdetect import detect as _detect_lang, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    _detect_lang = None

# Distinctive lyric themes (>=2 keyword hits to fire). Generic words like "love" are excluded —
# the audio mood models cover emotion; lyrics add lyrically-specific themes the audio can't know.
_LYRIC_THEMES = {   # EXHAUSTIVE capture, then wind-back: broad cross-firing phrases (df>8%% of library) pruned.
    'christmas': {"strong": ['christmas', 'santa', 'sleigh', 'mistletoe', 'reindeer', 'silent night', 'merry christmas', 'deck the halls', 'north pole', 'winter wonderland', 'yuletide', 'christmas tree', 'ho ho ho', 'jingle bell', 'feliz navidad', 'st nick', 'saint nick', 'frosty the snowman', 'jingle bells', 'auld lang syne', 'sleigh bells', 'this christmas', 'christmas time', 'christmas day', 'jolly old', 'little drummer', 'baby jesus', 'silver bells', 'christmas eve', 'father christmas', 'kris kringle', "season's greetings"],
        "weak": ['snow', 'bells', 'holiday', 'the season', 'presents', 'stockings', 'carol', 'this year', 'the chimney', 'gifts', 'tinsel', 'wrapping', 'fireplace', 'cold', 'december']},
    'summer': {"strong": ['summertime', 'summer night', 'summer love', 'in the summertime', 'endless summer', 'hot summer', 'summer breeze', 'long hot summer', 'that summer', 'in the summer', 'summer days', 'summer sun', 'cruel summer', 'boys of summer', 'summer rain', 'summer fling', 'one summer'],
        "weak": ['summer', 'sunshine', 'beach', 'in the sun', 'by the pool', 'ocean', 'waves', 'sand', 'surf', 'sunny', 'heat', 'poolside', 'barefoot', 'seaside', 'bikini', 'ice cream', 'july', 'sunburn', 'lemonade', 'boardwalk']},
    'winter': {"strong": ['let it snow', 'cold december', 'winter night', "baby it's cold outside", 'first snow', 'the snow falls', "winter's cold", 'snowed in', 'cold winter', 'frostbite', 'snow is falling', 'build a snowman', 'snow angel', "cold winter's night", 'walking in the snow', 'winter wind'],
        "weak": ['snow', 'frozen', 'cold', 'icy', 'snowflakes', 'freezing', 'frost', 'chill', 'fireplace', 'mittens', 'scarf', 'blizzard', 'shiver', 'december']},
    'rain': {"strong": ['in the rain', 'raindrops', 'pouring rain', 'rainy day', 'here comes the rain', 'stand the rain', 'rhythm of the rain', 'singing in the rain', 'let the rain', 'rain keeps falling', 'rain on me', 'rain rain', 'purple rain', 'when it rains', 'kiss the rain', 'walking in the rain', 'summer rain', 'cry in the rain'],
        "weak": ['raining', 'storm', 'thunder', 'umbrella', 'pouring', 'downpour', 'grey skies', 'wet', 'clouds', 'lightning', 'drizzle', 'puddles', 'drops']},
    'road': {"strong": ['on the road', 'the highway', 'open road', 'behind the wheel', 'hit the road', 'down the road', 'life is a highway', 'windows down', 'two lane', 'born to be wild', 'the freeway', 'road again', 'drive all night', 'down the highway', 'out on the road', 'cruisin', 'road trip', 'take a ride', 'the long road', 'endless road', 'road ahead'],
        "weak": ['road', 'driving', 'drive', 'wheel', 'miles', 'cruising', 'engine', 'headlights', 'gasoline', 'ride', 'highway', 'tires', 'rolling', 'the car', 'tank of gas', 'backseat']},
    'party': {"strong": ['dance floor', 'on the floor', "let's dance", 'in the club', 'party tonight', 'drop the beat', 'hands up', 'put your hands up', 'your hands in the air', 'raise your glass', 'til the morning', 'dance all night', 'get low', 'the club tonight', 'throw your hands', 'party people', 'shake it', 'party all night', 'turn this up', 'light it up tonight', 'up in the club', 'we own the night'],
        "weak": ['party', 'dance', 'dancing', 'club', 'tonight', 'all night', 'drinks', 'the dj', 'beat', 'move your body', 'turn it up', 'music', 'bass', 'wild', 'crazy night', 'up all night', 'the floor', 'good time', 'let loose', 'feel the music']},
    'celebration': {"strong": ['time of our lives', 'best day of my life', 'good times', "let's celebrate", 'raise a glass', 'living it up', 'what a night', 'time to celebrate', 'celebrate good', "tonight's the night", 'feeling good tonight', 'this is our moment', 'throw your hands up tonight'],
        "weak": ['celebrate', 'cheers', 'champagne', 'toast', 'good time', 'unforgettable', 'our night', 'feel so good', 'best night', 'living the dream', 'top of the world', 'victory', 'we made it', 'memories']},
    'drinking': {"strong": ['whiskey', 'tequila', 'margarita', 'bourbon', 'honky tonk', 'one more drink', 'another round', 'get drunk', 'bottle of', 'drink a beer', 'red solo cup', 'whisky', 'cold beer', 'shot of', 'the bottle', 'drink away', 'one more round', 'pour me another', 'drinkin', 'ice cold beer', 'shot of whiskey', 'cheap wine', 'rum and', 'vodka', 'double shot', 'gin and'],
        "weak": ['drink', 'drunk', 'beer', 'bar', 'wine', 'last call', 'shots', 'hangover', 'sippin', 'liquor', 'buzz', 'tipsy', 'bartender', 'pour', 'glass', 'booze', 'sober', 'wasted']},
    'money': {"strong": ['make it rain', 'material girl', 'mo money', 'cash rules', 'love of money', 'money money money', 'all about the money', 'stack my paper', 'hundred dollar', 'get money', "can't buy me", 'money on my mind', 'i get money', 'big money', 'spend my money', 'gold digger', 'secure the bag', 'make a million', 'cash money', 'dollar bills'],
        "weak": ['money', 'cash', 'rich', 'dollars', 'racks', 'stacks', 'bands', 'the bag', 'diamonds', 'paid', 'wealth', 'gold chain', 'broke', 'bills', 'fortune', 'greenbacks', 'benjamins', 'payday']},
    'hustle': {"strong": ['started from the bottom', 'the daily grind', 'work hard play hard', 'put in the work', 'self made', 'rise to the top', "grind don't stop", 'no days off', 'from the bottom', "can't knock the hustle", 'chase the dream', 'the rap game', 'came up', 'my hustle', 'stack my paper', 'grindin', 'secure the bag', 'run it up', 'the come up', "i'm a hustler", 'grind every day', 'work for it', 'earn my keep'],
        "weak": ['hustle', 'grind', 'work hard', 'come up', 'ambition', 'climbing', 'make it to the top', 'never quit', 'hip hop', 'the streets', 'get to the top', 'dream big', 'my goals', 'put in work']},
    'faith': {"strong": ['hallelujah', 'praise the lord', 'oh happy day', 'amazing grace', 'my lord', 'amen', 'the gospel', 'sweet jesus', 'jesus', 'salvation', 'oh lord', 'god above', 'washed my sins', 'take me to church', 'i can only imagine', 'precious lord', 'my savior', 'good god', 'glory hallelujah', 'thank you lord', "heaven's gates", 'the holy spirit', 'god is good', 'praise his name', 'lord have mercy', 'born again', 'the cross', 'kingdom come'],
        "weak": ['pray', 'prayer', 'faith', 'heaven', 'the lord', 'holy', 'blessed', 'church', 'my soul', 'believe in him', 'gospel', 'angels', 'grace', 'glory', 'worship', 'preacher', 'devil']},
    'homesick': {"strong": ['country roads', 'take me home', 'going home', 'home sweet home', 'my home town', 'sweet home', 'where i belong', 'homeward bound', 'back home again', 'the house that built me', 'carry me home', 'heading home', 'my old home', "i'm coming home", 'long way home', 'feels like home', 'find my way home', 'take me back home', 'road that leads me home', 'miss my home'],
        "weak": ['back home', 'roots', 'my town', 'hometown', 'the place i', 'my old', 'where i grew up', 'front porch', 'my mama', 'the farm']},
    'nostalgia': {"strong": ['the good old days', 'back in the day', 'those were the days', 'glory days', 'remember when', 'when we were young', 'wish you were here', 'take me back', 'the way we were', 'good old days', 'best days of my life', 'still remember', 'i remember when', 'back when we were', 'those summer nights', 'time of your life', 'the way it was', 'simpler times', 'those golden years'],
        "weak": ['memories', 'remember', 'used to', 'back then', 'younger days', 'the past', 'reminisce', 'growing up', 'yesterday', 'those days', 'long ago', 'old times', 'back in time', 'photographs', 'the old']},
    'youth': {"strong": ['forever young', 'we are young', 'my generation', 'young and wild', 'young dumb', 'when we were young', 'stay forever young', 'wild youth', 'our teenage', 'young and free', "live while we're young", 'we were just kids', 'back in high school', 'young love', 'just sixteen', 'when i was sixteen', 'wasted youth', 'young hearts'],
        "weak": ['young', 'youth', 'teenage', 'reckless', 'high school', 'seventeen', 'eighteen', 'kids', 'those years', 'growing up', 'rebellious', 'carefree']},
    'devotion': {"strong": ['marry me', 'marry you', 'will you marry', 'grow old with you', 'spend my life with', 'for the rest of my life', 'spend the rest of my life', "i'll never leave you", 'forever and ever', 'by your side forever', 'grow old together', "you're the one for me", 'put a ring on', 'down on one knee', 'always and forever', 'never gonna leave you', 'take you as my', 'my wedding day', 'be my wife', 'be my husband', 'till death do us', 'i still do', 'my better half'],
        "weak": ['marry', 'wedding', 'exchange vows', 'never let you go', 'spend forever with', 'faithful to you', 'my one and only', 'ring on my finger', 'commit', 'devoted', 'my soulmate', 'walk down the aisle']},
    'love': {"strong": ['i love you', 'fall in love', 'in love with you', 'all of me', 'love of my life', 'you complete me', "you're the one", 'i adore you', "you're my world", 'unchained melody', 'i will always love you', "i'm in love", 'my one and only', 'love you so', "i'm so in love", 'head over heels in love', 'crazy in love', 'loving you is', "you're all i need", 'my heart belongs to you'],
        "weak": ['my love', 'darling', 'you and i', 'be mine', 'my everything', 'loving you', 'sweetheart', 'my darling', 'hold you', 'kiss', 'embrace', "you're beautiful", 'my dear', 'precious']},
    'new_love': {"strong": ['love at first sight', 'head over heels', 'weak in the knees', 'swept off my feet', "can't take my eyes off you", 'crazy little thing called love', 'first time i saw you', 'got a crush', 'stars in my eyes', 'love struck', 'butterflies in my', 'what you do to me', 'just one look', 'never felt this way before', 'falling for you', 'you had me at', 'my heart skips'],
        "weak": ['crush', 'falling for', "can't stop thinking", 'new love', 'just met', 'nervous', 'heart skips', 'kiss me', 'hold your hand', 'the moment i saw', 'first date', 'blushing', 'tingle']},
    'desire': {"strong": ['make love to', "let's get it on", 'between the sheets', 'i want your body', 'turn me on', 'sex you up', 'take you home tonight', 'all over me', 'under the covers', 'get it on', 'feel your body', 'touch your body', 'i wanna touch', 'slow and sensual', 'body on me', 'make love all night', 'your bare skin', 'lick', 'taste your', 'grind on'],
        "weak": ['make love', 'your touch', 'your lips', 'sensual', 'hold you close', 'want you bad', 'your skin', 'closer tonight', 'tonight', 'seduce', 'tease', 'kiss your neck', 'hot', 'steam']},
    'longing': {"strong": ['i miss you', 'nothing compares to you', "can't make you love me", 'wish you were here', 'come back to me', 'missing you tonight', "ain't no sunshine", 'i long for you', 'every breath you take', 'i still need you', 'wishing you were here', 'miss you so much', 'i need you here', 'come back home', "i'll be waiting", 'yearning for you', 'pining'],
        "weak": ['miss you', 'without you', 'far away', 'come back', 'waiting', 'long distance', 'think of you', 'longing', 'need you', 'so far away', 'empty without', 'absence', 'crave']},
    'heartbreak': {"strong": ['broken heart', 'broke my heart', 'you broke me', 'heartbroken', 'breaking up', 'we are over', 'you left me', 'you walked out', 'love gone wrong', 'un-break my heart', 'tore us apart', 'when i was your man', "you don't love me anymore", 'cheated on me', 'torn apart', 'break my heart', 'you said goodbye', 'picking up the pieces', "we're through", "you don't want me", "should've stayed", "we're done", 'you played me', 'cried over you'],
        "weak": ['you left', 'goodbye', 'walk away', 'cheated', 'used to love', 'tears', 'your love', 'let you go', 'over now', 'leaving', 'should have stayed', 'broke up', 'fall apart', 'heartache', 'lies', 'betrayed', 'alone now', 'my mistake']},
    'grief': {"strong": ['rest in peace', 'gone too soon', 'tears in heaven', 'watching over me', "i'll be missing you", 'gone forever', 'heaven needed', 'never see you again', 'fly high', 'gone but not forgotten', 'see you again', 'in a better place', 'missing you brother', 'you in heaven', "i'll see you again", 'look down on me', 'rest easy', 'up in heaven', 'your spirit lives', 'died too young', 'my angel now', 'forever in my heart'],
        "weak": ["you're gone", 'lost you', 'heaven', 'the angels', 'your memory', 'passed away', 'goodbye forever', 'meet again', 'funeral', 'grave', 'mourning', 'candle', 'photograph of you', 'gone away']},
    'loneliness': {"strong": ['all by myself', 'so lonely', 'all alone', 'only the lonely', 'i walk alone', 'lonely road', 'i am a rock', 'feeling so alone', 'lonely night', 'empty room', 'talk to myself', 'no one to call', 'sit here alone', 'nobody loves me', 'alone again', 'lonely heart', 'solitude'],
        "weak": ['lonely', 'on my own', 'no one', 'nobody', 'empty', 'four walls', 'by myself', 'isolated', 'cold and empty', 'quiet house', 'no one understands', 'emptiness']},
    'sad': {"strong": ['everybody hurts', 'the sound of silence', 'tears keep falling', 'drowning in my tears', 'cry me a river', 'so much pain', 'my immortal', 'heavy heart', 'crying myself to sleep', 'cried so much', 'all this pain', 'my tears fall', "can't stop the tears", 'tears stream down', 'hurts so much', 'sinking down', 'drowning in sorrow'],
        "weak": ['crying', 'tears', 'so sad', 'the blues', 'hurts so bad', 'in pain', 'broken inside', 'sorrow', 'dark days', 'cry myself', 'my tears', 'aching', 'weeping', 'melancholy', 'despair', 'numb', 'hollow', 'suffering', 'wounded', 'misery']},
    'regret': {"strong": ['should have known', 'my biggest mistake', 'i was wrong', 'wish i could take it back', 'turn back time', 'take it all back', 'the things i said', 'i should have', "i'm so sorry", 'i wanna apologize', 'too late to apologize', "say i'm sorry", 'wish i never', "i'd do anything to", 'if i could go back', 'forgive me', 'my regret'],
        "weak": ['sorry', 'should have', 'regret', 'my fault', 'too late', 'looking back', 'wish i', 'take it back', 'my mistake', 'i was wrong', "shouldn't have", 'i blew it', "what i've done", 'apologize']},
    'moving_on': {"strong": ['i will survive', 'over you now', 'better off without you', "i don't need you", 'stronger now', 'best thing i never had', 'done with you', 'getting over you', 'moving on without', "i'm so much better", 'over you', "i don't need ya", 'no longer cry', "you don't deserve me", "i'm letting go", 'i deserve better', 'done with your'],
        "weak": ['moving on', 'over you', 'let you go', "i'm free", 'new chapter', 'find myself', 'no more tears', 'better without', 'done crying', 'hate you now', 'walk out the door', 'new beginning', "don't need you"]},
    'empowerment': {"strong": ['eye of the tiger', 'stand tall', 'take on the world', 'time to shine', 'rise above', 'born to win', 'i am the champion', 'nothing can stop me', 'i am unstoppable', 'i am the one', 'let it shine', 'i am a fighter', 'born this way', 'hear me roar', "i'm a survivor", 'we are the champions', "i'm still standing", 'stronger than ever', 'i will rise', 'my time is now', "i'm a warrior", 'power within'],
        "weak": ['rise up', 'stronger', 'unstoppable', 'never give up', 'conquer', 'invincible', 'my time', 'unbreakable', 'shine bright', 'i believe in me', 'fight', 'champion', 'powerful', 'strong', 'i can do it', 'no fear']},
    'confidence': {"strong": ['run the world', 'feeling myself', 'look at me now', "can't touch this", "i'm that girl", 'king of the world', 'do it my way', "i'm the best", 'watch me now', "i'm a boss", 'like a boss', "i'm so fancy", 'good as hell', 'sexy and i know it', "i'm a queen", 'baddest in the room', 'i run this', 'bow down to me'],
        "weak": ['swagger', 'own it', 'confident', 'queen', 'i shine', 'feeling good', 'unbothered', "i'm the one", 'i look good', 'bow down', 'flawless', 'fabulous', 'that body', 'strut']},
    'resilience': {"strong": ["i won't back down", 'i will survive', 'never give up', 'still standing', "i'll rise up", 'get back up', 'you raise me up', 'weather the storm', 'made it through', 'against all odds', 'i believe i can fly', "won't back down", "what doesn't kill you", "i'll keep fighting", 'survive this', 'stand back up', 'through the fire'],
        "weak": ['survive', 'keep going', 'hold on', 'never break', 'fight on', "don't give up", 'push through', 'stand my ground', 'carry on', 'stronger', 'endure', 'overcome', "won't quit", 'through it all']},
    'freedom': {"strong": ['break free', 'set me free', 'born to run', "i'm free", 'run away from', 'fly away', 'born to be wild', 'spread my wings', 'no chains', 'wild and free', 'breaking free', 'break away', 'freedom freedom', 'running free', 'free at last', 'unchain me', "i'm breaking out", 'no walls can hold'],
        "weak": ['freedom', 'run away', 'escape', 'my own way', 'no rules', 'fly free', 'let me go', 'liberated', 'break out', 'wide open', 'wings', 'untamed']},
    'wanderlust': {"strong": ['around the world', 'leave it all behind', 'see the world', 'take me away', 'wherever the road', 'new horizons', 'far off places', 'on the run', 'pack my bags', 'adventure awaits', "let's get lost", 'road less traveled', 'escape this town', 'run away with me', 'somewhere far away', 'chasing the horizon', 'never settle down', 'wander the earth', 'find a new place'],
        "weak": ['wanderlust', 'adventure', 'far away', 'horizon', 'the open road', 'drifting', 'roam', 'explore', 'no destination', 'traveling', 'journey', 'faraway', 'wander', 'get away']},
    'hope': {"strong": ['better days', 'brighter days', 'keep the faith', 'never lose hope', 'light at the end', 'a new dawn', "everything's gonna be alright", "we'll be alright", 'keep believing', 'hold on to hope', 'the sun will shine', 'gonna be okay', 'rise again', 'over the rainbow', 'gonna be alright', "it'll be alright", 'sun is gonna shine', 'things will get better', 'brand new day', 'light will guide you'],
        "weak": ['hold on', 'tomorrow', 'new day', "don't give up", 'make it through', 'brighter', 'the sun will', 'keep your head up', 'faith', 'one day', 'get better', 'stay strong']},
    'gratitude': {"strong": ['count my blessings', 'so grateful', 'give thanks', 'thank you for loving me', 'thank you for being', 'so blessed', 'forever grateful', 'thankful for you', 'i want to thank you', 'want to thank', 'thank you thank you', 'grateful for everything', 'blessed beyond'],
        "weak": ['thankful', 'grateful', 'thank you', 'blessings', 'lucky to have', 'appreciate you', 'so thankful', 'gratitude', "i'm so lucky", 'much obliged']},
    'friendship': {"strong": ["you've got a friend", 'count on me', 'stand by me', 'lean on me', 'little help from my friends', 'what friends are for', "you're my best friend", 'we are family', "i'll stand by you", 'through thick and thin', "i'll be there for you", 'call my name and', 'you can count on me', "that's what friends", "i'll come running", 'brothers and sisters', 'my ride or die', 'always have your back'],
        "weak": ['my friends', 'by my side', 'through it all', 'we got each other', 'stick together', 'my crew', 'day ones', 'my people', 'best friend', 'ride or die', 'homies', 'my squad', 'stand together', 'got your back']},
    'rebellion': {"strong": ['fight the power', 'killing in the name', 'power to the people', 'fight the system', 'up against the wall', "won't be silenced", 'take to the streets', 'rise up against', 'break the chains', "won't get fooled", 'burn it down', 'not gonna take it', 'stand and fight', 'raise your fist', "won't do what you tell me", 'tear down the walls', 'down with the', 'viva la revolution', 'the people united'],
        "weak": ['the system', 'revolution', 'anarchy', 'riot', 'fight back', 'resist', 'rebel', 'uprising', "won't comply", 'the man', 'oppression', 'injustice', 'protest', 'freedom fighters']},
    'angst': {"strong": ['smells like teen spirit', "i'm not okay", 'i hate everything', 'nobody understands me', "i can't take it anymore", 'breaking down inside', 'crawling in my skin', 'wanna scream', "i'm losing it", "i'm so sick of", 'no one understands', 'everything about you', "i'm a freak", 'tear me apart', "i'm falling apart inside", 'sick of this world', 'cut me open'],
        "weak": ['i hate', 'sick of', 'fed up', 'screaming', 'against the world', 'misunderstood', 'numb', 'falling apart', 'rage', "can't breathe", "i'm not", 'broken', 'scream', 'anger', 'frustrated', 'outcast']},
}

def _lrclib_search(artist, title):
    """Top LRCLIB match by artist+title (no duration needed → cache-driven). Lyrics are
    version-invariant, so the top fuzzy match is fine. None if no result/error."""
    url = "https://lrclib.net/api/search?" + urllib.parse.urlencode(
        {"artist_name": artist, "track_name": title})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": _LRCLIB_UA}), timeout=15) as r:
            res = json.loads(r.read().decode("utf-8", "ignore"))
        return res[0] if isinstance(res, list) and res else None
    except Exception:
        return None

def _analyze_lyrics(text):
    """(valence 0-1 or None, [themes], lang or None) from plain lyrics. VADER averaged per line
    (the whole-song compound saturates); themes from distinctive keyword counts."""
    tl = text.lower()
    # Fire a theme on >=2 DISTINCT phrases, OR one phrase repeated >=4x (a chorus hook like the
    # "christmas" in Last Christmas). Distinct-matching stops a repeated colloquialism (e.g.
    # "hallelujah" x3 in a funk song) from tripping a theme; the >=4 rule keeps single-hook songs.
    themes = []
    for _th, _sp in _LYRIC_THEMES.items():
        # a theme-unique STRONG phrase fires alone; otherwise need >=2 distinct phrases
        if any(s in tl for s in _sp["strong"]) or sum(1 for k in _sp["strong"] + _sp["weak"] if k in tl) >= 2:
            themes.append(_th)
    valence = None
    if _vader:
        comps = [_vader.polarity_scores(l)["compound"] for l in text.splitlines() if l.strip()]
        if comps:
            valence = round((sum(comps) / len(comps) + 1) / 2, 4)
    lang = None
    if _detect_lang:
        try:
            lang = _detect_lang(text[:2000])
        except Exception:
            lang = None
    return valence, themes, lang

def sync_lyrics(limit=None, force=False):
    """Refresh lyrics-derived sentiment/themes/language for cached tracks whose data is missing or
    stale (cache-driven; LRCLIB /search by artist+title). Targeted UPDATE sets lyrics_synced_at.
    force=True re-does every track (use after the lyric vocabulary changes)."""
    if _vader is None:
        log_msg("[WARN] vaderSentiment not installed — lyric_valence will be null (themes/lang only). "
                "pip install vaderSentiment langdetect for the full version.")
    conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_db_schema(conn)
    _ensure_meta_fields(conn)
    now = time.time()
    if force:
        # re-derive everything (the lyric vocabulary changed): clear lyric data + stamps so all are
        # due. Clearing lyric_lang too means a plain --sync-lyrics re-run resumes cleanly (the
        # cadence-migration below won't re-stamp half-done tracks). Run --force ONCE, then plain.
        conn.execute("UPDATE essentia_cache SET lyrics_synced_at=NULL, lyric_lang=NULL, "
                     "lyric_themes=NULL, lyric_valence=NULL")
        log_msg("[INFO] --force: cleared lyric data; re-deriving for the whole library.")
    else:
        # migrate: stamp pre-existing lyric data so it refreshes on cadence, not all at once
        conn.execute("UPDATE essentia_cache SET lyrics_synced_at=? "
                     "WHERE lyric_lang IS NOT NULL AND lyrics_synced_at IS NULL", (now,))
    conn.commit()
    rows = conn.execute(
        "SELECT rating_key, artist, title, release_date, year, lyrics_synced_at, lyric_lang "
        "FROM essentia_cache WHERE title IS NOT NULL").fetchall()
    todo = [(rk, art, tit) for rk, art, tit, rd, yr, syn, data in rows
            if _refresh_due(syn, rd, yr, data is not None, "lyrics", now)]
    if limit:
        todo = todo[:int(limit)]
    verbose = bool(limit) and int(limit) <= 25
    log_msg(f"[INFO] Lyrics sync: {len(todo)} due of {len(rows)} cached. "
            f"sentiment={'on' if _vader else 'off'}, lang={'on' if _detect_lang else 'off'}")
    conn.close()   # setup done; the writer thread below uses its own connection
    # ---- concurrent fetch: the work is just waiting on LRCLIB, so N workers ~= N x faster.
    # Workers fetch+analyse and queue results; a single writer thread (own conn) batches the UPDATEs.
    import concurrent.futures, threading, queue as _queue
    workers = 8
    if "--workers" in sys.argv:
        try:
            workers = max(1, int(sys.argv[sys.argv.index("--workers") + 1]))
        except Exception:
            pass
    log_msg(f"[INFO] fetching with {workers} concurrent workers.")
    q = _queue.Queue(maxsize=4000)
    _STOP = object()
    _nlp_lock = threading.Lock()       # vaderSentiment/langdetect aren't guaranteed thread-safe
    cnt = {"n": 0, "lyr": 0}
    start = time.time()
    _last = [0.0]

    def _writer():
        wc = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60)
        wc.execute("PRAGMA journal_mode=WAL")
        batch = []
        while True:
            item = q.get()
            if item is _STOP:
                break
            row, had, disp = item
            batch.append(row)
            cnt["n"] += 1
            cnt["lyr"] += 1 if had else 0
            if len(batch) >= 100:
                wc.executemany("UPDATE essentia_cache SET lyric_valence=?, lyric_themes=?, "
                               "lyric_lang=?, lyrics_synced_at=? WHERE rating_key=?", batch)
                wc.commit(); batch = []
            if verbose:
                log_msg(f"   {disp}")
            elif time.time() - _last[0] >= 1.0:
                _last[0] = time.time()
                rate = cnt["n"] / max(0.1, time.time() - start)
                eta = timedelta(seconds=int((len(todo) - cnt["n"]) / max(0.1, rate)))
                log_msg(f"Lyrics: [{cnt['n']}/{len(todo)}] {rate:.1f}/s | ETA {eta} ", end='\r')
        if batch:
            wc.executemany("UPDATE essentia_cache SET lyric_valence=?, lyric_themes=?, "
                           "lyric_lang=?, lyrics_synced_at=? WHERE rating_key=?", batch)
            wc.commit()
        wc.close()

    wt = threading.Thread(target=_writer, daemon=True)
    wt.start()

    def _fetch(item):
        rk, artist, title = item
        artist, title = artist or "", title or ""
        valence, themes, lang, had = None, [], "none", False
        d = _lrclib_search(artist, title) if (artist and title) else None
        if d:
            if d.get("instrumental") or not (d.get("plainLyrics") or "").strip():
                lang = "instrumental"
            else:
                with _nlp_lock:
                    valence, themes, _lang = _analyze_lyrics(d["plainLyrics"])
                lang = _lang or "unknown"; had = True
        disp = f"{artist[:16]:16} - {title[:20]:20} | valence={valence} themes={themes} lang={lang}"
        q.put(((valence, json.dumps(themes), lang, time.time(), rk), had, disp))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_fetch, todo))
    finally:
        q.put(_STOP)
        wt.join()
    log_msg(f"\n[INFO] Lyrics sync complete: {cnt['n']} processed, {cnt['lyr']} with lyrics.")


# --- 4. EXECUTION ---

if __name__ == "__main__":
    if "--sync-lyrics" in sys.argv:
        _lim = None
        if "--limit" in sys.argv:
            try:
                _lim = int(sys.argv[sys.argv.index("--limit") + 1])
            except Exception:
                _lim = None
        sync_lyrics(limit=_lim, force="--force" in sys.argv)
    elif "--sync-geo" in sys.argv:
        _lim = None
        if "--limit" in sys.argv:
            try:
                _lim = int(sys.argv[sys.argv.index("--limit") + 1])
            except Exception:
                _lim = None
        sync_artist_origin(limit=_lim)
    elif "--sync-metadata" in sys.argv:
        _lim = None
        if "--limit" in sys.argv:
            try:
                _lim = int(sys.argv[sys.argv.index("--limit") + 1])
            except Exception:
                _lim = None
        sync_lastfm_tags(limit=_lim)
    else:
        bulk_analyze()
