import os
import multiprocessing
import time
import json
import portalocker
import concurrent.futures
import logging
from datetime import timedelta, datetime
from meloday import (
    PLEX_URL, PLEX_TOKEN, MUSIC_LIBRARY, analyze_track_essentia, 
    save_essentia_cache, ESSENTIA_ENABLED, 
    ESSENTIA_CACHE_PATH, _essentia_cache, PlexServer, BASE_DIR
)

# --- LOGGING SETUP ---
LOG_FILE = os.path.join(BASE_DIR, "logs", "pre_analyze.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("PreAnalyze")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def log_msg(msg, level="info", log_only=False):
    """Filters for printable characters to ensure a clean, plain-text log."""
    if not msg: return
    clean_msg = "".join(ch for ch in str(msg) if ch.isprintable() or ch in ("\n", "\r", "\t"))
    clean_msg = clean_msg.replace('\x00', '')
    if not log_only: print(msg)
    if level == "error": logger.error(clean_msg)
    else: logger.info(clean_msg)

# --- 0. OPTIMISE WORKER COUNT ---

def get_optimal_workers(task_type="cpu"):
    try:
        # Detect logical threads (2, 22, or 24 on your specific CPUs)
        logical = os.cpu_count() or 1
        tier_reason = "Fallback"
        assigned = 4
        
        if task_type == "cpu":
            # --- TIER 1: Low-Power / Older (e.g., Atom C2338) ---
            if logical <= 4:
                tier_reason = "Tier 1: Low-Power / NAS (Limited cores)"
                assigned = 2
            
            # --- TIER 2: High-End / Hybrid (e.g., Ultra 7 155H) ---
            elif 16 < logical <= 22:
                tier_reason = "Tier 2: Hybrid Architecture (Skipping LP cores)"
                assigned = logical - 8
                
            # --- TIER 3: Flagship Desktop (e.g., Ultra 9 285K) ---
            else:
                tier_reason = "Tier 3: Standard / Flagship Desktop (High core count)"
                assigned = logical - 2

        elif task_type == "io":
            tier_reason = "I/O Optimized (Network/Disk Bound)"
            assigned = min(32, logical + 4)
        
        else:
            # Catch-all for typos in the task_type parameter
            tier_reason = f"Unknown task_type '{task_type}' - using default fallback"
            assigned = 4

        # Final diagnostic print
        log_msg(f"[WORKER CONFIG] Mode: {task_type.upper()} | {tier_reason}")
        log_msg(f"                Threads Detected: {logical} -> Assigned Workers: {assigned}")
        
        return assigned

    except Exception as e:
        log_msg(f"[WORKER CONFIG] ERROR: {e}. Defaulting to safe NAS fallback (2 workers).")
        return 2

# --- 1. CACHE UTILITIES ---

def load_essentia_cache_exclusive():
    """Reads the current cache from disk using a shared lock to ensure multi-process safety."""
    if os.path.exists(ESSENTIA_CACHE_PATH):
        try:
            with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='r', timeout=10) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# --- 2. WORKER WRAPPER ---

def analysis_worker(track_id):
    """
    Worker function to fetch and analyze a single track.
    RatingKey is passed instead of the object to minimize pickling overhead.
    Utilizes a local Plex session to ensure stability across multiple processes.
    """
    try:
        # Initialize a new local connection session for process isolation
        local_plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=60)
        track = local_plex.fetchItem(track_id)
        
        # This call now handles metadata syncing for cached tracks 
        # and full analysis for new tracks.
        result = analyze_track_essentia(track)
        return str(track_id), result
    except Exception:
        return str(track_id), None

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
        # Only process if not in cache, or if the metadata sync is older than 7 days
        if rk not in _essentia_cache:
            to_process.append(t.ratingKey)
        else:
            cached_data = _essentia_cache[rk]
            last_sync = cached_data.get("last_synced", 0)
            
            # Check if path has changed or if time limit has expired
            # get_local_path is imported via meloday (used in analyze_track_essentia logic)
            from meloday import get_local_path
            current_path = get_local_path(t)
            
            if cached_data.get("file_path") != current_path or (now_ts - last_sync) > 604800:
                to_process.append(t.ratingKey)

    num_to_process = len(to_process)
    if num_to_process == 0:
        log_msg("--- Success! All tracks are analyzed and metadata is up to date. ---")
        return

    log_msg(f"Found {len(all_tracks)} total tracks. Processing {num_to_process} for analysis/sync...")

    start_time = time.time()
    batch_size = 50
    completed = 0

    # 2. Maximize workers while managing memory via worker-restarting (Python 3.11+)
    # Workers restart after every 5 tracks to prevent memory bloat from accumulating
    workers = get_optimal_workers(task_type="cpu")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=5) as executor:
        for i in range(0, num_to_process, batch_size):
            batch_ids = to_process[i:i + batch_size]
            future_to_track = {executor.submit(analysis_worker, tid): tid for tid in batch_ids}
            
            for future in concurrent.futures.as_completed(future_to_track):
                rk, data = future.result()
                if data:
                    _essentia_cache[rk] = data
                completed += 1

            # Periodic status updates with time estimation
            elapsed = time.time() - start_time
            avg = elapsed / completed
            est = timedelta(seconds=int((num_to_process - completed) * avg))
            log_msg(f"Progress: [{completed}/{num_to_process}] | Est: {est} | In Cache: {len(_essentia_cache)} ", end='\r')
        
            # 3. Atomic Save: Merge memory with disk every 50 tracks to prevent data loss
            disk_cache = load_essentia_cache_exclusive()
            disk_cache.update(_essentia_cache)
            with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='w', timeout=60) as f:
                json.dump(disk_cache, f)

    # Final persistent save
    save_essentia_cache()
    log_msg(f"\n--- Success! Analysis and Sync complete. ---")
    log_msg(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()