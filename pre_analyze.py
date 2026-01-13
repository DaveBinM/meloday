import os
import time
import json
import portalocker
import concurrent.futures
from datetime import timedelta
from meloday import (
    plex, MUSIC_LIBRARY, analyze_track_essentia, 
    save_essentia_cache, ESSENTIA_ENABLED, 
    ESSENTIA_CACHE_PATH, _essentia_cache
)

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
    """
    try:
        track = plex.fetchItem(track_id)
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
        print("[ERROR] Essentia is not installed or enabled. Analysis cannot proceed.")
        return

    print(f"--- Starting Parallel Library Analysis & Metadata Sync: {MUSIC_LIBRARY} ---")
    
    # Fetch all tracks as a flat list for direct file access
    music_section = plex.library.section(MUSIC_LIBRARY)
    print("Fetching tracks from Plex... (This may take a minute)")
    all_tracks = music_section.search(libtype='track') 
    total = len(all_tracks)
    
    to_process = [t.ratingKey for t in all_tracks]
    num_to_process = len(to_process)
    
    print(f"Found {total} tracks. Processing analysis and metadata sync...")

    start_time = time.time()
    batch_size = 50
    completed = 0

    # Utilize ProcessPoolExecutor for true CPU parallelism
    with concurrent.futures.ProcessPoolExecutor() as executor:
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
            print(f"Progress: [{completed}/{num_to_process}] | Est: {est} | In Cache: {len(_essentia_cache)} ", end='\r')
        
            # Atomic Save: Merge memory with disk every 50 tracks to prevent data loss
            disk_cache = load_essentia_cache_exclusive()
            disk_cache.update(_essentia_cache)
            with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='w', timeout=60) as f:
                json.dump(disk_cache, f)

    # Final persistent save
    save_essentia_cache()
    print(f"\n--- Success! Analysis and Sync complete. ---")
    print(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()