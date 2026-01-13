import os
import time
import json
import portalocker
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

# --- 2. CORE ANALYSIS LOGIC ---

def bulk_analyze():
    """Iterates through the entire library to perform deep acoustic analysis."""
    if not ESSENTIA_ENABLED:
        print("[ERROR] Essentia is not installed or enabled. Analysis cannot proceed.")
        return

    print(f"--- Starting Full Library Analysis: {MUSIC_LIBRARY} ---")
    
    # Fetch all tracks as a flat list for direct file access
    music_section = plex.library.section(MUSIC_LIBRARY)
    print("Fetching tracks from Plex... (This may take a minute)")
    all_tracks = music_section.search(libtype='track') 
    total = len(all_tracks)
    
    print(f"Found {total} tracks. Beginning deep analysis (BPM, Key, Energy, Era)...")

    start_time = time.time()
    new_analyzed = 0

    for i, track in enumerate(all_tracks, 1):
        rk = str(track.ratingKey)
        
        # Simple skip for already-processed tracks (useful if you restart the script)
        if rk in _essentia_cache:
            continue
            
        # Analysis with Retry Logic for server stability
        max_retries = 3
        data = None
        for attempt in range(max_retries):
            try:
                # This call will display [DIAGNOSTIC] messages if issues occur
                data = analyze_track_essentia(track)
                break 
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n[RETRY] Attempt {attempt + 1} for '{track.title}': {e}")
                    time.sleep(2)
                else:
                    print(f"\n[SKIP] Failed to analyze '{track.title}' after {max_retries} attempts.")

        if data:
            new_analyzed += 1

        # Periodic status updates with time estimation
        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            avg = elapsed / i
            est = timedelta(seconds=int((total - i) * avg))
            print(f"Progress: [{i}/{total}] | Est: {est} | Analyzed: {new_analyzed} ", end='\r')
        
        # Atomic Save: Merge memory with disk every 50 tracks to prevent data loss
        if i % 50 == 0:
            disk_cache = load_essentia_cache_exclusive()
            disk_cache.update(_essentia_cache)
            with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='w', timeout=60) as f:
                json.dump(disk_cache, f)

    # Final persistent save
    save_essentia_cache()
    print(f"\n--- Success! Analysis complete. ---")
    print(f"Total tracks in cache: {len(_essentia_cache)}")

# --- 3. EXECUTION ---

if __name__ == "__main__":
    bulk_analyze()