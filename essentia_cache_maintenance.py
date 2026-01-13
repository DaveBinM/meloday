import os
import json
import portalocker
from plexapi.server import PlexServer
from meloday import (
    PLEX_URL, PLEX_TOKEN, MUSIC_LIBRARY, 
    ESSENTIA_CACHE_PATH
)

def cleanup_orphaned_cache_entries():
    """
    Removes entries from essentia_cache.json that no longer exist 
    as tracks in the specified Plex music library.
    """
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        print(f"[ERROR] Cache file not found at {ESSENTIA_CACHE_PATH}")
        return

    print(f"--- Starting Cache Cleanup: {MUSIC_LIBRARY} ---")
    
    try:
        # 1. Connect to Plex first (If this fails, the script exits safely)
        plex = PlexServer(PLEX_URL, PLEX_TOKEN)
        music = plex.library.section(MUSIC_LIBRARY)
        
        print("Fetching current library state from Plex...")
        # Use a set for O(1) lookup performance
        current_library_keys = {str(t.ratingKey) for t in music.search(libtype='track')}
        
        # SAFETY CHECK: If Plex returns 0 tracks, assume something is wrong (unmounted/empty)
        if not current_library_keys:
            print("[ABORT] Plex returned 0 tracks. Library may be unmounted or empty. Skipping cleanup.")
            return

        # 2. Open and lock the cache file ONLY after confirming Plex is accessible
        with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='r+', timeout=60) as f:
            cache = json.load(f)
            original_count = len(cache)
            
            if original_count == 0:
                print("Cache is already empty. Nothing to clean.")
                return

            # Identify keys in cache that are NOT in the current library
            orphaned_keys = [rk for rk in cache.keys() if rk not in current_library_keys]
            
            if not orphaned_keys:
                print("--- Success! No orphaned entries found. Cache is clean. ---")
                return

            # Remove the orphans
            for rk in orphaned_keys:
                del cache[rk]
            
            # Save the cleaned cache back to disk
            f.seek(0)
            json.dump(cache, f)
            f.truncate()
            
            print(f"\n--- Cleanup Complete ---")
            print(f"Entries Scanned: {original_count}")
            print(f"Orphans Removed: {len(orphaned_keys)}")
            print(f"Final Cache Size: {len(cache)}")

    except Exception as e:
        # If Plex is offline or there is a connection error, it lands here.
        # No file writes have happened yet, so the cache is safe.
        print(f"[ERROR] Cleanup failed: {e}")

if __name__ == "__main__":
    cleanup_orphaned_cache_entries()