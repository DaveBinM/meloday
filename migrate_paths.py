import os
import json
import portalocker
from plexapi.server import PlexServer
from meloday import (
    PLEX_URL, PLEX_TOKEN, MUSIC_LIBRARY, 
    ESSENTIA_CACHE_PATH, get_local_path
)

def migrate_cache_paths():
    """
    Populates 'file_path' for existing cache entries to prevent 
    unnecessary re-analysis after the code update.
    """
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        print(f"[ERROR] Cache file not found at {ESSENTIA_CACHE_PATH}")
        return

    print("--- Starting Cache Migration: Adding file paths ---")
    
    # Load the current cache with an exclusive lock
    try:
        with portalocker.Lock(ESSENTIA_CACHE_PATH, mode='r+', timeout=60) as f:
            cache = json.load(f)
            
            # Connect to Plex
            plex = PlexServer(PLEX_URL, PLEX_TOKEN)
            music = plex.library.section(MUSIC_LIBRARY)
            
            print("Fetching tracks from Plex for path matching...")
            all_tracks = {str(t.ratingKey): t for t in music.search(libtype='track')}
            
            updated_count = 0
            missing_count = 0
            
            for rk, data in cache.items():
                # Skip if already migrated
                if "file_path" in data:
                    continue
                
                track = all_tracks.get(rk)
                if track:
                    path = get_local_path(track)
                    if path:
                        data["file_path"] = path
                        updated_count += 1
                else:
                    missing_count += 1

                if updated_count % 100 == 0 and updated_count > 0:
                    print(f"Migrated {updated_count} tracks...")

            # Save the updated cache back to the file
            f.seek(0)
            json.dump(cache, f)
            f.truncate()
            
            print(f"\n--- Migration Complete ---")
            print(f"Updated: {updated_count} entries")
            print(f"Skipped (Not in Plex): {missing_count} entries")

    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")

if __name__ == "__main__":
    migrate_cache_paths()