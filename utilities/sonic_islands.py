import collections
import math
import csv
import os
import sys
import concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plexapi.server import PlexServer
from tqdm import tqdm
# WHY: Plex-metadata-only analysis — never decodes audio, so skip meloday's module-level TF
# runtime + model load (~0.7-1 GB RSS). setdefault so an operator override wins.
os.environ.setdefault("MELODAY_SKIP_TF_MODELS", "1")
import meloday
from meloday import (
    load_config, primary_artist, get_optimal_workers
)

# Load Meloday configuration
config = load_config()
PLEX_URL = config["plex"]["url"]
PLEX_TOKEN = config["plex"]["token"]
MUSIC_LIBRARY = config["plex"]["music_library"]
SONIC_LIMIT = config["playlist"].get("sonic_similarity_limit", 100)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORT_FILE = os.path.join(_PROJECT_ROOT, "isolated_artists_by_album.csv")

def analyze_and_export_isolation():
    # 1. Initialize Plex and pre-fetch album-level exclusions
    meloday.plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=120)
    meloday.prefetch_label_exclusions()

    music = meloday.plex.library.section(MUSIC_LIBRARY)
    print(f"--- Analyzing Sonic Density (Excluding label-excluded albums) ---")
    print("Fetching all artists and tracks...")

    # Phase 1: Collect all eligible tracks and build the artist mapping.
    # Fetching all tracks in bulk (container_size=5000) is far faster than fetching
    # per-artist via artist.tracks() — the latter makes one API call per artist.
    # grandparentTitle is the album artist, consistent with the previous artist-based approach.
    track_to_artist = {}   # ratingKey -> artist_name
    eligible_tracks = []

    print("Fetching tracks from Plex... (This may take a minute)")
    all_tracks = music.search(libtype='track', container_size=5000)
    artist_set = set()
    for track in tqdm(all_tracks, desc="Collecting tracks"):
        if str(track.parentRatingKey) in meloday._excluded_album_keys:
            continue
        artist_name = primary_artist(getattr(track, 'grandparentTitle', '') or '')
        track_to_artist[track.ratingKey] = artist_name
        eligible_tracks.append(track)
        artist_set.add(artist_name)

    print(f"Found {len(eligible_tracks)} eligible tracks across {len(artist_set)} artists.")

    if not eligible_tracks:
        print("[ERROR] No eligible tracks found.")
        return

    # Phase 2: Parallel sonic similarity checks.
    # sonicallySimilar() is a network-bound Plex API call — ThreadPoolExecutor
    # lets many requests fly concurrently without the overhead of spawning processes.
    def check_track(track):
        try:
            return track.ratingKey, len(track.sonicallySimilar(limit=SONIC_LIMIT))
        except Exception:
            return track.ratingKey, 0

    artist_map = collections.defaultdict(list)
    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(check_track, t): t for t in eligible_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(eligible_tracks), desc="Analyzing"):
            rk, count = future.result(timeout=120)
            artist_map[track_to_artist[rk]].append(count)

    # 3. Calculate averages for artists who have eligible (non-excluded) tracks
    final_stats = []
    for artist_name, neighbor_counts in artist_map.items():
        if neighbor_counts:
            avg_neighbors = sum(neighbor_counts) / len(neighbor_counts)
            final_stats.append((artist_name, avg_neighbors))

    # Sort by most isolated (lowest average matches)
    final_stats.sort(key=lambda x: x[1])

    # 4. Filter for Top 5%
    cutoff = math.ceil(len(final_stats) * 0.05)
    isolated_pool = final_stats[:cutoff]

    # 5. Output to CSV
    with open(EXPORT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Artist", "Average Sonic Matches"])
        for name, score in isolated_pool:
            writer.writerow([name, f"{score:.2f}"])

    print(f"\n[SUCCESS] Analysis complete. {len(isolated_pool)} artists exported.")
    print(f"Results saved to: {EXPORT_FILE}")

if __name__ == "__main__":
    analyze_and_export_isolation()
