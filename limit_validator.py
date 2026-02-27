"""
Meloday Limit Validator — Standalone Sonic Density Audit Tool

Scans the entire Plex music library and measures each track's sonic neighbourhood:
how many tracks Plex considers similar, and at what distance buckets they fall.

Use this to audit whether your current sonic_similarity_limit in config.yml is
appropriate for your library. A track is a 'placebo' if its true neighbor count is
below the configured limit, meaning Plex is returning everything it has — raising
the limit further won't improve that track's results.

Output: prints a density report to the console and exports a per-track CSV.
"""
import yaml
import os
import multiprocessing
import functools
import csv
import statistics
from plexapi.server import PlexServer
from tqdm import tqdm
import concurrent.futures

# --- Setup & Config ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

PLEX_URL = config["plex"]["url"]
PLEX_TOKEN = config["plex"]["token"]
MUSIC_LIBRARY = config["plex"]["music_library"]
CURRENT_LIMIT = config["playlist"].get("sonic_similarity_limit", 100)

plex = PlexServer(PLEX_URL, PLEX_TOKEN)
music = plex.library.section(MUSIC_LIBRARY)

@functools.lru_cache(maxsize=None)
def get_optimal_workers(task_type="cpu"):
    try:
        logical = os.cpu_count() or 1

        if task_type == "cpu":
            # Use all physical cores — Hyper-Threading gives little benefit for
            # compute-heavy work, and the main process is idle while workers run.
            try:
                import psutil
                physical = psutil.cpu_count(logical=False) or logical
                source = "psutil"
            except ImportError:
                physical = max(1, logical // 2) if logical > 2 else logical
                source = "estimated"
            assigned = max(1, physical)
            tier_reason = f"CPU-bound | Physical cores: {physical} ({source})"

        elif task_type == "io":
            # Threads release the GIL during network waits, so we can run far more than
            # the physical core count. Cap at 32 — enough to saturate Plex API throughput
            # without risking overwhelming a co-hosted server.
            assigned = min(32, logical + 4)
            tier_reason = "I/O Optimized (Network/Disk Bound)"

        else:
            tier_reason = f"Unknown task_type '{task_type}' — using default fallback"
            assigned = max(1, logical // 2)

        print(f"[WORKER CONFIG] Mode: {task_type.upper()} | {tier_reason}")
        print(f"                Threads Detected: {logical} -> Assigned Workers: {assigned}")

        return assigned

    except Exception as e:
        print(f"[WORKER CONFIG] ERROR: {e}. Defaulting to safe fallback (2 workers).")
        return 2

def check_track_neighborhood(track):
    """
    Fetches up to 500 sonic neighbours for a track (Plex's effective ceiling) and
    buckets them by distance. Returns None on any API failure so the caller can skip it.

    is_placebo=True means the track has fewer real neighbours than CURRENT_LIMIT,
    so raising the limit further would give no additional candidates for that track.
    """
    try:
        # Always request 500 — we want the full picture regardless of the configured limit
        similar = track.sonicallySimilar(limit=500)

        # Distance buckets mirror the meloday_optimizer.py bands
        buckets = {
            "ultra_tight": 0,  # 0.00–0.10: near-identical feel
            "strong_flow":  0, # 0.10–0.15: smooth transition
            "logical":      0, # 0.15–0.20: compatible but distinct
            "loose":        0  # 0.20–0.25: noticeable stylistic gap
        }

        for s in similar:
            dist = getattr(s, 'distance', 0.25)
            if dist <= 0.10:   buckets["ultra_tight"] += 1
            elif dist <= 0.15: buckets["strong_flow"]  += 1
            elif dist <= 0.20: buckets["logical"]       += 1
            else:              buckets["loose"]          += 1

        actual_count = len(similar)

        return {
            "title": track.title,
            "artist": track.grandparentTitle,
            "total_neighbors": actual_count,
            "ultra_tight": buckets["ultra_tight"],
            "strong_flow": buckets["strong_flow"],
            "logical":     buckets["logical"],
            "loose":       buckets["loose"],
            "is_placebo":  actual_count < CURRENT_LIMIT
        }
    except Exception:
        return None

def run_audit():
    print(f"--- Meloday Deep Density Audit ---")
    print("Fetching library tracks...")
    tracks = music.searchTracks()
    total = len(tracks)
    
    results = []
    print(f"Analyzing {total} tracks for neighborhood density...")
    
    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(check_track_neighborhood, t): t for t in tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total, desc="Auditing"):
            res = future.result()
            if res:
                results.append(res)

    # 2. Aggregation
    total_valid = len(results)
    avg_neighbors = statistics.mean([r["total_neighbors"] for r in results])
    
    # Global bucket totals
    sum_ultra = sum(r["ultra_tight"] for r in results)
    sum_strong = sum(r["strong_flow"] for r in results)
    sum_logical = sum(r["logical"] for r in results)
    sum_loose = sum(r["loose"] for r in results)
    
    print(f"\n" + "="*50)
    print(f" FINAL DENSITY REPORT ({total_valid} tracks)")
    print("="*50)
    print(f"Average Neighbors per Track: {avg_neighbors:.1f}")
    print(f"\nGLOBAL SIMILARITY DISTRIBUTION:")
    print(f"  [0.00 - 0.10] Ultra-Tight: {sum_ultra} matches")
    print(f"  [0.10 - 0.15] Strong Flow: {sum_strong} matches")
    print(f"  [0.15 - 0.20] Logical:     {sum_logical} matches")
    print(f"  [0.20 - 0.25] Loose:       {sum_loose} matches")
    
    # Recommended limit: the 85th-percentile usable-neighbour count across all tracks
    # (usable = ultra_tight + strong_flow + logical; excludes loose matches at >0.20).
    # This means 85% of your library has at least this many good neighbours —
    # setting the limit higher gives diminishing returns for most tracks.
    # Index 84 = 85th percentile in 0-indexed quantiles(n=100), consistent with meloday_optimizer.py.
    usable_counts = [r["ultra_tight"] + r["strong_flow"] + r["logical"] for r in results if r["total_neighbors"] > 0]
    recommended_limit = int(statistics.quantiles(usable_counts, n=100)[84]) if usable_counts else CURRENT_LIMIT

    print("\n" + "="*50)
    print(f" MELODAY RECOMMENDATION")
    print("="*50)
    print(f"Suggested sonic_similarity_limit: {recommended_limit}")
    
    # Advice based on the "Usable Pool"
    if recommended_limit > CURRENT_LIMIT:
        print(f"Advice: Your library has enough density to support a HIGHER limit (+{recommended_limit - CURRENT_LIMIT}).")
    else:
        print(f"Advice: Your current limit is sufficient for your library density.")

    # CSV Export
    with open("sonic_density_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "artist", "total_neighbors", "ultra_tight", "strong_flow", "logical", "loose", "is_placebo"])
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    run_audit()
