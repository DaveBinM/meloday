import yaml
import os
import multiprocessing
import csv
import statistics
from plexapi.server import PlexServer
from tqdm import tqdm
import concurrent.futures

# 1. Setup & Config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yml")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

PLEX_URL = config["plex"]["url"]
PLEX_TOKEN = config["plex"]["token"]
MUSIC_LIBRARY = config["plex"]["music_library"]
CURRENT_LIMIT = config["playlist"]["sonic_similarity_limit"] 

plex = PlexServer(PLEX_URL, PLEX_TOKEN)
music = plex.library.section(MUSIC_LIBRARY)

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
        print(f"[WORKER CONFIG] Mode: {task_type.upper()} | {tier_reason}")
        print(f"                Threads Detected: {logical} -> Assigned Workers: {assigned}")
        
        return assigned

    except Exception as e:
        print(f"[WORKER CONFIG] ERROR: {e}. Defaulting to safe NAS fallback (2 workers).")
        return 2

def check_track_neighborhood(track):
    try:
        # Request maximum ceiling to see the full potential of the library
        similar = track.sonicallySimilar(limit=500)
        
        # Bucketing logic
        buckets = {
            "ultra_tight": 0, # 0.00 - 0.10
            "strong_flow": 0, # 0.10 - 0.15
            "logical": 0,     # 0.15 - 0.20
            "loose": 0        # 0.20 - 0.25
        }
        
        for s in similar:
            dist = getattr(s, 'distance', 0.25)
            if dist <= 0.10: buckets["ultra_tight"] += 1
            elif dist <= 0.15: buckets["strong_flow"] += 1
            elif dist <= 0.20: buckets["logical"] += 1
            else: buckets["loose"] += 1

        actual_count = len(similar)
        
        return {
            "title": track.title,
            "artist": track.grandparentTitle,
            "total_neighbors": actual_count,
            "ultra_tight": buckets["ultra_tight"],
            "strong_flow": buckets["strong_flow"],
            "logical": buckets["logical"],
            "loose": buckets["loose"],
            "is_placebo": actual_count < CURRENT_LIMIT
        }
    except:
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
    
    # 3. Recommended Limit Calculation (Focusing on Tier 1-3)
    # This suggests a limit that covers 85% of usable (non-loose) matches
    usable_counts = [r["ultra_tight"] + r["strong_flow"] + r["logical"] for r in results if r["total_neighbors"] > 0]
    recommended_limit = int(statistics.quantiles(usable_counts, n=100)[85]) if usable_counts else CURRENT_LIMIT

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
