import yaml
import os
import sys
import statistics
import random
import concurrent.futures
import json
from datetime import datetime, timedelta
from collections import Counter
from plexapi.server import PlexServer
from tqdm import tqdm

# --- DEFAULTS ---
DEFAULT_SAMPLE_SIZE = 2000

def get_args():
    if len(sys.argv) < 2: return DEFAULT_SAMPLE_SIZE
    arg = sys.argv[1].upper()
    if arg == "ALL": return None
    try: return int(arg)
    except ValueError: return DEFAULT_SAMPLE_SIZE

def get_track_data(track, cache_data):
    try:
        rk = str(track.ratingKey)
        similar = track.sonicallySimilar(limit=500)
        technical = cache_data.get(rk, {})
        
        return {
            "neighbors": len(similar),
            "distances": [getattr(s, 'distance', 0.25) for s in similar if hasattr(s, 'distance')],
            "genres": [g.tag for g in track.genres],
            "bpm": technical.get("bpm"),
            "energy": technical.get("energy"),
            "year": technical.get("year") or track.year,
            "has_essentia": rk in cache_data
        }
    except Exception: return None

def run_optimizer():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    plex = PlexServer(config["plex"]["url"], config["plex"]["token"])
    music = plex.library.section(config["plex"]["music_library"])
    ess_cfg = config.get("essentia", {})
    
    # Load Cache
    cache_path = os.path.join(base_dir, ess_cfg.get("cache_path", "assets/essentia_cache.json"))
    cache_data = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f: cache_data = json.load(f)

    # --- BEHAVIORAL AUDIT ---
    history = music.history(mindate=datetime.now() - timedelta(days=90))
    unique_played = {entry.ratingKey for entry in history}
    total_library = music.totalViewSize()
    
    velocity = len(unique_played) / 90
    play_ratio = len(unique_played) / max(1, total_library)
    
    if play_ratio > 0.15: rec_exclude = 7
    elif play_ratio > 0.05: rec_exclude = 3
    else: rec_exclude = 1 
    
    if velocity < 2: rec_lookback = 120 
    elif velocity < 5: rec_lookback = 90
    else: rec_lookback = 60

    sample_size = get_args()
    all_tracks = music.searchTracks()
    total_count = len(all_tracks)
    target_tracks = all_tracks if (sample_size is None or sample_size >= total_count) else random.sample(all_tracks, sample_size)

    print(f"\n--- Meloday Universal Optimizer ---")
    print(f"MODE: {'FULL AUDIT' if sample_size is None else f'SAMPLING {sample_size}'}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(get_track_data, t, cache_data): t for t in target_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_tracks), desc="Analyzing"):
            res = future.result()
            if res: results.append(res)

    # --- CACHE WARNING ---
    essentia_hits = sum(1 for r in results if r['has_essentia'])
    hit_rate = (essentia_hits / len(results)) * 100
    if hit_rate < 80 and ess_cfg.get("enabled", False):
        print(f"\n[!] WARNING: Only {hit_rate:.1f}% of tracks are analyzed. Weights may be inaccurate.")
        print(f"    Recommendation: Run your analysis script before finalizing weights.")

    # Diversity & Density
    all_genres = [g for r in results for g in r['genres']]
    N = len(all_genres)
    diversity = 1 - (sum(n * (n - 1) for n in Counter(all_genres).values()) / (N * (N - 1))) if N > 1 else 0

    neighbor_counts = [r['neighbors'] for r in results if r['neighbors'] > 0]
    rec_limit = int(statistics.quantiles(neighbor_counts, n=100)[84]) if neighbor_counts else 150
    rec_dist = min(round(statistics.median([d for r in results for d in r['distances']]) if results else 0.20, 2), 0.25)

    def get_weight(data_list, base_val):
        clean = [x for x in data_list if x is not None]
        if len(clean) < 2: return base_val
        return round(min(base_val + ((statistics.stdev(clean) / statistics.mean(clean)) * 0.5), 0.20), 2)

    # Values for output/saving
    rec_hist_ratio = round(max(0.15, 0.40 - (diversity * 0.25)), 2)
    rec_genre_ratio = round(max(0.10, 0.05 + (diversity * 0.15)), 2)
    rec_key_weight = 0.20 if diversity < 0.4 else 0.15

    # Output Recommendations
    print(f"\n" + "="*50 + "\n RECOMMENDED CONFIGURATION\n" + "="*50)
    print(f"playlist:")
    print(f"  exclude_played_days: {rec_exclude}")
    print(f"  history_lookback_days: {rec_lookback}")
    print(f"  sonic_similar_limit: {rec_limit}")
    print(f"  historical_ratio: {rec_hist_ratio}")
    print(f"  genre_ratio: {rec_genre_ratio}")
    print(f"  sonic_similarity_distance: {rec_dist:.2f}")
    if ess_cfg.get("enabled", False):
        print(f"\nessentia:")
        print(f"  bpm_weight: {get_weight([r['bpm'] for r in results], 0.12):.2f}")
        print(f"  key_weight: {rec_key_weight:.2f}")
        print(f"  energy_weight: {get_weight([abs(r['energy']) for r in results if r['energy']], 0.08):.2f}")
        print(f"  era_weight: {get_weight([r['year'] for r in results if r['year']], 0.03):.2f}")
    print("="*50)

    # --- AUTO-APPLY LOGIC ---
    if config['playlist'].get('auto_apply_optimization', False):
        config['playlist']['exclude_played_days'] = rec_exclude
        config['playlist']['history_lookback_days'] = rec_lookback
        config['playlist']['sonic_similar_limit'] = rec_limit
        config['playlist']['historical_ratio'] = rec_hist_ratio
        config['playlist']['genre_ratio'] = rec_genre_ratio
        config['playlist']['sonic_similarity_distance'] = rec_dist

        if ess_cfg.get("enabled", False):
            config['essentia']['bpm_weight'] = get_weight([r['bpm'] for r in results], 0.12)
            config['essentia']['key_weight'] = rec_key_weight
            config['essentia']['energy_weight'] = get_weight([abs(r['energy']) for r in results if r['energy']], 0.08)
            config['essentia']['era_weight'] = get_weight([r['year'] for r in results if r['year']], 0.03)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n[INFO] Optimized values applied to {config_path}")

if __name__ == "__main__":
    run_optimizer()