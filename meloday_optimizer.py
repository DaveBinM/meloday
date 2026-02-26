import os
import multiprocessing
import functools
import sys
import statistics
import random
import concurrent.futures
import sqlite3
import logging
from datetime import datetime, timedelta
from collections import Counter
from plexapi.server import PlexServer
from tqdm import tqdm
from ruamel.yaml import YAML

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- LOGGING SETUP ---
LOG_FILE = os.path.join(BASE_DIR, "logs", "optimizer.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("Optimizer")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def log_msg(msg, level="info", log_only=False):
    """Filters for printable characters to ensure a clean, plain-text log."""
    if not msg: return
    
    # Remove non-printable characters and NUL bytes
    clean_msg = "".join(ch for ch in str(msg) if ch.isprintable() or ch in ("\n", "\r", "\t"))
    clean_msg = clean_msg.replace('\x00', '')
    
    if not log_only: 
        print(msg)  # Standard print for console
        
    if level == "error": 
        logger.error(clean_msg)
    else: 
        logger.info(clean_msg)

@functools.lru_cache(maxsize=None)
def get_optimal_workers(task_type="cpu"):
    try:
        logical = os.cpu_count() or 1

        if task_type == "cpu":
            try:
                import psutil
                physical = psutil.cpu_count(logical=False) or logical
                source = "psutil"
            except ImportError:
                physical = max(1, logical // 2) if logical > 2 else logical
                source = "estimated"
            assigned = max(1, physical - 1)
            tier_reason = f"CPU-bound | Physical cores: {physical} ({source})"

        elif task_type == "io":
            assigned = min(16, logical + 4)
            tier_reason = "I/O Optimized (Network/Disk Bound)"

        else:
            tier_reason = f"Unknown task_type '{task_type}' — using default fallback"
            assigned = max(1, logical // 2)

        log_msg(f"[WORKER CONFIG] Mode: {task_type.upper()} | {tier_reason}")
        log_msg(f"                Threads Detected: {logical} -> Assigned Workers: {assigned}")

        return assigned

    except Exception as e:
        log_msg(f"[WORKER CONFIG] ERROR: {e}. Defaulting to safe fallback (2 workers).")
        return 2

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
        
        buckets = {"ultra": 0, "strong": 0, "logical": 0, "loose": 0}
        distances = []
        for s in similar:
            dist = getattr(s, 'distance', 0.25)
            distances.append(dist)
            if dist <= 0.10: buckets["ultra"] += 1
            elif dist <= 0.15: buckets["strong"] += 1
            elif dist <= 0.20: buckets["logical"] += 1
            else: buckets["loose"] += 1
        
        return {
            "neighbors": len(similar),
            "usable_neighbors": buckets["ultra"] + buckets["strong"] + buckets["logical"],
            "buckets": buckets,
            "distances": distances,
            "genres": [g.tag for g in track.genres],
            "bpm": technical.get("bpm"),
            "energy": technical.get("energy"),
            "year": technical.get("year") or track.year,
            "has_essentia": rk in cache_data
        }
    except Exception: return None

def run_optimizer():
    # Initialize the YAML handler for RoundTrip (Comment preservation)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        # ruamel.yaml.load preserves the structure and comments
        config = yaml.load(f)

    plex = PlexServer(config['plex']['url'], config['plex']['token'])
    music = plex.library.section(config['plex']['music_library'])
    
    ess_cfg = config.get('essentia', {})
    cache_path = os.path.join(base_dir, ess_cfg.get('cache_path', 'assets/essentia_cache.db'))
    if cache_path.endswith('.json'):
        cache_path = cache_path[:-5] + '.db'

    cache_data = {}
    if os.path.exists(cache_path):
        try:
            conn = sqlite3.connect(cache_path, timeout=10)
            for row in conn.execute("SELECT rating_key, bpm, energy, year FROM essentia_cache"):
                rk, bpm, energy, year = row
                cache_data[rk] = {"bpm": bpm, "energy": energy, "year": year}
            conn.close()
        except Exception:
            pass

    sample_size = get_args()
    all_tracks = music.searchTracks()
    total_count = len(all_tracks)
    target_tracks = all_tracks if (sample_size is None or sample_size >= total_count) else random.sample(all_tracks, sample_size)

    log_msg(f"\n--- Meloday Universal Optimizer ---")
    log_msg(f"MODE: {'FULL AUDIT' if sample_size is None else f'SAMPLING {sample_size}'}")

    results = []
    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        # Start history fetch immediately so it runs concurrently with track analysis.
        # On large libraries, fetching 90 days of history can take 10-30 seconds —
        # overlapping it with the sonicallySimilar calls gives it for free.
        history_future = executor.submit(
            music.history, mindate=datetime.now() - timedelta(days=90)
        )

        futures = {executor.submit(get_track_data, t, cache_data): t for t in target_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_tracks), desc="Analyzing"):
            res = future.result()
            if res: results.append(res)

        history = history_future.result()

    unique_played = {entry.ratingKey for entry in history}

    if not results:
        log_msg("No data collected. Check Plex connection.", level="error")
        return

    # --- DIVERSITY & DENSITY ---
    all_genres = [g for r in results for g in r['genres']]
    N = len(all_genres)
    diversity = 1 - (sum(n * (n - 1) for n in Counter(all_genres).values()) / (N * (N - 1))) if N > 1 else 0

    # Bucket-aware density logic (targeting usable pool)
    usable_counts = [r['usable_neighbors'] for r in results if r['usable_neighbors'] > 0]
    rec_limit = int(statistics.quantiles(usable_counts, n=100)[84]) if usable_counts else 150
    rec_dist = min(round(statistics.median([d for r in results for d in r['distances']]) if results else 0.20, 2), 0.25)

    def get_weight(data_list, base_val):
        clean = [x for x in data_list if x is not None]
        if len(clean) < 2: return base_val
        return round(min(base_val + ((statistics.stdev(clean) / statistics.mean(clean)) * 0.5), 0.20), 2)

    # Values for output/saving
    rec_hist_ratio = round(max(0.15, 0.40 - (diversity * 0.25)), 2)
    rec_genre_ratio = round(max(0.10, 0.05 + (diversity * 0.15)), 2)
    rec_key_weight = 0.20 if diversity < 0.4 else 0.15

    # Pre-compute Essentia weights once — used in both output and auto-apply sections
    # to avoid rebuilding the same list comprehensions and re-running stdev/mean twice.
    essentia_results_exist = any(r['has_essentia'] for r in results)
    if essentia_results_exist:
        w_bpm    = get_weight([r['bpm'] for r in results], 0.12)
        w_energy = get_weight([abs(r['energy']) for r in results if r['energy']], 0.08)
        w_era    = get_weight([r['year'] for r in results if r['year']], 0.03)
    
    # Play Ratio: The % of your library explored in 3 months.
    # This is the "Universal Metric" that works from 5k to 150k tracks.
    play_ratio = len(unique_played) / max(1, total_count)
    
    # 1. BEST EXCLUSION TIME (The FLOW Guard)
    # Every 4% of depth adds 1 day. 
    # Capped at 4 days to keep "Sonic Twins" available.
    rec_exclude = min(4, max(1, round(play_ratio * 25)))
    
    # 2. BEST LOOKBACK TIME (The SEED Net)
    # Scales smoothly from 120 days (low depth) down to 30 days (high depth).
    # Logic: The more you listen, the faster your "vibe" changes, 
    # so we need a shorter lookback to find relevant seeds.
    rec_lookback = max(30, min(120, round(120 - (play_ratio * 300))))

    # --- CACHE WARNING ---
    essentia_hits = sum(1 for r in results if r['has_essentia'])
    hit_rate = (essentia_hits / len(results)) * 100
    if hit_rate < 80 and ess_cfg.get("enabled", False):
        log_msg(f"\n[!] WARNING: Only {hit_rate:.1f}% of tracks are analyzed. Weights may be inaccurate.")
        log_msg(f"    Recommendation: Run your analysis script before finalizing weights.")

    # Summing bucket totals for distribution log
    sum_ultra = sum(r['buckets']['ultra'] for r in results)
    sum_strong = sum(r['buckets']['strong'] for r in results)
    sum_logical = sum(r['buckets']['logical'] for r in results)
    sum_loose = sum(r['buckets']['loose'] for r in results)

    # --- OUTPUT ---
    log_msg("\n" + "="*50, log_only=True)
    log_msg(" GLOBAL SIMILARITY DISTRIBUTION (Audit Data):", log_only=True)
    log_msg(f"  Ultra-Tight (0.10): {sum_ultra}", log_only=True)
    log_msg(f"  Strong Flow (0.15): {sum_strong}", log_only=True)
    log_msg(f"  Logical     (0.20): {sum_logical}", log_only=True)
    log_msg(f"  Loose       (0.25): {sum_loose}", log_only=True)

    log_msg("\n" + "="*50)
    log_msg(" RECOMMENDED CONFIGURATION")
    log_msg("="*50)
    log_msg("playlist:")
    log_msg(f"  exclude_played_days: {rec_exclude}")
    log_msg(f"  history_lookback_days: {rec_lookback}")
    log_msg(f"  sonic_similarity_limit: {rec_limit}")
    log_msg(f"  historical_ratio: {rec_hist_ratio}")
    log_msg(f"  genre_ratio: {rec_genre_ratio}")
    log_msg(f"  sonic_similarity_distance: {rec_dist}")
    
    if essentia_results_exist:
        log_msg("\nessentia:")
        log_msg(f"  bpm_weight: {w_bpm:.2f}")
        log_msg(f"  key_weight: {rec_key_weight:.2f}")
        log_msg(f"  energy_weight: {w_energy:.2f}")
        log_msg(f"  era_weight: {w_era:.2f}")
    log_msg("="*50)

    # --- AUTO-APPLY LOGIC ---
    if config['playlist'].get('auto_apply_optimization', False):
        config['playlist']['exclude_played_days'] = rec_exclude
        config['playlist']['history_lookback_days'] = rec_lookback
        config['playlist']['sonic_similarity_limit'] = rec_limit
        config['playlist']['historical_ratio'] = rec_hist_ratio
        config['playlist']['genre_ratio'] = rec_genre_ratio
        config['playlist']['sonic_similarity_distance'] = rec_dist

        if ess_cfg.get("enabled", False) and essentia_results_exist:
            config['essentia']['bpm_weight'] = w_bpm
            config['essentia']['key_weight'] = rec_key_weight
            config['essentia']['energy_weight'] = w_energy
            config['essentia']['era_weight'] = w_era

        with open(config_path, 'w', encoding='utf-8') as f:
            # Saving with the YAML object preserves comments
            yaml.dump(config, f)
        log_msg("\n[INFO] Optimized values applied to config.yml (Comments preserved)")

if __name__ == "__main__":
    run_optimizer()