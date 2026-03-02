import os
import multiprocessing
import functools
import sys
import statistics
import random
import concurrent.futures
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from collections import Counter
from plexapi.server import PlexServer
from tqdm import tqdm
from ruamel.yaml import YAML

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def get_track_data(track, cache_data, limit=500):
    """
    Collects sonic neighbourhood data for a single track — called in parallel across the sample.
    Fetches up to `limit` sonic neighbours from Plex, buckets them by distance, and merges
    any available Essentia acoustic data (BPM, energy, year) from the local cache.
    Returns None on any failure so the caller can safely skip it.
    """
    try:
        rk = str(track.ratingKey)
        similar = track.sonicallySimilar(limit=limit)
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
            # Prefer cache for all metadata — it's already in memory and avoids
            # redundant attribute access on Plex objects. Fall back to the Plex
            # track object for tracks not yet in the Essentia cache.
            "genres": technical.get("genres") or [g.tag for g in track.genres],
            "styles": technical.get("styles") or [s.tag for s in getattr(track, "styles", [])],
            "moods":  technical.get("moods",  []),
            "bpm":    technical.get("bpm"),
            "energy": technical.get("energy"),
            "year":   technical.get("year") or track.year,
            "has_essentia": rk in cache_data
        }
    except Exception: return None

def run_optimizer():
    # Initialize the YAML handler for RoundTrip (Comment preservation)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
            for row in conn.execute("SELECT rating_key, bpm, energy, year, styles, genres, moods FROM essentia_cache"):
                rk, bpm, energy, year, styles_json, genres_json, moods_json = row
                cache_data[rk] = {
                    "bpm":    bpm,
                    "energy": energy,
                    "year":   year,
                    "styles": json.loads(styles_json) if styles_json else [],
                    "genres": json.loads(genres_json) if genres_json else [],
                    "moods":  json.loads(moods_json)  if moods_json  else [],
                }
            conn.close()
        except Exception:
            pass

    # Read playlist target size from config — used to derive all ratio recommendations below.
    MAX_TRACKS_CFG = config['playlist'].get('max_tracks', 50)

    # --- STYLE MEDIAN FROM CACHE ---
    # Count style tags only for tracks that actually have styles.
    # Genre counts are intentionally excluded — genres (typically 1–2 per track) and styles
    # (typically 4–10 per track) are different tag systems with different distributions.
    # Mixing them would drag the median down and cause rec_style_tag_depth and rec_style_ratio
    # to understate how many style slots styled tracks actually carry.
    # If no styled tracks exist in the cache, the default of 4 is used as a safe fallback.
    style_tag_counts = [len(entry["styles"]) for entry in cache_data.values() if entry.get("styles")]
    style_median_per_track = statistics.median(style_tag_counts) if style_tag_counts else 4

    # --- GENRE AND STYLE DIVERSITY FROM CACHE ---
    # Built from ALL cached tracks for maximum accuracy — not just the sonic sample.
    # These Counters drive global diversity scoring, anti-starvation floors, and
    # the audit log output. style_tag_counts (above) drives the per-track median logic.
    genre_counter = Counter(g for entry in cache_data.values() for g in entry.get("genres", []))
    style_counter = Counter(s for entry in cache_data.values() for s in entry.get("styles", []))

    N = sum(genre_counter.values())
    diversity = 1 - (sum(n * (n - 1) for n in genre_counter.values()) / (N * (N - 1))) if N > 1 else 0

    M = sum(style_counter.values())
    style_diversity = 1 - (sum(n * (n - 1) for n in style_counter.values()) / (M * (M - 1))) if M > 1 else 0

    # --- MOOD METRICS FROM CACHE ---
    # Count mood tags per track (tracks with no moods are excluded — they bypass mood caps).
    # mood_counter drives diversity, distinct count, and the anti-starvation floor.
    # mood_coverage_pct tells us how much of the library the mood cap will actually affect.
    mood_tag_counts = []
    mood_counter = Counter()
    for entry in cache_data.values():
        track_moods = entry.get("moods", [])
        if track_moods:
            mood_tag_counts.append(len(track_moods))
            for m in track_moods:
                mood_counter[m] += 1

    mood_median_per_track = statistics.median(mood_tag_counts) if mood_tag_counts else 3
    mood_tagged = sum(1 for entry in cache_data.values() if entry.get("moods"))
    mood_coverage_pct = (mood_tagged / max(1, len(cache_data))) * 100

    sample_size = get_args()
    all_tracks = music.searchTracks()
    total_count = len(all_tracks)

    # Scale the default sample to at least 5% of the library.
    # On a 5k library this stays at 2000 (40%); on a 300k library it grows to 15000 (5%).
    # Explicitly-passed values (including "ALL") are always respected as-is.
    if sample_size == DEFAULT_SAMPLE_SIZE:
        sample_size = max(DEFAULT_SAMPLE_SIZE, int(total_count * 0.05))

    target_tracks = all_tracks if (sample_size is None or sample_size >= total_count) else random.sample(all_tracks, sample_size)

    log_msg(f"\n--- Meloday Universal Optimizer ---")
    log_msg(f"MODE: {'FULL AUDIT' if sample_size is None else f'SAMPLING {sample_size} of {total_count}'}")

    results = []
    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        # Start history fetch immediately so it runs concurrently with track analysis.
        # On large libraries, fetching 90 days of history can take 10-30 seconds —
        # overlapping it with the sonicallySimilar calls gives it for free.
        history_future = executor.submit(
            music.history, mindate=datetime.now() - timedelta(days=90)
        )

        sonic_sample_limit = int(4.5 * MAX_TRACKS_CFG)
        futures = {executor.submit(get_track_data, t, cache_data, sonic_sample_limit): t for t in target_tracks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_tracks), desc="Analyzing"):
            res = future.result()
            if res: results.append(res)

        try:
            history = history_future.result(timeout=120)
        except Exception:
            log_msg("[WARN] History fetch timed out or failed. Play-ratio metrics will be unavailable.")
            history = []

    unique_played = {entry.ratingKey for entry in history}

    if not results:
        log_msg("No data collected. Check Plex connection.", level="error")
        return

    # --- SONIC DENSITY ---
    # Bucket-aware density logic (targeting usable pool)
    usable_counts = [r['usable_neighbors'] for r in results if r['usable_neighbors'] > 0]
    rec_limit = min(int(statistics.quantiles(usable_counts, n=100)[84]), 500, int(4.5 * MAX_TRACKS_CFG)) if usable_counts else min(150, int(4.5 * MAX_TRACKS_CFG))

    def get_weight(data_list, base_val):
        clean = [x for x in data_list if x is not None]
        if len(clean) < 2: return base_val
        return round(min(base_val + ((statistics.stdev(clean) / statistics.mean(clean)) * 0.5), 0.20), 2)

    # --- RATIO RECOMMENDATIONS ---

    # HISTORICAL RATIO: how much of the playlist can come from listening history.
    # Shorter playlists benefit from more history to feel personal; longer ones need
    # fresh sonic exploration to avoid repetition.
    # Scales from ~0.45 (small playlist) down to a hard floor of 0.30 (large playlist).
    # Floor raised from 0.15 → 0.30: at least 30% of tracks must come from actual listening
    # history to preserve temporal personalisation (the core Daylist-like signal) at all sizes.
    # In large libraries (100k–300k tracks) where even heavy listeners cover <5% of content,
    # the history pool per daypart is still comfortably deep enough to fill this floor.
    rec_hist_ratio = round(max(0.30, 0.50 - (MAX_TRACKS_CFG * 0.005) - (diversity * 0.10)), 2)

    # GENRE RATIO: genres are now purely a fallback for tracks with no style tags.
    # Fixed at 0.10 — no need to tune based on diversity; the style cap does that work.
    rec_genre_ratio = 0.10

    # STYLE RATIO: derived from the median style-tag count per track in the Essentia cache.
    # More distinct styles per track → more natural variety → we can afford a tighter per-style cap.
    # n_styles = how many distinct style slots we expect to need in a typical playlist.
    # Capped at MAX_TRACKS_CFG // 3 to prevent absurdly small per-style limits on tiny playlists.
    n_styles = max(4, min(round(style_median_per_track * 2), MAX_TRACKS_CFG // 3))
    rec_style_ratio = round(max(0.10, 1.0 / n_styles), 2)

    # How many style slots to check per track — the depth into a track's style list.
    # Derived from the median: if most tracks carry 3 styles, we check all 3.
    # Capped at 5 to avoid diminishing returns on heavily-tagged libraries.
    rec_style_tag_depth = max(1, min(round(style_median_per_track), 5))

    # Anti-starvation floor: if the library has very few distinct styles, the per-style cap
    # must be permissive enough that a full playlist can actually be assembled.
    # 1.5× headroom prevents the cap being so tight it starves shorter playlists.
    # Only ever raises rec_style_ratio — never lowers it.
    num_distinct_styles = len(style_counter)
    if num_distinct_styles > 0:
        starvation_floor = round(min(1.5 / num_distinct_styles, 0.25), 2)
        if starvation_floor > rec_style_ratio:
            rec_style_ratio = starvation_floor
            log_msg(f" [NOTE] style_ratio raised to {rec_style_ratio} (anti-starvation: only {num_distinct_styles} distinct styles detected).", log_only=True)

    # ARTIST RATIO: ensures at least 2 tracks per artist are possible (avoids single-artist dominance).
    # 2/MAX_TRACKS_CFG as a floor so no single artist can fill more than their fair share.
    # Capped at 0.10 — a playlist shouldn't lean heavily on one artist even if the library is small.
    rec_artist_ratio = round(min(0.10, max(2.0 / MAX_TRACKS_CFG, 0.04)), 2)

    # MOOD RATIO: caps any single primary mood to prevent vibe monotony.
    # n_moods = expected distinct mood slots needed, anchored to median mood tags per track.
    # Capped at MAX_TRACKS_CFG // 4 so the limit stays meaningful on small playlists.
    # Floor raised from 0.20 → 0.33: allows one mood to run through ~1/3 of the playlist,
    # matching Daylist's "one dominant vibe" character. At 0.20, the cap forced distribution
    # across 5+ moods regardless of playlist size — working against temporal coherence.
    n_moods = max(3, min(round(mood_median_per_track * 2), MAX_TRACKS_CFG // 4))
    rec_mood_ratio = round(max(0.33, 1.0 / n_moods), 2)

    # Anti-starvation floor: if very few distinct moods exist, the cap can't be so tight
    # it prevents a full playlist from being built. 1.5× headroom, ceiling at 0.50.
    num_distinct_moods = len(mood_counter)
    if num_distinct_moods > 0:
        mood_starvation_floor = round(min(1.5 / num_distinct_moods, 0.50), 2)
        if mood_starvation_floor > rec_mood_ratio:
            rec_mood_ratio = mood_starvation_floor
            log_msg(f" [NOTE] mood_ratio raised to {rec_mood_ratio} (anti-starvation: only {num_distinct_moods} distinct moods detected).", log_only=True)

    rec_key_weight = 0.20 if diversity < 0.4 else 0.15

    # Pre-compute Essentia weights from the full cache — more stable than a random sample
    # because BPM/energy/era variation is a library-wide property, not a sample property.
    # A biased sample could easily under-represent certain BPM ranges or eras and skew CoV.
    # essentia_hits remains sample-based — it's only used for the hit_rate diagnostic below.
    essentia_hits = sum(1 for r in results if r['has_essentia'])
    essentia_results_exist = any(entry.get('bpm') for entry in cache_data.values())
    if essentia_results_exist:
        w_bpm    = get_weight([entry['bpm']    for entry in cache_data.values() if entry.get('bpm')],    0.12)
        w_energy = get_weight([abs(entry['energy']) for entry in cache_data.values() if entry.get('energy')], 0.08)
        # Centre years around 1950 before computing CoV.
        # Raw years (mean ≈ 2011) give CoV ≈ 0.006 — nearly zero — so the formula
        # would always return the base value of 0.03 regardless of era spread.
        # Centering gives a meaningful CoV: e.g. stdev=12, mean_from_1950=61 → CoV=0.20.
        w_era    = get_weight([entry['year'] - 1950 for entry in cache_data.values() if entry.get('year')], 0.03)
    
    # Two signals used for exclusion and lookback recommendations:
    #   play_ratio   — coverage: % of library explored in 3 months.
    #                  Used for lookback: small/active libraries need a shorter window
    #                  so the playlist reflects current taste, not stale history.
    #   daily_unique — frequency: average unique tracks played per day.
    #                  Scale-independent; provides an uplift for heavy listeners.
    play_ratio   = len(unique_played) / max(1, total_count)
    daily_unique = len(unique_played) / 90

    if unique_played:
        # 1. BEST EXCLUSION TIME (The FLOW Guard)
        # Scaled directly from rec_limit — the actual sonic neighbour count.
        # A higher rec_limit means more bridge/sonic candidates are available, so a
        # longer exclusion window is safe (more variety, less repetition).
        # A lower rec_limit means fewer candidates, so exclusion stays short to avoid
        # starving bridging and sonic similarity searches.
        # 1 + round(rec_limit / 75) maps the typical 50–225 candidate range to 2–4 days.
        # Floor of 2: minimum freshness — no track on consecutive days.
        # Cap of 7: beyond this, sparse daypart pools risk running low.
        rec_exclude = min(7, max(2, 1 + round(rec_limit / 75)))

        # 2. BEST LOOKBACK TIME (The SEED Net)
        # Scales from 90 days (sparse/large library needs full depth) down to 30 days
        # (small/active library: recent taste matters more than long history).
        # Takes the stronger penalty from either signal so both large-library heavy
        # listeners and small-library active listeners get an appropriately short window.
        # Capped at 90 to match the optimizer's own history fetch window — recommending
        # more than 90 days would be inconsistent with the data actually analysed.
        rec_lookback = max(30, min(90, round(90 - max(play_ratio * 300, daily_unique * 0.5))))
    else:
        # History unavailable — fall back to config.yml defaults
        rec_exclude  = 3
        rec_lookback = 60

    # --- CACHE WARNINGS ---
    # hit_rate: % of the sonic sample covered by the Essentia cache.
    # Low coverage means sonicallySimilar() results are being evaluated with less acoustic context,
    # which may reduce the accuracy of rec_limit and rec_dist recommendations.
    hit_rate = (essentia_hits / len(results)) * 100
    if hit_rate < 80 and ess_cfg.get("enabled", False):
        log_msg(f"\n[!] WARNING: Only {hit_rate:.1f}% of the sonic sample is in the Essentia cache.")
        log_msg(f"    rec_limit and rec_dist may be less reliable. Run pre_analyze.py to improve coverage.")

    # Style coverage: if the styles column is unpopulated, the style diversity system
    # is entirely non-functional — every track falls through to the genre fallback.
    # This happens when pre_analyze.py was run before the styles column was added.
    # Computed from the full cache (not the sample) to match style_counter and diversity,
    # and to avoid false positives from the Plex-object fallback in get_track_data().
    style_hit_rate = (sum(1 for entry in cache_data.values() if entry.get('styles')) / max(1, len(cache_data))) * 100
    if style_hit_rate == 0:
        log_msg(f"\n[!] WARNING: 0% of cached tracks have style tags in the Essentia cache.")
        log_msg(f"    Style diversity is using genre as a fallback for every track.")
        log_msg(f"    Re-run pre_analyze.py to populate style tags and unlock multi-dimensional diversity.")
    elif style_hit_rate < 50:
        log_msg(f"\n[!] WARNING: Only {style_hit_rate:.1f}% of cached tracks have style tags.")
        log_msg(f"    Style diversity will be unreliable. Consider re-running pre_analyze.py.")

    # Untagged track rate: tracks with neither style nor genre bypass all diversity caps.
    # A high untagged rate means the playlist could be dominated by untagged content
    # (e.g. film/game scores) regardless of the style/genre ratios set.
    # Computed from the full cache — matches the coverage of genre_counter/style_counter.
    untagged = sum(1 for entry in cache_data.values() if not entry.get('styles') and not entry.get('genres'))
    untagged_pct = (untagged / max(1, len(cache_data))) * 100
    if untagged_pct > 30:
        log_msg(f"\n[NOTE] {untagged_pct:.1f}% of cached tracks have no style or genre tags.")
        log_msg(f"       These tracks bypass all diversity caps. Consider tagging your library,")
        log_msg(f"       or increasing style_ratio/genre_ratio to compensate.")

    # BPM doubles: Essentia sometimes detects tempo at 2× the true value for slow tracks.
    # These inflate the BPM distribution and skew bpm_weight calculations.
    # Computed from the full cache for a complete picture of the artefact rate.
    bpm_doubles = sum(1 for entry in cache_data.values() if entry.get('bpm') and entry['bpm'] >= 250)
    if bpm_doubles > 0:
        log_msg(f"\n[NOTE] {bpm_doubles} cached tracks have BPM ≥ 250 — likely Essentia tempo-doubling artefacts.")
        log_msg(f"       These may skew bpm_weight slightly. Re-analyze affected tracks to correct.")

    # Mood coverage: if very few cached tracks have mood tags, mood enforcement will only
    # apply to that fraction of the playlist. Warn when coverage is low so users understand
    # that mood_ratio will have limited impact until pre_analyze.py populates more moods.
    if ess_cfg.get("enabled", False) and mood_coverage_pct < 20:
        log_msg(f"\n[NOTE] Only {mood_coverage_pct:.1f}% of cached tracks have mood tags.")
        log_msg(f"       mood_ratio enforcement will only affect this portion of the playlist.")
        log_msg(f"       Re-run pre_analyze.py to improve mood coverage.")


    # Single-pass bucket summation
    sum_ultra = sum_strong = sum_logical = sum_loose = 0
    for r in results:
        b = r['buckets']
        sum_ultra   += b['ultra']
        sum_strong  += b['strong']
        sum_logical += b['logical']
        sum_loose   += b['loose']

    # Top styles for audit log — reuse the Counter already built above
    top_styles = style_counter.most_common(5)

    # --- OUTPUT ---
    log_msg("\n" + "="*50, log_only=True)
    log_msg(" GLOBAL SIMILARITY DISTRIBUTION (Audit Data):", log_only=True)
    log_msg(f"  Ultra-Tight (0.10): {sum_ultra}", log_only=True)
    log_msg(f"  Strong Flow (0.15): {sum_strong}", log_only=True)
    log_msg(f"  Logical     (0.20): {sum_logical}", log_only=True)
    log_msg(f"  Loose       (0.25): {sum_loose}", log_only=True)

    log_msg("\n" + "="*50, log_only=True)
    log_msg(f" STYLE DIVERSITY: {style_diversity:.3f} (Genre Diversity: {diversity:.3f})", log_only=True)
    log_msg(f" Median style tags per track (cache): {style_median_per_track:.1f} → n_styles target: {n_styles}", log_only=True)
    if top_styles:
        log_msg(" Top Styles in Library:", log_only=True)
        for style, count in top_styles:
            log_msg(f"  {style}: {count}", log_only=True)
    else:
        log_msg("  (No style tags found — consider tagging your library)", log_only=True)

    top_moods = mood_counter.most_common(5)
    log_msg("\n" + "="*50, log_only=True)
    log_msg(f" MOOD COVERAGE: {mood_coverage_pct:.1f}% ({mood_tagged} of {len(cache_data)} cached tracks)", log_only=True)
    log_msg(f" Distinct moods: {num_distinct_moods} | Median mood tags per track: {mood_median_per_track:.1f} → n_moods target: {n_moods}", log_only=True)
    if top_moods:
        log_msg(" Top Moods in Library:", log_only=True)
        for mood, count in top_moods:
            log_msg(f"  {mood}: {count}", log_only=True)
    else:
        log_msg("  (No mood tags found — re-run pre_analyze.py to populate)", log_only=True)

    log_msg("\n" + "="*50)
    log_msg(" RECOMMENDED CONFIGURATION")
    log_msg("="*50)
    log_msg("playlist:")
    log_msg(f"  exclude_played_days: {rec_exclude}")
    log_msg(f"  history_lookback_days: {rec_lookback}")
    log_msg(f"  sonic_similarity_limit: {rec_limit}")
    log_msg(f"  historical_ratio: {rec_hist_ratio}  (≤{round(MAX_TRACKS_CFG * rec_hist_ratio)} of {MAX_TRACKS_CFG} tracks from history)")
    log_msg(f"  style_ratio: {rec_style_ratio}       (≤{round(MAX_TRACKS_CFG * rec_style_ratio)} of {MAX_TRACKS_CFG} tracks per style)")
    log_msg(f"  style_tag_depth: {rec_style_tag_depth}         (check this many style slots per track; based on median {style_median_per_track:.1f} styles/track)")
    log_msg(f"  genre_ratio: {rec_genre_ratio}       (fallback only — applies to tracks with no style tags)")
    log_msg(f"  artist_ratio: {rec_artist_ratio}      (≤{round(MAX_TRACKS_CFG * rec_artist_ratio)} of {MAX_TRACKS_CFG} tracks per artist)")
    log_msg(f"  mood_ratio: {rec_mood_ratio}         (≤{round(MAX_TRACKS_CFG * rec_mood_ratio)} of {MAX_TRACKS_CFG} tracks per mood; {mood_coverage_pct:.0f}% coverage)")

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
        config['playlist']['style_ratio'] = rec_style_ratio
        config['playlist']['style_tag_depth'] = rec_style_tag_depth
        config['playlist']['artist_ratio'] = rec_artist_ratio
        config['playlist']['mood_ratio'] = rec_mood_ratio

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