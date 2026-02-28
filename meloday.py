import yaml
import os

# This must happen before any essentia imports to stop the SVM Info messages
os.environ['ESSENTIA_LOG_LEVEL'] = '1' 

import re
import random
import json
import functools
import unicodedata
import sqlite3
import traceback
import concurrent.futures
import multiprocessing
import logging
import argparse
from datetime import datetime, timedelta
from collections import Counter
from plexapi.server import PlexServer
from plexapi.audio import Track
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --- 1. ENVIRONMENT & ESSENTIA DETECTION ---
try:
    import essentia
    # SILENCE ESSENTIA BEFORE IMPORTING STANDARDS
    # This ensures that even during sub-process imports, the INFO flags are False
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Meloday Playlist Generator")
parser.add_argument('--debug', action='store_true', help="Enable verbose debug logging")
args, unknown = parser.parse_known_args()

DEBUG_MODE = args.debug

# --- LOGGING INITIALIZATION ---
LOG_FILE = os.path.join(BASE_DIR, "logs", "meloday_run.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Use 'a' (append) mode to prevent sub-processes from truncating the file on import
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

logger = logging.getLogger("Meloday")
logger.setLevel(logging.INFO)
# Clear existing handlers to prevent duplicates
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(file_handler)

def log_text(msg):
    """
    Strictly filters string to ensure no NUL bytes or non-printable 
    control characters trigger grep's binary detection, and flushes to disk.
    """
    if msg:
        # Keep only printable characters and standard whitespace
        clean_msg = "".join(ch for ch in str(msg) if ch.isprintable() or ch in ("\n", "\r", "\t"))
        # Remove NUL bytes which trigger binary detection in grep
        clean_msg = clean_msg.replace('\x00', '')
        logger.info(clean_msg)
        # Force the OS to write to disk immediately (integrated into all selection/filtering loops)
        file_handler.flush()

# Wrapper to ensure every print is also logged
def m_print(msg):
    print(msg)
    log_text(msg)

def d_print(msg):
    """Prints and logs only if DEBUG_MODE is True."""
    if DEBUG_MODE:
        # We use a prefix to make these easy to filter in the log file
        debug_msg = f"[DEBUG] {msg}"
        print(debug_msg)
        log_text(debug_msg)

def resolve_path(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)

def load_config(filepath="config.yml"):
    with open(os.path.join(BASE_DIR, filepath), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# --- 2. GLOBAL CONFIGURATION (Mapped from config.yml) ---
config = load_config()

# Plex Connection
PLEX_URL = config["plex"]["url"]
PLEX_TOKEN = config["plex"]["token"]
MUSIC_LIBRARY = config["plex"]["music_library"]
CHRISTMAS_COLLECTION_NAME = config["plex"]["christmas_collection"]
EXCLUDE_LABEL_NAME = config["plex"]["exclude_label"]

# Playlist & Logic Rules
EXCLUDE_PLAYED_DAYS = config["playlist"]["exclude_played_days"]
HISTORY_LOOKBACK_DAYS = config["playlist"]["history_lookback_days"]
MAX_TRACKS = config["playlist"]["max_tracks"]
SONIC_SIMILAR_LIMIT = MAX_TRACKS   # Sorting breadth always tracks playlist size; not user-configurable
HISTORICAL_RATIO = config["playlist"].get("historical_ratio", 0.3)
GENRE_RATIO = config["playlist"].get("genre_ratio", 0.15)
STYLE_RATIO = config["playlist"].get("style_ratio", 0.20)
STYLE_TAG_DEPTH = config["playlist"].get("style_tag_depth", 1)  # How many style slots to check per track; set by optimizer
ARTIST_RATIO = config["playlist"].get("artist_ratio", 0.05)
MOOD_RATIO = config["playlist"].get("mood_ratio", 0.35)
SONIC_SIMILARITY_SEARCH_LIMIT = max(config["playlist"].get("sonic_similarity_limit", 100), MAX_TRACKS * 2)
SONIC_SIMILARITY_DISTANCE = config["playlist"].get("sonic_similarity_distance", 0.25)

# Essentia Logic & Weights
ess_cfg = config.get("essentia", {})
ESSENTIA_ENABLED = ess_cfg.get("enabled", True) and ESSENTIA_AVAILABLE
ESSENTIA_CACHE_PATH = resolve_path(ess_cfg.get("cache_path", "assets/essentia_cache.db"), BASE_DIR)
# Auto-upgrade: if config still references the old .json path, silently redirect to .db
if ESSENTIA_CACHE_PATH.endswith('.json'):
    ESSENTIA_CACHE_PATH = ESSENTIA_CACHE_PATH[:-5] + '.db'
BPM_WEIGHT = ess_cfg.get("bpm_weight", 0.15)
KEY_WEIGHT = ess_cfg.get("key_weight", 0.10)
ENERGY_WEIGHT = ess_cfg.get("energy_weight", 0.10)
ERA_WEIGHT = ess_cfg.get("era_weight", 0.05)
PATH_MAPPING = ess_cfg.get("path_mapping", {})

# Bridging Configuration
BRIDGING_ENABLED = config.get("bridging", {}).get("enabled", True)
SMART_TRUNCATION_ENABLED = config.get("bridging", {}).get("smart_truncation", True)

# Seasonal Rules
xmas_cfg = config.get("seasonal", {}).get("christmas", {})
XMAS_START_MONTH = xmas_cfg.get("start_month", 12)
XMAS_START_DAY   = xmas_cfg.get("start_day", 1)
XMAS_END_MONTH   = xmas_cfg.get("end_month", 12)
XMAS_END_DAY     = xmas_cfg.get("end_day", 25)

# Asset Paths
COVER_IMAGE_DIR = resolve_path(config["directories"]["cover_images"], BASE_DIR)
FONTS_DIR       = resolve_path(config["directories"]["fonts"], BASE_DIR)
MOOD_MAP_PATH   = resolve_path(config["files"]["mood_map"], BASE_DIR)
FONT_MAIN_PATH   = resolve_path(config["fonts"]["main"], FONTS_DIR)
FONT_MELODAY_PATH = resolve_path(config["fonts"]["meloday"], FONTS_DIR)

# Musical Data Maps
CAMELOT_MAP = {
    'Ab minor': '1A', 'B major': '1B', 'Eb minor': '2A', 'F# major': '2B',
    'Bb minor': '3A', 'Db major': '3B', 'F minor': '4A', 'Ab major': '4B',
    'C minor': '5A', 'Eb major': '5B', 'G minor': '6A', 'Bb major': '6B',
    'D minor': '7A', 'F major': '7B', 'A minor': '8A', 'C major': '8B',
    'E minor': '9A', 'G major': '9B', 'B minor': '10A', 'D major': '10B',
    'F# minor': '11A', 'A major': '11B', 'Db minor': '12A', 'E major': '12B'
}

# Daypart Logic
PERIOD_PHRASES = config["period_phrases"]
time_periods = config["time_periods"]

# --- 3. SYSTEM INITIALIZATION ---
_essentia_cache = {}
_album_meta_cache = {}
_album_obj_cache = {}
_artist_obj_cache = {}
_metadata_tag_cache = {}   # {('album'|'artist', ratingKey, attr): [tag_strings]}
_global_sonic_cache = {}
_christmas_album_keys = set()
_excluded_album_keys = set()
plex = None

# --- 4. CORE FUNCTIONS ---

@functools.lru_cache(maxsize=None)
def get_optimal_workers(task_type="cpu"):
    try:
        logical = os.cpu_count() or 1

        if task_type == "cpu":
            # Use physical cores for CPU-bound tasks — hyperthreading gives little benefit
            # for compute-intensive work like Essentia audio analysis.
            try:
                import psutil
                physical = psutil.cpu_count(logical=False) or logical
                source = "psutil"
            except ImportError:
                # Approximate: assume 2 threads per core (x86 hyperthreading).
                # Slightly conservative for non-HT chips (ARM NAS, some AMD), but always safe.
                physical = max(1, logical // 2) if logical > 2 else logical
                source = "estimated"
            # Use all physical cores — the main process is idle (blocked on executor.map/as_completed)
            # while workers run, so there is nothing to reserve a core for.
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
            assigned = max(1, (logical) // 2)

        log_text(f"[WORKER CONFIG] Mode: {task_type.upper()} | {tier_reason}")
        log_text(f"                Threads Detected: {logical} -> Assigned Workers: {assigned}")

        return assigned

    except Exception as e:
        log_text(f"[WORKER CONFIG] ERROR: {e}. Defaulting to safe fallback (2 workers).")
        return 2

def validate_environment():
    """Checks for configuration errors and connectivity before the script runs."""
    m_print("--- Pre-flight Environment Check ---")
    errors = []

    # 1. Check Plex Connection & Library
    try:
        # Re-use global plex instance to verify connection
        music_section = plex.library.section(MUSIC_LIBRARY)
        m_print(f"[OK] Connected to Plex: {plex.friendlyName}")
    except Exception as e:
        errors.append(f"Plex Connection/Library Error: {e}")

    # 2. Check Directories & Fonts
    if not os.path.isdir(COVER_IMAGE_DIR):
        errors.append(f"Cover directory not found: {COVER_IMAGE_DIR}")
    if not os.path.exists(FONT_MAIN_PATH):
        errors.append(f"Main font file missing: {FONT_MAIN_PATH}")
    if not os.path.exists(FONT_MELODAY_PATH):
        errors.append(f"Branding font file missing: {FONT_MELODAY_PATH}")

    # 3. Check Essentia Path
    if ESSENTIA_ENABLED:
        cache_dir = os.path.dirname(ESSENTIA_CACHE_PATH)
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Essentia cache directory cannot be created: {e}")

    # 4. Check Seasonal Logic
    if XMAS_START_MONTH > 12 or XMAS_END_MONTH > 12:
        errors.append("Invalid seasonal month in config.yml (Must be 1-12)")

    if errors:
        m_print("[CRITICAL ERROR] Environment validation failed:")
        for err in errors:
            m_print(f"  - {err}")
        return False

    m_print("[OK] Environment validated. Proceeding to generation.")
    return True

def prefetch_seasonal_exclusions():
    """Populates a set of album ratingKeys that belong to the Christmas collection."""
    global _christmas_album_keys
    if not _in_christmas_window(datetime.now()):
        try:
            music_library = plex.library.section(MUSIC_LIBRARY)
            # Fetch the collection object
            collections = music_library.collections(title=CHRISTMAS_COLLECTION_NAME)
            if collections:
                # Get all album ratingKeys in this collection
                _christmas_album_keys = {str(album.ratingKey) for album in collections[0].items()}
                m_print(f"[OK] Pre-fetched {len(_christmas_album_keys)} Christmas albums for exclusion.")
        except Exception as e:
            m_print(f"[WARN] Failed to pre-fetch seasonal exclusions: {e}")

def prefetch_label_exclusions():
    """Fetches all album IDs that have the designated exclusion label."""
    global _excluded_album_keys
    try:
        music_library = plex.library.section(MUSIC_LIBRARY)
        # One bulk API call to find everything with the exclusion label
        excluded_albums = music_library.search(libtype='album', label=EXCLUDE_LABEL_NAME)
        _excluded_album_keys = {str(a.ratingKey) for a in excluded_albums}
        m_print(f"[OK] Pre-fetched {len(_excluded_album_keys)} albums with '{EXCLUDE_LABEL_NAME}' for exclusion.")
    except Exception as e:
        m_print(f"[WARN] Failed pre-fetching label exclusions: {e}")

# --- SQLite cache helpers ---

_UPSERT_SQL = """
    INSERT OR REPLACE INTO essentia_cache
    (rating_key, bpm, key, energy, year, artist, genres, styles, moods, file_path, last_synced)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

def _ensure_db_schema(conn):
    """Creates the essentia_cache table and sets optimal WAL pragmas."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS essentia_cache (
            rating_key TEXT PRIMARY KEY NOT NULL,
            bpm REAL, key TEXT, energy REAL, year INTEGER,
            artist TEXT, genres TEXT, styles TEXT, moods TEXT,
            file_path TEXT, last_synced REAL
        )
    """)
    # Migrate existing databases that pre-date the styles column
    try:
        conn.execute("ALTER TABLE essentia_cache ADD COLUMN styles TEXT")
    except Exception:
        pass  # Column already exists
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

def _entry_to_row(rk, d):
    return (rk, d.get("bpm"), d.get("key"), d.get("energy"), d.get("year"),
            d.get("artist"), json.dumps(d.get("genres") or []),
            json.dumps(d.get("styles") or []),
            json.dumps(d.get("moods") or []), d.get("file_path"), d.get("last_synced"))

def _row_to_entry(row):
    rk, bpm, key, energy, year, artist, genres_j, styles_j, moods_j, file_path, last_synced = row
    return rk, {
        "bpm": bpm, "key": key, "energy": energy, "year": year, "artist": artist,
        "genres": json.loads(genres_j) if genres_j else [],
        "styles": json.loads(styles_j) if styles_j else [],
        "moods": json.loads(moods_j) if moods_j else [],
        "file_path": file_path, "last_synced": last_synced
    }

def _migrate_json_to_sqlite():
    """One-time migration: imports the legacy JSON cache into the SQLite database."""
    json_path = ESSENTIA_CACHE_PATH[:-3] + '.json'
    if not os.path.exists(json_path):
        return
    m_print("[INFO] Migrating legacy JSON cache to SQLite — this runs once.")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=30)
        _ensure_db_schema(conn)
        conn.executemany(_UPSERT_SQL, [_entry_to_row(rk, d) for rk, d in old_data.items() if isinstance(d, dict)])
        conn.commit()
        conn.close()
        os.rename(json_path, json_path + ".bak")
        m_print(f"[INFO] Migration complete. {len(old_data)} entries moved. Old file renamed to .json.bak")
    except Exception as e:
        m_print(f"[WARN] JSON migration failed: {e}")

def load_essentia_cache():
    global _essentia_cache
    _migrate_json_to_sqlite()
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        _essentia_cache = {}
        return
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=10)
        _ensure_db_schema(conn)
        rows = conn.execute(
            "SELECT rating_key, bpm, key, energy, year, artist, genres, styles, moods, file_path, last_synced "
            "FROM essentia_cache"
        ).fetchall()
        conn.close()
        _essentia_cache = {rk: entry for rk, entry in (_row_to_entry(r) for r in rows)}
    except Exception as e:
        m_print(f"[WARN] Could not load Essentia cache: {e}")
        _essentia_cache = {}

def save_essentia_cache():
    os.makedirs(os.path.dirname(ESSENTIA_CACHE_PATH), exist_ok=True)
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60)
        _ensure_db_schema(conn)
        conn.executemany(_UPSERT_SQL, [_entry_to_row(rk, d) for rk, d in _essentia_cache.items()])
        conn.commit()
        conn.close()
    except Exception as e:
        m_print(f"[ERROR] Could not save Essentia cache safely: {e}")

def _upsert_cache_entries(entries):
    """Writes only the specified {rk: data} subset to SQLite, without touching the rest of the cache."""
    if not entries:
        return
    os.makedirs(os.path.dirname(ESSENTIA_CACHE_PATH), exist_ok=True)
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=60)
        _ensure_db_schema(conn)
        conn.executemany(_UPSERT_SQL, [_entry_to_row(rk, d) for rk, d in entries.items()])
        conn.commit()
        conn.close()
    except Exception as e:
        m_print(f"[ERROR] Could not save Essentia cache entries: {e}")

def get_local_path(track):
    if not track.locations: return None
    path = track.locations[0]
    for remote, local in PATH_MAPPING.items():
        if path.startswith(remote):
            path = path.replace(remote, local, 1)
            break
            
    if os.path.exists(path):
        return path
    else:
        m_print(f"[DIAGNOSTIC] File not found: {path}")
        return None

def analyze_track_essentia(track):
    rk = str(track.ratingKey)
    now_ts = datetime.now().timestamp()

    # Re-silence inside sub-processes to handle parallel worker logs
    if ESSENTIA_AVAILABLE:
        essentia.log.infoActive = False
        essentia.log.warningActive = False
    
    # Metadata Update Handling
    # If the track is already in the cache, we refresh the text fields from Plex
    # to ensure changes to genres/moods/artist names are captured.
    if rk in _essentia_cache and "energy" in _essentia_cache[rk]:
        data = _essentia_cache[rk]
        
        # Throttled Metadata Syncing
        # Only call Plex to update text metadata if the entry is older than 7 days (604800s).
        last_sync = data.get("last_synced", 0)
        file_path = get_local_path(track)
        
        # Check if the file path has changed (e.g., quality upgrade) while keeping same ratingKey
        if data.get("file_path") == file_path and (now_ts - last_sync) < 604800:
            # Backfill any tags missing or empty due to older cache entries.
            # "styles" not in data won't catch entries where styles=[] (loaded from NULL),
            # so check for falsy instead. Same applies to genres — older code used
            # track.genres directly without the album/artist fallback.
            backfilled = False
            if not data.get("styles"):
                data["styles"] = _resolve_tags(track, "styles")
                backfilled = True
            if not data.get("genres"):
                data["genres"] = _resolve_tags(track, "genres")
                backfilled = True
            if backfilled:
                data["last_synced"] = now_ts
            return data

        data["artist"] = norm_text(primary_artist(track_artist_name(track)))
        data["genres"] = _resolve_tags(track, "genres")
        data["styles"] = _resolve_tags(track, "styles")
        data["moods"]  = _resolve_tags(track, "moods")
        # Update year if it was previously null
        if data.get("year") is None:
            data["year"] = getattr(track, "year", None) or album_meta(track).get("year")

        # If path is still the same, we just update text metadata and return
        if data.get("file_path") == file_path:
            data["last_synced"] = now_ts
            return data
        # If path changed, we fall through to perform new acoustic analysis

    if not ESSENTIA_ENABLED: return None
    
    file_path = get_local_path(track)
    if not file_path: return None

    try:
        loader = es.MonoLoader(filename=file_path)
        audio = loader()

        # BPM & Key Extraction
        bpm = es.RhythmExtractor2013(method="multifeature")(audio)[0]
        key_alg = es.KeyExtractor()(audio)
        camelot = CAMELOT_MAP.get(f"{key_alg[0]} {key_alg[1]}", "0A")

        # Energy/Intensity via Integrated Loudness (EBU R128)
        # Mux mono signal to pseudo-stereo for loudness analyzer compatibility
        audio_stereo = es.StereoMuxer()(audio, audio)
        loudness_stats = es.LoudnessEBUR128()(audio_stereo)
        integrated_loudness = loudness_stats[2]

        # Year Fallback: Check track first, then album
        track_year = getattr(track, "year", None)
        if not track_year:
            track_year = album_meta(track).get("year")

        # Comprehensive Metadata Caching
        data = {
            "bpm": round(bpm, 2),
            "key": camelot,
            "energy": round(integrated_loudness, 2),
            "year": track_year,
            "artist": norm_text(primary_artist(track_artist_name(track))),
            "genres": _resolve_tags(track, "genres"),
            "styles": _resolve_tags(track, "styles"),
            "moods":  _resolve_tags(track, "moods"),
            "file_path": file_path,
            "last_synced": now_ts
        }
        _essentia_cache[rk] = data
        return data
    except Exception as e:
        m_print(f"[DIAGNOSTIC] Essentia failed for '{track.title}': {e}")
        return None

# Worker for Multiprocessing compatibility
def analysis_worker(track_id):
    try:
        # Silence Essentia explicitly in parallel worker processes
        if ESSENTIA_AVAILABLE:
            essentia.log.infoActive = False
            essentia.log.warningActive = False

        # Process-Safe Plex Connection
        # Initialize a new local connection session for process isolation.
        local_plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=60)
        track = local_plex.fetchItem(track_id)
        return str(track_id), analyze_track_essentia(track)
    except Exception:
        return str(track_id), None

def get_bpm_distance(bpm1, bpm2):
    if not bpm1 or not bpm2: return 0.5
    diffs = [abs(bpm1 - bpm2), abs(bpm1 - bpm2 * 2), abs(bpm1 * 2 - bpm2)]
    return min(min(diffs) / 20.0, 1.0)

def get_harmonic_distance(key1, key2):
    if key1 == "0A" or key2 == "0A": return 0.5
    idx1, type1 = int(key1[:-1]), key1[-1]
    idx2, type2 = int(key2[:-1]), key2[-1]
    dist = abs(idx1 - idx2)
    if dist > 6: dist = 12 - dist
    return (dist + (0 if type1 == type2 else 1)) / 7.0
    
def get_period_phrase(period):
    return PERIOD_PHRASES.get(period, f"in the {period}")

def _in_christmas_window(now: datetime) -> bool:
    """True if date is within the configured window in the server's local time."""
    # Create date objects for the current year to compare
    try:
        start_date = datetime(now.year, XMAS_START_MONTH, XMAS_START_DAY)
        end_date = datetime(now.year, XMAS_END_MONTH, XMAS_END_DAY)
        
        # Handle windows that cross into the next year (e.g., Dec 1 to Jan 5)
        if start_date > end_date:
            return now >= start_date or now <= end_date
            
        return start_date <= now <= end_date
    except ValueError:
        # Fallback to default behavior if config dates are invalid
        return (now.month == 12) and (1 <= now.day <= 25)

def _tag_list_contains(tags, needle: str) -> bool:
    """True if a Plex tag list contains a tag equal to needle (case-insensitive)."""
    if not tags:
        return False
    n = needle.strip().casefold()
    for t in tags:
        # Handle both Plex objects and lean cached strings
        val = t if isinstance(t, str) else (getattr(t, "tag", None) or getattr(t, "title", None))
        if isinstance(val, str) and val.strip().casefold() == n:
            return True
    return False

def has_label(obj, label_name: str) -> bool:
    """True if Plex item has a Labels tag equal to label_name (case-insensitive)."""
    try:
        # Handle both Plex objects and lean cached dicts
        labels = obj.get("labels") if isinstance(obj, dict) else getattr(obj, "labels", None)
        return _tag_list_contains(labels, label_name)
    except Exception as e:
        m_print(f"[WARN] Error checking label for object: {e}")
        return False

def _album_in_collection(album, collection_name: str) -> bool:
    """True if an Album is in the given Plex Collection name."""
    try:
        # collections are sometimes not populated until reload()
        # skip reload if we already have the lean dict
        if not isinstance(album, dict):
            try:
                album.reload()
            except Exception as e:
                m_print(f"[WARN] Error reloading album {album.title}: {e}")
        
        tags = album.get("collections") if isinstance(album, dict) else getattr(album, "collections", None)
        return _tag_list_contains(tags, collection_name)
    except Exception as e:
        m_print(f"[WARN] Error checking collection for album: {e}")
        return False

def filter_excluded_tracks(tracks, now=None):
    """Apply pre-fetched Christmas and 'noshare' exclusions."""
    if not tracks:
        return []
    now = now or datetime.now()
    in_xmas = _in_christmas_window(now)

    cleaned = []
    for t in tracks:
        parent_key = str(getattr(t, "parentRatingKey", ""))
        
        # Check pre-fetched Christmas set (Memory check)
        if not in_xmas and parent_key in _christmas_album_keys:
            log_text(f"[EXCLUSION] Track '{t.title}' skipped: Seasonal filtering.")
            continue

        # Check pre-fetched 'noshare' set (Memory check)
        if parent_key in _excluded_album_keys:
            log_text(f"[EXCLUSION] Track '{t.title}' skipped: Album has '{EXCLUDE_LABEL_NAME}' label.")
            continue

        cleaned.append(t)
    return cleaned

def remix_album_penalty(track) -> int:
    """Lower is better. Penalize remix releases (EPs/singles/albums titled 'Remix/Remixes')."""
    meta = album_meta(track)
    title = (meta.get("album_title") or "").casefold()
    subtype = (meta.get("album_subtype") or "").casefold()

    # Title-based detection catches: "Go Bang (Remixes) - EP", "Remixes", etc.
    if "remix" in title or "remixes" in title:
        return 1

    # If Plex subtype ever reports Remix, also treat it as a remix release.
    if "remix" in subtype:
        return 1

    return 0


@functools.lru_cache(maxsize=None)
def norm_text(s: str) -> str:
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s

def track_artist_name(track) -> str:
    """Best-effort track artist name.

    Plex can represent compilation albums in a few ways:
      - Sometimes track.grandparentTitle is the real track artist (ideal).
      - Sometimes track.grandparentTitle is 'Various Artists', and the real
        track artist is stored in track.originalTitle.
    This function prefers a non-VA grandparentTitle, then falls back to
    originalTitle (if it looks like an artist), then track.artist().title.
    """
    gp = getattr(track, "grandparentTitle", None)
    if isinstance(gp, str) and gp.strip() and not ((gp or '').strip().casefold() in {'various artists','various'}):
        return gp.strip()

    # On compilations, Plex often stores the real track artist here.
    ot = getattr(track, "originalTitle", None)
    if isinstance(ot, str) and ot.strip():
        # Avoid using originalTitle if it is identical to the track title.
        ttitle = getattr(track, "title", None)
        if not (isinstance(ttitle, str) and ttitle.strip().casefold() == ot.strip().casefold()):
            if not ((ot or '').strip().casefold() in {'various artists','various'}):
                return ot.strip()

    # As a last resort, try plexapi's artist() accessor.
    try:
        # Check cache first to fix redundant API calls and fragmented caching
        artist_key = getattr(track, "grandparentRatingKey", None)
        if artist_key and artist_key in _artist_obj_cache:
            a = _artist_obj_cache[artist_key]
        else:
            artist_obj = track.artist() if callable(getattr(track, "artist", None)) else None
            if artist_key and artist_obj:
                # Store only what we need in a lean dict
                a = {
                    "title": getattr(artist_obj, "title", None),
                    "userRating": getattr(artist_obj, "userRating", None)
                }
                _artist_obj_cache[artist_key] = a
            else:
                a = None
        
        at = a.get("title") if isinstance(a, dict) else (getattr(a, "title", None) if a else None)
        if isinstance(at, str) and at.strip() and not ((at or '').strip().casefold() in {'various artists','various'}):
            return at.strip()
    except Exception as e:
        m_print(f"[WARN] Error resolving artist for track {track.title}: {e}")

    # If we still can't tell, return whatever grandparentTitle we had (even if VA), or 'unknown'.
    if isinstance(gp, str) and gp.strip():
        return gp.strip()
    return "unknown"



# --- Dedup helpers: prefer studio albums over compilations/soundtracks ---
_FEAT_SPLIT_RE = re.compile(r"\s*(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)

@functools.lru_cache(maxsize=None)
def primary_artist(name: str) -> str:
    """Return the primary artist portion (strip 'feat./ft./featuring ...')."""
    if not name:
        return ""
    s = name.strip()
    s = _FEAT_SPLIT_RE.sub("", s)
    # Normalize whitespace and case for comparisons
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_various_artists(name: str) -> bool:
    return (name or "").strip().casefold() in {"various artists", "various"}

# Cache album metadata lookups so we don't spam Plex.
_album_meta_cache: dict[str, dict] = {}

# Unified caches for Plex objects to fix redundant API calls and fragmented caching
_album_obj_cache = {}
_artist_obj_cache = {}

def album_meta(track) -> dict:
    """Fetch album metadata (title, album-artist, subtype) with caching."""
    album_key = getattr(track, "parentRatingKey", None) or getattr(track, "parentKey", None) or getattr(track, "parentGuid", None)
    cache_key = str(album_key) if album_key is not None else str(getattr(track, "ratingKey", ""))
    if cache_key in _album_meta_cache:
        return _album_meta_cache[cache_key]

    meta = {
        "album_title": (getattr(track, "parentTitle", "") or "").strip(),
        "album_artist": "",
        "album_subtype": "",
        "year": None,
    }
    try:
        # Check lean cache first to fix redundant API calls
        if album_key and album_key in _album_obj_cache and isinstance(_album_obj_cache[album_key], dict):
            album_data = _album_obj_cache[album_key]
            meta["album_title"] = album_data.get("title", meta["album_title"])
            meta["album_artist"] = album_data.get("parentTitle", "")
            meta["year"] = album_data.get("year")
            meta["album_subtype"] = album_data.get("subtype", "")
        else:
            album = track.album() if callable(getattr(track, "album", None)) else None
            if album is not None:
                meta["album_title"] = (getattr(album, "title", meta["album_title"]) or meta["album_title"]).strip()
                meta["album_artist"] = (getattr(album, "parentTitle", "") or "").strip()
                meta["year"] = getattr(album, "year", None)
                # Plex may expose subtype/albumType differently depending on server/version.
                meta["album_subtype"] = (getattr(album, "subtype", "") or getattr(album, "albumType", "") or "").strip()
                # Fallback: query raw metadata XML and extract <Subformat tag="...">
                if not meta["album_subtype"]:
                    try:
                        # Process-Safe raw query logic
                        # Using track._server instead of the global plex instance for process isolation.
                        data = track._server.query(getattr(album, "key", f"/library/metadata/{album.ratingKey}"))
                        sub = data.find(".//Subformat")
                        if sub is not None:
                            tag = sub.get("tag", "") or ""
                            meta["album_subtype"] = tag.strip()
                    except Exception as e:
                        m_print(f"[WARN] Failed to query raw XML for album subtype ({track.title}): {e}")
                
                # Store only what we need in a lean dict for both caches
                if album_key:
                    _album_obj_cache[album_key] = {
                        "title": meta["album_title"],
                        "parentTitle": meta["album_artist"],
                        "year": meta["year"],
                        "subtype": meta["album_subtype"],
                        "userRating": getattr(album, "userRating", None),
                        "labels": [l.tag for l in getattr(album, "labels", [])],
                        "collections": [c.tag for c in getattr(album, "collections", [])],
                        "genres": [str(g) for g in getattr(album, "genres", [])],
                        "styles": [str(s) for s in getattr(album, "styles", [])],
                        "moods":  [str(m) for m in getattr(album, "moods",  [])],
                    }
    except Exception as e:
        m_print(f"[WARN] Error fetching metadata for {track.title}: {e}")

    _album_meta_cache[cache_key] = meta
    return meta

def _resolve_tags(track, attr):
    """Return tag list for attr with track → album → artist fallback, caching each level."""
    # 1. Track level — cheapest, no API call
    tags = [str(t) for t in (getattr(track, attr, None) or [])]
    if tags:
        return tags

    # 2. Album level — album_meta() is cached; only fetches once per album per run
    album_key = getattr(track, "parentRatingKey", None)
    if album_key:
        album_cache_key = ("album", album_key, attr)
        if album_cache_key not in _metadata_tag_cache:
            try:
                # album_meta() populates _album_obj_cache with genres/styles/moods
                album_meta(track)
                _metadata_tag_cache[album_cache_key] = _album_obj_cache.get(album_key, {}).get(attr) or []
            except Exception:
                _metadata_tag_cache[album_cache_key] = []
        tags = _metadata_tag_cache[album_cache_key]
        if tags:
            return tags

    # 3. Artist level — fetched and cached on first miss per artist
    artist_key = getattr(track, "grandparentRatingKey", None)
    if artist_key:
        artist_cache_key = ("artist", artist_key, attr)
        if artist_cache_key not in _metadata_tag_cache:
            try:
                artist = track.artist() if callable(getattr(track, "artist", None)) else None
                _metadata_tag_cache[artist_cache_key] = [str(t) for t in (getattr(artist, attr, None) or [])] if artist else []
            except Exception:
                _metadata_tag_cache[artist_cache_key] = []
        tags = _metadata_tag_cache[artist_cache_key]
        if tags:
            return tags

    return []

_COMPILATION_TITLE_RE = re.compile(
    r"\b("
    r"soundtrack|ost|o\.s\.t\.|"
    r"original\s+(?:motion\s+picture\s+)?soundtrack|"
    r"motion\s+picture\s+soundtrack|"
    r"music\s+from\s+the\s+(?:motion\s+picture|film)|"
    r"various\s+artists|"
    r"greatest\s+hits|best\s+of|"
    r"anthology|compilation|"
    r"triple\s*j"
    r")\b",
    re.IGNORECASE,
)
_LIVE_TITLE_RE = re.compile(r"\blive\b|unplugged|concert", re.IGNORECASE)

# Pre-compiled constants for clean_title — built once at import, not on every call
_VERSION_KEYWORDS_SORTED = sorted([
    "extended", "deluxe", "remaster", "remastered", "live", "acoustic", "edit",
    "version", "anniversary", "special edition", "radio edit", "album version",
    "original mix", "remix", "mix", "dub", "instrumental", "karaoke", "cover",
    "rework", "re-edit", "bootleg", "vip", "session", "alternate", "take",
    "mix cut", "cut", "dj mix"
], key=len, reverse=True)
_FEATURING_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"\(feat\.?.*?\)", r"\[feat\.?.*?\]", r"\(ft\.?.*?\)", r"\[ft\.?.*?\]",
    r"\bfeat\.?\s+\w+", r"\bfeaturing\s+\w+", r"\bft\.?\s+\w+",
    r" - .*mix$", r" - .*dub$", r" - .*remix$", r" - .*edit$", r" - .*version$",
]]
_KW_ALT = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in _VERSION_KEYWORDS_SORTED)
_PAREN_KW_RE   = re.compile(rf"\(\s*[^)]*(?:{_KW_ALT})[^)]*\)\s*", re.IGNORECASE)
_BRACKET_KW_RE = re.compile(rf"\[\s*[^\]]*(?:{_KW_ALT})[^\]]*\]\s*", re.IGNORECASE)
_KW_RES        = [re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE) for k in _VERSION_KEYWORDS_SORTED]
_EMPTY_PAREN_RE   = re.compile(r"\(\s*\)")
_EMPTY_BRACKET_RE = re.compile(r"\[\s*\]")
_TRAIL_DASH_RE    = re.compile(r"[\s-]+$")

def is_studio_album(track) -> bool:
    """Best-effort: treat compilations/soundtracks/live as non-studio; everything else as studio."""
    meta = album_meta(track)
    subtype = (meta.get("album_subtype") or "").casefold()
    if subtype:
        # These subtype strings vary, so check broadly.
        if any(x in subtype for x in ("compilation", "soundtrack")):
            return False
        if any(x in subtype for x in ("live", "ep", "single", "remix")):
            return False
        if "album" in subtype or "studio" in subtype:
            return True

    title = meta.get("album_title", "") or (getattr(track, "parentTitle", "") or "")
    if _COMPILATION_TITLE_RE.search(title):
        return False
    if _LIVE_TITLE_RE.search(title):
        return False

    # If we can't tell, assume it's a studio album.
    return True

def is_compilation_like(track) -> bool:
    meta = album_meta(track)
    subtype = (meta.get("album_subtype") or "").casefold()
    if any(x in subtype for x in ("compilation", "soundtrack")):
        return True
    title = meta.get("album_title", "") or (getattr(track, "parentTitle", "") or "")
    return bool(_COMPILATION_TITLE_RE.search(title))

def is_live_like(track) -> bool:
    meta = album_meta(track)
    subtype = (meta.get("album_subtype") or "").casefold()
    if "live" in subtype:
        return True
    title = meta.get("album_title", "") or (getattr(track, "parentTitle", "") or "")
    return bool(_LIVE_TITLE_RE.search(title))

def title_variant_rank(track) -> int:
    """Lower is better. Prefer plain/original titles when deduping."""
    raw = (getattr(track, "title", "") or "").strip().casefold()
    cleaned = clean_title(getattr(track, "title", "") or "").strip().casefold()

    # Best: already the base title (no version/remix tag removed)
    if raw == cleaned:
        return 0

    # Next best: explicitly "original mix"/"album version" type tags
    if re.search(r"\b(original\s+mix|album\s+version|single\s+version)\b", raw):
        return 1

    # Otherwise: remix/edit/live/etc variants
    return 2

def better_copy(a, b):
    """Choose which duplicate track entry to keep."""
    # 0) Prefer vastly sonically superior matches (difference > 0.08)
    # Historical tracks have an assumed distance of 0.00
    a_dist = getattr(a, "sonic_distance", 0.00)
    b_dist = getattr(b, "sonic_distance", 0.00)
    if abs(a_dist - b_dist) > 0.08:
        winner = a if a_dist < b_dist else b
        d_print(f"[DEDUPE] Sonic Priority: Kept '{winner.title}' ({winner.ratingKey}) due to distance ({min(a_dist, b_dist):.3f}).")
        return winner

    # 1) Prefer studio albums
    a_studio = is_studio_album(a)
    b_studio = is_studio_album(b)
    if a_studio != b_studio:
        winner = a if a_studio else b
        d_print(f"[DEDUPE] Format Priority: Kept '{winner.title}' (Studio Album preferred).")
        return winner

    # 2) Prefer the "plain/original" title within the same dedupe key
    a_rank = title_variant_rank(a)
    b_rank = title_variant_rank(b)
    if a_rank != b_rank:
        winner = a if a_rank < b_rank else b
        d_print(f"[DEDUPE] Variant Priority: Kept '{winner.title}' (Original title preferred over edit/remix).")
        return winner

    # 3) Prefer non-remix album titles (e.g., 'Changa' over 'Go Bang (Remixes) - EP')
    a_pen = remix_album_penalty(a)
    b_pen = remix_album_penalty(b)
    if a_pen != b_pen:
        winner = a if a_pen < b_pen else b
        d_print(f"[DEDUPE] Album Penalty Priority: Kept '{winner.title}' (Non-remix collection preferred).")
        return winner

    # Pre-fetch meta once
    a_meta = album_meta(a)
    b_meta = album_meta(b)

    # 4) Prefer compilation/soundtrack over live (when both are non-studio)
    a_comp = is_compilation_like(a)
    b_comp = is_compilation_like(b)
    a_live = is_live_like(a)
    b_live = is_live_like(b)

    # Explicit: compilation-like beats live-like if that's the head-to-head
    if a_comp and b_live and not b_comp and not a_live:
        return a
    if b_comp and a_live and not a_comp and not b_live:
        return b

    # Otherwise prefer non-live
    if a_live != b_live:
        return a if not a_live else b

    # 5) Prefer copies where album-artist matches the track primary artist
    a_track_artist = primary_artist(track_artist_name(a)).casefold()
    b_track_artist = primary_artist(track_artist_name(b)).casefold()

    a_album_artist = primary_artist(a_meta.get("album_artist", "")).casefold()
    b_album_artist = primary_artist(b_meta.get("album_artist", "")).casefold()

    a_match = bool(a_album_artist) and a_album_artist == a_track_artist
    b_match = bool(b_album_artist) and b_album_artist == b_track_artist
    if a_match != b_match:
        return a if a_match else b

    # 6) Prefer non-Various Artists albums
    a_va = is_various_artists(a_album_artist)
    b_va = is_various_artists(b_album_artist)
    if a_va != b_va:
        return b if a_va else a

    # 7) Prefer higher user rating if present
    a_rating = getattr(a, "userRating", None)
    b_rating = getattr(b, "userRating", None)
    if isinstance(a_rating, (int, float)) and isinstance(b_rating, (int, float)) and a_rating != b_rating:
        return a if a_rating > b_rating else b
    if isinstance(a_rating, (int, float)) and not isinstance(b_rating, (int, float)):
        return a
    if isinstance(b_rating, (int, float)) and not isinstance(a_rating, (int, float)):
        return b

    return a



# ---------------------------------------------------------------------
# HELPER: Print a simple progress bar (0-100%) with a message
def print_status(percent, message):
    """Print a progress bar with the given percentage and a status message."""
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    m_print(f"[{bar}] {percent:3d}%  {message}")

# ---------------------------------------------------------------------
def get_current_time_period():
    """Determine which daypart the current hour belongs to."""
    current_hour = datetime.now().hour

    for period, details in time_periods.items():
        if current_hour in details["hours"]:
            return period

    # Fallback if not found
    return "Late Night"

def load_descriptor_map(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        m_print(f"Error loading descriptor dictionary: {e}")
        return {}

def wrap_text(text, font, draw, max_width):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

# ---------------------------------------------------------------------
def fetch_historical_tracks(period):
    """Fetch tracks from Plex history that match the current daypart while excluding recently played tracks."""
    music_section = plex.library.section(MUSIC_LIBRARY)
    now = datetime.now()
    period_hours = set(time_periods[period]["hours"])

    history_start = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    log_text(f"[SELECTION] Scanning history for period '{period}' (Lookback: {HISTORY_LOOKBACK_DAYS} days).")

    all_history = list(music_section.history(mindate=history_start))

    # Single pass: build excluded_keys and filtered_entries simultaneously.
    # An entry that is recent (>= exclude_start) goes into excluded_keys only;
    # an entry that is in the right period but not recent goes into filtered_entries.
    excluded_keys = set()
    filtered_entries = []
    for entry in all_history:
        if not entry.viewedAt:
            continue
        if entry.viewedAt >= exclude_start:
            excluded_keys.add(entry.ratingKey)
        elif entry.viewedAt.hour in period_hours:
            filtered_entries.append(entry)

    # If no historical tracks found, try adjacent periods before falling back to all history
    if not filtered_entries:
        all_periods = list(time_periods.keys())
        period_idx = all_periods.index(period)
        prev_idx = (period_idx - 1) % len(all_periods)
        next_idx = (period_idx + 1) % len(all_periods)
        adjacent_hours = (
            set(time_periods[all_periods[prev_idx]]["hours"]) |
            set(time_periods[all_periods[next_idx]]["hours"])
        )
        adjacent_entries = [
            entry for entry in all_history
            if entry.ratingKey not in excluded_keys
            and entry.viewedAt.hour in adjacent_hours
        ]
        if adjacent_entries:
            log_text("[SELECTION] No direct daypart matches. Using adjacent-period history fallback.")
            filtered_entries = adjacent_entries
        else:
            log_text("[SELECTION] No daypart/adjacent matches found. Attempting generic history fallback.")
            fallback_entries = [
                entry for entry in all_history
                if entry.ratingKey not in excluded_keys
            ]
            if fallback_entries:
                filtered_entries = fallback_entries

    # --- OPTIMIZED BULK METADATA RESOLUTION ---
    # 1. Identify unique ratingKeys to avoid redundant API hits
    unique_keys = list({entry.ratingKey for entry in filtered_entries if entry.ratingKey})
    
    # 2. Resolve metadata for unique tracks ONLY
    def resolve_unique_track(rk):
        try:
            t = plex.fetchItem(rk)
            if t and getattr(t, "type", None) == "track":
                t.sonic_distance = 0.0
                return t
        except Exception: 
            return None

    # This is now 4-5x faster because we resolve ~150 tracks instead of 695
    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        resolved_list = list(executor.map(resolve_unique_track, unique_keys))
    
    # 3. Create a map for lightning-fast reconstruction
    track_map = {t.ratingKey: t for t in resolved_list if t}

    # 4. Reconstruct the weighted history (preserving duplicates for popularity weighting)
    resolved_tracks = []
    for entry in filtered_entries:
        t = track_map.get(entry.ratingKey)
        if t:
            resolved_tracks.append(t)
    
    # Audit log for historical pool size
    log_text(f"[SELECTION] Found {len(resolved_tracks)} historical tracks after exclusion/lookback filtering.")

    # 5. Filter against labels, seasonal exclusions, and low ratings
    # (Uses the optimized set-based logic)
    filtered_tracks = filter_excluded_tracks(resolved_tracks, now=now)

    track_play_counts = Counter()
    for track in filtered_tracks:
        track_play_counts[track] += 1

    sorted_tracks = sorted(filtered_tracks, key=lambda t: track_play_counts[t], reverse=True)
    split_index = max(1, len(sorted_tracks) // 4)
    popular_tracks = sorted_tracks[:split_index]
    rare_tracks = sorted_tracks[split_index:]

    balanced_selection = (
        random.sample(rare_tracks, min(len(rare_tracks), int(MAX_TRACKS * (1 - HISTORICAL_RATIO))))
        + random.sample(popular_tracks, min(len(popular_tracks), int(MAX_TRACKS * HISTORICAL_RATIO)))
    )

    # Shuffle so rare and popular tracks compete fairly for style/genre slots.
    # Without this, rare tracks (always first in the list) would claim all quota slots
    # before popular tracks of the same style get a chance.
    random.shuffle(balanced_selection)

    # Single-pass mood/style/genre cap — mirrors process_tracks() exactly.
    # Mood is checked first (primary vibe consistency axis); style and genre follow.
    # Checks up to STYLE_TAG_DEPTH style slots per track; rejects only if any checked
    # style is full; counts against all checked styles when accepted.
    # Genre is the fallback for tracks with no style tags at all.
    max_mood_limit  = int(MAX_TRACKS * MOOD_RATIO)
    max_style_limit = int(MAX_TRACKS * STYLE_RATIO)
    max_genre_limit = int(MAX_TRACKS * GENRE_RATIO)
    running_mood  = Counter()
    running_style = Counter()
    running_genre = Counter()
    capped = []
    for track in balanced_selection:
        t_moods = _resolve_tags(track, "moods")
        if t_moods and running_mood[t_moods[0]] >= max_mood_limit:
            continue
        t_styles = _resolve_tags(track, "styles")[:STYLE_TAG_DEPTH]
        if t_styles:
            if any(running_style[s] >= max_style_limit for s in t_styles):
                continue
            for s in t_styles:
                running_style[s] += 1
        else:
            t_genres = _resolve_tags(track, "genres")
            if t_genres:
                pg = t_genres[0]
                if running_genre[pg] >= max_genre_limit:
                    continue
                running_genre[pg] += 1
        if t_moods:
            running_mood[t_moods[0]] += 1
        capped.append(track)
    if len(capped) < len(balanced_selection):
        log_text(f"[SELECTION] Style/genre balancing trimmed {len(balanced_selection) - len(capped)} tracks from history pool.")
    balanced_selection = capped

    return balanced_selection, excluded_keys

def filter_low_rated_tracks(tracks):
    """Filter out tracks/albums/artists with a 2-star rating (rating <= 4), skipping ephemeral tracks that lack ratingKey or parentRatingKey."""
    filtered = []
    for track in tracks:
        try:
            if not getattr(track, "ratingKey", None) or not getattr(track, "parentRatingKey", None):
                continue
            
            # Use lean caches to fix redundant API calls and fragmented caching
            artist_key = getattr(track, "grandparentRatingKey", None)
            if artist_key and artist_key in _artist_obj_cache:
                artist = _artist_obj_cache[artist_key]
            else:
                artist_obj = track.artist() if callable(getattr(track, "artist", None)) else None
                if artist_key and artist_obj:
                    # Store only what we need in a lean dict
                    artist = {
                        "title": getattr(artist_obj, "title", None),
                        "userRating": getattr(artist_obj, "userRating", None)
                    }
                    _artist_obj_cache[artist_key] = artist
                else:
                    artist = None
                    
            artist_rating = artist.get("userRating") if isinstance(artist, dict) else (getattr(artist, "userRating", None) if artist else None)
            
            album_key = track.parentRatingKey
            if album_key in _album_obj_cache:
                album = _album_obj_cache[album_key]
            else:
                # Use album_meta to populate the lean dict in _album_obj_cache
                album_meta(track)
                album = _album_obj_cache.get(album_key)
                
            album_rating = album.get("userRating") if isinstance(album, dict) else (getattr(album, "userRating", None) if album else None)
            track_rating = getattr(track, "userRating", None)

            if artist_rating is not None and artist_rating <= 4:
                d_print(f"[EXCLUSION] Track '{track.title}' skipped: Artist rating is low ({artist_rating}).")
                continue
            if album_rating is not None and album_rating <= 4:
                d_print(f"[EXCLUSION] Track '{track.title}' skipped: Album rating is low ({album_rating}).")
                continue
            if track_rating is not None and track_rating <= 4:
                d_print(f"[EXCLUSION] Track '{track.title}' skipped: User rating is low ({track_rating}).")
                continue

            filtered.append(track)
        except Exception as e:
            m_print(f"  [!] Warning: Could not check rating for '{track.title}' - {e}. Skipping filter.")
    return filtered

@functools.lru_cache(maxsize=None)
def clean_title(title):
    title_clean = title.casefold().strip()

    # 1) Remove feat/ft patterns and dash-suffix patterns first
    for pat in _FEATURING_RES:
        title_clean = pat.sub("", title_clean).strip()

    # 2) Remove parenthetical/bracketed chunks that contain version keywords
    title_clean = _PAREN_KW_RE.sub(" ", title_clean)
    title_clean = _BRACKET_KW_RE.sub(" ", title_clean)

    # 3) Remove remaining standalone version keywords (not in brackets)
    for pat in _KW_RES:
        title_clean = pat.sub(" ", title_clean).strip()

    # 4) Cleanup
    title_clean = _EMPTY_PAREN_RE.sub("", title_clean)    # remove empty ()
    title_clean = _EMPTY_BRACKET_RE.sub("", title_clean)  # remove empty []
    title_clean = re.sub(r"\s+", " ", title_clean).strip()
    title_clean = _TRAIL_DASH_RE.sub("", title_clean)     # trim trailing spaces or hyphens

    return title_clean


def process_tracks(tracks):
    """
    Process tracks to remove duplicates and balance artist/genre representation.

    Dedup strategy:
        - Key on (cleaned title, primary track artist) so the same recording on
        different albums (studio vs compilation/soundtrack) collapses.
        - When duplicates exist, keep the "better" copy (prefer vastly sonically
        superior matches, then studio album, artist album, then userRating).
    """
    filtered_tracks = filter_low_rated_tracks(tracks)

    # Phase 1: choose best copy per (title, primary artist) key
    best_by_key = {}
    key_order = []

    for track in filtered_tracks:
        try:
            if not hasattr(track, "ratingKey") or not hasattr(track, "title"):
                continue

            title_key = norm_text(clean_title(track.title))
            artist_key = norm_text(primary_artist(track_artist_name(track)))
            track_key = (title_key, artist_key)

            if track_key in best_by_key:
                best_by_key[track_key] = better_copy(best_by_key[track_key], track)
            else:
                best_by_key[track_key] = track
                key_order.append(track_key)
        except Exception as e:
            m_print(f"[WARN] Error during duplicate comparison for {track.title}: {e}")
            continue

    deduped_tracks = [best_by_key[k] for k in key_order]

    # Phase 2: enforce artist + mood (vibe) + style (primary) / genre (fallback) balance
    unique_tracks = []
    artist_count = Counter()
    mood_count   = Counter()
    genre_count  = Counter()
    style_count  = Counter()
    artist_limit = round(MAX_TRACKS * ARTIST_RATIO)
    mood_limit   = int(MAX_TRACKS * MOOD_RATIO)
    style_limit  = int(MAX_TRACKS * STYLE_RATIO)
    genre_limit  = int(MAX_TRACKS * GENRE_RATIO)

    for track in deduped_tracks:
        try:
            artist_name = norm_text(primary_artist(track_artist_name(track)))
            if artist_count[artist_name] >= artist_limit:
                continue

            # Mood is the vibe-consistency axis — checked before style/genre so no single
            # mood dominates the playlist. Primary mood only (depth=1).
            track_moods = _resolve_tags(track, "moods")
            if track_moods and mood_count[track_moods[0]] >= mood_limit:
                continue

            # Styles are the primary diversity axis (more granular than genres).
            # Check up to STYLE_TAG_DEPTH style slots per track — a track is rejected if
            # ANY of its checked styles is already at the limit, giving a hard diversity cap.
            # When accepted, counts against ALL checked styles so load spreads across buckets.
            # Genre is the fallback only when a track carries no styles at all.
            track_styles = _resolve_tags(track, "styles")[:STYLE_TAG_DEPTH]
            track_genres = _resolve_tags(track, "genres")

            if track_styles:
                if any(style_count[s] >= style_limit for s in track_styles):
                    continue
                for s in track_styles:
                    style_count[s] += 1
            elif track_genres:
                primary_genre = track_genres[0]
                if genre_count[primary_genre] >= genre_limit:
                    continue
                genre_count[primary_genre] += 1

            if track_moods:
                mood_count[track_moods[0]] += 1
            artist_count[artist_name] += 1
            unique_tracks.append(track)
        except Exception as e:
            m_print(f"[WARN] Error enforcing limits for {track.title}: {e}")
            continue

    log_text(f"[DEDUPE] Pool processed. Kept {len(unique_tracks)} unique tracks after deduplication and balance checks.")
    return unique_tracks

def fetch_sonically_similar_tracks(reference_tracks, excluded_keys=None):
    """
    Fetches Plex sonicallySimilar candidates for each seed track in parallel,
    populates the global sonic distance cache, then filters out recently played,
    excluded, low-rated, and already-selected tracks.

    Returns a deduplicated list of candidate tracks ready for process_tracks().
    """
    similar_tracks = []
    now = datetime.now()
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    log_text(f"[SONIC] Searching similarities for {len(reference_tracks)} seed tracks.")

    def _fetch_one(track):
        try:
            sims = track.sonicallySimilar(limit=SONIC_SIMILARITY_SEARCH_LIMIT, maxDistance=SONIC_SIMILARITY_DISTANCE)
            return track, sims
        except Exception as e:
            m_print(f"Error fetching sonically similar tracks: {e}")
            return track, []

    io_workers = get_optimal_workers(task_type="io")
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
        fetched = list(executor.map(_fetch_one, reference_tracks))

    for track, similars in fetched:
        filtered_similars = []

        # Global Sonic Caching
        # Capture distance data now so the sorting algorithm doesn't have to call Plex again.
        rk_a = track.ratingKey
        if rk_a not in _global_sonic_cache:
            _global_sonic_cache[rk_a] = {}

        for s in similars:
            # Capture and reuse distance data
            dist = getattr(s, 'distance', SONIC_SIMILARITY_DISTANCE)
            _global_sonic_cache[rk_a][s.ratingKey] = dist
            s.sonic_distance = dist

            last_played = getattr(s, "lastViewedAt", None)

            # Exclude if it was played recently
            if last_played and last_played >= exclude_start:
                d_print(f"[SONIC] Skipping '{s.title}' ({s.ratingKey}): Recently played ({last_played}).")
                continue

            # Exclude if it's already in the excluded keys
            if excluded_keys and s.ratingKey in excluded_keys:
                d_print(f"[SONIC] Skipping '{s.title}': Already in selection/history list.")
                continue

            filtered_similars.append(s)

        # Run deduplication before adding similar tracks
        filtered_similars = filter_excluded_tracks(filtered_similars, now=now)
        final_similars = process_tracks(filter_low_rated_tracks(filtered_similars))
        similar_tracks.extend(final_similars)

    return similar_tracks


# --- OPTIMIZED SONIC SORTING LOGIC ---
def get_adj_dist(ka, kb, similarity_cache, meta_cache, limit=SONIC_SIMILAR_LIMIT):
    """
    Calculates a normalized distance between two tracks, 0.0 (identical) to 1.0 (dissimilar).

    Distance sources in priority order:
      1. Global sonic cache (_global_sonic_cache) — pre-computed Plex distances
      2. Local similarity_cache — built during the current sort pass
      3. Synthetic fallback — estimated from shared moods/styles/genres/era

    Essentia weights (BPM, key, energy, era) are added on top using squared penalties,
    which makes large jumps disproportionately costly compared to small drifts.
    A 'bridge bonus' (-0.08) rewards BPM/key-compatible cross-genre transitions.
    An artist penalty (+0.5) enforces hard separation between same-artist tracks.
    """
    # Safety: if either key is absent from meta_cache, return a neutral distance
    if ka not in meta_cache or kb not in meta_cache:
        return 0.20

    # 1. Check for raw sonic distance in global cache first
    base_dist = _global_sonic_cache.get(ka, {}).get(kb, None)
    
    if base_dist is None:
        # Check the local similarity matrix provided during sorting
        base_dist = similarity_cache.get(ka, {}).get(kb, None)
    
    if base_dist is None:
        # Fallback: Synthetic Distance based on metadata
        ma, mb = meta_cache[ka], meta_cache[kb]
        score = 0.8  # Start with high dissimilarity
        
        # Mood is the strongest tonal signal; style (e.g. "Emo", "Indie Rock") is
        # more specific than genre, so it ranks between moods and genres.
        shared_moods = ma["moods"] & mb["moods"]
        if shared_moods:
            score -= 0.4
        elif ma["styles"] & mb["styles"]:  # Style is more specific than genre
            score -= 0.3
        elif ma["genres"] & mb["genres"]:  # Genre is the coarsest fallback
            score -= 0.2
            
        # Era/Decade similarity prevents "time-travel" jumps
        if ma["year"] and mb["year"]:
            if abs(ma["year"] - mb["year"]) <= 5:
                score -= 0.1
        
        dist = score
    else:
        # Scale the 0-20 rank or 0.0-0.25 distance to a standard float
        if isinstance(base_dist, int):
            dist = (base_dist / limit) * 0.4
        else:
            dist = (base_dist / 0.25) * 0.4

    # 2. Add Essentia Logic for Tempo, Key, Energy, and Era
    if ESSENTIA_ENABLED:
        ea, eb = _essentia_cache.get(str(ka)), _essentia_cache.get(str(kb))
        if ea and eb and "energy" in ea and "energy" in eb:
            # Tempo & Key - Squaring the distance to penalize "jumps" over "flows"
            dist += ((get_bpm_distance(ea["bpm"], eb["bpm"]) ** 2) * BPM_WEIGHT)
            dist += ((get_harmonic_distance(ea["key"], eb["key"]) ** 2) * KEY_WEIGHT)

            # Energy/Loudness Jump Penalty - Using an exponent to punish volume clashes
            energy_diff = abs(ea["energy"] - eb["energy"])
            energy_dist = min(energy_diff / 10.0, 1.0) # 10dB diff = max penalty
            dist += ((energy_dist ** 2) * ENERGY_WEIGHT)
            
            # Era/Decade Jump Penalty - Squaring ensures decade jumps are much costlier than 2-3 year shifts
            if ea["year"] and eb["year"]:
                year_diff = abs(ea["year"] - eb["year"])
                year_dist = min(year_diff / 50.0, 1.0) # Penalty scales up to 50 years
                dist += ((year_dist ** 2) * ERA_WEIGHT)

            # Bridge Bonus Logic: Reward sonic compatibility across different genres
            ma, mb = meta_cache[ka], meta_cache[kb]
            if ma["genres"] != mb["genres"]:
                if abs(ea["bpm"] - eb["bpm"]) < 1.0 and get_harmonic_distance(ea["key"], eb["key"]) < 0.1:
                    dist -= 0.08

    # 3. Artist Clustering Penalty (Strict)
    if meta_cache[ka]["artist"] == meta_cache[kb]["artist"]:
        dist += 0.5 

    return min(dist, 1.0)

def write_transition_log(full_path, similarity_cache, meta_cache, limit=SONIC_SIMILAR_LIMIT):
    """Generates a log detailing the transition quality between tracks."""
    try:
        log_path = resolve_path("logs/transition_log.txt", BASE_DIR)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(f"Transition Quality Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("-" * 80 + "\n")
            for i in range(len(full_path) - 1):
                t1, t2 = full_path[i], full_path[i+1]
                dist = get_adj_dist(t1.ratingKey, t2.ratingKey, similarity_cache, meta_cache, limit)
                status = "[JUMP]" if dist > 0.4 else "[FLOW]"
                log.write(f"{status} {dist:.3f} | {t1.title} -> {t2.title}\n")
    except Exception as e:
        m_print(f"[WARN] Failed to write transition log: {e}")

def sort_by_sonic_similarity_refined(tracks, first_track, last_track, limit=SONIC_SIMILAR_LIMIT):
    """
    Orders `tracks` into the smoothest possible sonic path between first_track and last_track.

    Algorithm (two phases):
      1. Double-ended greedy: at each step, picks the remaining track whose combined
         'distance from current' + 'lookahead distance to end' is lowest.
         The 0.3 lookahead weight nudges the path toward the intended endpoint without
         being so strong that it pulls every track toward the end too early.
      2. 2-opt refinement: repeatedly reverses sub-sequences of the path when doing so
         reduces total cost. Uses a cubic penalty (d**3 * 20) for any edge > 0.4 distance
         so the optimizer strongly prefers eliminating hard jumps over minor smoothing.

    Returns (ordered_path, similarity_cache, meta_cache) — always a 3-tuple.
    """
    if not tracks:
        return [], {}, {}  # Always return a 3-tuple so callers can unpack safely

    all_involved = tracks + [first_track, last_track]
    similarity_cache = {}
    meta_cache = {}

    # Pre-compute metadata for all tracks once. _resolve_tags is cached, so
    # album/artist fallback calls are O(1) on subsequent tracks from the same album/artist.
    for track in all_involved:
        t_key = track.ratingKey
        ec = _essentia_cache.get(str(t_key), {})
        meta_cache[t_key] = {
            "artist": ec.get("artist") or norm_text(primary_artist(track_artist_name(track))),
            "genres": set(ec.get("genres") or _resolve_tags(track, "genres")),
            "moods":  set(ec.get("moods")  or _resolve_tags(track, "moods")),
            "styles": set(ec.get("styles") or _resolve_tags(track, "styles")),
            "year": ec.get("year") or getattr(track, "year", None)
        }

    # Fetch sonicallySimilar in parallel for tracks not already in the global cache
    tracks_needing_fetch = [t for t in all_involved if t.ratingKey not in _global_sonic_cache]

    def _fetch_similar(track):
        try:
            sims = track.sonicallySimilar(limit=limit)
            return track.ratingKey, {s.ratingKey: getattr(s, 'distance', i) for i, s in enumerate(sims)}
        except Exception:
            return track.ratingKey, {}

    if tracks_needing_fetch:
        io_workers = get_optimal_workers(task_type="io")
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as executor:
            for t_key, sims in executor.map(_fetch_similar, tracks_needing_fetch):
                similarity_cache[t_key] = sims

    # --- 1. BI-DIRECTIONAL GREEDY INITIALIZATION ---
    _dist_cache = {}

    def _adj(ka, kb):
        key = (ka, kb)
        if key not in _dist_cache:
            _dist_cache[key] = get_adj_dist(ka, kb, similarity_cache, meta_cache, limit)
        return _dist_cache[key]

    remaining = list(tracks)
    path = []
    current_key = first_track.ratingKey
    end_key = last_track.ratingKey

    log_text(f"[ORDERING] Initializing Greedy path (Start: '{first_track.title}', End: '{last_track.title}').")

    while remaining:
        # Greedy score = distance from current + (0.3 × distance to end).
        # The lookahead nudges the path toward the endpoint without fully committing
        # every step to it — pure greedy would tunnel straight to the end too early.
        best_idx, best_track = min(
            enumerate(remaining),
            key=lambda x: _adj(current_key, x[1].ratingKey) +
                          (_adj(x[1].ratingKey, end_key) * 0.3)
        )
        path.append(best_track)
        remaining[best_idx] = remaining[-1]
        remaining.pop()
        current_key = best_track.ratingKey

    # --- 2. 2-OPT REFINEMENT WITH JUMP PENALTY ---
    def edge_cost(ka, kb):
        d = _adj(ka, kb)
        # Cubic penalty makes hard jumps (>0.4) catastrophically expensive,
        # so the optimizer aggressively eliminates them over minor smoothing gains.
        return (d ** 3) * 20 if d > 0.4 else d

    def total_cost(p):
        full_path = [first_track] + p + [last_track]
        return sum(edge_cost(full_path[i].ratingKey, full_path[i+1].ratingKey) for i in range(len(full_path) - 1))

    start_cost = total_cost(path)
    n = len(path)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            prev_i = first_track.ratingKey if i == 0 else path[i - 1].ratingKey
            for j in range(i + 1, n):
                next_j = last_track.ratingKey if j == n - 1 else path[j + 1].ratingKey
                old = edge_cost(prev_i, path[i].ratingKey) + edge_cost(path[j].ratingKey, next_j)
                new = edge_cost(prev_i, path[j].ratingKey) + edge_cost(path[i].ratingKey, next_j)
                if new < old:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True

    end_cost = total_cost(path)
    log_text(f"[ORDERING] 2-opt refinement complete. Sonic path cost reduced from {start_cost:.2f} to {end_cost:.2f}.")
    
    return path, similarity_cache, meta_cache

def get_track_meta(track):
    rk = track.ratingKey
    ec = _essentia_cache.get(str(rk), {})
    return {
        "artist": ec.get("artist") or norm_text(primary_artist(track_artist_name(track))),
        "genres": set(ec.get("genres") or _resolve_tags(track, "genres")),
        "moods":  set(ec.get("moods")  or _resolve_tags(track, "moods")),
        "styles": set(ec.get("styles") or _resolve_tags(track, "styles")),
        "year": ec.get("year") or getattr(track, "year", None)
    }

# Bridge Pass [Smarter Bridge Track Selection]
def fill_sonic_gaps(path, limit=SONIC_SIMILARITY_SEARCH_LIMIT, similarity_cache=None, meta_cache=None):
    """Identifies jumps > 0.4 and attempts to insert a bridge track from the library."""
    if similarity_cache is None: similarity_cache = {} # Safety fallback
    if not path or len(path) < 2: return path
    final_path = []
    
    # Safety and Deduplication Logic
    # Track identity by (Normalized Title, Normalized Artist) to catch duplicates across different albums
    # Safety and Deduplication Logic
    existing_identities = {
        (norm_text(clean_title(t.title)), norm_text(primary_artist(track_artist_name(t)))) 
        for t in path
    }
    existing_keys = {t.ratingKey for t in path}
    
    # Track artist counts to respect global limits (matching process_tracks logic)
    artist_counts = Counter([norm_text(primary_artist(track_artist_name(t))) for t in path])
    artist_limit = round(MAX_TRACKS * ARTIST_RATIO)

    # Track mood/style/genre counts for soft diversity preference during bridge selection.
    # Uses the same multi-depth logic as process_tracks(): count against all STYLE_TAG_DEPTH
    # style slots per track so the over_limit check reflects true style saturation.
    mood_counts  = Counter()
    style_counts = Counter()
    genre_counts = Counter()
    for _t in path:
        _t_moods = _resolve_tags(_t, "moods")
        if _t_moods:
            mood_counts[_t_moods[0]] += 1
        _t_styles = _resolve_tags(_t, "styles")[:STYLE_TAG_DEPTH]
        if _t_styles:
            for _s in _t_styles:
                style_counts[_s] += 1
        else:
            _t_genres = _resolve_tags(_t, "genres")
            if _t_genres:
                genre_counts[_t_genres[0]] += 1
    mood_limit   = int(MAX_TRACKS * MOOD_RATIO)
    style_limit  = int(MAX_TRACKS * STYLE_RATIO)
    genre_limit  = int(MAX_TRACKS * GENRE_RATIO)
    
    now = datetime.now()
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    # Cross-gap rejection cache: tracks that failed a static filter (exclusion label,
    # seasonal, low rating) are skipped immediately in subsequent gap searches without
    # re-running any API calls. Context-dependent checks (already in playlist, artist
    # count, recency) are intentionally NOT cached since they change as bridges are added.
    bridge_rejected_rks = set()

    # Phase 1: Pre-scan all adjacent pairs to identify gaps and cache per-pair metadata.
    # get_adj_dist() uses only local lookups; this cheap pass lets us submit bridge
    # pre-fetches for exact gap positions before entering the evaluation loop.
    pre_dist   = {}
    pre_m_cache = {}
    gap_order  = []
    for _i in range(len(path) - 1):
        _t1, _t2 = path[_i], path[_i + 1]
        _mc = {
            _t1.ratingKey: (meta_cache.get(_t1.ratingKey) if meta_cache else None) or get_track_meta(_t1),
            _t2.ratingKey: (meta_cache.get(_t2.ratingKey) if meta_cache else None) or get_track_meta(_t2),
        }
        _d = get_adj_dist(_t1.ratingKey, _t2.ratingKey, similarity_cache, _mc, limit)
        pre_dist[_i]    = _d
        pre_m_cache[_i] = _mc
        if _d > 0.4:
            gap_order.append(_i)

    # Phase 2 — 1-ahead bridge pre-fetch with a single background thread.
    #
    # A single worker submits sonicallySimilar() calls to Plex back-to-back (one at
    # a time — same server load as sequential).  Because the executor starts the next
    # fetch immediately when the previous one completes, the fetch latency for gap i+1
    # overlaps with our candidate-evaluation work for gap i.
    #
    # Unlike fully-parallel pre-fetch (which overwhelms Plex, seen in testing),
    # this keeps at most ONE sonicallySimilar() request in-flight at any moment.
    # Savings ≈ min(T_eval, T_fetch) per gap — typically 15–30 s for a 15-gap playlist.
    _gap_futures: dict = {}
    _bridge_exe = concurrent.futures.ThreadPoolExecutor(max_workers=1) if gap_order else None
    if _bridge_exe:
        for _gidx in gap_order:
            _gap_futures[_gidx] = _bridge_exe.submit(
            )

    for i in range(len(path) - 1):
        t1, t2 = path[i], path[i+1]
        final_path.append(t1)

        m_cache = pre_m_cache[i]
        dist    = pre_dist[i]

        if dist > 0.4:
            log_text(f"[BRIDGE] Gap detected: {t1.title} -> {t2.title} (Gap: {dist:.3f}). Searching for candidate...")

            t1_artist_norm = norm_text(primary_artist(track_artist_name(t1)))
            t2_artist_norm = norm_text(primary_artist(track_artist_name(t2)))

            try:
                # Retrieve pre-fetched results from the background thread.
                # .result() blocks only if the fetch isn't done yet (rare for later gaps).
                potential_bridges = _gap_futures[i].result()
                candidates = [] # Collector for Best-Fit
                for bridge in potential_bridges:

                    # A0. CROSS-GAP REJECTION CACHE — cheapest possible check
                    if bridge.ratingKey in bridge_rejected_rks:
                        continue

                    # A. FAST ARTIST & IDENTITY CHECK
                    if bridge.ratingKey in existing_keys:
                        d_print(f"  [!] Skipping '{bridge.title}': Already in selection.")
                        continue

                    # C. FAST EXCLUSION CHECK
                    # Order matters: filter_excluded_tracks is O(1) set lookup (no API calls).
                    # filter_low_rated_tracks makes track.artist() API calls on cache misses.
                    # Running the fast check first means excluded tracks (noshare, seasonal)
                    # are dropped before any network round-trip is ever made.
                    if not filter_low_rated_tracks(filter_excluded_tracks([bridge])):
                        bridge_rejected_rks.add(bridge.ratingKey)
                        d_print(f"  [!] Skipping '{bridge.title}': Failed exclusion/rating filters.")
                        continue

                    b_artist_raw = track_artist_name(bridge)
                    b_artist_norm = norm_text(primary_artist(b_artist_raw))
                    b_identity = (norm_text(clean_title(bridge.title)), b_artist_norm)
                    
                    # Dedupe check
                    if b_identity in existing_identities:
                        d_print(f"  [!] Skipping '{bridge.title}': Already in selection.")
                        continue

                    # NEW: Global artist limit check
                    if artist_counts[b_artist_norm] >= artist_limit:
                        d_print(f"  [!] Skipping '{bridge.title}': Artist '{b_artist_raw}' saturated ({artist_counts[b_artist_norm]}).")
                        continue

                    # B. FAST RECENCY CHECK
                    last_p = getattr(bridge, "lastViewedAt", None)
                    if last_p and last_p >= exclude_start:
                        d_print(f"  [!] Skipping '{bridge.title}': Recently played.")
                        continue
                    
                    # NEW: Back-to-back check (prevents bridge matching t1 OR t2 artists)
                    if b_artist_norm == t1_artist_norm or b_artist_norm == t2_artist_norm:
                        d_print(f"  [!] Skipping '{bridge.title}': Artist back-to-back clash.")
                        continue

                    # D. METADATA & DISTANCE CHECK
                    bm = get_track_meta(bridge)
                    # Evaluate distance to both sides of the gap to find the best "middle ground"
                    d1 = get_adj_dist(t1.ratingKey, bridge.ratingKey, {}, {**m_cache, bridge.ratingKey: bm}, limit)
                    d2 = get_adj_dist(bridge.ratingKey, t2.ratingKey, {}, {**m_cache, bridge.ratingKey: bm}, limit)
                    
                    # Logic: If both legs of the new path are better than the single original jump, it is a net improvement for the "flow."
                    if d1 < dist and d2 < dist:
                        b_styles = _resolve_tags(bridge, "styles")[:STYLE_TAG_DEPTH]
                        b_genres = _resolve_tags(bridge, "genres")
                        if b_styles:
                            # over_limit if ANY checked style slot is saturated (hard cap)
                            over_limit = any(style_counts[s] >= style_limit for s in b_styles)
                        elif b_genres:
                            over_limit = genre_counts[b_genres[0]] >= genre_limit
                        else:
                            over_limit = False
                        # Also apply mood soft preference: deprioritise bridges that push a mood over limit
                        b_moods = _resolve_tags(bridge, "moods")
                        if b_moods and mood_counts[b_moods[0]] >= mood_limit:
                            over_limit = True
                        # Tuple sorts within-limits first (0), over-limits second (1), then by flow score
                        candidates.append((int(over_limit), d1 + d2, bridge, b_identity, b_artist_norm, b_styles, b_genres))

                # F. SELECTION
                if candidates:
                    # Within-limits candidates sort first (flag=0), over-limits second (flag=1);
                    # within each group, lowest combined flow score wins.
                    candidates.sort()
                    over_limit_flag, best_score, best_bridge, b_identity, b_artist_norm, b_styles, b_genres = candidates[0]

                    # E. AUDIO ANALYSIS (Only for the winner)
                    if ESSENTIA_ENABLED:
                        analyze_track_essentia(best_bridge)

                    if over_limit_flag:
                        log_text(f"[BRIDGE] Selected Best Fit: '{best_bridge.title}' (Combined Score: {best_score:.3f}). Inserting (style/genre limit exceeded — no within-limit option).")
                    else:
                        log_text(f"[BRIDGE] Selected Best Fit: '{best_bridge.title}' (Combined Score: {best_score:.3f}). Inserting.")
                    final_path.append(best_bridge)

                    # Update tracking sets, artist counter, and style/genre counts
                    existing_identities.add(b_identity)
                    existing_keys.add(best_bridge.ratingKey)
                    artist_counts[b_artist_norm] += 1
                    if b_styles:
                        for _s in b_styles:
                            style_counts[_s] += 1
                    elif b_genres:
                        genre_counts[b_genres[0]] += 1
                else:
                    log_text(f"[BRIDGE] Failure: No suitable bridge found for {t1.title} -> {t2.title}.")
            except Exception as e:
                m_print(f"  [ERR] Bridge error: {e}")

    # Shut down the 1-ahead pre-fetch executor now that all gaps have been evaluated.
    # wait=False: any in-flight requests (shouldn't be any) finish in the background.
    if _bridge_exe is not None:
        _bridge_exe.shutdown(wait=False)

    final_path.append(path[-1])

    # Smart Truncation Logic to stay within MAX_TRACKS
    if SMART_TRUNCATION_ENABLED and len(final_path) > MAX_TRACKS:
        m_print(f"[BRIDGE] Smart Truncating playlist from {len(final_path)} to {MAX_TRACKS}...")

        # Pre-compute metadata for every track once — it doesn't change during removal
        trunc_meta = {t.ratingKey: get_track_meta(t) for t in final_path}

        while len(final_path) > MAX_TRACKS:
            best_remove_idx = -1
            min_added_distance = float('inf')

            # Evaluate every track except the first and last
            for i in range(1, len(final_path) - 1):
                t_prev = final_path[i-1]
                t_next = final_path[i+1]

                # Metadata for neighbor distance calculation (reuse pre-computed dict)
                m_cache = {
                    t_prev.ratingKey: trunc_meta[t_prev.ratingKey],
                    t_next.ratingKey: trunc_meta[t_next.ratingKey],
                }

                # Calculate what the new jump would be if we removed final_path[i]
                new_dist = get_adj_dist(t_prev.ratingKey, t_next.ratingKey, {}, m_cache, limit)

                if new_dist < min_added_distance:
                    min_added_distance = new_dist
                    best_remove_idx = i

            # Remove the "least essential" track for sonic flow
            if best_remove_idx != -1:
                removed_track = final_path[best_remove_idx]
                log_text(f"[BRIDGE] Removed '{removed_track.title}' to minimize sonic impact during truncation.")
                final_path.pop(best_remove_idx)

        m_print(f"[OK] Smart truncation complete. Playlist optimized at {MAX_TRACKS} tracks.")

    return final_path
# ------------------------------------


def generate_playlist_title_and_description(period, tracks):
    descriptor_map = load_descriptor_map(MOOD_MAP_PATH)
    day_name = datetime.now().strftime("%A")

    top_genres = [str(g) for t in tracks for g in (t.genres or [])]
    top_moods = [str(m) for t in tracks for m in (t.moods or [])]
    genre_counts = Counter(top_genres)
    mood_counts = Counter(top_moods)

    sorted_genres = [g for g, _ in genre_counts.most_common()]
    sorted_moods = [m for m, _ in mood_counts.most_common()]

    most_common_genre = sorted_genres[0] if sorted_genres else "Eclectic"
    most_common_mood = sorted_moods[0] if sorted_moods else "Vibes"
    second_common_mood = sorted_moods[1] if len(sorted_moods) > 1 else None

    descriptor = random.choice(descriptor_map.get(second_common_mood, ["Vibrant"]))
    period_phrase = get_period_phrase(period)
    title = f"Meloday for {most_common_mood} {descriptor} {most_common_genre} {day_name} {period}"

    max_styles = 6
    highlight_styles = sorted_genres[:3] + sorted_moods[:3]
    highlight_styles = [s for s in highlight_styles if s not in {most_common_genre, most_common_mood}]
    highlight_styles = list(dict.fromkeys(highlight_styles))[:max_styles]
    
    # Ensure highlight styles are filled with whatever is available
    additional = sorted_genres + sorted_moods
    for s in additional:
        if len(highlight_styles) >= max_styles:
            break
        if s not in highlight_styles:
            highlight_styles.append(s)

    # Build the highlight phrase safely
    if len(highlight_styles) > 1:
        extra_info = f"Here's some {', '.join(highlight_styles[:-1])}, and {highlight_styles[-1]} tracks as well."
    elif len(highlight_styles) == 1:
        extra_info = f"Here's some {highlight_styles[0]} tracks as well."
    else:
        extra_info = "Enjoy this selection of your favorites."

    if second_common_mood:
        description = (
            f"You listened to {most_common_mood} and {most_common_genre} tracks on {day_name} {period_phrase}. "
            f"{extra_info}"
        )
    else:
        description = (
            f"You listened to {most_common_genre} and {most_common_mood} tracks on {day_name} {period_phrase}. "
            f"{extra_info}"
        )

    try:
        plex_account = plex.myPlexAccount()
        plex_user = plex_account.title.split()[0] if plex_account.title else plex_account.username
    except Exception:
        plex_user = "you"

    now = datetime.now()
    next_update_hour = (time_periods[period]["hours"][-1] + 1) % 24
    next_update = now.replace(hour=next_update_hour, minute=0, second=0)
    if next_update_hour < now.hour:
        next_update += timedelta(days=1)

    description += f"\n\nMade for {plex_user} • Next update at {next_update.strftime('%I:%M %p').lstrip('0')}."
    return title, description

def apply_text_to_cover(image_path, text):
    try:
        prefix = "Meloday for "
        if text.startswith(prefix):
            text = text[len(prefix):]

        image = Image.open(image_path).convert("RGBA")
        shadow_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        text_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        text_draw = ImageDraw.Draw(text_layer)

        try:
            font_main = ImageFont.truetype(FONT_MAIN_PATH, size=67)
            font_meloday = ImageFont.truetype(FONT_MELODAY_PATH, size=87)
        except IOError:
            font_main = ImageFont.load_default()
            font_meloday = ImageFont.load_default()

        text_box_width, text_box_right = 630, image.width - 110
        text_box_left = text_box_right - text_box_width
        y = 100

        lines = wrap_text(text, font_main, text_draw, text_box_width)
        for line in lines:
            bbox = text_draw.textbbox((0, 0), line, font=font_main)
            x = text_box_left + (text_box_width - (bbox[2] - bbox[0]))
            shadow_draw.text((x, y), line, font=font_main, fill=(0, 0, 0, 120))
            text_draw.text((x, y), line, font=font_main, fill=(255, 255, 255, 255))
            y += bbox[3] - bbox[1] + 10

        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=40))
        shadow_draw.text((110, image.height - 200), "Meloday", font=font_meloday, fill=(0, 0, 0, 120))
        text_draw.text((110, image.height - 200), "Meloday", font=font_meloday, fill=(255, 255, 255, 255))

        combined = Image.alpha_composite(image, shadow_layer)
        combined = Image.alpha_composite(combined, text_layer)
        new_path = image_path.replace(".webp", "_texted.webp")
        combined.convert("RGB").save(new_path)
        return new_path
    except Exception as e:
        m_print(f"[WARN] apply_text_to_cover failed: {e}")
        return image_path

def create_or_update_playlist(name, tracks, description, cover_file):
    existing_playlist = next((pl for pl in plex.playlists() if str(getattr(pl, "title", "")).startswith("Meloday for ")), None)
    valid_tracks = [t for t in tracks if getattr(t, "ratingKey", None)]
    
    if not valid_tracks:
        # Gracefully exit if no tracks found to fix the "Empty Result" crash
        m_print("[WARN] No valid tracks to add. Skipping playlist update.")
        return

    if existing_playlist:
        existing_playlist.removeItems(existing_playlist.items())
        existing_playlist.addItems(valid_tracks)
        existing_playlist.editTitle(name)
        existing_playlist.editSummary(description)
        playlist_obj = existing_playlist
    else:
        playlist_obj = plex.createPlaylist(name, items=valid_tracks)
        playlist_obj.editSummary(description)

    m_print(f"[OK] Playlist updated: {name} | items: {len(valid_tracks)}")

    cover_path = os.path.join(COVER_IMAGE_DIR, cover_file)
    if os.path.exists(cover_path):
        try:
            new_cover = apply_text_to_cover(cover_path, name)
            playlist_obj.uploadPoster(filepath=new_cover)
            m_print(f"[OK] Uploaded poster: {new_cover}")
        except Exception as e:
            m_print("[WARN] Poster upload failed (playlist still created):")
            log_text(f"[WARN] Poster upload failed: {str(e)}")
    else:
        m_print(f"[WARN] Cover file not found: {cover_path}")

def find_first_and_last_tracks(tracks, period):
    if not tracks: return None, None
    valid_hours = set(time_periods[period]["hours"])
    sorted_tracks = sorted(tracks, key=lambda t: t.lastViewedAt or datetime.max)
    first = next((t for t in sorted_tracks if t.lastViewedAt and t.lastViewedAt.hour in valid_hours), sorted_tracks[0])
    last = next((t for t in reversed(sorted_tracks) if t.lastViewedAt and t.lastViewedAt.hour in valid_hours), sorted_tracks[-1])
    return first, last

def main():
    # Force log truncation once at the start of the main process
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.truncate(0)

    # NEW: Initialize global plex connection only here
    global plex
    plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=60)

    log_text("=== MELODAY RUN STARTED ===")

    # NEW: Run environment validation
    if not validate_environment():
        log_text("=== MELODAY ABORTED DUE TO VALIDATION ERRORS ===")
        return

    prefetch_seasonal_exclusions()
    prefetch_label_exclusions()

    # Step 0% - Start
    print_status(0, "Starting track selection...")
    period = get_current_time_period()
    print_status(10, f"Period: {period}")

    # Confirm Essentia Status
    if ESSENTIA_ENABLED:
        m_print(f"[DIAGNOSTIC] Essentia is ACTIVE. Analyzing keys and BPM.")
        load_essentia_cache()
    else:
        reason = "Library not found" if not ESSENTIA_AVAILABLE else "Disabled in config"
        m_print(f"[DIAGNOSTIC] Essentia is INACTIVE ({reason}). Using standard sorting.")

    # Step 1: Fetch historical (Guarantee based on configured historical_ratio)
    print_status(20, "Fetching historical tracks...")
    historical, excluded_keys = fetch_historical_tracks(period)
    guaranteed = random.sample(historical, min(int(MAX_TRACKS * HISTORICAL_RATIO), len(historical)))

    # Step 2: Fetch similar
    print_status(30, "Fetching sonically similar tracks...")
    similar = fetch_sonically_similar_tracks(guaranteed, excluded_keys=excluded_keys)
    final_tracks = process_tracks(guaranteed + similar)

    # Step 3: Ensure we reach MAX_TRACKS
    print_status(40, "Combining & processing tracks...")
    progress = 40
    while len(final_tracks) < MAX_TRACKS:
        progress += 5
        print_status(progress, "Attempting to add more tracks...")
        more_h, more_e = fetch_historical_tracks(period)
        excluded_keys |= more_e
        more_s = fetch_sonically_similar_tracks(final_tracks, excluded_keys=excluded_keys)
        more_h_sample = random.sample(more_h, min(MAX_TRACKS - len(final_tracks), len(more_h)))
        prev_len = len(final_tracks)
        final_tracks = process_tracks(final_tracks + more_h_sample + more_s)[:MAX_TRACKS]
        if len(final_tracks) == prev_len: break

    if len(final_tracks) < MAX_TRACKS // 2:
        log_text(f"[WARN] Playlist is significantly short ({len(final_tracks)}/{MAX_TRACKS} tracks). "
                 f"Diversity caps may be too tight for the available candidate pool. "
                 f"Try running the optimizer, or increase style_ratio/genre_ratio in config.yml.")

    print_status(50, "Finding first & last historical tracks...")
    first, last = find_first_and_last_tracks(final_tracks[:MAX_TRACKS], period)
    middle = [t for t in final_tracks[:MAX_TRACKS] if t not in {first, last}]

    # Step 4: Sonic sort (GREEDY)
    similarity_cache = {}
    meta_cache = {}
    sort_breadth = 20

    if middle and first and last:
        if ESSENTIA_ENABLED:
            print_status(60, "Syncing metadata and analyzing new tracks...")
            all_tracks = [first, last] + middle
            now_ts = datetime.now().timestamp()
            to_analyze = []

            # PRE-FILTER: Check against your specific re-analyze logic
            for t in all_tracks:
                rk = str(t.ratingKey)
                file_path = get_local_path(t)
                
                if rk in _essentia_cache:
                    data = _essentia_cache[rk]
                    last_sync = data.get("last_synced", 0)
                    
                    # LOGIC: Skip only if path is identical AND synced within last 7 days
                    if data.get("file_path") == file_path and (now_ts - last_sync) < 604800:
                        continue 
                
                # If we are here, the track is missing or needs a fresh look
                to_analyze.append(t.ratingKey)

            if to_analyze:
                log_text(f"[DIAGNOSTIC] Cache miss/stale: Analyzing {len(to_analyze)} tracks.")
                cpu_workers = get_optimal_workers(task_type="cpu")
                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers, max_tasks_per_child=10) as executor:
                    # analysis_worker calls analyze_track_essentia internally
                    results = list(executor.map(analysis_worker, to_analyze))
                new_entries = {}
                for tid, data in results:
                    if data:
                        _essentia_cache[str(tid)] = data
                        new_entries[str(tid)] = data
                _upsert_cache_entries(new_entries)
            else:
                log_text("[OK] All tracks are cached and up-to-date.")

        print_status(70, "Double-ended 2-opt sonic refinement...")
        # Ensure sorting breadth covers the tracks in the list
        sort_breadth = max(SONIC_SIMILAR_LIMIT, len(middle) + 2)
        middle, similarity_cache, meta_cache = sort_by_sonic_similarity_refined(middle, first, last, limit=sort_breadth)

    final_ordered_tracks = [first] + middle + [last] if first and last else final_tracks[:MAX_TRACKS]

    # Bridge Pass
    if BRIDGING_ENABLED and len(final_ordered_tracks) > 2:
        # Step 4.5: Smooth technical gaps between vibe-compatible tracks.
        print_status(80, "Creating Sonic Bridges...")
        sb = max(SONIC_SIMILAR_LIMIT, len(middle) + 2) if middle else 20
        cache_keys_before_bridge = set(_essentia_cache.keys())
        final_ordered_tracks = fill_sonic_gaps(final_ordered_tracks, limit=sb, similarity_cache=similarity_cache, meta_cache=meta_cache)

        # Final Log Update after bridging and smart truncation
        # Refresh meta_cache for any newly added bridge tracks
        for t in final_ordered_tracks:
            if t.ratingKey not in meta_cache:
                meta_cache[t.ratingKey] = get_track_meta(t)

        if ESSENTIA_ENABLED:
            new_bridge_entries = {rk: _essentia_cache[rk] for rk in _essentia_cache if rk not in cache_keys_before_bridge}
            _upsert_cache_entries(new_bridge_entries)

    # Define a default sb (sort breadth) in case the bridge block was skipped
    sb_log = max(SONIC_SIMILAR_LIMIT, len(final_ordered_tracks)) 
    write_transition_log(final_ordered_tracks, similarity_cache, meta_cache, limit=sb_log)

    # Step 5: Playlist Update
    print_status(90, "Creating/Updating playlist...")
    title, desc = generate_playlist_title_and_description(period, final_ordered_tracks)
    create_or_update_playlist(title, final_ordered_tracks, desc, time_periods[period]['cover'])
    print_status(100, "Playlist creation/update complete!")
    log_text("=== MELODAY RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()