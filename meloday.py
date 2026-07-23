import yaml
import os

# These must be set before any essentia/TF imports — the C++ runtimes read them during dlopen
os.environ['ESSENTIA_LOG_LEVEL']    = '1'   # suppress essentia SVM info messages
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'   # suppress TF INFO+WARNING (incl. CUDA probe noise)

import re
import random
import json
import functools
import unicodedata
import sqlite3
import shutil
import traceback
import concurrent.futures
import multiprocessing
import logging
import argparse
import sys
import fcntl
import threading
from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
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
_raw_exclude = config["plex"]["exclude_label"]
EXCLUDE_LABEL_NAMES = [_raw_exclude] if isinstance(_raw_exclude, str) else list(_raw_exclude)

# Playlist & Logic Rules
EXCLUDE_PLAYED_DAYS = config["playlist"]["exclude_played_days"]
MAX_TRACK_MS = 15 * 60 * 1000   # exclude tracks longer than 15 minutes from all playlists
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
SONIC_SIMILARITY_DISTANCE = config["playlist"].get("sonic_similarity_distance", 0.20)

# Essentia Logic & Weights
ess_cfg = config.get("essentia", {})
ESSENTIA_ENABLED = ess_cfg.get("enabled", True) and ESSENTIA_AVAILABLE
ESSENTIA_CACHE_PATH = resolve_path(ess_cfg.get("cache_path", "assets/essentia_cache.db"), BASE_DIR)
# Auto-upgrade: if config still references the old .json path, silently redirect to .db
if ESSENTIA_CACHE_PATH.endswith('.json'):
    ESSENTIA_CACHE_PATH = ESSENTIA_CACHE_PATH[:-5] + '.db'
BPM_WEIGHT           = ess_cfg.get("bpm_weight",            0.15)
KEY_WEIGHT           = ess_cfg.get("key_weight",            0.10)
ENERGY_WEIGHT        = ess_cfg.get("energy_weight",         0.10)
ERA_WEIGHT           = ess_cfg.get("era_weight",            0.05)
DANCEABILITY_WEIGHT  = ess_cfg.get("danceability_weight",   0.08)
BRIGHTNESS_WEIGHT    = ess_cfg.get("brightness_weight",     0.06)
BEAT_CONF_WEIGHT     = ess_cfg.get("beat_confidence_weight", 0.07)
ONSET_RATE_WEIGHT    = ess_cfg.get("onset_rate_weight",      0.05)
AROUSAL_WEIGHT       = ess_cfg.get("arousal_weight",         0.10)
VALENCE_WEIGHT       = ess_cfg.get("valence_weight",         0.08)
VOCAL_WEIGHT         = ess_cfg.get("vocal_weight",           0.04)
PATH_MAPPING         = ess_cfg.get("path_mapping", {})

# Optional TF model inference (requires essentia-tensorflow + downloaded model files).
# Falls back silently when the package or model files are absent.
_TF_AVAILABLE     = False
_TF_MODELS_LOADED = False
_effnet_model = _musicnn_model = _av_model = _vocal_model = None
# When True, _fill_missing_acoustic re-runs the TF block for tracks that already have the other TF fields
# but never had their embeddings stored, so `pre_analyze.py --sync-embeddings` can backfill emb_effnet/
# emb_musicnn. Default off so normal (incremental) analysis never re-processes complete tracks. (audit Tier-4)
_FORCE_EMB_BACKFILL = False
# Optional high-level classification heads (run on the same EffNet/MusiCNN embeddings).
# Absent model files are skipped, so these fields simply stay null until the models exist.
_mood_models       = {}    # cache_field -> (head, "musicnn"|"effnet", positive_class_index)
_moodtheme_model   = None  # 56-class mood/theme sigmoid (EffNet)
_genre_model       = None  # 400-class Discogs genre (EffNet)
_MOODTHEME_CLASSES = []
_GENRE400_CLASSES  = []
# (cache_field, model basename in models_dir, embedding, index of the positive class)
_MOOD_HEAD_SPECS = (
    ("mood_happy",      "mood_happy-msd-musicnn-1",      "musicnn", 0),
    ("mood_sad",        "mood_sad-msd-musicnn-1",        "musicnn", 1),
    ("mood_aggressive", "mood_aggressive-msd-musicnn-1", "musicnn", 0),
    ("mood_relaxed",    "mood_relaxed-msd-musicnn-1",    "musicnn", 1),
    ("mood_party",      "mood_party-msd-musicnn-1",      "musicnn", 1),
    ("mood_acoustic",   "mood_acoustic-msd-musicnn-1",   "musicnn", 0),
    ("mood_electronic", "mood_electronic-msd-musicnn-1", "musicnn", 0),
    ("danceability_hl", "danceability-msd-musicnn-1",    "musicnn", 0),
)

# Cap each spawned analysis worker's BLAS/OpenMP/TF threads to 1 so N workers don't
# oversubscribe the CPU (N × all-core threads → thrash). Set in the parent before spawning;
# workers inherit at import. Local copy (pre_analyze has the same — meloday can't import it
# without a cycle). Used around the cache-miss re-analysis pool during playlist generation.
_THREAD_CAP_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")

def _set_thread_caps():
    """Cap every numeric-lib threadpool to 1 in subsequently-spawned workers; return prior values."""
    prev = {k: os.environ.get(k) for k in _THREAD_CAP_VARS}
    for k in _THREAD_CAP_VARS:
        os.environ[k] = "1"
    return prev

def _restore_env(prev):
    """Restore env vars captured by _set_thread_caps (None → unset)."""
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# ---------------------------------------------------------------------------
# Single-instance lock + runtime watchdog (shared by meloday.py and meloday_extras.py)
#
# WHY: overlapping cron fires (meloday.py at 7 hours/day; meloday_extras mood_mixes up to
# 3×/hour) had no mutual exclusion and no runtime bound, so a slow or hung run let the next
# fire stack on top. Each copy loads the ~3.8 GB essentia cache, so a few deep exhausts RAM
# (observed on prod: load ~30, 23G/30G, orphaned spawn_main workers). The flock guard makes a
# redundant run exit cleanly; the watchdog force-exits a run that overruns so it can never
# hold its lock (or its RAM) forever.
# ---------------------------------------------------------------------------

# Lockfiles live here. MUST be on a LOCAL filesystem — flock is a silent no-op on many
# network mounts, so never point this at the Plex share.
LOCK_DIR = resolve_path("assets/locks", BASE_DIR)

# Runtime safety limits, read from config.yml `runtime:` (all optional; 0 disables a watchdog).
_rt_cfg = config.get("runtime", {})
MELODAY_MAX_SECONDS     = int(float(_rt_cfg.get("meloday_max_minutes", 90)) * 60)
EXTRAS_MAX_SECONDS      = int(float(_rt_cfg.get("extras_max_minutes",  40)) * 60)
RESOLVE_TIMEOUT_SECONDS = int(_rt_cfg.get("resolve_timeout_seconds", 180))

# Holds the lock fd for the whole process lifetime. It's a raw os.open() fd, so (unlike a file
# object) the garbage collector never closes it — we still keep an explicit reference for clarity.
_INSTANCE_LOCK_FD = None

def single_instance_guard(name, log=log_text):
    """Acquire an exclusive, non-blocking flock so only one '<name>' run exists at a time.

    If another run already holds it, log a [SKIP] line and exit 0 (a redundant cron fire is a
    no-op, not an error). The kernel releases the lock automatically when this process dies —
    even on SIGKILL/OOM — which is why flock beats a stale-prone PID file here. Fails OPEN on
    infrastructure errors (unwritable lockdir) so a lock problem can never stop playlist runs.
    """
    global _INSTANCE_LOCK_FD
    try:
        os.makedirs(LOCK_DIR, exist_ok=True)
        # O_CREAT WITHOUT O_TRUNC: a contender that fails the flock below must not wipe the
        # holder's pid (open("w") truncates on open — before flock — so it can't be used here).
        fd = os.open(os.path.join(LOCK_DIR, f"{name}.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    except Exception as e:
        log(f"[WARN] single-instance lock unavailable ({e}); proceeding without it")
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"[SKIP] Another '{name}' run is active — exiting")
        os.close(fd)
        sys.exit(0)
    _INSTANCE_LOCK_FD = fd            # keep the fd for the whole process — do NOT close it
    try:
        os.ftruncate(fd, 0)          # only the holder rewrites the pid, so it stays readable for debugging
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
    except Exception:
        pass

def _worker_parent_sentinel():
    """Pool-worker initializer: force-exit this worker the moment its parent dies.

    WHY: stock spawn pool workers never notice a dead parent — the exit sentinel only arrives
    from a LIVE parent's manager thread, and sibling workers hold the call-queue pipe open so no
    EOF ever comes. After an OOM-kill / os._exit / crash of the parent, workers linger forever:
    idle ones block on the queue at 0% CPU, mid-inference ones keep burning CPU (observed on prod:
    ~70 orphans at 0.7-1.1 GB each). parent_process().join() returns the moment the parent dies,
    for ANY cause. Known limit: a worker whose main thread holds the GIL inside native code can't
    run this thread — reap_orphaned_workers() (SIGKILL, GIL-independent) sweeps that tail.
    """
    pp = multiprocessing.parent_process()
    if pp is None:
        return   # defensive: not a spawned child
    def _watch():
        pp.join()          # returns when the parent process dies
        os._exit(1)
    threading.Thread(target=_watch, name="parent-sentinel", daemon=True).start()

def _kill_child_processes(log=log_text):
    """Best-effort recursive kill of this process's children (analysis / spawn_main workers) so the
    watchdog's os._exit doesn't leave orphans reparented to init still burning CPU. Uses psutil
    (already a dependency); lazy-imported so a broken psutil never blocks the force-exit itself."""
    try:
        import psutil
        me = psutil.Process()
        # WHY: a single children() snapshot LEAKED workers — the analysis pool (max_tasks_per_child)
        # respawns replacements while we kill, so anything spawned after the snapshot survived the
        # sweep and was orphaned by the watchdog itself. Re-snapshot until no children remain (or 4
        # passes); a last-instant straggler is then caught by _worker_parent_sentinel once os._exit
        # lands and the parent is truly gone.
        for _pass in range(4):
            kids = me.children(recursive=True)
            if not kids:
                break
            for p in kids:
                try: p.kill()
                except Exception: pass
            psutil.wait_procs(kids, timeout=1)
    except Exception as e:
        log(f"[WATCHDOG] child cleanup skipped: {e}")

def reap_orphaned_workers(log=log_text):
    """Kill THIS venv's multiprocessing strays whose parent is gone (run at every entry-point start).

    WHY: workers can outlive every in-process cleanup — the parent may die by OOM-SIGKILL (no
    atexit), or a worker stuck in GIL-held native code can't run its parent-sentinel thread. Those
    orphans (plus their now-parentless resource_tracker) sat for days holding 0.7-1.1 GB each. Each
    scheduled run sweeps them: matches ONLY processes running this exact interpreter whose cmdline
    is a multiprocessing worker/tracker AND whose parent is missing/init/non-python — a live run's
    workers have a live python parent and are never touched. Best-effort: never blocks the run.
    """
    try:
        import psutil
        my_exe = os.path.realpath(sys.executable)
        victims = []
        for p in psutil.process_iter(attrs=["pid", "cmdline"]):
            try:
                cmd = " ".join(p.info["cmdline"] or ())
                if ("--multiprocessing-fork" not in cmd
                        and "multiprocessing.resource_tracker" not in cmd):
                    continue
                if os.path.realpath(p.exe()) != my_exe:
                    continue   # some other app's workers — never touch
                parent = p.parent()
                if parent is not None and parent.pid != 1 and "python" in parent.name().lower():
                    continue   # parent is a live python → an active run's worker
                victims.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        for p in victims:
            try: p.kill()
            except Exception: pass
        if victims:
            psutil.wait_procs(victims, timeout=3)
            log(f"[REAP] killed {len(victims)} orphaned worker process(es) from previous runs")
    except Exception as e:
        log(f"[REAP] orphan sweep skipped: {e}")

def start_watchdog(name, max_seconds, log=log_text):
    """Force-exit this process if it runs longer than max_seconds (0/None disables).

    A daemon thread — deliberately NOT signal.SIGALRM — because at timeout the main thread is
    almost always blocked in a C-level ProcessPoolExecutor.shutdown(wait=True) or a plexapi socket
    read, where a Python signal handler wouldn't run until the GIL returned (it can sit undelivered
    through the whole hang). A separate OS thread always runs, and os._exit is a direct _exit(2)
    that bypasses atexit and the stdlib concurrent.futures thread-join a hung worker would wedge.
    """
    if not max_seconds or max_seconds <= 0:
        return
    def _watch():
        threading.Event().wait(max_seconds)   # sleeps the full interval; never set → always times out
        log(f"[WATCHDOG] '{name}' exceeded {max_seconds}s — force-exiting")
        _kill_child_processes(log)
        os._exit(1)
    threading.Thread(target=_watch, name=f"watchdog-{name}", daemon=True).start()

# MELODAY_SKIP_TF_MODELS is set by pre_analyze.py in the main process before spawning
# workers. With 'spawn' context, workers re-import meloday.py; loading the TF C++
# runtime + three .pb models in every worker simultaneously uses ~700 MB–1 GB per
# worker and causes OOM on typical servers. Workers skip TF loading entirely;
# arousal/valence/vocal_presence are filled later during meloday.py playlist runs.
if ESSENTIA_AVAILABLE and not os.environ.get('MELODAY_SKIP_TF_MODELS'):
    try:
        _probe = getattr(es, "TensorflowPredictEffnetDiscogs", None)
        if _probe is not None:
            _TF_AVAILABLE = True
            _models_dir  = resolve_path(ess_cfg.get("models_dir", "assets/models"), BASE_DIR)
            _effnet_path  = os.path.join(_models_dir, "discogs-effnet-bs64-1.pb")
            _musicnn_path = os.path.join(_models_dir, "msd-musicnn-1.pb")
            _av_path      = os.path.join(_models_dir, "deam-msd-musicnn-2.pb")
            _vocal_path   = os.path.join(_models_dir, "voice_instrumental-discogs-effnet-1.pb")
            # MELODAY_EMB_ONLY (set by pre_analyze --sync-embeddings) loads ONLY the two embedding
            # extractors and skips av/vocal + the high-level heads — ~2 models instead of ~13, so
            # workers start faster and use ~half the RAM (more fit). WHY: the embedding backfill
            # needs only emb_effnet/emb_musicnn; the heads already exist on the tracks it targets,
            # so loading them is pure waste. Only the embedding files are required in this mode.
            _emb_only = bool(os.environ.get('MELODAY_EMB_ONLY'))
            _required_models = ([_effnet_path, _musicnn_path] if _emb_only
                                else [_effnet_path, _musicnn_path, _av_path, _vocal_path])
            if all(os.path.isfile(p) for p in _required_models):
                # Two embedding extractors (both expect 16 kHz mono):
                #   EffNet-Discogs → 1280-dim, feeds the voice/instrumental head.
                #   MusiCNN-MSD    →  200-dim, feeds the DEAM valence/arousal head.
                # The DEAM head is a *musicnn* model — feeding it EffNet's 1280-dim vectors
                # is what caused "Matrix size-incompatible [n,1280]x[200,100]" on every track.
                _effnet_model  = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=_effnet_path,  output="PartitionedCall:1")
                _musicnn_model = es.TensorflowPredictMusiCNN(
                    graphFilename=_musicnn_path, output="model/dense/BiasAdd")
                _TF_MODELS_LOADED = True   # the embeddings alone are enough to run the TF block

                if not _emb_only:
                    _av_model      = es.TensorflowPredict2D(
                        graphFilename=_av_path,      output="model/Identity:0")
                    _vocal_model   = es.TensorflowPredict2D(
                        graphFilename=_vocal_path,   output="model/Softmax")

                    # Optional high-level heads (downloaded into models_dir separately). Each runs
                    # on the embeddings already extracted above; missing files are skipped.
                    def _ld_head(_base, _out, _inp=None):
                        _p = os.path.join(_models_dir, _base + ".pb")
                        if not os.path.isfile(_p):
                            return None
                        try:  # per-head guard so one bad model can't skip the others
                            if _inp:
                                return es.TensorflowPredict2D(graphFilename=_p, input=_inp, output=_out)
                            return es.TensorflowPredict2D(graphFilename=_p, output=_out)
                        except Exception:
                            return None

                    def _ld_classes(_base):
                        try:
                            with open(os.path.join(_models_dir, _base + ".json")) as _jf:
                                return json.load(_jf).get("classes", [])
                        except Exception:
                            return []

                    for _field, _base, _emb, _idx in _MOOD_HEAD_SPECS:
                        _h = _ld_head(_base, "model/Softmax")
                        if _h is not None:
                            _mood_models[_field] = (_h, _emb, _idx)
                    _moodtheme_model   = _ld_head("mtg_jamendo_moodtheme-discogs-effnet-1", "model/Sigmoid")
                    _MOODTHEME_CLASSES = _ld_classes("mtg_jamendo_moodtheme-discogs-effnet-1")
                    _genre_model       = _ld_head("genre_discogs400-discogs-effnet-1", "PartitionedCall:0",
                                                  "serving_default_model_Placeholder")
                    _GENRE400_CLASSES  = _ld_classes("genre_discogs400-discogs-effnet-1")
    except Exception:
        pass

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
FONT_MAIN_PATH    = resolve_path(config["fonts"]["main"],    FONTS_DIR)
FONT_MELODAY_PATH = resolve_path(config["fonts"]["meloday"], FONTS_DIR)
FONT_LIGHT_PATH   = resolve_path(
    config.get("fonts", {}).get("light", config["fonts"]["main"]), FONTS_DIR
)

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
TITLE_PERIOD_NAMES = config.get("title_period_names", {})
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

                # Cap by available RAM — Essentia's MonoLoader decodes the full audio file
                # into a float32 array in memory. Peak per-worker usage (Python process base
                # + Essentia C++ buffers + decoded audio) is roughly 500 MB for typical tracks.
                # Reserve 15% of available RAM (min 1 GB) as headroom for the OS and other
                # processes that continue running during analysis.
                available_bytes = psutil.virtual_memory().available
                headroom = max(1024 ** 3, int(available_bytes * 0.15))
                ram_workers = max(1, (available_bytes - headroom) // (500 * 1024 * 1024))
                assigned = max(1, min(physical, ram_workers))
                tier_reason = (
                    f"CPU-bound | Physical cores: {physical} (psutil) | "
                    f"RAM: {available_bytes / 1024 ** 3:.1f} GB available → cap {ram_workers} workers"
                )
            except ImportError:
                # Approximate: assume 2 threads per core (x86 hyperthreading).
                # Slightly conservative for non-HT chips (ARM NAS, some AMD), but always safe.
                physical = max(1, logical // 2) if logical > 2 else logical
                assigned = max(1, physical)
                tier_reason = f"CPU-bound | Physical cores: {physical} (estimated)"

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
        _excluded_album_keys = set()
        for label in EXCLUDE_LABEL_NAMES:
            excluded_albums = music_library.search(libtype='album', label=label)
            _excluded_album_keys.update(str(a.ratingKey) for a in excluded_albums)
        m_print(f"[OK] Pre-fetched {len(_excluded_album_keys)} excluded albums across {len(EXCLUDE_LABEL_NAMES)} label(s).")
    except Exception as e:
        m_print(f"[WARN] Failed pre-fetching label exclusions: {e}")

# --- SQLite cache helpers ---

# All value columns, in the exact positional order _entry_to_row emits (rating_key prepended).
_CACHE_ALL_COLUMNS = (
    "bpm", "key", "energy", "year", "artist", "genres", "styles", "moods", "file_path",
    "track_updated_at", "album_updated_at", "artist_updated_at", "danceability", "brightness",
    "beat_confidence", "integrated_loudness", "onset_rate", "dynamic_complexity",
    "arousal", "valence", "vocal_presence",
    "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed", "mood_party", "mood_acoustic",
    "mood_electronic", "danceability_hl", "moodtheme", "genre_discogs", "emb_effnet", "emb_musicnn",
    "lastfm_artist_tags", "lastfm_track_tags", "artist_origin", "lastfm_listeners",
    "lyric_valence", "lyric_themes", "lyric_lang",
    "title", "release_date", "lastfm_synced_at", "geo_synced_at", "lyrics_synced_at",
    "lyric_themes_raw", "artist_mbid", "release_types",
)

# Per-process load profiles. WHY: the full 48-column load costs ~3.55 GB of RAM per process
# (measured: 164k rows; embeddings 978 MB, lyric_themes(+_raw) 1.09 GB, artist_origin 246 MB ...)
# and was the direct cause of prod OOM kills — yet each process READS only a subset:
#   core   — everything meloday.py's own run touches: DJ-order acoustics, tag resolution,
#            staleness timestamps, canonical_artist (artist_mbid), _entry_original_* (release_date),
#            PLUS the _fill_missing_acoustic presence-gate columns (integrated_loudness,
#            dynamic_complexity, mood_happy, moodtheme, genre_discogs) — dropping a gate column
#            would make every stale-path track re-decode audio to recompute "missing" TF heads.
#   extras — everything meloday_extras reads (embeddings, lyric/lastfm/origin enrichment, title,
#            release_types...) minus what it never touches: staleness timestamps, sync timestamps,
#            the mood_* heads (only ever written), artist_mbid, and dead-in-RAM lyric_themes_raw.
# Consumers all use entry.get(), so an unloaded column reads as None — same as an unpopulated one.
_CACHE_COLUMNS_CORE = (
    "bpm", "key", "energy", "year", "artist", "genres", "styles", "moods", "file_path",
    "track_updated_at", "album_updated_at", "artist_updated_at", "danceability", "brightness",
    "beat_confidence", "integrated_loudness", "onset_rate", "dynamic_complexity",
    "arousal", "valence", "vocal_presence",
    "mood_happy", "moodtheme", "genre_discogs", "release_date", "artist_mbid",
)
_CACHE_COLUMNS_EXTRAS = tuple(c for c in _CACHE_ALL_COLUMNS if c not in {
    "lyric_themes_raw",                                                # dead in RAM everywhere (SQL-only readers)
    "lastfm_synced_at", "geo_synced_at", "lyrics_synced_at",           # freshness checks are SQL-side (pre_analyze)
    "mood_happy", "mood_sad", "mood_aggressive", "mood_relaxed",       # TF heads: written by analysis,
    "mood_party", "mood_acoustic", "mood_electronic",                  #   never read by extras scoring
    "track_updated_at", "album_updated_at", "artist_updated_at",       # staleness is meloday/pre_analyze-only
    "artist_mbid",                                                     # extras uses the already-canonical artist
})
_CACHE_PROFILES = {"core": _CACHE_COLUMNS_CORE, "extras": _CACHE_COLUMNS_EXTRAS}

# INSERT ... ON CONFLICT DO UPDATE (NOT "INSERT OR REPLACE"). WHY: REPLACE deletes+reinserts the
# whole row, so every column absent from the in-memory entry would be written NULL. With profile
# loading, meloday's entries deliberately do NOT carry the extras/enrichment columns — a stale-path
# round-trip (loaded entry → metadata refresh → upsert, see analyze_track_essentia) under REPLACE
# would wipe embeddings/lyrics/Last.fm data for every stale track (the same bug class that once
# wiped ~67k embeddings via pre_analyze). Rule: columns in the CORE profile are always present in
# any entry meloday writes → plain assignment (fresh analysis wins); everything else is COALESCE'd
# (keep the stored value unless the incoming entry actually carries one). This also stops a
# stale-path write from clobbering a CONCURRENT pre_analyze sync's updates to those columns.
_UPSERT_SQL = (
    "INSERT INTO essentia_cache (rating_key, " + ", ".join(_CACHE_ALL_COLUMNS) + ")\n"
    "    VALUES (" + ", ".join("?" * (len(_CACHE_ALL_COLUMNS) + 1)) + ")\n"
    "    ON CONFLICT(rating_key) DO UPDATE SET\n      "
    + ",\n      ".join(
        (f"{c}=excluded.{c}" if c in _CACHE_COLUMNS_CORE
         else f"{c}=COALESCE(excluded.{c}, essentia_cache.{c})")
        for c in _CACHE_ALL_COLUMNS
    )
)

_CANONICAL_COLUMNS = (
    "rating_key TEXT PRIMARY KEY NOT NULL, "
    "bpm REAL, key TEXT, energy REAL, danceability REAL, brightness REAL, "
    "year INTEGER, artist TEXT, genres TEXT, styles TEXT, moods TEXT, "
    "file_path TEXT, track_updated_at REAL, album_updated_at REAL, artist_updated_at REAL, "
    "beat_confidence REAL, integrated_loudness REAL, onset_rate REAL, dynamic_complexity REAL, "
    "arousal REAL, valence REAL, vocal_presence REAL, "
    "mood_happy REAL, mood_sad REAL, mood_aggressive REAL, mood_relaxed REAL, "
    "mood_party REAL, mood_acoustic REAL, mood_electronic REAL, danceability_hl REAL, "
    "moodtheme TEXT, genre_discogs TEXT, emb_effnet BLOB, emb_musicnn BLOB, "
    "lastfm_artist_tags TEXT, lastfm_track_tags TEXT, artist_origin TEXT, "
    "lastfm_listeners INTEGER, lyric_valence REAL, lyric_themes TEXT, lyric_lang TEXT, "
    "title TEXT, release_date REAL, lastfm_synced_at REAL, geo_synced_at REAL, lyrics_synced_at REAL, "
    "lyric_themes_raw TEXT, artist_mbid TEXT, release_types TEXT"
)

# Per-process memo of DB file paths whose essentia_cache schema is already confirmed canonical.
# WHY: lets repeated connections in one process (load/save/upsert in meloday, every pre_analyze
# phase + worker) skip the ~40 failed ALTER TABLE attempts and the reorder PRAGMA scan after the
# first check. Safe: the schema is only ever mutated by this function, so a path that verified
# canonical stays canonical for the life of the process.
_SCHEMA_VERIFIED_PATHS = set()

def _ensure_db_schema(conn):
    """Creates or migrates the essentia_cache table to the canonical schema."""
    try:
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    except Exception:
        db_path = None
    # Fast path: this DB file's schema was already verified canonical this process.
    if db_path and db_path in _SCHEMA_VERIFIED_PATHS:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return

    conn.execute(f"CREATE TABLE IF NOT EXISTS essentia_cache ({_CANONICAL_COLUMNS})")
    conn.execute("PRAGMA journal_mode=WAL")

    # Step 1: Add any columns missing from older databases. Read existing columns ONCE and
    # ALTER only the absent ones — on a current DB this skips ~40 failed ALTER statements per
    # connection (the dominant per-connection cost once the schema is stable).
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(essentia_cache)").fetchall()}
    for col_name, col_def in (
        ("styles",               "styles TEXT"),
        ("danceability",         "danceability REAL"),
        ("brightness",           "brightness REAL"),
        ("track_updated_at",     "track_updated_at REAL"),
        ("album_updated_at",     "album_updated_at REAL"),
        ("artist_updated_at",    "artist_updated_at REAL"),
        ("beat_confidence",      "beat_confidence REAL"),
        ("integrated_loudness",  "integrated_loudness REAL"),
        ("onset_rate",           "onset_rate REAL"),
        ("dynamic_complexity",   "dynamic_complexity REAL"),
        ("arousal",              "arousal REAL"),
        ("valence",              "valence REAL"),
        ("vocal_presence",       "vocal_presence REAL"),
        ("mood_happy",           "mood_happy REAL"),
        ("mood_sad",             "mood_sad REAL"),
        ("mood_aggressive",      "mood_aggressive REAL"),
        ("mood_relaxed",         "mood_relaxed REAL"),
        ("mood_party",           "mood_party REAL"),
        ("mood_acoustic",        "mood_acoustic REAL"),
        ("mood_electronic",      "mood_electronic REAL"),
        ("danceability_hl",      "danceability_hl REAL"),
        ("moodtheme",            "moodtheme TEXT"),
        ("genre_discogs",        "genre_discogs TEXT"),
        ("emb_effnet",           "emb_effnet BLOB"),
        ("emb_musicnn",          "emb_musicnn BLOB"),
        ("lastfm_artist_tags",   "lastfm_artist_tags TEXT"),
        ("lastfm_track_tags",    "lastfm_track_tags TEXT"),
        ("artist_origin",        "artist_origin TEXT"),
        ("lastfm_listeners",     "lastfm_listeners INTEGER"),
        ("lyric_valence",        "lyric_valence REAL"),
        ("lyric_themes",         "lyric_themes TEXT"),
        ("lyric_lang",           "lyric_lang TEXT"),
        ("title",                "title TEXT"),
        ("release_date",         "release_date REAL"),
        ("lastfm_synced_at",     "lastfm_synced_at REAL"),
        ("geo_synced_at",        "geo_synced_at REAL"),
        ("lyrics_synced_at",     "lyrics_synced_at REAL"),
        ("lyric_themes_raw",     "lyric_themes_raw TEXT"),
        ("artist_mbid",          "artist_mbid TEXT"),
        ("release_types",        "release_types TEXT"),
    ):
        if col_name in existing_cols:
            continue  # already present — skip the doomed ALTER attempt
        try:
            conn.execute(f"ALTER TABLE essentia_cache ADD COLUMN {col_def}")
            m_print(f"[INFO] DB migration: added column '{col_name}' to essentia_cache.")
        except Exception:
            pass  # Column already exists (concurrent migration)

    # Step 2: Reorder table to canonical column sequence if needed.
    # Triggered when:
    #   - track_updated_at is missing (pre-migration DB still has last_synced)
    #   - last_synced still exists (needs to be removed from schema)
    #   - danceability is not immediately after energy (ALTER TABLE appended it at the tail)
    col_names = [row[1] for row in conn.execute("PRAGMA table_info(essentia_cache)").fetchall()]
    needs_reorder = (
        "track_updated_at" not in col_names or
        "last_synced" in col_names or
        ("danceability" in col_names and col_names.index("danceability") != col_names.index("energy") + 1) or
        "beat_confidence" not in col_names
    )
    reorder_ok = True
    if needs_reorder:
        m_print("[INFO] DB migration: reordering table to canonical schema...")
        try:
            db_path = conn.execute("PRAGMA database_list").fetchone()[2]
            # Use a timestamped backup name so each migration run gets its own backup,
            # regardless of whether a previous backup file already exists.
            import time as _time
            backup_path = db_path + f".pre_reorder_{int(_time.time())}.bak"
            if db_path:
                shutil.copy2(db_path, backup_path)
                m_print(f"[INFO] DB migration: database backed up to {os.path.basename(backup_path)}.")
        except Exception as backup_err:
            m_print(f"[WARN] DB migration: could not back up database: {backup_err}")
        try:
            # Carry EVERY canonical column through, in canonical order: copy it if the old table
            # had it, else default to NULL. track_updated_at falls back to the legacy last_synced
            # column. Deriving the column list from _CANONICAL_COLUMNS keeps this correct as the
            # schema grows — the previous hardcoded 22-column SELECT silently dropped every newer
            # column (moodtheme, genre_discogs, embeddings, lastfm, lyrics, …) on reorder and, once
            # the table reached 49 columns, failed outright ("49 columns but 22 values supplied").
            old_cols = set(col_names)
            canonical_names = [part.strip().split()[0] for part in _CANONICAL_COLUMNS.split(",")]
            select_exprs = []
            for c in canonical_names:
                if c == "track_updated_at" and "last_synced" in old_cols:
                    select_exprs.append(f"last_synced AS {c}")
                elif c in old_cols:
                    select_exprs.append(c)
                else:
                    select_exprs.append(f"NULL AS {c}")
            conn.execute("DROP TABLE IF EXISTS essentia_cache_reordered")
            conn.execute(f"CREATE TABLE essentia_cache_reordered ({_CANONICAL_COLUMNS})")
            conn.execute(
                f"INSERT INTO essentia_cache_reordered ({', '.join(canonical_names)}) "
                f"SELECT {', '.join(select_exprs)} FROM essentia_cache"
            )
            conn.commit()
            conn.execute("DROP TABLE essentia_cache")
            conn.execute("ALTER TABLE essentia_cache_reordered RENAME TO essentia_cache")
            conn.commit()
            m_print("[INFO] DB migration: schema reorder complete.")
        except Exception as reorder_err:
            reorder_ok = False
            m_print(f"[ERROR] DB migration: schema reorder failed: {reorder_err}")

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Memoise only when the schema is genuinely canonical now (no reorder needed, or it
    # succeeded), so a failed migration is retried on the next connection rather than skipped.
    if db_path and reorder_ok:
        _SCHEMA_VERIFIED_PATHS.add(db_path)

def _entry_to_row(rk, d):
    return (rk, d.get("bpm"), d.get("key"), d.get("energy"), d.get("year"),
            d.get("artist"), json.dumps(d.get("genres") or []),
            json.dumps(d.get("styles") or []),
            json.dumps(d.get("moods") or []), d.get("file_path"),
            d.get("track_updated_at"), d.get("album_updated_at"), d.get("artist_updated_at"),
            d.get("danceability"), d.get("brightness"),
            d.get("beat_confidence"), d.get("integrated_loudness"),
            d.get("onset_rate"), d.get("dynamic_complexity"),
            d.get("arousal"), d.get("valence"), d.get("vocal_presence"),
            d.get("mood_happy"), d.get("mood_sad"), d.get("mood_aggressive"),
            d.get("mood_relaxed"), d.get("mood_party"), d.get("mood_acoustic"),
            d.get("mood_electronic"), d.get("danceability_hl"),
            json.dumps(d["moodtheme"]) if d.get("moodtheme") is not None else None,
            json.dumps(d["genre_discogs"]) if d.get("genre_discogs") is not None else None,
            d.get("emb_effnet"), d.get("emb_musicnn"),
            json.dumps(d["lastfm_artist_tags"]) if d.get("lastfm_artist_tags") is not None else None,
            json.dumps(d["lastfm_track_tags"]) if d.get("lastfm_track_tags") is not None else None,
            json.dumps(d["artist_origin"]) if d.get("artist_origin") is not None else None,
            d.get("lastfm_listeners"), d.get("lyric_valence"),
            # extras-profile entries carry lyric_themes as its raw JSON TEXT (lazy parse) — pass a
            # str through unchanged so a round-trip can't double-encode it. Defensive: extras never
            # upserts cache entries, and the core profile never loads this column.
            (d["lyric_themes"] if isinstance(d.get("lyric_themes"), str)
             else json.dumps(d["lyric_themes"]) if d.get("lyric_themes") is not None else None),
            d.get("lyric_lang"), d.get("title"), d.get("release_date"),
            d.get("lastfm_synced_at"), d.get("geo_synced_at"), d.get("lyrics_synced_at"),
            json.dumps(d["lyric_themes_raw"]) if d.get("lyric_themes_raw") is not None else None,
            d.get("artist_mbid"),
            json.dumps(d["release_types"]) if d.get("release_types") is not None else None)

# JSON columns and their empty-value defaults: list-tags load as [] (callers iterate them
# directly), dict/enrichment columns load as None (callers guard with `or {}` / is-None checks).
_CACHE_LIST_COLUMNS = frozenset({"genres", "styles", "moods"})
# Small-vocabulary JSON columns: the same TEXT recurs across thousands of rows (measured:
# artist_origin 4,342 distinct / lastfm_artist_tags 5,331 / styles ~6k combos over 164k+ rows),
# so identical texts share ONE parsed object via _memo_json — this alone was ~600 MB of duplicate
# parsed objects. SAFE because no consumer mutates a loaded tag list/dict in place — only
# whole-key rebinds (verified across meloday + extras + pre_analyze); any future in-place
# mutation of an entry's tag fields would corrupt every sharer, so don't.
_CACHE_MEMO_JSON_COLUMNS = frozenset({
    "lastfm_artist_tags", "lastfm_track_tags", "artist_origin", "release_types",
})
# Per-track float-dict columns (whole texts ~unique — per-track confidences — so whole-object
# memoing can't help), but 98-99% of their KEY strings are duplicates ("Rock---Indie Rock",
# moodtheme tags): parse fresh, intern the keys.
_CACHE_FLOATDICT_COLUMNS = frozenset({"moodtheme", "genre_discogs"})
# lyric_themes is deliberately NOT parsed at load (extras profile only): 497 MB parsed vs 96 MB
# raw text, and it has a single consumer (_lyric_boost) — extras parses lazily on first touch
# via _entry_lyric_themes. lyric_themes_raw is loaded by no profile; the branch stays for safety.
_CACHE_JSON_COLUMNS = frozenset({"lyric_themes_raw"})
# Scalar columns with heavy duplication (artist ~4.5k distinct × ~37 copies; musical key ~24
# values; lyric_lang a few dozen codes).
_CACHE_INTERN_COLUMNS = frozenset({"artist", "key", "lyric_lang"})

_EMPTY_LIST = []       # shared []-default for the list columns (same no-mutation contract)
_JSON_PARSE_MEMO = {}  # raw JSON text -> parsed object; cleared after each load (keys hold the texts)

def _memo_json(text):
    try:
        return _JSON_PARSE_MEMO[text]
    except KeyError:
        v = _JSON_PARSE_MEMO[text] = json.loads(text)
        return v

def _row_to_entry(row, cols):
    """Builds a cache entry {col: value} from a (rating_key, *cols) row. Tuple-driven so the
    same code path serves every load profile; entry keys are the shared module-level column-name
    strings, so 164k+ entries don't duplicate key objects."""
    entry = {}
    for name, val in zip(cols, row[1:]):
        if name in _CACHE_MEMO_JSON_COLUMNS:
            entry[name] = _memo_json(val) if val else None
        elif name in _CACHE_LIST_COLUMNS:
            entry[name] = _memo_json(val) if val else _EMPTY_LIST
        elif name in _CACHE_FLOATDICT_COLUMNS:
            if val:
                gd = json.loads(val)
                # legacy rows may hold a plain list — pass through untouched (consumers handle both)
                entry[name] = ({sys.intern(k): v for k, v in gd.items()}
                               if isinstance(gd, dict) else gd)
            else:
                entry[name] = None
        elif name in _CACHE_JSON_COLUMNS:
            entry[name] = json.loads(val) if val else None
        elif name in _CACHE_INTERN_COLUMNS and isinstance(val, str):
            entry[name] = sys.intern(val)
        else:
            entry[name] = val
    return row[0], entry

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

def load_essentia_cache(profile="core"):
    """Loads the cache into meloday._essentia_cache, selecting ONLY the columns the calling
    process reads (see _CACHE_PROFILES). WHY: the full-column load cost ~3.55 GB per process and
    caused prod OOM kills; the core profile drops the ~2.5 GB of extras-only enrichment
    (embeddings, lyric/lastfm/origin data) that meloday.py never reads. Rows are streamed
    (no fetchall) so the raw row tuples never co-reside with the built dict — the old
    fetchall-then-build pattern transiently doubled peak RSS."""
    global _essentia_cache
    _migrate_json_to_sqlite()
    if not os.path.exists(ESSENTIA_CACHE_PATH):
        _essentia_cache = {}
        return
    cols = _CACHE_PROFILES[profile]
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=10)
        _ensure_db_schema(conn)
        # WHY: full-table scan of a ~1.5 GB db — mmap reads via the OS page cache
        # without a second userspace copy (roughly halves the cold-scan time).
        conn.execute("PRAGMA mmap_size=2147483648")
        cache = {}
        for row in conn.execute("SELECT rating_key, " + ", ".join(cols) + " FROM essentia_cache"):
            rk, entry = _row_to_entry(row, cols)
            cache[rk] = entry
        conn.close()
        # The memo's KEYS are the raw JSON texts — dead weight once loading ends (the parsed
        # values stay alive via the entries that share them). Drop them; a reload re-seeds.
        _JSON_PARSE_MEMO.clear()
        _essentia_cache = cache
    except Exception as e:
        m_print(f"[WARN] Could not load Essentia cache: {e}")
        _essentia_cache = {}

def _upsert_cache_entries(entries):
    """Writes only the specified {rk: data} subset to SQLite, without touching the rest of the cache."""
    if not entries:
        return
    os.makedirs(os.path.dirname(ESSENTIA_CACHE_PATH), exist_ok=True)
    try:
        conn = sqlite3.connect(ESSENTIA_CACHE_PATH, timeout=120)
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

def _fill_missing_acoustic(data, file_path, audio=None, track_title=""):
    """Compute any missing acoustic fields in data in-place.
    Loads audio from file_path only if at least one field needs computing.
    If audio is already loaded (full analysis path), pass it to avoid a second decode.
    Only runs the algorithms required for the missing fields."""
    needs_bpm        = data.get("bpm") is None
    needs_beat_conf  = data.get("beat_confidence") is None
    needs_key        = data.get("key") is None
    needs_energy     = data.get("energy") is None
    needs_int_loud   = data.get("integrated_loudness") is None
    needs_dance      = data.get("danceability") is None
    needs_brightness = data.get("brightness") is None
    needs_onset      = data.get("onset_rate") is None
    needs_dyn_comp   = data.get("dynamic_complexity") is None
    needs_tf         = (data.get("arousal") is None or data.get("valence") is None
                        or data.get("vocal_presence") is None
                        # backfill the high-level heads on existing tracks (only if the models
                        # are loaded — otherwise these stay null and don't force re-analysis)
                        or (bool(_mood_models) and data.get("mood_happy") is None)
                        or (_moodtheme_model is not None and data.get("moodtheme") is None)
                        or (_genre_model is not None and data.get("genre_discogs") is None)
                        # --sync-embeddings backfill: re-run the TF block for tracks that have the other TF
                        # fields but never had their embeddings stored (audit Tier-4). Gate on the embedding
                        # model, NOT _mood_models — emb-only mode loads no heads, so a _mood_models gate would
                        # make needs_tf False and skip the very backfill this flag requests.
                        or (_FORCE_EMB_BACKFILL and _effnet_model is not None and data.get("emb_effnet") is None))

    needs_base = any([needs_bpm, needs_beat_conf, needs_key, needs_energy, needs_int_loud,
                      needs_dance, needs_brightness, needs_onset, needs_dyn_comp])
    if not (needs_base or (needs_tf and _TF_MODELS_LOADED)):
        return

    # Base features decode at the native 44.1 kHz. The TF block decodes its own 16 kHz copy,
    # so when only TF fields are missing (the post-pass case) we skip this decode entirely.
    if audio is None and needs_base:
        audio = es.MonoLoader(filename=file_path)()

    # Each algorithm block is isolated in its own try/except so a failure in one
    # field does not prevent the remaining fields from being computed.

    # BPM + beat confidence share the same RhythmExtractor2013 call
    if needs_bpm or needs_beat_conf:
        try:
            rhythm_result = es.RhythmExtractor2013(method="multifeature")(audio)
            bpm        = rhythm_result[0]
            confidence = rhythm_result[2]
            if needs_bpm:
                if bpm >= 250:
                    bpm_retry = es.RhythmExtractor2013(method="degara")(audio)[0]
                    bpm = bpm_retry if bpm_retry < 250 else None
                    if bpm is None and track_title:
                        log_text(f"[WARN] BPM >= 250 for '{track_title}' after retry — storing as null (excluded from weight calculations).")
                data["bpm"] = round(bpm, 2) if bpm is not None else None
            if needs_beat_conf:
                data["beat_confidence"] = round(float(confidence), 4)
        except Exception as e:
            if track_title:
                log_text(f"[WARN] BPM/beat_confidence failed for '{track_title}': {e}")

    if needs_key:
        try:
            key_alg = es.KeyExtractor()(audio)
            data["key"] = CAMELOT_MAP.get(f"{key_alg[0]} {key_alg[1]}", "0A")
        except Exception:
            pass

    # energy (integratedLoudness, LUFS) and integrated_loudness (loudnessRange, LU) both come
    # from a single LoudnessEBUR128 call. LoudnessEBUR128 output tuple:
    #   [0] momentaryLoudness  — vector_real (one value per 400 ms block)
    #   [1] shortTermLoudness  — vector_real (one value per 3 s block)
    #   [2] integratedLoudness — real scalar, LUFS  → stored as "energy"
    #   [3] loudnessRange      — real scalar, LU    → stored as "integrated_loudness"
    # numpy column_stack creates the (N, 2) stereo array LoudnessEBUR128 expects — avoids
    # a dependency on es.StereoMuxer which is a streaming-only algorithm in some builds.
    if needs_energy or needs_int_loud:
        try:
            import numpy as _np
            audio_stereo = _np.column_stack([audio, audio])
            ebur = es.LoudnessEBUR128()(audio_stereo)
            if needs_energy:
                data["energy"] = round(float(ebur[2]), 2)
            if needs_int_loud:
                data["integrated_loudness"] = round(float(ebur[3]), 4)
        except Exception as e:
            if track_title:
                log_text(f"[WARN] LoudnessEBUR128 failed for '{track_title}': {e}")

    if needs_dance:
        try:
            data["danceability"] = round(min(float(es.Danceability()(audio)[0]) / 3.0, 1.0), 4)
        except Exception:
            pass

    if needs_brightness:
        try:
            _w = es.Windowing(type='hann')
            _spec = es.Spectrum()
            _centroid = es.Centroid(range=1.0)
            brightness_frames = [
                _centroid(_spec(_w(frame)))
                for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024, startFromZero=True)
            ]
            data["brightness"] = round(float(sum(brightness_frames) / len(brightness_frames)), 4) if brightness_frames else 0.0
        except Exception:
            pass

    if needs_onset:
        try:
            # OnsetRate returns (onsets_times_array, onset_rate_scalar) — index [1] is the rate.
            data["onset_rate"] = round(float(es.OnsetRate()(audio)[1]), 4)
        except Exception as e:
            if track_title:
                log_text(f"[WARN] OnsetRate failed for '{track_title}': {e}")

    if needs_dyn_comp:
        try:
            data["dynamic_complexity"] = round(float(es.DynamicComplexity()(audio)[0]), 4)
        except Exception:
            pass

    if needs_tf and _TF_MODELS_LOADED:
        try:
            import numpy as _np
            # The TF embedding networks require 16 kHz mono — the `audio` used for the base
            # features is 44.1 kHz (wrong rate for these models), so decode a dedicated copy.
            audio_tf    = es.MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()
            emb_effnet  = _np.asarray(_effnet_model(audio_tf))    # (n, 1280) → vocal head
            emb_musicnn = _np.asarray(_musicnn_model(audio_tf))   # (n,  200) → DEAM head
            # DEAM emits valence then arousal on the 1–9 annotation scale → normalise to 0–1.
            # voice_instrumental classes are [instrumental, voice] → vocal_presence = P(voice).
            # WHY guarded: run each model only if it's loaded AND its field is missing — skips the
            # wasted av/vocal passes on tracks that already have them (the --sync-embeddings backfill
            # targets exactly such tracks) and avoids touching the unloaded models in emb-only mode.
            if _av_model is not None and (data.get("valence") is None or data.get("arousal") is None):
                av = _np.asarray(_av_model(emb_musicnn))             # (n, 2): [valence, arousal], 1–9 scale
                if data.get("valence") is None:
                    data["valence"]        = round((float(av[:, 0].mean()) - 1.0) / 8.0, 4)
                if data.get("arousal") is None:
                    data["arousal"]        = round((float(av[:, 1].mean()) - 1.0) / 8.0, 4)
            if _vocal_model is not None and data.get("vocal_presence") is None:
                voc = _np.asarray(_vocal_model(emb_effnet))          # (n, 2): softmax [instrumental, voice]
                data["vocal_presence"] = round(float(voc[:, 1].mean()), 4)

            # High-level heads on the SAME embeddings; each filled only if currently missing.
            for _field, (_head, _emb_key, _idx) in _mood_models.items():
                if data.get(_field) is None:
                    _e = emb_musicnn if _emb_key == "musicnn" else emb_effnet
                    data[_field] = round(float(_np.asarray(_head(_e))[:, _idx].mean()), 4)
            if _moodtheme_model is not None and _MOODTHEME_CLASSES and data.get("moodtheme") is None:
                _mt = _np.asarray(_moodtheme_model(emb_effnet)).mean(axis=0)
                data["moodtheme"] = {
                    _MOODTHEME_CLASSES[_i]: round(float(_p), 3)
                    for _i, _p in sorted(enumerate(_mt), key=lambda kv: kv[1], reverse=True)[:8]
                    if float(_p) >= 0.05
                }
            if _genre_model is not None and _GENRE400_CLASSES and data.get("genre_discogs") is None:
                _gp = _np.asarray(_genre_model(emb_effnet)).mean(axis=0)
                data["genre_discogs"] = {
                    _GENRE400_CLASSES[_i]: round(float(_p), 3)
                    for _i, _p in sorted(enumerate(_gp), key=lambda kv: kv[1], reverse=True)[:6]
                    if float(_p) >= 0.03
                }
            # Cache the mean embeddings so any future head needs no audio re-read.
            # WHY no _mood_models gate: emb-only mode (--sync-embeddings) loads no heads, but must
            # still persist the vectors it was launched to backfill. emb_effnet/emb_musicnn are
            # always computed above whenever this TF block runs.
            # WHY the shape/finite guard: a file that fails the 16 kHz decode yields an empty audio_tf, so
            # the models emit an empty array; mean() of that is a NaN scalar → a 4-byte blob that later
            # crashes np.dot in the extras "sounds-like" scoring ((1,) vs (200/1280,)). Store only a
            # full-size, all-finite vector; else leave NULL — a missing embedding degrades gracefully
            # everywhere, a corrupt one does not. isfinite also rejects the (0,1280)→NaN-vector variant.
            if data.get("emb_effnet") is None:
                _ef = _np.atleast_2d(emb_effnet).mean(axis=0)
                _em = _np.atleast_2d(emb_musicnn).mean(axis=0)
                if (_ef.size == 1280 and _em.size == 200
                        and _np.all(_np.isfinite(_ef)) and _np.all(_np.isfinite(_em))):
                    data["emb_effnet"]  = _ef.astype(_np.float32).tobytes()
                    data["emb_musicnn"] = _em.astype(_np.float32).tobytes()
        except Exception as tf_err:
            log_text(f"[WARN] TF inference failed: {tf_err}")


_ORIG_DATE_RE = re.compile(r"\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?")
def _parse_original_ts(values):
    """First parseable original-release date among `values` (tag strings like '1975', '1975-08',
    '1975-08-01', a MB 'first-release-date', or a padded '1965-00-00') → unix timestamp with FULL
    day precision when present, else Jan 1 of the year. Rejects implausible years; None if nothing
    parses. Pre-1970 dates yield NEGATIVE timestamps (fine on Linux/macOS)."""
    for v in (values or []):
        m = _ORIG_DATE_RE.match(str(v))
        if not m:
            continue
        y = int(m.group(1))
        if y < 1000 or y > 2100:
            continue
        mo = int(m.group(2) or 1) or 1
        d  = int(m.group(3) or 1) or 1
        if not (1 <= mo <= 12):
            mo = 1
        for dd in (d if 1 <= d <= 31 else 1, 1):   # bad day (Feb 30 etc.) → fall back to the 1st
            try:
                return datetime(y, mo, dd).timestamp()
            except Exception:
                continue
    return None

def _entry_original_date(entry):
    """The track's original-release date (FULL precision) from the cache `release_date`, else None."""
    rd = entry.get("release_date")
    if rd:
        try:
            return datetime.fromtimestamp(rd).date()
        except Exception:
            pass
    return None

def _entry_original_year(entry):
    """Original-release YEAR: prefer the file-tag/MB `release_date`, soft-fallback to Plex `year`."""
    d = _entry_original_date(entry)
    return d.year if d else entry.get("year")

def _read_mb_file_tags(file_path):
    """Read the MusicBrainz tags Picard embeds in the audio file: the release-group type list (for
    compilation detection) and the artist MBID (for an exact geo lookup). Authoritative + local — no API.
    Handles FLAC/Vorbis (`releasetype`/`musicbrainz_artistid`), MP4/M4A (freeform `MusicBrainz Album Type`/
    `MusicBrainz Artist Id`) and MP3/ID3 (`TXXX:…`). Returns {"artist_mbid", "release_types": [lowercased]};
    empties on a tagless/unreadable file (never raises)."""
    try:
        import mutagen
        from mutagen.mp4 import MP4FreeForm
    except Exception:
        return {"artist_mbid": None, "release_types": [], "_tags_read": False}
    try:
        m = mutagen.File(file_path)
    except Exception:
        m = None
    if m is None or getattr(m, "tags", None) is None:
        return {"artist_mbid": None, "release_types": [], "_tags_read": False}
    tags   = m.tags
    keymap = {k.lower(): k for k in tags.keys()}

    def _norm(v):
        out = []
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if isinstance(x, MP4FreeForm):
                x = bytes(x).decode("utf-8", "ignore")
            elif isinstance(x, bytes):
                x = x.decode("utf-8", "ignore")
            out.append(str(x).strip().lower())
        return [s for s in out if s]

    def _get(*names):
        for n in names:
            k = keymap.get(n.lower())
            if k is not None:
                v = tags[k]
                if hasattr(v, "text"):            # ID3 frame -> .text list
                    v = list(v.text)
                return _norm(v)
        return []

    rtypes = _get("releasetype", "----:com.apple.iTunes:MusicBrainz Album Type",
                  "TXXX:MusicBrainz Album Type", "TXXX:RELEASETYPE", "musicbrainz_albumtype")
    ambid  = _get("musicbrainz_artistid", "----:com.apple.iTunes:MusicBrainz Artist Id",
                  "TXXX:MusicBrainz Artist Id")
    # ORIGINAL-release date (Picard) — full date where present, most-specific tag first. NOT plain
    # date/©day/TDRC (those are the edition/reissue date). release-group MBID is returned for the
    # MusicBrainz first-release-date fallback (sync_original_release_dates) — it is transient, not a
    # cache column, so the upsert ignores it.
    odate = _get("originaldate", "tdor", "TXXX:originaldate", "----:com.apple.iTunes:originaldate",
                 "originalyear", "tory", "TXXX:originalyear", "----:com.apple.iTunes:originalyear")
    rgid  = _get("musicbrainz_releasegroupid", "----:com.apple.iTunes:MusicBrainz Release Group Id",
                 "TXXX:MusicBrainz Release Group Id")
    out = {"artist_mbid": (ambid[0] if ambid else None), "release_types": rtypes,
           "release_group_mbid": (rgid[0] if rgid else None), "_tags_read": True}
    ots = _parse_original_ts(odate)
    if ots is not None:
        out["release_date"] = ots        # added ONLY when a tag parsed → data.update() never clobbers with None
    return out


def analyze_track_essentia(track, plex_track_ts=None, plex_album_ts=None, plex_artist_ts=None):
    rk = str(track.ratingKey)

    # Re-silence inside sub-processes to handle parallel worker logs
    if ESSENTIA_AVAILABLE:
        essentia.log.infoActive = False
        essentia.log.warningActive = False

    def _timestamps_changed(data):
        """True if any Plex-sourced timestamp differs from the cached value."""
        return (
            data.get("track_updated_at")  != plex_track_ts  or
            data.get("album_updated_at")  != plex_album_ts  or
            data.get("artist_updated_at") != plex_artist_ts
        )

    def _apply_timestamps(data):
        """Write the three Plex timestamps into a cache entry after a full metadata sync."""
        data["track_updated_at"]  = plex_track_ts
        data["album_updated_at"]  = plex_album_ts
        data["artist_updated_at"] = plex_artist_ts

    # Metadata Update Handling
    # If the track is already in the cache, we refresh the text fields from Plex
    # to ensure changes to genres/moods/artist names are captured.
    if rk in _essentia_cache and _essentia_cache[rk].get("energy") is not None:
        data = _essentia_cache[rk]
        file_path = get_local_path(track)

        # Re-sync only if Plex reports a change at the track, album, or artist level.
        # If all timestamps match (or are unavailable), skip metadata re-sync and
        # only backfill any missing acoustic/text fields — without touching timestamps.
        if data.get("file_path") == file_path and not _timestamps_changed(data):
            # Backfill any text tags missing from older cache entries.
            # "styles" not in data won't catch entries where styles=[] (loaded from NULL),
            # so check for falsy instead. Same applies to genres — older code used
            # track.genres directly without the album/artist fallback.
            if not data.get("styles"):
                data["styles"] = _resolve_tags(track, "styles")
            if not data.get("genres"):
                data["genres"] = _resolve_tags(track, "genres")
            # Backfill any missing acoustic fields without re-running the full pipeline.
            # Timestamps are intentionally not updated — no Plex metadata was re-synced.
            if ESSENTIA_ENABLED and file_path:
                try:
                    _fill_missing_acoustic(data, file_path)
                except Exception:
                    pass
            return data

        data["artist"] = canonical_artist(track, data)
        data["genres"] = _resolve_tags(track, "genres")
        data["styles"] = _resolve_tags(track, "styles")
        data["moods"]  = _resolve_tags(track, "moods")
        # Update year if it was previously null
        if data.get("year") is None:
            data["year"] = getattr(track, "year", None) or album_meta(track).get("year")

        # Same path but timestamps changed: sync text metadata, backfill missing acoustics,
        # then record the new Plex timestamps.
        if data.get("file_path") == file_path:
            if ESSENTIA_ENABLED:
                try:
                    _fill_missing_acoustic(data, file_path)
                except Exception:
                    pass
            _apply_timestamps(data)
            return data
        # If path changed, fall through to perform new acoustic analysis

    if not ESSENTIA_ENABLED:
        # Populate metadata-only entry (styles/moods/genres/year from Plex; acoustics remain null).
        # This lets pre_analyze.py and the optimizer work for users without Essentia.
        data = {
            "bpm": None, "key": None, "energy": None,
            "danceability": None, "brightness": None,
            "year": getattr(track, "year", None) or album_meta(track).get("year"),
            "artist": canonical_artist(track),
            "genres": _resolve_tags(track, "genres"),
            "styles": _resolve_tags(track, "styles"),
            "moods":  _resolve_tags(track, "moods"),
            "file_path": get_local_path(track),
            "track_updated_at": plex_track_ts,
            "album_updated_at": plex_album_ts,
            "artist_updated_at": plex_artist_ts,
            "beat_confidence": None, "integrated_loudness": None,
            "onset_rate": None, "dynamic_complexity": None,
            "arousal": None, "valence": None, "vocal_presence": None,
        }
        data.update(_read_mb_file_tags(data.get("file_path")))
        data["artist"] = canonical_artist(track, data)   # primary artist via MBID->MusicBrainz map
        _essentia_cache[rk] = data
        return data

    file_path = get_local_path(track)
    if not file_path: return None

    # RhythmExtractor2013 overflows its onset detection buffer on very long files
    # (DJ mixes, continuous albums). Skip acoustic analysis for tracks over 30 minutes
    # and cache metadata-only so the track can still appear in playlists.
    duration_ms = getattr(track, "duration", 0) or 0
    if duration_ms > 1_800_000:
        track_year = getattr(track, "year", None)
        if not track_year:
            track_year = album_meta(track).get("year")
        data = {
            "bpm": None, "key": None, "energy": None,
            "danceability": None, "brightness": None,
            "year": track_year,
            "artist": canonical_artist(track),
            "genres": _resolve_tags(track, "genres"),
            "styles": _resolve_tags(track, "styles"),
            "moods":  _resolve_tags(track, "moods"),
            "file_path": file_path,
            "track_updated_at": plex_track_ts,
            "album_updated_at": plex_album_ts,
            "artist_updated_at": plex_artist_ts,
            "beat_confidence": None, "integrated_loudness": None,
            "onset_rate": None, "dynamic_complexity": None,
            "arousal": None, "valence": None, "vocal_presence": None,
        }
        log_text(f"[DIAGNOSTIC] Skipping Essentia for '{track.title}' ({duration_ms // 60000}m — too long for RhythmExtractor).")
        data.update(_read_mb_file_tags(file_path))
        data["artist"] = canonical_artist(track, data)   # primary artist via MBID->MusicBrainz map
        _essentia_cache[rk] = data
        return data

    try:
        loader = es.MonoLoader(filename=file_path)
        audio = loader()

        # Year Fallback: Check track first, then album
        track_year = getattr(track, "year", None)
        if not track_year:
            track_year = album_meta(track).get("year")

        # Build entry with all non-acoustic fields, then compute acoustics via the shared helper.
        # Passing audio avoids a second MonoLoader decode — all algorithms reuse the same buffer.
        data = {
            "bpm": None, "key": None, "energy": None,
            "danceability": None, "brightness": None,
            "year": track_year,
            "artist": canonical_artist(track),
            "genres": _resolve_tags(track, "genres"),
            "styles": _resolve_tags(track, "styles"),
            "moods":  _resolve_tags(track, "moods"),
            "file_path": file_path,
            "track_updated_at": plex_track_ts,
            "album_updated_at": plex_album_ts,
            "artist_updated_at": plex_artist_ts,
            "beat_confidence": None, "integrated_loudness": None,
            "onset_rate": None, "dynamic_complexity": None,
            "arousal": None, "valence": None, "vocal_presence": None,
        }
        _fill_missing_acoustic(data, file_path, audio=audio, track_title=track.title)
        data.update(_read_mb_file_tags(file_path))
        data["artist"] = canonical_artist(track, data)   # primary artist via MBID->MusicBrainz map
        _essentia_cache[rk] = data
        return data
    except Exception as e:
        m_print(f"[DIAGNOSTIC] Essentia failed for '{track.title}': {e}")
        return None

# Worker for Multiprocessing compatibility
def analysis_worker(track_id, plex_track_ts=None, plex_album_ts=None, plex_artist_ts=None):
    try:
        # Silence Essentia explicitly in parallel worker processes
        if ESSENTIA_AVAILABLE:
            essentia.log.infoActive = False
            essentia.log.warningActive = False

        # Process-Safe Plex Connection
        # Initialize a new local connection session for process isolation.
        local_plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=120)
        track = local_plex.fetchItem(track_id)
        return str(track_id), analyze_track_essentia(track, plex_track_ts, plex_album_ts, plex_artist_ts)
    except Exception:
        return str(track_id), None

def get_bpm_distance(bpm1, bpm2):
    if not bpm1 or not bpm2: return 0.5
    diffs = [abs(bpm1 - bpm2), abs(bpm1 - bpm2 * 2), abs(bpm1 * 2 - bpm2)]
    return min(min(diffs) / 20.0, 1.0)

def get_harmonic_distance(key1, key2):
    if not key1 or not key2 or key1 == "0A" or key2 == "0A": return 0.5
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
    """Apply pre-fetched seasonal and label-based exclusions."""
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

        # Check pre-fetched excluded label set (memory check)
        if parent_key in _excluded_album_keys:
            log_text(f"[EXCLUSION] Track '{t.title}' skipped: album has an exclusion label.")
            continue

        # Length cap — exclude anything over 15 minutes
        if (getattr(t, "duration", 0) or 0) > MAX_TRACK_MS:
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
# The feat/ft markers MUST be word-bounded (\b): the old leading `\s*` let "ft" match INSIDE a word, so
# "Daft Punk" -> "Da", "Shift K3Y" -> "Shi", "Soft Cell" -> "So" (and "Soft Cell"/"SOFT PLAY" collided on
# "so"). `\b` mirrors clean_title's already-correct `\bft\.?\s+`. The ` & `/` x ` collab split is kept
# (it correctly reduces "A feat./& B" credits to the primary for dedup); band/duo names like
# "Simon & Garfunkel" are instead kept whole upstream via the artist_mbid canonical name (canonical_artist).
_FEAT_SPLIT_RE = re.compile(r"\b(?:feat\.?|ft\.?|featuring)\s+.*$|\s+[&x]\s+.*$", re.IGNORECASE)

@functools.lru_cache(maxsize=None)
def primary_artist(name: str) -> str:
    """Return the primary artist portion (strip 'feat./ft./featuring/& ...' collab suffixes)."""
    if not name:
        return ""
    s = name.strip()
    s = _FEAT_SPLIT_RE.sub("", s)
    # Normalize whitespace and case for comparisons
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- Cache artist key ---
# The cache `artist` is the track's PRIMARY artist, resolved from the file's MusicBrainz artist-ID via a
# `mbid -> MusicBrainz name` map (assets/mbid_artist_names.json). The MBID is the recording/track artist
# (first credit) and MusicBrainz names it authoritatively, so a real GROUP stays whole ("Mumford & Sons",
# "Simon & Garfunkel") while a COLLABORATION collapses to its primary ("David Bowie & BT" -> "David Bowie";
# "Max Berlin & Joakim" -> "Max Berlin"), and on compilations (incl. curator / DJ-mix comps where the album
# artist is the curator) the per-track performer is used, not the curator. CLASSICAL is the exception: the
# files tag the COMPOSER as the recording artist, but the user organises classical by PERFORMER, so those
# tracks keep the track_artist_name value instead. Fallback (no/unmapped MBID, ~2%, or classical):
# track_artist_name, feat-stripped, "&" kept so a band/duo on its own album stays whole.
MBID_ARTIST_MAP_PATH = os.path.join(BASE_DIR, "assets", "mbid_artist_names.json")
LASTFM_TOP_TRACKS_PATH = os.path.join(BASE_DIR, "assets", "lastfm_artist_top_tracks.json")

def _load_mbid_artist_map():
    try:
        with open(MBID_ARTIST_MAP_PATH, "r", encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if k and v}
    except (FileNotFoundError, ValueError, OSError):
        return {}

_MBID_ARTIST_MAP = _load_mbid_artist_map()
_FEAT_SUFFIX_RE = re.compile(r"\b(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)

def _is_classical(data) -> bool:
    """Classical tracks tag the COMPOSER as the recording artist; we keep the performer for them instead."""
    if not data:
        return False
    tags = [str(t).lower() for t in (data.get("genres") or []) + (data.get("styles") or [])]
    if any("classical" in t or "opera" in t for t in tags):
        return True
    gd = data.get("genre_discogs")
    return isinstance(gd, dict) and any(str(k).split("---")[0].strip().lower() == "classical" for k in gd)

def canonical_artist(track, data=None) -> str:
    """Normalised cache `artist` key: the PRIMARY artist via the artist_mbid -> MusicBrainz-name map (groups
    kept whole, collaborations -> primary, comp tracks -> performer). Classical keeps the performer; a missing
    or unmapped MBID falls back to track_artist_name (feat-stripped, "&" kept)."""
    if data and not _is_classical(data):
        nm = _MBID_ARTIST_MAP.get(data.get("artist_mbid"))
        if nm:
            return norm_text(nm)
    return norm_text(_FEAT_SUFFIX_RE.sub("", track_artist_name(track)).strip())

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
# Feat/ft credits only — always dropped (a credit, not a different recording).
# WHY: split out so lastfm_query_title can apply ONLY these (its version stripping is recording-aware, below),
# while clean_title keeps using the full set incl. the dash-version forms.
_FEAT_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"\(feat\b\.?.*?\)", r"\[feat\b\.?.*?\]", r"\(ft\b\.?.*?\)", r"\[ft\b\.?.*?\]",   # \b: don't eat "(FTampa Remix)"
    r"\bfeat\.?\s+\w+(?:\s+\w+)*", r"\bfeaturing\s+\w+(?:\s+\w+)*", r"\bft\.?\s+\w+(?:\s+\w+)*",
]]
_FEATURING_RES = _FEAT_RES + [re.compile(p, re.IGNORECASE) for p in [
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
def get_effective_now():
    """Return the current time in the travel timezone if the destination's current
    date falls within any configured travel window, otherwise return local time.
    Stripping tzinfo after conversion keeps all downstream code timezone-naive."""
    for trip in config.get("travel", []):
        try:
            dest_now = datetime.now(tz=ZoneInfo(trip["timezone"]))
            if date.fromisoformat(trip["start"]) <= dest_now.date() <= date.fromisoformat(trip["end"]):
                return dest_now.replace(tzinfo=None)
        except Exception:
            pass
    return datetime.now()

def _viewedAt_to_effective(dt):
    """Normalise a plexapi viewedAt timestamp to the effective timezone as a naive datetime.
    plexapi may return naive UTC or timezone-aware UTC; both are handled."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    for trip in config.get("travel", []):
        try:
            tz = ZoneInfo(trip["timezone"])
            dest_now = datetime.now(tz=tz)
            if date.fromisoformat(trip["start"]) <= dest_now.date() <= date.fromisoformat(trip["end"]):
                return dt.astimezone(tz).replace(tzinfo=None)
        except Exception:
            pass
    return dt.astimezone().replace(tzinfo=None)

# ---------------------------------------------------------------------
def get_current_time_period():
    """Determine which daypart the current hour belongs to."""
    current_hour = get_effective_now().hour

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
        m_print(f"Error loading descriptor map: {e}")
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
    now = get_effective_now()
    period_hours = set(time_periods[period]["hours"])

    history_start = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    log_text(f"[SELECTION] Scanning history for period '{period}' (Lookback: {HISTORY_LOOKBACK_DAYS} days).")

    all_history = list(music_section.history(mindate=history_start))

    # Single pass: build excluded_keys and filtered_entries simultaneously.
    # viewedAt from plexapi is UTC (naive or aware); normalise to the effective timezone
    # so that exclusion windows and period-hour matching are both timezone-correct.
    excluded_keys = set()
    filtered_entries = []
    for entry in all_history:
        if not entry.viewedAt:
            continue
        viewed = _viewedAt_to_effective(entry.viewedAt)
        if viewed >= exclude_start:
            excluded_keys.add(entry.ratingKey)
        elif viewed.hour in period_hours:
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
            and entry.viewedAt is not None
            # WHY: mirror the primary loop's tz handling — match on the effective-timezone
            # hour (not raw UTC) and guard None, or this fallback selects the wrong hours
            # under a travel/non-UTC tz and crashes on entries without a viewedAt.
            and _viewedAt_to_effective(entry.viewedAt).hour in adjacent_hours
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
    # 0. Pre-filter history entries before any API calls.
    #    parentRatingKey and userRating are available on raw history entries, so
    #    label, seasonal, and track-rating checks are free — no resolution needed.
    #    (Album/artist ratings require resolved objects and are checked later in process_tracks.)
    in_xmas = _in_christmas_window(now)
    filtered_entries = [
        e for e in filtered_entries
        if str(getattr(e, "parentRatingKey", "")) not in _excluded_album_keys
        and (in_xmas or str(getattr(e, "parentRatingKey", "")) not in _christmas_album_keys)
        and not (getattr(e, "userRating", None) is not None and getattr(e, "userRating", 10) <= 4)
    ]

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
        futures = {executor.submit(resolve_unique_track, rk): rk for rk in unique_keys}
        resolved_list = []
        for future in concurrent.futures.as_completed(futures):
            try:
                resolved_list.append(future.result(timeout=120))
            except Exception:
                resolved_list.append(None)
    
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

    # 5. Re-apply seasonal and label exclusions on resolved track objects.
    #    The pre-filter above works on raw history entries whose parentRatingKey may not
    #    be reliably populated in all PlexAPI versions. Running filter_excluded_tracks on
    #    the fully-resolved track objects guarantees Christmas music is excluded regardless.
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


# Typographic punctuation -> ASCII for Last.fm lookups.
# WHY: the RAW cache title was mismatching track.getInfo — Last.fm keeps SEPARATE pages for curly vs straight
# quotes (autocorrect won't merge them), so listener counts were silently wrong (confirmed: "I'm Gonna Be
# (500 Miles)" curly ~8k vs straight ~1M). Used by lastfm_query_title (+ norm_text-wrapped as top-tracks key).
_LASTFM_PUNCT = {0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-", 0x2026: "..."}
# Strip a trailing " - <version tag>" suffix.
# WHY: the _FEATURING_RES dash patterns don't cover "remaster", so "Yellow - Remastered" matched a low
# remaster page instead of the canonical track — this general suffix stripper fixes that.
_DASH_KW_RE = re.compile(rf"\s-\s.*(?:{_KW_ALT}).*$", re.IGNORECASE)

# "Different-recording" version tags: a DIFFERENT recording from the original (its own Last.fm page + its own
# identity), as opposed to a reissue (remaster/deluxe/radio edit/original mix = the SAME recording re-released).
# WHY: a remix/live/acoustic/etc. is its own track — it must read its OWN listener count and be a DISTINCT song,
# not borrow the canonical original's popularity (which pulled e.g. a DJ Enferno remix into a decade mix on the
# real "Party Rock Anthem" count). These keywords PROTECT a (paren)/[bracket]/" - " segment from being stripped
# by lastfm_query_title; reissue segments are still stripped.
_RECORDING_KEYWORDS = sorted([
    "remix", "live", "acoustic", "instrumental", "unplugged", "karaoke", "acapella", "a cappella", "demo",
    "dub", "cover", "rework", "re-edit", "bootleg", "vip", "rehearsal", "re-recorded", "rerecorded",
    "session", "sessions", "alternate", "take", "dj mix", "mix cut", "mashup", "flip", "refix", "reprise",
    "extended",   # extended mix/edit/version = a meaningfully different (longer) recording -> its own song
], key=len, reverse=True)
_RECORDING_ALT = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in _RECORDING_KEYWORDS)
# `…|\bre-?mix\w*`: the version-keyword strip matches "remix" as a substring (so it strips "(Remixed)",
# "(Remixes)", "(Remix1)"), so the KEEP test must match those same inflected forms or they'd collapse onto
# the studio cut. The stem keeps every remix-rooted form distinct (premix etc. excluded by the \b).
_RECORDING_KW_RE = re.compile(rf"\b(?:{_RECORDING_ALT})\b|\bre-?mix\w*", re.IGNORECASE)


def lastfm_query_title(title):
    """Title cleaned for a Last.fm track lookup / match. Folds typographic punctuation to ASCII, drops feat/ft
    credits, and strips REISSUE version tags (remaster / deluxe / radio edit / original mix — the same
    recording re-released) — but KEEPS apostrophes, bare words, AND any DIFFERENT-RECORDING segment
    (remix / live / acoustic / dub / …). Unlike clean_title (whose bare-word strip mangles real titles:
    "Take On Me" -> "on me", "Live" -> ""), this is safe to send to track.getInfo.
    WHY keep the different-recording tag: a remix is its own track — it must query its OWN Last.fm page (not
    the original's) and, wrapped in norm_text(), key as a DISTINCT song; reissues still collapse to canonical.
    Wrap in norm_text() for a case/apostrophe-insensitive match key."""
    if not title:
        return ""
    t = title.translate(_LASTFM_PUNCT)
    for pat in _FEAT_RES:                        # feat/ft credits — always dropped
        t = pat.sub("", t).strip()
    # Strip a version segment ONLY if it's a reissue; KEEP it when it names a different recording.
    # WHY: protecting remix/live/acoustic/… is what makes the remix query its own page + key as its own song.
    _keep_rec = lambda m: m.group(0) if _RECORDING_KW_RE.search(m.group(0)) else " "
    t = _PAREN_KW_RE.sub(_keep_rec, t)
    t = _BRACKET_KW_RE.sub(_keep_rec, t)
    t = _DASH_KW_RE.sub(lambda m: m.group(0) if _RECORDING_KW_RE.search(m.group(0)) else "", t)
    t = _EMPTY_PAREN_RE.sub("", t)
    t = _EMPTY_BRACKET_RE.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return _TRAIL_DASH_RE.sub("", t).strip()


def is_alt_recording(title):
    """True if the title carries a DIFFERENT-RECORDING tag (remix/live/acoustic/dub/…) in a version position —
    (parens), [brackets] or after ' - '. A bare title word ("Live and Let Die") is NOT a version tag, so it
    doesn't count; reissue tags (remaster/deluxe) are not alt recordings either.
    WHY: the resync re-fetches exactly these — their cached count must flip from the canonical to their own."""
    if not title:
        return False
    return any(_RECORDING_KW_RE.search(m.group(0))
               for rx in (_PAREN_KW_RE, _BRACKET_KW_RE, _DASH_KW_RE) for m in rx.finditer(title))


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
            orig = getattr(track, "originalTitle", None)
            artist_key = norm_text(primary_artist(orig)) if orig else norm_text(primary_artist(track_artist_name(track)))
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
    rejected_artist = rejected_mood = rejected_style = rejected_genre = 0

    for track in deduped_tracks:
        try:
            orig = getattr(track, "originalTitle", None)
            artist_name = norm_text(primary_artist(orig)) if orig else norm_text(primary_artist(track_artist_name(track)))
            if artist_count[artist_name] >= artist_limit:
                rejected_artist += 1
                continue

            # Mood is the vibe-consistency axis — checked before style/genre so no single
            # mood dominates the playlist. Primary mood only (depth=1).
            track_moods = _resolve_tags(track, "moods")
            if track_moods and mood_count[track_moods[0]] >= mood_limit:
                rejected_mood += 1
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
                    rejected_style += 1
                    continue
                for s in track_styles:
                    style_count[s] += 1
            elif track_genres:
                primary_genre = track_genres[0]
                if genre_count[primary_genre] >= genre_limit:
                    rejected_genre += 1
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
    if rejected_artist or rejected_mood or rejected_style or rejected_genre:
        log_text(f"[DEDUPE] Rejected — artist: {rejected_artist}, mood: {rejected_mood}, style: {rejected_style}, genre: {rejected_genre}")
    return unique_tracks

def fetch_sonically_similar_tracks(reference_tracks, excluded_keys=None):
    """
    Fetches Plex sonicallySimilar candidates for each seed track in parallel,
    populates the global sonic distance cache, then filters out recently played,
    excluded, low-rated, and already-selected tracks.

    Returns a deduplicated list of candidate tracks ready for process_tracks().
    """
    similar_tracks = []
    now = get_effective_now()
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
        futures = {executor.submit(_fetch_one, t): t for t in reference_tracks}
        fetched = []
        for future in concurrent.futures.as_completed(futures):
            try:
                fetched.append(future.result(timeout=120))
            except Exception as e:
                m_print(f"Error fetching sonically similar tracks: {e}")
                fetched.append((futures[future], []))

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
            if last_played and _viewedAt_to_effective(last_played) >= exclude_start:
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
            
        # Era/Decade similarity prevents "time-travel" jumps (original-release year, soft-fallback Plex year)
        _may, _mby = _entry_original_year(ma), _entry_original_year(mb)
        if _may and _mby:
            if abs(_may - _mby) <= 5:
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
        if ea and eb and ea.get("energy") is not None and eb.get("energy") is not None:
            # Tempo & Key - Squaring the distance to penalize "jumps" over "flows"
            dist += ((get_bpm_distance(ea["bpm"], eb["bpm"]) ** 2) * BPM_WEIGHT)
            dist += ((get_harmonic_distance(ea["key"], eb["key"]) ** 2) * KEY_WEIGHT)

            # Energy/Loudness Jump Penalty - Using an exponent to punish volume clashes
            energy_diff = abs(ea["energy"] - eb["energy"])
            energy_dist = min(energy_diff / 10.0, 1.0) # 10dB diff = max penalty
            dist += ((energy_dist ** 2) * ENERGY_WEIGHT)
            
            # Era/Decade Jump Penalty - Squaring ensures decade jumps are much costlier than 2-3 year shifts
            _eay, _eby = _entry_original_year(ea), _entry_original_year(eb)   # original-release yr, fallback Plex
            if _eay and _eby:
                year_diff = abs(_eay - _eby)
                year_dist = min(year_diff / 50.0, 1.0) # Penalty scales up to 50 years
                dist += ((year_dist ** 2) * ERA_WEIGHT)

            # Danceability Jump Penalty — punishes transitions between danceable and non-danceable tracks
            if ea.get("danceability") is not None and eb.get("danceability") is not None:
                dance_diff = abs(ea["danceability"] - eb["danceability"])
                dist += (dance_diff ** 2) * DANCEABILITY_WEIGHT

            # Brightness Jump Penalty — punishes jarring shifts between bright/trebly and dark/warm timbres
            if ea.get("brightness") is not None and eb.get("brightness") is not None:
                bright_diff = abs(ea["brightness"] - eb["brightness"])
                dist += (bright_diff ** 2) * BRIGHTNESS_WEIGHT

            # Beat confidence jump — penalises transitions from strong-groove to loose/ambient
            if ea.get("beat_confidence") is not None and eb.get("beat_confidence") is not None:
                bc_diff = min(abs(ea["beat_confidence"] - eb["beat_confidence"]) / 5.0, 1.0)  # raw Essentia scale → 0–1
                dist += (bc_diff ** 2) * BEAT_CONF_WEIGHT

            # Onset rate jump — penalises transitions between dense and sparse arrangements
            if ea.get("onset_rate") is not None and eb.get("onset_rate") is not None:
                or_diff = min(abs(ea["onset_rate"] - eb["onset_rate"]) / 10.0, 1.0)
                dist += (or_diff ** 2) * ONSET_RATE_WEIGHT

            # Arousal/valence/vocal jumps (TF features — only applied when both entries have data)
            if ea.get("arousal") is not None and eb.get("arousal") is not None:
                dist += ((ea["arousal"] - eb["arousal"]) ** 2) * AROUSAL_WEIGHT
                dist += ((ea["valence"] - eb["valence"]) ** 2) * VALENCE_WEIGHT
            if ea.get("vocal_presence") is not None and eb.get("vocal_presence") is not None:
                dist += ((ea["vocal_presence"] - eb["vocal_presence"]) ** 2) * VOCAL_WEIGHT

            # Bridge Bonus Logic: Reward sonic compatibility across different styles/genres.
            # Uses styles as the primary diversity axis (373 tags vs 43 genres — more granular).
            # Only the top STYLE_TAG_DEPTH styles are compared so that a track with 7–8 styles
            # doesn't share a minor tag with everything and lose the bonus unfairly.
            # Two tracks are cross-category when their top-N style sets are completely disjoint.
            # Falls back to genres only when both tracks have no style tags.
            ma, mb = meta_cache[ka], meta_cache[kb]
            a_styles = set(ea.get("styles", [])[:STYLE_TAG_DEPTH])
            b_styles = set(eb.get("styles", [])[:STYLE_TAG_DEPTH])
            if a_styles or b_styles:
                cross_category = not (a_styles & b_styles)
            else:
                cross_category = ma["genres"] != mb["genres"]
            if cross_category:
                # Require BPM and harmonic alignment as before, plus danceability and brightness
                # compatibility when data is available. Missing data does not block the bonus —
                # the None fallback ensures full coverage in libraries without acoustic analysis.
                dance_ok = (
                    ea.get("danceability") is None or eb.get("danceability") is None
                    or abs(ea["danceability"] - eb["danceability"]) < 0.2
                )
                bright_ok = (
                    ea.get("brightness") is None or eb.get("brightness") is None
                    or abs(ea["brightness"] - eb["brightness"]) < 0.1
                )
                if (abs(ea["bpm"] - eb["bpm"]) < 1.0
                        and get_harmonic_distance(ea["key"], eb["key"]) < 0.1
                        and dance_ok and bright_ok):
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
            "year": ec.get("year") or getattr(track, "year", None),
            "release_date": ec.get("release_date"),   # for _entry_original_year (original-release era)
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
            futures = {executor.submit(_fetch_similar, t): t for t in tracks_needing_fetch}
            for future in concurrent.futures.as_completed(futures):
                try:
                    t_key, sims = future.result(timeout=120)
                    similarity_cache[t_key] = sims
                except Exception:
                    pass

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
    max_passes = max(20, n)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            prev_i = first_track.ratingKey if i == 0 else path[i - 1].ratingKey
            for j in range(i + 1, n):
                next_j = last_track.ratingKey if j == n - 1 else path[j + 1].ratingKey
                old = edge_cost(prev_i, path[i].ratingKey) + edge_cost(path[j].ratingKey, next_j)
                new = edge_cost(prev_i, path[j].ratingKey) + edge_cost(path[i].ratingKey, next_j)
                if new < old:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
                    break  # restart with correct edge values after path mutation
            if improved:
                break

    end_cost = total_cost(path)
    capped = " (pass limit reached)" if passes >= max_passes else ""
    log_text(f"[ORDERING] 2-opt refinement complete in {passes} pass(es){capped}. Sonic path cost reduced from {start_cost:.2f} to {end_cost:.2f}.")
    
    return path, similarity_cache, meta_cache

def get_track_meta(track):
    rk = track.ratingKey
    ec = _essentia_cache.get(str(rk), {})
    return {
        "artist": ec.get("artist") or norm_text(primary_artist(track_artist_name(track))),
        "genres": set(ec.get("genres") or _resolve_tags(track, "genres")),
        "moods":  set(ec.get("moods")  or _resolve_tags(track, "moods")),
        "styles": set(ec.get("styles") or _resolve_tags(track, "styles")),
        "year": ec.get("year") or getattr(track, "year", None),
        "release_date": ec.get("release_date"),   # for _entry_original_year (original-release era)
    }

# Bridge Pass [Smarter Bridge Track Selection]
def fill_sonic_gaps(path, limit=SONIC_SIMILARITY_SEARCH_LIMIT, similarity_cache=None, meta_cache=None):
    """Identifies jumps > 0.4 and attempts to insert a bridge track from the library."""
    if similarity_cache is None: similarity_cache = {} # Safety fallback
    if not path or len(path) < 2: return path
    final_path = []
    
    # Safety and Deduplication Logic
    # Track identity by (Normalized Title, Normalized Artist) to catch duplicates across different albums.
    # We prefer originalTitle (track-level artist) over grandparentTitle (album artist) so that a track
    # appearing in a DJ set (album artist = DJ, track artist = real artist) registers under the real
    # artist rather than the DJ — preventing false positives when the DJ has their own unrelated song
    # with the same title.
    def _track_identity(t):
        title_key = norm_text(clean_title(t.title))
        orig = getattr(t, "originalTitle", None)
        artist_key = norm_text(primary_artist(orig)) if orig else norm_text(primary_artist(track_artist_name(t)))
        return title_key, artist_key

    existing_identities = {_track_identity(t) for t in path}
    existing_keys = {t.ratingKey for t in path}
    
    # Track artist counts to respect global limits (matching process_tracks logic)
    artist_counts = Counter([norm_text(primary_artist(track_artist_name(t))) for t in path])
    artist_limit = round(MAX_TRACKS * ARTIST_RATIO)

    
    # WHY: use the effective ("travel") timezone, not naive system-local — exclude_start is
    # compared against plexapi's naive-UTC lastViewedAt below, so both must be normalised the
    # same way (see fetch_sonically_similar_tracks). Otherwise the recency window is offset by
    # the server's UTC offset and ignores the travel tz.
    now = get_effective_now()
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
                path[_gidx].sonicallySimilar, SONIC_SIMILARITY_SEARCH_LIMIT
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
                potential_bridges = _gap_futures[i].result(timeout=30)
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
                    # Running the fast check first means excluded tracks (label-excluded, seasonal)
                    # are dropped before any network round-trip is ever made.
                    if not filter_low_rated_tracks(filter_excluded_tracks([bridge])):
                        bridge_rejected_rks.add(bridge.ratingKey)
                        d_print(f"  [!] Skipping '{bridge.title}': Failed exclusion/rating filters.")
                        continue

                    b_artist_raw = track_artist_name(bridge)
                    b_artist_norm = norm_text(primary_artist(b_artist_raw))
                    b_title_key = norm_text(clean_title(bridge.title))
                    b_identity = (b_title_key, b_artist_norm)
                    b_orig = getattr(bridge, "originalTitle", None)
                    b_track_artist_norm = norm_text(primary_artist(b_orig)) if b_orig else b_artist_norm

                    # Dedupe check: match on album artist identity OR track artist identity
                    if b_identity in existing_identities or (b_title_key, b_track_artist_norm) in existing_identities:
                        d_print(f"  [!] Skipping '{bridge.title}': Already in selection.")
                        continue

                    # NEW: Global artist limit check
                    if artist_counts[b_artist_norm] >= artist_limit:
                        d_print(f"  [!] Skipping '{bridge.title}': Artist '{b_artist_raw}' saturated ({artist_counts[b_artist_norm]}).")
                        continue

                    # B. FAST RECENCY CHECK
                    last_p = getattr(bridge, "lastViewedAt", None)
                    if last_p and _viewedAt_to_effective(last_p) >= exclude_start:
                        d_print(f"  [!] Skipping '{bridge.title}': Recently played.")
                        continue
                    
                    # NEW: Back-to-back check (prevents bridge matching t1 OR t2 artists)
                    if b_artist_norm == t1_artist_norm or b_artist_norm == t2_artist_norm:
                        d_print(f"  [!] Skipping '{bridge.title}': Artist back-to-back clash.")
                        continue

                    # D. METADATA & DISTANCE CHECK
                    # Resolve tags once — fast path via _essentia_cache, API fallback via _resolve_tags.
                    # This avoids the 3 duplicate _resolve_tags() calls that get_track_meta() + the
                    # diversity-check block would otherwise make for the same bridge candidate.
                    ec_b = _essentia_cache.get(str(bridge.ratingKey), {})
                    b_styles = list(ec_b.get("styles") or _resolve_tags(bridge, "styles"))[:STYLE_TAG_DEPTH]
                    b_genres = list(ec_b.get("genres") or _resolve_tags(bridge, "genres"))
                    b_moods  = list(ec_b.get("moods")  or _resolve_tags(bridge, "moods"))
                    bm = {
                        "artist": ec_b.get("artist") or norm_text(primary_artist(track_artist_name(bridge))),
                        "genres": set(b_genres),
                        "moods":  set(b_moods),
                        "styles": set(b_styles),
                        "year":   ec_b.get("year") or getattr(bridge, "year", None),
                        "release_date": ec_b.get("release_date"),   # original-release era (_entry_original_year)
                    }
                    # Evaluate distance to both sides of the gap to find the best "middle ground"
                    d1 = get_adj_dist(t1.ratingKey, bridge.ratingKey, {}, {**m_cache, bridge.ratingKey: bm}, limit)
                    d2 = get_adj_dist(bridge.ratingKey, t2.ratingKey, {}, {**m_cache, bridge.ratingKey: bm}, limit)

                    # Logic: If both legs of the new path are better than the single original jump, it is a net improvement for the "flow."
                    if d1 < dist and d2 < dist:
                        # Era penalty: same squared curve as Essentia scoring — max 0.05 at 50+ years.
                        # Era blending is intentional; sonic compatibility still dominates.
                        # Ignored when year data is unavailable.
                        b_year = bm.get("year")
                        t1_year = m_cache.get(t1.ratingKey, {}).get("year")
                        t2_year = m_cache.get(t2.ratingKey, {}).get("year")
                        era_penalty = 0.0
                        if b_year:
                            neighbour_years = [y for y in [t1_year, t2_year] if y]
                            if neighbour_years:
                                avg_neighbour_year = sum(neighbour_years) / len(neighbour_years)
                                era_penalty = (min(abs(b_year - avg_neighbour_year) / 50.0, 1.0) ** 2) * ERA_WEIGHT

                        # Sort by flow score alone — bridges are chosen for sonic fit, not diversity.
                        # Style/genre balance is already enforced in process_tracks; applying it here
                        # would sacrifice flow quality for marginal diversity gain on a handful of tracks.
                        # ratingKey is an int tiebreaker so Track objects are never compared directly.
                        candidates.append((d1 + d2 + era_penalty, bridge.ratingKey, bridge, b_identity, b_artist_norm, b_styles, b_genres))

                # F. SELECTION
                if candidates:
                    candidates.sort()
                    best_score, _, best_bridge, b_identity, b_artist_norm, b_styles, b_genres = candidates[0]

                    # E. AUDIO ANALYSIS (Only for the winner)
                    if ESSENTIA_ENABLED:
                        analyze_track_essentia(best_bridge)

                    log_text(f"[BRIDGE] Selected Best Fit: '{best_bridge.title}' (Combined Score: {best_score:.3f}). Inserting.")
                    final_path.append(best_bridge)

                    # Update tracking sets and artist counter
                    existing_identities.add(b_identity)
                    best_b_orig = getattr(best_bridge, "originalTitle", None)
                    if best_b_orig:
                        best_b_track_artist = norm_text(primary_artist(best_b_orig))
                        if best_b_track_artist != b_artist_norm:
                            existing_identities.add((b_identity[0], best_b_track_artist))
                    existing_keys.add(best_bridge.ratingKey)
                    artist_counts[b_artist_norm] += 1
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
            else:
                log_text("[WARN] Smart truncation: no removable candidate found — stopping early.")
                break

        m_print(f"[OK] Smart truncation complete. Playlist optimized at {MAX_TRACKS} tracks.")

    return final_path
# ------------------------------------


def generate_playlist_title_and_description(period, tracks):
    descriptor_map = load_descriptor_map(MOOD_MAP_PATH)
    now = get_effective_now()
    dawn_start = time_periods.get("Dawn", {}).get("hours", [5])[0]
    day_name = (now - timedelta(days=1) if now.hour < dawn_start else now).strftime("%A")

    # Use _resolve_tags() — the same resolution path as playlist selection —
    # so the title reflects what actually shaped the playlist, not just inline track tags.
    # Count primary mood per track to match the diversity-cap logic in process_tracks().
    style_counts = Counter()
    genre_counts = Counter()
    mood_counts  = Counter()
    for t in tracks:
        for s in _resolve_tags(t, "styles"):
            style_counts[str(s)] += 1
        for g in _resolve_tags(t, "genres"):
            genre_counts[str(g)] += 1
        moods = _resolve_tags(t, "moods")
        if moods:
            mood_counts[str(moods[0])] += 1

    sorted_styles = [s for s, _ in style_counts.most_common()]
    sorted_genres = [g for g, _ in genre_counts.most_common()]
    sorted_moods  = [m for m, _ in mood_counts.most_common()]

    most_common_mood   = sorted_moods[0] if sorted_moods else "Vibrant"
    second_common_mood = sorted_moods[1] if len(sorted_moods) > 1 else None

    # Descriptor from the moodmap: a punchy/colloquial translation of the primary mood,
    # used in the title. The formal mood name is then used in the description for clarity.
    descriptor = random.choice(descriptor_map.get(most_common_mood, ["Vibrant"]))

    # Two-tag phrase: prefer styles (primary diversity axis), fall back to genres.
    # Two tags reflect Meloday's varied nature — it's a curated mix, not a single-vibe channel.
    title_tags = sorted_styles[:2]
    if len(title_tags) < 2:
        title_tags += [g for g in sorted_genres if g not in title_tags][:2 - len(title_tags)]

    if len(title_tags) >= 2:
        tag_phrase = f"{title_tags[0]} & {title_tags[1]}"
    elif len(title_tags) == 1:
        tag_phrase = title_tags[0]
    else:
        tag_phrase = "Eclectic"

    def _apply_day(template, day):
        if "{day}" in template:
            return template.replace("{day}", day)
        return f"{day} {template}"

    period_phrase = _apply_day(get_period_phrase(period), day_name)
    title_period = _apply_day(TITLE_PERIOD_NAMES.get(period, get_period_phrase(period)), day_name)
    _NOCAP = {"at", "in", "on", "of", "to", "and", "or", "but", "the", "a", "an"}
    def _cover_case(s):
        words = s.split()
        return " ".join(w if w.lower() in _NOCAP else w.capitalize() for w in words)

    title = f"Meloday • {descriptor} {tag_phrase} for {title_period}"
    cover_title = f"Meloday • {descriptor} {tag_phrase} for {_cover_case(title_period)}"

    # Highlight tags for the description: prefer styles, then genres, then secondary moods.
    # Exclude whatever is already named in the title to avoid repetition.
    used = set(title_tags)
    highlight_tags = []
    for tag in sorted_styles[:4] + sorted_genres[:3] + sorted_moods[1:3]:
        if tag not in used and tag not in highlight_tags:
            highlight_tags.append(tag)
        if len(highlight_tags) >= 5:
            break

    if len(highlight_tags) > 1:
        extra_info = f"Here's some {', '.join(highlight_tags[:-1])}, and {highlight_tags[-1]} tracks as well."
    elif len(highlight_tags) == 1:
        extra_info = f"Here's some {highlight_tags[0]} tracks as well."
    else:
        extra_info = "Enjoy this selection of your favorites."

    # Use the formal mood name(s) in the description — no register clash with the slang title.
    article = "An" if most_common_mood[0].lower() in "aeiou" else "A"
    if second_common_mood:
        mood_phrase = f"{most_common_mood} and {second_common_mood}"
    else:
        mood_phrase = most_common_mood
    description = (
        f"{article} {mood_phrase.lower()} mix of {tag_phrase} from your {period_phrase} listening. "
        f"{extra_info}"
    )

    try:
        plex_account = plex.myPlexAccount()
        plex_user = plex_account.title.split()[0] if plex_account.title else plex_account.username
    except Exception:
        plex_user = "you"

    now = get_effective_now()
    next_update_hour = (time_periods[period]["hours"][-1] + 1) % 24
    next_update = now.replace(hour=next_update_hour, minute=0, second=0)
    if next_update_hour < now.hour:
        next_update += timedelta(days=1)

    description += f"\n\nMade for {plex_user} • Next update at {next_update.strftime('%I:%M %p').lstrip('0')}."
    return title, cover_title, description

def apply_text_to_cover(image_path, text):
    try:
        prefix = "Meloday • "
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

def create_or_update_playlist(name, tracks, description, cover_file, cover_name=None):
    # Use server-side title filtering to avoid fetching all playlists on every run.
    # The Python filter is kept as a safeguard in case plexapi's match is broader than expected.
    existing_playlist = next((pl for pl in plex.playlists(title="Meloday") if str(getattr(pl, "title", "")).startswith("Meloday")), None)
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
            new_cover = apply_text_to_cover(cover_path, cover_name or name)
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
    # Single-instance guard FIRST — before the log truncate below — so a redundant cron fire
    # that's about to skip doesn't wipe the running run's log. The lock frees automatically when
    # this process dies. The watchdog then bounds this run so a hang can't hold the lock forever.
    single_instance_guard("meloday")
    start_watchdog("meloday", MELODAY_MAX_SECONDS)

    # Force log truncation once at the start of the main process
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.truncate(0)

    # After the truncate so the [REAP] line survives in this run's log
    reap_orphaned_workers()   # sweep any workers a dead previous run left behind

    # NEW: Initialize global plex connection only here
    global plex
    plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=120)

    log_text("=== MELODAY RUN STARTED ===")

    # Log travel timezone if active
    for trip in config.get("travel", []):
        try:
            dest_now = datetime.now(tz=ZoneInfo(trip["timezone"]))
            if date.fromisoformat(trip["start"]) <= dest_now.date() <= date.fromisoformat(trip["end"]):
                msg = (
                    f"[TRAVEL] Timezone active: {trip['timezone']} "
                    f"({dest_now.strftime('%A %d %B %Y, %H:%M')}) "
                    f"[window: {trip['start']} → {trip['end']}]"
                )
                log_text(msg)
                print(msg)
                break
        except Exception:
            pass

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
    else:
        reason = "Library not found" if not ESSENTIA_AVAILABLE else "Disabled in config"
        m_print(f"[DIAGNOSTIC] Essentia is INACTIVE ({reason}). Using standard sorting.")
    load_essentia_cache()  # Always load — cached metadata (styles/moods/genres) benefits tag resolution even without acoustic data

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
    backfill_attempts = 0
    while len(final_tracks) < MAX_TRACKS:
        backfill_attempts += 1
        if backfill_attempts > 3:
            log_text(f"[WARN] Backfill stopped after 3 attempts — diversity caps or library size may be limiting results.")
            break
        progress += 5
        print_status(progress, "Attempting to add more tracks...")
        more_h, more_e = fetch_historical_tracks(period)
        excluded_keys |= more_e
        more_s = fetch_sonically_similar_tracks(final_tracks, excluded_keys=excluded_keys)
        more_h_sample = random.sample(more_h, min(MAX_TRACKS - len(final_tracks), len(more_h)))
        prev_len = len(final_tracks)
        # Pre-filter new candidates to avoid re-processing tracks already accepted.
        existing_rks = {t.ratingKey for t in final_tracks}
        new_candidates = [t for t in more_h_sample + more_s if t.ratingKey not in existing_rks]
        final_tracks = process_tracks(final_tracks + new_candidates)[:MAX_TRACKS]
        if len(final_tracks) == prev_len: break

    hist_keys = {t.ratingKey for t in guaranteed}
    hist_in_final = sum(1 for t in final_tracks if t.ratingKey in hist_keys)
    log_text(f"[COMPOSITION] {len(final_tracks)} tracks — {hist_in_final} from history, {len(final_tracks) - hist_in_final} from sonic discovery")

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
            to_analyze = []
            ts_map = {}  # ratingKey → (plex_track_ts, plex_album_ts, plex_artist_ts)

            # PRE-FILTER: Re-analyze only if Plex reports a change or the entry is missing.
            # album/artist timestamps are not on the track object — pass None here since
            # meloday's inline analysis path is a fast fallback for cache misses during
            # playlist generation. pre_analyze.py does the full three-level staleness check.
            for t in all_tracks:
                rk = str(t.ratingKey)
                file_path = get_local_path(t)
                plex_track_ts = t.updatedAt.timestamp() if getattr(t, "updatedAt", None) else None

                if rk in _essentia_cache:
                    data = _essentia_cache[rk]
                    if (data.get("file_path") == file_path and
                            data.get("track_updated_at") == plex_track_ts):
                        continue

                to_analyze.append(t.ratingKey)
                ts_map[t.ratingKey] = (plex_track_ts, None, None)

            if to_analyze:
                log_text(f"[DIAGNOSTIC] Cache miss/stale: Analyzing {len(to_analyze)} tracks.")
                cpu_workers = get_optimal_workers(task_type="cpu")
                new_entries = {}
                # Cap each spawned worker to 1 thread so N analysis workers don't oversubscribe the
                # CPU (N × all-core BLAS/TF). Restored right after the pool so the main process's own
                # numpy work (sonic refinement, embedding cosine) keeps full threading. Precautionary
                # — base Essentia is largely single-threaded — and matches the pre_analyze pools.
                _prev_caps = _set_thread_caps()
                # initializer: each worker self-terminates if this parent dies (OOM/watchdog/crash) —
                # without it, spawn workers survive a dead parent forever (see _worker_parent_sentinel).
                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_workers, max_tasks_per_child=10,
                                                            initializer=_worker_parent_sentinel) as executor:
                    # Use submit/as_completed so one crashing worker doesn't abort the whole batch
                    futures = {executor.submit(analysis_worker, tid, *ts_map[tid]): tid for tid in to_analyze}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            tid, data = future.result(timeout=120)
                            if data:
                                _essentia_cache[str(tid)] = data
                                new_entries[str(tid)] = data
                                # WHY: flush in batches so a watchdog kill (or any crash) mid-backlog
                                # never discards completed analysis — the next run resumes from a warmer
                                # cache and converges instead of re-analysing from zero. UPSERT is idempotent.
                                if len(new_entries) >= 50:
                                    _upsert_cache_entries(new_entries)
                                    new_entries = {}
                        except Exception as e:
                            log_text(f"[WARN] Analysis worker failed for track {futures[future]}: {e}")
                _restore_env(_prev_caps)
                _upsert_cache_entries(new_entries)   # flush the remainder (< 50 since the last batch)
            else:
                log_text("[OK] All tracks are cached and up-to-date.")

        print_status(70, "Double-ended 2-opt sonic refinement...")
        # Ensure sorting breadth covers the tracks in the list
        sort_breadth = max(SONIC_SIMILAR_LIMIT, len(middle) + 2)
        middle, similarity_cache, meta_cache = sort_by_sonic_similarity_refined(middle, first, last, limit=sort_breadth)
        # Feed sort-pass distances back into the global cache. Tracks that weren't in
        # _global_sonic_cache before the sort had their sonicallySimilar data stored
        # only in the local similarity_cache. Merging here means get_adj_dist() hits
        # the fast global-cache path (step 1) for all subsequent lookups in fill_sonic_gaps.
        for _rk, _sims in similarity_cache.items():
            if _rk not in _global_sonic_cache:
                _global_sonic_cache[_rk] = _sims

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
    title, cover_title, desc = generate_playlist_title_and_description(period, final_ordered_tracks)
    create_or_update_playlist(title, final_ordered_tracks, desc, time_periods[period]['cover'], cover_name=cover_title)
    print_status(100, "Playlist creation/update complete!")
    log_text("=== MELODAY RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()