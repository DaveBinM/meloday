#!/usr/bin/env python3
"""
meloday_extras.py
Supplementary "taste intelligence" playlists for a Plex music library.
Runs independently of meloday.py but shares read-only utilities from it.

Usage:
  python utilities/meloday_extras.py [--playlist PLAYLIST_ID] [--debug]

  PLAYLIST_ID: on_repeat | repeat_rewind | release_radar | discover_weekly |
               daily_mixes | rediscovery | time_capsule | deep_cuts | all
               (default: all)
"""

import io
import os
import json
import sys
import re
import math
import random
import heapq
import time
import logging
import argparse
import traceback
import colorsys
import concurrent.futures
from datetime import datetime, timedelta, timezone, date
from collections import Counter, defaultdict

# --- Plex / Meloday Imports ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import meloday  # required so load_essentia_cache() populates meloday._essentia_cache
from meloday import (
    load_config,
    load_essentia_cache,
    norm_text,
    primary_artist,
    lastfm_query_title,
    track_artist_name,
    get_bpm_distance,
    get_harmonic_distance,
    wrap_text,
    PLEX_URL,
    PLEX_TOKEN,
    MUSIC_LIBRARY,
    EXCLUDE_LABEL_NAMES,
    COVER_IMAGE_DIR,
    FONT_MAIN_PATH,
    FONT_MELODAY_PATH,
    FONT_LIGHT_PATH,
)
from plexapi.server import PlexServer

# --- Optional: numpy for k-means (Daily Mixes) ---
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

# --- Optional: requests for Last.fm (Release Radar) ---
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# --- Optional: PIL for cover generation ---
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# --- Optional: essentia-tensorflow for arousal/valence/vocal_presence ---
# Mirrors the _TF_MODELS_LOADED flag in meloday.py — True only when package and model files exist.
try:
    import essentia.standard as _es_probe
    _TF_AVAILABLE = getattr(_es_probe, "TensorflowPredictEffnetDiscogs", None) is not None
except Exception:
    _TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_BASE_DIR, "logs", "meloday_extras.log")
os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)

from logging.handlers import RotatingFileHandler
# WHY: this was mode="w", which wiped the log on every run — so an intermittent failure (e.g. an early-morning
# crash that froze the slate at the overnight state) left no trace to diagnose. Append + size-bounded rotation
# keeps several days of run history while capping disk.
_file_handler = RotatingFileHandler(_LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
_logger = logging.getLogger("MelodayExtras")
_logger.setLevel(logging.INFO)
_logger.addHandler(_file_handler)


def xlog(msg):
    if not msg:
        return
    clean = "".join(ch for ch in str(msg) if ch.isprintable() or ch in "\n\r\t").replace("\x00", "")
    print(clean)
    _logger.info(clean)
    _file_handler.flush()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
config = load_config()
_extras = config.get("extras", {})

DAILY_MIX_COUNT          = int(_extras.get("daily_mix_count", 6))
DISCOVER_WEEKLY_SIZE      = int(_extras.get("discover_weekly_size", 30))
RELEASE_RADAR_START_DAYS  = int(_extras.get("release_radar_start_days", 14))
RELEASE_RADAR_STEP_DAYS   = int(_extras.get("release_radar_step_days", 7))
RELEASE_RADAR_MIN_TRACKS  = int(_extras.get("release_radar_min_tracks", 50))
RELEASE_RADAR_MAX_DAYS    = int(_extras.get("release_radar_max_days", 90))
BIRTH_YEAR                = _extras.get("birth_year")  # int or None
LASTFM_API_KEY            = _extras.get("lastfm_api_key")
LASTFM_API_SECRET         = _extras.get("lastfm_api_secret")
ARTIST_RATIO              = config["playlist"].get("artist_ratio", 0.05)

_xmas_cfg               = config.get("seasonal", {}).get("christmas", {})
_XMAS_START_MONTH       = int(_xmas_cfg.get("start_month", 12))
_XMAS_START_DAY         = int(_xmas_cfg.get("start_day",   1))
_XMAS_END_MONTH         = int(_xmas_cfg.get("end_month",   12))
_XMAS_END_DAY           = int(_xmas_cfg.get("end_day",     25))
_CHRISTMAS_COLLECTION   = config.get("plex", {}).get("christmas_collection", "Christmas Music")


def _in_christmas_window(now):
    try:
        start = datetime(now.year, _XMAS_START_MONTH, _XMAS_START_DAY)
        end   = datetime(now.year, _XMAS_END_MONTH,   _XMAS_END_DAY)
        if start > end:
            return now >= start or now <= end
        return start <= now <= end
    except ValueError:
        return now.month == 12 and 1 <= now.day <= 25


PLAYLIST_IDS = [
    "on_repeat", "repeat_rewind", "release_radar", "discover_weekly",
    "daily_mixes", "rediscovery", "time_capsule", "time_machine", "deep_cuts",
    "top_songs", "all_time_favourites", "mood_mixes",
]

# How many days of history each playlist builder actually reads.
# release_radar uses no history in its builder — only the centroid (pre-computed).
# The 180-day value there covers the centroid computation, not the builder itself.
_HISTORY_LOOKBACK_DAYS = {
    "on_repeat":           30,    # fixed 30-day window
    "repeat_rewind":       70,    # 10-week peak window
    "release_radar":      180,    # centroid only — builder reads no history directly
    "discover_weekly":    365,    # played_keys filter; older plays still valid to exclude
    "daily_mixes":        180,    # 6 months gives stable acoustic clusters
    "rediscovery":        730,    # 24-month outer boundary + buffer
    "time_capsule":       180,    # recent gone-quiet window only; deep history via fetch_history_window
    "time_machine":       180,    # recent gone-quiet window only; deep history via fetch_history_window
    "deep_cuts":          180,    # 6-month artist ranking window
    # top_songs lookback is computed dynamically from top_songs_start_year config
    # all_time_favourites uses track.viewCount — no history needed
    "mood_mixes":         180,    # play-count weighting + rotation scoring
}

# Only these playlists need the listening centroid computed from history.
_CENTROID_PLAYLISTS = {"release_radar", "discover_weekly"}

# ---------------------------------------------------------------------------
# Cover Art — gradient colour palette (top RGB, bottom RGB) per playlist
# ---------------------------------------------------------------------------
_EXTRAS_COVER_COLORS = {
    # --- Meloday+ gap-fill mixes: 7 vibe gaps + 4-mix pop family ---
    "situationship": ((110, 95, 140), (55, 45, 75)),
    "sad_bangers": ((230, 70, 150), (45, 30, 90)),
    "power_ballads": ((200, 60, 80), (60, 20, 35)),
    "restless": ((40, 140, 150), (25, 30, 45)),
    "neoclassical": ((180, 200, 210), (95, 120, 145)),
    "yacht_rock": ((255, 200, 120), (70, 150, 180)),
    "swagger": ((205, 165, 60), (70, 40, 70)),
    "chart_pop": ((255, 90, 140), (120, 40, 150)),
    "dance_pop": ((255, 60, 170), (60, 30, 120)),
    "indie_pop": ((255, 170, 120), (60, 140, 150)),
    "synth_pop": ((230, 90, 210), (40, 40, 120)),
    # rock / electronic / scores gap-fill
    "indie_rock": ((210, 90, 70), (90, 40, 55)),
    "post_grunge": ((110, 120, 110), (45, 50, 48)),
    "rap_rock": ((180, 50, 40), (40, 30, 35)),
    "festival_edm": ((90, 70, 230), (200, 40, 170)),
    "soundtracks": ((200, 170, 90), (30, 40, 80)),
    "rave_cave": ((40, 230, 140), (90, 22, 150)),          # acid-green grid over UV purple (rave)
    # ---- 7 decade mixes (era) ----
    "decade_60s": ((240, 150, 60), (200, 70, 120)),
    "decade_70s": ((230, 160, 50), (150, 70, 40)),
    "decade_80s": ((255, 70, 180), (60, 200, 230)),
    "decade_90s": ((230, 60, 180), (40, 200, 200)),   # 90s hot-pink → cyan
    "decade_00s": ((80, 140, 240), (180, 100, 220)),
    "decade_10s": ((40, 200, 190), (255, 110, 120)),
    "decade_20s": ((150, 120, 255), (90, 200, 220)),
    # ---- 3 geo showcase mixes ----
    "scotland_scene":  ((200, 76, 24), (150, 52, 16)),     # Irn-Bru orange (tartan ground)
    "australia_scene": ((225, 125, 60), (140, 55, 50)),    # sunset orange/red
    "london_scene":    ((125, 85, 130), (48, 38, 68)),     # urban purple
    # ---- 3 geo HITS mixes (reuse the matching scene's gradient; star glyph distinguishes) ----
    "scottish_hits":   ((200, 76, 24), (150, 52, 16)),     # Irn-Bru orange (tartan ground)
    "australian_hits": ((225, 125, 60), (140, 55, 50)),    # sunset orange/red
    "london_hits":     ((125, 85, 130), (48, 38, 68)),     # urban purple
    "uk_scene":        ((20, 56, 140), (10, 28, 80)),      # Union Jack blue
    "uk_hits":         ((20, 56, 140), (10, 28, 80)),      # Union Jack blue
    "scotland_now":    ((30, 110, 170), (16, 56, 96)),     # Saltire blue (contemporary Scottish radio)
    "london_now":      ((196, 52, 58), (70, 18, 26)),      # London red (contemporary London radio)
    "uk_now":          ((20, 56, 140), (10, 28, 80)),      # Union Jack blue (union_jack bg hard-codes flag colours)
    "australia_now":   ((46, 140, 82), (212, 160, 40)),    # green & gold — green sky, golden sun (contemporary Australian radio)
    "stormy": ((60, 70, 95), (20, 24, 40)),
    "foggy": ((150, 158, 168), (70, 76, 88)),
    "snow_day": ((214, 228, 244), (140, 160, 196)),
    "heatwave": ((255, 150, 70), (200, 80, 60)),
    "frosty": ((210, 232, 240), (120, 170, 200)),
    "grey_skies": ((130, 135, 145), (64, 67, 76)),
    "windy": ((40, 150, 150), (18, 80, 90)),
    "clear_night": ((30, 40, 80), (150, 160, 200)),
    "festive": ((186, 34, 44), (20, 92, 52)),     # deep Christmas red → pine green
    "spring_bloom": ((120, 200, 120), (240, 150, 180)),
    "spring_acoustic": ((150, 190, 120), (80, 120, 70)),
    "spring_strings": ((180, 210, 190), (110, 150, 170)),
    "spring_jangle": ((140, 200, 170), (200, 160, 200)),
    "summer_heat": ((255, 128, 36), (190, 40, 28)),        # blazing hot orange → deep red
    "summer_breeze": ((255, 200, 120), (120, 180, 200)),
    "summer_roadtrip": ((255, 160, 60), (60, 150, 200)),
    "summer_tropical": ((255, 140, 60), (40, 170, 150)),
    "autumn_leaves": ((200, 120, 50), (110, 60, 30)),
    "autumn_jazz": ((180, 110, 60), (90, 50, 40)),
    "autumn_rain": ((110, 100, 110), (55, 50, 60)),
    "autumn_embers": ((190, 90, 40), (80, 40, 30)),
    "winter_frost": ((200, 224, 236), (110, 150, 185)),
    "winter_cosy": ((180, 90, 90), (80, 40, 50)),
    "winter_nights": ((40, 50, 90), (18, 22, 45)),
    "winter_jazz": ((150, 130, 170), (70, 55, 90)),
    "hopeful": ((240, 180, 80), (70, 110, 160)),
    "yearning": ((90, 110, 150), (40, 50, 80)),
    "triumphant": ((220, 160, 50), (150, 30, 40)),
    "serene": ((130, 190, 180), (50, 90, 95)),
    "tender": ((230, 170, 180), (120, 70, 90)),
    "defiant": ((200, 40, 50), (30, 30, 35)),
    "vulnerable": ((140, 120, 160), (60, 50, 75)),
    "awe_wonder": ((60, 90, 150), (20, 30, 70)),
    "grief_release": ((70, 80, 100), (30, 35, 50)),
    "sunrise": ((86, 58, 128), (255, 196, 138)),           # deep violet sky → peach/gold horizon
    "blue_hour": ((60, 80, 140), (25, 30, 65)),
    "midnight": ((30, 35, 70), (12, 14, 30)),
    "three_am": ((40, 60, 70), (18, 25, 32)),
    "golden_afternoon": ((240, 180, 90), (180, 120, 60)),
    "overcast": ((120, 125, 135), (60, 63, 70)),
    "starlit": ((30, 40, 80), (150, 160, 200)),
    "witching_hour": ((80, 40, 110), (25, 12, 40)),
    "monday_motivation": ((70, 140, 210), (30, 60, 120)),
    "midweek_reset": ((50, 150, 140), (20, 70, 70)),
    "friday_feeling": ((230, 70, 150), (120, 30, 110)),
    "sunday_scaries": ((90, 100, 130), (45, 50, 70)),
    "treat_yourself": ((220, 120, 180), (120, 40, 120)),
    "dinner_party": ((150, 60, 70), (70, 28, 40)),
    "housework_hustle": ((240, 150, 40), (150, 80, 20)),
    "study_session": ((70, 90, 110), (30, 40, 55)),
    "wind_down": ((90, 70, 130), (40, 30, 65)),
    "yoga_stretch": ((120, 180, 150), (50, 90, 80)),
    "meditation": ((90, 140, 170), (35, 60, 80)),
    "deep_reading": ((150, 110, 70), (70, 50, 30)),
    "creative_flow": ((180, 90, 200), (80, 40, 120)),
    "gaming": ((40, 200, 120), (15, 90, 60)),
    "gardening": ((110, 170, 70), (50, 90, 35)),
    "spa_bath": ((90, 180, 190), (35, 90, 100)),
    "power_nap": ((140, 130, 190), (60, 55, 95)),
    "throwback_anthems": ((230, 120, 60), (40, 110, 120)),
    "old_friends": ((230, 160, 80), (130, 80, 40)),
    "campfire": ((220, 110, 50), (90, 40, 25)),
    "cookout": ((238, 96, 52), (40, 120, 66)),         # BBQ red → picnic green (more contrast)
    "game_night": ((150, 80, 200), (70, 35, 110)),
    "singalong": ((240, 70, 120), (120, 30, 80)),
    "school_days": ((90, 140, 200), (230, 190, 60)),
    "memory_lane": ((180, 140, 110), (90, 65, 50)),
    "crush": ((255, 130, 170), (180, 60, 120)),
    "slow_burn": ((176, 66, 40), (58, 22, 28)),            # smouldering ember (smoke family)
    "moving_on": ((210, 130, 70), (60, 80, 120)),
    "loved_up": ((255, 120, 140), (200, 60, 90)),
    "long_distance": ((80, 70, 140), (150, 50, 70)),
    "flirty": ((255, 90, 150), (150, 30, 100)),
    "devotion": ((200, 60, 80), (110, 30, 50)),
    "wedding_day": ((250, 210, 190), (230, 150, 160)),
    "funk_disco": ((150, 40, 150), (70, 20, 90)),
    "neo_soul": ((120, 60, 120), (50, 25, 70)),
    "motown_soul": ((210, 90, 70), (120, 40, 30)),
    "after_hours_rnb": ((70, 40, 110), (30, 15, 55)),
    "acid_jazz": ((90, 120, 60), (40, 60, 30)),
    "boom_bap": ((120, 90, 50), (55, 40, 20)),
    "conscious_flow": ((80, 100, 90), (35, 50, 45)),
    "g_funk": ((180, 140, 40), (90, 60, 20)),
    "trap_mode": ((40, 40, 55), (15, 15, 25)),
    "lofi_beats": ((90, 80, 120), (40, 35, 60)),
    "house_party": ((230, 120, 180), (120, 40, 110)),
    "deep_house": ((40, 70, 90), (15, 30, 45)),
    "techno": ((40, 210, 220), (46, 16, 64)),              # neon cyan circuit traces over deep purple (warehouse)
    "trance": ((60, 120, 220), (20, 40, 130)),
    "dnb": ((40, 180, 160), (15, 70, 70)),
    "bass_drop": ((90, 40, 140), (35, 15, 60)),
    "uk_garage": ((60, 200, 220), (110, 30, 170)),         # neon club lasers (cyan → purple)
    "synthwave": ((200, 40, 160), (40, 20, 90)),
    "industrial": ((70, 75, 80), (25, 28, 32)),
    "vaporwave": ((255, 140, 200), (90, 60, 160)),
    "downtempo": ((50, 60, 90), (20, 25, 45)),
    "hyperpop": ((255, 90, 180), (120, 30, 150)),
    "classic_rock": ((180, 80, 40), (90, 35, 20)),
    "heavy_riffs": ((140, 30, 30), (45, 15, 18)),
    "punk_energy": ((230, 40, 90), (110, 15, 45)),
    "garage_grunge": ((110, 110, 90), (45, 45, 35)),
    "emo_poppunk": ((210, 60, 120), (90, 25, 70)),
    "britpop_rock": ((90, 160, 200), (35, 70, 110)),
    "blues_bar": ((168, 104, 48), (34, 30, 52)),           # dim smoky bar — warm amber over indigo
    "psych_haze": ((120, 60, 180), (50, 25, 90)),
    "prog_rock": ((60, 90, 130), (25, 40, 65)),
    "stoner_rock": ((150, 80, 40), (60, 30, 18)),
    "reggae_dub": ((40, 150, 70), (200, 170, 40)),
    "afrobeat": ((240, 150, 30), (120, 60, 15)),
    "latin_heat": ((255, 90, 60), (160, 30, 80)),
    "bossa_samba": ((40, 170, 140), (20, 90, 80)),
    "celtic_folk": ((40, 130, 90), (20, 65, 50)),
    "ska": ((40, 40, 40), (220, 200, 40)),
    "bebop": ((180, 140, 60), (90, 60, 25)),
    "swing_bigband": ((210, 160, 60), (110, 70, 25)),
    "smooth_jazz": ((90, 110, 140), (40, 55, 75)),
    "country_roads": ((210, 150, 70), (110, 70, 30)),
    "outlaw_country": ((150, 100, 60), (70, 45, 25)),
    "bluegrass": ((150, 180, 80), (70, 90, 35)),
    "rockabilly_surf": ((60, 160, 200), (25, 80, 120)),
    "cinematic_epic": ((180, 150, 60), (30, 40, 80)),
    "ambient_drift": ((70, 90, 120), (25, 35, 55)),
    "post_rock": ((60, 80, 110), (25, 35, 55)),
    "chiptune": ((60, 200, 120), (20, 90, 60)),
    "gospel": ((220, 170, 50), (120, 70, 20)),
    "glasgow_folk": ((90, 110, 120), (40, 55, 65)),
    "glasgow_dream": ((110, 90, 140), (45, 35, 70)),
    "glasgow_indie": ((100, 120, 150), (45, 55, 80)),
    "glasgow_soul": ((150, 110, 70), (70, 50, 30)),
    "glasgow_postrock": ((60, 80, 100), (25, 35, 50)),
    "glasgow_anthems": ((120, 130, 160), (50, 60, 90)),
    "glasgow_synth": ((90, 80, 150), (35, 30, 80)),
    "glasgow_postpunk": ((80, 90, 110), (30, 35, 50)),
    "glasgow_house": ((110, 90, 130), (45, 35, 65)),
    "glasgow_underground": ((70, 80, 95), (25, 30, 40)),
    "glasgow_bass": ((150, 90, 170), (60, 30, 90)),
    "glasgow_late": ((50, 60, 85), (20, 25, 42)),
    "london_dub": ((150, 60, 50), (60, 25, 30)),
    "london_soul": ((160, 70, 80), (65, 28, 40)),
    "london_jazz": ((130, 90, 60), (55, 40, 30)),
    "london_triphop": ((70, 60, 90), (28, 25, 45)),
    "london_mod": ((180, 60, 60), (70, 30, 40)),
    "london_britpop": ((120, 70, 150), (50, 28, 75)),
    "london_indie": ((150, 80, 90), (60, 32, 45)),
    "london_calling": ((180, 40, 50), (70, 18, 28)),
    "london_garage": ((170, 120, 40), (70, 50, 18)),
    "london_grime": ((120, 40, 55), (45, 15, 25)),
    "london_dubstep": ((90, 50, 110), (35, 20, 50)),
    "london_jungle": ((150, 40, 70), (60, 18, 35)),
    "melbourne_folk": ((60, 120, 120), (28, 55, 55)),
    "melbourne_dream": ((90, 140, 150), (40, 65, 70)),
    "melbourne_soul": ((40, 140, 120), (20, 70, 60)),
    "melbourne_sunset": ((255, 140, 90), (150, 60, 90)),
    "melbourne_indie": ((60, 150, 140), (28, 70, 65)),
    "melbourne_pubrock": ((200, 90, 50), (90, 40, 25)),
    "melbourne_hiphop": ((40, 110, 120), (20, 55, 60)),
    "melbourne_postpunk": ((50, 70, 80), (20, 28, 35)),
    "melbourne_psych": ((90, 60, 150), (40, 25, 75)),
    "melbourne_garagepunk": ((220, 70, 60), (100, 30, 30)),
    "melbourne_club": ((40, 170, 170), (20, 80, 90)),
    "melbourne_techno": ((40, 90, 95), (18, 40, 45)),
    "on_repeat":      ((255, 140, 40),  (180,  60, 20)),   # warm amber → deep orange
    "repeat_rewind":  ((140,  60, 220), ( 70,  20, 140)),  # violet → deep indigo
    "release_radar":  (( 20, 105, 200), ( 10,  45, 110)),  # sky blue → navy
    "discover_weekly":((200, 100, 210), ( 20,  40, 115)),  # warm lavender → deep navy
    "daily_mix_1":    (( 50, 110, 230), ( 20,  50, 160)),  # blue
    "daily_mix_2":    ((220,  70,  70), (160,  20,  40)),  # coral red
    "daily_mix_3":    (( 60, 185,  80), ( 20, 100,  40)),  # green
    "daily_mix_4":    ((220, 130,  20), (150,  70,  10)),  # amber orange
    "daily_mix_5":    ((150,  55, 215), ( 80,  20, 145)),  # purple
    "daily_mix_6":    (( 20, 165, 180), ( 10,  85, 115)),  # teal cyan
    "rediscovery":        ((200,  80, 120), (110,  30,  70)),  # rose → deep rose
    "time_capsule":       ((185, 135,  65), ( 95,  55,  20)),  # warm sepia → dark amber
    "time_machine":       ((150,  95, 175), ( 60,  35,  95)),  # nostalgic violet → deep indigo
    "deep_cuts":          (( 50,  65, 110), ( 20,  30,  65)),  # slate → near-black blue
    # Top Songs / All-Time
    "top_songs":          ((200, 160,  30), (140,  90,  10)),  # warm gold
    "all_time_favourites":((220, 180,  20), (160, 120,  10)),  # bright gold
    # Mood / Activity Mixes (12 profiles)
    "workout":            ((220,  50,  50), (160,  20,  20)),  # energetic red
    "running":            ((220, 100,  30), (160,  60,  10)),  # orange
    "party":              ((220,  80, 180), (150,  20, 120)),  # vivid magenta
    "happy":              ((235, 168,  28), (188, 112,   8)),  # warm orange-yellow
    "morning":            ((252, 185,  62), (242, 132,  28)),  # warm orange-golden sunrise
    "focus":              (( 50,  80, 160), ( 20,  40, 100)),  # cool blue
    "dinner":             ((120,  50, 140), ( 70,  20,  90)),  # warm purple
    "chill":              (( 40, 150, 130), ( 20,  80,  80)),  # teal
    "rainy_day":          (( 80, 110, 160), ( 40,  60, 110)),  # grey-blue
    "melancholy":         (( 60,  50, 130), ( 30,  20,  90)),  # muted indigo
    "late_night":         (( 58,  18,  88), ( 28,   8,  52)),  # distinctly purple
    "sleep":              (( 30,  30, 100), ( 10,  10,  60)),  # deep navy
    "sunny":              ((255, 200,  30), (220, 140,  10)),  # bright sunshine yellow
    "cosy":               ((205, 145,  80), (140,  85,  35)),  # warm amber-caramel
    # --- Mood / Emotional ---
    "nostalgia_mix":      ((168, 118,  90), (100,  62,  42)),  # warm sepia-terracotta
    "dreamy_mix":         ((160, 120, 200), ( 80,  60, 140)),  # soft lavender
    "moody_mix":          (( 55,  55, 110), ( 25,  25,  70)),  # dark slate blue
    "emotional":          ((180,  70, 110), (110,  30,  65)),  # deep rose-mauve
    "bittersweet":        ((200, 130,  60), (130,  70,  25)),  # warm amber-rust
    "cathartic":          (( 44,  70, 128), (224,  96,  72)),  # deep blue → coral (high contrast through smoke)
    "confidence_boost":   ((192, 158,  42), (135, 105,  14)),  # deeper brass-gold
    "empowering":         ((170,  50, 220), (100,  20, 160)),  # bright violet
    "euphoric":           ((240,  80, 140), (170,  20,  90)),  # hot pink-coral
    "angst_mix":          ((150,  30,  30), ( 80,  10,  10)),  # dark red-charcoal
    "romantic_mix":       ((210, 120, 150), (140,  65,  90)),  # soft rose-dusty pink
    "daydreaming":        ((130, 145, 215), ( 75,  85, 165)),  # soft periwinkle-lavender
    "fresh_start":        ((100, 200, 150), ( 40, 130,  90)),  # mint-sage green
    # --- Aesthetic / Time-of-Day ---
    "main_character":     (( 40,  50,  90), ( 15,  20,  55)),  # dramatic navy
    "golden_hour":        ((230, 170,  60), (170, 100,  20)),  # warm gold-amber
    "sunset_mix":         ((220, 110,  80), (160,  55,  40)),  # coral-orange-pink
    "after_dark":         (( 18,  14,  58), (  7,   5,  30)),  # deep blue-black
    # --- Time / Occasion ---
    "after_work":         ((180, 155, 100), (110,  90,  50)),  # warm khaki-neutral
    "friday_night":       (( 40,  80, 200), ( 20,  40, 130)),  # electric blue-navy
    "weekend_mix":        ((110, 150, 220), ( 60,  90, 170)),  # sky blue-periwinkle
    "sunday_morning":     ((230, 200, 130), (175, 140,  75)),  # warm cream-yellow
    "lazy_sunday":        ((178, 148, 175), (110,  88, 108)),  # soft lavender-haze
    "brunch_mix":         ((140, 200, 100), ( 80, 140,  50)),  # fresh green-lime
    "date_night":         ((120,  30,  70), ( 70,  10,  40)),  # deep wine-burgundy
    # --- Activity ---
    "driving_mix":        (( 70,  90, 130), ( 35,  50,  85)),  # road grey-asphalt blue
    "night_drive":        (( 20,  60,  90), ( 10,  30,  60)),  # deep teal-midnight
    "driving_singalong":  (( 60, 140, 220), ( 25,  75, 160)),  # bright sky blue
    "road_trip":          ((222, 172,  90), (160, 108,  42)),  # sandy ochre-desert
    "commute_mix":        ((100, 115, 140), ( 55,  65,  90)),  # steel grey-slate
    "walking_mix":        (( 80, 175, 100), ( 35, 110,  55)),  # grass green-nature
    # --- Social / Nostalgia ---
    "party_throwback":    ((230,  60, 150), (160,  15,  90)),  # neon pink-electric
    # --- Weather / Season ---
    "beach_vibes":        (( 30, 175, 180), ( 10, 100, 130)),  # turquoise-ocean
    "summer_evening":     ((200, 130, 100), (120,  65,  85)),  # warm coral-lilac twilight
    "autumn_mix":         ((195, 110,  40), (125,  60,  15)),  # burnt orange-rust
    "winter_mix":         ((110, 145, 185), ( 55,  85, 135)),  # icy blue-grey
    "spring_mix":         ((210, 155, 175), (155,  90, 115)),  # blossom pink
    # --- Romance ---
    "modern_romance":     ((170,  90, 160), (100,  40, 105)),  # warm purple-mauve
    "late_night_romance": (( 28,  16,  78), ( 12,   6,  48)),  # deep midnight navy-blue
    "romantic_dinner":    ((145,  30,  55), ( 85,  10,  30)),  # dark red-wine
    "love_songs":         ((225, 130, 155), (165,  70,  95)),  # blush pink-rose
    "slow_dance":         ((145, 100, 175), ( 85,  50, 120)),  # soft purple-lavender
    "candlelight":        ((215, 150,  50), (150,  90,  20)),  # warm amber-gold
    "first_date":         ((220, 140, 110), (160,  80,  60)),  # coral-peach
    "romantic_jazz":      ((140,  80,  30), ( 85,  40,  10)),  # dark cognac-amber
    "jazz_dinner":        (( 80,  70,  75), ( 45,  38,  42)),  # dark slate-warm grey
    "string_quartet":     ((195, 170, 140), (130, 100,  70)),  # elegant cream-warm
    "strings_romance":    ((200, 160, 155), (135,  95,  90)),  # soft rose-cream
    "piano_romance":      (( 50,  40,  55), ( 20,  15,  25)),  # deep charcoal-black
    "acoustic_romance":   ((205, 130, 120), (145,  72,  80)),  # warm dusty rose-blush
    "indie_romance":      ((155, 110, 140), ( 92,  55,  88)),  # dusty mauve-plum
    "synthpop_romance":   ((160,  60, 180), ( 90,  20, 130)),  # electric mauve-neon pink
    # --- Gap fills ---
    "evening_unwind":     ((120, 100, 160), ( 65,  50, 110)),  # soft lavender-slate purple
    "heartbreak":         ((130,  40,  60), ( 70,  15,  30)),  # deep crimson-charcoal
    "pre_party":          ((230, 120,  50), (170,  65,  15)),  # electric coral-gold
    "cool_down":          (( 60, 160, 140), ( 25,  95,  85)),  # soft seafoam-teal
    "cooking_mix":        ((210, 120,  50), (145,  65,  20)),  # warm terracotta-orange
    "deep_work":          (( 45,  65, 100), ( 18,  30,  60)),  # dark steel blue
    "folk_acoustic":      ((150, 110,  60), ( 90,  60,  25)),  # warm earth-wood
    "celebration":        ((240, 200,  60), (190, 140,  20)),  # bright champagne-gold
}


# Background style per cover key — (style_name, variant_int). Bespoke per playlist: the family
# reflects the playlist's genre/mood; weather is literal (rain->ripples/rainfall, etc.); variant
# indices beyond a family's base table are the base pattern h-mirrored (see _render_bg). Generated
# collision-free + genre-true; guarded by cover_validator.py. daily_mix/release_radar left as-is.
_COVER_BG_STYLES = {
    # --- electronic ---
    "after_dark": ("laser_fan", 0),
    "bass_drop": ("concentric_pulse", 0),
    "dance_pop": ("disco_ball", 0),
    "decade_00s": ("spiral", 0),
    "decade_20s": ("equalizer", 1),
    "decade_80s": ("concentric_pulse", 1),
    "deep_house": ("circuit", 1),
    "dnb": ("spiral", 1),
    "festival_edm": ("starburst", 1),
    "glasgow_bass": ("waveform", 0),
    "glasgow_house": ("laser_fan", 1),
    "glasgow_synth": ("grid_perspective", 0),
    "glasgow_underground": ("grid_perspective", 1),
    "house_party": ("equalizer", 2),
    "london_dubstep": ("waveform", 1),
    "london_garage": ("laser_fan", 2),
    "london_jungle": ("waveform", 2),
    "melbourne_club": ("grid_perspective", 2),
    "melbourne_techno": ("grid_perspective", 3),
    "rave_cave": ("grid_perspective", 4),
    "synth_pop": ("waveform", 3),
    "synthwave": ("grid_perspective", 5),
    "techno": ("circuit", 0),
    "time_machine": ("concentric_pulse", 2),
    "trance": ("circuit", 2),
    "uk_garage": ("laser_fan", 3),
    "vaporwave": ("grid_perspective", 6),
    # --- hiphop ---
    "after_hours_rnb": ("lounge", 0),
    "boom_bap": ("cassette", 0),
    "conscious_flow": ("halftone", 0),
    "g_funk": ("brushstrokes", 0),
    "london_grime": ("cityscape", 0),
    "melbourne_hiphop": ("cassette", 1),
    "swagger": ("cassette", 2),
    "trap_mode": ("waveform", 4),
    # --- soul_funk ---
    "decade_60s": ("circles", 0),
    "funk_disco": ("disco_ball", 1),
    "glasgow_soul": ("disco_ball", 2),
    "gospel": ("stained_glass", 0),
    "london_soul": ("disco_ball", 3),
    "melbourne_soul": ("disco_ball", 4),
    "motown_soul": ("disco_ball", 5),
    "neo_soul": ("vinyl_grooves", 1),
    "yacht_rock": ("sun_horizon", 0),
    # --- rock ---
    "autumn_embers": ("low_poly", 0),
    "britpop_rock": ("mod_target", 0),
    "classic_rock": ("amp_stack", 0),
    "emo_poppunk": ("guitar", 0),
    "garage_grunge": ("amp_stack", 1),
    "glasgow_anthems": ("amp_stack", 2),
    "glasgow_indie": ("guitar", 1),
    "indie_rock": ("guitar", 2),
    "london_britpop": ("mod_target", 1),
    "london_indie": ("guitar", 3),
    "melbourne_indie": ("guitar", 4),
    "melbourne_psych": ("prism", 0),
    "melbourne_pubrock": ("amp_stack", 3),
    "post_grunge": ("triangles", 0),
    "power_ballads": ("smoke", 0),
    "prog_rock": ("prism", 1),
    "psych_haze": ("geometric", 0),
    "rockabilly_surf": ("amp_stack", 4),
    "stoner_rock": ("desert", 0),
    # --- metal_punk ---
    "defiant": ("amp_stack", 5),
    "glasgow_postpunk": ("guitar", 5),
    "heavy_riffs": ("amp_stack", 6),
    "industrial": ("shards", 1),
    "london_calling": ("guitar", 6),
    "melbourne_garagepunk": ("guitar", 7),
    "melbourne_postpunk": ("guitar", 8),
    "punk_energy": ("guitar", 9),
    "rap_rock": ("shards", 0),
    # --- jazz ---
    "acid_jazz": ("jazz_club", 0),
    "autumn_jazz": ("jazz_club", 1),
    "bebop": ("jazz_club", 2),
    "blues_bar": ("smoke", 1),
    "bossa_samba": ("jazz_club", 3),
    "dinner": ("candle_glow", 0),
    "jazz_dinner": ("jazz_club", 4),
    "london_jazz": ("jazz_club", 5),
    "romantic_jazz": ("jazz_club", 6),
    "smooth_jazz": ("jazz_club", 7),
    "swing_bigband": ("brushstrokes", 1),
    "winter_jazz": ("jazz_club", 8),
    # --- classical ---
    "neoclassical": ("strings", 0),
    "spring_strings": ("strings", 1),
    "string_quartet": ("strings", 2),
    # --- folk ---
    "acoustic_romance": ("mountains", 0),
    "autumn_leaves": ("woodgrain", 0),
    "autumn_mix": ("mountains", 1),
    "bluegrass": ("woodgrain", 1),
    "campfire": ("pine_forest", 0),
    "celtic_folk": ("pine_forest", 1),
    "cooking_mix": ("gingham", 0),
    "country_roads": ("mountains", 2),
    "gardening": ("blossom", 0),
    "glasgow_folk": ("meadow", 0),
    "melbourne_folk": ("meadow", 1),
    "old_friends": ("woodgrain", 2),
    "outlaw_country": ("desert", 1),
    "road_trip": ("open_road", 0),
    "spring_acoustic": ("meadow", 2),
    "spring_bloom": ("meadow", 3),
    "spring_mix": ("meadow", 4),
    "summer_breeze": ("pine_forest", 3),
    "summer_roadtrip": ("open_road", 1),
    # --- latin_global ---
    "afrobeat": ("kente", 0),
    "beach_vibes": ("beach", 0),
    "decade_70s": ("vinyl_grooves", 0),
    "latin_heat": ("palm_sunburst", 1),
    "london_dub": ("beach", 1),
    "melbourne_sunset": ("palm_sunburst", 0),
    "reggae_dub": ("tropical_leaves", 0),
    "summer_tropical": ("beach", 2),
    # --- chiptune ---
    "chiptune": ("pixel_grid", 0),
    "gaming": ("pixel_grid", 1),
    # --- cinematic ---
    "cinematic_epic": ("film_strip", 0),
    "main_character": ("film_strip", 1),
    "soundtracks": ("film_strip", 2),
    # --- pop ---
    "brunch_mix": ("gingham", 1),
    "chart_pop": ("floating_circles", 0),
    "decade_10s": ("equalizer", 0),
    "decade_90s": ("diamond", 0),
    "fresh_start": ("halftone", 1),
    "happy": ("floating_circles", 1),
    "hyperpop": ("pixel_grid", 2),
    "indie_pop": ("diamond", 1),
    "london_mod": ("mod_target", 2),
    "sad_bangers": ("halftone", 2),
    "school_days": ("cassette", 3),
    "ska": ("checkerboard", 0),
    "spring_jangle": ("blossom", 1),
    "throwback_anthems": ("cassette", 4),
    # --- ambient ---
    "ambient_drift": ("cosmos", 0),
    "awe_wonder": ("cosmos", 1),
    "daydreaming": ("clouds", 0),
    "deep_cuts": ("aurora", 0),
    "downtempo": ("starfield", 1),
    "dreamy_mix": ("clouds", 1),
    "evening_unwind": ("aurora", 1),
    "glasgow_dream": ("clouds", 2),
    "glasgow_late": ("smoke", 2),
    "glasgow_postrock": ("crescendo", 0),
    "late_night": ("lounge", 1),
    "lofi_beats": ("cassette", 5),
    "london_triphop": ("smoke", 3),
    "melbourne_dream": ("clouds", 3),
    "midnight": ("moonlight", 0),
    "night_drive": ("cosmos", 2),
    "post_rock": ("crescendo", 1),
    "rediscovery": ("starfield", 2),
    "starlit": ("starfield", 0),
    "winter_nights": ("moonlight", 1),
    # --- atmos ---
    "autumn_rain": ("rainfall", 0),
    "blue_hour": ("clouds", 4),
    "clear_night": ("moonlight", 2),
    "foggy": ("smoke", 4),
    "frosty": ("snowfall", 0),
    "golden_hour": ("sun_horizon", 1),
    "grey_skies": ("clouds", 5),
    "heatwave": ("sun_horizon", 2),
    "morning": ("waves", 0),
    "overcast": ("clouds", 6),
    "rainy_day": ("ripples", 0),
    "snow_day": ("snowfall", 1),
    "stormy": ("clouds", 7),
    "summer_evening": ("waves", 1),
    "sunny": ("starburst", 0),
    "sunrise": ("sun_horizon", 3),
    "sunset_mix": ("sun_horizon", 4),
    "windy": ("clouds", 8),
    "winter_frost": ("snowfall", 2),
    "winter_mix": ("rainfall", 1),
    # --- romance ---
    "candlelight": ("candle_glow", 1),
    "crush": ("arc_sweep", 0),
    "date_night": ("arc_sweep", 1),
    "devotion": ("arc_sweep", 2),
    "first_date": ("brushstrokes", 2),
    "flirty": ("diamond", 2),
    "indie_romance": ("floating_circles", 2),
    "late_night_romance": ("bokeh", 0),
    "long_distance": ("candle_glow", 3),
    "love_songs": ("bokeh", 1),
    "loved_up": ("bokeh", 3),
    "modern_romance": ("bokeh", 2),
    "piano_romance": ("arc_sweep", 3),
    "romantic_dinner": ("candle_glow", 2),
    "romantic_mix": ("brushstrokes", 3),
    "slow_burn": ("smoke", 5),
    "slow_dance": ("diamond", 3),
    "strings_romance": ("strings", 3),
    "synthpop_romance": ("floating_circles", 3),
    "wedding_day": ("wedding_rings", 0),
    # --- party ---
    "all_time_favourites": ("circles", 1),
    "cookout": ("gingham", 2),
    "dinner_party": ("gingham", 3),
    "euphoric": ("confetti", 2),
    "festive": ("holiday_lights", 0),
    "friday_feeling": ("starburst", 2),
    "friday_night": ("disco_ball", 6),
    "game_night": ("circles", 2),
    "party_throwback": ("confetti", 3),
    "singalong": ("starburst", 3),
    "summer_heat": ("sun_horizon", 5),
    "treat_yourself": ("circles", 3),
    "weekend_mix": ("confetti", 4),
    # --- energetic ---
    "celebration": ("confetti", 0),
    "commute_mix": ("traffic", 0),
    "confidence_boost": ("lightning", 0),
    "driving_mix": ("open_road", 2),
    "driving_singalong": ("traffic", 1),
    "empowering": ("lightning", 1),
    "housework_hustle": ("shards", 2),
    "monday_motivation": ("lightning", 2),
    "party": ("laser_fan", 4),
    "running": ("motion", 0),
    "triumphant": ("lightning", 3),
    "walking_mix": ("motion", 1),
    "workout": ("motion", 2),
    # --- emotional ---
    "angst_mix": ("low_poly", 1),
    "bittersweet": ("triangles", 1),
    "cathartic": ("smoke", 6),
    "emotional": ("low_poly", 2),
    "grief_release": ("rainfall", 2),
    "heartbreak": ("waves", 2),
    "hopeful": ("triangles", 2),
    "melancholy": ("low_poly", 3),
    "memory_lane": ("cassette", 6),
    "moody_mix": ("rainfall", 3),
    "moving_on": ("waves", 3),
    "nostalgia_mix": ("triangles", 3),
    "restless": ("starfield", 3),
    "situationship": ("brushstrokes", 4),
    "sunday_scaries": ("low_poly", 4),
    "tender": ("rainfall", 4),
    "three_am": ("moonlight", 3),
    "vulnerable": ("waves", 4),
    "witching_hour": ("triangles", 4),
    "yearning": ("moonlight", 4),
    # --- calm ---
    "after_work": ("aurora", 2),
    "chill": ("aurora", 3),
    "cool_down": ("woodgrain", 3),
    "cosy": ("pine_forest", 2),
    "creative_flow": ("starfield", 4),
    "deep_reading": ("grid_paper", 0),
    "deep_work": ("grid_paper", 1),
    "focus": ("grid_paper", 2),
    "golden_afternoon": ("sun_horizon", 6),
    "lazy_sunday": ("aurora", 4),
    "meditation": ("zen", 0),
    "midweek_reset": ("woodgrain", 4),
    "power_nap": ("starfield", 5),
    "serene": ("aurora", 5),
    "spa_bath": ("zen", 1),
    "study_session": ("grid_paper", 3),
    "sunday_morning": ("clouds", 9),
    "wind_down": ("zen", 2),
    "winter_cosy": ("woodgrain", 5),
    "yoga_stretch": ("zen", 3),
    # --- place_scot ---
    "scotland_scene": ("tartan", 0),
    # --- place_lon ---
    "london_scene": ("cityscape", 1),
    # --- place_aus ---
    "australia_scene": ("sun_horizon", 7),
    # --- 3 geo HITS mixes: same family as the matching scene, distinct variant (star glyph distinguishes) ---
    "scottish_hits": ("tartan", 1),
    "scotland_now": ("tartan", 2),
    "london_hits": ("cityscape", 2),
    "london_now": ("cityscape", 3),
    "australian_hits": ("sun_horizon", 8),
    "australia_now": ("sun_horizon", 9),
    # --- UK (whole) scene + hits + now: the Union Jack family ---
    "uk_scene": ("union_jack", 0),
    "uk_hits": ("union_jack", 1),
    "uk_now": ("union_jack", 2),
    # --- misc ---
    "discover_weekly": ("radial", 0),
    "folk_acoustic": ("acoustic_guitar", 0),
    "on_repeat": ("chevrons", 0),
    "pre_party": ("confetti", 1),
    "repeat_rewind": ("geometric", 1),
    "sleep": ("moonlight", 5),
    "time_capsule": ("radial", 1),
}

# Per-profile icon overlay — drawn on top of the background before text.
# Keys not listed here get no icon (plain background only).
_PROFILE_ICON = {
    # Per-profile glyph — restrained: only where a clear literal symbol fits AND it doesn't
    # compete with the art. Romance hearts via _HEART_MODE; date_night/brunch_mix rotate.
    # autumn_leaves uses the multicolour falling-leaves treatment in _draw_icon_overlay.
    "sad_bangers": "nightlife", "decade_80s": "music_note_2", "decade_10s": "music_note",
    "foggy": "foggy", "snow_day": "snowflake",
    "frosty": "ac_unit", "study_session": "menu_book", "deep_reading": "menu_book",
    "gaming": "stadia_controller", "power_nap": "bedtime", "chiptune": "stadia_controller",
    "romantic_mix": "favorite", "modern_romance": "favorite", "late_night_romance": "favorite",
    "love_songs": "favorite", "slow_dance": "favorite", "date_night": "wine_bar",
    "first_date": "favorite", "acoustic_romance": "favorite", "indie_romance": "favorite",
    "synthpop_romance": "favorite", "heartbreak": "heart_broken",
    "brunch_mix": "brunch_dining", "rainy_day": "rainy",
    "fresh_start": "clear_day",
    "winter_mix": "snowflake", "spring_mix": "local_florist", "running": "directions_run",
    "lazy_sunday": "cloud", "evening_unwind": "self_improvement", "cool_down": "spa",
    "pre_party": "nightlife", "friday_night": "nightlife", "celebration": "flare",
    "weekend_mix": "weekend", "commute_mix": "route", "walking_mix": "directions_walk",
    "on_repeat": "repeat", "repeat_rewind": "replay", "release_radar": "radar",
    "rediscovery": "travel_explore", "discover_weekly": "explore", "time_capsule": "history",
    "crush": "favorite", "flirty": "favorite", "devotion": "favorite",
    "loved_up": "favorite", "long_distance": "favorite", "autumn_leaves": "eco",
    "heatwave": "thermostat", "cooking_mix": "skillet",
    "gospel": "church",
    "scottish_hits": "star", "australian_hits": "star", "london_hits": "star",
    "uk_hits": "star",
    "scotland_now": "radio", "london_now": "radio", "uk_now": "radio", "australia_now": "radio",
}

# Profiles whose icon rotates weekly — the per-ISO-week rng picks one each Monday.
_PROFILE_ICON_ROTATE = {
    "date_night": ["wine_bar", "local_bar", "brunch_dining"],
    "brunch_mix": ["brunch_dining", "bakery_dining", "egg_alt"],
}

# Per-glyph placement metadata for _draw_icon_overlay.
#   extent     ≈ glyph radius in px at scale 1.0 (size = 2*extent*scale → hero proportion)
#   base_scale size multiplier; tilt = max |rotation|° (0 = upright); anchor = (x,y) fractions
#   kind       "single" or "cluster" (music notes). Romance clusters are set via _HEART_MODE.
_ICON_DEFAULT_META = {"extent": 205, "base_scale": 1.0, "tilt": 10, "anchor": (0.50, 0.40), "kind": "single"}
_ICON_META = {
    # Music profiles render a composed note cluster (mixes music_note + music_note_2).
    "music_note":          {"kind": "cluster", "extent": 215},
    "beach_access":        {"tilt_bias": -16, "tilt": 5},   # umbrella leans clockwise
    "piano":               {"extent": 215},                 # a touch larger so the keys read
    # Concept / informational glyphs — upright, a touch more central.
    "repeat":              {"tilt": 0, "anchor": (0.50, 0.42)},
    "replay":              {"tilt": 0, "anchor": (0.50, 0.42)},
    "radar":               {"tilt": 0, "anchor": (0.50, 0.42)},
    "history":             {"tilt": 0, "anchor": (0.50, 0.42)},
    "travel_explore":      {"tilt": 0, "anchor": (0.50, 0.42)},
    "explore":             {"tilt": 0, "anchor": (0.50, 0.42)},
    "center_focus_strong": {"tilt": 0},
    "trending_up":         {"tilt": 6},
    "candle":              {"extent": 185, "tilt": 12, "anchor": (0.58, 0.45)},  # off-centre (lower-right) + tilted
    "star_shine":          {"extent": 220},
}

# Per-profile overrides (merged over _ICON_META) — a deliberate placement spread across
# same-glyph siblings (sun / moon profiles) or a bespoke size.
_ICON_PROFILE_OVERRIDE = {
    # round 3: keep glyphs clear of busy art (smaller / moved off the focal element)
    "first_date":     {"anchor": (0.30, 0.30), "base_scale": 0.66},   # hearts small, off the candle
    "heatwave":       {"anchor": (0.30, 0.28), "base_scale": 0.70},   # smaller, higher
    "long_distance":  {"anchor": (0.30, 0.27), "base_scale": 0.66},   # heart up-left, off the candle flame
    "decade_10s":     {"anchor": (0.50, 0.15), "base_scale": 0.55},   # notes up high, clear of the EQ bars
    "repeat_rewind":  {"anchor": (0.74, 0.42)},                       # glyph to the right
    "rediscovery":    {"anchor": (0.75, 0.62), "base_scale": 0.50},   # small, lower-right (declutter cosmos)
    "main_character": {"anchor": (0.50, 0.37), "base_scale": 1.12},
    "night_drive":    {"anchor": (0.62, 0.40)},
    "late_night":     {"anchor": (0.62, 0.41)},
    "sleep":          {"anchor": (0.50, 0.38)},
    "golden_hour":    {"anchor": (0.40, 0.37)},
    "sunset_mix":     {"anchor": (0.50, 0.36)},
    "summer_evening": {"anchor": (0.40, 0.37)},
    # Release Radar — small, lower-right quadrant, semi-transparent, no shadow.
    "release_radar":  {"anchor": (0.74, 0.60), "base_scale": 0.45, "alpha": 120, "shadow": False},
    # Rainy Day — cloud up in the top-right, mirrored (rain falls left), ripples drawn below it.
    "rainy_day":      {"anchor": (0.70, 0.25), "flip": True, "base_scale": 0.86},  # cloud up top; ripples land low
    # Party — Pre-Party lower-left, Friday Night lower-right (with scattered music notes).
    "pre_party":      {"anchor": (0.30, 0.62)},
    "friday_night":   {"anchor": (0.70, 0.62)},
    # Winter — cloud sits in the top-right quadrant; flakes fall below it (cloud drawn on top).
    "winter_mix":     {"anchor": (0.66, 0.33)},
    # Time Capsule — larger, up in the top-right quadrant; Rediscovery — larger, centred.
    "time_capsule":   {"anchor": (0.73, 0.27), "base_scale": 1.30},
    "rediscovery":    {"base_scale": 1.30},
    # Discover Weekly — small, tucked into the bottom-right quadrant.
    "discover_weekly": {"anchor": (0.74, 0.62), "base_scale": 0.45},
}

# Heart arrangement per romance profile — deliberately spread so no two covers match.
_HEART_MODE = {
    "romantic_mix":       "cluster",
    "love_songs":         "trio",
    "first_date":         "pair",
    "modern_romance":     "solitary",
    "late_night_romance": "cluster",
    "slow_dance":         "pair",
    "date_night":         "solitary",
    "acoustic_romance":   "solitary",
    "indie_romance":      "cluster",
    "synthpop_romance":   "pair",
}

# Mood-appropriate icon colour per glyph — the icon's natural / emotional colour, chosen to
# suit both the symbol and the vibe. Abstract concept glyphs (repeat, replay, radar, history,
# album, travel_explore, explore, route, directions_*) are intentionally omitted: they have no
# natural colour, so they fall back to an in-hue tint of the cover palette. _ensure_icon_contrast
# then nudges whatever colour is chosen just enough to stay legible on the background.
_ICON_COLOR = {
    "ac_unit": (180, 215, 235),
    "air": (190, 220, 220),
    "foggy": (170, 178, 190),
    "bedtime": (170, 160, 210),
    "hot_tub": (110, 200, 205),
    "menu_book": (210, 180, 130),
    "potted_plant": (120, 180, 90),
    "self_improvement": (110, 170, 200),
    "wb_twilight": (250, 180, 100),
    "headphones": (150, 140, 210),
    "mic": (240, 180, 90),
    "movie": (220, 190, 90),
    "stadia_controller": (90, 210, 130),
    # Love / warmth — rose-red
    "favorite":              (240,  90, 115),
    "heart_broken":          (214, 102, 122),
    "wine_bar":              (214, 116, 138),   # wine rose
    "local_bar":             (224, 140, 110),   # cocktail amber
    "brunch_dining":         (236, 196, 140),   # brunch warm
    "bakery_dining":         (232, 186, 128),   # pastry gold
    "egg_alt":               (246, 210, 150),   # sunny yolk
    # Fire / energy — orange-red
    "local_fire_department": (255, 125,  55),
    "whatshot":              (255, 135,  60),
    "fitness_center":        (255, 152,  92),
    "directions_run":        (255, 162,  98),
    # Sun / day / uplift — golden
    "wb_sunny":              (255, 200,  70),
    "clear_day":             (255, 210,  95),
    "mood":                  (255, 205,  85),   # happy smiley
    "star_shine":            (255, 214, 110),   # spotlight gold
    "trending_up":           (255, 206, 110),
    "weekend":               (255, 200, 120),
    # Night / moon — pale silver-blue
    "bedtime":               (224, 228, 246),
    "moon_stars":            (224, 228, 246),
    "partly_cloudy_night":   (210, 218, 240),
    "dark_mode":             (208, 214, 240),
    # Rain / water / storm — cool blue
    "rainy":                 (150, 198, 238),
    "water_drop":            (140, 196, 240),
    "thunderstorm":          (176, 192, 224),
    # Snow — icy white-blue
    "snowflake":             (224, 240, 255),
    # Dreamy / calm — soft lavender-white, spa
    "cloud":                 (222, 226, 244),
    "self_improvement":      (198, 222, 214),
    "spa":                   (190, 220, 212),
    "center_focus_strong":   (202, 216, 232),
    # Nature — autumn amber, spring pink, beach warm
    "forest":                (220, 138,  66),
    "local_florist":         (246, 158, 196),
    "beach_access":          (255, 178, 102),
    "sailing":               (222, 230, 240),   # white sail / nautical
    # Food / drink — warm creams
    "local_cafe":            (226, 196, 150),   # latte
    "restaurant":            (238, 200, 150),
    "skillet":               (236, 192, 132),
    # Candle — warm glow
    "candle":                (255, 198, 120),
    # Music — warm cream accent
    "music_note":            (244, 230, 206),
    "music_note_2":          (244, 230, 206),
    "piano":                 (246, 246, 250),   # white keys (two-tone detail draws the black keys)
    # Party / celebration — vivid festive
    "celebration":           (255, 140, 175),
    "festival":              (255, 150, 165),
    "nightlife":             (236, 150, 210),
    "flare":                 (255, 214, 130),   # festive gold spark
    # --- added in the cover redesign ---
    "eco":                   (130, 200,  96),   # leaf green
    "bolt":                  (255, 214,  90),   # electric yellow
    "palette":               (236, 210, 170),   # warm cream
    "casino":                (236, 110, 120),   # festive red
    "church":                (236, 206, 142),   # warm gold
    "thermostat":            (255, 150,  80),   # heat orange
    "album":                 (226, 216, 202),   # vinyl cream
    "redeem":                (242, 152, 182),   # gift pink
    "trophy":                (255, 210, 110),   # gold
    "star":                  (255, 210, 110),   # gold — geo HITS mixes
    "radio":                 (245, 246, 250),   # near-white — Scotland Now radio
    "outdoor_grill":         (246, 142,  82),   # grill orange
}

# Per-profile icon-colour overrides (mood nuance where a shared glyph serves different moods).
# Merged over _ICON_COLOR. Empty by default — add entries as specific covers want tuning.
_PROFILE_ICON_COLOR = {}

# Top Songs year-specific covers — 30 colours + 10 background styles cycling by year offset.
# Index: (year - top_songs_start_year) % 30
_TOP_SONGS_YEAR_PALETTES = [
    ((220,  60,  60), (155,  20,  20)),  #  0 — crimson red
    ((230, 105,  35), (170,  55,  10)),  #  1 — fire orange
    ((235, 185,  35), (175, 125,  10)),  #  2 — golden amber
    ((190, 205,  30), (130, 145,  10)),  #  3 — yellow-green
    ((100, 195,  45), ( 50, 135,  15)),  #  4 — lime
    (( 35, 175,  85), ( 12, 115,  45)),  #  5 — emerald
    (( 25, 180, 145), ( 10, 120,  95)),  #  6 — teal
    (( 25, 160, 195), (  8,  95, 145)),  #  7 — cyan
    (( 30, 120, 220), (  8,  55, 165)),  #  8 — cerulean blue
    (( 55,  70, 215), ( 20,  28, 160)),  #  9 — cobalt
    (( 90,  40, 205), ( 45,  12, 150)),  # 10 — electric indigo
    ((130,  28, 205), ( 80,   8, 150)),  # 11 — deep violet
    ((175,  30, 195), (120,  10, 140)),  # 12 — rich purple
    ((215,  40, 180), (155,  12, 125)),  # 13 — magenta
    ((225,  45, 130), (170,  15,  80)),  # 14 — hot pink
    ((225,  50,  90), (165,  18,  50)),  # 15 — rose red
    ((205,  75,  70), (150,  30,  30)),  # 16 — deep coral
    ((210, 120,  50), (150,  65,  15)),  # 17 — terracotta
    ((195, 160,  50), (135, 105,  12)),  # 18 — mustard
    ((155, 195,  55), ( 95, 135,  18)),  # 19 — warm yellow-green
    (( 70, 185,  85), ( 28, 125,  42)),  # 20 — spring green
    (( 28, 170, 150), ( 10, 110, 100)),  # 21 — seafoam
    (( 18, 140, 205), (  6,  78, 150)),  # 22 — sky blue
    (( 58,  88, 215), ( 22,  42, 160)),  # 23 — royal blue
    ((108,  48, 215), ( 55,  18, 160)),  # 24 — purple-violet
    ((185,  42, 190), (130,  12, 135)),  # 25 — purple-magenta
    ((215,  48, 120), (160,  15,  70)),  # 26 — deep rose
    ((215, 115,  42), (155,  62,  12)),  # 27 — warm orange
    ((215, 185,  38), (155, 128,  10)),  # 28 — yellow gold
    (( 52, 185, 115), ( 18, 125,  62)),  # 29 — mint green
]
# All 13 generators with their maximum v value.  Used for year-seeded random
# selection so every year gets a truly unique style+variant combination.
_TOP_SONGS_STYLE_POOL = [
    ("geometric",        8),
    ("circles",          6),
    ("radial",           6),
    ("waves",            9),
    ("floating_circles", 6),
    ("rays",             5),
    ("arc_sweep",        4),
    ("aurora",           7),
    ("triangles",        6),
    ("diamond",          3),
    ("starburst",        4),
    ("chevrons",         2),
    ("spiral",           2),
]

# Top Songs [Year] — 20 curated, mutually-distinct covers ((top, bottom), style, variant), indexed
# by `year % 20`. WHY curated (not random per year as before): guarantees 20 visibly-different,
# individually-bold "year in review" covers that never collide; glyph-free (the big year is the hero).
_TOP_SONGS_COVERS = [
    (((220,  60,  60), (120, 20, 30)), "starburst",        0),   # crimson burst
    (((235, 130,  40), (150, 50, 10)), "sun_horizon",      2),   # orange sunset
    (((240, 190,  50), (170, 110, 10)), "confetti",        0),   # gold confetti
    (((150, 200,  60), ( 60, 120, 30)), "equalizer",       0),   # lime EQ
    (((60,  200, 120), ( 20, 100, 60)), "circuit",         0),   # green circuit
    (((40,  200, 200), ( 15,  90, 110)), "concentric_pulse", 0), # teal pulse
    (((60,  150, 235), ( 20,  60, 150)), "grid_perspective", 0), # blue grid
    (((120, 110, 235), ( 50,  40, 150)), "starfield",      0),   # indigo starfield
    (((180,  90, 220), ( 90,  30, 140)), "laser_fan",      0),   # violet lasers
    (((230,  80, 200), (130,  20, 120)), "halftone",       0),   # magenta halftone
    (((240,  90, 150), (150,  30,  90)), "waveform",       0),   # pink waveform
    (((255, 120,  90), (170,  50,  50)), "triangles",      0),   # coral triangles
    (((210, 150,  70), (110,  70,  30)), "vinyl_grooves",  0),   # amber vinyl
    (((90,  190, 160), ( 30, 100,  90)), "mountains",      0),   # mint mountains
    (((230, 170,  60), ( 60,  80, 140)), "marquee_lights", 0),   # gold marquee
    (((200,  70, 110), ( 70,  25,  60)), "spiral",         0),   # rose spiral
    (((70,  170, 210), ( 25,  80, 120)), "chevrons",       0),   # sky chevrons
    (((180, 200,  80), ( 90, 110,  30)), "low_poly",       0),   # chartreuse low-poly
    (((240, 120, 180), (120,  40, 120)), "floating_circles", 0), # bubblegum circles
    (((110, 205, 205), ( 40, 110, 120)), "aurora",         0),   # aqua aurora
]

# _MOOD_PROFILE_KEYS is defined after _MOOD_MIX_NAMES below.


# ---------------------------------------------------------------------------
# Playlist descriptions — picked randomly each run for variety
# ---------------------------------------------------------------------------
_DESCRIPTIONS = {
    # --- Meloday+ gap-fill mixes ---
    "situationship": ["Almost something.", "Mixed signals on repeat.", "Undefined, unresolved, undeniable.", "Neither here nor there."],
    "sad_bangers": ["Cry on the dancefloor.", "Heartbreak you can dance to.", "Sad songs with a beat.", "Big feelings, bigger hooks."],
    "power_ballads": ["Lighters up.", "Big, dramatic, all heart.", "The ones with the key change.", "Sing it to the rafters."],
    "restless": ["Can't switch off.", "Wired and racing.", "Too much energy, nowhere to put it.", "Restless, on the edge."],
    "neoclassical": ["Quiet, modern, contemplative.", "Piano, strings, stillness.", "Arnalds, Richter, Einaudi and kin.", "Space to think."],
    "yacht_rock": ["Smooth sailing.", "Soft rock, top down.", "Polished, warm, breezy.", "Mellow gold."],
    "swagger": ["Walk in like you own it.", "Pure attitude.", "Cocky, stylish, untouchable.", "Strut to this."],
    "chart_pop": ["The songs everyone knows.", "Chart-toppers and earworms.", "Pure pop, all hits.", "Radio-ready."],
    "dance_pop": ["Pop you can dance to.", "Hands up.", "Uptempo pop bangers.", "Move."],
    "indie_pop": ["Jangly, bright, indie pop.", "Cooler than the charts.", "Hooks with heart.", "Sunny indie."],
    "synth_pop": ["Synths and hooks.", "Neon-lit electropop.", "80s sheen, modern pulse.", "Glittering synth-pop."],
    # rock / electronic / scores gap-fill
    "indie_rock": ["Guitar-forward indie and alt-rock.", "Anthems, hooks and jangle.", "Big choruses, skinny ties.", "Indie at full volume."],
    "post_grunge": ["Post-grunge radio rock.", "Loud-quiet-loud, big choruses.", "Flannel hooks, full volume.", "The 2000s rock that ruled the radio."],
    "rap_rock": ["Rap-rock and nu-metal energy.", "Riffs, rhymes and attitude.", "Drop the bass and break stuff.", "Turn-of-the-millennium aggression."],
    "festival_edm": ["Big-room, main-stage EDM.", "Hands up — festival drops.", "Build, drop, repeat.", "The anthems that close the set."],
    "soundtracks": ["Scores and soundtracks, big screen to game.", "Sweeping themes and end credits.", "Cinematic and orchestral, score-forward.", "Music from the movies, shows and games."],
    "rave_cave": ["Donk, hardstyle and hard trance.", "Hands up — full send.", "Fast, loud, relentless.", "Rave till the lights come up."],
    # ---- 7 decade mixes (era) ----
    "decade_60s": ["The sound of the sixties.", "Where it all kicked off.", "Sixties gold."],
    "decade_70s": ["Seventies grooves and gold.", "Flares, funk and rock.", "The sound of the seventies."],
    "decade_80s": ["Big hair, bigger choruses.", "Synths, neon and shoulder pads.", "Pure eighties.", "The decade that never left."],
    "decade_90s": ["Nineties, all day.", "The sound of the nineties.", "Back to the '90s."],
    "decade_00s": ["The sound of the noughties.", "Two-thousands throwbacks.", "Y2K and beyond."],
    "decade_10s": ["The 2010s on repeat.", "Last decade's biggest.", "The sound of the 2010s."],
    "decade_20s": ["The sound of right now.", "This decade so far.", "Twenty-twenties fresh."],
    "stormy": ["For when the sky cracks open.", "Thunder, lightning and drama.", "Big, dark, electric skies."],
    "foggy": ["Lost in the mist.", "Soft-focus, low-visibility calm.", "Music for a world gone grey and still."],
    "snow_day": ["Snow's falling — stay in.", "Hushed, white-blanket calm.", "Watch the flakes come down."],
    "heatwave": ["Too hot to move.", "Shimmering, sun-baked and slow.", "Melt into the heat."],
    "frosty": ["Cold, clear and crystalline.", "Frost on the glass, breath in the air.", "Crisp and bright and freezing."],
    "grey_skies": ["Flat grey overhead.", "Muted music for a colourless day.", "Cloud cover, inside and out."],
    "windy": ["Lean into the gusts.", "Blustery, restless and brisk.", "The wind's really up."],
    "clear_night": ["Cloudless, starlit and still.", "Look up — the whole sky's out.", "A calm, clear night."],
    "festive": ["Deck the halls.", "All the Christmas classics.", "'Tis the season — holiday favourites."],
    "spring_bloom": ["Everything's in bloom.", "Bright, fresh spring pop.", "The first warm days, in song."],
    "spring_acoustic": ["Acoustic songs for a spring morning.", "Folk, fresh air and new leaves.", "Unplugged and budding."],
    "spring_strings": ["Strings to greet the thaw.", "Graceful, hopeful, orchestral spring.", "Music blooming like the season."],
    "spring_jangle": ["Jangly guitars and spring sun.", "Indie-pop with a skip in its step.", "Twee, bright and blooming."],
    "summer_heat": ["Dancefloor in the sun.", "Disco-funk summer heat.", "Sweat it out under the lights."],
    "summer_breeze": ["Top down, breeze blowing.", "Smooth, sun-warmed soft rock.", "Easy yacht-rock summer."],
    "summer_roadtrip": ["Windows down, full volume.", "Summer singalongs for the open road.", "Every chorus, top of your lungs."],
    "summer_tropical": ["Palm trees and warm rhythms.", "Latin, reggae and tropical heat.", "Beach-bar summer grooves."],
    "autumn_leaves": ["Crunching leaves and cardigans.", "Folk songs for falling leaves.", "Acoustic, amber autumn."],
    "autumn_jazz": ["Warm jazz for cool evenings.", "Smoky, soulful autumn.", "A glass of red and a saxophone."],
    "autumn_rain": ["Rain on falling leaves.", "Moody, grey-skied autumn.", "Wistful songs for wet windows."],
    "autumn_embers": ["Bonfires and worn-in rock.", "Heartland rock for autumn nights.", "Warm, gritty and golden."],
    "winter_frost": ["Icy, weightless and still.", "Ambient frost and clear air.", "Crystalline winter calm."],
    "winter_cosy": ["Fireside warmth and soul.", "Cosy, intimate winter R&B.", "Blankets, candles and slow jams."],
    "winter_nights": ["Long, dark, electronic nights.", "Downtempo for the deep midwinter.", "Hypnotic after-dark winter."],
    "winter_jazz": ["Cool jazz for cold nights.", "Smooth, snowed-in standards.", "A nightcap and a slow tune."],
    "hopeful": ["Songs that look on the bright side.", "A lift when you need one — bright and hopeful.", "Optimism, one track at a time."],
    "yearning": ["For when you want what you can't quite reach.", "Aching, wistful, full of longing.", "The sound of quiet yearning."],
    "triumphant": ["Chest-out, fists-up victory songs.", "The comeback, the win, the big finish.", "Music for your montage moment."],
    "serene": ["Calm, settled, completely at ease.", "Lower your shoulders and breathe.", "Stillness you can sink into."],
    "tender": ["Soft, warm and full of heart.", "Gentle songs that hold you close.", "Tenderness, turned up just enough."],
    "defiant": ["Chin up, no apologies.", "Songs for standing your ground.", "Defiant, fierce and unbothered."],
    "vulnerable": ["Bare, honest and quietly raw.", "The songs you play with the lights low.", "Open-hearted and unguarded."],
    "awe_wonder": ["Music that makes the world feel vast.", "Goosebumps and wide-open skies.", "Wonder, awe and the sublime."],
    "grief_release": ["For the heaviest days — and letting go.", "Sit with it; let it out.", "Songs for grief, and for release."],
    "sunrise": ["First light, slow and golden.", "The world waking up.", "Easy songs for a new morning."],
    "blue_hour": ["That hush between day and night.", "Twilight blues and long shadows.", "The quiet glow of dusk."],
    "midnight": ["The small hours, lights low.", "After everyone's asleep.", "Deep, still and nocturnal."],
    "three_am": ["Wide awake when you shouldn't be.", "Restless thoughts at 3am.", "The soundtrack to insomnia."],
    "golden_afternoon": ["Sun-warmed and in no hurry.", "Hazy, golden, easy hours.", "A lazy afternoon in the light."],
    "overcast": ["Grey skies and inward thoughts.", "Muted songs for a flat, cloudy day.", "Low light, low key."],
    "starlit": ["Lie back and look up.", "Vast, shimmering and far away.", "Music under an open sky."],
    "witching_hour": ["Something's stirring in the dark.", "Mysterious, eerie and a little strange.", "The hour the shadows move."],
    "monday_motivation": ["Reset, refocus, get after it.", "Beat the Monday blues.", "Fuel for a fresh start to the week."],
    "midweek_reset": ["Hump-day steady-as-she-goes.", "Find your second wind.", "Keep the week rolling."],
    "friday_feeling": ["The week's done — let's go.", "That Friday-afternoon buzz.", "Clock off and turn it up."],
    "sunday_scaries": ["That Sunday-evening sinking feeling.", "Soft songs for the weekend's last hours.", "Easing into the week ahead."],
    "treat_yourself": ["You earned it.", "A little luxury, a lot of attitude.", "Songs for spoiling yourself."],
    "dinner_party": ["Good food, good company, good taste.", "Smooth, grown-up background warmth.", "Set the table and pour the wine."],
    "housework_hustle": ["Blast it and blitz the chores.", "Make cleaning almost fun.", "Mop, dust, dance, repeat."],
    "study_session": ["Heads down, focus on.", "Low-distraction songs for deep work.", "Get in the zone and stay there."],
    "wind_down": ["Ease off and decompress.", "The slow descent into evening.", "Unspool the day, gently."],
    "yoga_stretch": ["Flow, breathe, stretch it out.", "Calm movement, calm mind.", "Find your balance."],
    "meditation": ["Be still. Just breathe.", "Beatless calm for a quiet mind.", "Drift inward."],
    "deep_reading": ["Lose yourself in a good book.", "Quiet, wordless company for reading.", "Turn the page, mind at ease."],
    "creative_flow": ["Get in the zone and make something.", "Songs that keep ideas moving.", "Fuel for the flow state."],
    "gaming": ["Locked in, controller in hand.", "High-energy fuel for the grind.", "Game on."],
    "gardening": ["Hands in the soil, sun on your back.", "Easy songs for the garden.", "Grow something good."],
    "spa_bath": ["Run the bath, dim the lights.", "Pure, warm, weightless calm.", "Soak it all away."],
    "power_nap": ["Twenty minutes of soft drift.", "Recharge without falling all the way under.", "A gentle reset."],
    "throwback_anthems": ["The bangers you grew up on.", "Old favourites, full volume.", "Throw it back."],
    "old_friends": ["Songs that feel like an old mate.", "For catch-ups and long laughs.", "The crew, the memories, the tunes."],
    "campfire": ["Gather round, pass the guitar.", "Acoustic warmth under the stars.", "Songs for the embers."],
    "cookout": ["Fire up the grill.", "Backyard, sunshine, good people.", "Summer cookout soundtrack."],
    "game_night": ["Snacks out, scores settled.", "Light, fun background for the table.", "Let the games begin."],
    "singalong": ["Everybody knows the words.", "Top-of-your-lungs anthems.", "Sing it loud."],
    "school_days": ["Back to the bus and the bell.", "The songs of your school years.", "Young, loud and carefree."],
    "memory_lane": ["A gentle walk through old memories.", "Faded photos, warm feelings.", "Take the long way back."],
    "crush": ["That giddy, can't-stop-smiling feeling.", "New crush, butterflies and all.", "Falling, fast."],
    "slow_burn": ["Tension that builds, slow and sweet.", "The long, smouldering wait.", "Take it slow."],
    "moving_on": ["Closing one chapter, opening the next.", "Healing, and heading forward.", "Better off, and you know it."],
    "loved_up": ["Head over heels and glowing.", "Pure loved-up bliss.", "Everything's better with them."],
    "long_distance": ["Miles apart, close at heart.", "Songs for missing someone.", "Counting down to next time."],
    "flirty": ["A little wink, a little tease.", "Light, fun and flirtatious.", "Turn on the charm."],
    "devotion": ["I'm yours — all in.", "Songs of deep, steady love.", "Devotion, plain and true."],
    "wedding_day": ["First dances and happy tears.", "The big day, set to music.", "Here's to forever."],
    "funk_disco": ["Get down — funk, disco and pure groove.", "Mirrorball-ready funk and disco.", "Strut, shimmer and let the bassline lead."],
    "neo_soul": ["Smooth, slow-burning soul.", "Quiet storm — velvet neo-soul.", "Candlelit soul with a slow, deep pocket."],
    "motown_soul": ["Classic soul that moves you.", "Motown gold and timeless soul.", "Handclaps, horns and Hitsville magic."],
    "after_hours_rnb": ["Late-night, lights-low R&B.", "Slow jams for after hours.", "Lights low, slow jams on repeat."],
    "acid_jazz": ["Where jazz meets the dancefloor.", "Funky, jazzy, effortlessly cool.", "Jazzy chords over a club-ready groove."],
    "boom_bap": ["Head-nodding 90s boom bap.", "Crates, breaks and bars.", "Boom, bap and a fat dusty break."],
    "conscious_flow": ["Thoughtful rap with something to say.", "Jazzy, conscious hip-hop.", "Bars with a message, beats with soul."],
    "g_funk": ["Top down, West Coast bounce.", "G-funk whine and laid-back flows.", "Synth whine, low-riding West Coast cool."],
    "trap_mode": ["Hi-hats, 808s and menace.", "Hard trap and drill energy.", "808s, hi-hats and pure menace."],
    "lofi_beats": ["Dusty beats to relax/study to.", "Lo-fi loops and mellow haze.", "Crackle, haze and a head-nod loop."],
    "house_party": ["Four-on-the-floor, all night.", "Feel-good house grooves.", "Four-on-the-floor till the lights come up."],
    "deep_house": ["Deep, hypnotic, late-night house.", "Sub-bass and 3am grooves.", "Deep, rolling grooves for the small hours."],
    "techno": ["Relentless, hypnotic techno.", "Warehouse pressure, all night.", "Steel, smoke and relentless 4/4."],
    "trance": ["Hands-up, eyes-closed trance.", "Uplifting trance euphoria.", "Big builds, bigger drops, hands in the air."],
    "dnb": ["Rolling breaks and deep bass.", "174 — drum & bass rollers.", "Breakbeats and sub-bass at full tilt."],
    "bass_drop": ["Wobble, weight and the drop.", "Dubstep and heavyweight bass.", "Brace for the wobble and the weight."],
    "uk_garage": ["Skippy 2-step and garage swing.", "UKG — pirate-radio bounce.", "Skippy beats and a basement bounce."],
    "synthwave": ["Neon nights and 80s synths.", "Retro-futurist synthwave drive.", "Neon, chrome and an endless night drive."],
    "industrial": ["Cold, mechanical, relentless.", "Industrial clang and EBM pulse.", "Clang, grind and machine-shop menace."],
    "vaporwave": ["Hazy, melted retro nostalgia.", "Vaporwave drift and pastel synths.", "Melted malls and pastel nostalgia."],
    "downtempo": ["Slow, deep electronic drift.", "Downtempo grooves after dark.", "Slow, deep and hypnotic after dark."],
    "hyperpop": ["Maximalist, candy-coloured chaos.", "Hyperpop sugar rush and glitch.", "Sugar-rush chaos turned up to 11."],
    "classic_rock": ["Turn it up — classic rock anthems.", "Guitar heroes and stadium riffs.", "Big riffs, bigger choruses, no apologies."],
    "heavy_riffs": ["Down-tuned riffs and full power.", "Heavy metal, loud and proud.", "Down-tuned, dialled-up and heavy."],
    "punk_energy": ["Fast, loud, three chords.", "Punk — no rules, all energy.", "Three chords and the truth, fast."],
    "garage_grunge": ["Fuzzed-out garage and grunge.", "Raw riffs and flannel energy.", "Fuzz, flannel and feedback."],
    "emo_poppunk": ["Heart-on-sleeve, full volume.", "Emo anthems and pop-punk hooks.", "Heart on sleeve, foot on distortion."],
    "britpop_rock": ["Swaggering Britpop singalongs.", "Madchester groove and 90s guitars.", "Swagger, hooks and a knowing wink."],
    "blues_bar": ["Smoky, late-night blues.", "Electric blues and barroom soul.", "Smoke, whiskey and a wailing guitar."],
    "psych_haze": ["Swirling, reverb-soaked haze.", "Psychedelic and shoegaze drift.", "Swirling reverb and melting colours."],
    "prog_rock": ["Ambitious, sprawling art-rock.", "Prog epics and odd time signatures.", "Odd meters and epic suites."],
    "stoner_rock": ["Fuzzed-out, sun-baked riffs.", "Stoner grooves and desert haze.", "Fuzzed-out, sun-baked and heavy."],
    "reggae_dub": ["Easy skank and deep dub.", "Reggae riddims and dub echoes.", "One drop, deep echo, easy skank."],
    "afrobeat": ["Polyrhythmic Afrobeat fire.", "Horns, percussion and groove.", "Horns, polyrhythms and unstoppable groove."],
    "latin_heat": ["Salsa, cumbia and reggaetón heat.", "Turn it up — Latin fire.", "Brass, percussion and pure fuego."],
    "bossa_samba": ["Sunlit bossa and samba sway.", "Brazilian warmth and swing.", "Sunlit sway from Rio to your room."],
    "celtic_folk": ["Fiddles, reels and folk fire.", "Celtic and traditional folk.", "Fiddles, reels and a roaring chorus."],
    "ska": ["Pick it up — ska and 2-tone.", "Off-beat horns and skank.", "Pick it up — off-beat horns and skank."],
    "bebop": ["Fast, fearless bebop.", "Hard bop fire and virtuosity.", "Dizzying runs and fearless swing."],
    "swing_bigband": ["Swing out — big band brass.", "Jump, jive and swing.", "Brass, bounce and a packed dancefloor."],
    "smooth_jazz": ["Easy, smooth and unhurried.", "Lounge jazz and cocktail cool.", "Mellow, polished and unhurried."],
    "country_roads": ["Windows down, country roads.", "Modern country and honky-tonk.", "Open highway, tailgate down."],
    "outlaw_country": ["Whiskey-soaked outlaw country.", "Alt-country grit and twang.", "Dust, grit and a rebel heart."],
    "bluegrass": ["Banjo rolls and fiddle runs.", "High, lonesome bluegrass.", "Banjo rolls and high lonesome harmony."],
    "rockabilly_surf": ["Twang, reverb and rockabilly roll.", "Surf's up — retro rock & roll.", "Twang, reverb and a greaser grin."],
    "cinematic_epic": ["Sweeping, big-screen scores.", "Epic, cinematic orchestral drama.", "Strings soaring over the big screen."],
    "ambient_drift": ["Weightless ambient and drone.", "Beatless drift and deep calm.", "Weightless tones and endless space."],
    "post_rock": ["Slow builds to soaring peaks.", "Instrumental post-rock crescendos.", "Quiet to thunder, the long crescendo."],
    "chiptune": ["Press start — 8-bit anthems.", "Chiptune and video-game gold.", "Pixel bleeps and boss-fight energy."],
    "gospel": ["Hands up — gospel joy.", "Soaring choirs and spirit.", "Lift your voice and raise the roof."],
    "glasgow_folk": ["Scottish folk, fireside-warm.", "Glasgow's folk heart.", "Fireside songs from the Clyde."],
    "glasgow_dream": ["Reverb-washed Glasgow dream-pop.", "Hazy, jangling and wistful.", "Jangle and haze, Glasgow-grey skies."],
    "glasgow_indie": ["Twee, jangly Glasgow indie.", "Belle & Sebastian's hometown sound.", "Wry, tender and impossibly charming."],
    "glasgow_soul": ["Glasgow's blue-eyed soul.", "Northern soul and funk warmth.", "Northern soul stomp with a Scottish heart."],
    "glasgow_postrock": ["Mogwai-loud, slow-burning swells.", "Glasgow post-rock crescendos.", "Mogwai-loud, slow and overwhelming."],
    "glasgow_anthems": ["Franz Ferdinand-style art-rock swagger.", "Glasgow guitar anthems.", "Art-school swagger and dancefloor riffs."],
    "glasgow_synth": ["Chvrches-bright Glasgow synth-pop.", "Neon synths and big hooks.", "Big synths, bigger feelings."],
    "glasgow_postpunk": ["Postcard-era Glasgow post-punk.", "Wiry, angular and restless.", "Wiry, angular Postcard-era cool."],
    "glasgow_house": ["Optimo-style left-field house.", "Glasgow's dancefloor underground.", "Left-field grooves from the Sub Club."],
    "glasgow_underground": ["Sub Club pressure — Glasgow techno.", "Subterranean, relentless and deep.", "Subterranean techno pressure."],
    "glasgow_bass": ["LuckyMe wonky, maximal bass.", "Hudson Mohawke-style future bass.", "Wonky, maximal, LuckyMe-bright bass."],
    "glasgow_late": ["After the club — Glasgow downtempo.", "Late, low and hypnotic.", "After the club, the comedown glow."],
    "london_dub": ["Notting Hill dub and lovers rock.", "London's reggae soundsystem.", "Soundsystem weight from Notting Hill."],
    "london_soul": ["Brit-soul à la Winehouse and Sade.", "Smoky London soul.", "Smoky Brit-soul, late and lovelorn."],
    "london_jazz": ["The new London jazz wave.", "Ezra Collective broken-beat jazz.", "The new wave, broken and brilliant."],
    "london_triphop": ["Smoky, cinematic trip-hop.", "London after midnight.", "Rain-soaked, cinematic and cool."],
    "london_mod": ["Sharp-dressed 60s London mod.", "The Kinks, The Who and Carnaby cool.", "Sharp suits and Carnaby swagger."],
    "london_britpop": ["Blur-vs-Oasis London Britpop.", "Cigarettes, alcohol and big choruses.", "Cigarettes, alcohol and big choruses."],
    "london_indie": ["Libertines-era London indie.", "Skinny jeans and jagged guitars.", "Jagged guitars and skinny-jean cool."],
    "london_calling": ["The only band that matters — London punk.", "Clash-era riot energy.", "Riot energy and pavement poetry."],
    "london_garage": ["Pirate-radio London garage.", "Skippy 2-step swing.", "Pirate-radio 2-step and bassline."],
    "london_grime": ["140 bpm — London grime.", "Skepta, Stormzy, square-wave bass.", "140, square-wave bass, all fire."],
    "london_dubstep": ["Croydon weight — London dubstep.", "Half-step pressure and sub-bass.", "Croydon half-step and deep sub."],
    "london_jungle": ["Amen breaks and ragga jungle.", "London's 174 pressure.", "Amen breaks and ragga pressure."],
    "melbourne_folk": ["Melbourne's folk songwriters.", "Acoustic warmth, southern skies.", "Acoustic warmth under southern skies."],
    "melbourne_dream": ["Hazy Melbourne dream-pop.", "Jangle, reverb and warmth.", "Reverb, jangle and golden haze."],
    "melbourne_soul": ["Future-soul à la Hiatus Kaiyote.", "Melbourne's neo-soul groove.", "Future-soul groove with a jazz heart."],
    "melbourne_sunset": ["Rooftop, golden-hour indie.", "Sun-soaked Melbourne summer.", "Rooftop, golden hour, salt air."],
    "melbourne_indie": ["Courtney Barnett-style slacker indie.", "Wry, jangly Melbourne guitars.", "Slacker charm and deadpan wit."],
    "melbourne_pubrock": ["Beer-soaked Aussie pub rock.", "Cold Chisel singalong energy.", "Beer-barn riffs and singalong roar."],
    "melbourne_hiphop": ["Aussie flows and laid-back beats.", "Melbourne hip-hop.", "Laid-back flows, sun-warmed beats."],
    "melbourne_postpunk": ["Birthday Party-dark post-punk.", "Melbourne's gothic underbelly.", "Birthday Party-dark and brooding."],
    "melbourne_psych": ["King Gizzard-style psych churn.", "Melbourne's garage-psych engine.", "Garage-psych churn, eyes spinning."],
    "melbourne_garagepunk": ["Amyl & the Sniffers snarl.", "Melbourne garage-punk fury.", "Snarl, sweat and three chords."],
    "melbourne_club": ["Melbourne bounce, peak-time.", "Big-room club energy.", "Peak-time bounce, hands up."],
    "melbourne_techno": ["Warehouse techno, southside.", "Hypnotic Melbourne techno.", "Hypnotic, southside warehouse techno."],
    "on_repeat": [
        "The songs you can't stop playing right now.",
        "Your most played, on heavy rotation.",
        "Can't get enough of these right now.",
        "The tracks you keep coming back to.",
        "Your current obsessions, all in one place.",
    ],
    "repeat_rewind": [
        "The songs that defined your recent past.",
        "Back to what you couldn't stop playing a few weeks ago.",
        "Your recent obsessions, revisited.",
        "What you were listening to last month.",
        "A look back at what had you hooked.",
    ],
    "release_radar": [
        "New music matched to your taste. Updated every Friday.",
        "Fresh releases, picked for you. Updated every Friday.",
        "The latest from artists you love and more. Updated every Friday.",
        "What's new in your world. Updated every Friday.",
        "New music Friday, tailored to you.",
    ],
    "discover_weekly": [
        "Your weekly mixtape of fresh music, chosen just for you. Updated every Monday.",
        "Music you haven't heard yet but probably will love. Updated every Monday.",
        "Hand-picked tracks you've never played. Updated every Monday.",
        "Something new for your ears, every Monday.",
        "Fresh picks from outside your usual rotation. Updated every Monday.",
    ],
    "rediscovery": [
        "Songs you used to love. Time to rediscover them.",
        "Old favourites you haven't heard in a while.",
        "Dust these off — you used to play them all the time.",
        "Music you loved and forgot about. Until now.",
        "They've been waiting for you to come back.",
    ],
    "time_capsule": [
        "A journey back to {era}. Made just for you.",
        "Music from {era} that helped shape your taste.",
        "Taking you back to {era}.",
        "Your soundtrack from {era}.",
        "The songs that defined {era} for you.",
    ],
    "time_machine": [
        "Back to {era}.",
        "What you were playing, {era}.",
        "Rewind to {era}.",
        "Your {era}, on repeat.",
        "This time, {era}.",
    ],
    "deep_cuts": [
        "Go deeper with the artists you love.",
        "The tracks your favourite artists don't get enough credit for.",
        "Hidden gems from artists you play all the time.",
        "Less played. Just as good.",
        "Beyond the obvious, from the artists you know best.",
    ],
    "top_songs": [
        "Your most played songs of {year}.",
        "The tracks that defined your {year}.",
        "What you couldn't stop playing in {year}.",
        "A look back at everything you loved in {year}.",
        "Your {year} in music.",
    ],
    "all_time_favourites": [
        "Your all-time most played tracks.",
        "The songs you've always come back to.",
        "Your personal greatest hits.",
        "The tracks that have stood the test of time.",
        "Everything you keep playing, year after year.",
    ],
    "workout": [
        "Time to put in the work.",
        "Push harder with these tracks.",
        "Music to match your intensity.",
        "Keep moving. Keep pushing.",
        "Your training soundtrack.",
    ],
    "running": [
        "Keep your pace up.",
        "Music matched to your stride.",
        "Every kilometre, these tracks.",
        "Run further. Run faster.",
        "Built for the road.",
    ],
    "party": [
        "Get the party started.",
        "Music that moves you.",
        "Turn it up.",
        "Dance. Repeat.",
        "The floor is yours.",
    ],
    "happy": [
        "Good vibes only.",
        "Music to lift your mood.",
        "Because today is a good day.",
        "Bright tracks for bright moments.",
        "Put a smile on it.",
    ],
    "morning": [
        "Start your day right.",
        "Music to ease you in.",
        "A gentle start to the day.",
        "Rise and press play.",
        "Good morning.",
    ],
    "focus": [
        "Music to help you concentrate.",
        "Zone in.",
        "Clear your head. Get it done.",
        "Deep work, deeper sound.",
        "Find your focus.",
    ],
    "dinner": [
        "The perfect soundtrack for dinner.",
        "Music for the table.",
        "Easy listening for easy evenings.",
        "Set the mood.",
        "Slow down and savour it.",
    ],
    "chill": [
        "Take it easy.",
        "Relax. You've earned it.",
        "Laid-back sounds for downtime.",
        "Nothing to do. Nowhere to be.",
        "Just breathe.",
    ],
    "rainy_day": [
        "For when the skies are grey.",
        "Music for a quiet day indoors.",
        "Rain on the window. Tea in hand.",
        "Overcast and reflective.",
        "Let the weather in.",
    ],
    "melancholy": [
        "Music for deeper moods.",
        "Sometimes you need to feel it.",
        "Sit with it for a while.",
        "Dark, honest, real.",
        "For when the mood is heavy.",
    ],
    "late_night": [
        "For the late-night hours.",
        "The city's asleep. You're not.",
        "After midnight.",
        "Dark and danceable.",
        "The night is still young.",
    ],
    "sleep": [
        "Wind down and drift off.",
        "Quiet sounds for quiet moments.",
        "Let it fade to sleep.",
        "Slow, soft, still.",
        "Close your eyes.",
    ],
    "sunny": [
        "Turn up the sunshine.",
        "Music as bright as the day.",
        "Good weather, good music.",
        "Windows down. Volume up.",
        "The sun's out.",
    ],
    "cosy": [
        "Warm music for cold days.",
        "Stay in. Turn it on.",
        "Warm sounds for cold weather.",
        "Pull a blanket over. Press play.",
        "Outside is cold. In here is warm.",
    ],
    # --- Energy / Intensity ---
    "empowering": [
        "Own every moment.",
        "Music that makes you feel unstoppable.",
        "Stand taller. Play louder.",
        "Your power playlist.",
        "For when you need to feel invincible.",
    ],
    "confidence_boost": [
        "Walk in like you own it.",
        "Head up. Volume up.",
        "Music that makes you feel like you.",
        "Turn confidence up to eleven.",
        "For when you need to feel your best.",
    ],
    "cathartic": [
        "Let it out.",
        "Feel it all.",
        "Release. Repeat.",
        "Music for a good cry or a good scream.",
        "A safe place for big feelings.",
    ],
    "angst_mix": [
        "Channel the frustration.",
        "Loud, raw, and real.",
        "For when you've had enough.",
        "Turn it up and let it out.",
        "Feels so good to feel this bad.",
    ],
    "celebration": [
        "Pop the cork. Press play.",
        "Something worth celebrating.",
        "Music for the best moments.",
        "You deserve this.",
        "Here's to tonight.",
    ],
    "euphoric": [
        "Sky-high and unstoppable.",
        "Total bliss.",
        "Pure elation, in playlist form.",
        "That feeling when everything clicks.",
        "When life sounds like this.",
    ],
    # --- Calm / Gentle ---
    "daydreaming": [
        "Let your mind wander.",
        "Head in the clouds.",
        "Drift away.",
        "Music for staring out of the window.",
        "Somewhere between here and there.",
    ],
    "lazy_sunday": [
        "Nowhere to be. Nothing to do.",
        "A Sunday kind of easy.",
        "Low effort. High comfort.",
        "Sundays were made for this.",
        "The softest kind of day.",
    ],
    "sunday_morning": [
        "Sunday morning, slow and easy.",
        "Coffee, light, and good music.",
        "No rush. No plans. Just this.",
        "Ease into Sunday.",
        "For the slow mornings you love.",
    ],
    "deep_work": [
        "Tune out. Tune in.",
        "No distractions. Just depth.",
        "Music for when the work matters.",
        "Long sessions need long playlists.",
        "Where deep work meets quiet music.",
    ],
    "evening_unwind": [
        "The day is done. Wind it down.",
        "Ease out of the day.",
        "Soft landing after a long one.",
        "Let the evening do its thing.",
        "From the chaos of the day to calm.",
    ],
    "folk_acoustic": [
        "Just wood, wire, and voice.",
        "Stripped back. Still brilliant.",
        "Music in its purest form.",
        "Real instruments. Real feelings.",
        "Close your eyes. Hear everything.",
    ],
    "emotional": [
        "Music that means something.",
        "Feel every word of it.",
        "Big feelings deserve big music.",
        "For the moments that hit hard.",
        "Honest, open, emotional.",
    ],
    # --- Upbeat / Outdoor ---
    "beach_vibes": [
        "Sun, sand, and something to listen to.",
        "Pure summer energy.",
        "Flip-flops optional.",
        "Music for when you can smell the sea.",
        "As close as music gets to the beach.",
    ],
    "fresh_start": [
        "New day. New chapter.",
        "Clean slate. Press play.",
        "Start fresh.",
        "Music for new beginnings.",
        "Whatever comes next, you're ready.",
    ],
    "spring_mix": [
        "The season is turning.",
        "Brighter days, brighter sounds.",
        "Music as light as spring.",
        "A soundtrack for bloom season.",
        "Spring is here. Ears open.",
    ],
    "brunch_mix": [
        "Good food. Better music.",
        "Bottomless brunch energy.",
        "Weekend morning, fully loaded.",
        "For the long brunches.",
        "Make it a Saturday.",
    ],
    "weekend_mix": [
        "The weekend has officially started.",
        "Saturday sounds. Sunday vibes.",
        "Two days of this.",
        "Finally.",
        "Absolutely nothing urgent.",
    ],
    "cooking_mix": [
        "Music for when you're in the kitchen.",
        "Cook something good. Play something better.",
        "Chopping, stirring, pressing play.",
        "The kitchen has a soundtrack.",
        "Tonight, you're cooking.",
    ],
    "summer_evening": [
        "Warm air. Warm sounds.",
        "The best part of summer.",
        "When the heat of the day breaks.",
        "Long summer nights.",
        "Golden hour lasts longer in summer.",
    ],
    # --- Romantic ---
    "romantic_mix": [
        "Set the mood.",
        "Romance, curated.",
        "For the two of you.",
        "Slow it down.",
        "Music that means more when you're not alone.",
    ],
    "modern_romance": [
        "Love songs for right now.",
        "Modern and heartfelt.",
        "Contemporary love, well soundtracked.",
        "How it feels to fall now.",
        "Romance, updated.",
    ],
    "slow_dance": [
        "For dancing close.",
        "Take their hand.",
        "The slowest, sweetest songs.",
        "Move together.",
        "Nothing else in the room.",
    ],
    "love_songs": [
        "Songs that say it better than words.",
        "Love, in playlist form.",
        "For the people worth a love song.",
        "Music made for feeling this way.",
        "All the love songs.",
    ],
    "first_date": [
        "Nervous energy and first impressions.",
        "Music for the butterflies.",
        "Hopeful, warm, a little nervous.",
        "For that first hello.",
        "The beginning of something.",
    ],
    "acoustic_romance": [
        "Quiet and close.",
        "Just a guitar and a feeling.",
        "Intimate and acoustic.",
        "Soft enough to hear every word.",
        "Music for close quarters.",
    ],
    "indie_romance": [
        "Romance with a little edge.",
        "Indie hearts, big feelings.",
        "Love songs for those who skip the obvious ones.",
        "Not your usual love playlist.",
        "For when the feelings are real but the pop's too much.",
    ],
    "late_night_romance": [
        "Just you, them, and past midnight.",
        "Quiet lights. Closer now.",
        "Romance after dark.",
        "The night is for this.",
        "No one else awake.",
    ],
    "piano_romance": [
        "Romance built on 88 keys.",
        "Just the piano and a feeling.",
        "Beautiful, delicate, true.",
        "Keys pressed gently.",
        "For when only the piano will do.",
    ],
    "strings_romance": [
        "Strings to match the feeling.",
        "Romance, orchestrated.",
        "Beautiful and sweeping.",
        "When only strings will do.",
        "As big as the feeling itself.",
    ],
    "string_quartet": [
        "Four instruments. Everything you need.",
        "The intimacy of a string quartet.",
        "Perfectly balanced. Deeply moving.",
        "Chamber music for your chamber.",
        "Nothing added. Nothing missing.",
    ],
    # --- Atmospheric / Seasonal ---
    "golden_hour": [
        "Chase the light.",
        "That hour when the world turns gold.",
        "Music for the glow.",
        "Everything looks better right now.",
        "Catch it before it goes.",
    ],
    "sunset_mix": [
        "Watch it go down.",
        "Music for the end of the light.",
        "Every ending is beautiful.",
        "Sunset coming. Volume up.",
        "The sky is doing something special.",
    ],
    "autumn_mix": [
        "The year is turning amber.",
        "Music for the falling leaves.",
        "Autumn has its own soundtrack.",
        "Wrapped up and wondering.",
        "The most reflective season.",
    ],
    "winter_mix": [
        "Cold outside. Warm in here.",
        "Music for the shortest days.",
        "Winter is here.",
        "Dark skies, good music.",
        "Deep winter listening.",
    ],
    "main_character": [
        "You're the main character today.",
        "For walking like the world is watching.",
        "Your scene. Your music.",
        "Live in the moment. Soundtrack optional.",
        "Every hero needs a theme.",
    ],
    # --- Driving / Activity ---
    "driving_mix": [
        "Built for the open road.",
        "Miles passing. Music playing.",
        "Windows cracked. Speed up slightly.",
        "Driving music, exactly that.",
        "Keep moving.",
    ],
    "night_drive": [
        "Late roads and empty lanes.",
        "The city at night, through the windscreen.",
        "Driving into the dark.",
        "Night driving has its own rules.",
        "Headlights on. Volume up.",
    ],
    "driving_singalong": [
        "Sing it like no one's watching. They're not.",
        "Full volume. Full chorus. Full commitment.",
        "You know every word.",
        "Hands on the wheel. Vocals on point.",
        "The car is your concert venue.",
    ],
    "road_trip": [
        "Long roads need great playlists.",
        "Pack the car. Hit play.",
        "Miles to go. Enjoy every one.",
        "The journey is half the fun.",
        "Road trip, officially started.",
    ],
    "commute_mix": [
        "The commute, made better.",
        "From door to desk with a soundtrack.",
        "Better than silence on the way in.",
        "Make the journey count.",
        "Getting there.",
    ],
    "walking_mix": [
        "Best foot forward.",
        "Music to walk to.",
        "Step by step.",
        "For the daily walk.",
        "The path is more interesting with these.",
    ],
    # --- Dinner / Evening ---
    "jazz_dinner": [
        "Jazz for the table.",
        "Sophisticated dining, properly soundtracked.",
        "Low, warm, effortless.",
        "Dinner goes better with jazz.",
        "The jazz is playing.",
    ],
    "romantic_jazz": [
        "Jazz and romance. The classic combination.",
        "Low lights. Low tempo. High feeling.",
        "Romance, with a brushed snare.",
        "Late evening, jazz turned down low.",
        "Blue notes and warm feelings.",
    ],
    "candlelight": [
        "Soft sounds for soft light.",
        "Music for the candles.",
        "Intimate, warm, unhurried.",
        "Turn the lights down. Press play.",
        "Everything is gentler right now.",
    ],
    "date_night": [
        "Tonight has potential.",
        "A perfect evening, properly soundtracked.",
        "Date night, set to music.",
        "Music for a night with someone special.",
        "This is going well.",
    ],
    "romantic_dinner": [
        "Dinner for two, sounding great.",
        "Fine dining, finer playlist.",
        "Tonight's reservation includes a soundtrack.",
        "Set the table. Set the mood.",
        "Music as good as the food.",
    ],
    # --- Mood / Feel ---
    "dreamy_mix": [
        "Lost in sound.",
        "Music for drifting.",
        "Soft edges and slow tempo.",
        "Somewhere between awake and asleep.",
        "Float away.",
    ],
    "moody_mix": [
        "Lean into the mood.",
        "Dark and interesting.",
        "For the complicated days.",
        "Not everything has to be uplifting.",
        "Mood: exactly this.",
    ],
    "bittersweet": [
        "Joy and sadness, together.",
        "Both at once.",
        "Happy-sad. You know the feeling.",
        "For those feelings that are hard to name.",
        "Mixed emotions, beautiful music.",
    ],
    "heartbreak": [
        "Let yourself feel it.",
        "For the ache.",
        "Music for the hard moments.",
        "It helps to have the right playlist.",
        "Some songs know exactly what you're going through.",
    ],
    # --- Memory / Late Night ---
    "nostalgia_mix": [
        "Back when everything felt different.",
        "A detour through memory lane.",
        "For the part of you that misses it.",
        "Music that takes you somewhere else.",
        "Then and now.",
    ],
    "synthpop_romance": [
        "Romance with a synth pulse.",
        "Shimmering keys and big feelings.",
        "Love songs with an electronic edge.",
        "Modern love, analogue soul.",
        "Synths, arpeggios, and a feeling you can't shake.",
    ],
    # --- Party / Social ---
    "friday_night": [
        "Friday. Finally.",
        "The weekend starts here.",
        "Tonight's the night.",
        "End the week right.",
        "Friday energy, amplified.",
    ],
    "pre_party": [
        "Getting ready. Getting pumped.",
        "The warmup before the warmup.",
        "Pregame playlist, confirmed.",
        "Turn it up while you get ready.",
        "The party starts here.",
    ],
    "party_throwback": [
        "Classics that still hit hard.",
        "Throwback bangers for the party.",
        "You forgot about some of these. You're welcome.",
        "The old stuff still works.",
        "Turn it up and go back.",
    ],
    "after_dark": [
        "The night belongs to you.",
        "Deep and dark and danceable.",
        "After dark, everything changes.",
        "Low lights, high tempo.",
        "Where the night leads.",
    ],
    "after_work": [
        "Clocked off. Tuned in.",
        "Shake off the workday.",
        "The commute home is different now.",
        "From desk to done.",
        "Office hours over. This begins.",
    ],
    "cool_down": [
        "Take it down a notch.",
        "The session is over. The music isn't.",
        "Breath slowing. Sounds slowing.",
        "Recovery, set to music.",
        "Ease out of it.",
    ],
}


def _pick_description(playlist_id, era=None, styles=None):
    """Pick a random description for the given playlist, formatting any placeholders."""
    pool = _DESCRIPTIONS.get(playlist_id, [])
    if not pool:
        return ""
    desc = random.choice(pool)
    if era:
        desc = desc.replace("{era}", era).replace("{year}", era)
    return desc


# ---------------------------------------------------------------------------
# Mood / Activity Mix profiles — acoustic target fingerprints
# ---------------------------------------------------------------------------
_MOOD_PROFILES = {
    # --- Meloday+ gap-fill mixes: 7 vibe gaps + 4-mix pop family ---
# ── situationship → "Situationship Mix" ───────────────────────────────────────────────────
# Theme:    Yearning, searching, bittersweet, wistful, tense.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "situationship": {"bpm": 90, "energy": -15, "danceability": 0.30, "brightness": 0.16, "beat_confidence": 0.50, "onset_rate": 3.2, "dynamic_complexity": 0.60, "arousal": 0.46, "valence": 0.42, "vocal_presence": 0.74},
# ── sad_bangers → "Sad Bangers Mix" ───────────────────────────────────────────────────────
# Theme:    Cathartic, bittersweet, melancholy, energetic, lively.
# Sound:    Upbeat 124bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · soft daypart lean · lyric-themes · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "sad_bangers": {"bpm": 124, "energy": -8, "danceability": 0.70, "brightness": 0.34, "beat_confidence": 0.78, "onset_rate": 6.0, "dynamic_complexity": 0.40, "arousal": 0.80, "valence": 0.32, "vocal_presence": 0.72},
# ── power_ballads → "Lighters Up" ─────────────────────────────────────────────────────────
# Theme:    Dramatic, passionate, theatrical, rousing, anthemic.
# Sound:    Slow 78bpm, mid energy, low groove; warm-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · floor 100k listeners · pop +1 · lyric-themes · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "power_ballads": {"bpm": 78, "energy": -11, "danceability": 0.24, "brightness": 0.26, "beat_confidence": 0.48, "onset_rate": 3.2, "dynamic_complexity": 0.72, "arousal": 0.55, "valence": 0.52, "vocal_presence": 0.86},
# ── restless → "Can't Switch Off" ─────────────────────────────────────────────────────────
# Theme:    Tense, anxious, urgent, nervous, searching.
# Sound:    Mid 116bpm, mid energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:defiant_intense
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "restless": {"bpm": 116, "energy": -11, "danceability": 0.40, "brightness": 0.24, "beat_confidence": 0.62, "onset_rate": 5.5, "dynamic_complexity": 0.55, "arousal": 0.70, "valence": 0.40, "vocal_presence": 0.62},
# ── neoclassical → "Neoclassical Calm" ────────────────────────────────────────────────────
# Theme:    Peaceful, elegant, graceful, reflective, poignant.
# Sound:    Slow 78bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: neo-classical, modern composition, chamber music, classical crossover, contemporary instrumental, classical.
# Criteria: style gate parent {classical, stage & screen} · pop -1 · soft daypart lean · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "neoclassical": {"bpm": 78, "energy": -19, "danceability": 0.16, "brightness": 0.22, "beat_confidence": 0.22, "onset_rate": 1.8, "dynamic_complexity": 0.76, "arousal": 0.24, "valence": 0.52, "vocal_presence": 0.18},
# ── yacht_rock → "Yacht Rock Mix" ─────────────────────────────────────────────────────────
# Theme:    Smooth, warm, mellow, sophisticated, sunny.
# Sound:    Mid 102bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soft rock, adult contemporary, sophisti-pop, blue-eyed soul, pop-soul, quiet storm.
# Criteria: style gate parent {funk / soul, pop, rock} · pop +0.5 · lyric-themes · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "yacht_rock": {"bpm": 102, "energy": -13, "danceability": 0.48, "brightness": 0.36, "beat_confidence": 0.60, "onset_rate": 4.0, "dynamic_complexity": 0.50, "arousal": 0.46, "valence": 0.70, "vocal_presence": 0.68},
# ── swagger → "Swagger Mix" ───────────────────────────────────────────────────────────────
# Theme:    Swaggering, brash, confident, stylish, street-smart.
# Sound:    Mid-slow 96bpm, mid energy, very danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: contemporary rap, hardcore rap, contemporary r&b, g-funk, funk, west coast rap.
# Criteria: style gate parent {funk / soul, hip hop} · pop +1 · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "swagger": {"bpm": 96, "energy": -11, "danceability": 0.62, "brightness": 0.22, "beat_confidence": 0.66, "onset_rate": 4.0, "dynamic_complexity": 0.42, "arousal": 0.58, "valence": 0.60, "vocal_presence": 0.80},
# ── chart_pop → "Pop Hits" ────────────────────────────────────────────────────────────────
# Theme:    Happy, lively, bright, fun, upbeat.
# Sound:    Mid 116bpm, mid energy, very danceable; bright, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: contemporary pop/rock, dance-pop, teen pop, vocal pop, traditional pop, pop idol….
# Criteria: style gate parent {electronic, pop, rock} · floor 500k listeners · pop +1 · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "chart_pop": {"bpm": 116, "energy": -9, "danceability": 0.66, "brightness": 0.50, "beat_confidence": 0.70, "onset_rate": 4.5, "dynamic_complexity": 0.42, "arousal": 0.62, "valence": 0.78, "vocal_presence": 0.80},
# ── dance_pop → "Dancefloor Pop" ──────────────────────────────────────────────────────────
# Theme:    Energetic, celebratory, fun, exuberant, lively.
# Sound:    Upbeat 122bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: dance-pop, dance-rock, alternative dance, euro-dance, eurodance, hi-nrg….
# Criteria: style gate parent {electronic, pop, rock} · floor 400k listeners · pop +1 · moodclass · cat:electronic_edm_pop
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "dance_pop": {"bpm": 122, "energy": -8, "danceability": 0.74, "brightness": 0.46, "beat_confidence": 0.78, "onset_rate": 5.0, "dynamic_complexity": 0.40, "arousal": 0.72, "valence": 0.75, "vocal_presence": 0.76},
# ── indie_pop → "Indie Darlings" ──────────────────────────────────────────────────────────
# Theme:    Bright, playful, sweet, quirky, sparkling.
# Sound:    Mid 112bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: indie pop, left-field pop, jangle pop, twee pop, chamber pop, baroque pop….
# Criteria: style gate parent {electronic, pop, rock} · floor 150k listeners · pop +1 · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "indie_pop": {"bpm": 112, "energy": -12, "danceability": 0.50, "brightness": 0.40, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.50, "arousal": 0.55, "valence": 0.68, "vocal_presence": 0.74},
# ── synth_pop → "Synth-Pop Nights" ────────────────────────────────────────────────────────
# Theme:    Stylish, sparkling, bright, lively, sophisticated.
# Sound:    Mid 116bpm, mid energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: synth pop, synth-pop, synthwave, new wave, new romantic, neo-electro….
# Criteria: style gate parent {electronic, pop, rock} · floor 150k listeners · pop +1 · soft daypart lean · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "synth_pop": {"bpm": 116, "energy": -10, "danceability": 0.58, "brightness": 0.42, "beat_confidence": 0.72, "onset_rate": 4.5, "dynamic_complexity": 0.45, "arousal": 0.62, "valence": 0.66, "vocal_presence": 0.72},
    # --- Meloday+ rock / electronic / scores gap-fill (play-history-driven) ---
# ── indie_rock → "Indie Anthems" ──────────────────────────────────────────────────────────
# Theme:    Lively, stylish, energetic, rousing, wry.
# Sound:    Upbeat 124bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: alternative/indie rock, indie rock, college rock, jangle pop, garage rock revival, modern rock.
# Criteria: style gate parent {rock} · floor 150k listeners · pop +1 · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "indie_rock": {"bpm": 124, "energy": -10, "danceability": 0.50, "brightness": 0.33, "beat_confidence": 0.72, "onset_rate": 5.2, "dynamic_complexity": 0.46, "arousal": 0.70, "valence": 0.58, "vocal_presence": 0.72},
# ── post_grunge → "Post-Grunge" ───────────────────────────────────────────────────────────
# Theme:    Brooding, gritty, intense, angst-ridden, cathartic.
# Sound:    Upbeat 126bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Last.fm-tag-gated (audio model can't name it): post-grunge, post grunge.
# Criteria: Last.fm gate · floor 150k listeners · pop +1 · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "post_grunge": {"bpm": 126, "energy": -8, "danceability": 0.46, "brightness": 0.24, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.46, "arousal": 0.78, "valence": 0.46, "vocal_presence": 0.70},
# ── rap_rock → "Rap-Rock & Nu-Metal" ──────────────────────────────────────────────────────
# Theme:    Aggressive, brash, intense, swaggering, rebellious.
# Sound:    Upbeat 128bpm, high energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: rap-rock, rap rock, rap-metal, rap metal, nu metal, nü metal….
# Criteria: style gate parent {hip hop, rock} · floor 150k listeners · pop +1 · moodclass · cat:rock_heavy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "rap_rock": {"bpm": 128, "energy": -7, "danceability": 0.52, "brightness": 0.22, "beat_confidence": 0.84, "onset_rate": 6.0, "dynamic_complexity": 0.44, "arousal": 0.86, "valence": 0.45, "vocal_presence": 0.66},
# ── festival_edm → "EDM Anthems" ──────────────────────────────────────────────────────────
# Theme:    Euphoric, exuberant, uplifting, sparkling, ecstatic.
# Sound:    Upbeat 128bpm, high energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: edm, big room, electro house, future bass, complextro.
# Criteria: style gate parent {electronic} · floor 400k listeners · pop +1 · moodclass · cat:electronic_edm_pop
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "festival_edm": {"bpm": 128, "energy": -6, "danceability": 0.72, "brightness": 0.48, "beat_confidence": 0.84, "onset_rate": 5.6, "dynamic_complexity": 0.32, "arousal": 0.86, "valence": 0.78, "vocal_presence": 0.48},
# ── soundtracks → "Soundtracks & Scores" ──────────────────────────────────────────────────
# Theme:    Epic, dramatic, majestic, atmospheric, reflective.
# Sound:    Mid-slow 92bpm, low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soundtrack, original score, film score, tv soundtrack, film music, movie theme….
# Criteria: style gate parent {classical, stage & screen} · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "soundtracks": {"bpm": 92, "energy": -15, "danceability": 0.24, "brightness": 0.22, "beat_confidence": 0.42, "onset_rate": 2.8, "dynamic_complexity": 0.70, "arousal": 0.40, "valence": 0.50, "vocal_presence": 0.18},
# ── rave_cave → "Rave Cave" ───────────────────────────────────────────────────────────────
# Theme:    Euphoric, aggressive, pounding, relentless, ecstatic.
# Sound:    Upbeat 132bpm, mid energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: donk, hard house, hard trance, hardstyle, hard techno, schranz….
# Criteria: style gate parent {electronic} · 50/50 balanced · moodclass · cat:electronic_edm_pop
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "rave_cave": {"bpm": 132, "energy": -8, "danceability": 0.40, "brightness": 0.17, "beat_confidence": 0.62, "onset_rate": 4.3, "dynamic_complexity": 0.44, "arousal": 0.66, "valence": 0.66, "vocal_presence": 0.45},
    # ---- 7 decade mixes (era) ----
# ── decade_60s → "60s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  1960–1969 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 1960-1969 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_60s": {"bpm": 120, "energy": -12, "danceability": 0.5, "brightness": 0.30, "beat_confidence": 0.62, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.58, "valence": 0.72, "vocal_presence": 0.82},
# ── decade_70s → "70s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  1970–1979 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 1970-1979 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_70s": {"bpm": 116, "energy": -11, "danceability": 0.56, "brightness": 0.33, "beat_confidence": 0.68, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.6, "valence": 0.68, "vocal_presence": 0.78},
# ── decade_80s → "80s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  1980–1989 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 1980-1989 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_80s": {"bpm": 122, "energy": -9, "danceability": 0.6, "brightness": 0.42, "beat_confidence": 0.76, "onset_rate": 5.5, "dynamic_complexity": 0.44, "arousal": 0.66, "valence": 0.7, "vocal_presence": 0.74},
# ── decade_90s → "90s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  1990–1999 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 1990-1999 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_90s": {"bpm": 110, "energy": -9, "danceability": 0.56, "brightness": 0.4, "beat_confidence": 0.72, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.6, "valence": 0.6, "vocal_presence": 0.74},
# ── decade_00s → "00s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  2000–2009 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 2000-2009 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_00s": {"bpm": 116, "energy": -8, "danceability": 0.6, "brightness": 0.43, "beat_confidence": 0.76, "onset_rate": 5.5, "dynamic_complexity": 0.44, "arousal": 0.64, "valence": 0.62, "vocal_presence": 0.76},
# ── decade_10s → "10s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  2010–2019 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 2010-2019 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_10s": {"bpm": 114, "energy": -7, "danceability": 0.62, "brightness": 0.46, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.42, "arousal": 0.62, "valence": 0.6, "vocal_presence": 0.74},
# ── decade_20s → "20s Mix" ────────────────────────────────────────────────────────────────
# Theme:    Rousing, energetic, stylish, playful.
# Sound:    N/A — the decade's hits, any style or sound (centroid below is metadata only, unused).
# Era/Geo:  2020–2029 · any origin.
# Music:    The decade's biggest hits by global Last.fm listeners — any genre or sound.
# Criteria: no genre gate · top hits by Last.fm listeners · 2020-2029 year-window · comp/VA-gated · anthem-rescue (artist top-5 ≥150k) · cat:era
# Flow:     Day-seeded shuffle (showcase — no DJ re-order).
# Enhance:  —
    "decade_20s": {"bpm": 112, "energy": -7, "danceability": 0.62, "brightness": 0.46, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.42, "arousal": 0.6, "valence": 0.58, "vocal_presence": 0.74},
    # ---- 3 geo showcase mixes: origin hard-gated, then an EQUAL-weight artist rotation (every eligible
    #      in-origin artist cycles in evenly, longest-unseen first); each artist's top-10 tracks by global
    #      Last.fm popularity, one picked at random. No sound/genre gate; the centroid below is metadata
    #      only (required _MOOD_PROFILES entry / cover art), never used for selection. DJ-ordered (SMOOTH). ----
# ── scotland_scene → "Sounds of Scotland" ─────────────────────────────────────────────────
# Theme:    Geo mix — the Scottish scene, all genres.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: scotland.
# Music:    Origin gate + equal-weight artist rotation; each artist's top-10 by global Last.fm popularity.
# Criteria: no genre/sound gate · origin-gated · equal-weight rotation (no floor) · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "scotland_scene":  {"bpm": 116, "energy": -9, "danceability": 0.54, "brightness": 0.40, "beat_confidence": 0.74, "onset_rate": 5.2, "dynamic_complexity": 0.48, "arousal": 0.60, "valence": 0.55, "vocal_presence": 0.76},
# ── australia_scene → "Sounds of Australia" ───────────────────────────────────────────────
# Theme:    Geo mix — the Australian scene, all genres.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: australia.
# Music:    Origin gate + equal-weight artist rotation; each artist's top-10 by global Last.fm popularity.
# Criteria: no genre/sound gate · origin-gated · equal-weight rotation (no floor) · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "australia_scene": {"bpm": 122, "energy": -7, "danceability": 0.56, "brightness": 0.45, "beat_confidence": 0.78, "onset_rate": 5.6, "dynamic_complexity": 0.44, "arousal": 0.67, "valence": 0.63, "vocal_presence": 0.78},
# ── london_scene → "Sounds of London" ─────────────────────────────────────────────────────
# Theme:    Geo mix — the London scene, all genres.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: london.
# Music:    Origin gate + equal-weight artist rotation; each artist's top-10 by global Last.fm popularity.
# Criteria: no genre/sound gate · origin-gated · equal-weight rotation (no floor) · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "london_scene":    {"bpm": 118, "energy": -8, "danceability": 0.63, "brightness": 0.44, "beat_confidence": 0.79, "onset_rate": 5.4, "dynamic_complexity": 0.41, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.73},
    # ---- 3 geo HITS mixes: SAME hard origin gate as the scenes above, but selected like the decade mixes —
    #      the top-200 songs by GLOBAL Last.fm listeners from that origin (an auto per-location floor: London's
    #      bar sits far higher than Scotland's). Genuine, widely-known hits; repetition across days is fine.
    #      Centroid is metadata only (required entry / cover art). PINNED. DJ-ordered (SMOOTH). ----
# ── scottish_hits → "Scottish Hits" ───────────────────────────────────────────────────────
# Theme:    Geo HITS mix — Scotland's biggest, anyone-would-know songs.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: scotland.
# Music:    Origin gate + top-200 by global Last.fm listeners + each artist's own top-5 anthem-rescue (genuine hits).
# Criteria: no genre/sound gate · origin-gated · top-200 + anthem-rescue · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "scottish_hits":   {"bpm": 116, "energy": -9, "danceability": 0.54, "brightness": 0.40, "beat_confidence": 0.74, "onset_rate": 5.2, "dynamic_complexity": 0.48, "arousal": 0.60, "valence": 0.55, "vocal_presence": 0.76},
# ── australian_hits → "Australian Hits" ───────────────────────────────────────────────────
# Theme:    Geo HITS mix — Australia's biggest, anyone-would-know songs.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: australia.
# Music:    Origin gate + top-200 by global Last.fm listeners + each artist's own top-5 anthem-rescue (genuine hits).
# Criteria: no genre/sound gate · origin-gated · top-200 + anthem-rescue · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "australian_hits": {"bpm": 122, "energy": -7, "danceability": 0.56, "brightness": 0.45, "beat_confidence": 0.78, "onset_rate": 5.6, "dynamic_complexity": 0.44, "arousal": 0.67, "valence": 0.63, "vocal_presence": 0.78},
# ── london_hits → "London Hits" ───────────────────────────────────────────────────────────
# Theme:    Geo HITS mix — London's biggest, anyone-would-know songs.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: london.
# Music:    Origin gate + top-200 by global Last.fm listeners + each artist's own top-5 anthem-rescue (genuine hits).
# Criteria: no genre/sound gate · origin-gated · top-200 + anthem-rescue · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "london_hits":     {"bpm": 118, "energy": -8, "danceability": 0.63, "brightness": 0.44, "beat_confidence": 0.79, "onset_rate": 5.4, "dynamic_complexity": 0.41, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.73},
# ── uk_scene → "Sounds of the UK" ─────────────────────────────────────────────────────────
# Theme:    Geo mix — the UK scene (England/Scotland/Wales/NI), all genres.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: united kingdom.
# Music:    Origin gate + equal-weight artist rotation; each artist's top-10 by global Last.fm popularity.
# Criteria: no genre/sound gate · origin-gated · equal-weight rotation (no floor) · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "uk_scene":        {"bpm": 118, "energy": -8, "danceability": 0.60, "brightness": 0.42, "beat_confidence": 0.78, "onset_rate": 5.4, "dynamic_complexity": 0.44, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.75},
# ── uk_hits → "UK Hits" ───────────────────────────────────────────────────────────────────
# Theme:    Geo HITS mix — the UK's biggest, anyone-would-know songs.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  Any era · origin-gated: united kingdom.
# Music:    Origin gate + HYBRID: listeners≥400k OR the artist's own Last.fm top-5 (≥150k) — rescues under-counted classics; no top-N cap.
# Criteria: no genre/sound gate · origin-gated · floor+anthem hybrid · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "uk_hits":         {"bpm": 118, "energy": -8, "danceability": 0.60, "brightness": 0.42, "beat_confidence": 0.78, "onset_rate": 5.4, "dynamic_complexity": 0.44, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.75},
# ── scotland_now → "Scotland Now" ─────────────────────────────────────────────────────────
# Theme:    Scotland-only radio — current Scottish releases + a few classic throwbacks.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  ~90% last 5 years + ~10% throwback · origin-gated: scotland.
# Music:    Origin gate + most-popular contemporary singles (last 5y) + ~10% throwback classics · 1-per-artist.
# Criteria: no genre/sound gate · origin-gated · 90% contemporary + 10% throwback · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "scotland_now":    {"bpm": 116, "energy": -9, "danceability": 0.54, "brightness": 0.40, "beat_confidence": 0.74, "onset_rate": 5.2, "dynamic_complexity": 0.48, "arousal": 0.60, "valence": 0.55, "vocal_presence": 0.76},
# ── london_now → "London Now" ─────────────────────────────────────────────────────────────
# Theme:    London-only radio — current London releases + a few classic throwbacks.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  ~90% last 5 years (leaning newest) + ~10% throwback · origin-gated: london.
# Music:    Origin gate + most-popular contemporary singles (last 5y, newer-leaning) + ~10% throwback classics · 1-per-artist.
# Criteria: no genre/sound gate · origin-gated · 90% contemporary + 10% throwback · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "london_now":      {"bpm": 118, "energy": -8, "danceability": 0.63, "brightness": 0.44, "beat_confidence": 0.79, "onset_rate": 5.4, "dynamic_complexity": 0.41, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.73},
# ── uk_now → "UK Now" ─────────────────────────────────────────────────────────────────────
# Theme:    UK-wide radio — current British releases + a few classic throwbacks.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  ~90% last 5 years (leaning newest) + ~10% throwback · origin-gated: united kingdom.
# Music:    Origin gate + most-popular contemporary singles (last 5y, newer-leaning) + ~10% throwback classics · 1-per-artist.
# Criteria: no genre/sound gate · origin-gated · 90% contemporary + 10% throwback · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "uk_now":          {"bpm": 118, "energy": -8, "danceability": 0.60, "brightness": 0.42, "beat_confidence": 0.78, "onset_rate": 5.4, "dynamic_complexity": 0.44, "arousal": 0.62, "valence": 0.58, "vocal_presence": 0.75},
# ── australia_now → "Australia Now" ───────────────────────────────────────────────────────
# Theme:    Australia-wide radio — current Australian releases + a few classic throwbacks.
# Sound:    No sound/genre gate — any genre from the origin (centroid below is metadata only).
# Era/Geo:  ~90% last 5 years (leaning newest) + ~10% throwback · origin-gated: australia.
# Music:    Origin gate + most-popular contemporary singles (last 5y, newer-leaning) + ~10% throwback classics · 1-per-artist.
# Criteria: no genre/sound gate · origin-gated · 90% contemporary + 10% throwback · PINNED · cat:geo_scene
# Flow:     DJ-ordered (SMOOTH); day-seeded start for day-to-day variety.
# Enhance:  —
    "australia_now":   {"bpm": 122, "energy": -7, "danceability": 0.56, "brightness": 0.45, "beat_confidence": 0.78, "onset_rate": 5.6, "dynamic_complexity": 0.44, "arousal": 0.67, "valence": 0.63, "vocal_presence": 0.78},
    # ---- 25 weather/seasonal mixes ----
# ── stormy → "Stormy Mix" ─────────────────────────────────────────────────────────────────
# Theme:    Dramatic, ominous, brooding, volatile, intense.
# Sound:    Mid-slow 80bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · weather-gated · lyric-themes · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "stormy": {"bpm": 80, "energy": -12, "danceability": 0.3, "brightness": 0.12, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.62, "arousal": 0.45, "valence": 0.35, "vocal_presence": 0.45},
# ── foggy → "Foggy Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Soothing, mysterious, atmospheric, eerie, dreamy.
# Sound:    Slow 78bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "foggy": {"bpm": 78, "energy": -16, "danceability": 0.25, "brightness": 0.14, "beat_confidence": 0.42, "onset_rate": 2, "dynamic_complexity": 0.6, "arousal": 0.25, "valence": 0.45, "vocal_presence": 0.45},
# ── snow_day → "Snow Day Mix" ─────────────────────────────────────────────────────────────
# Theme:    Soothing, gentle, delicate, playful.
# Sound:    Mid-slow 88bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "snow_day": {"bpm": 88, "energy": -15, "danceability": 0.3, "brightness": 0.3, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.35, "valence": 0.65, "vocal_presence": 0.5},
# ── heatwave → "Heatwave Mix" ─────────────────────────────────────────────────────────────
# Theme:    Languid, sultry, sparkling, mellow, dreamy.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "heatwave": {"bpm": 92, "energy": -13, "danceability": 0.4, "brightness": 0.4, "beat_confidence": 0.5, "onset_rate": 3.5, "dynamic_complexity": 0.55, "arousal": 0.35, "valence": 0.58, "vocal_presence": 0.5},
# ── frosty → "Frosty Mix" ─────────────────────────────────────────────────────────────────
# Theme:    Delicate, austere, refined, gentle.
# Sound:    Mid-slow 82bpm, low energy, low groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "frosty": {"bpm": 82, "energy": -15, "danceability": 0.28, "brightness": 0.32, "beat_confidence": 0.48, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.3, "valence": 0.55, "vocal_presence": 0.45},
# ── grey_skies → "Grey Skies Mix" ─────────────────────────────────────────────────────────
# Theme:    Melancholy, reflective, austere, somber, introspective.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · weather-gated · lyric-themes · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "grey_skies": {"bpm": 82, "energy": -15, "danceability": 0.26, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.3, "valence": 0.4, "vocal_presence": 0.6},
# ── windy → "Windy Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Nervous, driving, volatile, energetic.
# Sound:    Mid 110bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "windy": {"bpm": 110, "energy": -11, "danceability": 0.45, "brightness": 0.3, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.55, "arousal": 0.62, "valence": 0.55, "vocal_presence": 0.6},
# ── clear_night → "Clear Night Mix" ───────────────────────────────────────────────────────
# Theme:    Soothing, nocturnal, ethereal, dreamy, spacious.
# Sound:    Slow 76bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "clear_night": {"bpm": 76, "energy": -16, "danceability": 0.25, "brightness": 0.16, "beat_confidence": 0.42, "onset_rate": 2, "dynamic_complexity": 0.62, "arousal": 0.2, "valence": 0.55, "vocal_presence": 0.45},
# ── festive → "Festive Mix" ───────────────────────────────────────────────────────────────
# Theme:    Joyous, warm, nostalgic, celebratory, cheerful.
# Sound:    Mid 100bpm, mid energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Holiday-signal gated (seasonal title keyword or Plex Holiday flag).
# Criteria: holiday gate · pop +0.5 · lyric-themes · moodclass · cat:festive
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "festive": {"bpm": 100, "energy": -11, "danceability": 0.45, "brightness": 0.4, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.55, "arousal": 0.55, "valence": 0.78, "vocal_presence": 0.7},
# ── spring_bloom → "Spring Bloom Mix" ─────────────────────────────────────────────────────
# Theme:    Bright, joyous, sunny, lively.
# Sound:    Mid 105bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · season:spring · lyric-themes · moodclass · cat:season_spring
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "spring_bloom": {"bpm": 105, "energy": -12, "danceability": 0.48, "brightness": 0.42, "beat_confidence": 0.65, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.75, "vocal_presence": 0.65},
# ── spring_acoustic → "Spring Acoustic Mix" ───────────────────────────────────────────────
# Theme:    Warm, earthy, gentle, pastoral, wistful.
# Sound:    Mid-slow 95bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: folk, singer/songwriter, americana, indie folk.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · season:spring · moodclass · cat:season_spring
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "spring_acoustic": {"bpm": 95, "energy": -14, "danceability": 0.32, "brightness": 0.28, "beat_confidence": 0.55, "onset_rate": 4, "dynamic_complexity": 0.62, "arousal": 0.4, "valence": 0.65, "vocal_presence": 0.7},
# ── spring_strings → "Spring Strings Mix" ─────────────────────────────────────────────────
# Theme:    Graceful, uplifting, elegant, optimistic, soothing.
# Sound:    Mid-slow 92bpm, low energy, low groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: classical, modern composition, chamber, orchestral.
# Criteria: style gate parent {classical, stage & screen} · season:spring · cat:season_spring
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "spring_strings": {"bpm": 92, "energy": -14, "danceability": 0.28, "brightness": 0.3, "beat_confidence": 0.52, "onset_rate": 3.5, "dynamic_complexity": 0.68, "arousal": 0.45, "valence": 0.62, "vocal_presence": 0.35},
# ── spring_jangle → "Spring Jangle Mix" ───────────────────────────────────────────────────
# Theme:    Wistful, charming, lively, wry, bright.
# Sound:    Mid 110bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: indie pop, jangle pop, dream pop, twee pop.
# Criteria: style gate parent {rock} · pop +0.5 · season:spring · moodclass · cat:season_spring
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "spring_jangle": {"bpm": 110, "energy": -12, "danceability": 0.45, "brightness": 0.32, "beat_confidence": 0.68, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.58, "valence": 0.68, "vocal_presence": 0.7},
# ── summer_heat → "Summer Heat Mix" ───────────────────────────────────────────────────────
# Theme:    Lively, exuberant, sexy, carefree, euphoric.
# Sound:    Upbeat 118bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: disco, funk, club/dance.
# Criteria: style gate parent {electronic, funk / soul} · pop +1 · season:summer · lyric-themes · moodclass · cat:season_summer
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "summer_heat": {"bpm": 118, "energy": -10, "danceability": 0.72, "brightness": 0.42, "beat_confidence": 0.8, "onset_rate": 5.5, "dynamic_complexity": 0.4, "arousal": 0.7, "valence": 0.8, "vocal_presence": 0.55},
# ── summer_breeze → "Summer Breeze Mix" ───────────────────────────────────────────────────
# Theme:    Mellow, warm, smooth, easygoing, sunny.
# Sound:    Mid 100bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soft rock, adult contemporary, sophisti-pop.
# Criteria: style gate parent {pop, rock} · pop +0.5 · season:summer · lyric-themes · moodclass · cat:season_summer
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "summer_breeze": {"bpm": 100, "energy": -13, "danceability": 0.45, "brightness": 0.4, "beat_confidence": 0.58, "onset_rate": 4, "dynamic_complexity": 0.52, "arousal": 0.45, "valence": 0.7, "vocal_presence": 0.65},
# ── summer_roadtrip → "Summer Roadtrip Mix" ───────────────────────────────────────────────
# Theme:    Carefree, exuberant, anthemic, rousing, joyous.
# Sound:    Mid 116bpm, mid energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +0.5 · season:summer · lyric-themes · moodclass · cat:season_summer
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "summer_roadtrip": {"bpm": 116, "energy": -11, "danceability": 0.55, "brightness": 0.4, "beat_confidence": 0.74, "onset_rate": 5.5, "dynamic_complexity": 0.48, "arousal": 0.65, "valence": 0.78, "vocal_presence": 0.72},
# ── summer_tropical → "Summer Tropical Mix" ───────────────────────────────────────────────
# Theme:    Warm, sunny, lively, sensual, carefree.
# Sound:    Mid 102bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: latin, reggae, afro, tropical, bossa.
# Criteria: style gate parent {electronic, folk, world, & country, funk / soul, latin, reggae} · pop +0.5 · season:summer · lyric-themes · moodclass · cat:season_summer
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "summer_tropical": {"bpm": 102, "energy": -11, "danceability": 0.62, "brightness": 0.4, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.6, "valence": 0.78, "vocal_presence": 0.62},
# ── autumn_leaves → "Autumn Leaves Mix" ───────────────────────────────────────────────────
# Theme:    Warm, nostalgic, wistful, rustic, mellow.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: folk, singer/songwriter, americana.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · season:autumn · lyric-themes · moodclass · cat:season_autumn
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "autumn_leaves": {"bpm": 90, "energy": -14, "danceability": 0.32, "brightness": 0.26, "beat_confidence": 0.55, "onset_rate": 4, "dynamic_complexity": 0.6, "arousal": 0.4, "valence": 0.55, "vocal_presence": 0.7},
# ── autumn_jazz → "Autumn Jazz Mix" ───────────────────────────────────────────────────────
# Theme:    Smooth, warm, sophisticated, mellow, sultry.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: jazz, soul, smooth jazz.
# Criteria: style gate parent {funk / soul, jazz} · season:autumn · moodclass · cat:season_autumn
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "autumn_jazz": {"bpm": 92, "energy": -13, "danceability": 0.38, "brightness": 0.24, "beat_confidence": 0.58, "onset_rate": 4.5, "dynamic_complexity": 0.62, "arousal": 0.42, "valence": 0.6, "vocal_presence": 0.55},
# ── autumn_rain → "Autumn Rain Mix" ───────────────────────────────────────────────────────
# Theme:    Melancholy, wistful, reflective, brooding, somber.
# Sound:    Mid-slow 84bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · season:autumn · lyric-themes · moodclass · cat:season_autumn
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "autumn_rain": {"bpm": 84, "energy": -14, "danceability": 0.28, "brightness": 0.16, "beat_confidence": 0.52, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.35, "valence": 0.42, "vocal_presence": 0.62},
# ── autumn_embers → "Autumn Embers Mix" ───────────────────────────────────────────────────
# Theme:    Warm, rousing, earthy, gritty, nostalgic.
# Sound:    Mid 108bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: classic rock, blues rock, southern rock, arena rock, aor.
# Criteria: style gate parent {rock} · pop +0.5 · season:autumn · lyric-themes · moodclass · cat:season_autumn
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "autumn_embers": {"bpm": 108, "energy": -11, "danceability": 0.45, "brightness": 0.28, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.52, "arousal": 0.6, "valence": 0.55, "vocal_presence": 0.65},
# ── winter_frost → "Winter Frost Mix" ─────────────────────────────────────────────────────
# Theme:    Delicate, ethereal, atmospheric, soothing, austere.
# Sound:    Mid-slow 80bpm, low energy, low groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: ambient, modern composition, classical, experimental ambient.
# Criteria: style gate parent {classical, electronic, stage & screen} · season:winter · moodclass · cat:season_winter
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "winter_frost": {"bpm": 80, "energy": -16, "danceability": 0.26, "brightness": 0.3, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.66, "arousal": 0.25, "valence": 0.55, "vocal_presence": 0.3},
# ── winter_cosy → "Winter Cosy Mix" ───────────────────────────────────────────────────────
# Theme:    Warm, intimate, smooth, tender, sensual.
# Sound:    Mid-slow 88bpm, low energy, moderate groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soul, neo-soul, quiet storm, smooth soul.
# Criteria: style gate parent {funk / soul} · pop +0.5 · season:winter · lyric-themes · moodclass · cat:season_winter
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "winter_cosy": {"bpm": 88, "energy": -14, "danceability": 0.42, "brightness": 0.24, "beat_confidence": 0.58, "onset_rate": 4, "dynamic_complexity": 0.55, "arousal": 0.35, "valence": 0.62, "vocal_presence": 0.78},
# ── winter_nights → "Winter Nights Mix" ───────────────────────────────────────────────────
# Theme:    Nocturnal, hypnotic, atmospheric, mellow, dreamy.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: downtempo, ambient techno, electronica, trip-hop.
# Criteria: style gate parent {electronic} · pop +0.5 · scheduled 18-3 · season:winter · lyric-themes · moodclass · cat:season_winter
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "winter_nights": {"bpm": 92, "energy": -14, "danceability": 0.42, "brightness": 0.16, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.55, "arousal": 0.32, "valence": 0.5, "vocal_presence": 0.45},
# ── winter_jazz → "Winter Jazz Mix" ───────────────────────────────────────────────────────
# Theme:    Smooth, warm, mellow, sophisticated, intimate.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: smooth jazz, jazz, lounge, vocal jazz.
# Criteria: style gate parent {jazz} · season:winter · moodclass · cat:season_winter
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "winter_jazz": {"bpm": 90, "energy": -14, "danceability": 0.38, "brightness": 0.24, "beat_confidence": 0.55, "onset_rate": 4, "dynamic_complexity": 0.62, "arousal": 0.32, "valence": 0.6, "vocal_presence": 0.55},
    # ---- 50 added mood/vibe mixes (emotional & contextual) ----
# ── hopeful → "Brighter Days" ─────────────────────────────────────────────────────────────
# Theme:    Optimistic, uplifting, bright, innocent.
# Sound:    Mid-slow 95bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "hopeful": {"bpm": 95, "energy": -13, "danceability": 0.35, "brightness": 0.3, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.58, "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.65},
# ── yearning → "Longing" ──────────────────────────────────────────────────────────────────
# Theme:    Yearning, wistful, tender, plaintive.
# Sound:    Mid-slow 88bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:heartbreak_longing
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "yearning": {"bpm": 88, "energy": -15, "danceability": 0.25, "brightness": 0.14, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.62, "arousal": 0.42, "valence": 0.45, "vocal_presence": 0.7},
# ── triumphant → "Victory Lap" ────────────────────────────────────────────────────────────
# Theme:    Triumphant, rousing, majestic, anthemic, epic.
# Sound:    Mid 110bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "triumphant": {"bpm": 110, "energy": -10, "danceability": 0.45, "brightness": 0.34, "beat_confidence": 0.68, "onset_rate": 4.5, "dynamic_complexity": 0.55, "arousal": 0.68, "valence": 0.75, "vocal_presence": 0.62},
# ── serene → "Calm Waters" ────────────────────────────────────────────────────────────────
# Theme:    Soothing, peaceful, calm, gentle.
# Sound:    Slow 78bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "serene": {"bpm": 78, "energy": -18, "danceability": 0.22, "brightness": 0.2, "beat_confidence": 0.42, "onset_rate": 2, "dynamic_complexity": 0.65, "arousal": 0.16, "valence": 0.68, "vocal_presence": 0.4},
# ── tender → "Soft Spot" ──────────────────────────────────────────────────────────────────
# Theme:    Tender, gentle, warm, sentimental, sweet.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "tender": {"bpm": 82, "energy": -16, "danceability": 0.28, "brightness": 0.22, "beat_confidence": 0.48, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.28, "valence": 0.68, "vocal_presence": 0.68},
# ── defiant → "No Apologies" ──────────────────────────────────────────────────────────────
# Theme:    Defiant, fierce, brash, rebellious.
# Sound:    Upbeat 120bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:defiant_intense
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "defiant": {"bpm": 120, "energy": -9, "danceability": 0.45, "brightness": 0.24, "beat_confidence": 0.75, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.72, "valence": 0.45, "vocal_presence": 0.68},
# ── vulnerable → "Heart on Sleeve" ────────────────────────────────────────────────────────
# Theme:    Vulnerable, delicate, intimate.
# Sound:    Mid-slow 80bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:heartbreak_longing
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "vulnerable": {"bpm": 80, "energy": -16, "danceability": 0.24, "brightness": 0.14, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.62, "arousal": 0.35, "valence": 0.35, "vocal_presence": 0.7},
# ── awe_wonder → "Awe & Wonder Mix" ───────────────────────────────────────────────────────
# Theme:    Majestic, ethereal, epic, spiritual, reverent.
# Sound:    Mid-slow 92bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · soft daypart lean · cat:dreamy_ethereal
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "awe_wonder": {"bpm": 92, "energy": -13, "danceability": 0.25, "brightness": 0.22, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.7, "arousal": 0.45, "valence": 0.6, "vocal_presence": 0.4},
# ── grief_release → "Letting Go" ──────────────────────────────────────────────────────────
# Theme:    Elegiac, plaintive, somber, cathartic, anguished.
# Sound:    Slow 72bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:heartbreak_longing
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "grief_release": {"bpm": 72, "energy": -16, "danceability": 0.2, "brightness": 0.12, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.68, "arousal": 0.38, "valence": 0.18, "vocal_presence": 0.62},
# ── sunrise → "Sunrise Mix" ───────────────────────────────────────────────────────────────
# Theme:    Optimistic, gentle, bright, warm.
# Sound:    Mid-slow 96bpm, low energy, moderate groove; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 5-9 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "sunrise": {"bpm": 96, "energy": -14, "danceability": 0.35, "brightness": 0.4, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.55, "arousal": 0.38, "valence": 0.68, "vocal_presence": 0.55},
# ── blue_hour → "Blue Hour Mix" ───────────────────────────────────────────────────────────
# Theme:    Wistful, atmospheric, reflective, nocturnal, poignant.
# Sound:    Mid-slow 85bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 16-21 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "blue_hour": {"bpm": 85, "energy": -15, "danceability": 0.28, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.3, "valence": 0.55, "vocal_presence": 0.55},
# ── midnight → "Midnight Mix" ─────────────────────────────────────────────────────────────
# Theme:    Nocturnal, dark, intimate, hypnotic, brooding.
# Sound:    Mid-slow 80bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 22-4 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "midnight": {"bpm": 80, "energy": -15, "danceability": 0.35, "brightness": 0.08, "beat_confidence": 0.52, "onset_rate": 2.8, "dynamic_complexity": 0.55, "arousal": 0.38, "valence": 0.4, "vocal_presence": 0.55},
# ── three_am → "3am Thoughts" ─────────────────────────────────────────────────────────────
# Theme:    Nocturnal, lonely, nervous, hypnotic, weary.
# Sound:    Mid-slow 84bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · scheduled 0-4 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "three_am": {"bpm": 84, "energy": -15, "danceability": 0.32, "brightness": 0.1, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.42, "valence": 0.35, "vocal_presence": 0.55},
# ── golden_afternoon → "Golden Afternoon Mix" ─────────────────────────────────────────────
# Theme:    Warm, mellow, languid, agreeable, summery.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 12-18 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "golden_afternoon": {"bpm": 92, "energy": -14, "danceability": 0.35, "brightness": 0.34, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.55, "arousal": 0.4, "valence": 0.68, "vocal_presence": 0.6},
# ── overcast → "Overcast Mix" ─────────────────────────────────────────────────────────────
# Theme:    Melancholy, reflective, austere, somber, introspective.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "overcast": {"bpm": 82, "energy": -15, "danceability": 0.26, "brightness": 0.14, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.3, "valence": 0.4, "vocal_presence": 0.6},
# ── starlit → "Starlit Mix" ───────────────────────────────────────────────────────────────
# Theme:    Ethereal, dreamy, spacious, atmospheric, meditative.
# Sound:    Slow 76bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · scheduled 21-4 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "starlit": {"bpm": 76, "energy": -17, "danceability": 0.22, "brightness": 0.16, "beat_confidence": 0.42, "onset_rate": 2, "dynamic_complexity": 0.65, "arousal": 0.2, "valence": 0.6, "vocal_presence": 0.45},
# ── witching_hour → "Witching Hour Mix" ───────────────────────────────────────────────────
# Theme:    Eerie, mysterious, ominous, nocturnal.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · scheduled 22-4 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "witching_hour": {"bpm": 90, "energy": -14, "danceability": 0.32, "brightness": 0.1, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.58, "arousal": 0.45, "valence": 0.35, "vocal_presence": 0.5},
# ── monday_motivation → "Monday Motivation Mix" ───────────────────────────────────────────
# Theme:    Rousing, energetic, confident.
# Sound:    Mid 110bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · scheduled Mon 5-12 · lyric-themes · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "monday_motivation": {"bpm": 110, "energy": -11, "danceability": 0.48, "brightness": 0.34, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.48, "arousal": 0.65, "valence": 0.65, "vocal_presence": 0.62},
# ── midweek_reset → "Midweek Reset Mix" ───────────────────────────────────────────────────
# Theme:    Rousing, reflective, driving, cheerful.
# Sound:    Mid 100bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled Tue,Wed,Thu 7-20 · lyric-themes · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "midweek_reset": {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.3, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.5, "arousal": 0.5, "valence": 0.6, "vocal_presence": 0.6},
# ── friday_feeling → "Finally Friday" ─────────────────────────────────────────────────────
# Theme:    Exuberant, lively, carefree, fun, celebratory.
# Sound:    Mid 116bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · scheduled Fri 11-19 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "friday_feeling": {"bpm": 116, "energy": -10, "danceability": 0.58, "brightness": 0.38, "beat_confidence": 0.75, "onset_rate": 5.5, "dynamic_complexity": 0.42, "arousal": 0.66, "valence": 0.78, "vocal_presence": 0.62},
# ── sunday_scaries → "Sunday Scaries Mix" ─────────────────────────────────────────────────
# Theme:    Nervous, wistful, bittersweet, weary, reflective.
# Sound:    Mid-slow 86bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled Sun 15-23 · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "sunday_scaries": {"bpm": 86, "energy": -14, "danceability": 0.3, "brightness": 0.16, "beat_confidence": 0.52, "onset_rate": 3, "dynamic_complexity": 0.58, "arousal": 0.45, "valence": 0.4, "vocal_presence": 0.62},
# ── treat_yourself → "Treat Yourself Mix" ─────────────────────────────────────────────────
# Theme:    Stylish, confident, hedonistic, sexy, exuberant.
# Sound:    Mid 108bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "treat_yourself": {"bpm": 108, "energy": -11, "danceability": 0.55, "brightness": 0.32, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.6, "valence": 0.72, "vocal_presence": 0.62},
# ── dinner_party → "Dinner Party Mix" ─────────────────────────────────────────────────────
# Theme:    Sophisticated, warm, smooth, stylish, cosmopolitan.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: vocal jazz, soul, lounge, bossa nova, smooth.
# Criteria: style gate parent {funk / soul, jazz} · pop +0.5 · scheduled 18-22 · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "dinner_party": {"bpm": 100, "energy": -13, "danceability": 0.45, "brightness": 0.28, "beat_confidence": 0.6, "onset_rate": 4, "dynamic_complexity": 0.55, "arousal": 0.45, "valence": 0.65, "vocal_presence": 0.55},
# ── housework_hustle → "Tidy Up" ──────────────────────────────────────────────────────────
# Theme:    Lively, exuberant, playful, carefree.
# Sound:    Mid 116bpm, mid energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop +1 · soft daypart lean · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "housework_hustle": {"bpm": 116, "energy": -11, "danceability": 0.58, "brightness": 0.4, "beat_confidence": 0.74, "onset_rate": 5.5, "dynamic_complexity": 0.42, "arousal": 0.64, "valence": 0.74, "vocal_presence": 0.62},
# ── study_session → "Brain Food" ──────────────────────────────────────────────────────────
# Theme:    Cerebral, calm, reflective, mellow, hypnotic.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · soft daypart lean · moodclass · cat:focus_study
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "study_session": {"bpm": 90, "energy": -15, "danceability": 0.3, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.3},
# ── wind_down → "Wind-Down Mix" ───────────────────────────────────────────────────────────
# Theme:    Soothing, calm, gentle, languid, relaxed.
# Sound:    Slow 72bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 19-23 · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "wind_down": {"bpm": 72, "energy": -17, "danceability": 0.25, "brightness": 0.2, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.62, "arousal": 0.18, "valence": 0.62, "vocal_presence": 0.45},
# ── yoga_stretch → "Yoga & Stretch Mix" ───────────────────────────────────────────────────
# Theme:    Soothing, gentle, meditative, graceful, spiritual.
# Sound:    Mid-slow 90bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · soft daypart lean · moodclass · cat:wellness_sleep
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "yoga_stretch": {"bpm": 90, "energy": -16, "danceability": 0.28, "brightness": 0.22, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.6, "arousal": 0.25, "valence": 0.62, "vocal_presence": 0.35},
# ── meditation → "Meditation Mix" ─────────────────────────────────────────────────────────
# Theme:    Meditative, soothing, spiritual, spacious, devotional.
# Sound:    Slow 64bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · soft daypart lean · moodclass · cat:wellness_sleep
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "meditation": {"bpm": 64, "energy": -20, "danceability": 0.12, "brightness": 0.14, "beat_confidence": 0.25, "onset_rate": 1, "dynamic_complexity": 0.72, "arousal": 0.12, "valence": 0.6, "vocal_presence": 0.15},
# ── deep_reading → "Lost in a Book" ───────────────────────────────────────────────────────
# Theme:    Cerebral, calm, atmospheric, intimate, reflective.
# Sound:    Mid-slow 82bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · moodclass · cat:focus_study
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "deep_reading": {"bpm": 82, "energy": -17, "danceability": 0.2, "brightness": 0.16, "beat_confidence": 0.42, "onset_rate": 2, "dynamic_complexity": 0.65, "arousal": 0.3, "valence": 0.5, "vocal_presence": 0.25},
# ── creative_flow → "In the Flow" ─────────────────────────────────────────────────────────
# Theme:    Freewheeling, lively, hypnotic, playful, kinetic.
# Sound:    Mid 102bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · cat:focus_study
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "creative_flow": {"bpm": 102, "energy": -13, "danceability": 0.42, "brightness": 0.26, "beat_confidence": 0.6, "onset_rate": 4.5, "dynamic_complexity": 0.55, "arousal": 0.46, "valence": 0.62, "vocal_presence": 0.45},
# ── gaming → "Game On" ────────────────────────────────────────────────────────────────────
# Theme:    Energetic, intense, driving, kinetic, exciting.
# Sound:    Upbeat 130bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:workout_energy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "gaming": {"bpm": 130, "energy": -9, "danceability": 0.52, "brightness": 0.26, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.72, "valence": 0.55, "vocal_presence": 0.4},
# ── gardening → "Green Thumb" ─────────────────────────────────────────────────────────────
# Theme:    Warm, pastoral, sunny, carefree, earthy.
# Sound:    Mid 100bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "gardening": {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.36, "beat_confidence": 0.6, "onset_rate": 4.5, "dynamic_complexity": 0.52, "arousal": 0.42, "valence": 0.7, "vocal_presence": 0.6},
# ── spa_bath → "Spa Day" ──────────────────────────────────────────────────────────────────
# Theme:    Soothing, relaxed, gentle, delicate.
# Sound:    Slow 70bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · soft daypart lean · moodclass · cat:wellness_sleep
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "spa_bath": {"bpm": 70, "energy": -18, "danceability": 0.18, "brightness": 0.18, "beat_confidence": 0.4, "onset_rate": 1.5, "dynamic_complexity": 0.68, "arousal": 0.16, "valence": 0.65, "vocal_presence": 0.3},
# ── power_nap → "Forty Winks" ─────────────────────────────────────────────────────────────
# Theme:    Delicate, languid, dreamy, atmospheric, hypnotic.
# Sound:    Slow 66bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · soft daypart lean · moodclass · cat:wellness_sleep
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "power_nap": {"bpm": 66, "energy": -19, "danceability": 0.15, "brightness": 0.14, "beat_confidence": 0.3, "onset_rate": 1.2, "dynamic_complexity": 0.7, "arousal": 0.12, "valence": 0.55, "vocal_presence": 0.25},
# ── throwback_anthems → "Throwback Anthems Mix" ───────────────────────────────────────────
# Theme:    Nostalgic, exuberant, celebratory, fun, rousing.
# Sound:    Upbeat 118bpm, mid energy, danceable; warm-toned.
# Era/Geo:  up to 2016 · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · anthem gate (top-10 + 100k floor) · pop +1 · soft daypart lean · lyric-themes · moodclass · cat:nostalgic_throwback
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "throwback_anthems": {"bpm": 118, "energy": -10, "danceability": 0.55, "brightness": 0.34, "beat_confidence": 0.72, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.68, "valence": 0.78, "vocal_presence": 0.7},
# ── old_friends → "Old Friends Mix" ───────────────────────────────────────────────────────
# Theme:    Warm, nostalgic, joyous, sentimental, good-natured.
# Sound:    Mid 105bpm, low energy, danceable; warm-toned.
# Era/Geo:  up to 2018 · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · soft daypart lean · lyric-themes · moodclass · cat:nostalgic_throwback
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "old_friends": {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.3, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.5, "arousal": 0.5, "valence": 0.72, "vocal_presence": 0.7},
# ── campfire → "Campfire Mix" ─────────────────────────────────────────────────────────────
# Theme:    Warm, rustic, earthy, gentle, earnest.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: folk, singer/songwriter, americana, indie folk, acoustic.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · lyric-themes · moodclass · cat:folk_acoustic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "campfire": {"bpm": 92, "energy": -14, "danceability": 0.32, "brightness": 0.26, "beat_confidence": 0.52, "onset_rate": 3.5, "dynamic_complexity": 0.6, "arousal": 0.35, "valence": 0.65, "vocal_presence": 0.68},
# ── cookout → "Cookout Mix" ───────────────────────────────────────────────────────────────
# Theme:    Sunny, carefree, fun, warm, summery.
# Sound:    Mid 108bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · soft daypart lean · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "cookout": {"bpm": 108, "energy": -11, "danceability": 0.52, "brightness": 0.36, "beat_confidence": 0.68, "onset_rate": 5, "dynamic_complexity": 0.48, "arousal": 0.56, "valence": 0.78, "vocal_presence": 0.65},
# ── game_night → "Game Night Mix" ─────────────────────────────────────────────────────────
# Theme:    Playful, fun, lively, witty, exuberant.
# Sound:    Mid 112bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled Fri-Sun 18-24 · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "game_night": {"bpm": 112, "energy": -11, "danceability": 0.52, "brightness": 0.34, "beat_confidence": 0.68, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.55, "valence": 0.75, "vocal_presence": 0.62},
# ── singalong → "Singalong Mix" ───────────────────────────────────────────────────────────
# Theme:    Anthemic, joyous, rousing, exuberant, celebratory.
# Sound:    Upbeat 120bpm, mid energy, danceable; warm-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · floor 100k listeners · pop +1 · soft daypart lean · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "singalong": {"bpm": 120, "energy": -10, "danceability": 0.5, "brightness": 0.34, "beat_confidence": 0.74, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.7, "valence": 0.78, "vocal_presence": 0.78},
# ── school_days → "School Days Mix" ───────────────────────────────────────────────────────
# Theme:    Nostalgic, playful, bittersweet, lively, carefree.
# Sound:    Mid 116bpm, mid energy, danceable; warm-toned.
# Era/Geo:  1990–2014 · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:nostalgic_throwback
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "school_days": {"bpm": 116, "energy": -11, "danceability": 0.52, "brightness": 0.34, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.62, "valence": 0.62, "vocal_presence": 0.7},
# ── memory_lane → "Memory Lane Mix" ───────────────────────────────────────────────────────
# Theme:    Nostalgic, wistful, sentimental, tender, reflective.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  up to 2016 · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:nostalgic_throwback
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "memory_lane": {"bpm": 90, "energy": -14, "danceability": 0.32, "brightness": 0.22, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.58, "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.68},
# ── crush → "Crushing" ────────────────────────────────────────────────────────────────────
# Theme:    Sweet, playful, optimistic, gleeful, tender.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "crush": {"bpm": 100, "energy": -12, "danceability": 0.45, "brightness": 0.3, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.72, "vocal_presence": 0.7},
# ── slow_burn → "Slow Burn Mix" ───────────────────────────────────────────────────────────
# Theme:    Yearning, tender, intimate, sensual, sultry.
# Sound:    Mid-slow 84bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "slow_burn": {"bpm": 84, "energy": -15, "danceability": 0.3, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.58, "arousal": 0.4, "valence": 0.58, "vocal_presence": 0.75},
# ── moving_on → "Over It" ─────────────────────────────────────────────────────────────────
# Theme:    Bittersweet, optimistic, rousing, cathartic, defiant.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:heartbreak_longing
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "moving_on": {"bpm": 100, "energy": -12, "danceability": 0.45, "brightness": 0.26, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.52, "arousal": 0.5, "valence": 0.5, "vocal_presence": 0.72},
# ── loved_up → "Loved Up Mix" ─────────────────────────────────────────────────────────────
# Theme:    Ecstatic, warm, tender, sentimental, joyous.
# Sound:    Mid-slow 96bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "loved_up": {"bpm": 96, "energy": -12, "danceability": 0.45, "brightness": 0.3, "beat_confidence": 0.6, "onset_rate": 4, "dynamic_complexity": 0.52, "arousal": 0.5, "valence": 0.8, "vocal_presence": 0.75},
# ── long_distance → "Miles Apart" ─────────────────────────────────────────────────────────
# Theme:    Yearning, wistful, tender, poignant.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "long_distance": {"bpm": 82, "energy": -15, "danceability": 0.28, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.6, "arousal": 0.4, "valence": 0.45, "vocal_presence": 0.72},
# ── flirty → "Make a Move" ────────────────────────────────────────────────────────────────
# Theme:    Playful, sexy, fun, lively, sensual.
# Sound:    Mid 104bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "flirty": {"bpm": 104, "energy": -12, "danceability": 0.52, "brightness": 0.3, "beat_confidence": 0.65, "onset_rate": 4.5, "dynamic_complexity": 0.48, "arousal": 0.55, "valence": 0.72, "vocal_presence": 0.7},
# ── devotion → "All Yours" ────────────────────────────────────────────────────────────────
# Theme:    Devotional, tender, earnest, warm, reverent.
# Sound:    Mid-slow 86bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "devotion": {"bpm": 86, "energy": -14, "danceability": 0.32, "brightness": 0.22, "beat_confidence": 0.52, "onset_rate": 3, "dynamic_complexity": 0.58, "arousal": 0.34, "valence": 0.7, "vocal_presence": 0.75},
# ── wedding_day → "Wedding Day Mix" ───────────────────────────────────────────────────────
# Theme:    Joyous, tender, celebratory, romantic, triumphant.
# Sound:    Mid 100bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "wedding_day": {"bpm": 100, "energy": -11, "danceability": 0.45, "brightness": 0.34, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.82, "vocal_presence": 0.72},
    # ---- 86 added mixes (50 genre gaps + 36 city scenes) ----
# ── funk_disco → "Funk & Disco Mix" ───────────────────────────────────────────────────────
# Theme:    Rollicking, lively, stylish, exuberant, sexy.
# Sound:    Mid 116bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: funk, disco, funky breaks, neo-disco, boogie, euro-disco….
# Criteria: style gate parent {electronic, funk / soul} · pop +0.5 · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "funk_disco": {"bpm": 116, "energy": -11, "danceability": 0.74, "brightness": 0.42, "beat_confidence": 0.8, "onset_rate": 5.5, "dynamic_complexity": 0.4, "arousal": 0.7, "valence": 0.8, "vocal_presence": 0.62},
# ── neo_soul → "Neo-Soul & Quiet Storm Mix" ───────────────────────────────────────────────
# Theme:    Sensual, smooth, warm, sophisticated, intimate.
# Sound:    Mid-slow 82bpm, low energy, moderate groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: neo soul, contemporary r&b, quiet storm.
# Criteria: style gate parent {funk / soul} · pop +0.5 · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "neo_soul": {"bpm": 82, "energy": -15, "danceability": 0.4, "brightness": 0.22, "beat_confidence": 0.45, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.35, "valence": 0.62, "vocal_presence": 0.8},
# ── motown_soul → "Motown & Classic Soul Mix" ─────────────────────────────────────────────
# Theme:    Joyous, lively, warm, celebratory, sweet.
# Sound:    Upbeat 122bpm, mid energy, very danceable; bright, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soul, rhythm & blues, funk, disco.
# Criteria: style gate parent {funk / soul} · pop +0.5 · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "motown_soul": {"bpm": 122, "energy": -11, "danceability": 0.62, "brightness": 0.4, "beat_confidence": 0.72, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.62, "valence": 0.82, "vocal_presence": 0.85},
# ── after_hours_rnb → "After-Hours R&B Mix" ───────────────────────────────────────────────
# Theme:    Sensual, sultry, nocturnal, smooth, intimate.
# Sound:    Mid-slow 96bpm, low energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: contemporary r&b, alternative r&b, new jack swing, quiet storm.
# Criteria: style gate parent {electronic, funk / soul} · pop +0.5 · scheduled 21-4 · lyric-themes · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "after_hours_rnb": {"bpm": 96, "energy": -13, "danceability": 0.5, "brightness": 0.16, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.45, "arousal": 0.45, "valence": 0.55, "vocal_presence": 0.78},
# ── acid_jazz → "Acid Jazz & Jazz-Funk Mix" ───────────────────────────────────────────────
# Theme:    Lively, rollicking, stylish, cosmopolitan, smooth.
# Sound:    Mid 108bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: acid jazz, jazz-funk, soul jazz, jazz-house, fusion, clubjazz.
# Criteria: style gate parent {electronic, funk / soul, jazz} · pop +0.5 · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "acid_jazz": {"bpm": 108, "energy": -12, "danceability": 0.55, "brightness": 0.3, "beat_confidence": 0.6, "onset_rate": 4.5, "dynamic_complexity": 0.55, "arousal": 0.62, "valence": 0.65, "vocal_presence": 0.55},
# ── boom_bap → "Boom Bap Mix" ─────────────────────────────────────────────────────────────
# Theme:    Swaggering, confident, stylish, street-smart, gritty.
# Sound:    Mid-slow 92bpm, low energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: boom bap, hardcore hip-hop, jazzy hip-hop, conscious.
# Criteria: style gate parent {hip hop} · pop +0.5 · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "boom_bap": {"bpm": 92, "energy": -12, "danceability": 0.55, "brightness": 0.18, "beat_confidence": 0.62, "onset_rate": 4, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.55, "vocal_presence": 0.8},
# ── conscious_flow → "Conscious Flow Mix" ─────────────────────────────────────────────────
# Theme:    Thoughtful, reflective, stylish, earnest, literate.
# Sound:    Mid-slow 90bpm, low energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: conscious, jazzy hip-hop, instrumental, boom bap.
# Criteria: style gate parent {hip hop} · pop +0.5 · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "conscious_flow": {"bpm": 90, "energy": -13, "danceability": 0.48, "brightness": 0.2, "beat_confidence": 0.55, "onset_rate": 3.8, "dynamic_complexity": 0.55, "arousal": 0.5, "valence": 0.5, "vocal_presence": 0.78},
# ── g_funk → "G-Funk & West Coast Mix" ────────────────────────────────────────────────────
# Theme:    Laid-back, swaggering, sunny, confident, stylish.
# Sound:    Mid-slow 94bpm, low energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: g-funk, gangsta.
# Criteria: style gate parent {hip hop} · pop +0.5 · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "g_funk": {"bpm": 94, "energy": -12, "danceability": 0.58, "brightness": 0.24, "beat_confidence": 0.6, "onset_rate": 4, "dynamic_complexity": 0.42, "arousal": 0.55, "valence": 0.6, "vocal_presence": 0.78},
# ── trap_mode → "Trap Mode Mix" ───────────────────────────────────────────────────────────
# Theme:    Dark, aggressive, brash, menacing, swaggering.
# Sound:    Fast 140bpm, mid energy, very danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: trap, cloud rap, crunk, gangsta.
# Criteria: style gate parent {hip hop} · pop +0.5 · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "trap_mode": {"bpm": 140, "energy": -9, "danceability": 0.62, "brightness": 0.1, "beat_confidence": 0.72, "onset_rate": 3, "dynamic_complexity": 0.35, "arousal": 0.78, "valence": 0.35, "vocal_presence": 0.62},
# ── lofi_beats → "Lo-Fi Beats Mix" ────────────────────────────────────────────────────────
# Theme:    Mellow, hypnotic, relaxed, nostalgic, soothing.
# Sound:    Mid-slow 84bpm, low energy, danceable; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: instrumental hip-hop, lo-fi, trip-hop, downbeat.
# Criteria: style gate parent {electronic} · pop -1 · soft daypart lean · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "lofi_beats": {"bpm": 84, "energy": -16, "danceability": 0.45, "brightness": 0.14, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.48, "arousal": 0.25, "valence": 0.5, "vocal_presence": 0.22},
# ── house_party → "House Party Mix" ───────────────────────────────────────────────────────
# Theme:    Lively, exuberant, stylish, carefree, euphoric.
# Sound:    Upbeat 123bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: house, tech-house, progressive house, club/dance, euro-dance.
# Criteria: style gate parent {electronic} · pop +0.5 · lyric-themes · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "house_party": {"bpm": 123, "energy": -9, "danceability": 0.78, "brightness": 0.4, "beat_confidence": 0.8, "onset_rate": 5.5, "dynamic_complexity": 0.35, "arousal": 0.72, "valence": 0.74, "vocal_presence": 0.55},
# ── deep_house → "Deep House Late Mix" ────────────────────────────────────────────────────
# Theme:    Hypnotic, stylish, nocturnal, smooth, cosmopolitan.
# Sound:    Upbeat 122bpm, mid energy, very danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: deep house, microhouse, minimal techno, tech-house, left-field house.
# Criteria: style gate parent {electronic} · pop +0.5 · soft daypart lean · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "deep_house": {"bpm": 122, "energy": -11, "danceability": 0.66, "brightness": 0.18, "beat_confidence": 0.72, "onset_rate": 4.5, "dynamic_complexity": 0.4, "arousal": 0.55, "valence": 0.55, "vocal_presence": 0.4},
# ── techno → "Techno Warehouse Mix" ───────────────────────────────────────────────────────
# Theme:    Hypnotic, dark, driving, intense, nocturnal.
# Sound:    Upbeat 130bpm, mid energy, very danceable; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: techno, minimal techno, detroit techno, acid house, industrial dance.
# Criteria: style gate parent {electronic} · pop +0.5 · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "techno": {"bpm": 130, "energy": -9, "danceability": 0.62, "brightness": 0.12, "beat_confidence": 0.82, "onset_rate": 5.5, "dynamic_complexity": 0.32, "arousal": 0.8, "valence": 0.4, "vocal_presence": 0.2},
# ── trance → "Trance Heights Mix" ─────────────────────────────────────────────────────────
# Theme:    Euphoric, uplifting, exuberant, sparkling, ecstatic.
# Sound:    Upbeat 138bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: trance, progressive trance, goa trance, euro-dance, hi-nrg.
# Criteria: style gate parent {electronic} · pop +0.5 · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "trance": {"bpm": 138, "energy": -8, "danceability": 0.66, "brightness": 0.4, "beat_confidence": 0.82, "onset_rate": 5.8, "dynamic_complexity": 0.38, "arousal": 0.82, "valence": 0.7, "vocal_presence": 0.45},
# ── dnb → "Drum & Bass Mix" ───────────────────────────────────────────────────────────────
# Theme:    Energetic, kinetic, driving, rousing, exciting.
# Sound:    Fast 174bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: jungle/drum'n'bass, breakbeat, idm, bass music.
# Criteria: style gate parent {electronic} · pop +0.5 · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "dnb": {"bpm": 174, "energy": -8, "danceability": 0.55, "brightness": 0.28, "beat_confidence": 0.85, "onset_rate": 7.5, "dynamic_complexity": 0.4, "arousal": 0.85, "valence": 0.55, "vocal_presence": 0.45},
# ── bass_drop → "Bass Drop Mix" ───────────────────────────────────────────────────────────
# Theme:    Heavy, dark, intense, menacing, visceral.
# Sound:    Fast 142bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: dubstep, bass music, grime, trap (edm).
# Criteria: style gate parent {electronic, hip hop} · pop +0.5 · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "bass_drop": {"bpm": 142, "energy": -8, "danceability": 0.58, "brightness": 0.16, "beat_confidence": 0.8, "onset_rate": 4.5, "dynamic_complexity": 0.38, "arousal": 0.82, "valence": 0.42, "vocal_presence": 0.4},
# ── uk_garage → "UK Garage & 2-Step Mix" ──────────────────────────────────────────────────
# Theme:    Stylish, lively, swaggering, sexy, exuberant.
# Sound:    Upbeat 134bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: uk garage, garage, bass music, broken beat, bassline.
# Criteria: style gate parent {electronic} · pop +0.5 · lyric-themes · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "uk_garage": {"bpm": 134, "energy": -10, "danceability": 0.7, "brightness": 0.3, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.4, "arousal": 0.7, "valence": 0.62, "vocal_presence": 0.55},
# ── synthwave → "Synthwave & Retrowave Mix" ───────────────────────────────────────────────
# Theme:    Nostalgic, nocturnal, stylish, hypnotic, atmospheric.
# Sound:    Mid 110bpm, low energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: synthwave, neo-electro, new romantic.
# Criteria: style gate parent {electronic} · pop +0.5 · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "synthwave": {"bpm": 110, "energy": -12, "danceability": 0.5, "brightness": 0.2, "beat_confidence": 0.65, "onset_rate": 3.8, "dynamic_complexity": 0.42, "arousal": 0.55, "valence": 0.55, "vocal_presence": 0.3},
# ── industrial → "Industrial & EBM Mix" ───────────────────────────────────────────────────
# Theme:    Aggressive, harsh, menacing, cold, intense.
# Sound:    Upbeat 126bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: industrial, electro-industrial, industrial metal, industrial dance.
# Criteria: style gate parent {electronic, rock} · pop +0.5 · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-arc: ease-in → peak → wind-down (octave-BPM + Camelot key + energy).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "industrial": {"bpm": 126, "energy": -8, "danceability": 0.5, "brightness": 0.14, "beat_confidence": 0.8, "onset_rate": 5, "dynamic_complexity": 0.4, "arousal": 0.82, "valence": 0.3, "vocal_presence": 0.4},
# ── vaporwave → "Vaporwave & Chillsynth Mix" ──────────────────────────────────────────────
# Theme:    Hypnotic, dreamy, nostalgic, trippy, languid.
# Sound:    Mid-slow 80bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: vaporwave, chillwave, ambient pop, plunderphonics.
# Criteria: style gate parent {electronic} · pop -1 · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "vaporwave": {"bpm": 80, "energy": -16, "danceability": 0.42, "brightness": 0.22, "beat_confidence": 0.45, "onset_rate": 2.5, "dynamic_complexity": 0.5, "arousal": 0.25, "valence": 0.55, "vocal_presence": 0.3},
# ── downtempo → "Downtempo Drift Mix" ─────────────────────────────────────────────────────
# Theme:    Hypnotic, mellow, atmospheric, nocturnal, dreamy.
# Sound:    Mid-slow 96bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: downtempo, trip-hop, chillwave, idm, ambient techno.
# Criteria: style gate parent {electronic} · pop -1 · soft daypart lean · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "downtempo": {"bpm": 96, "energy": -15, "danceability": 0.42, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.3, "valence": 0.52, "vocal_presence": 0.4},
# ── hyperpop → "Hyperpop & Glitch Mix" ────────────────────────────────────────────────────
# Theme:    Manic, exuberant, playful, brash, ecstatic.
# Sound:    Fast 150bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: hyperpop, glitch, bubblegum, social media pop.
# Criteria: style gate parent {electronic, pop} · pop +0.5 · moodclass · cat:electronic_edm_pop
# Flow:     DJ energy-arc: ease-in → peak → wind-down (octave-BPM + Camelot key + energy).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "hyperpop": {"bpm": 150, "energy": -8, "danceability": 0.66, "brightness": 0.45, "beat_confidence": 0.78, "onset_rate": 6.5, "dynamic_complexity": 0.38, "arousal": 0.8, "valence": 0.62, "vocal_presence": 0.62},
# ── classic_rock → "Classic Rock Mix" ─────────────────────────────────────────────────────
# Theme:    Rousing, swaggering, brash, confident, exuberant.
# Sound:    Upbeat 122bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: album rock, arena rock, hard rock, blues-rock, southern rock, american trad rock.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_classic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "classic_rock": {"bpm": 122, "energy": -10, "danceability": 0.5, "brightness": 0.32, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.48, "arousal": 0.68, "valence": 0.62, "vocal_presence": 0.62},
# ── heavy_riffs → "Heavy Riffs Mix" ───────────────────────────────────────────────────────
# Theme:    Aggressive, intense, fierce, brash, visceral.
# Sound:    Upbeat 130bpm, high energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: heavy metal, hard rock, alternative metal, nü metal, funk metal, metalcore.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_heavy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "heavy_riffs": {"bpm": 130, "energy": -7, "danceability": 0.45, "brightness": 0.18, "beat_confidence": 0.82, "onset_rate": 6, "dynamic_complexity": 0.45, "arousal": 0.85, "valence": 0.35, "vocal_presence": 0.55},
# ── punk_energy → "Punk Energy Mix" ───────────────────────────────────────────────────────
# Theme:    Aggressive, rebellious, brash, raucous, defiant.
# Sound:    Fast 165bpm, mid energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: pop punk, punk revival, hardcore punk, skatepunk, punk/new wave.
# Criteria: style gate parent {rock} · pop +0.5 · lyric-themes · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "punk_energy": {"bpm": 165, "energy": -8, "danceability": 0.42, "brightness": 0.26, "beat_confidence": 0.85, "onset_rate": 7.5, "dynamic_complexity": 0.45, "arousal": 0.88, "valence": 0.45, "vocal_presence": 0.7},
# ── garage_grunge → "Garage & Grunge Mix" ─────────────────────────────────────────────────
# Theme:    Gritty, brash, rebellious, raw, fierce.
# Sound:    Upbeat 124bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: grunge, garage rock revival, garage punk, proto-punk, noise-rock.
# Criteria: style gate parent {rock} · pop +0.5 · lyric-themes · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "garage_grunge": {"bpm": 124, "energy": -9, "danceability": 0.45, "brightness": 0.2, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.78, "valence": 0.45, "vocal_presence": 0.65},
# ── emo_poppunk → "Emo & Pop-Punk Mix" ────────────────────────────────────────────────────
# Theme:    Angst-ridden, cathartic, earnest, yearning, fierce.
# Sound:    Upbeat 135bpm, mid energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: emo, emo-pop, pop punk, post-hardcore, screamo.
# Criteria: style gate parent {rock} · pop +0.5 · lyric-themes · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "emo_poppunk": {"bpm": 135, "energy": -9, "danceability": 0.45, "brightness": 0.24, "beat_confidence": 0.8, "onset_rate": 6, "dynamic_complexity": 0.45, "arousal": 0.78, "valence": 0.4, "vocal_presence": 0.78},
# ── britpop_rock → "Britpop & Madchester Mix" ─────────────────────────────────────────────
# Theme:    Swaggering, lively, stylish, wry, exuberant.
# Sound:    Upbeat 120bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: brit pop.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "britpop_rock": {"bpm": 120, "energy": -10, "danceability": 0.52, "brightness": 0.3, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.65, "valence": 0.62, "vocal_presence": 0.7},
# ── blues_bar → "Blues Bar Mix" ───────────────────────────────────────────────────────────
# Theme:    Gritty, earthy, passionate, gutsy, sultry.
# Sound:    Mid 100bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: blues-rock, electric blues, chicago blues, regional blues, punk blues.
# Criteria: style gate parent {blues, rock} · pop +0.5 · lyric-themes · moodclass · cat:rock_classic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "blues_bar": {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.24, "beat_confidence": 0.6, "onset_rate": 4, "dynamic_complexity": 0.55, "arousal": 0.58, "valence": 0.45, "vocal_presence": 0.68},
# ── psych_haze → "Psych Haze Mix" ─────────────────────────────────────────────────────────
# Theme:    Hypnotic, dreamy, trippy, atmospheric, druggy.
# Sound:    Mid-slow 96bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: neo-psychedelia, shoegaze, space rock, dream pop, kraut rock.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_psych
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "psych_haze": {"bpm": 96, "energy": -14, "danceability": 0.35, "brightness": 0.16, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.58, "arousal": 0.42, "valence": 0.5, "vocal_presence": 0.55},
# ── prog_rock → "Prog & Art Rock Mix" ─────────────────────────────────────────────────────
# Theme:    Complex, elaborate, cerebral, epic, sprawling.
# Sound:    Mid 110bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Last.fm-tag-gated (audio model can't name it): progressive rock, prog rock.
# Criteria: Last.fm gate · pop +0.5 · moodclass · cat:rock_psych
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "prog_rock": {"bpm": 110, "energy": -12, "danceability": 0.4, "brightness": 0.22, "beat_confidence": 0.55, "onset_rate": 4, "dynamic_complexity": 0.62, "arousal": 0.58, "valence": 0.5, "vocal_presence": 0.55},
# ── stoner_rock → "Stoner & Desert Rock Mix" ──────────────────────────────────────────────
# Theme:    Heavy, hypnotic, gritty, brooding, druggy.
# Sound:    Mid 110bpm, mid energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: stoner metal, doom metal, acid rock, space rock.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_heavy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "stoner_rock": {"bpm": 110, "energy": -9, "danceability": 0.42, "brightness": 0.16, "beat_confidence": 0.72, "onset_rate": 4.5, "dynamic_complexity": 0.5, "arousal": 0.7, "valence": 0.4, "vocal_presence": 0.55},
# ── reggae_dub → "Reggae & Dub Mix" ───────────────────────────────────────────────────────
# Theme:    Warm, laid-back, mellow, spiritual, sunny.
# Sound:    Slow 76bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: roots reggae, dub, dancehall, ska, contemporary reggae, reggae-pop.
# Criteria: style gate parent {reggae} · pop +0.5 · lyric-themes · moodclass · cat:reggae_ska
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "reggae_dub": {"bpm": 76, "energy": -13, "danceability": 0.48, "brightness": 0.28, "beat_confidence": 0.6, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.4, "valence": 0.65, "vocal_presence": 0.6},
# ── afrobeat → "Afrobeat Mix" ─────────────────────────────────────────────────────────────
# Theme:    Warm, lively, exuberant, spiritual, celebratory.
# Sound:    Mid 110bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: afrobeat, highlife, african, afro-cuban, soukous.
# Criteria: style gate parent {electronic, folk, world, & country, funk / soul} · pop +0.5 · lyric-themes · moodclass · cat:world_latin
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "afrobeat": {"bpm": 110, "energy": -11, "danceability": 0.6, "brightness": 0.34, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.65, "valence": 0.72, "vocal_presence": 0.6},
# ── latin_heat → "Latin Heat Mix" ─────────────────────────────────────────────────────────
# Theme:    Warm, lively, sexy, exuberant, celebratory.
# Sound:    Mid 100bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: latin pop, salsa, cumbia, reggaeton, latin dance, tropical.
# Criteria: style gate parent {electronic, latin} · pop +0.5 · lyric-themes · moodclass · cat:world_latin
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "latin_heat": {"bpm": 100, "energy": -11, "danceability": 0.62, "brightness": 0.36, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.7, "valence": 0.78, "vocal_presence": 0.65},
# ── bossa_samba → "Bossa & Samba Mix" ─────────────────────────────────────────────────────
# Theme:    Warm, smooth, sophisticated, sensual, mellow.
# Sound:    Mid-slow 95bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: bossa, samba, latin jazz, mpb.
# Criteria: style gate parent {jazz, latin} · pop +0.5 · lyric-themes · moodclass · cat:world_latin
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "bossa_samba": {"bpm": 95, "energy": -15, "danceability": 0.45, "brightness": 0.3, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.62, "arousal": 0.4, "valence": 0.65, "vocal_presence": 0.55},
# ── celtic_folk → "Celtic & Folk Traditions Mix" ──────────────────────────────────────────
# Theme:    Rousing, earthy, nostalgic, warm, pastoral.
# Sound:    Mid 110bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: celtic, celtic rock, celtic fusion, british folk, traditional celtic.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · lyric-themes · moodclass · cat:folk_acoustic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "celtic_folk": {"bpm": 110, "energy": -13, "danceability": 0.35, "brightness": 0.28, "beat_confidence": 0.6, "onset_rate": 4.5, "dynamic_complexity": 0.62, "arousal": 0.55, "valence": 0.6, "vocal_presence": 0.62},
# ── ska → "Ska & Two-Tone Mix" ────────────────────────────────────────────────────────────
# Theme:    Lively, exuberant, playful, rousing, carefree.
# Sound:    Fast 145bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: ska, ska-punk, third wave ska revival, ska revival.
# Criteria: style gate parent {reggae, rock} · pop +0.5 · moodclass · cat:reggae_ska
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "ska": {"bpm": 145, "energy": -10, "danceability": 0.58, "brightness": 0.34, "beat_confidence": 0.8, "onset_rate": 6.5, "dynamic_complexity": 0.45, "arousal": 0.75, "valence": 0.72, "vocal_presence": 0.65},
# ── bebop → "Bebop Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Energetic, elaborate, kinetic, complex, exciting.
# Sound:    Fast 165bpm, low energy, moderate groove; warm-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: hard bop, bop, post-bop, avant-garde jazz.
# Criteria: style gate parent {jazz} · pop +0.5 · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "bebop": {"bpm": 165, "energy": -12, "danceability": 0.4, "brightness": 0.3, "beat_confidence": 0.65, "onset_rate": 7, "dynamic_complexity": 0.72, "arousal": 0.78, "valence": 0.55, "vocal_presence": 0.2},
# ── swing_bigband → "Swing & Big Band Mix" ────────────────────────────────────────────────
# Theme:    Lively, exuberant, celebratory, brassy, playful.
# Sound:    Fast 150bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: swing, big band, swing, retro swing, traditional pop.
# Criteria: style gate parent {jazz, pop} · pop +0.5 · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "swing_bigband": {"bpm": 150, "energy": -12, "danceability": 0.55, "brightness": 0.34, "beat_confidence": 0.7, "onset_rate": 6, "dynamic_complexity": 0.65, "arousal": 0.72, "valence": 0.78, "vocal_presence": 0.62},
# ── smooth_jazz → "Smooth Jazz & Lounge Mix" ──────────────────────────────────────────────
# Theme:    Smooth, mellow, sophisticated, relaxed, elegant.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: smooth jazz, crossover jazz, lounge, cool, quiet storm.
# Criteria: style gate parent {funk / soul, jazz} · pop +0.5 · soft daypart lean · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "smooth_jazz": {"bpm": 92, "energy": -16, "danceability": 0.38, "brightness": 0.24, "beat_confidence": 0.45, "onset_rate": 3, "dynamic_complexity": 0.62, "arousal": 0.32, "valence": 0.58, "vocal_presence": 0.45},
# ── country_roads → "Country Roads Mix" ───────────────────────────────────────────────────
# Theme:    Warm, earnest, cheerful, nostalgic.
# Sound:    Mid 105bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: country, honky tonk, country rock.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · lyric-themes · moodclass · cat:country
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "country_roads": {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.34, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.52, "arousal": 0.55, "valence": 0.65, "vocal_presence": 0.72},
# ── outlaw_country → "Outlaw & Alt-Country Mix" ───────────────────────────────────────────
# Theme:    Gritty, earthy, earnest, rebellious, weary.
# Sound:    Mid 108bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: honky tonk, bluegrass, country rock, country.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · lyric-themes · moodclass · cat:country
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "outlaw_country": {"bpm": 108, "energy": -13, "danceability": 0.42, "brightness": 0.26, "beat_confidence": 0.6, "onset_rate": 4.5, "dynamic_complexity": 0.58, "arousal": 0.58, "valence": 0.52, "vocal_presence": 0.72},
# ── bluegrass → "Bluegrass & Banjo Mix" ───────────────────────────────────────────────────
# Theme:    Lively, rousing, earthy, pastoral, rustic.
# Sound:    Upbeat 120bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: bluegrass, progressive bluegrass, country-folk, string bands, new acoustic.
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · lyric-themes · moodclass · cat:country
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "bluegrass": {"bpm": 120, "energy": -13, "danceability": 0.45, "brightness": 0.36, "beat_confidence": 0.65, "onset_rate": 6.5, "dynamic_complexity": 0.65, "arousal": 0.62, "valence": 0.68, "vocal_presence": 0.65},
# ── rockabilly_surf → "Rockabilly & Surf Mix" ─────────────────────────────────────────────
# Theme:    Lively, playful, rousing, nostalgic, exuberant.
# Sound:    Fast 150bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: rockabilly, surf, rockabilly revival, psychobilly, rock & roll.
# Criteria: style gate parent {rock} · pop +0.5 · moodclass · cat:rock_classic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "rockabilly_surf": {"bpm": 150, "energy": -11, "danceability": 0.55, "brightness": 0.36, "beat_confidence": 0.78, "onset_rate": 6.5, "dynamic_complexity": 0.5, "arousal": 0.72, "valence": 0.7, "vocal_presence": 0.6},
# ── cinematic_epic → "Cinematic Epic Mix" ─────────────────────────────────────────────────
# Theme:    Epic, dramatic, majestic, monumental, theatrical.
# Sound:    Mid-slow 95bpm, low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: soundtrack, score, neo-romantic.
# Criteria: style gate parent {classical, stage & screen} · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "cinematic_epic": {"bpm": 95, "energy": -13, "danceability": 0.25, "brightness": 0.24, "beat_confidence": 0.45, "onset_rate": 3, "dynamic_complexity": 0.7, "arousal": 0.55, "valence": 0.5, "vocal_presence": 0.2},
# ── ambient_drift → "Ambient Drift Mix" ───────────────────────────────────────────────────
# Theme:    Atmospheric, meditative, ethereal, soothing, spacious.
# Sound:    Slow 62bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: ambient, dark ambient, new age, experimental ambient.
# Criteria: style gate parent {electronic} · pop -1 · soft daypart lean · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "ambient_drift": {"bpm": 62, "energy": -20, "danceability": 0.12, "brightness": 0.1, "beat_confidence": 0.2, "onset_rate": 1, "dynamic_complexity": 0.75, "arousal": 0.1, "valence": 0.5, "vocal_presence": 0.06},
# ── post_rock → "Post-Rock Crescendo Mix" ─────────────────────────────────────────────────
# Theme:    Atmospheric, majestic, cathartic, epic, brooding.
# Sound:    Mid 100bpm, low energy, moderate groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: post rock, math rock.
# Criteria: style gate parent {rock} · pop -1 · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "post_rock": {"bpm": 100, "energy": -13, "danceability": 0.3, "brightness": 0.18, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.68, "arousal": 0.55, "valence": 0.48, "vocal_presence": 0.25},
# ── chiptune → "8-Bit & Game Mix" ─────────────────────────────────────────────────────────
# Theme:    Playful, lively, nostalgic, exuberant, quirky.
# Sound:    Upbeat 130bpm, mid energy, danceable; bright, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: chiptune.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_edm_pop
# Flow:     DJ energy-arc: ease-in → peak → wind-down (octave-BPM + Camelot key + energy).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "chiptune": {"bpm": 130, "energy": -10, "danceability": 0.55, "brightness": 0.45, "beat_confidence": 0.75, "onset_rate": 6, "dynamic_complexity": 0.42, "arousal": 0.72, "valence": 0.65, "vocal_presence": 0.2},
# ── gospel → "Gospel & Choir Mix" ─────────────────────────────────────────────────────────
# Theme:    Spiritual, joyous, uplifting, reverent, exuberant.
# Sound:    Mid 112bpm, mid energy, danceable; bright, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: gospel.
# Criteria: style gate parent {funk / soul} · pop +0.5 · soft daypart lean · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "gospel": {"bpm": 112, "energy": -11, "danceability": 0.48, "brightness": 0.4, "beat_confidence": 0.65, "onset_rate": 5, "dynamic_complexity": 0.55, "arousal": 0.65, "valence": 0.72, "vocal_presence": 0.85},
# ── glasgow_folk → "Glasgow Folk Mix" ─────────────────────────────────────────────────────
# Theme:    Earnest, nostalgic, warm, pastoral, wistful.
# Sound:    Mid-slow 88bpm, low energy, low groove; warm-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: folk, folk rock, neofolk, celtic.
# Criteria: style gate parent {folk, world, & country, rock} · cat:folk_acoustic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_folk": {"bpm": 88, "energy": -15, "danceability": 0.28, "brightness": 0.26, "beat_confidence": 0.5, "onset_rate": 4, "dynamic_complexity": 0.65, "arousal": 0.45, "valence": 0.6, "vocal_presence": 0.7},
# ── glasgow_dream → "Glasgow Dream Mix" ───────────────────────────────────────────────────
# Theme:    Dreamy, hypnotic, atmospheric, wistful, ethereal.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: dream pop, shoegaze, noise pop, neo-psychedelia.
# Criteria: style gate parent {rock} · moodclass · cat:rock_psych
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_dream": {"bpm": 92, "energy": -15, "danceability": 0.3, "brightness": 0.16, "beat_confidence": 0.52, "onset_rate": 3.5, "dynamic_complexity": 0.58, "arousal": 0.4, "valence": 0.55, "vocal_presence": 0.55},
# ── glasgow_indie → "Glasgow Indie Mix" ───────────────────────────────────────────────────
# Theme:    Wry, wistful, witty, stylish, literate.
# Sound:    Mid 110bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: indie pop, twee pop, c-86, jangle pop, sophisti-pop, chamber pop.
# Criteria: style gate parent {rock} · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_indie": {"bpm": 110, "energy": -14, "danceability": 0.42, "brightness": 0.28, "beat_confidence": 0.6, "onset_rate": 5, "dynamic_complexity": 0.52, "arousal": 0.55, "valence": 0.58, "vocal_presence": 0.72},
# ── glasgow_soul → "Glasgow Soul Mix" ─────────────────────────────────────────────────────
# Theme:    Warm, stylish, smooth, sophisticated, lively.
# Sound:    Mid 104bpm, low energy, danceable; warm-toned, vocal-forward.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: blue-eyed soul, pop-soul, northern soul, funk.
# Criteria: style gate parent {funk / soul} · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "glasgow_soul": {"bpm": 104, "energy": -12, "danceability": 0.55, "brightness": 0.3, "beat_confidence": 0.65, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.65, "vocal_presence": 0.78},
# ── glasgow_postrock → "Glasgow Post-Rock Mix" ────────────────────────────────────────────
# Theme:    Atmospheric, majestic, cathartic, brooding, epic.
# Sound:    Mid 100bpm, low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: post rock, math rock.
# Criteria: style gate parent {rock} · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_postrock": {"bpm": 100, "energy": -14, "danceability": 0.28, "brightness": 0.16, "beat_confidence": 0.55, "onset_rate": 3.5, "dynamic_complexity": 0.68, "arousal": 0.55, "valence": 0.48, "vocal_presence": 0.2},
# ── glasgow_anthems → "Glasgow Anthems Mix" ───────────────────────────────────────────────
# Theme:    Swaggering, exuberant, lively, confident, rousing.
# Sound:    Upbeat 120bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: indie rock, dance-rock, britpop, new wave/post-punk revival.
# Criteria: style gate parent {rock} · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_anthems": {"bpm": 120, "energy": -10, "danceability": 0.55, "brightness": 0.3, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.45, "arousal": 0.68, "valence": 0.62, "vocal_presence": 0.68},
# ── glasgow_synth → "Glasgow Synth Mix" ───────────────────────────────────────────────────
# Theme:    Stylish, nocturnal, hypnotic, atmospheric, yearning.
# Sound:    Mid 116bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: synth pop, new wave, electro, dance-rock.
# Criteria: style gate parent {electronic, pop, rock} · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_synth": {"bpm": 116, "energy": -11, "danceability": 0.52, "brightness": 0.22, "beat_confidence": 0.7, "onset_rate": 4.5, "dynamic_complexity": 0.42, "arousal": 0.55, "valence": 0.6, "vocal_presence": 0.6},
# ── glasgow_postpunk → "Glasgow Post-Punk Mix" ────────────────────────────────────────────
# Theme:    Angular, rebellious, brash, nervous, defiant.
# Sound:    Upbeat 132bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: post-punk, new wave/post-punk revival, punk.
# Criteria: style gate parent {rock} · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_postpunk": {"bpm": 132, "energy": -10, "danceability": 0.48, "brightness": 0.22, "beat_confidence": 0.78, "onset_rate": 6, "dynamic_complexity": 0.48, "arousal": 0.78, "valence": 0.5, "vocal_presence": 0.62},
# ── glasgow_house → "Glasgow House Mix" ───────────────────────────────────────────────────
# Theme:    Hypnotic, stylish, lively, cosmopolitan, nocturnal.
# Sound:    Upbeat 124bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: house, left-field house, tech-house, disco.
# Criteria: style gate parent {electronic} · lyric-themes · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "glasgow_house": {"bpm": 124, "energy": -10, "danceability": 0.7, "brightness": 0.26, "beat_confidence": 0.78, "onset_rate": 5, "dynamic_complexity": 0.4, "arousal": 0.65, "valence": 0.6, "vocal_presence": 0.45},
# ── glasgow_underground → "Glasgow Underground Mix" ───────────────────────────────────────
# Theme:    Hypnotic, dark, driving, nocturnal, intense.
# Sound:    Upbeat 130bpm, mid energy, very danceable; dark-toned, near-instrumental.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: techno, minimal techno, detroit techno, acid house.
# Criteria: style gate parent {electronic} · pop -1 · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_underground": {"bpm": 130, "energy": -9, "danceability": 0.62, "brightness": 0.12, "beat_confidence": 0.82, "onset_rate": 5.5, "dynamic_complexity": 0.34, "arousal": 0.8, "valence": 0.42, "vocal_presence": 0.2},
# ── glasgow_bass → "Glasgow Bass Mix" ─────────────────────────────────────────────────────
# Theme:    Kinetic, playful, quirky, exuberant, bright.
# Sound:    Fast 140bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: idm, bassline, dubstep, breakbeat.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_bass": {"bpm": 140, "energy": -9, "danceability": 0.58, "brightness": 0.3, "beat_confidence": 0.78, "onset_rate": 5, "dynamic_complexity": 0.4, "arousal": 0.78, "valence": 0.55, "vocal_presence": 0.45},
# ── glasgow_late → "Glasgow Late Mix" ─────────────────────────────────────────────────────
# Theme:    Hypnotic, nocturnal, mellow, atmospheric, stylish.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · geo-tiered to Glasgow.
# Music:    Genre-pure: downtempo, trip-hop, electronica, ambient techno.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "glasgow_late": {"bpm": 90, "energy": -15, "danceability": 0.42, "brightness": 0.16, "beat_confidence": 0.5, "onset_rate": 3, "dynamic_complexity": 0.55, "arousal": 0.3, "valence": 0.5, "vocal_presence": 0.4},
# ── london_dub → "London Dub Mix" ─────────────────────────────────────────────────────────
# Theme:    Warm, laid-back, hypnotic, spiritual, mellow.
# Sound:    Slow 76bpm, low energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: dub, roots reggae, dancehall, reggae-pop.
# Criteria: style gate parent {reggae} · moodclass · cat:reggae_ska
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_dub": {"bpm": 76, "energy": -13, "danceability": 0.48, "brightness": 0.24, "beat_confidence": 0.58, "onset_rate": 3, "dynamic_complexity": 0.58, "arousal": 0.4, "valence": 0.62, "vocal_presence": 0.55},
# ── london_soul → "London Soul Mix" ───────────────────────────────────────────────────────
# Theme:    Passionate, sultry, stylish, smooth, sophisticated.
# Sound:    Mid-slow 96bpm, low energy, danceable; warm-toned, vocal-forward.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: blue-eyed soul, neo-soul, contemporary r&b, acid jazz.
# Criteria: style gate parent {funk / soul} · lyric-themes · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "london_soul": {"bpm": 96, "energy": -12, "danceability": 0.52, "brightness": 0.26, "beat_confidence": 0.62, "onset_rate": 4, "dynamic_complexity": 0.52, "arousal": 0.5, "valence": 0.62, "vocal_presence": 0.82},
# ── london_jazz → "London Jazz Mix" ───────────────────────────────────────────────────────
# Theme:    Lively, spiritual, cosmopolitan, cerebral, warm.
# Sound:    Mid 108bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: contemporary jazz, jazz-funk, spiritual jazz, afro-beat, acid jazz.
# Criteria: style gate parent {jazz} · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_jazz": {"bpm": 108, "energy": -12, "danceability": 0.5, "brightness": 0.28, "beat_confidence": 0.62, "onset_rate": 5, "dynamic_complexity": 0.62, "arousal": 0.62, "valence": 0.62, "vocal_presence": 0.45},
# ── london_triphop → "London Trip-Hop Mix" ────────────────────────────────────────────────
# Theme:    Nocturnal, hypnotic, brooding, atmospheric, theatrical.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: trip-hop, downtempo, downbeat, idm.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_chill
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_triphop": {"bpm": 90, "energy": -15, "danceability": 0.42, "brightness": 0.14, "beat_confidence": 0.52, "onset_rate": 3, "dynamic_complexity": 0.58, "arousal": 0.32, "valence": 0.48, "vocal_presence": 0.55},
# ── london_mod → "London Mod Mix" ─────────────────────────────────────────────────────────
# Theme:    Lively, acerbic, stylish, rousing, exuberant.
# Sound:    Upbeat 124bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: mod, beat, merseybeat, freakbeat, british invasion, british rhythm & blues….
# Criteria: style gate parent {rock} · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_mod": {"bpm": 124, "energy": -11, "danceability": 0.52, "brightness": 0.34, "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.48, "arousal": 0.68, "valence": 0.65, "vocal_presence": 0.68},
# ── london_britpop → "London Britpop Mix" ─────────────────────────────────────────────────
# Theme:    Swaggering, witty, lively, stylish, wry.
# Sound:    Upbeat 122bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: brit pop.
# Criteria: style gate parent {rock} · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_britpop": {"bpm": 122, "energy": -10, "danceability": 0.52, "brightness": 0.3, "beat_confidence": 0.7, "onset_rate": 5, "dynamic_complexity": 0.45, "arousal": 0.65, "valence": 0.62, "vocal_presence": 0.7},
# ── london_indie → "London Indie Mix" ─────────────────────────────────────────────────────
# Theme:    Angular, lively, brash, nervous, stylish.
# Sound:    Upbeat 130bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: indie rock, new wave/post-punk revival, garage rock revival.
# Criteria: style gate parent {rock} · moodclass · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_indie": {"bpm": 130, "energy": -10, "danceability": 0.52, "brightness": 0.28, "beat_confidence": 0.78, "onset_rate": 6, "dynamic_complexity": 0.45, "arousal": 0.72, "valence": 0.58, "vocal_presence": 0.68},
# ── london_calling → "London Calling Mix" ─────────────────────────────────────────────────
# Theme:    Rebellious, brash, defiant, raucous, gritty.
# Sound:    Fast 142bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: punk, post-punk, oi!, new wave.
# Criteria: style gate parent {rock} · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_calling": {"bpm": 142, "energy": -9, "danceability": 0.45, "brightness": 0.26, "beat_confidence": 0.82, "onset_rate": 6.5, "dynamic_complexity": 0.48, "arousal": 0.78, "valence": 0.55, "vocal_presence": 0.68},
# ── london_garage → "London Garage Mix" ───────────────────────────────────────────────────
# Theme:    Stylish, lively, swaggering, sexy, exuberant.
# Sound:    Upbeat 134bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: uk garage, garage, broken beat, bassline.
# Criteria: style gate parent {electronic} · lyric-themes · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "london_garage": {"bpm": 134, "energy": -10, "danceability": 0.7, "brightness": 0.3, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.4, "arousal": 0.7, "valence": 0.62, "vocal_presence": 0.55},
# ── london_grime → "London Grime Mix" ─────────────────────────────────────────────────────
# Theme:    Aggressive, brash, menacing, gritty, defiant.
# Sound:    Fast 140bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: grime, uk drill, bass music.
# Criteria: style gate parent {electronic, hip hop} · lyric-themes · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "london_grime": {"bpm": 140, "energy": -9, "danceability": 0.58, "brightness": 0.14, "beat_confidence": 0.8, "onset_rate": 4, "dynamic_complexity": 0.38, "arousal": 0.82, "valence": 0.42, "vocal_presence": 0.62},
# ── london_dubstep → "London Dubstep Mix" ─────────────────────────────────────────────────
# Theme:    Heavy, dark, menacing, hypnotic, intense.
# Sound:    Fast 142bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: dubstep, bass music, uk garage.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_dubstep": {"bpm": 142, "energy": -9, "danceability": 0.55, "brightness": 0.14, "beat_confidence": 0.78, "onset_rate": 4, "dynamic_complexity": 0.4, "arousal": 0.8, "valence": 0.42, "vocal_presence": 0.4},
# ── london_jungle → "London Jungle Mix" ───────────────────────────────────────────────────
# Theme:    Kinetic, manic, driving, rousing, exciting.
# Sound:    Fast 172bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to London.
# Music:    Genre-pure: jungle, drum n bass, breakbeat, breaks, big beat.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_bass
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "london_jungle": {"bpm": 172, "energy": -8, "danceability": 0.55, "brightness": 0.24, "beat_confidence": 0.85, "onset_rate": 7.5, "dynamic_complexity": 0.42, "arousal": 0.85, "valence": 0.55, "vocal_presence": 0.45},
# ── melbourne_folk → "Melbourne Folk Mix" ─────────────────────────────────────────────────
# Theme:    Warm, earnest, wistful, pastoral, gentle.
# Sound:    Mid-slow 92bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: folk, folk rock, neofolk, celtic.
# Criteria: style gate parent {folk, world, & country, rock} · cat:folk_acoustic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_folk": {"bpm": 92, "energy": -15, "danceability": 0.3, "brightness": 0.28, "beat_confidence": 0.52, "onset_rate": 4, "dynamic_complexity": 0.62, "arousal": 0.45, "valence": 0.6, "vocal_presence": 0.72},
# ── melbourne_dream → "Melbourne Dream Mix" ───────────────────────────────────────────────
# Theme:    Dreamy, hypnotic, atmospheric, wistful, mellow.
# Sound:    Mid-slow 96bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: dream pop, jangle pop, indie pop, neo-psychedelia.
# Criteria: style gate parent {rock} · moodclass · cat:rock_psych
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_dream": {"bpm": 96, "energy": -14, "danceability": 0.34, "brightness": 0.2, "beat_confidence": 0.55, "onset_rate": 4, "dynamic_complexity": 0.55, "arousal": 0.42, "valence": 0.58, "vocal_presence": 0.58},
# ── melbourne_soul → "Melbourne Soul Mix" ─────────────────────────────────────────────────
# Theme:    Passionate, stylish, warm, lively, sophisticated.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned, vocal-forward.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: soul, funk, neo soul, jazz-funk.
# Criteria: style gate parent {funk / soul} · lyric-themes · moodclass · cat:soul_funk_rnb
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "melbourne_soul": {"bpm": 100, "energy": -12, "danceability": 0.55, "brightness": 0.28, "beat_confidence": 0.62, "onset_rate": 4.5, "dynamic_complexity": 0.55, "arousal": 0.55, "valence": 0.65, "vocal_presence": 0.78},
# ── melbourne_sunset → "Melbourne Sunset Mix" ─────────────────────────────────────────────
# Theme:    Warm, carefree, sunny, summery, mellow.
# Sound:    Mid 105bpm, low energy, danceable; bright.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: surf, indie pop, tropical, sunshine pop.
# Criteria: style gate parent {electronic, pop, rock} · moodclass · cat:pop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_sunset": {"bpm": 105, "energy": -12, "danceability": 0.5, "brightness": 0.42, "beat_confidence": 0.62, "onset_rate": 5, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.72, "vocal_presence": 0.65},
# ── melbourne_indie → "Melbourne Indie Mix" ───────────────────────────────────────────────
# Theme:    Wry, witty, laid-back, stylish, charming.
# Sound:    Upbeat 118bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: indie rock, alternative/indie rock, jangle pop.
# Criteria: style gate parent {rock} · cat:rock_indie
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_indie": {"bpm": 118, "energy": -11, "danceability": 0.48, "brightness": 0.3, "beat_confidence": 0.68, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.62, "valence": 0.6, "vocal_presence": 0.72},
# ── melbourne_pubrock → "Melbourne Pub Rock Mix" ──────────────────────────────────────────
# Theme:    Rousing, gutsy, swaggering, brash, raucous.
# Sound:    Upbeat 118bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: aussie rock, pub rock, album rock, hard rock, heartland rock.
# Criteria: style gate parent {rock} · cat:rock_classic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_pubrock": {"bpm": 118, "energy": -10, "danceability": 0.48, "brightness": 0.28, "beat_confidence": 0.72, "onset_rate": 5, "dynamic_complexity": 0.48, "arousal": 0.7, "valence": 0.6, "vocal_presence": 0.65},
# ── melbourne_hiphop → "Melbourne Hip-Hop Mix" ────────────────────────────────────────────
# Theme:    Laid-back, earnest, witty, street-smart, warm.
# Sound:    Mid-slow 92bpm, low energy, danceable; dark-toned, vocal-forward.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: trap, cloud rap, conscious, boom bap, hardcore hip-hop.
# Criteria: style gate parent {hip hop} · lyric-themes · moodclass · cat:hiphop
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "melbourne_hiphop": {"bpm": 92, "energy": -12, "danceability": 0.52, "brightness": 0.24, "beat_confidence": 0.62, "onset_rate": 4, "dynamic_complexity": 0.5, "arousal": 0.55, "valence": 0.55, "vocal_presence": 0.8},
# ── melbourne_postpunk → "Melbourne Post-Punk Mix" ────────────────────────────────────────
# Theme:    Dark, angular, brooding, menacing, intense.
# Sound:    Upbeat 128bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: post-punk, goth rock, new wave.
# Criteria: style gate parent {rock} · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_postpunk": {"bpm": 128, "energy": -9, "danceability": 0.46, "brightness": 0.16, "beat_confidence": 0.78, "onset_rate": 5.5, "dynamic_complexity": 0.5, "arousal": 0.78, "valence": 0.45, "vocal_presence": 0.62},
# ── melbourne_psych → "Melbourne Psych Mix" ───────────────────────────────────────────────
# Theme:    Hypnotic, trippy, manic, driving, druggy.
# Sound:    Upbeat 132bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: psychedelic rock, garage rock.
# Criteria: style gate parent {rock} · moodclass · cat:rock_psych
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_psych": {"bpm": 132, "energy": -9, "danceability": 0.45, "brightness": 0.2, "beat_confidence": 0.78, "onset_rate": 6, "dynamic_complexity": 0.52, "arousal": 0.8, "valence": 0.5, "vocal_presence": 0.55},
# ── melbourne_garagepunk → "Melbourne Garage Punk Mix" ────────────────────────────────────
# Theme:    Aggressive, raucous, rebellious, brash, fierce.
# Sound:    Fast 150bpm, mid energy, danceable; dark-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: punk, pop punk, hardcore, melodic hardcore, garage rock.
# Criteria: style gate parent {rock} · moodclass · cat:rock_punk
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_garagepunk": {"bpm": 150, "energy": -8, "danceability": 0.45, "brightness": 0.22, "beat_confidence": 0.82, "onset_rate": 7, "dynamic_complexity": 0.48, "arousal": 0.85, "valence": 0.45, "vocal_presence": 0.65},
# ── melbourne_club → "Melbourne Club Mix" ─────────────────────────────────────────────────
# Theme:    Exuberant, lively, euphoric, driving, bright.
# Sound:    Upbeat 128bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: house, progressive house, electro, club/dance.
# Criteria: style gate parent {electronic} · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_club": {"bpm": 128, "energy": -8, "danceability": 0.74, "brightness": 0.34, "beat_confidence": 0.82, "onset_rate": 5.5, "dynamic_complexity": 0.35, "arousal": 0.8, "valence": 0.68, "vocal_presence": 0.45},
# ── melbourne_techno → "Melbourne Techno Mix" ─────────────────────────────────────────────
# Theme:    Hypnotic, dark, driving, nocturnal, intense.
# Sound:    Upbeat 130bpm, mid energy, very danceable; dark-toned, near-instrumental.
# Era/Geo:  Any era · geo-tiered to Melbourne.
# Music:    Genre-pure: techno, minimal techno, tech-house, acid house.
# Criteria: style gate parent {electronic} · pop -1 · moodclass · cat:electronic_house_techno
# Flow:     DJ energy-ARC: ease-in → peak ~75% → wind-down (beatmatched/harmonic transitions).
# Enhance:  emb_effnet sub-style cohesion (applied)
    "melbourne_techno": {"bpm": 130, "energy": -9, "danceability": 0.62, "brightness": 0.12, "beat_confidence": 0.82, "onset_rate": 5.5, "dynamic_complexity": 0.34, "arousal": 0.8, "valence": 0.42, "vocal_presence": 0.2},
    # General activity / mood profiles (rotation by acoustic fit)
    # beat_confidence: groove/pulse strength (0=loose, 1=driving)
    # onset_rate: note density in onsets/sec
    # dynamic_complexity: loudness variance (low=compressed, high=dynamic; 0.3=EDM, 0.6=folk/jazz)
    # arousal/valence/vocal_presence: TF-derived — None-safe, only scored when data exists
# ── workout → "Beast Mode" ────────────────────────────────────────────────────────────────
# Theme:    Energetic, aggressive, powerful, intense, triumphant.
# Sound:    Fast 150bpm, high energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · scheduled MWF 16-19 / TuTh 5-8 · lyric-themes · moodclass · cat:workout_energy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  integrated_loudness steadiness + emb_musicnn sounds-like cohesion (applied)
    "workout":    {"bpm": 150, "energy": -7,  "danceability": 0.60, "brightness": 0.30,
                   "beat_confidence": 0.85, "onset_rate": 7.0, "dynamic_complexity": 0.40,
                   "arousal": 0.80, "valence": 0.65, "vocal_presence": 0.60},
# ── running → "Runner's High" ─────────────────────────────────────────────────────────────
# Theme:    Energetic, powerful, lively, rousing, driving.
# Sound:    Fast 160bpm, high energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · scheduled TuTh 12-14 / weekends 8-12 · lyric-themes · cat:workout_energy
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  integrated_loudness steadiness + emb_musicnn sounds-like cohesion (applied)
    "running":    {"bpm": 160, "energy": -6,  "danceability": 0.45, "brightness": 0.28,
                   "beat_confidence": 0.80, "onset_rate": 6.0, "dynamic_complexity": 0.42,
                   "arousal": 0.85, "valence": 0.60, "vocal_presence": 0.55},
# ── party → "Party Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Celebratory, energetic, euphoric, fun, carefree.
# Sound:    Upbeat 125bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "party":      {"bpm": 125, "energy": -9,  "danceability": 0.78, "brightness": 0.38,
                   "beat_confidence": 0.75, "onset_rate": 6.5, "dynamic_complexity": 0.35,
                   "arousal": 0.75, "valence": 0.80, "vocal_presence": 0.70},
# ── happy → "Happy Hits" ──────────────────────────────────────────────────────────────────
# Theme:    Happy, lively, euphoric, joyous, cheerful.
# Sound:    Upbeat 118bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop +1 · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "happy":      {"bpm": 118, "energy": -11, "danceability": 0.65, "brightness": 0.48,
                   "beat_confidence": 0.65, "onset_rate": 5.5, "dynamic_complexity": 0.45,
                   "arousal": 0.65, "valence": 0.85, "vocal_presence": 0.75},
# ── focus → "In the Zone" ─────────────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, meditative, relaxed, atmospheric.
# Sound:    Mid-slow 90bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · scheduled Mon-Fri 7-15 · moodclass · cat:focus_study
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  integrated_loudness steadiness + emb_musicnn sounds-like cohesion (applied)
    "focus":      {"bpm":  90, "energy": -18, "danceability": 0.22, "brightness": 0.10,
                   "beat_confidence": 0.30, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                   "arousal": 0.25, "valence": 0.55, "vocal_presence": 0.15},
# ── chill → "Chill Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Calm, laid-back, mellow, relaxed, peaceful.
# Sound:    Mid-slow 82bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "chill":      {"bpm":  82, "energy": -15, "danceability": 0.32, "brightness": 0.16,
                   "beat_confidence": 0.45, "onset_rate": 3.0, "dynamic_complexity": 0.55,
                   "arousal": 0.30, "valence": 0.65, "vocal_presence": 0.45},
# ── melancholy → "In My Feels" ────────────────────────────────────────────────────────────
# Theme:    Sad, melancholy, bittersweet, somber, plaintive.
# Sound:    Slow 68bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "melancholy": {"bpm":  68, "energy": -15, "danceability": 0.15, "brightness": 0.07,
                   "beat_confidence": 0.35, "onset_rate": 2.5, "dynamic_complexity": 0.60,
                   "arousal": 0.25, "valence": 0.20, "vocal_presence": 0.65},
    # Time-of-day profiles (boosted when current time matches their window)
# ── morning → "Rise & Shine" ──────────────────────────────────────────────────────────────
# Theme:    Cheerful, calm, peaceful, gentle, springlike.
# Sound:    Mid 100bpm, low energy, moderate groove; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · hard-time (5, 12) · lyric-themes · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "morning":    {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.45,
                   "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.55,
                   "arousal": 0.50, "valence": 0.70, "vocal_presence": 0.55},
# ── dinner → "Dinner Mix" ─────────────────────────────────────────────────────────────────
# Theme:    Romantic, sophisticated, elegant, smooth, mellow.
# Sound:    Mid-slow 88bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · hard-time (17, 21) · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "dinner":     {"bpm":  88, "energy": -19, "danceability": 0.25, "brightness": 0.22,
                   "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.65,
                   "arousal": 0.25, "valence": 0.60, "vocal_presence": 0.40},
# ── late_night → "Late Night Feels" ───────────────────────────────────────────────────────
# Theme:    Dark, introspective, brooding, atmospheric, hypnotic.
# Sound:    Slow 78bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · hard-time (22, 2) · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "late_night": {"bpm":  78, "energy": -14, "danceability": 0.42, "brightness": 0.05,
                   "beat_confidence": 0.50, "onset_rate": 3.5, "dynamic_complexity": 0.52,
                   "arousal": 0.40, "valence": 0.35, "vocal_presence": 0.50},
# ── sleep → "Drift Off" ───────────────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, dreamy, soothing, languid.
# Sound:    Slow 65bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · hard-time (21, 4) · moodclass · cat:wellness_sleep
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  integrated_loudness steadiness + emb_musicnn sounds-like cohesion (applied)
    "sleep":      {"bpm":  65, "energy": -23, "danceability": 0.10, "brightness": 0.03,
                   "beat_confidence": 0.15, "onset_rate": 1.0, "dynamic_complexity": 0.70,
                   "arousal": 0.10, "valence": 0.50, "vocal_presence": 0.10},
    # Weather-triggered profiles (boosted when conditions match)
# ── rainy_day → "Rainy Day Mix" ───────────────────────────────────────────────────────────
# Theme:    Melancholy, bittersweet, introspective, nostalgic, wistful.
# Sound:    Slow 72bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · weather-gated · lyric-themes · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "rainy_day":  {"bpm":  72, "energy": -16, "danceability": 0.18, "brightness": 0.09,
                   "beat_confidence": 0.30, "onset_rate": 2.0, "dynamic_complexity": 0.62,
                   "arousal": 0.25, "valence": 0.30, "vocal_presence": 0.60},
# ── sunny → "Sunny Mix" ───────────────────────────────────────────────────────────────────
# Theme:    Happy, lively, carefree, fun, joyous.
# Sound:    Mid 108bpm, mid energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · weather-gated · lyric-themes · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "sunny":      {"bpm": 108, "energy": -11, "danceability": 0.58, "brightness": 0.52,
                   "beat_confidence": 0.65, "onset_rate": 5.0, "dynamic_complexity": 0.48,
                   "arousal": 0.70, "valence": 0.85, "vocal_presence": 0.65},
# ── cosy → "Cosy Mix" ─────────────────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, warm, reassuring, nostalgic.
# Sound:    Slow 75bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · weather-gated · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "cosy":       {"bpm":  75, "energy": -16, "danceability": 0.20, "brightness": 0.18,
                   "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.62,
                   "arousal": 0.25, "valence": 0.65, "vocal_presence": 0.45},
    # ----------------------------------------------------------------
    # Mood / Emotional
    # ----------------------------------------------------------------
# ── nostalgia_mix → "Take Me Back" ────────────────────────────────────────────────────────
# Theme:    Nostalgic, wistful, bittersweet, sentimental, autumnal.
# Sound:    Mid-slow 88bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:nostalgic_throwback
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "nostalgia_mix":    {"bpm":  88, "energy": -14, "danceability": 0.28, "brightness": 0.15,
                         "beat_confidence": 0.35, "onset_rate": 2.5, "dynamic_complexity": 0.58,
                         "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.70},
# ── dreamy_mix → "Dreamy Mix" ─────────────────────────────────────────────────────────────
# Theme:    Dreamy, ethereal, atmospheric, spacey, trippy.
# Sound:    Mid-slow 85bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:dreamy_ethereal
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "dreamy_mix":       {"bpm":  85, "energy": -18, "danceability": 0.25, "brightness": 0.12,
                         "beat_confidence": 0.25, "onset_rate": 2.0, "dynamic_complexity": 0.60,
                         "arousal": 0.20, "valence": 0.55, "vocal_presence": 0.50},
# ── moody_mix → "In a Mood" ───────────────────────────────────────────────────────────────
# Theme:    Dark, introspective, brooding, melancholy, gloomy.
# Sound:    Slow 78bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "moody_mix":        {"bpm":  78, "energy": -16, "danceability": 0.22, "brightness": 0.08,
                         "beat_confidence": 0.40, "onset_rate": 2.8, "dynamic_complexity": 0.58,
                         "arousal": 0.35, "valence": 0.30, "vocal_presence": 0.60},
# ── emotional → "All the Feels" ───────────────────────────────────────────────────────────
# Theme:    Passionate, poignant, bittersweet, intense, powerful.
# Sound:    Mid-slow 95bpm, low energy, moderate groove; warm-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "emotional":        {"bpm":  95, "energy": -12, "danceability": 0.35, "brightness": 0.25,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.65,
                         "arousal": 0.50, "valence": 0.45, "vocal_presence": 0.85},
# ── bittersweet → "Happy-Sad" ─────────────────────────────────────────────────────────────
# Theme:    Bittersweet, nostalgic, wistful, sad, melancholy.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:melancholy_blue
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "bittersweet":      {"bpm":  82, "energy": -15, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.35, "valence": 0.45, "vocal_presence": 0.70},
# ── cathartic → "Let It Out" ──────────────────────────────────────────────────────────────
# Theme:    Intense, powerful, dramatic, triumphant, passionate.
# Sound:    Mid 108bpm, mid energy, moderate groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "cathartic":        {"bpm": 108, "energy": -10, "danceability": 0.38, "brightness": 0.22,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.70,
                         "arousal": 0.70, "valence": 0.40, "vocal_presence": 0.80},
# ── confidence_boost → "Feelin' Myself" ───────────────────────────────────────────────────
# Theme:    Confident, powerful, lively, gutsy, swaggering.
# Sound:    Upbeat 118bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "confidence_boost": {"bpm": 118, "energy":  -9, "danceability": 0.58, "brightness": 0.35,
                         "beat_confidence": 0.70, "onset_rate": 5.5, "dynamic_complexity": 0.40,
                         "arousal": 0.68, "valence": 0.72, "vocal_presence": 0.70},
# ── empowering → "Unstoppable" ────────────────────────────────────────────────────────────
# Theme:    Triumphant, powerful, rousing, anthemic, mighty.
# Sound:    Upbeat 128bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · soft daypart lean · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "empowering":       {"bpm": 128, "energy":  -8, "danceability": 0.50, "brightness": 0.30,
                         "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.55,
                         "arousal": 0.75, "valence": 0.70, "vocal_presence": 0.75},
# ── euphoric → "Cloud Nine" ───────────────────────────────────────────────────────────────
# Theme:    Euphoric, ecstatic, lively, carefree, thrilling.
# Sound:    Upbeat 132bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "euphoric":         {"bpm": 132, "energy":  -8, "danceability": 0.72, "brightness": 0.42,
                         "beat_confidence": 0.80, "onset_rate": 6.5, "dynamic_complexity": 0.35,
                         "arousal": 0.85, "valence": 0.90, "vocal_presence": 0.65},
# ── angst_mix → "Big Feelings" ────────────────────────────────────────────────────────────
# Theme:    Angry, aggressive, intense, nervous, rebellious.
# Sound:    Upbeat 138bpm, mid energy, moderate groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:defiant_intense
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "angst_mix":        {"bpm": 138, "energy":  -9, "danceability": 0.40, "brightness": 0.22,
                         "beat_confidence": 0.65, "onset_rate": 6.0, "dynamic_complexity": 0.55,
                         "arousal": 0.78, "valence": 0.25, "vocal_presence": 0.82},
# ── romantic_mix → "Romantic Mix" ─────────────────────────────────────────────────────────
# Theme:    Romantic, passionate, tender, intimate, sensual.
# Sound:    Mid-slow 88bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "romantic_mix":     {"bpm":  88, "energy": -16, "danceability": 0.28, "brightness": 0.20,
                         "beat_confidence": 0.32, "onset_rate": 2.5, "dynamic_complexity": 0.60,
                         "arousal": 0.28, "valence": 0.70, "vocal_presence": 0.72},
# ── daydreaming → "Head in the Clouds" ────────────────────────────────────────────────────
# Theme:    Dreamy, atmospheric, calm, peaceful, ethereal.
# Sound:    Mid-slow 80bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:dreamy_ethereal
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "daydreaming":      {"bpm":  80, "energy": -19, "danceability": 0.20, "brightness": 0.10,
                         "beat_confidence": 0.22, "onset_rate": 1.8, "dynamic_complexity": 0.62,
                         "arousal": 0.18, "valence": 0.60, "vocal_presence": 0.55},
# ── fresh_start → "Fresh Start Mix" ───────────────────────────────────────────────────────
# Theme:    Optimistic, idealistic, cheerful, bright, springlike.
# Sound:    Mid 105bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled Mon-Fri 5-9 / weekends 8-10 · lyric-themes · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "fresh_start":      {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.40,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.78, "vocal_presence": 0.65},
    # ----------------------------------------------------------------
    # Aesthetic / Time-of-Day (general pool; soft time boost in rotation)
    # ----------------------------------------------------------------
# ── main_character → "Main Character Mix" ─────────────────────────────────────────────────
# Theme:    Confident, dramatic, powerful, swaggering, triumphant.
# Sound:    Mid 115bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:euphoric_triumphant
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "main_character":   {"bpm": 115, "energy": -10, "danceability": 0.48, "brightness": 0.30,
                         "beat_confidence": 0.62, "onset_rate": 5.0, "dynamic_complexity": 0.55,
                         "arousal": 0.65, "valence": 0.58, "vocal_presence": 0.72},
# ── golden_hour → "Golden Hour Mix" ───────────────────────────────────────────────────────
# Theme:    Nostalgic, wistful, warm, optimistic, summery.
# Sound:    Mid-slow 95bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 16-21 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "golden_hour":      {"bpm":  95, "energy": -13, "danceability": 0.35, "brightness": 0.28,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.55,
                         "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.62},
# ── sunset_mix → "Sunset Mix" ─────────────────────────────────────────────────────────────
# Theme:    Nostalgic, wistful, bittersweet, mellow, introspective.
# Sound:    Mid-slow 85bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 16-21 · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "sunset_mix":       {"bpm":  85, "energy": -16, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.35, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.32, "valence": 0.62, "vocal_presence": 0.55},
# ── after_dark → "After Dark Mix" ─────────────────────────────────────────────────────────
# Theme:    Dark, brooding, atmospheric, nocturnal, sensual.
# Sound:    Mid 105bpm, low energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 20-4 · lyric-themes · moodclass · cat:time_of_day
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "after_dark":       {"bpm": 105, "energy": -12, "danceability": 0.52, "brightness": 0.08,
                         "beat_confidence": 0.60, "onset_rate": 4.5, "dynamic_complexity": 0.42,
                         "arousal": 0.48, "valence": 0.38, "vocal_presence": 0.55},
    # ----------------------------------------------------------------
    # Time / Occasion (general pool; soft time boost in rotation)
    # ----------------------------------------------------------------
# ── after_work → "Clock Out" ──────────────────────────────────────────────────────────────
# Theme:    Lively, relaxed, mellow, laid-back, easygoing.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled Mon-Fri 15-20 · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "after_work":       {"bpm": 100, "energy": -12, "danceability": 0.45, "brightness": 0.30,
                         "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.48, "valence": 0.65, "vocal_presence": 0.60},
# ── friday_night → "Friday Night Mix" ─────────────────────────────────────────────────────
# Theme:    Fun, energetic, carefree, lively, celebratory.
# Sound:    Upbeat 118bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · scheduled Fri 18-3 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "friday_night":     {"bpm": 118, "energy": -10, "danceability": 0.60, "brightness": 0.35,
                         "beat_confidence": 0.68, "onset_rate": 5.5, "dynamic_complexity": 0.42,
                         "arousal": 0.68, "valence": 0.78, "vocal_presence": 0.68},
# ── weekend_mix → "Weekend Mix" ───────────────────────────────────────────────────────────
# Theme:    Lively, happy, carefree, relaxed, fun.
# Sound:    Mid 100bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled weekends 8-23 · lyric-themes · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "weekend_mix":      {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.32,
                         "beat_confidence": 0.50, "onset_rate": 4.0, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.72, "vocal_presence": 0.62},
# ── sunday_morning → "Slow Sunday" ────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, gentle, warm, relaxed.
# Sound:    Mid-slow 82bpm, low energy, low groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled Sun 7-13 · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "sunday_morning":   {"bpm":  82, "energy": -16, "danceability": 0.28, "brightness": 0.30,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.32, "valence": 0.72, "vocal_presence": 0.60},
# ── lazy_sunday → "Lazy Sunday Mix" ───────────────────────────────────────────────────────
# Theme:    Calm, peaceful, relaxed, laid-back, easygoing.
# Sound:    Slow 70bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · soft daypart lean · scheduled weekends 10-18 · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "lazy_sunday":      {"bpm":  70, "energy": -19, "danceability": 0.20, "brightness": 0.20,
                         "beat_confidence": 0.28, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.68, "vocal_presence": 0.55},
# ── brunch_mix → "Bottomless Brunch" ──────────────────────────────────────────────────────
# Theme:    Happy, cheerful, lively, carefree, fun.
# Sound:    Mid 100bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled weekends 8-14 · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "brunch_mix":       {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.35,
                         "beat_confidence": 0.50, "onset_rate": 3.8, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.78, "vocal_presence": 0.65},
# ── date_night → "Date Night Mix" ─────────────────────────────────────────────────────────
# Theme:    Romantic, intimate, smooth, elegant, tender.
# Sound:    Mid-slow 88bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 18-24 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "date_night":       {"bpm":  88, "energy": -17, "danceability": 0.28, "brightness": 0.22,
                         "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.68, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Activity / Driving
    # ----------------------------------------------------------------
# ── driving_mix → "Windows Down" ──────────────────────────────────────────────────────────
# Theme:    Energetic, lively, powerful, confident, driving.
# Sound:    Mid 110bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:driving
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "driving_mix":      {"bpm": 110, "energy": -10, "danceability": 0.50, "brightness": 0.32,
                         "beat_confidence": 0.62, "onset_rate": 4.8, "dynamic_complexity": 0.50,
                         "arousal": 0.58, "valence": 0.68, "vocal_presence": 0.70},
# ── night_drive → "Night Drive Mix" ───────────────────────────────────────────────────────
# Theme:    Dark, brooding, atmospheric, introspective, nocturnal.
# Sound:    Mid 100bpm, low energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 19-4 · lyric-themes · moodclass · cat:driving
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "night_drive":      {"bpm": 100, "energy": -12, "danceability": 0.48, "brightness": 0.08,
                         "beat_confidence": 0.55, "onset_rate": 3.8, "dynamic_complexity": 0.45,
                         "arousal": 0.42, "valence": 0.35, "vocal_presence": 0.52},
# ── driving_singalong → "Sing in the Car" ─────────────────────────────────────────────────
# Theme:    Lively, fun, energetic, carefree, cheerful.
# Sound:    Mid 112bpm, mid energy, danceable; warm-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:driving
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "driving_singalong":{"bpm": 112, "energy": -10, "danceability": 0.55, "brightness": 0.38,
                         "beat_confidence": 0.65, "onset_rate": 5.0, "dynamic_complexity": 0.48,
                         "arousal": 0.62, "valence": 0.78, "vocal_presence": 0.85},
# ── road_trip → "Open Road" ───────────────────────────────────────────────────────────────
# Theme:    Lively, energetic, carefree, fun, optimistic.
# Sound:    Mid 108bpm, mid energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:driving
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "road_trip":        {"bpm": 108, "energy": -11, "danceability": 0.48, "brightness": 0.35,
                         "beat_confidence": 0.58, "onset_rate": 4.5, "dynamic_complexity": 0.52,
                         "arousal": 0.58, "valence": 0.72, "vocal_presence": 0.72},
# ── commute_mix → "Beat the Traffic" ──────────────────────────────────────────────────────
# Theme:    Lively, mellow, laid-back, easygoing, relaxed.
# Sound:    Mid 100bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled MWF 6-8 / MWF 15-17 · lyric-themes · cat:driving
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "commute_mix":      {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.28,
                         "beat_confidence": 0.52, "onset_rate": 4.0, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.62, "vocal_presence": 0.65},
# ── walking_mix → "Walk It Out" ───────────────────────────────────────────────────────────
# Theme:    Lively, energetic, carefree, optimistic, fun.
# Sound:    Mid 105bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "walking_mix":      {"bpm": 105, "energy": -12, "danceability": 0.48, "brightness": 0.32,
                         "beat_confidence": 0.58, "onset_rate": 4.5, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.68, "vocal_presence": 0.68},
    # ----------------------------------------------------------------
    # Social / Nostalgia
    # ----------------------------------------------------------------
# ── party_throwback → "Party Throwback Mix" ───────────────────────────────────────────────
# Theme:    Celebratory, energetic, fun, carefree, boisterous.
# Sound:    Upbeat 128bpm, mid energy, very danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "party_throwback":  {"bpm": 128, "energy":  -8, "danceability": 0.72, "brightness": 0.42,
                         "beat_confidence": 0.75, "onset_rate": 6.0, "dynamic_complexity": 0.38,
                         "arousal": 0.78, "valence": 0.78, "vocal_presence": 0.70},
    # ----------------------------------------------------------------
    # Weather / Season (managed by _WEATHER_PROFILES / _SEASONAL_PROFILES)
    # ----------------------------------------------------------------
# ── beach_vibes → "Beach Vibes Mix" ───────────────────────────────────────────────────────
# Theme:    Happy, carefree, summery, relaxed, fun.
# Sound:    Mid 100bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · weather-gated · lyric-themes · moodclass · cat:weather
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "beach_vibes":      {"bpm": 100, "energy": -13, "danceability": 0.55, "brightness": 0.48,
                         "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.82, "vocal_presence": 0.65},
# ── summer_evening → "Summer Evening Mix" ─────────────────────────────────────────────────
# Theme:    Warm, carefree, relaxed, summery, mellow.
# Sound:    Mid 100bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:season_summer
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "summer_evening":   {"bpm": 100, "energy": -13, "danceability": 0.50, "brightness": 0.30,
                         "beat_confidence": 0.52, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.48, "valence": 0.75, "vocal_presence": 0.62},
# ── autumn_mix → "Autumn Mix" ─────────────────────────────────────────────────────────────
# Theme:    Nostalgic, wistful, melancholy, introspective, bittersweet.
# Sound:    Mid-slow 80bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:season_autumn
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "autumn_mix":       {"bpm":  80, "energy": -15, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.65},
# ── winter_mix → "Winter Mix" ─────────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, melancholy, atmospheric, nostalgic.
# Sound:    Slow 75bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:season_winter
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "winter_mix":       {"bpm":  75, "energy": -17, "danceability": 0.22, "brightness": 0.12,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.45, "vocal_presence": 0.60},
# ── spring_mix → "Spring Mix" ─────────────────────────────────────────────────────────────
# Theme:    Optimistic, cheerful, springlike, bright, sunny.
# Sound:    Mid 105bpm, low energy, danceable; bright.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:season_spring
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "spring_mix":       {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.42,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.52,
                         "arousal": 0.55, "valence": 0.78, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Romance
    # ----------------------------------------------------------------
# ── modern_romance → "Modern Romance Mix" ─────────────────────────────────────────────────
# Theme:    Romantic, passionate, tender, poignant, yearning.
# Sound:    Mid-slow 88bpm, low energy, moderate groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "modern_romance":   {"bpm":  88, "energy": -15, "danceability": 0.32, "brightness": 0.22,
                         "beat_confidence": 0.40, "onset_rate": 3.0, "dynamic_complexity": 0.52,
                         "arousal": 0.35, "valence": 0.70, "vocal_presence": 0.75},
# ── late_night_romance → "Late Night Romance Mix" ─────────────────────────────────────────
# Theme:    Romantic, intimate, sensual, yearning, dark.
# Sound:    Slow 72bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 21-3 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "late_night_romance":{"bpm": 72, "energy": -18, "danceability": 0.22, "brightness": 0.08,
                         "beat_confidence": 0.28, "onset_rate": 2.0, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.62, "vocal_presence": 0.72},
# ── romantic_dinner → "Romantic Dinner Mix" ───────────────────────────────────────────────
# Theme:    Romantic, elegant, sophisticated, smooth, intimate.
# Sound:    Mid-slow 80bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 18-22 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "romantic_dinner":  {"bpm":  80, "energy": -20, "danceability": 0.20, "brightness": 0.18,
                         "beat_confidence": 0.25, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.62, "vocal_presence": 0.55},
# ── love_songs → "Love Songs" ─────────────────────────────────────────────────────────────
# Theme:    Romantic, passionate, tender, poignant, earnest.
# Sound:    Mid-slow 90bpm, low energy, moderate groove; warm-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "love_songs":       {"bpm":  90, "energy": -14, "danceability": 0.30, "brightness": 0.25,
                         "beat_confidence": 0.42, "onset_rate": 3.0, "dynamic_complexity": 0.62,
                         "arousal": 0.42, "valence": 0.75, "vocal_presence": 0.82},
# ── slow_dance → "Slow Dance Mix" ─────────────────────────────────────────────────────────
# Theme:    Romantic, tender, intimate, sensual, passionate.
# Sound:    Slow 68bpm, very low energy, low groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "slow_dance":       {"bpm":  68, "energy": -18, "danceability": 0.18, "brightness": 0.15,
                         "beat_confidence": 0.28, "onset_rate": 1.8, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.68, "vocal_presence": 0.78},
# ── candlelight → "Candlelight Mix" ───────────────────────────────────────────────────────
# Theme:    Romantic, elegant, peaceful, gentle, intimate.
# Sound:    Slow 72bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · scheduled 18-24 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "candlelight":      {"bpm":  72, "energy": -20, "danceability": 0.15, "brightness": 0.15,
                         "beat_confidence": 0.22, "onset_rate": 1.5, "dynamic_complexity": 0.70,
                         "arousal": 0.18, "valence": 0.62, "vocal_presence": 0.65},
# ── first_date → "First Date Mix" ─────────────────────────────────────────────────────────
# Theme:    Romantic, optimistic, sweet, tender, gentle.
# Sound:    Mid-slow 95bpm, low energy, moderate groove; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "first_date":       {"bpm":  95, "energy": -14, "danceability": 0.38, "brightness": 0.30,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.55,
                         "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.70},
# ── romantic_jazz → "Romantic Jazz Mix" ───────────────────────────────────────────────────
# Theme:    Romantic, sophisticated, smooth, elegant, intimate.
# Sound:    Slow 78bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: vocal jazz, smooth jazz, cool, piano jazz, crossover jazz, jazz-pop….
# Criteria: style gate parent {jazz} · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "romantic_jazz":    {"bpm":  78, "energy": -20, "danceability": 0.22, "brightness": 0.18,
                         "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.70,
                         "arousal": 0.22, "valence": 0.65, "vocal_presence": 0.72},
# ── jazz_dinner → "Jazz Dinner Mix" ───────────────────────────────────────────────────────
# Theme:    Sophisticated, elegant, smooth, mellow, relaxed.
# Sound:    Mid-slow 85bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: vocal jazz, jazz, cool, smooth jazz, crossover jazz, piano jazz….
# Criteria: style gate parent {jazz} · scheduled 18-22 · moodclass · cat:jazz_lounge
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "jazz_dinner":      {"bpm":  85, "energy": -21, "danceability": 0.25, "brightness": 0.20,
                         "beat_confidence": 0.28, "onset_rate": 2.8, "dynamic_complexity": 0.72,
                         "arousal": 0.20, "valence": 0.60, "vocal_presence": 0.50},
# ── string_quartet → "String Quartet Mix" ─────────────────────────────────────────────────
# Theme:    Elegant, graceful, sophisticated, refined, stately.
# Sound:    Mid-slow 80bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: chamber music, classical crossover, orchestral, neo-classical, modern composition, concerto….
# Criteria: style gate parent {classical, stage & screen} · moodclass · cat:instrumental_cinematic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "string_quartet":   {"bpm":  80, "energy": -19, "danceability": 0.18, "brightness": 0.20,
                         "beat_confidence": 0.20, "onset_rate": 2.0, "dynamic_complexity": 0.75,
                         "arousal": 0.22, "valence": 0.58, "vocal_presence": 0.30},
# ── strings_romance → "Strings & Romance Mix" ─────────────────────────────────────────────
# Theme:    Romantic, elegant, graceful, majestic, lush.
# Sound:    Slow 75bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: orchestral, chamber music, classical crossover, neo-classical, modern composition, chamber pop….
# Criteria: style gate parent {classical, stage & screen} · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "strings_romance":  {"bpm":  75, "energy": -20, "danceability": 0.15, "brightness": 0.18,
                         "beat_confidence": 0.18, "onset_rate": 1.8, "dynamic_complexity": 0.75,
                         "arousal": 0.20, "valence": 0.65, "vocal_presence": 0.32},
# ── piano_romance → "Piano Romance Mix" ───────────────────────────────────────────────────
# Theme:    Romantic, peaceful, tender, intimate, gentle.
# Sound:    Slow 72bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: piano jazz, neo-classical, modern composition, contemporary instrumental, keyboard, classical crossover….
# Criteria: style gate parent {classical, stage & screen} · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "piano_romance":    {"bpm":  72, "energy": -22, "danceability": 0.15, "brightness": 0.16,
                         "beat_confidence": 0.18, "onset_rate": 1.5, "dynamic_complexity": 0.72,
                         "arousal": 0.18, "valence": 0.62, "vocal_presence": 0.35},
# ── acoustic_romance → "Acoustic Romance Mix" ─────────────────────────────────────────────
# Theme:    Romantic, earnest, tender, organic, intimate.
# Sound:    Slow 78bpm, very low energy, low groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Seed-defined by sound: Jack Johnson, Jason Mraz, Donavon Frankenreiter, …
# Criteria: style gate · pop +0.5 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "acoustic_romance": {"bpm":  78, "energy": -18, "danceability": 0.22, "brightness": 0.20,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.65,
                         "arousal": 0.28, "valence": 0.68, "vocal_presence": 0.78},
# ── indie_romance → "Indie Romance Mix" ───────────────────────────────────────────────────
# Theme:    Romantic, dreamy, melancholy, bittersweet, wistful.
# Sound:    Mid-slow 85bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: indie rock, indie pop, dream pop, indie folk, indie electronic, twee pop….
# Criteria: style gate parent {rock} · pop +0.5 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "indie_romance":    {"bpm":  85, "energy": -17, "danceability": 0.28, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.62,
                         "arousal": 0.32, "valence": 0.65, "vocal_presence": 0.72},
# ── synthpop_romance → "Synth-Pop Romance Mix" ────────────────────────────────────────────
# Theme:    Romantic, dreamy, nostalgic, yearning, poignant.
# Sound:    Mid 100bpm, low energy, danceable; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: synth pop, synthwave, new romantic, new wave, indie electronic, dance-pop….
# Criteria: style gate parent {electronic, pop, rock} · pop +0.5 · lyric-themes · moodclass · cat:romantic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion + lyric_lang=en soft lean (applied)
    "synthpop_romance": {"bpm": 100, "energy": -13, "danceability": 0.48, "brightness": 0.15,
                         "beat_confidence": 0.55, "onset_rate": 3.8, "dynamic_complexity": 0.45,
                         "arousal": 0.40, "valence": 0.68, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Gap fills
    # ----------------------------------------------------------------
# ── evening_unwind → "Unwind" ─────────────────────────────────────────────────────────────
# Theme:    Calm, peaceful, mellow, relaxed, smooth.
# Sound:    Slow 78bpm, very low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · scheduled 18-23 · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "evening_unwind":   {"bpm":  78, "energy": -17, "danceability": 0.22, "brightness": 0.15,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.58, "vocal_presence": 0.50},
# ── heartbreak → "Broken Hearts Club" ─────────────────────────────────────────────────────
# Theme:    Sad, yearning, lonely, anguished, tragic.
# Sound:    Slow 75bpm, low energy, low groove; dark-toned, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · lyric-themes · moodclass · cat:heartbreak_longing
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "heartbreak":       {"bpm":  75, "energy": -14, "danceability": 0.18, "brightness": 0.07,
                         "beat_confidence": 0.32, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.45, "valence": 0.12, "vocal_presence": 0.88},
# ── pre_party → "Pre-Party Mix" ───────────────────────────────────────────────────────────
# Theme:    Energetic, fun, carefree, lively, celebratory.
# Sound:    Upbeat 122bpm, mid energy, very danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · soft daypart lean · scheduled Fri,Sat 17-23 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "pre_party":        {"bpm": 122, "energy":  -9, "danceability": 0.65, "brightness": 0.38,
                         "beat_confidence": 0.72, "onset_rate": 5.8, "dynamic_complexity": 0.40,
                         "arousal": 0.72, "valence": 0.75, "vocal_presence": 0.68},
# ── cool_down → "Catch Your Breath" ───────────────────────────────────────────────────────
# Theme:    Calm, relaxed, peaceful, mellow, gentle.
# Sound:    Slow 78bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · moodclass · cat:calm_unwind
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "cool_down":        {"bpm":  78, "energy": -14, "danceability": 0.28, "brightness": 0.20,
                         "beat_confidence": 0.42, "onset_rate": 3.0, "dynamic_complexity": 0.55,
                         "arousal": 0.30, "valence": 0.60, "vocal_presence": 0.50},
# ── cooking_mix → "Kitchen Disco" ─────────────────────────────────────────────────────────
# Theme:    Lively, happy, carefree, fun, cheerful.
# Sound:    Mid 102bpm, low energy, danceable; warm-toned.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · soft daypart lean · moodclass · cat:happy_bright
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion (applied)
    "cooking_mix":      {"bpm": 102, "energy": -12, "danceability": 0.48, "brightness": 0.38,
                         "beat_confidence": 0.55, "onset_rate": 4.2, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.75, "vocal_presence": 0.65},
# ── deep_work → "Locked In" ───────────────────────────────────────────────────────────────
# Theme:    Calm, atmospheric, meditative, peaceful, soothing.
# Sound:    Mid-slow 88bpm, very low energy, low groove; dark-toned, near-instrumental.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate).
# Criteria: no genre gate · pop -1 · scheduled Mon-Fri 7-15 · moodclass · cat:focus_study
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  integrated_loudness steadiness + emb_musicnn sounds-like cohesion (applied)
    "deep_work":        {"bpm":  88, "energy": -17, "danceability": 0.18, "brightness": 0.08,
                         "beat_confidence": 0.25, "onset_rate": 1.8, "dynamic_complexity": 0.68,
                         "arousal": 0.22, "valence": 0.52, "vocal_presence": 0.12},
# ── folk_acoustic → "Folk & Acoustic Mix" ─────────────────────────────────────────────────
# Theme:    Earthy, organic, rustic, pastoral, earnest.
# Sound:    Mid-slow 82bpm, low energy, low groove; dark-toned.
# Era/Geo:  Any era · any origin.
# Music:    Genre-pure: indie folk, contemporary folk, folk-rock, folk-pop, singer/songwriter, americana….
# Criteria: style gate parent {folk, world, & country, rock} · pop +0.5 · moodclass · cat:folk_acoustic
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_effnet sub-style cohesion (applied)
    "folk_acoustic":    {"bpm":  82, "energy": -16, "danceability": 0.25, "brightness": 0.22,
                         "beat_confidence": 0.38, "onset_rate": 2.5, "dynamic_complexity": 0.68,
                         "arousal": 0.30, "valence": 0.62, "vocal_presence": 0.75},
# ── celebration → "Celebration Mix" ───────────────────────────────────────────────────────
# Theme:    Celebratory, triumphant, happy, euphoric, lively.
# Sound:    Upbeat 125bpm, mid energy, very danceable; bright, vocal-forward.
# Era/Geo:  Any era · any origin.
# Music:    Mood/centroid-led (no genre gate), with lyric-theme pull.
# Criteria: no genre gate · pop +1 · lyric-themes · moodclass · cat:party_fun
# Flow:     DJ smooth: greedy+2-opt on beatmatch (octave-BPM) + Camelot key + energy.
# Enhance:  emb_musicnn sounds-like cohesion + lyric_lang=en soft lean (applied)
    "celebration":      {"bpm": 125, "energy":  -8, "danceability": 0.62, "brightness": 0.42,
                         "beat_confidence": 0.72, "onset_rate": 5.8, "dynamic_complexity": 0.45,
                         "arousal": 0.78, "valence": 0.90, "vocal_presence": 0.78},
}

_MOOD_MIX_NAMES = {
    # --- Meloday+ gap-fill mixes ---
    "situationship": "Situationship Mix • Meloday+",
    "sad_bangers": "Sad Bangers Mix • Meloday+",
    "power_ballads": "Lighters Up • Meloday+",
    "restless": "Can't Switch Off • Meloday+",
    "neoclassical": "Neoclassical Calm • Meloday+",
    "yacht_rock": "Yacht Rock Mix • Meloday+",
    "swagger": "Swagger Mix • Meloday+",
    "chart_pop": "Pop Hits • Meloday+",
    "dance_pop": "Dancefloor Pop • Meloday+",
    "indie_pop": "Indie Darlings • Meloday+",
    "synth_pop": "Synth-Pop Nights • Meloday+",
    # --- Meloday+ rock / electronic / scores gap-fill ---
    "indie_rock": "Indie Anthems • Meloday+",
    "post_grunge": "Post-Grunge • Meloday+",
    "rap_rock": "Rap-Rock & Nu-Metal • Meloday+",
    "festival_edm": "EDM Anthems • Meloday+",
    "soundtracks": "Soundtracks & Scores • Meloday+",
    "rave_cave": "Rave Cave • Meloday+",
    # ---- 7 decade mixes (era) ----
    "decade_60s": "60s Mix • Meloday+",
    "decade_70s": "70s Mix • Meloday+",
    "decade_80s": "80s Mix • Meloday+",
    "decade_90s": "90s Mix • Meloday+",
    "decade_00s": "00s Mix • Meloday+",
    "decade_10s": "10s Mix • Meloday+",
    "decade_20s": "20s Mix • Meloday+",
    # ---- 3 geo showcase mixes ----
    "scotland_scene":  "Sounds of Scotland • Meloday+",
    "australia_scene": "Sounds of Australia • Meloday+",
    "london_scene":    "Sounds of London • Meloday+",
    # ---- 3 geo HITS mixes ----
    "scottish_hits":   "Scottish Hits • Meloday+",
    "australian_hits": "Australian Hits • Meloday+",
    "london_hits":     "London Hits • Meloday+",
    "uk_scene":        "Sounds of the UK • Meloday+",
    "uk_hits":         "UK Hits • Meloday+",
    "scotland_now":    "Scotland Now • Meloday+",
    "london_now":      "London Now • Meloday+",
    "uk_now":          "UK Now • Meloday+",
    "australia_now":   "Australia Now • Meloday+",
    "stormy": "Stormy Mix • Meloday+",
    "foggy": "Foggy Mix • Meloday+",
    "snow_day": "Snow Day Mix • Meloday+",
    "heatwave": "Heatwave Mix • Meloday+",
    "frosty": "Frosty Mix • Meloday+",
    "grey_skies": "Grey Skies Mix • Meloday+",
    "windy": "Windy Mix • Meloday+",
    "clear_night": "Clear Night Mix • Meloday+",
    "festive": "Festive Mix • Meloday+",
    "spring_bloom": "Spring Bloom Mix • Meloday+",
    "spring_acoustic": "Spring Acoustic Mix • Meloday+",
    "spring_strings": "Spring Strings Mix • Meloday+",
    "spring_jangle": "Spring Jangle Mix • Meloday+",
    "summer_heat": "Summer Heat Mix • Meloday+",
    "summer_breeze": "Summer Breeze Mix • Meloday+",
    "summer_roadtrip": "Summer Roadtrip Mix • Meloday+",
    "summer_tropical": "Summer Tropical Mix • Meloday+",
    "autumn_leaves": "Autumn Leaves Mix • Meloday+",
    "autumn_jazz": "Autumn Jazz Mix • Meloday+",
    "autumn_rain": "Autumn Rain Mix • Meloday+",
    "autumn_embers": "Autumn Embers Mix • Meloday+",
    "winter_frost": "Winter Frost Mix • Meloday+",
    "winter_cosy": "Winter Cosy Mix • Meloday+",
    "winter_nights": "Winter Nights Mix • Meloday+",
    "winter_jazz": "Winter Jazz Mix • Meloday+",
    "hopeful": "Brighter Days • Meloday+",
    "yearning": "Longing • Meloday+",
    "triumphant": "Victory Lap • Meloday+",
    "serene": "Calm Waters • Meloday+",
    "tender": "Soft Spot • Meloday+",
    "defiant": "No Apologies • Meloday+",
    "vulnerable": "Heart on Sleeve • Meloday+",
    "awe_wonder": "Awe & Wonder Mix • Meloday+",
    "grief_release": "Letting Go • Meloday+",
    "sunrise": "Sunrise Mix • Meloday+",
    "blue_hour": "Blue Hour Mix • Meloday+",
    "midnight": "Midnight Mix • Meloday+",
    "three_am": "3am Thoughts • Meloday+",
    "golden_afternoon": "Golden Afternoon Mix • Meloday+",
    "overcast": "Overcast Mix • Meloday+",
    "starlit": "Starlit Mix • Meloday+",
    "witching_hour": "Witching Hour Mix • Meloday+",
    "monday_motivation": "Monday Motivation Mix • Meloday+",
    "midweek_reset": "Midweek Reset Mix • Meloday+",
    "friday_feeling": "Finally Friday • Meloday+",
    "sunday_scaries": "Sunday Scaries Mix • Meloday+",
    "treat_yourself": "Treat Yourself Mix • Meloday+",
    "dinner_party": "Dinner Party Mix • Meloday+",
    "housework_hustle": "Tidy Up • Meloday+",
    "study_session": "Brain Food • Meloday+",
    "wind_down": "Wind-Down Mix • Meloday+",
    "yoga_stretch": "Yoga & Stretch Mix • Meloday+",
    "meditation": "Meditation Mix • Meloday+",
    "deep_reading": "Lost in a Book • Meloday+",
    "creative_flow": "In the Flow • Meloday+",
    "gaming": "Game On • Meloday+",
    "gardening": "Green Thumb • Meloday+",
    "spa_bath": "Spa Day • Meloday+",
    "power_nap": "Forty Winks • Meloday+",
    "throwback_anthems": "Throwback Anthems Mix • Meloday+",
    "old_friends": "Old Friends Mix • Meloday+",
    "campfire": "Campfire Mix • Meloday+",
    "cookout": "Cookout Mix • Meloday+",
    "game_night": "Game Night Mix • Meloday+",
    "singalong": "Singalong Mix • Meloday+",
    "school_days": "School Days Mix • Meloday+",
    "memory_lane": "Memory Lane Mix • Meloday+",
    "crush": "Crushing • Meloday+",
    "slow_burn": "Slow Burn Mix • Meloday+",
    "moving_on": "Over It • Meloday+",
    "loved_up": "Loved Up Mix • Meloday+",
    "long_distance": "Miles Apart • Meloday+",
    "flirty": "Make a Move • Meloday+",
    "devotion": "All Yours • Meloday+",
    "wedding_day": "Wedding Day Mix • Meloday+",
    "funk_disco": "Funk & Disco Mix • Meloday+",
    "neo_soul": "Neo-Soul & Quiet Storm Mix • Meloday+",
    "motown_soul": "Motown & Classic Soul Mix • Meloday+",
    "after_hours_rnb": "After-Hours R&B Mix • Meloday+",
    "acid_jazz": "Acid Jazz & Jazz-Funk Mix • Meloday+",
    "boom_bap": "Boom Bap Mix • Meloday+",
    "conscious_flow": "Conscious Flow Mix • Meloday+",
    "g_funk": "G-Funk & West Coast Mix • Meloday+",
    "trap_mode": "Trap Mode Mix • Meloday+",
    "lofi_beats": "Lo-Fi Beats Mix • Meloday+",
    "house_party": "House Party Mix • Meloday+",
    "deep_house": "Deep House Late Mix • Meloday+",
    "techno": "Techno Warehouse Mix • Meloday+",
    "trance": "Trance Heights Mix • Meloday+",
    "dnb": "Drum & Bass Mix • Meloday+",
    "bass_drop": "Bass Drop Mix • Meloday+",
    "uk_garage": "UK Garage & 2-Step Mix • Meloday+",
    "synthwave": "Synthwave & Retrowave Mix • Meloday+",
    "industrial": "Industrial & EBM Mix • Meloday+",
    "vaporwave": "Vaporwave & Chillsynth Mix • Meloday+",
    "downtempo": "Downtempo Drift Mix • Meloday+",
    "hyperpop": "Hyperpop & Glitch Mix • Meloday+",
    "classic_rock": "Classic Rock Mix • Meloday+",
    "heavy_riffs": "Heavy Riffs Mix • Meloday+",
    "punk_energy": "Punk Energy Mix • Meloday+",
    "garage_grunge": "Garage & Grunge Mix • Meloday+",
    "emo_poppunk": "Emo & Pop-Punk Mix • Meloday+",
    "britpop_rock": "Britpop & Madchester Mix • Meloday+",
    "blues_bar": "Blues Bar Mix • Meloday+",
    "psych_haze": "Psych Haze Mix • Meloday+",
    "prog_rock": "Prog & Art Rock Mix • Meloday+",
    "stoner_rock": "Stoner & Desert Rock Mix • Meloday+",
    "reggae_dub": "Reggae & Dub Mix • Meloday+",
    "afrobeat": "Afrobeat Mix • Meloday+",
    "latin_heat": "Latin Heat Mix • Meloday+",
    "bossa_samba": "Bossa & Samba Mix • Meloday+",
    "celtic_folk": "Celtic & Folk Traditions Mix • Meloday+",
    "ska": "Ska & Two-Tone Mix • Meloday+",
    "bebop": "Bebop Mix • Meloday+",
    "swing_bigband": "Swing & Big Band Mix • Meloday+",
    "smooth_jazz": "Smooth Jazz & Lounge Mix • Meloday+",
    "country_roads": "Country Roads Mix • Meloday+",
    "outlaw_country": "Outlaw & Alt-Country Mix • Meloday+",
    "bluegrass": "Bluegrass & Banjo Mix • Meloday+",
    "rockabilly_surf": "Rockabilly & Surf Mix • Meloday+",
    "cinematic_epic": "Cinematic Epic Mix • Meloday+",
    "ambient_drift": "Ambient Drift Mix • Meloday+",
    "post_rock": "Post-Rock Crescendo Mix • Meloday+",
    "chiptune": "8-Bit & Game Mix • Meloday+",
    "gospel": "Gospel & Choir Mix • Meloday+",
    "glasgow_folk": "Glasgow Folk Mix • Meloday+",
    "glasgow_dream": "Glasgow Dream Mix • Meloday+",
    "glasgow_indie": "Glasgow Indie Mix • Meloday+",
    "glasgow_soul": "Glasgow Soul Mix • Meloday+",
    "glasgow_postrock": "Glasgow Post-Rock Mix • Meloday+",
    "glasgow_anthems": "Glasgow Anthems Mix • Meloday+",
    "glasgow_synth": "Glasgow Synth Mix • Meloday+",
    "glasgow_postpunk": "Glasgow Post-Punk Mix • Meloday+",
    "glasgow_house": "Glasgow House Mix • Meloday+",
    "glasgow_underground": "Glasgow Underground Mix • Meloday+",
    "glasgow_bass": "Glasgow Bass Mix • Meloday+",
    "glasgow_late": "Glasgow Late Mix • Meloday+",
    "london_dub": "London Dub Mix • Meloday+",
    "london_soul": "London Soul Mix • Meloday+",
    "london_jazz": "London Jazz Mix • Meloday+",
    "london_triphop": "London Trip-Hop Mix • Meloday+",
    "london_mod": "London Mod Mix • Meloday+",
    "london_britpop": "London Britpop Mix • Meloday+",
    "london_indie": "London Indie Mix • Meloday+",
    "london_calling": "London Calling Mix • Meloday+",
    "london_garage": "London Garage Mix • Meloday+",
    "london_grime": "London Grime Mix • Meloday+",
    "london_dubstep": "London Dubstep Mix • Meloday+",
    "london_jungle": "London Jungle Mix • Meloday+",
    "melbourne_folk": "Melbourne Folk Mix • Meloday+",
    "melbourne_dream": "Melbourne Dream Mix • Meloday+",
    "melbourne_soul": "Melbourne Soul Mix • Meloday+",
    "melbourne_sunset": "Melbourne Sunset Mix • Meloday+",
    "melbourne_indie": "Melbourne Indie Mix • Meloday+",
    "melbourne_pubrock": "Melbourne Pub Rock Mix • Meloday+",
    "melbourne_hiphop": "Melbourne Hip-Hop Mix • Meloday+",
    "melbourne_postpunk": "Melbourne Post-Punk Mix • Meloday+",
    "melbourne_psych": "Melbourne Psych Mix • Meloday+",
    "melbourne_garagepunk": "Melbourne Garage Punk Mix • Meloday+",
    "melbourne_club": "Melbourne Club Mix • Meloday+",
    "melbourne_techno": "Melbourne Techno Mix • Meloday+",
    # Original 14
    "workout":            "Beast Mode • Meloday+",
    "running":            "Runner's High • Meloday+",
    "party":              "Party Mix • Meloday+",
    "happy":              "Happy Hits • Meloday+",
    "focus":              "In the Zone • Meloday+",
    "chill":              "Chill Mix • Meloday+",
    "melancholy":         "In My Feels • Meloday+",
    "morning":            "Rise & Shine • Meloday+",
    "dinner":             "Dinner Mix • Meloday+",
    "late_night":         "Late Night Feels • Meloday+",
    "sleep":              "Drift Off • Meloday+",
    "rainy_day":          "Rainy Day Mix • Meloday+",
    "sunny":              "Sunny Mix • Meloday+",
    "cosy":               "Cosy Mix • Meloday+",
    # Mood / Emotional
    "nostalgia_mix":      "Take Me Back • Meloday+",
    "dreamy_mix":         "Dreamy Mix • Meloday+",
    "moody_mix":          "In a Mood • Meloday+",
    "emotional":          "All the Feels • Meloday+",
    "bittersweet":        "Happy-Sad • Meloday+",
    "cathartic":          "Let It Out • Meloday+",
    "confidence_boost":   "Feelin' Myself • Meloday+",
    "empowering":         "Unstoppable • Meloday+",
    "euphoric":           "Cloud Nine • Meloday+",
    "angst_mix":          "Big Feelings • Meloday+",
    "romantic_mix":       "Romantic Mix • Meloday+",
    "daydreaming":        "Head in the Clouds • Meloday+",
    "fresh_start":        "Fresh Start Mix • Meloday+",
    # Aesthetic / Time-of-Day
    "main_character":     "Main Character Mix • Meloday+",
    "golden_hour":        "Golden Hour Mix • Meloday+",
    "sunset_mix":         "Sunset Mix • Meloday+",
    "after_dark":         "After Dark Mix • Meloday+",
    # Time / Occasion
    "after_work":         "Clock Out • Meloday+",
    "friday_night":       "Friday Night Mix • Meloday+",
    "weekend_mix":        "Weekend Mix • Meloday+",
    "sunday_morning":     "Slow Sunday • Meloday+",
    "lazy_sunday":        "Lazy Sunday Mix • Meloday+",
    "brunch_mix":         "Bottomless Brunch • Meloday+",
    "date_night":         "Date Night Mix • Meloday+",
    # Activity
    "driving_mix":        "Windows Down • Meloday+",
    "night_drive":        "Night Drive Mix • Meloday+",
    "driving_singalong":  "Sing in the Car • Meloday+",
    "road_trip":          "Open Road • Meloday+",
    "commute_mix":        "Beat the Traffic • Meloday+",
    "walking_mix":        "Walk It Out • Meloday+",
    # Social / Nostalgia
    "party_throwback":    "Party Throwback Mix • Meloday+",
    # Weather / Season
    "beach_vibes":        "Beach Vibes Mix • Meloday+",
    "summer_evening":     "Summer Evening Mix • Meloday+",
    "autumn_mix":         "Autumn Mix • Meloday+",
    "winter_mix":         "Winter Mix • Meloday+",
    "spring_mix":         "Spring Mix • Meloday+",
    # Romance
    "modern_romance":     "Modern Romance Mix • Meloday+",
    "late_night_romance": "Late Night Romance Mix • Meloday+",
    "romantic_dinner":    "Romantic Dinner Mix • Meloday+",
    "love_songs":         "Love Songs • Meloday+",
    "slow_dance":         "Slow Dance Mix • Meloday+",
    "candlelight":        "Candlelight Mix • Meloday+",
    "first_date":         "First Date Mix • Meloday+",
    "romantic_jazz":      "Romantic Jazz Mix • Meloday+",
    "jazz_dinner":        "Jazz Dinner Mix • Meloday+",
    "string_quartet":     "String Quartet Mix • Meloday+",
    "strings_romance":    "Strings & Romance Mix • Meloday+",
    "piano_romance":      "Piano Romance Mix • Meloday+",
    "acoustic_romance":   "Acoustic Romance Mix • Meloday+",
    "indie_romance":      "Indie Romance Mix • Meloday+",
    "synthpop_romance":   "Synth-Pop Romance Mix • Meloday+",
    # Gap fills
    "evening_unwind":     "Unwind • Meloday+",
    "heartbreak":         "Broken Hearts Club • Meloday+",
    "pre_party":          "Pre-Party Mix • Meloday+",
    "cool_down":          "Catch Your Breath • Meloday+",
    "cooking_mix":        "Kitchen Disco • Meloday+",
    "deep_work":          "Locked In • Meloday+",
    "folk_acoustic":      "Folk & Acoustic Mix • Meloday+",
    "celebration":        "Celebration Mix • Meloday+",
}

# Mood-profile keys get a bottom-bar text treatment (Spotify Niche/Mood Mix style)
# rather than the standard near-bottom overlay used by the other extras.
_MOOD_PROFILE_KEYS = set(_MOOD_MIX_NAMES.keys())

# ---------------------------------------------------------------------------
# Rotation infrastructure: time windows, weather/season sets, categories
# ---------------------------------------------------------------------------

# Hard time-of-day profiles — actively added/removed by the boundary cron.
# Windows use (start, end) in 0–23. If start > end the window wraps past midnight.
_TIME_BIASED_PROFILES = {
    "morning":    ( 5, 12),   # 5am–noon
    "dinner":     (17, 21),   # 5pm–9pm
    "late_night": (22,  2),   # 10pm–2am
    "sleep":      (21,  4),   # 9pm–4am
}

# Soft time boosts — general-pool profiles that score better during a time window
# but are NOT managed by the boundary cron. (start_hour, end_hour, reduction_amount)
_TIME_SOFT_BOOSTS = {
    "brunch_mix":         ( 9, 13, 0.12),   # 9am–1pm
    "golden_hour":        (15, 20, 0.15),   # 3pm–8pm
    "after_work":         (16, 20, 0.12),   # 4pm–8pm
    "sunset_mix":         (17, 21, 0.15),   # 5pm–9pm
    "date_night":         (17,  2, 0.10),   # 5pm–2am
    "friday_night":       (17,  2, 0.10),   # 5pm–2am (any evening, not just Friday)
    "after_dark":         (22,  4, 0.15),   # 10pm–4am
    "night_drive":        (20,  4, 0.12),   # 8pm–4am
    "late_night_romance": (21,  3, 0.12),   # 9pm–3am
    "focus":              ( 7, 15, 0.15),   # 7am–3pm work window
    "deep_work":          ( 7, 15, 0.12),   # 7am–3pm work window
    "commute_mix":        ( 6,  9, 0.10),   # morning commute
    "sunday_morning":     ( 7, 14, 0.08),   # any morning (day-of-week not tracked)
    "lazy_sunday":        (11, 18, 0.07),   # any quiet afternoon
    "evening_unwind":     (19, 23, 0.15),   # 7pm–11pm (dinner-to-late-night gap)
    "pre_party":          (17, 23, 0.12),   # 5pm–11pm evenings
    "cooking_mix":        (17, 20, 0.10),   # 5pm–8pm dinner-prep window
    # --- new mood/vibe mixes: daypart leans (hard-gated ones get their in-window pull) ---
    "sunrise":            ( 5,  9, 0.20),
    "blue_hour":          (17, 20, 0.12),
    "golden_afternoon":   (12, 17, 0.12),
    "midnight":           (22,  3, 0.15),
    "three_am":           ( 0,  4, 0.15),
    "witching_hour":      (22,  4, 0.12),
    "starlit":            (21,  4, 0.12),
    "wind_down":          (19, 23, 0.15),
    "dinner_party":       (18, 22, 0.15),
    "power_nap":          (13, 16, 0.12),
    "study_session":      ( 8, 17, 0.12),
    "spa_bath":           (19, 23, 0.12),
    "yoga_stretch":       ( 6, 10, 0.08),
    "meditation":         ( 6, 10, 0.06),
    "monday_motivation":  ( 6, 11, 0.15),
    "friday_feeling":     (12, 18, 0.15),
    "sunday_scaries":     (16, 22, 0.12),
    # new-romance evening lean
    "crush":              (18, 24, 0.08),
    "slow_burn":          (18, 24, 0.08),
    "loved_up":           (18, 24, 0.08),
    "long_distance":      (18, 24, 0.08),
    "flirty":             (18, 24, 0.08),
    "devotion":           (18, 24, 0.08),
    # weekend/social evening + daytime windows
    "throwback_anthems":  (18,  1, 0.08),
    "old_friends":        (12, 23, 0.06),
    "game_night":         (18, 23, 0.08),
    "singalong":          (18,  1, 0.06),
    "treat_yourself":     (17, 24, 0.10),
    "housework_hustle":   ( 9, 15, 0.10),
    "cookout":            (12, 18, 0.10),
    "gospel":             ( 8, 12, 0.08),   # Sunday-service morning lean (+ {6} weekday boost)
    # genre mixes with a clear daypart
    "lofi_beats":         ( 9, 18, 0.10),
    "smooth_jazz":        (18, 23, 0.12),
    "ambient_drift":      (21,  5, 0.12),
    "downtempo":          (21,  3, 0.10),
    "deep_house":         (21,  4, 0.10),
    # --- audit-added daypart leans (Tier 1/2) ---
    "synth_pop":          (19,  2, 0.08),   # WHY: name is "Synth-Pop Nights" but had no evening bias (synth_pop-1)
    "situationship":      (18, 24, 0.08),   # WHY: parity with its new-romance siblings' evening lean (situationship-1)
    "sad_bangers":        (20,  2, 0.06),   # WHY: cathartic dance-cry peaks on a night out (sad_bangers-1)
    "restless":           (21,  3, 0.06),   # WHY: its late-night/intrusive-thoughts frame skews night (restless-1)
    "neoclassical":       (19, 24, 0.06),   # WHY: contemplative listening skews evening (neoclassical-1)
    "triumphant":         ( 6, 11, 0.06),   # WHY: morning-motivation context (triumphant-1)
    "empowering":         ( 6, 11, 0.06),   # WHY: morning-motivation context (empowering-1)
    "awe_wonder":         (20,  1, 0.06),   # WHY: awe/cosmic listening skews night (awe_wonder-1)
    "moving_on":          (18, 24, 0.06),   # WHY: parity with romance siblings (moving_on-1)
    # Romance MOODS (the unscheduled ones) lean evening so a rotating few surface each night in the
    # anytime tier — "a date can be any night", but we don't force the whole romance shelf nightly.
    "love_songs":         (18, 24, 0.08),
    "slow_dance":         (18, 24, 0.08),
    "romantic_mix":       (18, 24, 0.08),
    "modern_romance":     (18, 24, 0.08),
    "first_date":         (18, 24, 0.08),
    "indie_romance":      (18, 24, 0.06),
    "acoustic_romance":   (18, 24, 0.06),
    "synthpop_romance":   (19,  2, 0.06),
    "romantic_jazz":      (18, 24, 0.08),
    "piano_romance":      (19, 24, 0.06),
    "strings_romance":    (18, 24, 0.06),
}

# Day-of-week boosts — applied on top of soft time boosts for profiles with a natural weekday.
# 0=Monday … 6=Sunday. Amount is an additional score reduction (lower = more likely selected).
_WEEKDAY_BOOSTS = {
    "friday_night":    ({4},     0.15),   # Friday evenings
    "pre_party":       ({4, 5},  0.10),   # Fri/Sat
    "weekend_mix":     ({5, 6},  0.10),   # Sat/Sun all day
    "brunch_mix":      ({5, 6},  0.08),   # Weekend brunch
    "sunday_morning":  ({6},     0.12),   # Sunday only
    "lazy_sunday":     ({6},     0.10),   # Sunday only
    "celebration":     ({4, 5},  0.08),   # Fri/Sat
    "party_throwback": ({4, 5},  0.06),   # Fri/Sat (slight)
    "focus":           ({0, 1, 2, 3, 4}, 0.12),  # Mon-Fri work boost
    "deep_work":       ({0, 1, 2, 3, 4}, 0.10),  # Mon-Fri work boost
    # --- new day-bound / weekend-social mixes ---
    "monday_motivation": ({0},     0.18),   # Monday
    "friday_feeling":    ({4},     0.15),   # Friday
    "sunday_scaries":    ({6},     0.12),   # Sunday
    "midweek_reset":     ({2, 3},  0.10),   # Wed/Thu
    "throwback_anthems": ({4, 5},  0.10),   # Fri/Sat
    "old_friends":       ({5, 6},  0.08),   # weekend
    "game_night":        ({4, 5},  0.10),   # Fri/Sat
    "singalong":         ({4, 5},  0.08),   # Fri/Sat
    "treat_yourself":    ({4, 5},  0.08),   # Fri/Sat
    "housework_hustle":  ({5, 6},  0.10),   # weekend chores
    "cookout":           ({5, 6},  0.10),   # weekend
    "gospel":            ({6},     0.10),   # Sunday
    "sad_bangers":       ({4, 5},  0.06),   # WHY: night-out lean (audit sad_bangers-1)
}

# (The old hard day-of-week + hour gate dicts — _WEEKDAY_RESTRICTED / _HOUR_RESTRICTED — were replaced by
#  the richer per-profile _PROFILE_SCHEDULE below, which expresses day-dependent, multi-window timing.)

# ---------------------------------------------------------------------------
# Per-profile SCHEDULE — the "right playlist at the right time" gate (replaces the old single-window hour
# + weekday gate dicts). Each value is a LIST of windows (dayset, start_hour, end_hour); dayset is a
# frozenset of weekdays (0=Mon … 6=Sun) or None = any day; (start,end) wraps midnight via _in_time_window.
# Windows OR together, so one profile can be day-dependent AND multi-window (e.g. workout: MWF evening +
# Tue/Thu early morning). A SCHEDULED profile surfaces ONLY in its windows (the uncapped "context" tier);
# a profile NOT here is an "anytime" rotating mix. City mixes (geo tier), weather/seasonal profiles, and
# the 4 cron mixes (_TIME_PROFILES) are deliberately NOT scheduled here.
# ---------------------------------------------------------------------------
_SCHED_MWF      = frozenset({0, 2, 4})        # office/gym days
_SCHED_TT       = frozenset({1, 3})           # WFH / early-workout / lunch-run days
_SCHED_WEEKDAYS = frozenset({0, 1, 2, 3, 4})
_SCHED_WEEKEND  = frozenset({5, 6})
_SCHED_FRI_WKND = frozenset({4, 5, 6})        # "going out" nights
_WORKOUT_WINDOWS = [(_SCHED_MWF, 16, 19), (_SCHED_TT, 5, 8)]   # MWF gym / Tue-Thu early

_PROFILE_SCHEDULE = {
    # wake / early morning
    "fresh_start":       [(_SCHED_WEEKDAYS, 5, 9), (_SCHED_WEEKEND, 8, 10)],
    "sunrise":           [(None, 5, 9)],
    "sunday_morning":    [(frozenset({6}), 7, 13)],
    # commute — MWF office days only, morning + evening home leg (Tue/Thu WFH)
    "commute_mix":       [(_SCHED_MWF, 6, 8), (_SCHED_MWF, 15, 17)],
    # work — all weekdays (office MWF + WFH Tue/Thu)
    "focus":             [(_SCHED_WEEKDAYS, 7, 15)],
    "deep_work":         [(_SCHED_WEEKDAYS, 7, 15)],
    "study_session":     [(_SCHED_WEEKDAYS, 7, 15)],
    # lunch (reused mixes) + weekend cookout
    "cookout":           [(None, 12, 14), (_SCHED_WEEKEND, 12, 18)],
    "golden_afternoon":  [(None, 12, 18)],
    # midday run (Tue/Thu) + weekend morning
    "running":           [(_SCHED_TT, 12, 14), (_SCHED_WEEKEND, 8, 12)],
    # walk Henry — weekday late afternoon / weekend morning
    "walking_mix":       [(_SCHED_WEEKDAYS, 16, 18), (_SCHED_WEEKEND, 8, 10)],
    # workout — MWF evening / Tue-Thu early AM
    "workout":           list(_WORKOUT_WINDOWS),
    "cool_down":         [(_SCHED_MWF, 18, 19), (_SCHED_TT, 7, 9)],
    # cook dinner / dinner guests / a date — ANY night
    "cooking_mix":       [(None, 17, 19)],
    "dinner_party":      [(None, 18, 22)],
    "romantic_dinner":   [(None, 18, 22)],
    "jazz_dinner":       [(None, 18, 22)],
    # evening (any night)
    "evening_unwind":    [(None, 18, 23)],
    "wind_down":         [(None, 19, 23)],
    "date_night":        [(None, 18, 24)],
    "candlelight":       [(None, 18, 24)],
    "blue_hour":         [(None, 16, 21)],
    "golden_hour":       [(None, 16, 21)],
    "sunset_mix":        [(None, 16, 21)],
    "game_night":        [(_SCHED_FRI_WKND, 18, 24)],
    # night / late
    "late_night_romance":[(None, 21, 3)],
    "after_dark":        [(None, 20, 4)],
    "night_drive":       [(None, 19, 4)],
    "after_hours_rnb":   [(None, 21, 4)],
    "midnight":          [(None, 22, 4)],
    "three_am":          [(None, 0, 4)],
    "witching_hour":     [(None, 22, 4)],
    "starlit":           [(None, 21, 4)],
    "winter_nights":     [(None, 18, 3)],
    # going out (Fri + weekend nights)
    "friday_night":      [(frozenset({4}), 18, 3)],
    "pre_party":         [(frozenset({4, 5}), 17, 23)],
    "party":             [(_SCHED_FRI_WKND, 20, 4)],
    "party_throwback":   [(frozenset({4, 5}), 20, 3)],
    "celebration":       [(_SCHED_FRI_WKND, 18, 24)],
    # weekday occasion / day-named
    "after_work":        [(_SCHED_WEEKDAYS, 15, 20)],
    "monday_motivation": [(frozenset({0}), 5, 12)],
    "friday_feeling":    [(frozenset({4}), 11, 19)],
    "sunday_scaries":    [(frozenset({6}), 15, 23)],
    "midweek_reset":     [(frozenset({1, 2, 3}), 7, 20)],
    # weekend flexible (reused)
    "brunch_mix":        [(_SCHED_WEEKEND, 8, 14)],
    "weekend_mix":       [(_SCHED_WEEKEND, 8, 23)],
    "lazy_sunday":       [(_SCHED_WEEKEND, 10, 18)],
    "road_trip":         [(_SCHED_WEEKEND, 9, 18)],
    "campfire":          [(_SCHED_WEEKEND, 16, 21)],
    "folk_acoustic":     [(_SCHED_WEEKEND, 9, 18)],
    # wellness routines
    "yoga_stretch":      [(None, 6, 10)],
    "meditation":        [(None, 6, 10)],
    "spa_bath":          [(None, 19, 23)],
    "power_nap":         [(None, 13, 16)],
}
# Dance / EDM family — present when going out (Fri + weekend nights) AND during workouts (high-energy).
_DANCE_SCHEDULE_KEYS = {"rave_cave", "festival_edm", "techno", "trance", "dnb", "house_party",
                        "uk_garage", "bass_drop", "deep_house", "hyperpop", "industrial",
                        "dance_pop", "funk_disco"}
for _dk in _DANCE_SCHEDULE_KEYS:
    _PROFILE_SCHEDULE.setdefault(_dk, []).append((_SCHED_FRI_WKND, 20, 4))
    _PROFILE_SCHEDULE[_dk].extend(_WORKOUT_WINDOWS)

# Weather-conditional profiles — require weather data; add/remove when conditions match.
_WEATHER_PROFILES = {"rainy_day", "sunny", "cosy", "beach_vibes", "stormy", "foggy", "snow_day", "heatwave", "frosty", "grey_skies", "windy", "clear_night"}
# NOTE: "overcast" is deliberately NOT weather-gated — it is the anytime introspective twin of the
# weather-gated "grey_skies" (same mood signals), available all day rather than only when it's cloudy. (audit overcast-1)

# Season-conditional profiles — triggered by current calendar season; no weather API needed.
_SEASONAL_PROFILES = {"autumn_mix", "winter_mix", "spring_mix", "summer_evening", "festive"}

# Always-present geo showcase mixes — pinned into every general run (never rotated, never removed); like
# the daily mixes they're always there, and their content refreshes once a day via their date-seed shuffle.
_PINNED_PROFILES  = {"scotland_scene", "australia_scene", "london_scene",
                     "scottish_hits", "australian_hits", "london_hits",
                     "uk_scene", "uk_hits", "scotland_now", "london_now", "uk_now", "australia_now"}
_TIME_PROFILES    = set(_TIME_BIASED_PROFILES.keys())
# Retired mixes — dropped from the slate because no membership signal survives the pure-Discogs gate:
# lofi_beats (Discogs has no "Lo-Fi"; the chillhop sound IS Downtempo, already its own mix).
# (adult_alt was retired for the same reason and is now FULLY removed, not a soft-retired stub. WHY: its
# style gate matched 0 tracks, so every lingering config entry — centroid, name, floor, etc. — was dead.)
_RETIRED_PROFILES = {"lofi_beats"}
_GENERAL_PROFILES = (set(_MOOD_PROFILES)
                     - _TIME_PROFILES
                     - _WEATHER_PROFILES
                     - _SEASONAL_PROFILES
                     - _PINNED_PROFILES
                     - _RETIRED_PROFILES)

# Category labels for diversity-aware rotation (max 2 per category in active slots).
_PROFILE_CATEGORY = {
    # rock_classic
    "classic_rock": "rock_classic", "blues_bar": "rock_classic", "rockabilly_surf": "rock_classic",
    "melbourne_pubrock": "rock_classic",
    # rock_indie
    "indie_rock": "rock_indie", "britpop_rock": "rock_indie", "glasgow_indie": "rock_indie",
    "glasgow_anthems": "rock_indie", "london_indie": "rock_indie", "london_britpop": "rock_indie",
    "london_mod": "rock_indie", "melbourne_indie": "rock_indie",
    # rock_punk
    "punk_energy": "rock_punk", "garage_grunge": "rock_punk", "emo_poppunk": "rock_punk",
    "post_grunge": "rock_punk", "london_calling": "rock_punk", "glasgow_postpunk": "rock_punk",
    "melbourne_postpunk": "rock_punk", "melbourne_garagepunk": "rock_punk",
    # rock_heavy
    "heavy_riffs": "rock_heavy", "stoner_rock": "rock_heavy", "rap_rock": "rock_heavy",
    # rock_psych
    "psych_haze": "rock_psych", "prog_rock": "rock_psych", "glasgow_dream": "rock_psych",
    "melbourne_dream": "rock_psych", "melbourne_psych": "rock_psych",
    # pop
    "chart_pop": "pop", "indie_pop": "pop", "yacht_rock": "pop", "synth_pop": "pop", "glasgow_synth": "pop",
    "melbourne_sunset": "pop",
    # electronic_house_techno
    "techno": "electronic_house_techno", "deep_house": "electronic_house_techno",
    "house_party": "electronic_house_techno", "trance": "electronic_house_techno",
    "industrial": "electronic_house_techno", "glasgow_house": "electronic_house_techno",
    "glasgow_underground": "electronic_house_techno", "melbourne_club": "electronic_house_techno",
    "melbourne_techno": "electronic_house_techno",
    # electronic_bass
    "dnb": "electronic_bass", "bass_drop": "electronic_bass", "uk_garage": "electronic_bass",
    "glasgow_bass": "electronic_bass", "london_garage": "electronic_bass", "london_grime": "electronic_bass",
    "london_dubstep": "electronic_bass", "london_jungle": "electronic_bass",
    # electronic_edm_pop
    "festival_edm": "electronic_edm_pop", "rave_cave": "electronic_edm_pop", "hyperpop": "electronic_edm_pop",
    "dance_pop": "electronic_edm_pop", "chiptune": "electronic_edm_pop",
    # electronic_chill
    "synthwave": "electronic_chill", "vaporwave": "electronic_chill", "downtempo": "electronic_chill",
    "lofi_beats": "electronic_chill", "glasgow_late": "electronic_chill", "london_triphop": "electronic_chill",
    # hiphop
    "swagger": "hiphop", "boom_bap": "hiphop", "conscious_flow": "hiphop", "g_funk": "hiphop",
    "trap_mode": "hiphop", "melbourne_hiphop": "hiphop",
    # soul_funk_rnb
    "neo_soul": "soul_funk_rnb", "motown_soul": "soul_funk_rnb", "after_hours_rnb": "soul_funk_rnb",
    "gospel": "soul_funk_rnb", "funk_disco": "soul_funk_rnb", "glasgow_soul": "soul_funk_rnb",
    "london_soul": "soul_funk_rnb", "melbourne_soul": "soul_funk_rnb",
    # jazz_lounge
    "acid_jazz": "jazz_lounge", "bebop": "jazz_lounge", "swing_bigband": "jazz_lounge",
    "smooth_jazz": "jazz_lounge", "jazz_dinner": "jazz_lounge", "dinner_party": "jazz_lounge",
    "london_jazz": "jazz_lounge",
    # folk_acoustic
    "folk_acoustic": "folk_acoustic", "campfire": "folk_acoustic", "celtic_folk": "folk_acoustic",
    "glasgow_folk": "folk_acoustic", "melbourne_folk": "folk_acoustic",
    # country
    "country_roads": "country", "outlaw_country": "country", "bluegrass": "country",
    # world_latin
    "latin_heat": "world_latin", "afrobeat": "world_latin", "bossa_samba": "world_latin",
    # reggae_ska
    "reggae_dub": "reggae_ska", "ska": "reggae_ska", "london_dub": "reggae_ska",
    # instrumental_cinematic
    "soundtracks": "instrumental_cinematic", "cinematic_epic": "instrumental_cinematic",
    "ambient_drift": "instrumental_cinematic", "post_rock": "instrumental_cinematic",
    "neoclassical": "instrumental_cinematic", "string_quartet": "instrumental_cinematic",
    "glasgow_postrock": "instrumental_cinematic",
    # happy_bright
    "happy": "happy_bright", "brunch_mix": "happy_bright", "cooking_mix": "happy_bright",
    "housework_hustle": "happy_bright", "gardening": "happy_bright", "walking_mix": "happy_bright",
    "fresh_start": "happy_bright", "weekend_mix": "happy_bright",
    # party_fun
    "party": "party_fun", "celebration": "party_fun", "pre_party": "party_fun", "party_throwback": "party_fun",
    "friday_night": "party_fun", "friday_feeling": "party_fun", "cookout": "party_fun",
    "game_night": "party_fun", "singalong": "party_fun", "treat_yourself": "party_fun",
    # euphoric_triumphant
    "euphoric": "euphoric_triumphant", "empowering": "euphoric_triumphant",
    "confidence_boost": "euphoric_triumphant", "triumphant": "euphoric_triumphant",
    "main_character": "euphoric_triumphant", "cathartic": "euphoric_triumphant",
    "monday_motivation": "euphoric_triumphant", "midweek_reset": "euphoric_triumphant",
    "hopeful": "euphoric_triumphant", "power_ballads": "euphoric_triumphant",
    # defiant_intense
    "defiant": "defiant_intense", "angst_mix": "defiant_intense", "restless": "defiant_intense",
    # melancholy_blue
    "melancholy": "melancholy_blue", "moody_mix": "melancholy_blue", "bittersweet": "melancholy_blue",
    "emotional": "melancholy_blue", "sad_bangers": "melancholy_blue", "sunday_scaries": "melancholy_blue",
    # heartbreak_longing
    "heartbreak": "heartbreak_longing", "yearning": "heartbreak_longing",
    "grief_release": "heartbreak_longing", "vulnerable": "heartbreak_longing",
    "moving_on": "heartbreak_longing",
    # nostalgic_throwback
    "nostalgia_mix": "nostalgic_throwback", "memory_lane": "nostalgic_throwback",
    "old_friends": "nostalgic_throwback", "school_days": "nostalgic_throwback",
    "throwback_anthems": "nostalgic_throwback",
    # romantic
    "situationship": "romantic", "romantic_mix": "romantic", "modern_romance": "romantic",
    "love_songs": "romantic", "slow_dance": "romantic", "first_date": "romantic", "date_night": "romantic",
    "late_night_romance": "romantic", "romantic_dinner": "romantic", "dinner": "romantic",
    "candlelight": "romantic", "crush": "romantic", "flirty": "romantic", "devotion": "romantic",
    "loved_up": "romantic", "slow_burn": "romantic", "long_distance": "romantic", "wedding_day": "romantic",
    "indie_romance": "romantic", "acoustic_romance": "romantic", "synthpop_romance": "romantic",
    "romantic_jazz": "romantic", "piano_romance": "romantic", "strings_romance": "romantic",
    # calm_unwind
    "chill": "calm_unwind", "lazy_sunday": "calm_unwind", "sunday_morning": "calm_unwind",
    "evening_unwind": "calm_unwind", "cool_down": "calm_unwind", "wind_down": "calm_unwind",
    "after_work": "calm_unwind", "serene": "calm_unwind", "tender": "calm_unwind",
    # dreamy_ethereal
    "dreamy_mix": "dreamy_ethereal", "daydreaming": "dreamy_ethereal", "awe_wonder": "dreamy_ethereal",
    # workout_energy
    "workout": "workout_energy", "running": "workout_energy", "gaming": "workout_energy",
    # focus_study
    "focus": "focus_study", "deep_work": "focus_study", "study_session": "focus_study",
    "deep_reading": "focus_study", "creative_flow": "focus_study",
    # wellness_sleep
    "meditation": "wellness_sleep", "spa_bath": "wellness_sleep", "yoga_stretch": "wellness_sleep",
    "sleep": "wellness_sleep", "power_nap": "wellness_sleep",
    # driving
    "driving_mix": "driving", "road_trip": "driving", "night_drive": "driving", "driving_singalong": "driving",
    "commute_mix": "driving",
    # weather
    "rainy_day": "weather", "sunny": "weather", "cosy": "weather", "beach_vibes": "weather",
    "stormy": "weather", "foggy": "weather", "snow_day": "weather", "heatwave": "weather", "frosty": "weather",
    "grey_skies": "weather", "windy": "weather", "clear_night": "weather", "overcast": "weather",
    # time_of_day
    "morning": "time_of_day", "sunrise": "time_of_day", "golden_hour": "time_of_day",
    "golden_afternoon": "time_of_day", "blue_hour": "time_of_day", "sunset_mix": "time_of_day",
    "after_dark": "time_of_day", "late_night": "time_of_day", "midnight": "time_of_day",
    "three_am": "time_of_day", "witching_hour": "time_of_day", "starlit": "time_of_day",
    # season_autumn
    "autumn_embers": "season_autumn", "autumn_jazz": "season_autumn", "autumn_leaves": "season_autumn",
    "autumn_rain": "season_autumn", "autumn_mix": "season_autumn",
    # season_winter
    "winter_cosy": "season_winter", "winter_frost": "season_winter", "winter_jazz": "season_winter",
    "winter_nights": "season_winter", "winter_mix": "season_winter",
    # season_spring
    "spring_acoustic": "season_spring", "spring_bloom": "season_spring", "spring_jangle": "season_spring",
    "spring_strings": "season_spring", "spring_mix": "season_spring",
    # season_summer
    "summer_breeze": "season_summer", "summer_heat": "season_summer", "summer_roadtrip": "season_summer",
    "summer_tropical": "season_summer", "summer_evening": "season_summer",
    # festive
    "festive": "festive",
    # era
    "decade_60s": "era", "decade_70s": "era", "decade_80s": "era", "decade_90s": "era", "decade_00s": "era",
    "decade_10s": "era", "decade_20s": "era",
    # geo_scene (3 pinned showcase scenes + 3 pinned geo HITS mixes — all share showcase behaviour)
    "scotland_scene": "geo_scene", "australia_scene": "geo_scene", "london_scene": "geo_scene",
    "scottish_hits": "geo_scene", "australian_hits": "geo_scene", "london_hits": "geo_scene",
    "uk_scene": "geo_scene", "uk_hits": "geo_scene", "scotland_now": "geo_scene",
    "london_now": "geo_scene", "uk_now": "geo_scene", "australia_now": "geo_scene",
}

# Mood tag signals per profile: (positive_substrings, negative_substrings).
# Substring matching against lowercased Plex/MusicBrainz mood tags from the Essentia cache.
# Tracks with matching positive tags get a distance boost; conflicting tags get a penalty.
# Profiles where emotional character is the key distinguishing feature benefit most.
_PROFILE_MOOD_SIGNALS = {
    # --- Meloday+ gap-fill mixes ---
    "situationship": (["yearning", "searching", "bittersweet", "wistful", "tense", "anxious"], ["euphoric", "carefree", "celebratory", "joyous"]),
    "sad_bangers": (["cathartic", "bittersweet", "melancholy", "energetic", "lively", "danceable"], []),
    "power_ballads": (["dramatic", "passionate", "theatrical", "rousing", "anthemic", "yearning"], ["aggressive", "boisterous", "frivolous"]),
    "restless": (["tense", "anxious", "urgent", "nervous", "searching", "brooding"], ["calm", "peaceful", "serene", "mellow"]),
    "neoclassical": (["peaceful", "elegant", "graceful", "reflective", "poignant", "serene"], ["aggressive", "energetic", "boisterous", "intense"]),
    "yacht_rock": (["smooth", "warm", "mellow", "sophisticated", "sunny", "easygoing"], ["aggressive", "intense", "angry", "abrasive"]),
    "swagger": (["swaggering", "brash", "confident", "stylish", "street-smart", "bravado", "gutsy"], ["sad", "melancholy", "tender", "vulnerable"]),
    "chart_pop": (["happy", "lively", "bright", "fun", "upbeat", "sunny", "exuberant"], ["aggressive", "brooding", "bleak"]),
    "dance_pop": (["energetic", "celebratory", "fun", "exuberant", "lively", "euphoric"], ["brooding", "bleak", "melancholy"]),
    "indie_pop": (["bright", "playful", "sweet", "quirky", "sparkling", "earnest"], ["aggressive", "menacing", "bleak"]),
    "synth_pop": (["stylish", "sparkling", "bright", "lively", "sophisticated", "nocturnal"], ["aggressive", "rustic", "raw"]),
    # rock / electronic / scores gap-fill
    "indie_rock": (["lively", "stylish", "energetic", "rousing", "wry", "earnest"], ["sad", "somber", "aggressive"]),
    "post_grunge": (["brooding", "gritty", "intense", "angst-ridden", "cathartic", "fierce"], ["calm", "peaceful", "cheerful"]),
    "rap_rock": (["aggressive", "brash", "intense", "swaggering", "rebellious", "visceral"], ["calm", "peaceful", "gentle"]),
    "festival_edm": (["euphoric", "exuberant", "uplifting", "sparkling", "ecstatic", "exciting"], ["sad", "melancholy", "gritty"]),
    "soundtracks": (["epic", "dramatic", "majestic", "atmospheric", "reflective", "cinematic"], ["aggressive", "silly", "trashy"]),
    "rave_cave": (["euphoric", "aggressive", "pounding", "relentless", "ecstatic", "frenzied"], ["calm", "gentle", "mellow", "sombre"]),
    # ---- 7 decade mixes (era) ----
    "decade_60s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_70s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_80s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_90s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_00s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_10s": (["rousing", "energetic", "stylish", "playful"], []),
    "decade_20s": (["rousing", "energetic", "stylish", "playful"], []),
    "stormy": (["dramatic", "ominous", "brooding", "volatile", "intense"], []),
    "foggy": (["soothing", "mysterious", "atmospheric", "eerie", "dreamy"], []),
    "snow_day": (["soothing", "gentle", "delicate", "playful"], []),
    "heatwave": (["languid", "sultry", "sparkling", "mellow", "dreamy"], []),
    "frosty": (["delicate", "austere", "refined", "gentle"], []),
    "grey_skies": (["melancholy", "reflective", "austere", "somber", "introspective"], []),
    "windy": (["nervous", "driving", "volatile", "energetic"], []),
    "clear_night": (["soothing", "nocturnal", "ethereal", "dreamy", "spacious"], []),
    "festive": (["joyous", "warm", "nostalgic", "celebratory", "cheerful"], []),
    "spring_bloom": (["bright", "joyous", "sunny", "lively"], []),
    "spring_acoustic": (["warm", "earthy", "gentle", "pastoral", "wistful"], []),
    "spring_strings": (["graceful", "uplifting", "elegant", "optimistic", "soothing"], []),
    "spring_jangle": (["wistful", "charming", "lively", "wry", "bright"], []),
    "summer_heat": (["lively", "exuberant", "sexy", "carefree", "euphoric"], []),
    "summer_breeze": (["mellow", "warm", "smooth", "easygoing", "sunny"], []),
    "summer_roadtrip": (["carefree", "exuberant", "anthemic", "rousing", "joyous"], []),
    "summer_tropical": (["warm", "sunny", "lively", "sensual", "carefree"], []),
    "autumn_leaves": (["warm", "nostalgic", "wistful", "rustic", "mellow"], []),
    "autumn_jazz": (["smooth", "warm", "sophisticated", "mellow", "sultry"], []),
    "autumn_rain": (["melancholy", "wistful", "reflective", "brooding", "somber"], []),
    "autumn_embers": (["warm", "rousing", "earthy", "gritty", "nostalgic"], []),
    "winter_frost": (["delicate", "ethereal", "atmospheric", "soothing", "austere"], []),
    "winter_cosy": (["warm", "intimate", "smooth", "tender", "sensual"], []),
    "winter_nights": (["nocturnal", "hypnotic", "atmospheric", "mellow", "dreamy"], []),
    "winter_jazz": (["smooth", "warm", "mellow", "sophisticated", "intimate"], []),
    "hopeful": (["optimistic", "uplifting", "bright", "innocent"], []),
    "yearning": (["yearning", "wistful", "tender", "plaintive"], []),
    "triumphant": (["triumphant", "rousing", "majestic", "anthemic", "epic"], []),
    "serene": (["soothing", "peaceful", "calm", "gentle"], []),
    "tender": (["tender", "gentle", "warm", "sentimental", "sweet"], []),
    "defiant": (["defiant", "fierce", "brash", "rebellious"], []),
    "vulnerable": (["vulnerable", "delicate", "intimate"], []),
    "awe_wonder": (["majestic", "ethereal", "epic", "spiritual", "reverent"], []),
    "grief_release": (["elegiac", "plaintive", "somber", "cathartic", "anguished"], []),
    "sunrise": (["optimistic", "gentle", "bright", "warm"], []),
    "blue_hour": (["wistful", "atmospheric", "reflective", "nocturnal", "poignant"], []),
    "midnight": (["nocturnal", "dark", "intimate", "hypnotic", "brooding"], []),
    "three_am": (["nocturnal", "lonely", "nervous", "hypnotic", "weary"], []),
    "golden_afternoon": (["warm", "mellow", "languid", "agreeable", "summery"], []),
    "overcast": (["melancholy", "reflective", "austere", "somber", "introspective"], []),
    "starlit": (["ethereal", "dreamy", "spacious", "atmospheric", "meditative"], []),
    "witching_hour": (["eerie", "mysterious", "ominous", "nocturnal"], []),
    "monday_motivation": (["rousing", "energetic", "confident"], []),
    "midweek_reset": (["rousing", "reflective", "driving", "cheerful"], []),
    "friday_feeling": (["exuberant", "lively", "carefree", "fun", "celebratory"], []),
    "sunday_scaries": (["nervous", "wistful", "bittersweet", "weary", "reflective"], []),
    "treat_yourself": (["stylish", "confident", "hedonistic", "sexy", "exuberant"], []),
    "dinner_party": (["sophisticated", "warm", "smooth", "stylish", "cosmopolitan"], []),
    "housework_hustle": (["lively", "exuberant", "playful", "carefree"], []),
    "study_session": (["cerebral", "calm", "reflective", "mellow", "hypnotic"], []),
    "wind_down": (["soothing", "calm", "gentle", "languid", "relaxed"], []),
    "yoga_stretch": (["soothing", "gentle", "meditative", "graceful", "spiritual"], []),
    "meditation": (["meditative", "soothing", "spiritual", "spacious", "devotional"], []),
    "deep_reading": (["cerebral", "calm", "atmospheric", "intimate", "reflective"], []),
    "creative_flow": (["freewheeling", "lively", "hypnotic", "playful", "kinetic"], []),
    "gaming": (["energetic", "intense", "driving", "kinetic", "exciting"], []),
    "gardening": (["warm", "pastoral", "sunny", "carefree", "earthy"], []),
    "spa_bath": (["soothing", "relaxed", "gentle", "delicate"], []),
    "power_nap": (["delicate", "languid", "dreamy", "atmospheric", "hypnotic"], []),
    "throwback_anthems": (["nostalgic", "exuberant", "celebratory", "fun", "rousing"], []),
    "old_friends": (["warm", "nostalgic", "joyous", "sentimental", "good-natured"], []),
    "campfire": (["warm", "rustic", "earthy", "gentle", "earnest"], []),
    "cookout": (["sunny", "carefree", "fun", "warm", "summery"], []),
    "game_night": (["playful", "fun", "lively", "witty", "exuberant"], []),
    "singalong": (["anthemic", "joyous", "rousing", "exuberant", "celebratory"], []),
    "school_days": (["nostalgic", "playful", "bittersweet", "lively", "carefree"], []),
    "memory_lane": (["nostalgic", "wistful", "sentimental", "tender", "reflective"], []),
    "crush": (["sweet", "playful", "optimistic", "gleeful", "tender"], []),
    "slow_burn": (["yearning", "tender", "intimate", "sensual", "sultry"], []),
    "moving_on": (["bittersweet", "optimistic", "rousing", "cathartic", "defiant"], []),
    "loved_up": (["ecstatic", "warm", "tender", "sentimental", "joyous"], []),
    "long_distance": (["yearning", "wistful", "tender", "poignant"], []),
    "flirty": (["playful", "sexy", "fun", "lively", "sensual"], []),
    "devotion": (["devotional", "tender", "earnest", "warm", "reverent"], []),
    "wedding_day": (["joyous", "tender", "celebratory", "romantic", "triumphant"], []),
    "funk_disco": (["rollicking", "lively", "stylish", "exuberant", "sexy"], ["sad", "melancholy", "aggressive"]),
    "neo_soul": (["sensual", "smooth", "warm", "sophisticated", "intimate", "mellow"], ["aggressive", "intense", "angry"]),
    "motown_soul": (["joyous", "lively", "warm", "celebratory", "sweet", "exuberant"], ["dark", "aggressive", "melancholy"]),
    "after_hours_rnb": (["sensual", "sultry", "nocturnal", "smooth", "intimate", "stylish"], ["aggressive", "intense", "cheerful"]),
    "acid_jazz": (["lively", "rollicking", "stylish", "cosmopolitan", "smooth"], ["aggressive", "sad", "angry"]),
    "boom_bap": (["swaggering", "confident", "stylish", "street-smart", "gritty"], ["calm", "peaceful", "gentle"]),
    "conscious_flow": (["thoughtful", "reflective", "stylish", "earnest", "literate", "smooth"], ["aggressive", "hostile", "angry"]),
    "g_funk": (["laid-back", "swaggering", "sunny", "confident", "stylish", "mellow"], ["aggressive", "intense", "sad"]),
    "trap_mode": (["dark", "aggressive", "brash", "menacing", "swaggering", "intense"], ["calm", "peaceful", "gentle"]),
    "lofi_beats": (["mellow", "hypnotic", "relaxed", "nostalgic", "soothing", "atmospheric"], ["aggressive", "intense", "angry"]),
    "house_party": (["lively", "exuberant", "stylish", "carefree", "euphoric", "sparkling"], ["sad", "melancholy", "aggressive"]),
    "deep_house": (["hypnotic", "stylish", "nocturnal", "smooth", "cosmopolitan", "atmospheric"], ["aggressive", "cheerful", "angry"]),
    "techno": (["hypnotic", "dark", "driving", "intense", "nocturnal"], ["calm", "peaceful", "cheerful"]),
    "trance": (["euphoric", "uplifting", "exuberant", "sparkling", "ecstatic", "exciting"], ["sad", "melancholy", "gritty"]),
    "dnb": (["energetic", "kinetic", "driving", "rousing", "exciting", "manic"], ["calm", "peaceful", "languid"]),
    "bass_drop": (["heavy", "dark", "intense", "menacing", "visceral", "aggressive"], ["calm", "peaceful", "gentle"]),
    "uk_garage": (["stylish", "lively", "swaggering", "sexy", "exuberant"], ["sad", "aggressive", "melancholy"]),
    "synthwave": (["nostalgic", "nocturnal", "stylish", "hypnotic", "atmospheric", "theatrical"], ["aggressive", "angry", "gritty"]),
    "industrial": (["aggressive", "harsh", "menacing", "cold", "intense", "mechanical"], ["calm", "peaceful", "warm"]),
    "vaporwave": (["hypnotic", "dreamy", "nostalgic", "trippy", "languid", "atmospheric"], ["aggressive", "intense", "angry"]),
    "downtempo": (["hypnotic", "mellow", "atmospheric", "nocturnal", "dreamy", "stylish"], ["aggressive", "intense", "angry"]),
    "hyperpop": (["manic", "exuberant", "playful", "brash", "ecstatic"], ["calm", "peaceful", "somber"]),
    "classic_rock": (["rousing", "swaggering", "brash", "confident", "exuberant", "gutsy"], ["calm", "peaceful", "gentle"]),
    "heavy_riffs": (["aggressive", "intense", "fierce", "brash", "visceral", "hostile"], ["calm", "peaceful", "gentle"]),
    "punk_energy": (["aggressive", "rebellious", "brash", "raucous", "defiant", "fierce"], ["calm", "peaceful", "gentle"]),
    "garage_grunge": (["gritty", "brash", "rebellious", "raw", "fierce", "angst-ridden"], ["calm", "peaceful", "elegant"]),
    "emo_poppunk": (["angst-ridden", "cathartic", "earnest", "yearning", "fierce", "anguished"], ["calm", "peaceful", "stately"]),
    "britpop_rock": (["swaggering", "lively", "stylish", "wry", "exuberant", "confident"], ["sad", "aggressive", "somber"]),
    "blues_bar": (["gritty", "earthy", "passionate", "gutsy", "sultry", "raw"], ["aggressive", "cheerful", "euphoric"]),
    "psych_haze": (["hypnotic", "dreamy", "trippy", "atmospheric", "druggy", "spacey"], ["aggressive", "intense", "angry"]),
    "prog_rock": (["complex", "elaborate", "cerebral", "epic", "sprawling", "dramatic"], ["aggressive", "naive", "angry"]),
    "stoner_rock": (["heavy", "hypnotic", "gritty", "brooding", "druggy"], ["calm", "peaceful", "cheerful"]),
    "reggae_dub": (["warm", "laid-back", "mellow", "spiritual", "sunny", "hypnotic"], ["aggressive", "intense", "angry"]),
    "afrobeat": (["warm", "lively", "exuberant", "spiritual", "celebratory"], ["sad", "aggressive", "cold"]),
    "latin_heat": (["warm", "lively", "sexy", "exuberant", "celebratory", "fiery"], ["sad", "melancholy", "cold"]),
    "bossa_samba": (["warm", "smooth", "sophisticated", "sensual", "mellow", "graceful"], ["aggressive", "intense", "harsh"]),
    "celtic_folk": (["rousing", "earthy", "nostalgic", "warm", "pastoral"], ["aggressive", "cold", "menacing"]),
    "ska": (["lively", "exuberant", "playful", "rousing", "carefree", "brassy"], ["sad", "somber", "menacing"]),
    "bebop": (["energetic", "elaborate", "kinetic", "complex", "exciting", "cerebral"], ["calm", "languid", "somber"]),
    "swing_bigband": (["lively", "exuberant", "celebratory", "brassy", "playful", "swinging"], ["sad", "aggressive", "somber"]),
    "smooth_jazz": (["smooth", "mellow", "sophisticated", "relaxed", "elegant", "stylish"], ["aggressive", "intense", "angry"]),
    "country_roads": (["warm", "earnest", "cheerful", "nostalgic"], ["aggressive", "dark", "menacing"]),
    "outlaw_country": (["gritty", "earthy", "earnest", "rebellious", "weary", "rustic"], ["cheerful", "euphoric", "slick"]),
    "bluegrass": (["lively", "rousing", "earthy", "pastoral", "rustic"], ["dark", "aggressive", "menacing"]),
    "rockabilly_surf": (["lively", "playful", "rousing", "nostalgic", "exuberant", "rustic"], ["sad", "somber", "menacing"]),
    "cinematic_epic": (["epic", "dramatic", "majestic", "monumental", "theatrical"], ["aggressive", "silly", "trashy"]),
    "ambient_drift": (["atmospheric", "meditative", "ethereal", "soothing", "spacious", "hypnotic"], ["aggressive", "energetic", "brash"]),
    "post_rock": (["atmospheric", "majestic", "cathartic", "epic", "brooding", "dramatic"], ["aggressive", "silly", "trashy"]),
    "chiptune": (["playful", "lively", "nostalgic", "exuberant", "quirky", "bright"], ["sad", "somber", "menacing"]),
    "gospel": (["spiritual", "joyous", "uplifting", "reverent", "exuberant", "devotional"], ["aggressive", "dark", "menacing"]),
    "glasgow_folk": (["earnest", "nostalgic", "warm", "pastoral", "wistful", "gentle"], ["aggressive", "intense", "brash"]),
    "glasgow_dream": (["dreamy", "hypnotic", "atmospheric", "wistful", "ethereal", "mellow"], ["aggressive", "intense", "angry"]),
    "glasgow_indie": (["wry", "wistful", "witty", "stylish", "literate", "charming"], ["aggressive", "menacing", "hostile"]),
    "glasgow_soul": (["warm", "stylish", "smooth", "sophisticated", "lively", "passionate"], ["aggressive", "dark", "menacing"]),
    "glasgow_postrock": (["atmospheric", "majestic", "cathartic", "brooding", "epic", "dramatic"], ["aggressive", "silly", "cheerful"]),
    "glasgow_anthems": (["swaggering", "exuberant", "lively", "confident", "rousing", "stylish"], ["sad", "somber", "calm"]),
    "glasgow_synth": (["stylish", "nocturnal", "hypnotic", "atmospheric", "yearning"], ["aggressive", "gritty", "harsh"]),
    "glasgow_postpunk": (["angular", "rebellious", "brash", "nervous", "defiant"], ["calm", "peaceful", "warm"]),
    "glasgow_house": (["hypnotic", "stylish", "lively", "cosmopolitan", "nocturnal"], ["aggressive", "sad", "angry"]),
    "glasgow_underground": (["hypnotic", "dark", "driving", "nocturnal", "intense"], ["calm", "peaceful", "cheerful"]),
    "glasgow_bass": (["kinetic", "playful", "quirky", "exuberant", "bright"], ["calm", "somber", "languid"]),
    "glasgow_late": (["hypnotic", "nocturnal", "mellow", "atmospheric", "stylish", "dreamy"], ["aggressive", "intense", "cheerful"]),
    "london_dub": (["warm", "laid-back", "hypnotic", "spiritual", "mellow", "sensual"], ["aggressive", "intense", "cold"]),
    "london_soul": (["passionate", "sultry", "stylish", "smooth", "sophisticated", "warm"], ["aggressive", "intense", "menacing"]),
    "london_jazz": (["lively", "spiritual", "cosmopolitan", "cerebral", "warm"], ["aggressive", "sad", "cold"]),
    "london_triphop": (["nocturnal", "hypnotic", "brooding", "atmospheric", "theatrical", "stylish"], ["aggressive", "cheerful", "euphoric"]),
    "london_mod": (["lively", "acerbic", "stylish", "rousing", "exuberant", "brassy"], ["sad", "somber", "menacing"]),
    "london_britpop": (["swaggering", "witty", "lively", "stylish", "wry", "confident"], ["sad", "aggressive", "somber"]),
    "london_indie": (["angular", "lively", "brash", "nervous", "stylish"], ["calm", "somber", "gentle"]),
    "london_calling": (["rebellious", "brash", "defiant", "raucous", "gritty", "fierce"], ["calm", "peaceful", "elegant"]),
    "london_garage": (["stylish", "lively", "swaggering", "sexy", "exuberant"], ["sad", "aggressive", "somber"]),
    "london_grime": (["aggressive", "brash", "menacing", "gritty", "defiant", "intense"], ["calm", "peaceful", "gentle"]),
    "london_dubstep": (["heavy", "dark", "menacing", "hypnotic", "intense", "visceral"], ["calm", "peaceful", "cheerful"]),
    "london_jungle": (["kinetic", "manic", "driving", "rousing", "exciting", "raucous"], ["calm", "peaceful", "languid"]),
    "melbourne_folk": (["warm", "earnest", "wistful", "pastoral", "gentle", "nostalgic"], ["aggressive", "intense", "brash"]),
    "melbourne_dream": (["dreamy", "hypnotic", "atmospheric", "wistful", "mellow", "ethereal"], ["aggressive", "intense", "angry"]),
    "melbourne_soul": (["passionate", "stylish", "warm", "lively", "sophisticated", "sensual"], ["aggressive", "dark", "menacing"]),
    "melbourne_sunset": (["warm", "carefree", "sunny", "summery", "mellow", "playful"], ["aggressive", "dark", "menacing"]),
    "melbourne_indie": (["wry", "witty", "laid-back", "stylish", "charming", "lively"], ["aggressive", "menacing", "somber"]),
    "melbourne_pubrock": (["rousing", "gutsy", "swaggering", "brash", "raucous", "gritty"], ["calm", "peaceful", "elegant"]),
    "melbourne_hiphop": (["laid-back", "earnest", "witty", "street-smart", "warm", "stylish"], ["aggressive", "hostile", "cold"]),
    "melbourne_postpunk": (["dark", "angular", "brooding", "menacing", "intense", "nervous"], ["calm", "cheerful", "warm"]),
    "melbourne_psych": (["hypnotic", "trippy", "manic", "driving", "druggy", "kinetic"], ["calm", "peaceful", "gentle"]),
    "melbourne_garagepunk": (["aggressive", "raucous", "rebellious", "brash", "fierce", "gritty"], ["calm", "peaceful", "elegant"]),
    "melbourne_club": (["exuberant", "lively", "euphoric", "driving", "bright", "energetic"], ["sad", "somber", "calm"]),
    "melbourne_techno": (["hypnotic", "dark", "driving", "nocturnal", "intense"], ["calm", "peaceful", "cheerful"]),
    "melancholy":  (["sad", "melancholy", "bittersweet", "somber", "plaintive", "elegiac",
                     "lonely", "gloomy", "poignant"],
                   ["happy", "euphoric", "lively", "energetic", "fun"]),
    "happy":       (["happy", "lively", "euphoric", "joyous", "cheerful", "jovial",
                     "carefree", "fun", "positive"],
                   ["sad", "melancholy", "dark", "gloomy"]),
    "workout":     (["energetic", "aggressive", "powerful", "intense", "triumphant",
                     "rousing", "athletic"],
                   ["calm", "peaceful", "sad", "melancholy"]),
    "running":     (["energetic", "powerful", "lively", "rousing", "driving"],
                   ["calm", "peaceful"]),
    "party":       (["celebratory", "energetic", "euphoric", "fun", "carefree", "boisterous"],
                   ["sad", "calm", "melancholy"]),
    "focus":       (["calm", "peaceful", "meditative", "relaxed", "atmospheric", "cerebral"],
                   ["aggressive", "intense", "boisterous"]),
    "chill":       (["calm", "laid-back", "mellow", "relaxed", "peaceful", "easygoing"],
                   ["aggressive", "intense", "angry"]),
    "sleep":       (["calm", "peaceful", "dreamy", "soothing", "languid", "meditative"],
                   ["energetic", "aggressive", "lively"]),
    "late_night":  (["dark", "introspective", "brooding", "atmospheric", "hypnotic", "nocturnal"],
                   ["happy", "cheerful", "fun"]),
    "morning":     (["cheerful", "calm", "peaceful", "gentle", "springlike", "bright"],
                   ["dark", "aggressive", "sad"]),
    "dinner":      (["romantic", "sophisticated", "elegant", "smooth", "mellow"],
                   ["aggressive", "intense", "angry"]),
    "rainy_day":        (["melancholy", "bittersweet", "introspective", "nostalgic", "wistful", "plaintive", "poignant", "somber", "reflective"],
                        ["lively", "energetic", "euphoric"]),
    "sunny":            (["happy", "lively", "carefree", "fun", "joyous", "cheerful", "summery"],
                        ["sad", "dark", "melancholy"]),
    "cosy":             (["calm", "peaceful", "warm", "reassuring", "nostalgic", "gentle"],
                        ["aggressive", "intense", "energetic"]),
    # New profiles
    "nostalgia_mix":    (["nostalgic", "wistful", "bittersweet", "sentimental", "autumnal"],
                        ["aggressive", "energetic", "euphoric"]),
    "dreamy_mix":       (["dreamy", "ethereal", "atmospheric", "spacey", "trippy"],
                        ["aggressive", "intense", "angry"]),
    "moody_mix":        (["dark", "introspective", "brooding", "melancholy", "gloomy"],
                        ["happy", "lively", "fun", "euphoric"]),
    "emotional":        (["passionate", "poignant", "bittersweet", "intense", "powerful", "cathartic"],
                        []),
    "bittersweet":      (["bittersweet", "nostalgic", "wistful", "sad", "melancholy", "poignant"],
                        ["lively", "energetic", "euphoric"]),
    "cathartic":        (["intense", "powerful", "dramatic", "triumphant", "passionate", "cathartic", "anguished", "visceral", "fiery"],
                        ["calm", "peaceful", "soothing"]),
    "confidence_boost": (["confident", "powerful", "lively", "gutsy", "swaggering", "bravado", "stylish"],
                        ["sad", "melancholy", "dark"]),
    "empowering":       (["triumphant", "powerful", "rousing", "anthemic", "mighty", "heroic"],
                        ["sad", "melancholy", "dark"]),
    "angst_mix":        (["angry", "aggressive", "intense", "nervous", "rebellious"],
                        ["calm", "peaceful", "happy", "cheerful"]),
    "romantic_mix":     (["romantic", "passionate", "tender", "intimate", "sensual"],
                        ["aggressive", "angry", "intense"]),
    "daydreaming":      (["dreamy", "atmospheric", "calm", "peaceful", "ethereal"],
                        ["energetic", "aggressive", "intense"]),
    "fresh_start":      (["optimistic", "idealistic", "cheerful", "bright", "springlike"],
                        ["dark", "aggressive", "sad"]),
    "euphoric":         (["euphoric", "ecstatic", "lively", "carefree", "thrilling"],
                        ["sad", "dark", "melancholy"]),
    "beach_vibes":      (["happy", "carefree", "summery", "relaxed", "fun", "sunny"],
                        ["sad", "dark", "aggressive"]),
    "autumn_mix":       (["nostalgic", "wistful", "melancholy", "introspective", "bittersweet", "autumnal"],
                        ["lively", "euphoric", "energetic"]),
    "winter_mix":       (["calm", "peaceful", "melancholy", "atmospheric", "nostalgic", "wintry"],
                        ["energetic", "aggressive", "lively"]),
    "spring_mix":       (["optimistic", "cheerful", "springlike", "bright", "sunny"],
                        ["dark", "sad", "aggressive"]),
    "love_songs":       (["romantic", "passionate", "tender", "poignant", "earnest"],
                        ["aggressive", "angry"]),
    "slow_dance":       (["romantic", "tender", "intimate", "sensual", "passionate"],
                        ["aggressive", "energetic", "intense"]),
    "candlelight":      (["romantic", "elegant", "peaceful", "gentle", "intimate"],
                        ["aggressive", "intense", "energetic"]),
    "acoustic_romance": (["romantic", "earnest", "tender", "organic", "intimate"],
                        ["aggressive", "angry"]),
    "indie_romance":    (["romantic", "dreamy", "melancholy", "bittersweet", "wistful"],
                        ["aggressive", "intense"]),
    "modern_romance":   (["romantic", "passionate", "tender", "poignant", "yearning"],
                        ["aggressive", "angry", "boisterous", "energetic"]),
    "late_night_romance":(["romantic", "intimate", "sensual", "yearning", "dark"],
                        ["energetic", "lively", "fun", "boisterous"]),
    "romantic_dinner":  (["romantic", "elegant", "sophisticated", "smooth", "intimate"],
                        ["aggressive", "intense", "energetic", "boisterous"]),
    "first_date":       (["romantic", "optimistic", "sweet", "tender", "gentle", "poignant"],
                        ["aggressive", "intense", "angry"]),
    "romantic_jazz":    (["romantic", "sophisticated", "smooth", "elegant", "intimate"],
                        ["aggressive", "intense", "energetic"]),
    "jazz_dinner":      (["sophisticated", "elegant", "smooth", "mellow", "relaxed"],
                        ["aggressive", "intense", "energetic", "angry"]),
    "string_quartet":   (["elegant", "graceful", "sophisticated", "refined", "stately"],
                        ["aggressive", "intense", "energetic", "angry"]),
    "strings_romance":  (["romantic", "elegant", "graceful", "majestic", "lush"],
                        ["aggressive", "intense", "energetic"]),
    "piano_romance":    (["romantic", "peaceful", "tender", "intimate", "gentle", "poignant"],
                        ["aggressive", "intense", "energetic", "angry"]),
    "synthpop_romance": (["romantic", "dreamy", "nostalgic", "yearning", "poignant"],
                        ["aggressive", "angry", "intense"]),
    "party_throwback":  (["celebratory", "energetic", "fun", "carefree", "boisterous"],
                        ["sad", "calm", "melancholy"]),
    "heartbreak":       (["sad", "yearning", "lonely", "anguished", "tragic", "bleak", "desperate"],
                        ["happy", "lively", "euphoric", "carefree"]),
    "pre_party":        (["energetic", "fun", "carefree", "lively", "celebratory", "boisterous"],
                        ["sad", "melancholy", "calm", "peaceful"]),
    "celebration":      (["celebratory", "triumphant", "happy", "euphoric", "lively", "joyous"],
                        ["sad", "melancholy", "dark"]),
    "folk_acoustic":    (["earthy", "organic", "rustic", "pastoral", "earnest"],
                        ["aggressive", "intense", "brash"]),
    "deep_work":        (["calm", "atmospheric", "meditative", "peaceful", "soothing", "cerebral"],
                        ["aggressive", "intense", "boisterous", "energetic"]),
    "evening_unwind":   (["calm", "peaceful", "mellow", "relaxed", "smooth", "easygoing"],
                        ["aggressive", "intense", "energetic"]),
    "cool_down":        (["calm", "relaxed", "peaceful", "mellow", "gentle"],
                        ["aggressive", "intense", "angry"]),
    "cooking_mix":      (["lively", "happy", "carefree", "fun", "cheerful"],
                        ["sad", "intense", "aggressive", "melancholy"]),
    # Time-of-day / lifestyle
    "after_work":       (["lively", "relaxed", "mellow", "laid-back", "easygoing"],
                        ["aggressive", "intense", "angry"]),
    "after_dark":       (["dark", "brooding", "atmospheric", "nocturnal", "sensual", "hypnotic"],
                        ["happy", "cheerful", "fun"]),
    "brunch_mix":       (["happy", "cheerful", "lively", "carefree", "fun"],
                        ["sad", "dark", "aggressive", "melancholy"]),
    "sunday_morning":   (["calm", "peaceful", "gentle", "warm", "relaxed", "soothing"],
                        ["aggressive", "energetic", "intense"]),
    "lazy_sunday":      (["calm", "peaceful", "relaxed", "laid-back", "easygoing", "mellow"],
                        ["aggressive", "energetic", "intense"]),
    "golden_hour":      (["nostalgic", "wistful", "warm", "optimistic", "summery"],
                        ["aggressive", "angry", "intense"]),
    "sunset_mix":       (["nostalgic", "wistful", "bittersweet", "mellow", "introspective"],
                        ["aggressive", "lively", "energetic"]),
    "summer_evening":   (["warm", "carefree", "relaxed", "summery", "mellow"],
                        ["aggressive", "intense", "dark"]),
    "friday_night":     (["fun", "energetic", "carefree", "lively", "celebratory", "boisterous"],
                        ["sad", "melancholy", "calm"]),
    "weekend_mix":      (["lively", "happy", "carefree", "relaxed", "fun", "cheerful"],
                        ["aggressive", "angry", "sad"]),
    "date_night":       (["romantic", "intimate", "smooth", "elegant", "tender"],
                        ["aggressive", "intense", "energetic"]),
    # Activity / driving
    "driving_mix":      (["energetic", "lively", "powerful", "confident", "driving"],
                        ["calm", "peaceful", "melancholy"]),
    "night_drive":      (["dark", "brooding", "atmospheric", "introspective", "nocturnal"],
                        ["happy", "lively", "fun", "cheerful"]),
    "driving_singalong":(["lively", "fun", "energetic", "carefree", "cheerful"],
                        ["sad", "dark", "melancholy"]),
    "road_trip":        (["lively", "energetic", "carefree", "fun", "optimistic"],
                        ["sad", "melancholy", "dark"]),
    "commute_mix":      (["lively", "mellow", "laid-back", "easygoing", "relaxed"],
                        ["aggressive", "angry"]),
    "walking_mix":      (["lively", "energetic", "carefree", "optimistic", "fun"],
                        ["aggressive", "intense", "angry"]),
    # Aesthetic
    "main_character":   (["confident", "dramatic", "powerful", "swaggering", "triumphant", "theatrical"],
                        ["sad", "melancholy", "dark", "gloomy"]),
}


def _mood_tag_boost(entry, profile_key):
    """
    Distance adjustment from mood tag compatibility.
    Negative = boost (track emotionally fits this profile).
    Positive = penalty (track emotionally conflicts with this profile).
    Returns 0.0 when no mood data is available (neutral — acoustic features take over).
    """
    signals = _PROFILE_MOOD_SIGNALS.get(profile_key)
    if not signals:
        return 0.0
    positive_subs, negative_subs = signals
    track_moods = [m.lower() for m in (entry.get("moods") or [])]
    if not track_moods:
        return 0.0

    def _matches(subs):
        return any(sub in mood for mood in track_moods for sub in subs)

    boost = 0.0
    if _matches(positive_subs):
        boost -= 0.12  # meaningful push toward tracks that feel emotionally right
    if _matches(negative_subs):
        boost += 0.10  # discourage tracks that feel emotionally wrong
    return boost


# Style/genre signals per profile: (positive_substrings, negative_substrings), matched as
# substrings against the AllMusic styles in entry["styles"] (+ entry["genres"]).
# For STYLE-DEFINED profiles (a genre *is* the identity — the synth-pop / folk / jazz /
# classical / indie mixes) the positives are REQUIRED: _build_mix_tracks keeps only tracks
# carrying one of them, so the mix is genre-pure. The rest (focus/deep_work) are a soft nudge.
_PROFILE_STYLE_SIGNALS = {
    # --- Meloday+ gap-fill mixes (power_ballads is a SOFT nudge — not in _STYLE_DEFINED_PROFILES;
    # the others are hard-gated. Pop lists carry BOTH Plex + genre_discogs spellings) ---
    "power_ballads": (["arena rock", "album rock", "soft rock", "adult contemporary", "hard rock", "contemporary pop/rock"], ["rap", "techno", "drill"]),
    "neoclassical": (["neo-classical", "modern composition", "chamber music", "classical crossover", "contemporary instrumental", "classical"], ["film score", "original score", "soundtracks", "rap", "metal", "punk"]),
    "yacht_rock": (["soft rock", "adult contemporary", "sophisti-pop", "blue-eyed soul", "pop-soul", "quiet storm"], ["metal", "rap", "punk", "techno", "drill", "hardcore"]),  # WHY: dropped "am pop" — substring ⊂ "dream pop" (2845-track leak) (audit substr-fix)
    "swagger": (["contemporary rap", "hardcore rap", "contemporary r&b", "g-funk", "funk", "west coast rap"], ["folk", "ambient", "classical", "metal", "country"]),
    "chart_pop": (["contemporary pop/rock", "dance-pop", "teen pop", "vocal pop", "traditional pop", "pop idol", "social media pop", "europop", "euro-pop", "power pop", "bubblegum", "sunshine pop", "brill building pop", "pop-soul"], ["punk", "metal", "experimental", "country"]),
    "dance_pop": (["dance-pop", "dance-rock", "alternative dance", "euro-dance", "eurodance", "hi-nrg", "nu-disco", "italo-disco", "euro-disco", "post-disco", "electroclash", "eurobeat"], ["metal", "folk", "ambient", "country"]),
    "indie_pop": (["indie pop", "left-field pop", "jangle pop", "twee pop", "chamber pop", "baroque pop", "noise pop", "psychedelic pop", "sunshine pop", "bedroom pop", "sophisti-pop", "c-86", "indie electronic"], ["metal", "rap", "techno", "hardcore"]),
    "synth_pop": (["synth pop", "synth-pop", "synthwave", "new wave", "new romantic", "neo-electro", "electroclash", "sophisti-pop", "indie electronic"], ["metal", "country", "folk", "gospel"]),
    # rock / electronic / scores gap-fill (Plex + genre_discogs spellings)
    "indie_rock": (["alternative/indie rock", "indie rock", "college rock", "jangle pop", "garage rock revival", "modern rock"], ["rap", "drill", "techno", "edm", "gospel"]),
    "post_grunge": (["post-grunge", "post grunge"], ["rap", "drill", "techno", "edm", "gospel"]),
    "rap_rock": (["rap-rock", "rap rock", "rap-metal", "rap metal", "nu metal", "nü metal", "funk metal", "rapcore"], ["ambient", "folk", "gospel", "classical"]),
    "festival_edm": (["edm", "big room", "electro house", "future bass", "complextro"], ["metal", "country", "folk", "ambient", "gospel"]),
    "soundtracks": (["soundtrack", "original score", "film score", "tv soundtrack", "film music", "movie theme", "video game music", "soundtracks", "orchestral", "modern composition"], ["rap", "punk", "drill"]),
    "rave_cave": (["donk", "hard house", "hard trance", "hardstyle", "hard techno", "schranz", "gabber", "happy hardcore", "jumpstyle", "makina"], ["ambient", "folk", "classical", "gospel", "rap", "k-pop"]),
    "festive": (["christmas", "holidays"], []),
    "spring_acoustic": (["folk", "singer/songwriter", "americana", "indie folk"], []),
    "spring_strings": (["classical", "modern composition", "chamber", "orchestral"], []),
    "spring_jangle": (["indie pop", "jangle pop", "dream pop", "twee pop"], []),
    "summer_heat": (["disco", "funk", "club/dance"], []),  # dropped "house" (22k swamped the disco/funk intent)
    "summer_breeze": (["soft rock", "adult contemporary", "sophisti-pop"], []),  # WHY: dropped "am pop" (⊂ "dream pop"); +sophisti-pop keeps breadth (audit substr-fix)
    "summer_tropical": (["latin", "reggae", "afro", "tropical", "bossa"], []),
    "autumn_leaves": (["folk", "singer/songwriter", "americana"], []),
    "autumn_jazz": (["jazz", "soul", "smooth jazz"], []),
    "autumn_embers": (["classic rock", "blues rock", "southern rock", "arena rock", "aor"], []),  # Discogs subgenres
    "winter_frost": (["ambient", "modern composition", "classical", "experimental ambient"], []),
    "winter_cosy": (["soul", "neo-soul", "quiet storm", "smooth soul"], []),
    "winter_nights": (["downtempo", "ambient techno", "electronica", "trip-hop"], []),
    "winter_jazz": (["smooth jazz", "jazz", "lounge", "vocal jazz"], []),
    "dinner_party": (["vocal jazz", "soul", "lounge", "bossa nova", "smooth"], []),
    "study_session": (["instrumental", "classical", "ambient", "post-rock", "score"], []),
    "yoga_stretch": (["ambient", "new age", "instrumental", "classical", "downtempo"], []),
    "meditation": (["ambient", "new age", "modern composition", "experimental ambient"], []),
    "deep_reading": (["instrumental", "classical", "ambient", "post-rock", "score"], []),
    "gaming": (["electronic", "big beat", "drum", "synthwave", "hard rock"], []),
    "spa_bath": (["ambient", "new age", "instrumental", "downtempo", "classical"], []),
    "power_nap": (["ambient", "new age", "modern composition", "downtempo"], []),
    "campfire": (["folk", "singer/songwriter", "americana", "indie folk", "acoustic"], []),
    "cookout": (["soul", "funk", "reggae", "r&b"], []),
    "singalong": (["album rock", "arena rock", "power pop"], []),
    "funk_disco": (["funk", "disco", "funky breaks", "neo-disco", "boogie", "euro-disco", "post-disco"], ["metal", "rap", "ambient"]),
    "neo_soul": (["neo soul", "contemporary r&b", "quiet storm"], ["metal", "punk", "edm"]),  # Discogs subgenres
    "motown_soul": (["soul", "rhythm & blues", "funk", "disco"], ["metal", "rap", "techno"]),  # Discogs subgenres (classic soul/funk)
    "after_hours_rnb": (["contemporary r&b", "alternative r&b", "new jack swing", "quiet storm"], ["metal", "punk", "country"]),
    "acid_jazz": (["acid jazz", "jazz-funk", "soul jazz", "jazz-house", "fusion", "clubjazz"], ["metal", "screamo", "drill"]),
    "boom_bap": (["boom bap", "hardcore hip-hop", "jazzy hip-hop", "conscious"], ["metal", "country", "ambient"]),  # Discogs subgenres
    "conscious_flow": (["conscious", "jazzy hip-hop", "instrumental", "boom bap"], ["metal", "screamo", "edm"]),  # Discogs subgenres
    "g_funk": (["g-funk", "gangsta"], ["metal", "punk", "ambient"]),  # Discogs subgenres
    "trap_mode": (["trap", "cloud rap", "crunk", "gangsta"], ["folk", "ambient", "classical"]),  # Discogs subgenres
    "lofi_beats": (["instrumental hip-hop", "lo-fi", "trip-hop", "downbeat"], ["metal", "punk", "hardcore"]),  # dropped "downtempo" (pulled 13k generic downtempo, not lo-fi beats)
    "house_party": (["house", "tech-house", "progressive house", "club/dance", "euro-dance"], ["metal", "country", "ambient"]),
    "deep_house": (["deep house", "microhouse", "minimal techno", "tech-house", "left-field house"], ["metal", "punk", "country"]),
    "techno": (["techno", "minimal techno", "detroit techno", "acid house", "industrial dance"], ["folk", "country", "gospel"]),
    "trance": (["trance", "progressive trance", "goa trance", "euro-dance", "hi-nrg"], ["metal", "country", "blues"]),
    "dnb": (["jungle", "drum'n'bass", "breakbeat", "idm", "bass music"], ["folk", "country", "ambient"]),  # WHY: split the "jungle/drum'n'bass" composite — word-boundary matching treats them as two separate positives (audit substr-fix-systemic)
    "bass_drop": (["dubstep", "bass music", "grime", "trap (edm)"], ["folk", "country", "jazz"]),
    "uk_garage": (["uk garage", "garage", "bass music", "broken beat", "bassline"], ["metal", "country", "folk"]),
    "synthwave": (["synthwave", "neo-electro", "new romantic"], ["metal", "country", "gospel"]),  # dropped bare "electro" (substring of the "electronic" parent → pulled the whole electronic pool)
    "industrial": (["industrial", "electro-industrial", "industrial metal", "industrial dance"], ["folk", "country", "gospel"]),
    "vaporwave": (["vaporwave", "chillwave", "ambient pop", "plunderphonics"], ["metal", "punk", "hardcore"]),
    "downtempo": (["downtempo", "trip-hop", "chillwave", "idm", "ambient techno"], ["metal", "punk", "hardcore"]),
    "hyperpop": (["hyperpop", "glitch", "bubblegum", "social media pop"], ["metal", "blues", "country"]),
    "classic_rock": (["album rock", "arena rock", "hard rock", "blues-rock", "southern rock", "american trad rock"], ["techno", "rap", "edm"]),
    "heavy_riffs": (["heavy metal", "hard rock", "alternative metal", "nü metal", "funk metal", "metalcore"], ["ambient", "folk", "gospel"]),
    "punk_energy": (["pop punk", "punk revival", "hardcore punk", "skatepunk", "punk/new wave"], ["ambient", "jazz", "gospel"]),
    "garage_grunge": (["grunge", "garage rock revival", "garage punk", "proto-punk", "noise-rock"], ["ambient", "gospel", "classical"]),
    "emo_poppunk": (["emo", "emo-pop", "pop punk", "post-hardcore", "screamo"], ["ambient", "jazz", "classical"]),
    "britpop_rock": (["brit pop"], ["metal", "techno", "gospel"]),  # Discogs subgenre
    "blues_bar": (["blues-rock", "electric blues", "chicago blues", "regional blues", "punk blues"], ["edm", "techno", "gospel"]),
    "psych_haze": (["neo-psychedelia", "shoegaze", "space rock", "dream pop", "kraut rock"], ["gospel", "country", "rap"]),
    "prog_rock": (["progressive rock", "prog rock"], ["rap", "country", "gospel"]),  # Last.fm community tags (Discogs can't name prog)
    "stoner_rock": (["stoner metal", "doom metal", "acid rock", "space rock"], ["gospel", "folk", "jazz"]),
    "reggae_dub": (["roots reggae", "dub", "dancehall", "ska", "contemporary reggae", "reggae-pop"], ["metal", "techno", "screamo"]),
    "afrobeat": (["afrobeat", "highlife", "african", "afro-cuban", "soukous"], ["metal", "techno", "screamo"]),  # Discogs subgenres (library-thin)
    "latin_heat": (["latin pop", "salsa", "cumbia", "reggaeton", "latin dance", "tropical"], ["metal", "ambient", "screamo"]),
    "bossa_samba": (["bossa", "samba", "latin jazz", "mpb"], ["metal", "techno", "screamo"]),  # Discogs subgenres (bossa matches bossa nova/bossanova)
    "celtic_folk": (["celtic", "celtic rock", "celtic fusion", "british folk", "traditional celtic"], ["techno", "rap", "metal"]),
    "ska": (["ska", "ska-punk", "third wave ska revival", "ska revival"], ["ambient", "techno", "drill"]),
    "bebop": (["hard bop", "bop", "post-bop", "avant-garde jazz"], ["edm", "metal", "gospel"]),
    "swing_bigband": (["swing", "big band", "swing", "retro swing", "traditional pop"], ["metal", "techno", "screamo"]),
    "smooth_jazz": (["smooth jazz", "crossover jazz", "lounge", "cool", "quiet storm"], ["metal", "punk", "drill"]),
    "country_roads": (["country", "honky tonk", "country rock"], ["techno", "metal", "drill"]),  # Discogs subgenres
    "outlaw_country": (["honky tonk", "bluegrass", "country rock", "country"], ["techno", "edm", "gospel"]),  # Discogs subgenres
    "bluegrass": (["bluegrass", "progressive bluegrass", "country-folk", "string bands", "new acoustic"], ["techno", "metal", "drill"]),
    "rockabilly_surf": (["rockabilly", "surf", "rockabilly revival", "psychobilly", "rock & roll"], ["techno", "drill", "gospel"]),
    "cinematic_epic": (["soundtrack", "score", "neo-romantic"], ["rap", "punk", "drill"]),  # Discogs subgenres
    "ambient_drift": (["ambient", "dark ambient", "new age", "experimental ambient"], ["rap", "punk", "metal"]),
    "post_rock": (["post rock", "math rock"], ["rap", "drill", "gospel"]),  # Discogs subgenres
    "chiptune": (["chiptune"], ["metal", "gospel", "country"]),  # Discogs subgenre (≈ 8-bit Misfits; library-thin)
    "gospel": (["gospel"], ["metal", "techno", "drill"]),  # Discogs subgenre
    "glasgow_folk": (["folk", "folk rock", "neofolk", "celtic"], ["metal", "techno", "drill"]),  # Discogs subgenres
    "glasgow_dream": (["dream pop", "shoegaze", "noise pop", "neo-psychedelia"], ["metal", "techno", "drill"]),
    "glasgow_indie": (["indie pop", "twee pop", "c-86", "jangle pop", "sophisti-pop", "chamber pop"], ["metal", "techno", "drill"]),
    "glasgow_soul": (["blue-eyed soul", "pop-soul", "northern soul", "funk"], ["metal", "screamo", "drill"]),
    "glasgow_postrock": (["post rock", "math rock"], ["rap", "drill", "gospel"]),  # Discogs subgenres
    "glasgow_anthems": (["indie rock", "dance-rock", "britpop", "new wave/post-punk revival"], ["metal", "techno", "ambient"]),
    "glasgow_synth": (["synth pop", "new wave", "electro", "dance-rock"], ["metal", "country", "gospel"]),
    "glasgow_postpunk": (["post-punk", "new wave/post-punk revival", "punk"], ["ambient", "gospel", "classical"]),
    "glasgow_house": (["house", "left-field house", "tech-house", "disco"], ["metal", "country", "folk"]),
    "glasgow_underground": (["techno", "minimal techno", "detroit techno", "acid house"], ["folk", "country", "gospel"]),
    "glasgow_bass": (["idm", "bassline", "dubstep", "breakbeat"], ["folk", "country", "gospel"]),  # Discogs subgenres
    "glasgow_late": (["downtempo", "trip-hop", "electronica", "ambient techno"], ["metal", "punk", "gospel"]),
    "london_dub": (["dub", "roots reggae", "dancehall", "reggae-pop"], ["metal", "techno", "screamo"]),
    "london_soul": (["blue-eyed soul", "neo-soul", "contemporary r&b", "acid jazz"], ["metal", "punk", "drill"]),
    "london_jazz": (["contemporary jazz", "jazz-funk", "spiritual jazz", "afro-beat", "acid jazz"], ["metal", "drill", "screamo"]),
    "london_triphop": (["trip-hop", "downtempo", "downbeat", "idm"], ["metal", "punk", "gospel"]),
    "london_mod": (["mod", "beat", "merseybeat", "freakbeat", "british invasion", "british rhythm & blues", "garage rock", "garage rock revival"], ["metal", "techno", "drill"]),  # WHY: word-boundary matched so "mod"≠"modern"; adds beat/garage 60s scene (audit london_mod-1)
    "london_britpop": (["brit pop"], ["metal", "techno", "gospel"]),  # Discogs subgenre
    "london_indie": (["indie rock", "new wave/post-punk revival", "garage rock revival"], ["metal", "techno", "gospel"]),
    "london_calling": (["punk", "post-punk", "oi!", "new wave"], ["ambient", "gospel", "classical"]),
    "london_garage": (["uk garage", "garage", "broken beat", "bassline"], ["metal", "country", "folk"]),
    "london_grime": (["grime", "uk drill", "bass music"], ["folk", "country", "ambient"]),
    "london_dubstep": (["dubstep", "bass music", "uk garage"], ["folk", "country", "gospel"]),
    "london_jungle": (["jungle", "drum n bass", "breakbeat", "breaks", "big beat"], ["folk", "country", "ambient"]),  # Discogs subgenres
    "melbourne_folk": (["folk", "folk rock", "neofolk", "celtic"], ["metal", "techno", "drill"]),  # Discogs subgenres
    "melbourne_dream": (["dream pop", "jangle pop", "indie pop", "neo-psychedelia"], ["metal", "techno", "drill"]),
    "melbourne_soul": (["soul", "funk", "neo soul", "jazz-funk"], ["metal", "screamo", "drill"]),  # Discogs subgenres
    "melbourne_sunset": (["surf", "indie pop", "tropical", "sunshine pop"], ["metal", "drill", "techno"]),
    "melbourne_indie": (["indie rock", "alternative/indie rock", "jangle pop"], ["metal", "techno", "drill"]),
    "melbourne_pubrock": (["aussie rock", "pub rock", "album rock", "hard rock", "heartland rock"], ["techno", "ambient", "gospel"]),
    "melbourne_hiphop": (["trap", "cloud rap", "conscious", "boom bap", "hardcore hip-hop"], ["metal", "ambient", "gospel"]),  # Discogs subgenres
    "melbourne_postpunk": (["post-punk", "goth rock", "new wave"], ["gospel", "ambient", "country"]),
    "melbourne_psych": (["psychedelic rock", "garage rock"], ["gospel", "country", "ambient"]),  # Discogs subgenres
    "melbourne_garagepunk": (["punk", "pop punk", "hardcore", "melodic hardcore", "garage rock"], ["ambient", "gospel", "classical"]),  # Discogs subgenres
    "melbourne_club": (["house", "progressive house", "electro", "club/dance"], ["folk", "country", "gospel"]),
    "melbourne_techno": (["techno", "minimal techno", "tech-house", "acid house"], ["folk", "country", "gospel"]),
    # ----- style-defined genre mixes (positives required) -----
    "synthpop_romance": (["synth pop", "synthwave", "new romantic", "new wave",
                          "indie electronic", "dance-pop", "sophisti-pop", "dream pop",
                          "left-field pop", "neo-electro"],
                         ["metal", "rap", "country", "folk", "jazz", "punk", "hardcore", "blues"]),
    "folk_acoustic":    (["indie folk", "contemporary folk", "folk-rock", "folk-pop",
                          "singer/songwriter", "americana", "new acoustic", "alt-country",
                          "anti-folk", "folk revival", "neo-traditional folk", "folk"],
                         ["metal", "rap", "edm", "techno", "house", "club/dance", "synth"]),
    # acoustic_romance: no genre positive — selected by the Jack-Johnson seed centroid + romantic lyric
    # themes (_SEED_ARTISTS / _PROFILE_LYRIC_THEMES), so _has_required_style returns True (ungated).
    "indie_romance":    (["indie rock", "indie pop", "dream pop", "indie folk", "indie electronic",
                          "twee pop", "chamber pop", "sadcore", "shoegaze", "slowcore",
                          "jangle pop", "noise pop", "alternative singer/songwriter"],
                         ["metal", "rap", "edm", "country", "hardcore", "club/dance"]),
    "romantic_jazz":    (["vocal jazz", "smooth jazz", "cool", "piano jazz", "crossover jazz",
                          "jazz-pop", "torch songs", "bossa nova", "standards", "lounge",
                          "saxophone jazz", "post-bop", "swing", "jazz blues"],
                         ["jazz-rap", "jazz-rock", "rap", "metal", "punk", "edm"]),
    "jazz_dinner":      (["vocal jazz", "jazz", "cool", "smooth jazz", "crossover jazz",
                          "piano jazz", "standards", "lounge", "bossa nova", "traditional pop",
                          "swing", "cocktail", "post-bop"],
                         ["jazz-rap", "jazz-rock", "rap", "metal", "punk", "hardcore", "edm"]),
    "string_quartet":   (["chamber music", "classical crossover", "orchestral", "neo-classical",
                          "modern composition", "concerto", "chamber pop", "baroque pop",
                          "symphony", "modal music"],
                         ["rap", "metal", "edm", "punk", "club/dance"]),
    "strings_romance":  (["orchestral", "chamber music", "classical crossover", "neo-classical",
                          "modern composition", "chamber pop", "baroque pop"],
                         ["rap", "metal", "edm", "punk", "club/dance"]),
    "piano_romance":    (["piano jazz", "neo-classical", "modern composition",
                          "contemporary instrumental", "keyboard", "classical crossover",
                          "instrumental pop", "chamber music"],
                         ["metal", "rap", "punk", "edm", "club/dance", "hardcore"]),
    # ----- soft style preference (instrumental focus; not required) -----
    "focus":     (["orchestral", "instrumental", "ambient", "classical", "score",
                   "soundtrack", "new age", "film music", "post-rock"],
                  ["singer/songwriter"]),
    "deep_work": (["orchestral", "instrumental", "ambient", "classical", "score",
                   "soundtrack", "new age", "film music", "post-rock"],
                  ["singer/songwriter"]),
}

# Profiles whose identity IS a genre — positives above are required (the candidate pool is
# hard-filtered to them in _build_mix_tracks). focus/deep_work stay a soft nudge.
_STYLE_DEFINED_PROFILES = {
    # --- Meloday+ gap-fill mixes (hard style gate) ---
    "neoclassical", "yacht_rock", "swagger",
    "chart_pop", "dance_pop", "indie_pop", "synth_pop",
    "indie_rock", "post_grunge", "rap_rock", "festival_edm", "soundtracks", "rave_cave",
    "festive",
    "spring_acoustic",
    "spring_strings",
    "spring_jangle",
    "summer_heat",
    "summer_breeze",
    "summer_tropical",
    "autumn_leaves",
    "autumn_jazz",
    "autumn_embers",
    "winter_frost",
    "winter_cosy",
    "winter_nights",
    "winter_jazz",
    # genre-character mood mixes. The instrumental ones (meditation/spa/yoga/power_nap/study/
    # deep_reading) were relaxed once TF vocal_presence landed — they now lean instrumental via a
    # heavier per-profile vocal weight (_INSTRUMENTAL_VOCAL) + soft _PROFILE_STYLE_SIGNALS, for
    # more cross-genre variety. campfire/dinner_party stay gated (vocal genres, not instrumental).
    "campfire", "dinner_party",
    "funk_disco",
    "neo_soul",
    "motown_soul",
    "after_hours_rnb",
    "acid_jazz",
    "boom_bap",
    "conscious_flow",
    "g_funk",
    "trap_mode",
    "lofi_beats",
    "house_party",
    "deep_house",
    "techno",
    "trance",
    "dnb",
    "bass_drop",
    "uk_garage",
    "synthwave",
    "industrial",
    "vaporwave",
    "downtempo",
    "hyperpop",
    "classic_rock",
    "heavy_riffs",
    "punk_energy",
    "garage_grunge",
    "emo_poppunk",
    "britpop_rock",
    "blues_bar",
    "psych_haze",
    "prog_rock",
    "stoner_rock",
    "reggae_dub",
    "afrobeat",
    "latin_heat",
    "bossa_samba",
    "celtic_folk",
    "ska",
    "bebop",
    "swing_bigband",
    "smooth_jazz",
    "country_roads",
    "outlaw_country",
    "bluegrass",
    "rockabilly_surf",
    "cinematic_epic",
    "ambient_drift",
    "post_rock",
    "chiptune",
    "gospel",
    "glasgow_folk",
    "glasgow_dream",
    "glasgow_indie",
    "glasgow_soul",
    "glasgow_postrock",
    "glasgow_anthems",
    "glasgow_synth",
    "glasgow_postpunk",
    "glasgow_house",
    "glasgow_underground",
    "glasgow_bass",
    "glasgow_late",
    "london_dub",
    "london_soul",
    "london_jazz",
    "london_triphop",
    "london_mod",
    "london_britpop",
    "london_indie",
    "london_calling",
    "london_garage",
    "london_grime",
    "london_dubstep",
    "london_jungle",
    "melbourne_folk",
    "melbourne_dream",
    "melbourne_soul",
    "melbourne_sunset",
    "melbourne_indie",
    "melbourne_pubrock",
    "melbourne_hiphop",
    "melbourne_postpunk",
    "melbourne_psych",
    "melbourne_garagepunk",
    "melbourne_club",
    "melbourne_techno",
    "synthpop_romance", "folk_acoustic", "acoustic_romance", "indie_romance",
    "romantic_jazz", "jazz_dinner", "string_quartet", "strings_romance", "piano_romance",
}


# The Discogs-400 genre model is MULTI-LABEL: it emits its top-6 subgenre guesses per track, each with a
# confidence, and the low-probability tail is noise — it sprays narrow microgenres (e.g.
# "Electronic---Hardstyle" at 0.06) onto tracks that aren't remotely that genre. A gate positive matching
# one of those sprays drags pure misfits into a tight mix: an indie-electronic track (Metronomy), a
# dream-pop track (Mansionair), an emo-pop track (Sleeping with Sirens) and a film-score cue (Koji Kondo)
# all leaked into Rave Cave on a <0.11 "Hardstyle"/"Hard Trance" guess. So for genre MEMBERSHIP we only
# trust a Discogs subgenre the classifier was reasonably confident about. 0.12 cleanly separated the noise
# sprays (the misfits sat at 0.06–0.11; the entire <0.05 tail was Camila Cabello / RHCP / Janet Jackson)
# from genuine tags (Hannah Laing's real hardstyle 0.15, Darude's hard-trance 0.21) and thinned no pool
# below usable size. Plex styles are human-curated (no confidence) and always count.
_DISCOGS_TAG_FLOOR = 0.12

def _track_style_tags(entry, min_conf=0.0):
    """Lowercased style/genre tags for matching: Plex STYLES (granular, human-curated) + the Discogs-400
    genre classifier (each 'Category---Style' split into both parts). The broad Plex GENRES (the ~7
    mega-buckets like "Pop/Rock", "Electronic") are deliberately EXCLUDED — they're too coarse to
    gate on and pollute membership (a single "Pop/Rock" bucket can't separate indie-rock from
    chart-pop). Discogs carries genre (parent + 400 subgenres, ~98% coverage); Plex styles are the
    granular fallback for the ~2% not yet Discogs-analysed. When `min_conf` > 0, a Discogs subgenre is
    included only if the classifier's confidence for it is at least that (the gate passes
    `_DISCOGS_TAG_FLOOR` so a low-probability spray can't grant genre membership); Plex styles, carrying
    no confidence, always count."""
    tags = [t.lower() for t in (entry.get("styles") or [])]
    gd = entry.get("genre_discogs") or {}
    if isinstance(gd, dict):
        for k, sc in gd.items():
            if min_conf <= 0 or sc is None or sc >= min_conf:
                tags += [p.lower() for p in k.split("---")]
    else:
        for k in gd:                                  # legacy list form: no confidence, always count
            tags += [p.lower() for p in str(k).split("---")]
    return tags


# Every style-gated mix requires its DOMINANT Discogs parent genre to fall within an allowed set, so a
# stray cross-genre subgenre tag (the Discogs-400 classifier spraying "hardstyle" onto pop, "smooth
# jazz" onto a rock track, etc.) can't drag an off-genre track into the mix. Discogs is the genre
# source of truth (parent + 400 subgenres, ~98% covered); the broad Plex genres are not used. Assigned
# by genre family — a single parent for clean genres, a set for genuinely cross-parent mixes (jazz+soul,
# the pop scatter, disco = funk/soul+electronic, etc.). One entry per _STYLE_DEFINED_PROFILES mix.
_PROFILE_GENRE_PARENT = {
    # --- Electronic ---
    "techno": {"electronic"}, "deep_house": {"electronic"}, "trance": {"electronic"},
    "house_party": {"electronic"}, "downtempo": {"electronic"}, "ambient_drift": {"electronic"},
    "synthwave": {"electronic"}, "vaporwave": {"electronic"}, "dnb": {"electronic"},
    "lofi_beats": {"electronic"}, "london_triphop": {"electronic"},
    # WHY: tropical names latin/reggae/bossa, not just electronic tropical-house (audit summer_tropical-1)
    "summer_tropical": {"electronic", "latin", "reggae", "folk, world, & country", "funk / soul"},
    "uk_garage": {"electronic"}, "festival_edm": {"electronic"}, "winter_nights": {"electronic"},
    "chiptune": {"electronic"}, "london_dubstep": {"electronic"}, "london_jungle": {"electronic"},
    "glasgow_house": {"electronic"}, "glasgow_underground": {"electronic"}, "glasgow_late": {"electronic"},
    "glasgow_bass": {"electronic"},
    "glasgow_synth": {"electronic", "rock", "pop"},  # WHY: admits new-wave/synth-pop (Rock/Pop parent) e.g. Simple Minds (audit glasgow_synth-1)
    "melbourne_club": {"electronic"},
    "melbourne_techno": {"electronic"}, "london_garage": {"electronic"},
    "summer_heat": {"electronic", "funk / soul"},  # WHY: admits the disco/funk it names (Funk/Soul parent), +1479 (audit summer_heat-1)
    "rave_cave": {"electronic"},
    # --- Rock ---
    "classic_rock": {"rock"}, "heavy_riffs": {"rock"}, "indie_rock": {"rock"},
    "post_grunge": {"rock"}, "prog_rock": {"rock"}, "punk_energy": {"rock"},
    "emo_poppunk": {"rock"}, "garage_grunge": {"rock"}, "stoner_rock": {"rock"},
    "post_rock": {"rock"}, "britpop_rock": {"rock"}, "psych_haze": {"rock"},
    # WHY: both admit AM-pop/blue-eyed-soul (Pop / Funk&Soul parent) (audit yacht_rock-1 / summer_breeze-1)
    "yacht_rock": {"rock", "pop", "funk / soul"}, "summer_breeze": {"rock", "pop"},
    "spring_jangle": {"rock"}, "autumn_embers": {"rock"}, "rockabilly_surf": {"rock"},
    "indie_romance": {"rock"}, "london_calling": {"rock"}, "london_britpop": {"rock"},
    "glasgow_anthems": {"rock"}, "glasgow_postpunk": {"rock"}, "glasgow_postrock": {"rock"},
    "glasgow_indie": {"rock"}, "glasgow_dream": {"rock"}, "melbourne_indie": {"rock"},
    "melbourne_pubrock": {"rock"}, "melbourne_postpunk": {"rock"}, "melbourne_dream": {"rock"},
    "melbourne_garagepunk": {"rock"}, "melbourne_psych": {"rock"}, "london_indie": {"rock"},
    # WHY: london_mod is 60s mod/beat/garage ROCK — was wrongly under Classical, so it played modern-classical/film-scores (audit london_mod-1)
    "london_mod": {"rock"},
    # --- Jazz ---
    "winter_jazz": {"jazz"}, "jazz_dinner": {"jazz"}, "romantic_jazz": {"jazz"},
    # WHY: smooth_jazz admits quiet-storm (Funk/Soul); swing_bigband admits traditional-pop (Pop) (audit smooth_jazz-1 / swing_bigband-1)
    "smooth_jazz": {"jazz", "funk / soul"}, "bebop": {"jazz"}, "london_jazz": {"jazz"}, "swing_bigband": {"jazz", "pop"},
    "autumn_jazz": {"funk / soul", "jazz"},
    # --- Hip-Hop ---
    "boom_bap": {"hip hop"}, "trap_mode": {"hip hop"}, "conscious_flow": {"hip hop"},
    "g_funk": {"hip hop"}, "melbourne_hiphop": {"hip hop"},
    # --- Funk / Soul ---
    "neo_soul": {"funk / soul"}, "motown_soul": {"funk / soul"}, "glasgow_soul": {"funk / soul"},
    "london_soul": {"funk / soul"}, "melbourne_soul": {"funk / soul"},
    "dinner_party": {"funk / soul", "jazz"},  # WHY: names vocal-jazz/bossa (Jazz parent), rejected under funk/soul-only (audit dinner_party-1)
    "winter_cosy": {"funk / soul"}, "gospel": {"funk / soul"},
    "funk_disco": {"electronic", "funk / soul"}, "after_hours_rnb": {"electronic", "funk / soul"},
    # --- Folk, World, & Country ---
    "folk_acoustic": {"folk, world, & country", "rock"}, "spring_acoustic": {"folk, world, & country", "rock"},
    "autumn_leaves": {"folk, world, & country", "rock"}, "campfire": {"folk, world, & country", "rock"},
    "celtic_folk": {"folk, world, & country", "rock"}, "country_roads": {"folk, world, & country", "rock"},
    "outlaw_country": {"folk, world, & country", "rock"}, "bluegrass": {"folk, world, & country", "rock"},
    "glasgow_folk": {"folk, world, & country", "rock"},
    "melbourne_folk": {"folk, world, & country", "rock"},
    # --- Classical / Stage & Screen ---
    "neoclassical": {"classical", "stage & screen"}, "string_quartet": {"classical", "stage & screen"},
    "strings_romance": {"classical", "stage & screen"}, "piano_romance": {"classical", "stage & screen"},
    "spring_strings": {"classical", "stage & screen"}, "cinematic_epic": {"classical", "stage & screen"},
    "soundtracks": {"classical", "stage & screen"},
    "winter_frost": {"classical", "stage & screen", "electronic"},  # WHY: admits ambient (Electronic parent), the frost half currently rejected (audit winter_frost-1)
    # --- Pop (scatters across parents) ---
    "chart_pop": {"electronic", "pop", "rock"}, "synth_pop": {"electronic", "pop", "rock"},
    "synthpop_romance": {"electronic", "pop", "rock"}, "dance_pop": {"electronic", "pop", "rock"},
    "indie_pop": {"electronic", "pop", "rock"}, "melbourne_sunset": {"electronic", "pop", "rock"},
    "festive": {"electronic", "pop", "rock"},
    # --- Cross-parent / edge ---
    "industrial": {"electronic", "rock"},
    # WHY: "dub"⊂"dubstep" leaked 59% dubstep into these reggae mixes; reggae-only parent rejects Electronic dubstep (audit dub fix)
    "reggae_dub": {"reggae"}, "london_dub": {"reggae"}, "bass_drop": {"electronic", "hip hop"},
    "latin_heat": {"electronic", "latin"}, "rap_rock": {"hip hop", "rock"},
    "ska": {"reggae", "rock"}, "blues_bar": {"blues", "rock"},
    "acid_jazz": {"electronic", "funk / soul", "jazz"}, "bossa_samba": {"jazz", "latin"},
    "london_grime": {"electronic", "hip hop"}, "swagger": {"funk / soul", "hip hop"},
    "afrobeat": {"electronic", "folk, world, & country", "funk / soul"}, "hyperpop": {"electronic", "pop"},
}


def _discogs_subgenres(entry, min_conf):
    """Floored Discogs SUBGENRE names (the part after '---'), lowercased — the matchable genre tags for
    the membership gate. A subgenre counts only when the classifier's confidence >= min_conf, so a
    low-probability multi-label spray can't grant membership. The PARENT half of each 'Parent---Sub' is
    deliberately excluded here (the parent is enforced separately in _has_required_style) — otherwise a
    positive that is a substring of its parent name (e.g. "soul" ⊂ "Funk / Soul", "folk" ⊂ "Folk, World,
    & Country") would match the whole parent and pull every track in the family."""
    gd = entry.get("genre_discogs") or {}
    if isinstance(gd, dict):
        return [k.split("---")[-1].lower() for k, sc in gd.items() if (sc is None or sc >= min_conf)]
    return [str(k).split("---")[-1].lower() for k in gd]


# Word-boundary genre matching. WHY: the membership gate matched a positive against a Discogs subgenre as a
# plain SUBSTRING (`sub in tag`), so a short positive leaked into an unrelated subgenre — "mod" ⊂ "modern",
# "am pop" ⊂ "dream pop", "dub" ⊂ "dubstep". Matching whole TOKENS instead fixes the entire leak class while
# still matching every legitimate multi-word / hyphenated positive ("deep house", "synth-pop", "contemporary
# pop/rock") and the space↔hyphen spelling variants. Replaces the per-profile _STYLE_EXACT_PROFILES hack.
# (audit substr-fix-systemic)
_STYLE_TOK_RE    = re.compile(r"[\s\-&/,']+")   # boundaries: space, hyphen, ampersand, slash, comma, apostrophe
_STYLE_TOK_CACHE = {}                            # genre string -> token tuple (bounded: a few hundred subgenres)

def _style_tokens(s):
    toks = _STYLE_TOK_CACHE.get(s)
    if toks is None:
        toks = tuple(t for t in _STYLE_TOK_RE.split(s.lower()) if t)
        _STYLE_TOK_CACHE[s] = toks
    return toks

def _positive_matches(positive, tag):
    """True if `positive`'s tokens occur as a contiguous run within `tag`'s tokens. So "mod" matches the
    subgenre "mod" but NOT "modern"/"modal"; "house" matches "deep house"/"tech-house" but not "microhouse";
    "synth-pop" matches the "synth pop" spelling variant. Composite positives must be pre-split (see dnb)."""
    pt = _style_tokens(positive)
    if not pt:
        return False
    tt = _style_tokens(tag)
    n = len(pt)
    return any(tt[i:i + n] == pt for i in range(len(tt) - n + 1))


def _entry_lastfm_tags(entry):
    """Lowercased Last.fm community (crowd) tags for a track — artist + track level. These are genre
    LABELS supplied by listeners, distinct from the Discogs audio-model classifier and from Plex's genre
    tags. They're the membership source for format/scene genres an audio model can't name (post-grunge,
    progressive rock): Discogs has no such subgenre, but the Last.fm community tags them accurately."""
    out = set()
    for f in ("lastfm_artist_tags", "lastfm_track_tags"):
        v = entry.get(f)
        if isinstance(v, dict):
            out |= {t.lower() for t in v}
        elif isinstance(v, (list, tuple)):
            out |= {str(t).lower() for t in v}
    return out


# Format/scene genres the Discogs audio model has no label for and the acoustic centroid can't isolate
# (verified: a Pink-Floyd/Rush-seeded centroid pulled Tiësto DJ mixes; a Nickelback/Creed one pulled
# Springsteen/McCartney). Their membership comes from Last.fm community tags instead — still RANKED on our
# own centroid + leans. The positive list in _PROFILE_STYLE_SIGNALS is matched against the Last.fm tags.
_LASTFM_GATED = {"post_grunge", "prog_rock"}

# (Retired _STYLE_EXACT_PROFILES: london_mod's "mod" no longer leaks into "modern" now that the gate uses
#  word-boundary token matching for ALL profiles — see _positive_matches above. audit substr-fix-systemic.)

# festive (Christmas) is a calendar event, not a genre or a sound — neither Discogs nor the centroid can
# express it. Gate on a holiday signal: a seasonal keyword in the title, or Plex flagging it Holiday/
# Christmas (the one place a Plex tag is still consulted, precisely because "Christmas" isn't a sound).
_FESTIVE_TITLE_KW = ("christmas", "xmas", "santa", "sleigh", "jingle", "noel", "navidad", "mistletoe",
                     "yuletide", "wonderland", "auld lang", "let it snow", "holly jolly", "silent night",
                     "feliz navidad", "deck the hall")


def _has_required_style(entry, profile_key):
    """True if the track qualifies for a style-defined mix. Membership source by mix:
      • most mixes → a Discogs SUBGENRE positive (confidence-floored), AND the dominant Discogs parent
        within the mix's allowed set (_PROFILE_GENRE_PARENT). Plex styles are NOT consulted.
      • _LASTFM_GATED mixes → a Last.fm community-tag positive (Discogs can't name these genres).
      • festive → a holiday signal (seasonal title keyword or Plex Holiday flag).
      • mixes with no positive list (e.g. the centroid-defined acoustic_romance) → always True; the seed
        centroid + lyric themes do the selecting in _build_mix_tracks.
    Untagged tracks return False for positive-gated mixes, so a mix never includes unconfirmable tracks."""
    if profile_key == "festive":
        title = (entry.get("title") or "").lower()
        if any(k in title for k in _FESTIVE_TITLE_KW):
            return True
        return any("holiday" in s.lower() or "christmas" in s.lower() for s in (entry.get("styles") or []))

    sig = _PROFILE_STYLE_SIGNALS.get(profile_key)
    positive_subs = sig[0] if sig else None

    if positive_subs:
        if profile_key in _LASTFM_GATED:
            tags = _entry_lastfm_tags(entry)                     # crowd genre labels (Discogs can't name these)
        else:
            tags = _discogs_subgenres(entry, _DISCOGS_TAG_FLOOR)  # Discogs subgenres only, confidence-floored
        # Word-boundary token match (see _positive_matches): "mod" matches the subgenre "mod" but not
        # "modern", "dub" not "dubstep", "am pop" not "dream pop" — the systemic fix for the leak class.
        if not tags or not any(_positive_matches(sub, tag) for tag in tags for sub in positive_subs):
            return False

    # Dominant-Discogs-parent constraint — independent of the positive match, so it also applies to a
    # positive-less centroid mix that sets a parent. Skipped for Last.fm-gated mixes (the community tag is
    # the authority) and for mixes with no parent. A tie with any non-allowed parent rejects, so a stray
    # cross-genre subgenre spray (e.g. "Electronic---Hardstyle" on a pop track) can't drag it in.
    req_parents = _PROFILE_GENRE_PARENT.get(profile_key)
    if req_parents and profile_key not in _LASTFM_GATED:
        gd = entry.get("genre_discogs") or {}
        keys = list(gd.keys()) if isinstance(gd, dict) else (gd or [])
        if keys:
            parents = Counter(str(k).split("---")[0].strip().lower() for k in keys)
            mx = max(parents.values())
            top = [p for p, v in parents.items() if v == mx]
            if not all(p in req_parents for p in top):
                return False
    return True


# Era windows for the nostalgia mixes, so "throwbacks" are actually old. Computed relative to
# the current year so the window never goes stale. (None = open end.)
_NOW_YEAR = date.today().year
_PROFILE_YEAR_WINDOW = {
    # ---- 7 decade mixes (era) ----
    "decade_60s": (1960, 1969),
    "decade_70s": (1970, 1979),
    "decade_80s": (1980, 1989),
    "decade_90s": (1990, 1999),
    "decade_00s": (2000, 2009),
    "decade_10s": (2010, 2019),
    "decade_20s": (2020, 2029),
    "throwback_anthems": (None, _NOW_YEAR - 10),   # nothing from the last ~10 years (throwback distance)
    "old_friends":       (None, _NOW_YEAR - 8),
    "memory_lane":       (None, _NOW_YEAR - 10),
    "school_days":       (1990, _NOW_YEAR - 12),
}


def _year_in_window(entry, window):
    """True if the track's release year falls in the (min, max) window (None = open end).
    Tracks with no year are excluded while a window is active — we can't confirm they're old."""
    y = entry.get("year")
    if not y:
        return False
    lo, hi = window
    return (lo is None or y >= lo) and (hi is None or y <= hi)


# ---------------------------------------------------------------------------
# Unfillable-profile guard — cheap per-run yield estimate
# ---------------------------------------------------------------------------
_MIN_PROFILE_YIELD = 25          # a profile must clear this many HARD-gate-passing cache entries to be
                                 # worth surfacing — below it the mix is visibly thin/repetitive
_YIELD_FILLABLE    = 10 ** 9     # sentinel for acoustic-only profiles (no hard gate → always fillable)
_YIELD_CACHE       = {}          # (id(cache), profile_key, cap) -> capped yield, per cache object


def _profile_yield(profile_key, essentia_cache, cap=_MIN_PROFILE_YIELD):
    """Count cache entries passing the profile's HARD gates only — style (_STYLE_DEFINED_PROFILES /
    _has_required_style), year-window (_PROFILE_YEAR_WINDOW), geo (_PROFILE_GEO_GATE / _origin_match) —
    reusing the exact predicates _build_mix_tracks applies. Acoustic-only profiles (no hard gate) are
    always fillable (returns _YIELD_FILLABLE, no scan). Counting stops at `cap` (we only need to know if a
    profile clears the bar), so fillable profiles early-exit cheaply and only genuinely-thin ones scan the
    whole cache. Pure dict scan, no Plex calls. Memoised per cache object."""
    ck = (id(essentia_cache), profile_key, cap)
    v = _YIELD_CACHE.get(ck)
    if v is not None:
        return v
    style_gated = profile_key in _STYLE_DEFINED_PROFILES
    yw  = _PROFILE_YEAR_WINDOW.get(profile_key)
    geo = _PROFILE_GEO_GATE.get(profile_key)
    if not (style_gated or yw or geo):
        _YIELD_CACHE[ck] = _YIELD_FILLABLE
        return _YIELD_FILLABLE
    smy    = _song_min_year_map(essentia_cache) if yw else None
    lo, hi = yw if yw else (None, None)
    n = 0
    for e in essentia_cache.values():
        if style_gated and not _has_required_style(e, profile_key):
            continue
        if yw:
            oy = smy.get(_entry_song_key(e)) or e.get("year")
            if oy is None or (lo is not None and oy < lo) or (hi is not None and oy > hi):
                continue
        if geo and not _origin_match(e, geo):
            continue
        n += 1
        if n >= cap:                 # cleared the bar — no need to count further
            break
    _YIELD_CACHE[ck] = n
    return n


def _style_tag_boost(entry, profile_key):
    """
    Distance adjustment from genre/style tag compatibility. Orders tracks within a profile by
    how well their styles fit. Style-defined profiles get a stronger positive pull (the hard
    pool filter already guarantees purity); soft profiles (focus/deep_work) get a gentle nudge.
    Returns 0.0 for profiles with no style signal defined.
    """
    signals = _PROFILE_STYLE_SIGNALS.get(profile_key)
    if not signals:
        return 0.0
    positive_subs, negative_subs = signals
    tags = _track_style_tags(entry)
    if not tags:
        return 0.0

    def _matches(subs):
        return any(sub in tag for tag in tags for sub in subs)

    boost = 0.0
    if _matches(positive_subs):
        boost -= 0.18 if profile_key in _STYLE_DEFINED_PROFILES else 0.12
    if _matches(negative_subs):
        boost += 0.10
    return boost


# ---------------------------------------------------------------------------
# Essentia high-level mood/theme scoring (calibrated per-track classifiers)
# ---------------------------------------------------------------------------
_MOODTHEME_WEIGHT = 0.40   # pull toward tracks whose mood/theme tags match the mix's theme
_MOODCLASS_WEIGHT = 0.25   # pull on the calibrated production-quality mood classes

# Profile -> mtg_jamendo mood/theme tags it wants (from the 56-tag model vocabulary). A track
# scoring high in these tags is pulled into the mix. Only profiles with a clear theme are listed.
_PROFILE_MOODTHEME = {
    # --- Meloday+ gap-fill mixes (neoclassical omitted — instrumental) ---
    "situationship": ["love", "melancholic", "emotional"],
    "sad_bangers": ["energetic", "party", "melancholic", "sad"],
    "power_ballads": ["ballad", "epic", "dramatic", "emotional"],
    "restless": ["energetic", "dark", "emotional"],
    "yacht_rock": ["relaxing", "soft", "summer"],
    "swagger": ["cool", "groovy", "energetic"],
    "chart_pop": ["happy", "positive", "upbeat", "energetic"],
    "dance_pop": ["party", "energetic", "upbeat", "happy"],
    "indie_pop": ["happy", "positive", "melodic"],
    "synth_pop": ["energetic", "cool", "retro", "upbeat"],
    # rock / electronic / scores gap-fill
    "indie_rock": ["energetic", "upbeat", "cool", "melodic"],
    "post_grunge": ["energetic", "dark", "heavy", "powerful"],
    "rap_rock": ["energetic", "heavy", "dark", "powerful"],
    "festival_edm": ["party", "energetic", "uplifting", "upbeat"],
    "soundtracks": ["film", "epic", "dramatic", "soundscape"],
    "rave_cave": ["party", "energetic", "powerful", "uplifting"],
    # Seasonal / weather
    "festive": ["christmas", "holiday"], "summer_heat": ["summer", "party"],
    "summer_breeze": ["summer", "relaxing"], "summer_roadtrip": ["summer", "travel"],
    "summer_tropical": ["summer", "groovy"], "summer_evening": ["summer", "relaxing"],
    "beach_vibes": ["summer", "relaxing"], "cookout": ["summer", "fun"],
    "sunny": ["happy", "summer", "upbeat"], "spring_bloom": ["happy", "positive"],
    "autumn_rain": ["melancholic", "calm"], "winter_nights": ["dark", "calm"],
    "rainy_day": ["melancholic", "calm", "relaxing"], "cosy": ["calm", "soft", "relaxing"],
    # Romance
    "romantic_mix": ["love", "romantic"], "love_songs": ["love", "romantic"],
    "modern_romance": ["love", "romantic"], "late_night_romance": ["love", "romantic", "sexy"],
    "slow_dance": ["love", "romantic"], "first_date": ["love", "happy"],
    "romantic_dinner": ["love", "romantic"], "acoustic_romance": ["love", "romantic"],
    "indie_romance": ["love", "romantic"], "synthpop_romance": ["love", "romantic"],
    "piano_romance": ["love", "romantic", "emotional"], "romantic_jazz": ["love", "romantic"],
    "crush": ["love", "happy", "fun"], "slow_burn": ["sexy", "romantic", "love"],
    "loved_up": ["love", "happy", "romantic"], "long_distance": ["love", "melancholic", "emotional"],
    "flirty": ["sexy", "fun", "love"], "devotion": ["love", "romantic", "emotional"],
    "wedding_day": ["love", "happy", "romantic"], "moving_on": ["hopeful", "emotional"],
    "heartbreak": ["sad", "melancholic", "emotional"],
    # Emotional
    "grief_release": ["sad", "melancholic", "emotional"], "melancholy": ["melancholic", "sad"],
    "triumphant": ["epic", "powerful", "motivational", "uplifting"],
    "hopeful": ["hopeful", "uplifting", "inspiring"], "yearning": ["melancholic", "emotional"],
    "serene": ["calm", "relaxing"], "tender": ["soft", "emotional", "love"],
    "defiant": ["powerful", "heavy", "energetic"], "vulnerable": ["emotional", "sad", "soft"],
    "awe_wonder": ["epic", "dream", "space"], "bittersweet": ["melancholic", "emotional"],
    "empowering": ["powerful", "motivational", "uplifting"], "angst_mix": ["dark", "heavy", "emotional"],
    "moody_mix": ["dark", "melancholic"], "euphoric": ["uplifting", "energetic", "party"],
    "nostalgia_mix": ["retro", "melancholic"], "cathartic": ["powerful", "emotional", "energetic"],
    # Activity
    "meditation": ["meditative", "calm", "relaxing"], "yoga_stretch": ["calm", "relaxing", "meditative"],
    "spa_bath": ["relaxing", "calm", "soft"], "power_nap": ["calm", "soft", "dream"],
    "workout": ["sport", "energetic", "motivational"], "running": ["sport", "energetic", "motivational"],
    "gaming": ["action", "energetic", "epic"], "study_session": ["background", "calm"],
    "deep_reading": ["background", "calm"], "creative_flow": ["inspiring", "motivational"],
    "cooking_mix": ["fun", "upbeat"], "cool_down": ["calm", "relaxing"],
    "gardening": ["calm", "nature", "happy"], "housework_hustle": ["fun", "upbeat", "energetic"],
    # Cinematic / atmospheric
    "cinematic_epic": ["epic", "film", "movie", "action", "trailer"], "post_rock": ["epic", "dramatic"],
    "ambient_drift": ["soundscape", "calm", "space"], "dreamy_mix": ["dream", "soft", "calm"],
    "midnight": ["dark", "calm"], "three_am": ["dark", "calm", "melancholic"],
    "witching_hour": ["dark", "dramatic"], "sunrise": ["hopeful", "calm", "uplifting"],
    "golden_afternoon": ["summer", "relaxing"], "starlit": ["space", "dream", "calm"],
    "blue_hour": ["melancholic", "calm"], "overcast": ["melancholic", "calm"],
    "after_dark": ["dark", "sexy"], "night_drive": ["dark", "energetic"],
    # Party / upbeat / occasion
    "party": ["party", "fun", "energetic"], "celebration": ["party", "happy", "fun"],
    "friday_night": ["party", "fun", "energetic"], "pre_party": ["party", "energetic"],
    "happy": ["happy", "fun", "positive"], "friday_feeling": ["fun", "happy", "upbeat"],
    "party_throwback": ["party", "retro", "fun"], "main_character": ["powerful", "energetic", "cool"],
    "confidence_boost": ["powerful", "motivational", "energetic"],
    "monday_motivation": ["motivational", "energetic", "uplifting"],
    "sunday_scaries": ["melancholic", "calm"], "wind_down": ["calm", "relaxing"],
    "treat_yourself": ["sexy", "fun", "cool"], "midweek_reset": ["motivational", "positive"],
    "dinner_party": ["cool", "groovy"], "evening_unwind": ["calm", "relaxing"],
    # Social / nostalgia / travel
    "throwback_anthems": ["retro", "fun", "energetic"], "old_friends": ["happy", "fun"],
    "campfire": ["calm", "nature", "soft"], "singalong": ["fun", "energetic", "upbeat"],
    "memory_lane": ["retro", "melancholic"], "school_days": ["retro", "fun"],
    "game_night": ["fun", "funny"], "road_trip": ["travel", "fun", "energetic"],
    "driving_mix": ["travel", "energetic"], "driving_singalong": ["travel", "fun", "upbeat"],
    "walking_mix": ["calm", "happy"], "commute_mix": ["energetic", "motivational"],
    # Genre mixes with a clear theme
    "bossa_samba": ["relaxing", "summer"], "reggae_dub": ["relaxing", "groovy"],
    "afrobeat": ["groovy", "energetic"], "smooth_jazz": ["relaxing", "calm"],
    "gospel": ["uplifting", "inspiring", "powerful"], "lofi_beats": ["calm", "relaxing", "background"],
    "celtic_folk": ["nature", "epic"], "latin_heat": ["party", "energetic", "groovy"],
}

# Profile -> {calibrated mood-class field: 1 to want high / 0 to want low}. Captures production
# qualities (acoustic, aggressive, electronic, danceable) the 56 theme tags don't.
_PROFILE_MOODCLASS = {
    # --- Meloday+ gap-fill mixes ---
    "sad_bangers": {"danceability_hl": 1, "mood_party": 1}, "neoclassical": {"mood_acoustic": 1},
    "dance_pop": {"danceability_hl": 1, "mood_party": 1},
    "rap_rock": {"mood_aggressive": 1}, "post_grunge": {"mood_aggressive": 1},
    "festival_edm": {"danceability_hl": 1, "mood_party": 1, "mood_electronic": 1},
    "rave_cave": {"danceability_hl": 1, "mood_party": 1, "mood_electronic": 1, "mood_aggressive": 1},
    "campfire": {"mood_acoustic": 1}, "acoustic_romance": {"mood_acoustic": 1},
    "folk_acoustic": {"mood_acoustic": 1}, "spring_acoustic": {"mood_acoustic": 1},
    "celtic_folk": {"mood_acoustic": 1}, "country_roads": {"mood_acoustic": 1},
    "bluegrass": {"mood_acoustic": 1},
    "heavy_riffs": {"mood_aggressive": 1}, "punk_energy": {"mood_aggressive": 1},
    "angst_mix": {"mood_aggressive": 1}, "defiant": {"mood_aggressive": 1},
    "stoner_rock": {"mood_aggressive": 1}, "garage_grunge": {"mood_aggressive": 1},
    "industrial": {"mood_aggressive": 1, "mood_electronic": 1},
    "techno": {"mood_electronic": 1}, "deep_house": {"mood_electronic": 1},
    "trance": {"mood_electronic": 1}, "synthwave": {"mood_electronic": 1},
    "dnb": {"mood_electronic": 1}, "house_party": {"danceability_hl": 1, "mood_party": 1},
    "funk_disco": {"danceability_hl": 1}, "summer_heat": {"danceability_hl": 1, "mood_party": 1},
    "uk_garage": {"danceability_hl": 1, "mood_electronic": 1},
    "meditation": {"mood_relaxed": 1, "mood_aggressive": 0}, "spa_bath": {"mood_relaxed": 1},
    "sleep": {"mood_relaxed": 1, "mood_aggressive": 0}, "yoga_stretch": {"mood_relaxed": 1},
}


def _moodtheme_boost(entry, profile_key):
    """Pull tracks whose Essentia mood/theme tags match the profile's theme. `moodtheme` is
    {tag: prob}; returns 0.0 when either side is absent (so it no-ops until tracks are analysed)."""
    wanted = _PROFILE_MOODTHEME.get(profile_key)
    mt = entry.get("moodtheme")
    if not wanted or not mt:
        return 0.0
    return -_MOODTHEME_WEIGHT * min(1.0, sum(mt.get(t, 0.0) for t in wanted))


def _moodclass_boost(entry, profile_key):
    """Pull on the calibrated mood-class fields a profile cares about (high or low). No-ops for
    profiles without a spec or tracks without the data."""
    spec = _PROFILE_MOODCLASS.get(profile_key)
    if not spec:
        return 0.0
    total, n = 0.0, 0
    for field, want_high in spec.items():
        v = entry.get(field)
        if v is None:
            continue
        total += v if want_high else (1.0 - v)
        n += 1
    return -_MOODCLASS_WEIGHT * (total / n) if n else 0.0


# --- Broaden the two universal Essentia signals to EVERY profile, derived from its own intent
# (Plex mood signals + centroid). Hand-mapped entries above are preserved as overrides, so this
# only fills the gaps. Also lifts Last.fm coverage, since _lastfm_tag_boost reuses _PROFILE_MOODTHEME.
_MOOD_TO_THEME = {   # Plex mood signal -> mtg_jamendo moodtheme tags
    "lively": ["energetic"], "energetic": ["energetic"], "rousing": ["energetic", "powerful"],
    "exuberant": ["energetic", "fun", "happy"], "driving": ["energetic", "powerful"],
    "brash": ["energetic", "powerful"], "euphoric": ["uplifting", "energetic"],
    "playful": ["fun", "happy"], "fun": ["fun"], "carefree": ["fun", "happy"],
    "cheerful": ["happy", "fun"], "joyous": ["happy", "uplifting"], "bright": ["happy", "positive"],
    "sunny": ["happy", "summer"], "optimistic": ["hopeful", "positive", "uplifting"],
    "hopeful": ["hopeful", "uplifting"], "celebratory": ["party", "happy"],
    "confident": ["powerful", "cool"], "swaggering": ["cool", "powerful"], "powerful": ["powerful"],
    "intense": ["powerful", "dramatic"], "dramatic": ["dramatic", "epic"],
    "aggressive": ["heavy", "powerful", "dark"], "rebellious": ["powerful", "heavy"],
    "gritty": ["dark", "powerful"], "warm": ["positive", "soft"], "stylish": ["cool"],
    "sophisticated": ["cool"], "elegant": ["cool", "soft"], "smooth": ["cool", "relaxing"],
    "laid-back": ["relaxing", "calm", "cool"], "mellow": ["relaxing", "calm"],
    "calm": ["calm", "relaxing"], "relaxed": ["relaxing", "calm"],
    "soothing": ["calm", "relaxing", "soft"], "peaceful": ["calm", "meditative"],
    "gentle": ["soft", "calm"], "spiritual": ["meditative", "uplifting"],
    "atmospheric": ["soundscape", "dream"], "ethereal": ["dream", "soundscape", "soft"],
    "dreamy": ["dream", "soft"], "hypnotic": ["dream", "deep"], "nocturnal": ["dark", "calm"],
    "dark": ["dark"], "brooding": ["dark", "melancholic"], "nervous": ["dark", "dramatic"],
    "tender": ["soft", "emotional"], "intimate": ["love", "soft", "sexy"], "sensual": ["sexy", "love"],
    "romantic": ["love", "romantic"], "passionate": ["emotional", "powerful", "love"],
    "wistful": ["melancholic", "emotional"], "melancholy": ["melancholic", "sad"],
    "poignant": ["emotional", "melancholic"], "bittersweet": ["melancholic", "emotional"],
    "yearning": ["emotional", "melancholic"], "reflective": ["melancholic", "meditative", "emotional"],
    "introspective": ["melancholic", "meditative", "emotional"], "earnest": ["emotional"],
    "nostalgic": ["retro", "melancholic"], "earthy": ["calm", "nature"],
}


def _derive_essentia_signals():
    """Fill moodtheme + mood-class leans for every profile that wasn't hand-mapped, from its own
    intent: moodtheme from its Plex mood signals (via _MOOD_TO_THEME), mood classes from its
    centroid (the calibrated valence/arousal/danceability axes)."""
    for k in _MOOD_PROFILES:
        if k not in _PROFILE_MOODTHEME:
            tags = []
            for m in _PROFILE_MOOD_SIGNALS.get(k, ([], []))[0]:
                for t in _MOOD_TO_THEME.get(m.lower(), []):
                    if t not in tags:
                        tags.append(t)
            if tags:
                _PROFILE_MOODTHEME[k] = tags[:4]
        if k not in _PROFILE_MOODCLASS:
            c = _MOOD_PROFILES[k]
            v, a, d = c.get("valence", 0.55), c.get("arousal", 0.5), c.get("danceability", 0.45)
            spec = {}
            if v >= 0.66:
                spec["mood_happy"] = 1
            elif v <= 0.50:
                spec["mood_sad"] = 1
            if a >= 0.58 and v <= 0.58:
                spec["mood_aggressive"] = 1
            elif a <= 0.42:
                spec["mood_relaxed"] = 1
            if d >= 0.50 and a >= 0.52:
                spec["mood_party"] = 1
            if spec:
                _PROFILE_MOODCLASS[k] = spec


_derive_essentia_signals()

# Specific romance/date criteria — these mixes share "love" but feel genuinely different, so each
# gets its own moodtheme + lyric sub-feeling to stay distinct (their centroids already differ:
# Flirty is fast/playful, Slow Dance slow/intimate, Wedding joyous, Late Night sultry, Indie wistful).
_PROFILE_MOODTHEME.update({
    "romantic_mix": ["love", "romantic", "emotional"], "love_songs": ["love", "romantic"],
    "modern_romance": ["love", "romantic", "positive"], "date_night": ["love", "cool", "sexy"],
    "late_night_romance": ["sexy", "love", "dark"], "romantic_dinner": ["love", "cool", "relaxing"],
    "slow_dance": ["love", "romantic", "soft"], "first_date": ["love", "happy", "hopeful"],
    "crush": ["happy", "fun", "love"], "loved_up": ["love", "happy", "uplifting"],
    "flirty": ["sexy", "fun"], "devotion": ["love", "emotional", "soft"],
    "tender": ["soft", "emotional", "love"], "slow_burn": ["sexy", "emotional", "dark"],
    "wedding_day": ["love", "happy", "party"], "romantic_jazz": ["love", "cool", "soft"],
    "piano_romance": ["love", "calm", "soft"], "acoustic_romance": ["love", "soft", "emotional"],
    "indie_romance": ["love", "melancholic", "dream"], "synthpop_romance": ["love", "retro", "dream"],
    # subtle differentiation of the calm/melancholic, calm/relaxing and summer-chill clusters
    "autumn_rain": ["melancholic", "calm", "soft"], "blue_hour": ["melancholic", "dream", "calm"],
    "overcast": ["melancholic", "calm", "soundscape"], "sunday_scaries": ["melancholic", "dark", "calm"],
    "serene": ["calm", "meditative", "soft"], "wind_down": ["calm", "relaxing"],
    "evening_unwind": ["calm", "relaxing", "soft"], "cool_down": ["calm", "relaxing", "positive"],
    "golden_afternoon": ["summer", "relaxing", "happy"], "beach_vibes": ["summer", "relaxing", "fun"],
    "summer_evening": ["summer", "relaxing", "soft"],
})


_LASTFM_WEIGHT = 0.30   # pull toward tracks whose Last.fm community tags match the mix's theme

# Last.fm-specific tag words — the occasion/activity/era folksonomy the 56 model tags miss
# (a track tagged "workout"/"rainy day"/"makeout" won't contain the model words sport/melancholic/
# sexy). Merged with _PROFILE_MOODTHEME when matching the community tags.
_PROFILE_LASTFM = {
    "rainy_day": ["rain"], "workout": ["workout", "gym"], "running": ["running"],
    "study_session": ["study", "concentration", "focus"], "deep_reading": ["reading", "study"],
    "road_trip": ["road trip", "driving"], "driving_mix": ["driving", "road trip"],
    "driving_singalong": ["driving", "road trip"], "night_drive": ["night drive", "driving"],
    "sleep": ["sleep"], "chill": ["chill"], "lazy_sunday": ["lazy sunday", "chill"],
    "cosy": ["cosy", "chill"], "late_night_romance": ["makeout", "sensual"],
    "slow_burn": ["makeout", "sensual"], "flirty": ["sexy"], "festive": ["xmas", "holiday"],
    "happy": ["feel good"], "heartbreak": ["breakup", "heartbreak"], "morning": ["morning"],
    "brunch_mix": ["sunday morning"], "focus": ["focus", "concentration"],
    "deep_work": ["focus", "concentration"], "summer_roadtrip": ["road trip"],
    "beach_vibes": ["beach"], "cookout": ["bbq", "summer"], "main_character": ["confidence"],
    "confidence_boost": ["confidence", "empowering"], "throwback_anthems": ["throwback", "nostalgia"],
    "nostalgia_mix": ["nostalgia"], "pre_party": ["pregame", "party"], "meditation": ["meditation"],
}


def _lastfm_tag_boost(entry, profile_key):
    """Pull tracks whose Last.fm tags match the profile's theme words (_PROFILE_MOODTHEME plus the
    Last.fm-specific _PROFILE_LASTFM folksonomy). Track-level tags (precise, per-song) count double
    the artist-level tags (broad); weights are Last.fm's 0–100 counts. No-op until tags are synced."""
    wanted = list(_PROFILE_MOODTHEME.get(profile_key, ())) + list(_PROFILE_LASTFM.get(profile_key, ()))
    if not wanted:
        return 0.0
    tt = entry.get("lastfm_track_tags") or {}
    at = entry.get("lastfm_artist_tags") or {}
    if not tt and not at:
        return 0.0

    def _match(tags):
        return sum(w for t, w in tags.items() if any(word in t for word in wanted))

    score = 2.0 * _match(tt) + _match(at)
    return -_LASTFM_WEIGHT * min(1.0, score / 150.0)


# ---------------------------------------------------------------------------
# Geographic origin (MusicBrainz hierarchy, with a Last.fm scene-tag fallback)
# ---------------------------------------------------------------------------
_ORIGIN_WEIGHT  = 0.50   # strong pull toward local artists in a city/region mix
_ORIGIN_PENALTY = 0.15   # gentle push-out of confirmed non-local artists (keeps the genre though)

# Origin specs are a single `places` set (place names matched against the artist's consolidated place
# hierarchy — the MB heading area/region/country doesn't matter) + an optional `scene` set (Last.fm
# scene-tag fallback). NOTE: the glasgow_/london_/melbourne_ CITY mixes are deliberately NOT here — they
# use a tiered, country-bounded geo gate instead (see _PROFILE_GEO_TIERS below), not this soft boost.
_PROFILE_ORIGIN = {}

# Geographically-rooted genre mixes: prefer artists from the genre's home turf. The style gate keeps
# the genre coherent; origin just refines toward authentic local artists (and gently pushes out
# confirmed non-locals). Place names are LOWERCASE and matched against the consolidated `places`, so
# the MB heading doesn't matter. Extra (currently low-content) countries future-proof the geo sync.
_PROFILE_ORIGIN.update({
    "celtic_folk":  {"places": {"ireland", "scotland", "wales"}},
    "latin_heat":   {"places": {"mexico", "spain", "colombia", "argentina", "chile", "puerto rico",
                                "cuba", "venezuela", "peru", "dominican republic"}},
    "reggae_dub":   {"places": {"jamaica"}},
    "afrobeat":     {"places": {"nigeria", "south africa", "ghana", "senegal", "mali"}},
    "bossa_samba":  {"places": {"brazil"}},
    # match the UK PLACE NAME, not country_code: 15% of UK artists lack a GB code but have a UK place
    "britpop_rock": {"places": {"united kingdom", "england", "scotland", "wales", "northern ireland"}},
})

# Geo SHOWCASE mixes (category "geo_scene"): a HARD origin gate — only tracks whose artist is from the place
# — then ranked by Last.fm popularity, exactly like the decade mixes. `places` matched vs the artist's
# consolidated place hierarchy; `scene` adds a Last.fm scene-tag fallback.
_PROFILE_GEO_GATE = {
    "scotland_scene":  {"places": {"scotland"},  "scene": {"scotland"}},
    "australia_scene": {"places": {"australia"}, "scene": {"australia"}},
    "london_scene":    {"places": {"london"},    "scene": {"london"}},
    "scottish_hits":   {"places": {"scotland"},  "scene": {"scotland"}},
    "australian_hits": {"places": {"australia"}, "scene": {"australia"}},
    "london_hits":     {"places": {"london"},    "scene": {"london"}},
    "uk_scene":        {"places": {"united kingdom", "england", "scotland", "wales", "northern ireland", "great britain"}, "scene": {"british"}},
    "uk_hits":         {"places": {"united kingdom", "england", "scotland", "wales", "northern ireland", "great britain"}, "scene": {"british"}},
    "scotland_now":    {"places": {"scotland"},  "scene": {"scotland"}},
    "london_now":      {"places": {"london"},    "scene": {"london"}},
    "uk_now":          {"places": {"united kingdom", "england", "scotland", "wales", "northern ireland", "great britain"}, "scene": {"british"}},
    "australia_now":   {"places": {"australia"}, "scene": {"australia"}},
}
# Geo HITS mixes — same origin gate as the scenes above, but selected like the decade mixes (top-N by global
# Last.fm listeners) instead of the scenes' equal-weight artist rotation. Routed via _is_geo_hits in _build_mix_tracks.
_GEO_HITS_PROFILES = {"scottish_hits", "australian_hits", "london_hits", "uk_hits"}

# City STYLE mixes (glasgow_/london_/melbourne_*): a TIERED, COUNTRY-BOUNDED geo gate (replaces the old soft
# origin boost). Selection fills from the strictest tier first and tops up outward — city, then region, then
# the whole nation — and DROPS anything from outside the country (or with unknown origin): a HARD bound, no
# full-pool fallback. Each tier is an _origin_match spec (matched against the artist's consolidated `places`,
# with a Last.fm scene-tag fallback on the city/region tiers). Tier 3 = the whole nation; the UK set names
# the constituent countries too, since ~15% of UK artists carry a UK place name but no GB code. The pinned
# showcase scenes (_PROFILE_GEO_GATE) are excluded — london_scene shares the london_ prefix but keeps its own
# single hard gate + popularity rotation. This changes SELECTION only; _dj_order still orders picks sonically.
_UK_PLACES = {"united kingdom", "england", "scotland", "wales", "northern ireland"}
_GEO_TIERS_BY_CITY = {
    "glasgow":   [{"places": {"glasgow"},   "scene": {"glasgow"}},
                  {"places": {"scotland"},  "scene": {"scotland"}},
                  {"places": _UK_PLACES}],
    "london":    [{"places": {"london"},    "scene": {"london"}},
                  {"places": {"england"},   "scene": {"england"}},
                  {"places": _UK_PLACES}],
    "melbourne": [{"places": {"melbourne"}, "scene": {"melbourne"}},
                  {"places": {"victoria"},  "scene": {"victoria"}},
                  {"places": {"australia"}}],
}
_PROFILE_GEO_TIERS = {}
for _pk in _MOOD_PROFILES:
    if _pk in _PROFILE_GEO_GATE:          # never touch the pinned showcase scenes (london_scene shares prefix)
        continue
    for _pre, _city in (("glasgow_", "glasgow"), ("london_", "london"), ("melbourne_", "melbourne")):
        if _pk.startswith(_pre):
            _PROFILE_GEO_TIERS[_pk] = _GEO_TIERS_BY_CITY[_city]
            break


def _origin_match(entry, spec):
    """True if the track's artist is from one of the spec's places. MusicBrainz files the same place
    under inconsistent headings (area / region / country / begin_area / city), so we consolidate EVERY
    place field into ONE lowercased `places` set and match the spec's place names against that — the
    spec doesn't care which heading MB used. spec keys:
      `places` — set of lowercase place names (matched against the artist's consolidated places);
      `scene`  — optional set of scene tags (Last.fm fallback: catches artists tagged with a scene
                 whose MB origin is only a birthplace, e.g. someone tagged "glasgow")."""
    o = entry.get("artist_origin") or {}
    places = {p.lower() for p in (o.get("places") or []) if isinstance(p, str)}
    for field in ("begin_area", "area", "city", "region", "country", "subdivisions"):
        v = o.get(field)
        if isinstance(v, str) and v:
            places.add(v.lower())
        elif isinstance(v, (list, tuple)):
            places.update(x.lower() for x in v if isinstance(x, str))
    if {p.lower() for p in spec.get("places", ())} & places:
        return True
    scene = spec.get("scene")
    if scene:
        lf = list(entry.get("lastfm_artist_tags") or {}) + list(entry.get("lastfm_track_tags") or {})
        if any(s in t for s in (x.lower() for x in scene) for t in lf):
            return True
    return False


def _origin_boost(entry, profile_key):
    """Strongly favour local artists in a city/region mix; gently push out confirmed non-local
    ones. Tracks with no origin data yet are neutral (no-op), so it switches on as the geo syncs."""
    spec = _PROFILE_ORIGIN.get(profile_key)
    if not spec:
        return 0.0
    if _origin_match(entry, spec):
        return -_ORIGIN_WEIGHT
    # known origin but not a match -> mild penalty; unknown origin -> neutral
    return _ORIGIN_PENALTY if entry.get("artist_origin") else 0.0


# ---------------------------------------------------------------------------
# Popularity lean (Last.fm global listeners) — hits vs deep cuts
# ---------------------------------------------------------------------------
_POP_WEIGHT = 0.20
# Profile -> lean: >0 wants well-known/hits, <0 wants deep cuts/obscure; the MAGNITUDE scales the nudge
# (genre mixes use +0.5 = a lighter recognisability touch than the +1 hits mixes). Most mixes are neutral
# (no entry). The decade mixes lean to the hits people remember; focus/underground mixes dig deep.
_PROFILE_POPULARITY = {
    # --- Meloday+ gap-fill mixes (pop mixes lean to hits; neoclassical digs deep) ---
    "sad_bangers": 1, "power_ballads": 1, "neoclassical": -1, "swagger": 1,
    "chart_pop": 1, "dance_pop": 1, "indie_pop": 1, "synth_pop": 1,
    "indie_rock": 1, "post_grunge": 1, "rap_rock": 1, "festival_edm": 1,
    # hits — the recognisable, well-known songs (decades, throwbacks, parties, sing-alongs, motivation)
    "decade_60s": 1, "decade_70s": 1, "decade_80s": 1, "decade_90s": 1, "decade_00s": 1,
    "decade_10s": 1, "decade_20s": 1,
    "throwback_anthems": 1, "party_throwback": 1, "memory_lane": 1, "school_days": 1, "old_friends": 1,
    "party": 1, "celebration": 1, "friday_night": 1, "friday_feeling": 1, "pre_party": 1, "singalong": 1,
    "happy": 1, "euphoric": 1, "main_character": 1, "confidence_boost": 1, "monday_motivation": 1,
    "road_trip": 1, "driving_singalong": 1, "driving_mix": 1, "summer_heat": 1, "cookout": 1,
    "workout": 1, "running": 1, "housework_hustle": 1, "wedding_day": 1,
    # deep cuts — discovery / focus / ambient / underground, where obscurity is a feature
    "deep_work": -1, "focus": -1, "study_session": -1, "deep_reading": -1, "creative_flow": -1,
    "ambient_drift": -1, "meditation": -1, "spa_bath": -1, "yoga_stretch": -1, "power_nap": -1,
    "sleep": -1, "lofi_beats": -1, "vaporwave": -1, "downtempo": -1, "post_rock": -1, "starlit": -1,
    "three_am": -1, "witching_hour": -1, "glasgow_underground": -1, "melbourne_techno": -1,
    # recognisability nudge — a LIGHTER (+0.5) lean than the hits mixes, so these genre mixes surface a
    # few familiar anchors while staying mood-led + diverse + discovery-friendly (broad data-reliable set;
    # city/scene, classical/score/instrumental, and deep-cut mixes deliberately left neutral)
    "classic_rock": 0.5, "britpop_rock": 0.5, "heavy_riffs": 0.5, "punk_energy": 0.5, "garage_grunge": 0.5,
    "stoner_rock": 0.5, "emo_poppunk": 0.5, "prog_rock": 0.5, "rockabilly_surf": 0.5, "psych_haze": 0.5,
    "house_party": 0.5, "deep_house": 0.5, "uk_garage": 0.5, "techno": 0.5, "trance": 0.5, "dnb": 0.5,
    "bass_drop": 0.5, "hyperpop": 0.5, "industrial": 0.5, "synthwave": 0.5,
    "motown_soul": 0.5, "neo_soul": 0.5, "funk_disco": 0.5, "after_hours_rnb": 0.5, "acid_jazz": 0.5,
    "boom_bap": 0.5, "g_funk": 0.5, "conscious_flow": 0.5, "trap_mode": 0.5, "gospel": 0.5,
    "bebop": 0.5, "smooth_jazz": 0.5,
    "blues_bar": 0.5, "bluegrass": 0.5, "outlaw_country": 0.5, "country_roads": 0.5,
    "afrobeat": 0.5, "bossa_samba": 0.5, "celtic_folk": 0.5, "latin_heat": 0.5, "reggae_dub": 0.5,
    "ska": 0.5, "swing_bigband": 0.5, "yacht_rock": 0.5, "folk_acoustic": 0.5,
    "festive": 0.5, "autumn_embers": 0.5, "autumn_leaves": 0.5, "spring_jangle": 0.5, "spring_acoustic": 0.5,
    "summer_breeze": 0.5, "summer_tropical": 0.5, "winter_cosy": 0.5, "winter_nights": 0.5,
    "summer_roadtrip": 0.5,  # WHY: highway-singalong recognisability nudge (audit summer_roadtrip-1)
    "acoustic_romance": 0.5, "indie_romance": 0.5, "synthpop_romance": 0.5,
    "dinner_party": 0.5, "campfire": 0.5,
}


# Hard floor on Last.fm listeners — "only songs people actually know" (generalises the decade mixes'
# _ERA_MIN_LISTENERS to any profile). Applied in _build_mix_tracks with a depth fallback so a floor
# never starves a mix; the pop mixes use it to stay recognisable rather than dredging up deep cuts.
_PROFILE_MIN_LISTENERS = {
    "chart_pop": 500_000, "dance_pop": 400_000, "indie_pop": 150_000, "synth_pop": 150_000,
    "indie_rock": 150_000, "post_grunge": 150_000, "rap_rock": 150_000, "festival_edm": 400_000,
    "power_ballads": 100_000, "singalong": 100_000,  # WHY: keep "Lighters Up"/"Singalong" to songs people know (audit power_ballads-1 / singalong-1)
}

# Artist top-tracks gate (Throwback Anthems): keep only an artist's OWN Last.fm top-N tracks that ALSO clear
# the global-listener floor — both must hold (an artist's signature song AND a genuinely famous one).
# WHY: throwback_anthems is acoustic-vibe-ranked (category "nostalgic_throwback") with no popularity floor, so obscure
# tracks that merely fit the vibe were displacing artists' actual hits. The per-artist rank map is built by
# `pre_analyze.py --sync-top-tracks` (assets/lastfm_artist_top_tracks.json: {artist: {norm_title: rank}});
# the floor reads the lastfm_listeners column.
def _load_lastfm_top_tracks():
    try:
        with open(meloday.LASTFM_TOP_TRACKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}

_LASTFM_TOP_TRACKS = _load_lastfm_top_tracks()

def _artist_top_rank(entry):
    """1-based rank of this track within its artist's Last.fm top tracks, or None (unmapped artist / not a top
    track). Matches on norm_text(lastfm_query_title(title)) — the SAME key the map was built with."""
    ranks = _LASTFM_TOP_TRACKS.get(entry.get("artist") or "")
    if not ranks:
        return None
    return ranks.get(norm_text(lastfm_query_title(entry.get("title") or "")))

_PROFILE_ANTHEM_GATE = {
    # 100k floor calibrated from the cache; RE-TUNE after the listener-column fix re-syncs (curly-quote /
    # remaster counts rise, so a few more tracks clear the floor).
    "throwback_anthems": {"top_n": 10, "min_listeners": 100_000},
}


def _popularity_boost(entry, profile_key):
    """Lean a mix toward well-known tracks (lean > 0) or deep cuts (lean < 0); the lean MAGNITUDE
    scales the nudge (0.5 = half strength, a lighter recognisability touch for genre mixes).
    lastfm_listeners is log-scaled (~10k ~ 0.6, a few million ~ 1.0). No-op for neutral / missing data."""
    lean = _PROFILE_POPULARITY.get(profile_key)
    if not lean:
        return 0.0
    listeners = entry.get("lastfm_listeners")
    if not listeners:
        return 0.0
    pop = min(1.0, math.log10(listeners + 1) / 6.5)
    return -_POP_WEIGHT * abs(lean) * (pop if lean > 0 else (1.0 - pop))


# ---------------------------------------------------------------------------
# Listening-hour affinity — nudge toward tracks you usually play around now
# ---------------------------------------------------------------------------
_HOUR_WEIGHT = 0.10
_HOUR_AFFINITY = {}   # rating_key -> 0..1 share of plays near the current hour (set per build)


def _listening_hour_boost(rk):
    """Small nudge toward tracks the user usually plays around this hour (computed from play
    history in build_mood_mixes). No-op for tracks with no such history."""
    return -_HOUR_WEIGHT * _HOUR_AFFINITY.get(rk, 0.0)


# ---------------------------------------------------------------------------
# Lyrics (LRCLIB): theme match (strong, unique) + light sentiment nudge
# ---------------------------------------------------------------------------
_LYRIC_THEME_WEIGHT   = 0.35   # strong pull for lyrically-defined mixes (festive/summer/road)
_LYRIC_VALENCE_WEIGHT = 0.12   # light only — lyric sentiment is noisy and overlaps mood_sad/happy

# Profile -> lyric themes it wants (from the sync's _LYRIC_THEMES vocabulary).
_PROFILE_LYRIC_THEMES = {
    # acoustic_romance — within the Jack-Johnson seed-centroid sound, lift the genuine love songs.
    "acoustic_romance": ["romantic", "tender", "affectionate", "intimate", "devotional", "romantic_tenderness",
                         "warm_affection", "tender_devotion", "heartfelt", "warmhearted", "romantic_warmth",
                         "tender_reassurance", "romantic_longing", "dreamy_romantic", "warm_tenderness",
                         "romantic_possibility", "mutual_affection", "romantic_euphoria"],
    # --- Meloday+ gap-fill mixes (neoclassical + the pop mixes omitted — not lyric-anchored) ---
    "situationship": ["yearning", "restless", "anxious", "conflicted", "uncertain", "bittersweet", "vulnerable", "unresolved_attachment", "relationship_limbo", "mixed_signals", "push_pull_dynamics", "drifting_apart", "emotional_distance", "communication_breakdown", "fear_of_abandonment", "searching_for_intimacy", "missing_someone"],
    "sad_bangers": ["euphoric_sadness", "cathartic", "bittersweet", "melancholic", "dancefloor_catharsis", "sad_banger", "post_breakup_longing", "living_in_the_moment"],
    "power_ballads": ["dramatic", "passionate", "yearning", "triumphant", "overcoming_obstacles", "self_empowerment"],
    "restless": ["restless", "anxious", "tense", "urgent", "frustrated", "brooding", "intrusive_thoughts", "late_night", "survival_mode", "feeling_trapped", "feeling_stuck", "time_running_out", "emotional_overwhelm", "inner_turmoil", "late_night_introspection", "keep_moving_forward"],
    "yacht_rock": ["warm", "relaxed", "carefree", "romantic", "bright", "gentle", "living_in_the_moment", "summer_romance", "mutual_affection"],
    "swagger": ["swaggering", "confident", "bold", "assertive", "cocky", "bragging_rights", "luxury_flex", "status_flex", "wealth_flex", "hustle_mindset"],
    # Rebuilt onto the emergent canonical lyric vocab (assets/lyric_vocab.json, 471 moods + 586
    # themes). Each profile's wanted moods + themes are flattened into one list; the boost pulls a
    # song carrying ANY of them and the exclusion VETO pushes out songs that exclude them. Seasonal/
    # weather mixes map to their emotional CHARACTER. Texture/instrumental mixes (decades, ambient,
    # jazz, meditation, geo) get no entry. Regenerate via build_lyric_vocab + the profile mapper.
    # --- activity ---
    "bluegrass": ["upbeat", "earnest", "resilient", "hopeful", "nostalgic", "playful", "working_class_struggle", "perseverance", "family_responsibility", "overcoming_adversity", "community_support"],
    "blues_bar": ["melancholic", "somber", "sad", "heartbreak", "reflective", "tired", "grief_and_loss", "loneliness", "missing_someone", "trying_to_forget", "crying_in_private", "night_drive"],
    "commute_mix": ["reflective", "thoughtful", "calm", "melancholic", "focused", "restless", "late_night_introspection", "searching_for_direction", "rumination", "feeling_stuck", "keep_moving_forward"],
    "driving_mix": ["focused", "driven", "energized", "confident", "restless", "adventurous", "night_drive", "searching_for_direction", "escapism", "keep_moving_forward"],
    "driving_singalong": ["energetic", "upbeat", "cheerful", "confident", "anthemic", "adventurous", "living_in_the_moment", "weekend_escape", "collective_momentum", "music_as_escape", "party_energy"],
    "night_drive": ["late_night_introspection", "hypnotic", "reflective", "brooding", "dreamlike", "night_drive", "late_night", "searching_for_meaning", "memory_loop"],
    "outlaw_country": ["defiant", "gritty", "resigned", "reflective", "rebellious", "wry", "haunted", "anti_authoritarian", "burning_bridges", "running_from_consequences", "regret", "standing_your_ground", "working_class_struggle"],
    "road_trip": ["adventurous", "restless", "hopeful", "reflective", "energetic", "searching_for_direction", "escapism", "leaving_the_past_behind", "weekend_escape"],
    # --- atmospheric ---
    "after_dark": ["mysterious", "intimate", "brooding", "dreamlike", "late_night_introspection", "seductive", "late_night", "dreamlike_imagery", "romantic_limbo", "seeking_connection", "unspoken_desire"],
    "main_character": ["confident", "bold", "mysterious", "playful", "anthemic", "identity_as_performance", "self_mythologizing", "status_flex", "living_in_the_moment", "performative_identity"],
    "stormy": ["tense", "brooding", "uneasy", "dramatic", "dark", "volatile", "relationship_conflict", "warning_signs_ignored", "inner_turmoil", "dread", "emotional_overwhelm"],
    # --- calm ---
    "candlelight": ["tender", "intimate", "calm", "warm", "serene", "reflective", "emotional_safety", "late_night_intimacy", "mutual_support", "unconditional_support", "healing"],
    "romantic_jazz": ["romantic", "tender", "intimate", "warm", "reverent", "late_night_intimacy", "mutual_affection", "invitation_to_intimacy", "romantic_devotion", "enduring_love"],
    # --- emotional ---
    "bittersweet": ["bittersweet", "nostalgic", "wistful", "tender", "reflective", "sad", "hopeful", "moving_on", "nostalgia_for_the_past", "post_breakup_longing", "letting_go", "healing", "enduring_love"],
    "defiant": ["defiant", "rebellious", "unyielding", "angry", "assertive", "bold", "defiance_against_authority", "rebellion_against_authority", "stand_your_ground", "anti_conformity", "anti_authoritarian", "reclaiming_agency"],
    "grey_skies": ["reflective", "wistful", "melancholic", "somber", "calm", "introspective", "nature_as_emotional_mirror", "nostalgia_for_the_past", "loneliness", "searching_for_closure"],
    "grief_release": ["grief_heavy", "mourning", "reflective", "somber", "tender", "cathartic", "accepting", "grief_and_loss", "grief_and_mourning", "letting_go", "healing", "anticipatory_grief"],
    "heartbreak": ["heartbroken", "grief_stricken", "sad", "aching", "lonely", "devastated", "heartbreak", "lost_love", "breaking_point", "cant_move_on", "post_breakup_longing", "emotional_abandonment"],
    "hopeful": ["hopeful", "optimistic", "encouraging", "warm", "earnest", "bright", "second_chance", "healing", "new_romance", "personal_growth", "overcoming_obstacles"],
    "melancholy": ["melancholic", "wistful", "sad", "reflective", "somber", "lonely", "brooding", "nostalgia_for_the_past", "missing_someone", "loneliness", "rumination", "loss_of_direction"],
    "moody_mix": ["brooding", "reflective", "melancholic", "tender", "guarded", "uncertain", "late_night_introspection", "inner_conflict", "rumination", "emotional_distance", "identity_questioning"],
    "nostalgia_mix": ["nostalgic", "wistful", "reflective", "sentimental", "melancholic", "haunted_by_memory", "nostalgia_for_the_past", "haunting_memories", "memory_loop", "missing_someone", "trying_to_forget"],
    "rainy_day": ["melancholic", "wistful", "reflective", "somber", "gentle", "introspective", "late_night_introspection", "missing_someone", "nostalgia_for_the_past", "heartbreak", "grief_and_loss"],
    "tender": ["tender", "gentle", "warm", "comforting", "affectionate", "intimate", "emotional_safety", "mutual_support", "unconditional_support", "healing", "reassurance_seeking"],
    "triumphant": ["triumphant", "confident", "empowered", "uplifted", "resilient", "bold", "overcoming_obstacles", "self_empowerment", "personal_growth", "prove_them_wrong", "reclaiming_agency"],
    "vulnerable": ["vulnerable", "earnest", "confessional", "anxious", "tender", "vulnerability", "seeking_reassurance", "asking_for_help", "emotional_accountability", "unspoken_truths", "relationship_boundaries"],
    "yearning": ["yearning", "wistful", "aching", "vulnerable", "reflective", "melancholic", "unresolved_longing", "missing_someone", "post_separation_longing", "unrequited_love", "waiting_for_return"],
    # --- energy ---
    "angst_mix": ["angst", "anxious", "restless", "conflicted", "brooding", "overthinking", "hurt", "inner_conflict", "relationship_conflict", "self_doubt"],
    "cathartic": ["cathartic", "intense", "exhilarated", "agitated", "healing", "inner_turmoil", "breaking_point"],
    "celebration": ["celebratory", "joyful", "festive", "exuberant", "grateful", "bright", "uplifted", "party_energy", "collective_momentum", "community_solidarity", "personal_growth"],
    "confidence_boost": ["confident", "swagger", "bold", "assertive", "playful_confident", "empowered", "self_belief", "self_empowerment", "bragging_rights", "status_flex", "reclaiming_agency", "standing_your_ground"],
    "emo_poppunk": ["angst", "defiant", "angry", "hurt", "restless", "intense", "coming_of_age", "relationship_breakdown", "self_doubt", "burning_bridges", "identity_conflict"],
    "empowering": ["empowered", "inspiring", "confident", "resilient", "motivational", "uplifted", "self_empowerment", "self_belief", "reclaiming_agency", "overcoming_doubt", "perseverance", "personal_transformation"],
    "euphoric": ["euphoric", "exuberant", "joyful", "ecstatic", "uplifted", "bright", "triumphant", "living_in_the_moment", "club_energy", "dancefloor_catharsis", "collective_momentum", "mutual_affection"],
    "friday_night": ["hyped", "carefree_energy", "playful", "flirty", "energetic", "celebratory", "party_energy", "club_energy", "weekend_escape", "hookup_energy", "flirtation", "living_in_the_moment"],
    "garage_grunge": ["angst", "brooding", "gritty", "angry", "alienated", "defiant", "inner_turmoil", "emotional_dysregulation", "social_isolation", "identity_conflict", "self_destruction", "falling_apart", "rebellion_against_authority"],
    "london_grime": ["intense", "aggressive", "defiant", "urgent", "bold", "edgy", "competitive", "stand_your_ground", "conflict_escalation", "survival_mindset", "power_struggle", "reclaiming_agency", "anti_authoritarian"],
    "party": ["party_energy", "celebratory", "hyped", "playful", "carefree", "energetic", "club_energy", "living_in_the_moment", "collective_momentum", "hookup_energy", "weekend_escape"],
    "party_throwback": ["nostalgic", "cheerful", "festive", "energetic", "playful", "excited", "nostalgia_for_the_past", "party_energy", "collective_momentum", "memory_loop", "living_in_the_moment"],
    "pre_party": ["anticipatory", "excited", "energized", "playful", "flirty", "hyped", "party_energy", "club_energy", "night_drive", "flirtation", "living_in_the_moment"],
    "punk_energy": ["rebellious", "defiant", "aggressive", "angry", "high_energy", "bold", "anti_conformity", "defiance_against_authority", "rebellion_against_authority", "stand_your_ground", "burning_bridges", "conflict_escalation", "collective_momentum"],
    "running": ["adrenaline_rush", "driven", "energized", "focused", "restless", "breathless", "action_call", "keep_moving_forward", "perseverance", "escape_attempt", "time_pressure", "survival_mindset"],
    "trap_mode": ["ambitious", "cocky", "swaggering_confidence", "driven", "assertive", "aggressive", "tense", "status_flex", "wealth_flex", "hustle_mindset", "power_dynamics", "survival_mindset", "reclaiming_agency", "luxury_flex"],
    "workout": ["high_energy", "driven", "aggressive", "adrenaline_driven", "focused", "action_call", "perseverance", "survival_mindset", "collective_momentum"],
    # --- global ---
    "afrobeat": ["joyful", "energetic", "festive", "playful", "upbeat", "warm", "celebratory", "party_energy", "collective_momentum", "community_solidarity", "living_in_the_moment", "dancefloor_catharsis"],
    "bossa_samba": ["playful", "carefree", "romantic", "flirtatious", "bright", "cheerful", "sensual", "flirtation", "mutual_attraction", "summer_romance", "living_in_the_moment", "desire"],
    "celtic_folk": ["nostalgic", "wistful", "reflective", "gentle", "earnest", "warm", "mournful", "homesickness", "nostalgia_for_the_past", "searching_for_home", "nature_as_emotional_mirror", "grief_and_loss", "family_responsibility", "coming_of_age"],
    "latin_heat": ["passionate", "flirtatious", "playful", "sensual", "energetic", "warm", "carefree_energy", "flirtation", "sexual_tension", "mutual_attraction", "party_energy", "living_in_the_moment"],
    "reggae_dub": ["peaceful", "relaxed", "hopeful", "warm", "grounded", "reassuring", "liberated", "community_solidarity", "community_support", "mutual_support", "healing", "spiritual_reassurance"],
    # --- groove ---
    "after_hours_rnb": ["intimate", "seductive", "relaxed", "confident", "late_night", "invitation_to_intimacy", "mutual_desire", "romantic_pursuit"],
    "boom_bap": ["thoughtful", "observant", "confident", "grounded", "critical", "self_reflective", "serious", "social_commentary", "social_critique", "self_reflection", "identity_questioning", "hustle_mindset", "prove_them_wrong", "moral_accountability"],
    "conscious_flow": ["thoughtful", "reflective", "calm", "meditative", "self_reflective", "serene", "self_reflection", "existential_reflection", "searching_for_meaning", "inner_conflict", "healing"],
    "funk_disco": ["playful", "cheerful", "energetic", "confident", "bright", "festive", "party_energy", "club_energy", "dancefloor_catharsis", "living_in_the_moment", "collective_momentum"],
    "g_funk": ["playful", "confident", "swaggering", "carefree", "seductive", "upbeat", "party_energy", "club_energy", "sexual_playfulness", "flirtation", "luxury_flex", "night_drive", "mutual_attraction"],
    "glasgow_house": ["energetic", "rowdy", "cheerful", "uplifted", "gritty", "bold", "party_energy", "club_energy", "collective_momentum", "community_solidarity", "dancefloor_catharsis"],
    "glasgow_soul": ["heartfelt", "brooding", "vulnerable", "gritty", "reflective", "melancholic", "resilient", "working_class_struggle", "heartbreak", "emotional_resilience", "personal_growth", "self_reckoning"],
    "gospel": ["reverent", "hopeful", "uplifting", "devotional", "comforting", "transcendent", "spiritual_guidance", "spiritual_reassurance", "spiritual_renewal", "divine_guidance", "healing", "community_support"],
    "house_party": ["festive", "cheerful", "playful", "energetic", "rowdy", "confident", "party_energy", "club_energy", "collective_momentum", "community_solidarity", "living_in_the_moment"],
    "london_garage": ["energetic", "confident", "playful", "hypnotic", "restless", "hyped", "party_energy", "club_energy", "late_night", "dancefloor_catharsis", "living_in_the_moment"],
    "london_soul": ["reflective", "intimate", "earnest", "wistful", "confident", "romantic_uncertainty", "identity_search", "love_with_conditions", "self_reflection"],
    "melbourne_hiphop": ["thoughtful", "confident", "motivated", "reflective", "gritty", "self_aware", "ambitious", "self_determination", "identity_search", "personal_growth", "hustle_mindset", "social_commentary", "overcoming_adversity", "reclaiming_agency"],
    "melbourne_soul": ["warm", "hopeful", "reflective", "gentle", "content", "intimate", "mutual_support", "healing", "romantic_possibility", "self_discovery", "weekend_escape"],
    "motown_soul": ["joyful", "upbeat", "energetic", "romantic", "celebratory", "bright", "warm", "mutual_affection", "falling_in_love", "romantic_promise", "love_at_first_sight", "community_solidarity"],
    "neo_soul": ["intimate", "warm", "confident", "reflective", "sensual", "calm", "late_night_intimacy", "romantic_invitation", "mutual_affection", "music_as_escape"],
    "uk_garage": ["energetic", "driven", "playful", "hypnotic", "restless", "party_energy", "club_energy", "dancefloor_catharsis", "living_in_the_moment", "late_night"],
    # --- occasion ---
    "friday_feeling": ["relaxed", "upbeat", "anticipatory", "carefree", "cheerful", "liberated", "playful", "weekend_escape", "party_energy", "living_in_the_moment", "collective_momentum"],
    "midweek_reset": ["calm", "reflective", "grounded", "reassuring", "hopeful", "relaxed", "starting_over", "self_reflection", "healing", "seeking_clarity"],
    "monday_motivation": ["motivated", "driven", "focused", "energetic", "optimistic", "determined", "hustle_mindset", "action_call", "perseverance", "starting_over", "personal_growth", "keep_moving_forward"],
    "treat_yourself": ["carefree", "joyful", "celebratory", "playful", "relaxed", "bright", "living_in_the_moment", "weekend_escape", "luxury_flex", "club_energy"],
    # --- romantic ---
    "acoustic_romance": ["romantic", "intimate", "tender", "earnest", "heartfelt", "gentle", "falling_in_love", "unspoken_feelings", "mutual_affection", "romantic_reassurance", "enduring_love"],
    "crush": ["infatuated", "flirty", "dreamy", "eager", "nervous", "hopeful", "infatuation", "instant_attraction", "unspoken_feelings", "mutual_attraction", "romantic_possibility"],
    "date_night": ["romantic", "flirty", "playful", "confident", "affectionate", "sensual", "flirtation", "mutual_attraction", "invitation_to_intimacy", "early_stage_romance", "romantic_persuasion"],
    "devotion": ["devoted", "devotional", "tender", "reverent", "earnest", "warm", "steadfast", "devotion", "enduring_love", "mutual_devotion", "romantic_devotion", "unconditional_love"],
    "first_date": ["flirtatious", "excited", "playful", "hopeful", "curious", "anticipatory"],
    "flirty": ["flirty", "playful", "cheeky", "confident", "mischievous", "seductive", "flirtation", "mutual_attraction", "sexual_playfulness", "invitation_to_intimacy"],
    "indie_romance": ["intimate", "wistful", "dreamy", "tender", "reflective", "romantic", "vulnerable", "falling_in_love", "romantic_pining", "mutual_affection", "unspoken_feelings", "romantic_uncertainty"],
    "late_night_romance": ["romantic", "intimate", "tender", "dreamy", "reflective", "late_night_intimacy", "unspoken_feelings", "yearning", "post_separation_longing", "romantic_pining"],
    "long_distance": ["yearning", "hopeful_longing", "wistful", "vulnerable", "lonely", "reflective", "uncertain", "long_term_commitment", "missing_someone", "post_separation_longing", "waiting_for_return", "seeking_reassurance"],
    "love_songs": ["romantic", "affectionate", "tender", "devoted", "heartfelt", "passionate", "falling_in_love", "mutual_affection", "all_consuming_love", "enduring_love", "romantic_devotion"],
    "loved_up": ["euphoric", "joyful", "affectionate", "romantic", "bright", "playful", "content", "mutual_affection", "mutual_desire", "falling_in_love", "all_consuming_love", "new_romance"],
    "modern_romance": ["romantic", "playful", "confident", "flirty", "hopeful", "curious", "instant_connection", "mixed_signals", "seeking_connection", "romantic_possibility"],
    "moving_on": ["resigned", "reflective", "wistful", "hopeful", "relieved", "moving_on", "letting_go", "healing", "self_reclamation", "post_breakup_longing", "trying_to_hold_on", "leaving_the_past_behind"],
    "romantic_dinner": ["romantic", "tender", "warm", "affectionate", "intimate", "reverent", "mutual_affection", "invitation_to_intimacy", "romantic_reassurance", "enduring_love", "intimacy_as_escape"],
    "romantic_mix": ["romantic", "affectionate", "tender", "passionate", "warm", "devoted", "falling_in_love", "mutual_affection", "mutual_desire", "all_consuming_love", "romantic_possibility"],
    "slow_burn": ["romantic", "anticipatory", "tender", "yearning", "intimate", "romantic_pining", "unspoken_desire", "mutual_attraction", "pursuit_of_intimacy"],
    "slow_dance": ["romantic", "tender", "intimate", "gentle", "devoted", "falling_in_love", "mutual_affection", "invitation_to_intimacy", "romantic_devotion", "all_consuming_love"],
    "synthpop_romance": ["romantic", "dreamy", "euphoric", "playful", "electric", "flirty", "falling_in_love", "instant_connection", "romantic_escape", "mutual_attraction", "love_at_first_sight"],
    "wedding_day": ["devoted", "joyful", "romantic", "celebratory", "grateful", "warm", "content", "long_term_commitment", "promise_of_forever", "mutual_devotion", "unconditional_love"],
    # --- seasonal ---
    "autumn_embers": ["brooding", "reflective", "warm", "tender", "nostalgic", "dark_tenderness", "haunting_memories", "unresolved_longing", "letting_go", "emotional_resilience"],
    "autumn_leaves": ["reflective", "nostalgic", "wistful", "gentle", "melancholic", "thoughtful", "letting_go_of_the_past", "nostalgia_for_the_past", "moving_on", "haunting_memory"],
    "autumn_rain": ["wistful", "melancholic", "reflective", "nostalgic", "somber", "gentle", "nostalgia_for_the_past", "haunting_memory", "missing_someone", "grief_and_loss", "late_night_introspection"],
    "festive": ["festive", "joyful", "celebratory", "upbeat", "cheerful", "playful", "christmas_spirit", "community_solidarity", "party_energy", "family_responsibility", "mutual_support"],
    "spring_bloom": ["hopeful", "uplifted", "bright", "optimistic", "gentle", "starting_over", "personal_growth", "healing", "new_romance"],
    "summer_breeze": ["carefree", "relaxed", "warm", "bright", "lighthearted", "peaceful", "living_in_the_moment", "weekend_escape", "romantic_possibility", "mutual_affection", "escapism"],
    "summer_heat": ["carefree_energy", "euphoric", "playful", "flirtatious", "sensual", "bright", "summer_romance", "hookup_energy", "flirtation", "living_in_the_moment", "party_energy"],
    "summer_roadtrip": ["adventurous", "carefree", "energized", "reflective", "hopeful", "bright", "escapism", "leaving_the_past_behind", "searching_for_direction"],
    "summer_tropical": ["euphoric", "carefree", "playful", "sensual", "exuberant", "bright", "summer_romance", "escapism", "party_energy", "living_in_the_moment"],
    "winter_cosy": ["cozy", "warm", "comforted", "gentle", "content", "secure", "mutual_support", "emotional_safety", "unconditional_love", "healing"],
    "winter_nights": ["reflective", "intimate", "tender", "nostalgic", "melancholic", "cozy", "late_night_intimacy", "missing_someone", "nostalgia_for_the_past", "homesickness"],
    # --- social ---
    "campfire": ["cozy", "intimate", "warm", "reflective", "gentle", "nostalgic", "community_solidarity", "searching_for_meaning"],
    "cookout": ["festive", "cheerful", "playful", "warm", "upbeat", "community_solidarity", "party_energy", "family_responsibility", "mutual_support", "call_and_response"],
    "memory_lane": ["nostalgic", "wistful", "reflective", "sentimental", "melancholic", "tender", "nostalgia_for_the_past", "haunting_memories", "memory_loop", "missing_someone", "searching_for_home"],
    "old_friends": ["nostalgic", "warm", "affectionate", "sentimental", "grateful", "mutual_support", "nostalgia_for_the_past", "community_solidarity", "enduring_love"],
    "school_days": ["nostalgic", "reflective", "sentimental", "wistful", "playful", "earnest", "coming_of_age", "nostalgia_for_the_past", "memory_loop", "identity_search"],
    "throwback_anthems": ["anthemic", "nostalgic", "confident", "uplifted", "celebratory", "energetic", "nostalgia_for_the_past", "collective_momentum", "party_energy", "bragging_rights", "living_in_the_moment"],
    # --- upbeat ---
    "beach_vibes": ["carefree", "relaxed", "bright", "playful", "warm", "upbeat", "escapism", "weekend_escape", "living_in_the_moment", "romantic_possibility", "party_energy"],
    "country_roads": ["adventurous", "nostalgic", "warm", "peaceful", "reflective", "nature_as_emotional_mirror", "weekend_escape", "homesickness", "searching_for_home", "living_in_the_moment"],
    "fresh_start": ["optimistic", "uplifted", "motivated", "hopeful", "energized", "resilient", "starting_over", "self_reclamation", "personal_growth", "leaving_the_past_behind", "second_chance"],
    "morning": ["bright", "upbeat", "cheerful", "calm", "optimistic", "warm", "starting_over", "living_in_the_moment", "self_reassurance"],
    "summer_evening": ["warm", "romantic", "reflective", "relaxed", "gentle", "wistful", "summer_romance", "late_night_intimacy", "mutual_affection", "nostalgia_for_the_past", "romantic_possibility"],
    "sunny": ["bright", "cheerful", "uplifted", "hopeful", "warm", "optimistic", "light_in_the_dark", "personal_growth", "overcoming_adversity", "mutual_support", "living_in_the_moment"],
    "weekend_mix": ["carefree", "upbeat", "relaxed", "cheerful", "playful", "liberated", "bright", "weekend_escape", "living_in_the_moment", "party_energy", "music_as_escape", "collective_momentum"],
}
# Profile -> desired lyric sentiment (+1 positive lyrics, -1 sad lyrics). Light nudge only.
_PROFILE_LYRIC_VALENCE = {
    # --- Meloday+ gap-fill mixes ---
    "situationship": -1, "sad_bangers": -1, "restless": -1, "yacht_rock": 1, "swagger": 1,
    "heartbreak": -1, "grief_release": -1, "melancholy": -1, "angst_mix": -1, "moody_mix": -1,
    "happy": 1, "euphoric": 1, "sunny": 1, "celebration": 1, "confidence_boost": 1,
}


# Profiles annotated "lyric_lang=en gate" in their _MOOD_PROFILES comment — English-leaning mixes where a
# foreign-language vocal breaks the vibe. From those 116 Enhance: notes MINUS 5 language-native genres
# (latin_heat / bossa_samba / afrobeat / reggae_dub / celtic_folk) whose en-lean would wrongly demote the
# Spanish / Portuguese / African / Gaelic vocals that DEFINE the genre → 111 keys.
_LYRIC_EN_PROFILES = {
    "situationship", "sad_bangers", "power_ballads", "restless", "yacht_rock", "swagger", "stormy", "grey_skies",
    "festive", "spring_bloom", "summer_heat", "summer_breeze", "summer_roadtrip", "summer_tropical",
    "autumn_leaves", "autumn_rain", "autumn_embers", "winter_cosy", "winter_nights", "hopeful", "yearning",
    "triumphant", "tender", "defiant", "vulnerable", "grief_release", "monday_motivation", "midweek_reset",
    "friday_feeling", "treat_yourself", "throwback_anthems", "old_friends", "campfire", "cookout", "school_days",
    "memory_lane", "crush", "slow_burn", "moving_on", "loved_up", "long_distance", "flirty", "devotion",
    "wedding_day", "funk_disco", "neo_soul", "motown_soul", "after_hours_rnb", "boom_bap", "conscious_flow",
    "g_funk", "trap_mode", "house_party", "uk_garage", "punk_energy", "garage_grunge", "emo_poppunk", "blues_bar",
    "country_roads", "outlaw_country",
    "bluegrass", "gospel", "glasgow_soul", "glasgow_house", "london_soul", "london_garage", "london_grime",
    "melbourne_soul", "melbourne_hiphop", "party", "melancholy", "morning", "rainy_day", "sunny", "nostalgia_mix",
    "moody_mix", "bittersweet", "cathartic", "confidence_boost", "empowering", "euphoric", "angst_mix",
    "romantic_mix", "fresh_start", "main_character", "after_dark", "friday_night", "weekend_mix", "date_night",
    "driving_mix", "night_drive", "driving_singalong", "road_trip", "commute_mix", "party_throwback", "beach_vibes",
    "summer_evening", "modern_romance", "late_night_romance", "romantic_dinner", "love_songs", "slow_dance",
    "candlelight", "first_date", "romantic_jazz", "acoustic_romance", "indie_romance", "synthpop_romance",
    "heartbreak", "pre_party", "celebration"
}
_LYRIC_LANG_PENALTY = 0.25                       # soft down-rank for a foreign-language vocal in an en mix
_LYRIC_LANG_KEEP    = {"en", "none", "instrumental"}   # English, instrumentals, and unknown/empty are kept

def _lyric_lang_penalty(entry, profile_key):
    """SOFT down-rank (never exclude) for a KNOWN foreign-language vocal track in an English-leaning mix
    (the _LYRIC_EN_PROFILES set). English / instrumental / unknown are untouched. lower score = better,
    so a positive penalty demotes. WHY soft: a strong-fitting foreign track can still surface; we only
    nudge the vibe English (user choice)."""
    if profile_key not in _LYRIC_EN_PROFILES:
        return 0.0
    lang = (entry.get("lyric_lang") or "").lower()
    return _LYRIC_LANG_PENALTY if (lang and lang not in _LYRIC_LANG_KEEP) else 0.0

def _lyric_boost(entry, profile_key):
    """Pull tracks whose lyrics match the mix's wanted moods/themes; PUSH OUT tracks whose lyrics
    actively EXCLUDE them (the veto — e.g. a danceable breakup song kept out of a party mix). The new
    `lyric_themes` is a 3-group dict {moods, themes, excluded_themes} of {tag: weight}; a legacy keyword
    list (pre-backfill) is treated as themes at weight 1.0. No-op for un-mapped profiles / untagged."""
    wanted = _PROFILE_LYRIC_THEMES.get(profile_key)
    if not wanted:
        return 0.0
    boost = 0.0
    lt = entry.get("lyric_themes")
    if isinstance(lt, dict):
        # A POSITIVE match wins: a song carrying a wanted mood/theme belongs even if it also excludes one
        # (genuine party songs exclude breakup themes yet are obviously party). Only veto when there's NO
        # positive match AND the lyrics actively exclude a wanted tag (a sad-banger kept out of party).
        pos = {**(lt.get("moods") or {}), **(lt.get("themes") or {})}
        matched = [pos[t] for t in wanted if t in pos]
        if matched:
            boost -= _LYRIC_THEME_WEIGHT * max(matched)            # pull: core fires fully, faint a quarter
        elif any(t in (lt.get("excluded_themes") or {}) for t in wanted):
            boost += _LYRIC_THEME_WEIGHT * 1.5                     # excludes a wanted tag, no positive -> push OUT
    elif isinstance(lt, list):                                     # legacy keyword list (pre-backfill)
        if any(t in lt for t in wanted):
            boost -= _LYRIC_THEME_WEIGHT
    # lyric sentiment lean — legacy VADER; lv is None for batch-tagged tracks, so this is a no-op there
    # (the happy/sad valence profiles re-map to mood tags in the Phase D mapping rebuild).
    lean = _PROFILE_LYRIC_VALENCE.get(profile_key)
    lv = entry.get("lyric_valence")
    if lean and lv is not None:
        boost -= _LYRIC_VALENCE_WEIGHT * (lv if lean > 0 else (1.0 - lv))
    return boost


# ---------------------------------------------------------------------------
# Mood Mix Context Helpers — time-of-day and weather
# ---------------------------------------------------------------------------

def _get_active_hour():
    """Return the current hour (0–23) in the user's active timezone.
    Uses the travel timezone if a trip window is active, otherwise local time.
    Mirrors the same logic as meloday_cron.py."""
    from zoneinfo import ZoneInfo
    travel = config.get("travel", [])
    for trip in travel:
        try:
            tz = ZoneInfo(trip["timezone"])
            dest_today = datetime.now(tz=tz).date()
            start = date.fromisoformat(trip["start"])
            end   = date.fromisoformat(trip["end"])
            if start <= dest_today <= end:
                return datetime.now(tz=tz).hour
        except Exception:
            pass
    return datetime.now().hour


def _in_time_window(hour, window):
    """True if `hour` (0–23) falls within (start, end). If start > end the window wraps past midnight."""
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _get_active_weekday():
    """Current weekday (0=Mon … 6=Sun) in the user's active timezone (travel-aware), mirroring
    _get_active_hour — so day-of-week stays consistent with the active hour on a trip."""
    from zoneinfo import ZoneInfo
    travel = config.get("travel", [])
    for trip in travel:
        try:
            tz = ZoneInfo(trip["timezone"])
            dest_today = datetime.now(tz=tz).date()
            start = date.fromisoformat(trip["start"])
            end   = date.fromisoformat(trip["end"])
            if start <= dest_today <= end:
                return datetime.now(tz=tz).weekday()
        except Exception:
            pass
    return datetime.now().weekday()


def _get_active_date():
    """Current date in the user's active timezone (travel-aware), mirroring _get_active_hour/_get_active_weekday.
    WHY: the slot index cur_gslot mixes this day ordinal with the (travel-aware) hour — they must share a
    timezone, or the day boundary desyncs from the hour during travel and rotation mis-fires/stalls."""
    from zoneinfo import ZoneInfo
    travel = config.get("travel", [])
    for trip in travel:
        try:
            tz = ZoneInfo(trip["timezone"])
            dest_today = datetime.now(tz=tz).date()
            start = date.fromisoformat(trip["start"])
            end   = date.fromisoformat(trip["end"])
            if start <= dest_today <= end:
                return dest_today
        except Exception:
            pass
    return datetime.now().date()


def _is_scheduled(profile_key):
    """True if the profile has a _PROFILE_SCHEDULE (it's a time/activity 'context' mix, not 'anytime')."""
    return profile_key in _PROFILE_SCHEDULE


def _in_schedule(profile_key, hour, weekday):
    """True if `profile_key` has a schedule window matching (hour, weekday). Windows OR together; a
    dayset of None means any day; (start,end) wraps midnight. Unscheduled profiles return False
    (they belong to the rotating 'anytime' tier, not the context tier)."""
    for dayset, start, end in _PROFILE_SCHEDULE.get(profile_key, ()):
        if (dayset is None or weekday in dayset) and _in_time_window(hour, (start, end)):
            return True
    return False


def _get_weather(location):
    """
    Fetch current weather from wttr.in.
    Returns a dict with temp_c, condition, code, and lat (for season detection),
    or None on failure. Location can be a city name, coordinates, or airport code.
    """
    if not location or not _REQUESTS_AVAILABLE:
        return None
    try:
        resp = _requests.get(
            f"https://wttr.in/{location}",
            params={"format": "j1"},
            timeout=8,
            headers={"User-Agent": "meloday-extras/1.0"},
        )
        resp.raise_for_status()
        data    = resp.json()
        current = data["current_condition"][0]
        # latitude in wttr.in JSON is a plain string ("-37.813"), not a nested list
        nearest = data.get("nearest_area", [{}])[0]
        lat_str = nearest.get("latitude", "0") if isinstance(nearest, dict) else "0"
        return {
            "temp_c":    int(current.get("temp_C", 20)),
            "condition": current.get("weatherDesc", [{}])[0].get("value", "").lower(),
            "code":      int(current.get("weatherCode", 113)),
            "wind_kmph": int(current.get("windspeedKmph", 0)),
            "lat":       float(lat_str),
        }
    except Exception as e:
        xlog(f"[WARN] Weather fetch failed for '{location}': {e}")
        return None


def _current_season(lat):
    """
    Return 'summer', 'autumn', 'winter', or 'spring' for the current month,
    adjusted for hemisphere (negative latitude = Southern Hemisphere).
    """
    month = datetime.now().month
    # Northern Hemisphere season by month, then flip for Southern
    if month in (12, 1, 2):
        base = "winter"
    elif month in (3, 4, 5):
        base = "spring"
    elif month in (6, 7, 8):
        base = "summer"
    else:
        base = "autumn"
    if lat >= 0:
        return base
    return {"winter": "summer", "summer": "winter",
            "spring": "autumn",  "autumn": "spring"}[base]


# wttr.in weather codes (subset)
_RAIN_CODES  = {176, 263, 266, 293, 296, 299, 302, 305, 308, 311, 314, 353, 356, 359}
_SNOW_CODES  = {179, 182, 185, 227, 230, 323, 326, 329, 332, 335, 338, 368, 371, 374}
_CLEAR_CODES = {113}                  # sunny/clear
_STORM_CODES = {200, 386, 389, 392, 395}        # thundery
_FOG_CODES   = {143, 248, 260}                  # mist / fog / freezing fog
_CLOUD_CODES = {119, 122}                        # cloudy / overcast (116 = partly cloudy, left mild)

def _weather_boost(profile_key, weather):
    """
    Distance reduction (negative = boost) or penalty (positive = discourage)
    for a profile based on current weather. Thresholds are season-adjusted so
    that a mild Melbourne winter day doesn't trigger Cosy all season, but an
    unusually cold summer day does.
    """
    if weather is None:
        return 0.0

    code   = weather.get("code", 113)
    temp   = weather.get("temp_c", 20)
    season = _current_season(weather.get("lat", 0.0))

    # Season-adjusted "feels cold" and "warm & sunny" thresholds.
    # In Southern Hemisphere winter (Jun–Aug), typical Melbourne days are 10–15°C;
    # Cosy should only trigger on genuinely cold outliers, not every winter day.
    cold_threshold = {"summer": 18, "autumn": 14, "winter": 12, "spring": 15}[season]
    warm_threshold = {"summer": 24, "autumn": 18, "winter": 15, "spring": 20}[season]

    if profile_key == "rainy_day":
        if code in _RAIN_CODES:                            return -0.40
        if code in _CLEAR_CODES and temp > warm_threshold: return +0.20  # sunny → discourage
    elif profile_key == "sunny":
        if code in _CLEAR_CODES and temp > warm_threshold: return -0.40  # warm & clear → boost
        if code in _RAIN_CODES:                            return +0.20  # raining → discourage
    elif profile_key == "cosy":
        if temp < cold_threshold:                          return -0.40  # cold for the season
        if code in _SNOW_CODES:                            return -0.40  # snow (rare in AU)
        if temp > warm_threshold + 5:                      return +0.15  # too warm → discourage
    elif profile_key == "beach_vibes":
        if code in _CLEAR_CODES and temp > warm_threshold + 5: return -0.40  # hot & sunny
        if code in _RAIN_CODES:                               return +0.20
    elif profile_key == "stormy":
        if code in _STORM_CODES:                              return -0.40
    elif profile_key == "foggy":
        if code in _FOG_CODES:                                return -0.40
    elif profile_key == "snow_day":
        if code in _SNOW_CODES:                               return -0.40
    elif profile_key == "heatwave":
        if code in _CLEAR_CODES and temp > warm_threshold + 10: return -0.40  # genuinely hot
    elif profile_key == "frosty":
        if temp <= cold_threshold - 6:                        return -0.40   # genuinely freezing
        if code in _SNOW_CODES:                               return -0.30
    elif profile_key == "grey_skies":
        if code in _CLOUD_CODES:                              return -0.35
    elif profile_key == "windy":
        if weather.get("wind_kmph", 0) > 28:                  return -0.35   # blustery
    elif profile_key == "clear_night":
        hour = datetime.now().hour
        if code in _CLEAR_CODES and (hour >= 21 or hour < 5): return -0.40   # clear & after dark

    return 0.0


def _season_active(profile_key, lat=0.0):
    """
    True if the current calendar season matches the profile's target season.
    Does not require a weather API call — uses local date only.
    Latitude (from weather data when available) adjusts for hemisphere.
    """
    if profile_key == "festive":                       # Christmas/Holiday window, not a season
        return _in_christmas_window(datetime.now())
    season = _current_season(lat)
    return {
        "autumn_mix":     season == "autumn",
        "winter_mix":     season == "winter",
        "spring_mix":     season == "spring",
        "summer_evening": season == "summer",
    }.get(profile_key, False)


# Season each rotating seasonal-style mix belongs to — eligible only when in season, so it
# rotates within the general pool during its season and disappears off-season.
_PROFILE_SEASON = {
    "spring_bloom": "spring", "spring_acoustic": "spring", "spring_strings": "spring", "spring_jangle": "spring",
    "summer_heat": "summer", "summer_breeze": "summer", "summer_roadtrip": "summer", "summer_tropical": "summer",
    "autumn_leaves": "autumn", "autumn_jazz": "autumn", "autumn_rain": "autumn", "autumn_embers": "autumn",
    "winter_frost": "winter", "winter_cosy": "winter", "winter_nights": "winter", "winter_jazz": "winter",
}


def _profile_season_ok(profile_key, lat=0.0):
    """A seasonal-style profile is only eligible in its season; everything else is always eligible."""
    s = _PROFILE_SEASON.get(profile_key)
    return s is None or _current_season(lat) == s


def _mood_rotation_score(profile_key, acoustic_dist, current_hour, weather):
    """
    Hybrid rotation score for mood mix selection. Lower = selected.
    Base is acoustic distance; context (time/weather/soft-time) applies reductions.
    """
    score = acoustic_dist

    # Hard time-of-day boost (time-managed profiles only)
    window = _TIME_BIASED_PROFILES.get(profile_key)
    if window and _in_time_window(current_hour, window):
        score -= 0.35  # strong enough to override pure acoustic fit

    # Soft time boost (general-pool profiles with a time affinity)
    soft = _TIME_SOFT_BOOSTS.get(profile_key)
    if soft:
        start, end, amount = soft
        if _in_time_window(current_hour, (start, end)):
            score -= amount

    # Day-of-week boost (e.g. Friday Night only on Fri/Sat; Sunday Morning only on Sunday)
    weekday_entry = _WEEKDAY_BOOSTS.get(profile_key)
    if weekday_entry:
        days, amount = weekday_entry
        if _get_active_weekday() in days:           # travel-aware, consistent with the schedule gate
            score -= amount

    # Weather boost
    score += _weather_boost(profile_key, weather)

    return score  # can go negative; sort ascending so lowest = most selected


def _daily_mix_description(styles_list):
    """
    Format Daily Mix playlist description — matches Spotify's 'Genre1, Genre2 and more.' format.
    Shows all styles (up to 3) followed by 'and more.' — e.g. 'Indie Rock, Alternative and more.'
    """
    if not styles_list:
        return "A mix made just for you."
    return ", ".join(styles_list) + " and more."


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Meloday Extras — Supplementary Playlists")
    p.add_argument(
        "--playlist", default="all",
        choices=PLAYLIST_IDS + ["all"],
        help="Which playlist to generate (default: all)",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--reselect-moods", action="store_true",
        help="Force full reselection of general mood mixes (runs weekly).",
    )
    p.add_argument(
        "--time-context", action="store_true",
        help="Only add/remove time-of-day mood mixes based on current hour. "
             "Runs at each time boundary (5am, noon, 5pm, 9pm, 10pm, 2am, 4am). "
             "Does not touch general or weather mixes.",
    )
    p.add_argument(
        "--weather-context", action="store_true",
        help="Only add/remove weather mood mixes based on current conditions. "
             "Cheap — run hourly so weather mixes track the weather through the day. "
             "Does not touch general, seasonal, or time mixes.",
    )
    args, _ = p.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Plex playlist CRUD (no cover art — simpler than meloday.py's version)
# ---------------------------------------------------------------------------
def _upsert_extras_playlist(plex, name, tracks, description,
                            cover_key=None, cover_title=None, cover_subtitle=None,
                            cover_tracks=None, existing_playlists=None):
    """
    cover_tracks:        pass the mix's track list to generate a Daily Mix collage cover.
    existing_playlists:  pre-fetched {title: playlist_obj} dict; avoids a full plex.playlists()
                         call per update. Populated in main() and passed through.
    """
    # Single chokepoint for EVERY extras playlist: show the canonical (studio-original) copy of each song.
    # Post-selection, order-preserving — ranking/length unchanged; live/remix/extended keep their own copy.
    tracks = _canonicalize_tracks(plex, tracks, meloday._essentia_cache)
    valid = [t for t in tracks if getattr(t, "ratingKey", None)]
    if not valid:
        xlog(f"[WARN] '{name}': no valid tracks — skipping.")
        return
    try:
        existing = (existing_playlists or {}).get(name) or \
                   next((pl for pl in plex.playlists() if getattr(pl, "title", "") == name), None)
        if existing:
            existing.removeItems(existing.items())
            existing.addItems(valid)
            existing.editTitle(name)
            existing.editSummary(description)
            playlist_obj = existing
        else:
            playlist_obj = plex.createPlaylist(name, items=valid)
            playlist_obj.editSummary(description)
        xlog(f"[OK] '{name}' — {len(valid)} tracks.")

        if cover_key and cover_title:
            if cover_tracks is not None:
                cover_path = _generate_daily_mix_cover(
                    plex, cover_tracks, cover_key, cover_title, cover_subtitle)
            else:
                cover_path = _generate_extras_cover(cover_key, cover_title, cover_subtitle)
            if cover_path:
                playlist_obj.uploadPoster(filepath=cover_path)
                xlog(f"[OK] Cover uploaded: {os.path.basename(cover_path)}")
    except Exception as e:
        xlog(f"[ERROR] Failed to upsert '{name}': {e}")


# ===========================================================================
# Shared Utilities
# ===========================================================================

def fetch_full_history(music, lookback_days=548):
    """Fetch up to lookback_days of play history; normalises viewedAt to UTC-aware."""
    mindate = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
    try:
        entries = list(music.history(mindate=mindate))
    except Exception as e:
        xlog(f"[ERROR] history fetch failed: {e}")
        return []
    for e in entries:
        if e.viewedAt and e.viewedAt.tzinfo is None:
            e.viewedAt = e.viewedAt.replace(tzinfo=timezone.utc)
    return entries


def fetch_history_window(music, start_dt, end_dt, cap=20000):
    """Play history in [start_dt, end_dt] (UTC-aware). plexapi's history() is lower-bound only, so the upper
    bound is added to the raw endpoint (joinArgs renders the `viewedAt>`/`viewedAt<` keys as `>=`/`<=`); a
    Python re-filter guarantees the range even if the server ignores a bound. Empty on failure."""
    from plexapi import utils as _putils
    args = {"viewedAt>": int(start_dt.timestamp()), "viewedAt<": int(end_dt.timestamp()),
            "sort": "viewedAt:asc", "librarySectionID": music.key, "accountID": 1}
    key = "/status/sessions/history/all" + _putils.joinArgs(args)
    try:
        entries = music._server.fetchItems(key, maxresults=cap)
    except Exception as e:
        xlog(f"[WARN] history window {start_dt:%Y-%m-%d}..{end_dt:%Y-%m-%d} fetch failed: {e}")
        return []
    out = []
    for e in entries:
        v = e.viewedAt
        if v is None:
            continue
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        if start_dt <= v <= end_dt:
            out.append(e)
    return out


def _year_play_count(music, year):
    """Cheap total play count for a calendar year via the history endpoint's totalSize (no full fetch)."""
    from plexapi import utils as _putils
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    args = {"viewedAt>": int(start.timestamp()), "viewedAt<": int(end.timestamp()),
            "librarySectionID": music.key, "accountID": 1, "X-Plex-Container-Size": 0}
    key = "/status/sessions/history/all" + _putils.joinArgs(args)
    try:
        return int(music._server.query(key).get("totalSize") or 0)
    except Exception:
        return 0


MAX_TRACK_MS = 15 * 60 * 1000   # exclude tracks longer than 15 minutes from all playlists


def _too_long(track):
    """True if a track exceeds the 15-minute cap (excluded from every playlist)."""
    return (getattr(track, "duration", 0) or 0) > MAX_TRACK_MS


def resolve_tracks_by_keys(plex, rating_keys, workers=16):
    """Parallel fetchItem for a list of ratingKeys. Returns {rk_str: Track}.
    Tracks over the 15-minute cap are dropped here — the single choke-point feeding
    every extras builder (callers already skip rks missing from the map)."""
    keys = list(dict.fromkeys(str(k) for k in rating_keys))  # deduplicate, preserve order
    result = {}

    def _fetch(rk):
        try:
            t = plex.fetchItem(int(rk))
            return rk, t if getattr(t, "type", None) == "track" else None
        except Exception:
            return rk, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch, rk): rk for rk in keys}
        for f in concurrent.futures.as_completed(futures):
            rk, t = f.result()
            if t and not _too_long(t):
                result[rk] = t
    return result


def build_excluded_album_keys(music):
    """Returns set of album ratingKeys that carry any configured exclusion label,
    plus Christmas/seasonal albums when outside their configured window."""
    excluded = set()
    for label in EXCLUDE_LABEL_NAMES:
        try:
            albums = music.search(libtype="album", label=label)
            excluded.update(str(a.ratingKey) for a in albums)
        except Exception:
            pass
    if not _in_christmas_window(datetime.now()):
        try:
            cols = music.collections(title=_CHRISTMAS_COLLECTION)
            if cols:
                xmas_keys = {str(a.ratingKey) for a in cols[0].items()}
                excluded.update(xmas_keys)
                xlog(f"[OK] Excluded {len(xmas_keys)} Christmas albums (outside seasonal window)")
        except Exception as e:
            xlog(f"[WARN] Could not fetch Christmas collection '{_CHRISTMAS_COLLECTION}': {e}")
    return excluded


# ---------------------------------------------------------------------------
# Audio embeddings — "sounds-like" similarity (emb_effnet: 1280-d, mean-pooled, then L2-normalised)
# ---------------------------------------------------------------------------
# emb_effnet is the EffNet-Discogs genre-head embedding computed during analysis (~35% coverage today,
# growing via `pre_analyze.py --sync-embeddings`). It captures sub-style/timbre the 10-d acoustic centroid
# can't. EVERY consumer treats a None vector as "fall back to the acoustic path", so partial coverage and a
# missing numpy degrade gracefully. WHY: emb_effnet/emb_musicnn were collected but unused (audit Tier-4).
_EMB_FIELD  = "emb_effnet"
_EMB_WEIGHT = 0.25          # "sounds-like" cohesion pull in _combined_score (self-cohesion centroid, or a
                            # profile's seed centroid). Tuned: lifts top-50 pairwise sim ~0.83->0.93 (mood) /
                            # 0.71->0.77 (genre); the lift flattens past 0.25, so 0.25 avoids over-dominating
                            # the acoustic/tag terms while keeping most of the cohesion gain.

def _track_emb(entry, which="effnet"):
    """L2-normalised float32 embedding for a track, or None (missing blob / numpy absent / empty).
    `which`: 'effnet' (1280-d Discogs — genre / sub-style) or 'musicnn' (200-d MSD — vibe / sounds-like)."""
    if not _NUMPY_AVAILABLE:
        return None
    blob = entry.get("emb_musicnn" if which == "musicnn" else _EMB_FIELD)
    if not blob:
        return None
    try:
        v = np.frombuffer(blob, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if v.size == 0:
        return None
    n = float(np.linalg.norm(v))
    return (v / n) if n else None

def _emb_cosine(a, b):
    """Cosine similarity of two already-normalised embeddings (a dot product); 0.0 if either is None."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))

def _emb_centroid(rks, essentia_cache, which="effnet"):
    """Mean of the normalised `which` embeddings over a set of ratingKeys, renormalised. None if <5 present."""
    if not _NUMPY_AVAILABLE:
        return None
    vs = [v for v in (_track_emb(essentia_cache.get(str(rk), {}), which) for rk in rks) if v is not None]
    if len(vs) < 5:
        return None
    m = np.mean(vs, axis=0)
    n = float(np.linalg.norm(m))
    return (m / n) if n else None


def compute_listening_centroid(history_entries, essentia_cache, top_n=100):
    """
    Compute acoustic/taste centroid from the top-N most-played tracks.
    Returns dict with bpm/energy/danceability/brightness/year means
    plus styles_counter and genres_counter weighted by play count,
    and an "emb" key — the mean normalised embedding of those tracks (None if numpy/coverage absent).
    """
    play_counts = Counter(str(e.ratingKey) for e in history_entries)
    top_keys = [rk for rk, _ in play_counts.most_common(top_n)]

    # Weighted sums — avoids extending large lists for heavily-played tracks
    wsum  = defaultdict(float)
    wcount = defaultdict(float)
    styles_counter = Counter()
    genres_counter = Counter()
    lastfm_counter = Counter()

    for rk in top_keys:
        entry = essentia_cache.get(rk)
        if not entry:
            continue
        count = play_counts[rk]
        for field in ("bpm", "energy", "danceability", "brightness", "year",
                      "beat_confidence", "onset_rate", "dynamic_complexity",
                      "integrated_loudness", "arousal", "valence", "vocal_presence"):
            val = entry.get(field)
            if val is not None:
                wsum[field]   += val * count
                wcount[field] += count
        for s in (entry.get("styles") or []):
            styles_counter[s] += count
        for g in (entry.get("genres") or []):
            genres_counter[g] += count
        for tag in (entry.get("lastfm_track_tags") or {}):        # community taste tags (Last.fm)
            lastfm_counter[tag] += count
        for tag in (entry.get("lastfm_artist_tags") or {}):
            lastfm_counter[tag] += count * 0.4                    # artist-level tags weighted lighter

    return {
        "bpm":                wsum["bpm"]               / wcount["bpm"]               if wcount["bpm"]               else None,
        "energy":             wsum["energy"]             / wcount["energy"]             if wcount["energy"]             else None,
        "danceability":       wsum["danceability"]       / wcount["danceability"]       if wcount["danceability"]       else None,
        "brightness":         wsum["brightness"]         / wcount["brightness"]         if wcount["brightness"]         else None,
        "year":               wsum["year"]               / wcount["year"]               if wcount["year"]               else None,
        "beat_confidence":    wsum["beat_confidence"]    / wcount["beat_confidence"]    if wcount["beat_confidence"]    else None,
        "onset_rate":         wsum["onset_rate"]         / wcount["onset_rate"]         if wcount["onset_rate"]         else None,
        "dynamic_complexity": wsum["dynamic_complexity"] / wcount["dynamic_complexity"] if wcount["dynamic_complexity"] else None,
        "integrated_loudness":wsum["integrated_loudness"]/ wcount["integrated_loudness"]if wcount["integrated_loudness"]else None,
        "arousal":            wsum["arousal"]            / wcount["arousal"]            if wcount["arousal"]            else None,
        "valence":            wsum["valence"]            / wcount["valence"]            if wcount["valence"]            else None,
        "vocal_presence":     wsum["vocal_presence"]     / wcount["vocal_presence"]     if wcount["vocal_presence"]     else None,
        "styles_counter": styles_counter,
        "genres_counter": genres_counter,
        "lastfm_counter": lastfm_counter,
        "emb": _emb_centroid(top_keys, essentia_cache),   # mean normalised embedding of the top-N played
    }


# Mood-tag → (valence, arousal) lexicon, on Russell's circumplex (valence = pleasant↔
# unpleasant, arousal = activated↔calm). Used as a FALLBACK for the TF-derived valence/
# arousal columns while those are still being generated: when a track has no TF value,
# we estimate it from its mood tags so the per-profile valence/arousal targets in
# _MOOD_PROFILES still discriminate. Real TF values, once present, always take precedence.
# Exact, lower-cased map of every Plex/AllMusic mood in the library (338) → (valence, arousal)
# on Russell's circumplex: valence = pleasant↔unpleasant, arousal = activated↔calm.
_MOOD_AFFECT = {
    "rousing": (0.72, 0.80), "dramatic": (0.45, 0.70), "energetic": (0.65, 0.85),
    "reflective": (0.45, 0.28), "stylish": (0.62, 0.45), "confident": (0.70, 0.62),
    "yearning": (0.40, 0.45), "playful": (0.80, 0.62), "passionate": (0.62, 0.72),
    "atmospheric": (0.45, 0.32), "earnest": (0.55, 0.40), "intense": (0.40, 0.82),
    "theatrical": (0.55, 0.65), "exuberant": (0.85, 0.80), "intimate": (0.62, 0.32),
    "bittersweet": (0.42, 0.38), "fun": (0.82, 0.65), "bright": (0.78, 0.58),
    "amiable/good-natured": (0.78, 0.45), "lively": (0.78, 0.72), "brooding": (0.28, 0.45),
    "sophisticated": (0.58, 0.40), "warm": (0.72, 0.40), "celebratory": (0.85, 0.78),
    "wistful": (0.42, 0.32), "freewheeling": (0.72, 0.62), "hypnotic": (0.45, 0.40),
    "laid-back/mellow": (0.60, 0.28), "ambitious": (0.62, 0.62), "sentimental": (0.52, 0.35),
    "searching": (0.42, 0.42), "romantic": (0.68, 0.42), "nocturnal": (0.40, 0.42),
    "poignant": (0.32, 0.35), "aggressive": (0.25, 0.88), "swaggering": (0.60, 0.65),
    "dreamy": (0.58, 0.25), "urgent": (0.42, 0.82), "tense/anxious": (0.25, 0.72),
    "melancholy": (0.25, 0.30), "brash": (0.52, 0.78), "cathartic": (0.45, 0.72),
    "boisterous": (0.75, 0.82), "slick": (0.60, 0.50), "literate": (0.55, 0.38),
    "quirky": (0.72, 0.55), "gentle": (0.65, 0.25), "exciting": (0.72, 0.82),
    "sensual": (0.62, 0.45), "smooth": (0.62, 0.35), "refined": (0.60, 0.38),
    "summery": (0.80, 0.55), "carefree": (0.82, 0.52), "cheerful": (0.88, 0.60),
    "plaintive": (0.30, 0.30), "bravado": (0.58, 0.70), "sweet": (0.78, 0.45),
    "organic": (0.58, 0.38), "confrontational": (0.30, 0.80), "soothing": (0.65, 0.18),
    "eerie": (0.30, 0.45), "earthy": (0.55, 0.38), "lush": (0.62, 0.40),
    "elegant": (0.60, 0.35), "detached": (0.42, 0.30), "angst-ridden": (0.25, 0.62),
    "relaxed": (0.65, 0.25), "visceral": (0.42, 0.78), "happy": (0.90, 0.62),
    "street-smart": (0.55, 0.58), "whimsical": (0.75, 0.52), "trippy": (0.50, 0.45),
    "sexy": (0.65, 0.55), "volatile": (0.30, 0.78), "rollicking": (0.78, 0.75),
    "cerebral": (0.50, 0.35), "fiery": (0.45, 0.82), "ominous": (0.22, 0.55),
    "pulsing": (0.50, 0.70), "introspective": (0.42, 0.30), "autumnal": (0.45, 0.32),
    "complex": (0.50, 0.45), "somber": (0.25, 0.25), "rambunctious": (0.72, 0.82),
    "light": (0.72, 0.45), "strong": (0.58, 0.62), "provocative": (0.45, 0.62),
    "sparkling": (0.80, 0.62), "witty": (0.72, 0.52), "uplifting": (0.85, 0.62),
    "melodic": (0.65, 0.45), "raucous": (0.55, 0.82), "flowing": (0.58, 0.40),
    "thrilling": (0.72, 0.85), "suspenseful": (0.35, 0.62), "joyous": (0.90, 0.72),
    "elaborate": (0.55, 0.48), "calm/peaceful": (0.65, 0.15), "irreverent": (0.62, 0.60),
    "delicate": (0.60, 0.25), "gritty": (0.40, 0.62), "wry": (0.55, 0.42),
    "rowdy": (0.62, 0.82), "rebellious": (0.40, 0.78), "ethereal": (0.55, 0.25),
    "menacing": (0.22, 0.62), "druggy": (0.45, 0.40), "uncompromising": (0.42, 0.65),
    "gutsy": (0.55, 0.68), "driving": (0.55, 0.80), "soft/quiet": (0.60, 0.18),
    "hedonistic": (0.68, 0.65), "self-conscious": (0.42, 0.40), "animated": (0.75, 0.70),
    "enigmatic": (0.45, 0.40), "innocent": (0.72, 0.40), "bleak": (0.18, 0.30),
    "agreeable": (0.72, 0.42), "spacey": (0.50, 0.25), "sad": (0.18, 0.28),
    "austere": (0.38, 0.28), "nostalgic": (0.48, 0.35), "sexual": (0.60, 0.55),
    "restrained": (0.48, 0.30), "humorous": (0.78, 0.55), "kinetic": (0.62, 0.80),
    "ironic": (0.50, 0.45), "explosive": (0.45, 0.88), "cynical/sarcastic": (0.35, 0.50),
    "powerful": (0.55, 0.78), "epic": (0.58, 0.75), "gloomy": (0.20, 0.30),
    "effervescent": (0.82, 0.65), "shimmering": (0.68, 0.45), "thoughtful": (0.52, 0.32),
    "fierce": (0.38, 0.82), "eccentric": (0.60, 0.52), "clinical": (0.45, 0.32),
    "weary": (0.30, 0.25), "sprawling": (0.50, 0.45), "positive": (0.82, 0.55),
    "anguished/distraught": (0.18, 0.65), "indulgent": (0.58, 0.48), "spooky": (0.28, 0.48),
    "reckless": (0.45, 0.75), "knotty": (0.45, 0.50), "tough": (0.45, 0.65),
    "euphoric": (0.92, 0.82), "spiritual": (0.58, 0.30), "nervous/jittery": (0.30, 0.72),
    "angry": (0.20, 0.85), "fractured": (0.35, 0.55), "bombastic": (0.52, 0.78),
    "bitter": (0.22, 0.50), "flashy": (0.62, 0.62), "gleeful": (0.88, 0.70),
    "brassy": (0.62, 0.70), "paranoid": (0.25, 0.62), "crunchy": (0.50, 0.62),
    "confessional": (0.42, 0.40), "acerbic": (0.35, 0.52), "lyrical": (0.58, 0.40),
    "dark": (0.25, 0.45), "pastoral": (0.58, 0.28), "striding": (0.58, 0.60),
    "airy": (0.65, 0.32), "harsh": (0.28, 0.70), "reserved": (0.48, 0.28),
    "campy": (0.65, 0.58), "fantastic/fantasy-like": (0.60, 0.45), "reverent": (0.55, 0.30),
    "precious": (0.62, 0.38), "serious": (0.42, 0.40), "springlike": (0.78, 0.50),
    "optimistic": (0.82, 0.55), "snide": (0.38, 0.48), "technical": (0.50, 0.45),
    "anthemic": (0.70, 0.75), "graceful": (0.62, 0.32), "brittle": (0.38, 0.40),
    "turbulent": (0.32, 0.72), "wintry": (0.40, 0.28), "narrative": (0.52, 0.40),
    "giddy": (0.85, 0.72), "sleazy": (0.42, 0.52), "sparse": (0.45, 0.25),
    "hostile": (0.20, 0.78), "sugary": (0.82, 0.55), "manic": (0.55, 0.85),
    "triumphant": (0.80, 0.78), "defiant": (0.45, 0.72), "circular": (0.50, 0.40),
    "cold": (0.35, 0.30), "vulnerable": (0.35, 0.35), "tender": (0.65, 0.30),
    "marching": (0.55, 0.62), "ebullient": (0.85, 0.72), "extroverted": (0.75, 0.65),
    "insular": (0.42, 0.30), "swinging": (0.72, 0.62), "outrageous": (0.62, 0.70),
    "unsettling": (0.28, 0.55), "languid": (0.50, 0.20), "mysterious": (0.42, 0.40),
    "cosmopolitan": (0.60, 0.45), "meandering": (0.48, 0.30), "belligerent": (0.25, 0.78),
    "lonely": (0.20, 0.30), "silly": (0.78, 0.58), "ramshackle": (0.52, 0.50),
    "threatening": (0.22, 0.65), "rustic": (0.55, 0.35), "nihilistic": (0.22, 0.50),
    "reassuring/consoling": (0.65, 0.28), "bouncy": (0.80, 0.65), "understated": (0.52, 0.32),
    "grim": (0.20, 0.42), "meditative": (0.55, 0.15), "hungry": (0.48, 0.62),
    "scary": (0.22, 0.62), "devotional": (0.58, 0.32), "ecstatic": (0.93, 0.85),
    "angular": (0.45, 0.55), "capricious": (0.58, 0.55), "majestic": (0.65, 0.62),
    "stately": (0.58, 0.45), "sarcastic": (0.38, 0.48), "declamatory": (0.50, 0.58),
    "spicy": (0.62, 0.62), "trashy": (0.48, 0.58), "naive": (0.65, 0.42),
    "vulgar": (0.42, 0.58), "pure": (0.68, 0.35), "greasy": (0.45, 0.52),
    "lazy": (0.55, 0.22), "opulent": (0.62, 0.45), "exploratory": (0.55, 0.45),
    "mighty": (0.58, 0.72), "messy": (0.45, 0.55), "rhapsodic": (0.62, 0.55),
    "improvisatory": (0.55, 0.48), "philosophical": (0.50, 0.32), "malevolent": (0.18, 0.62),
    "cartoonish": (0.70, 0.62), "outraged": (0.22, 0.80), "sardonic": (0.38, 0.48),
    "dignified/noble": (0.58, 0.42), "sunny": (0.88, 0.58), "narcotic": (0.45, 0.20),
    "virile": (0.55, 0.65), "hyper": (0.62, 0.88), "motoric": (0.52, 0.72),
    "monumental": (0.58, 0.68), "poetic": (0.55, 0.35), "difficult": (0.40, 0.45),
    "suffocating": (0.20, 0.55), "heroic": (0.65, 0.70), "elegiac": (0.30, 0.32),
    "scattered": (0.45, 0.48), "feverish": (0.42, 0.78), "resolute": (0.55, 0.55),
    "severe": (0.30, 0.55), "desperate": (0.20, 0.68), "erotic": (0.60, 0.55),
    "exotic": (0.58, 0.48), "dissonant": (0.35, 0.58), "jovial": (0.85, 0.62),
    "ornate": (0.58, 0.45), "perky": (0.82, 0.65), "seductive": (0.62, 0.50),
    "mechanical": (0.42, 0.48), "child-like": (0.75, 0.50), "magical": (0.70, 0.50),
    "sprightly": (0.80, 0.65), "loose": (0.58, 0.45), "mystical": (0.52, 0.35),
    "apocalyptic": (0.20, 0.70), "regretful": (0.28, 0.35), "macabre": (0.25, 0.50),
    "radiant": (0.82, 0.58), "satirical": (0.50, 0.50), "feral": (0.35, 0.78),
    "athletic": (0.62, 0.78), "concise": (0.52, 0.45), "spontaneous": (0.65, 0.62),
    "benevolent": (0.72, 0.40), "savage": (0.25, 0.82), "transparent/translucent": (0.55, 0.30),
    "spacious": (0.55, 0.28), "heavy": (0.35, 0.65), "patriotic": (0.62, 0.60),
    "arid": (0.40, 0.30), "tight": (0.50, 0.55), "tragic": (0.18, 0.45),
    "negative": (0.22, 0.40), "demonic": (0.15, 0.72), "comic": (0.78, 0.58),
    "martial": (0.48, 0.68), "sacred": (0.58, 0.30), "hymn-like": (0.58, 0.28),
    "quaint": (0.62, 0.35), "energetic yearning": (0.45, 0.62), "funereal": (0.18, 0.25),
    "cute": (0.80, 0.50), "swingin'": (0.72, 0.62), "dreamy brooding": (0.38, 0.35),
    "charming": (0.78, 0.48), "easygoing": (0.68, 0.32), "mannered": (0.52, 0.42),
    "energetic melancholy": (0.35, 0.60), "heavy brooding": (0.28, 0.55),
    "energetic anxious": (0.32, 0.72), "dramatic emotion": (0.45, 0.65), "suave": (0.62, 0.45),
    "sultry": (0.58, 0.45), "euphoric energy": (0.90, 0.85), "evocative": (0.52, 0.40),
    "intriguing": (0.55, 0.45), "edgy": (0.42, 0.62), "aggressive power": (0.28, 0.85),
    "wild": (0.50, 0.82), "depressed": (0.12, 0.25), "solemn": (0.35, 0.30),
    "upbeat pop groove": (0.82, 0.68), "dark pop intensity": (0.35, 0.68),
    "awakening": (0.65, 0.50), "dark urgent": (0.28, 0.72), "fiery groove": (0.55, 0.75),
    "wary": (0.35, 0.50), "energetic abstract groove": (0.55, 0.70),
    "energetic dreamy": (0.55, 0.55), "dreamy pulse": (0.55, 0.45), "other": (0.50, 0.45),
    "idealistic": (0.68, 0.50), "stirring": (0.62, 0.62), "sober": (0.42, 0.32),
    "determined": (0.58, 0.62), "happy excitement": (0.88, 0.78),
}

# Valence/arousal define a "mood", so the emotional axis dominates the acoustic distance
# (texture dims carry weight 1.0). Applies equally to TF and proxy-derived values, so the
# behaviour is correct now and once TF columns are populated.
_VALENCE_WEIGHT = 4.0
_AROUSAL_WEIGHT = 2.0
_VOCAL_WEIGHT   = 1.0   # instrumental↔vocal axis (TF value, or genre/style proxy as fallback)
_BC_DIFF_SCALE  = 3.9   # beat_confidence/dynamic_complexity are RAW Essentia (centroids calibrated to raw
_DC_DIFF_SCALE  = 11.2  # via _REAL_DIST); normalise the DIFF to 0–1 like onset_rate (≈ cache p99−p1)


# --- Target calibration --------------------------------------------------------------------
# The DEAM TF **arousal** clusters tightly mid-scale (μ0.50 σ0.10, ~0.31–0.66), so hand-set
# targets that assumed a full 0–1 spread were off-scale and the arousal axis stopped
# discriminating (every high-energy mix targeted 0.85, unreachable). Z-score-remap arousal onto
# the real distribution — preserving each profile's relative ordering but fitting the reachable
# range. Only arousal is remapped: valence already matches the real spread (rescaling it just
# shifts targets away from where mood-boosted selections land), and vocal_presence is bimodal
# (a z-score overshoots into extremes a blended mix can't reach — handled via _VOCAL_WEIGHT
# instead). Stored targets stay readable as "intent"; this runs once at import.
_REAL_DIST = {   # feature: (mean, sd, clamp_lo≈p1, clamp_hi≈p99), measured from the live cache
    "arousal": (0.50, 0.10, 0.28, 0.70),
    # beat_confidence & dynamic_complexity are stored RAW (Essentia scale) but the profiles author them
    # 0–1; remap the profile targets onto the raw cache distribution so both axes discriminate again.
    "beat_confidence":    (1.70, 1.07, 0.0,  3.73),
    "dynamic_complexity": (4.96, 2.17, 1.80, 13.02),
}
_CALIB = {}   # feature -> (intended_mean, intended_sd, real_mean, real_sd, lo, hi)


def _calibrate_value(feature, v):
    """Map an intended-scale value (a hand-set target or a proxy estimate) onto the real cache
    distribution, so targets, real TF values and proxy fallbacks all share one scale."""
    c = _CALIB.get(feature)
    if c is None or v is None:
        return v
    imu, isd, rmu, rsd, lo, hi = c
    return min(hi, max(lo, rmu + (v - imu) * (rsd / isd)))


def _calibrate_targets():
    """Rescale every profile's valence/arousal/vocal_presence target onto the real distribution
    (runs once at import). Records the per-feature transform in _CALIB so the proxies match."""
    for feature, (rmu, rsd, lo, hi) in _REAL_DIST.items():
        vals = [p[feature] for p in _MOOD_PROFILES.values() if p.get(feature) is not None]
        if not vals:
            continue
        imu = sum(vals) / len(vals)
        isd = (sum((v - imu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
        _CALIB[feature] = (imu, isd, rmu, rsd, lo, hi)
        for p in _MOOD_PROFILES.values():
            if p.get(feature) is not None:
                p[feature] = _calibrate_value(feature, p[feature])


_calibrate_targets()

# Instrumental-intent profiles weight the (now real) vocal axis more heavily, so they lean
# instrumental on the data instead of a hard genre gate — relaxing them to any low-vocal track
# (more variety) while still staying mostly instrumental. Per-profile so the other ~225 mixes,
# which read vocal at the normal _VOCAL_WEIGHT, are untouched.
_INSTRUMENTAL_VOCAL = {"meditation", "spa_bath", "yoga_stretch", "power_nap", "study_session",
                       "deep_reading", "focus", "deep_work", "sleep"}
for _k in _INSTRUMENTAL_VOCAL:
    if _k in _MOOD_PROFILES:
        _MOOD_PROFILES[_k]["vocal_weight"] = 3.0

# Rave Cave leans instrumental for a different reason than the focus/ambient mixes above: its
# hard-dance gate can't tell genuine instrumental rave (Hannah Laing's donk measures vocal ~0.45)
# from vocal-pop that the Discogs classifier mis-tags with hard-dance subgenres (vocal ~0.85-0.99
# — Katy Perry, Girls Aloud, etc.). Lowering the target (0.62 -> 0.45) + up-weighting the vocal
# axis separates them on the one dimension that actually differs, and as a bonus keeps genuine
# instrumental REMIXES of pop tracks (e.g. Girls Aloud "Life Got Cold (29 Palms remix)", voc 0.43)
# while dropping the vocal originals. Weight 2.0 (< the 3.0 above) so moderate-vocal dance
# (vocal trance/house, voc ~0.5-0.6) still fits. Trade-off: vocal-pop crossovers go too.
if "rave_cave" in _MOOD_PROFILES:
    _MOOD_PROFILES["rave_cave"]["vocal_weight"] = 2.0


def _entry_affect_proxy(entry):
    """(valence, arousal) estimated from an entry's mood tags, or (None, None) when none
    match. _MOOD_AFFECT covers the full Plex mood vocabulary exactly, so a plain lookup
    suffices. Memoised on the entry; only a fallback for absent TF valence/arousal."""
    cached = entry.get("_affect_proxy")
    if cached is not None:
        return cached
    vs, as_ = [], []
    for m in (entry.get("moods") or []):
        va = _MOOD_AFFECT.get(m.strip().lower())
        if va is not None:
            vs.append(va[0])
            as_.append(va[1])
    result = ((_calibrate_value("valence", sum(vs) / len(vs)),
               _calibrate_value("arousal", sum(as_) / len(as_))) if vs else (None, None))
    try:
        entry["_affect_proxy"] = result
    except (TypeError, AttributeError):
        pass
    return result


# Genre/style cues for a vocal-presence estimate, used only until a track's TF vocal_presence
# is computed (the real value always wins). Vocal-forward genres read high, instrumental ones
# low, everything else mid (most popular music is vocal-led).
_VOCAL_CUES = (
    "a cappella", "choral", "gospel", "doo wop", "vocal jazz", "barbershop",
    "singer/songwriter", "spoken word",
)
_INSTRUMENTAL_CUES = (
    "instrumental", "ambient", "new age", "drone", "modern composition", "chamber",
    "orchestral", "film score", "original score", "soundtrack", "post-rock", "math rock",
    "video game", "chiptune", "classical", "exotica", "minimal techno", "ambient techno",
)


def _entry_vocal_proxy(entry):
    """Coarse vocal-presence estimate from an entry's genres/styles — only a fallback for
    tracks that don't yet have a TF vocal_presence value. Memoised on the entry."""
    cached = entry.get("_vocal_proxy")
    if cached is not None:
        return cached
    tags = [t.lower() for t in (entry.get("styles") or []) + (entry.get("genres") or [])]
    if any(cue in t for t in tags for cue in _VOCAL_CUES):
        result = 0.88
    elif any(cue in t for t in tags for cue in _INSTRUMENTAL_CUES):
        result = 0.18
    else:
        result = 0.60
    result = _calibrate_value("vocal_presence", result)
    try:
        entry["_vocal_proxy"] = result
    except (TypeError, AttributeError):
        pass
    return result


def _acoustic_distance_to_centroid(entry, centroid):
    """
    Weighted normalised Euclidean distance between a track entry and a centroid.
    Returns 0.5 (neutral) when insufficient data is available.

    Every dimension is None-safe — only scored when both entry and centroid have data —
    so the function degrades gracefully on libraries without certain columns. The emotional
    axis (valence/arousal) is weighted heavily (_VALENCE_WEIGHT / _AROUSAL_WEIGHT) so two
    profiles with similar tempo/texture but different moods (e.g. rainy_day vs cosy) stay
    distinct. When a track lacks TF valence/arousal, a tag-derived proxy stands in.
    """
    num = 0.0
    den = 0.0

    def _add(d2, w):
        nonlocal num, den
        num += w * d2
        den += w

    if entry.get("bpm") and centroid.get("bpm"):
        _add(((entry["bpm"] - centroid["bpm"]) / 200.0) ** 2, 1.0)
    if entry.get("energy") is not None and centroid.get("energy") is not None:
        _add(((entry["energy"] - centroid["energy"]) / 23.0) ** 2, 1.0)
    if entry.get("danceability") is not None and centroid.get("danceability") is not None:
        _add((entry["danceability"] - centroid["danceability"]) ** 2, 1.0)
    if entry.get("brightness") is not None and centroid.get("brightness") is not None:
        _add((entry["brightness"] - centroid["brightness"]) ** 2, 1.0)
    if entry.get("year") and centroid.get("year"):
        _add(min(abs(entry["year"] - centroid["year"]) / 100.0, 1.0) ** 2, 1.0)
    if entry.get("beat_confidence") is not None and centroid.get("beat_confidence") is not None:
        _add(min(abs(entry["beat_confidence"] - centroid["beat_confidence"]) / _BC_DIFF_SCALE, 1.0) ** 2, 1.0)
    if entry.get("onset_rate") is not None and centroid.get("onset_rate") is not None:
        _add(min(abs(entry["onset_rate"] - centroid["onset_rate"]) / 10.0, 1.0) ** 2, 1.0)
    if entry.get("dynamic_complexity") is not None and centroid.get("dynamic_complexity") is not None:
        _add(min(abs(entry["dynamic_complexity"] - centroid["dynamic_complexity"]) / _DC_DIFF_SCALE, 1.0) ** 2, 1.0)
    # Emotional axis — TF value preferred, tag-derived proxy as fallback. Weighted heavily.
    if centroid.get("valence") is not None or centroid.get("arousal") is not None:
        ev, ea = entry.get("valence"), entry.get("arousal")
        if ev is None or ea is None:
            pv, pa = _entry_affect_proxy(entry)
            ev = ev if ev is not None else pv
            ea = ea if ea is not None else pa
        if ev is not None and centroid.get("valence") is not None:
            _add((ev - centroid["valence"]) ** 2, _VALENCE_WEIGHT)
        if ea is not None and centroid.get("arousal") is not None:
            _add((ea - centroid["arousal"]) ** 2, _AROUSAL_WEIGHT)
    # Vocal axis — TF value preferred, genre/style proxy as fallback until TF data lands.
    if centroid.get("vocal_presence") is not None:
        ev = entry.get("vocal_presence")
        if ev is None:
            ev = _entry_vocal_proxy(entry)
        if ev is not None:
            _add((ev - centroid["vocal_presence"]) ** 2, centroid.get("vocal_weight", _VOCAL_WEIGHT))
    if den == 0:
        return 0.5
    return min(math.sqrt(num / den), 1.0)


def _tag_overlap_score(entry, centroid):
    """Taste overlap with the listening profile: Last.fm community tags (richest signal) blended with
    AllMusic styles/genres."""
    # Last.fm community-tag overlap — overlap coefficient (robust to the large artist-tag sets)
    top_lf = {t for t, _ in centroid.get("lastfm_counter", Counter()).most_common(30)}
    track_lf = set(entry.get("lastfm_track_tags") or {}) | set(entry.get("lastfm_artist_tags") or {})
    lf = (len(top_lf & track_lf) / min(len(top_lf), len(track_lf))) if (top_lf and track_lf) else None
    # AllMusic styles / genres Jaccard (existing)
    top_styles = {s for s, _ in centroid.get("styles_counter", Counter()).most_common(10)}
    top_genres = {g for g, _ in centroid.get("genres_counter", Counter()).most_common(5)}
    track_styles = set(entry.get("styles") or [])
    track_genres = set(entry.get("genres") or [])
    sg = None
    if top_styles and track_styles:
        sg = len(top_styles & track_styles) / len(top_styles | track_styles)
    elif top_genres and track_genres:
        sg = len(top_genres & track_genres) / len(top_genres | track_genres)
    # blend — Last.fm primary; fall back to whichever signal is present
    if lf is not None and sg is not None:
        return 0.6 * lf + 0.4 * sg
    return lf if lf is not None else (sg if sg is not None else 0.0)


def acoustic_affinity(rk, centroid, essentia_cache):
    """0–1 similarity. Base = 60% acoustic distance (inverted) + 40% tag overlap. When the centroid carries
    an embedding ("emb", e.g. compute_listening_centroid) AND the track has one, blend the base 50/50 with
    the embedding cosine ("sounds-like"). A track (or centroid) without an embedding falls back to the base
    — graceful under partial coverage. WHY: sharper discovery than the 10-d centroid + tags (audit Tier-4)."""
    entry = essentia_cache.get(str(rk), {})
    base = (0.6 * (1.0 - _acoustic_distance_to_centroid(entry, centroid))
            + 0.4 * _tag_overlap_score(entry, centroid))
    cen_emb = centroid.get("emb") if isinstance(centroid, dict) else None
    if cen_emb is not None:
        v = _track_emb(entry)
        if v is not None:
            emb_aff = 0.5 + 0.5 * _emb_cosine(v, cen_emb)   # cosine [-1,1] -> affinity [0,1]
            return 0.5 * base + 0.5 * emb_aff
    return base


def _album_acoustic_centroid(rks, essentia_cache):
    """Mean acoustic vector for a collection of ratingKeys."""
    accum = defaultdict(list)
    for rk in rks:
        e = essentia_cache.get(str(rk), {})
        for f in ("bpm", "energy", "danceability", "brightness", "year"):
            if e.get(f) is not None:
                accum[f].append(e[f])
    return {f: (sum(v) / len(v)) if v else None for f, v in accum.items()}


def _pick_representative_tracks(tracks, essentia_cache, n=2):
    """
    Pick the n tracks most acoustically central to the album.
    Falls back to positional selection (tracks 0 and 2) if no acoustic data.
    """
    if not tracks:
        return []
    rks = [str(t.ratingKey) for t in tracks]
    centroid = _album_acoustic_centroid(rks, essentia_cache)
    if not any(centroid.get(f) for f in ("bpm", "energy", "danceability", "brightness")):
        indices = [0, 2] if len(tracks) > 2 else list(range(min(n, len(tracks))))
        return [tracks[i] for i in indices[:n]]
    scored = [(_acoustic_distance_to_centroid(essentia_cache.get(str(t.ratingKey), {}), centroid), t)
              for t in tracks]
    scored.sort(key=lambda x: x[0])
    return [t for _, t in scored[:n]]


def _round_robin_interleave(track_lists, cap):
    """Fair interleave across multiple track lists, stopping at cap."""
    result = []
    queues = [list(lst) for lst in track_lists if lst]
    while queues and len(result) < cap:
        next_queues = []
        for q in queues:
            if q and len(result) < cap:
                result.append(q.pop(0))
            if q:
                next_queues.append(q)
        queues = next_queues
    return result


def is_low_rated(track, threshold=4):
    r = getattr(track, "userRating", None)
    return r is not None and r <= threshold


def _rating_multiplier(ur):
    """
    Score multiplier for history-based playlists.
    Plex internal scale: 5 = 2.5★ (neutral), 10 = 5★ (loved).
    Unrated tracks are neutral — no penalty. Tracks ≤ 4 are excluded upstream.
    """
    if ur is None or ur <= 5: return 1.0
    if ur >= 9:               return 1.20  # 4.5–5★ — Spotify "liked" equivalent
    if ur >= 7:               return 1.10  # 3.5–4★ — enjoyed
    return 1.05                             # 3★ — mildly positive


_RATING_WEIGHT = 0.10   # moderate loved-track pull (gentle ≈0.05 / strong ≈0.15); ×tier below

# 50/50 anchor/discovery balance for select genre mixes (piloted on rave_cave; roll out by adding keys).
# Anchors = tracks you'll KNOW (played, or Last.fm listeners ≥ floor) or already LIKE; the other half is
# fame-blind best-FIT discovery. Anchor share is capped by availability (niche genres auto-skew discovery).
_BALANCED_PROFILES = {"rave_cave"}
_ANCHOR_LISTENERS  = 100_000
_ANCHOR_RATIO      = 0.50


def _rating_dist_bonus(ur):
    """
    Distance reduction favouring tracks you really like (>=3.5★ = Plex rating 7): they sort as if
    acoustically closer to the target. Shared by the mood mixes, Daily Mixes and Artist Deep Cuts.
    3★ (rating 6) is the baseline 'good' so near-neutral; <=2★ is excluded upstream by is_low_rated;
    unrated = neutral. Scales with _RATING_WEIGHT. (Discover Weekly + decade/geo showcases don't call this.)
    """
    if ur is None or ur < 6: return 0.0
    if   ur >= 10: tier = 1.10   # 5★
    elif ur >= 9:  tier = 1.00   # 4.5★
    elif ur >= 8:  tier = 0.90   # 4★
    elif ur >= 7:  tier = 0.80   # 3.5★ — "really like" threshold
    else:          tier = 0.20   # 3★ — baseline 'good'
    return _RATING_WEIGHT * tier


def _artist_key(track):
    """Normalised primary artist key for deduplication and capping."""
    name = (getattr(track, "grandparentTitle", "") or
            getattr(track, "originalTitle", "") or "")
    return norm_text(primary_artist(name))


def _song_key(track):
    """
    Deduplication key: normalised (track_artist, lastfm_query_title) pair.
    Uses track_artist_name() which resolves the actual performing artist even
    on compilation albums where grandparentTitle is 'Various Artists' — in that
    case it falls back to track.originalTitle (where Plex stores the real artist).
    Uses lastfm_query_title: strips reissue suffixes (Remastered/Deluxe/Radio Edit) but KEEPS
    different-recording tags (remix/live/acoustic), so those count as their OWN distinct songs.
    """
    artist = norm_text(primary_artist(track_artist_name(track)))
    # WHY: lastfm_query_title (not clean_title) so a different RECORDING (remix/live/acoustic) keys as its OWN
    # song — independently selectable in a mix — while reissues (remaster/deluxe) still collapse onto the original.
    title  = norm_text(lastfm_query_title(getattr(track, "title", "") or ""))
    return (artist, title)


def _dedup_filter(tracks, essentia_cache=None):
    """
    Remove duplicate songs from a track list, keeping the most CANONICAL copy of each (studio/original
    — not a live / remix / demo / instrumental / compilation version) at the song's best (highest-
    scored) position. Two tracks are the same song by normalised (artist, lastfm_query_title) key — so
    "Stars" from a studio album and "Stars" from a compilation are treated as one entry.
    """
    return _dedup_canonical(tracks, essentia_cache)


# ===========================================================================
# Last.fm Helpers (optional — graceful no-op when key absent)
# ===========================================================================
_lastfm_artist_tops = {}  # artist_name_lower -> {track_title_lower: play_count}


def _lastfm_top_tracks_for_artist(artist_name):
    """
    Fetch top 50 tracks for an artist from Last.fm with global play counts.
    Returns {track_title_lower: int}. Cached per run. Returns {} on any failure.
    """
    if not LASTFM_API_KEY or not _REQUESTS_AVAILABLE:
        return {}
    clean = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", artist_name)).strip()
    cache_key = clean.lower()
    if cache_key in _lastfm_artist_tops:
        return _lastfm_artist_tops[cache_key]

    for attempt in range(3):
        try:
            resp = _requests.get(
                "http://ws.audioscrobbler.com/2.0/",
                params={
                    "method": "artist.gettoptracks",
                    "artist": clean,
                    "limit": 50,
                    "api_key": LASTFM_API_KEY,
                    "format": "json",
                },
                timeout=10,
            )
            data = resp.json()
            tracks = data.get("toptracks", {}).get("track", [])
            result = {
                t["name"].lower(): int(t.get("playcount", 0))
                for t in tracks if t.get("name")
            }
            _lastfm_artist_tops[cache_key] = result
            return result
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                xlog(f"[WARN] Last.fm lookup failed for '{artist_name}': {e}")

    _lastfm_artist_tops[cache_key] = {}
    return {}


def _lastfm_popularity_score(track_title, artist_name):
    """Return the Last.fm global play count for a track, or 0 on failure/miss."""
    top = _lastfm_top_tracks_for_artist(artist_name)
    if not top:
        return 0
    title_lower = track_title.lower()
    if title_lower in top:
        return top[title_lower]
    # Partial match fallback
    for k, v in top.items():
        if title_lower in k or k in title_lower:
            return v
    return 0


_lastfm_track_cache = {}  # (artist_lower, title_lower) -> int playcount


def _lastfm_track_playcount(track_title, artist_name):
    """
    Fetch the global play count for a specific track via track.getInfo.
    Returns int playcount, or 0 on failure / track not found on Last.fm.
    Cached per (artist, title) within the run. Uses a short timeout with no
    retries so brand-new or unindexed tracks fail fast rather than hanging.
    """
    if not LASTFM_API_KEY or not _REQUESTS_AVAILABLE:
        return 0
    cache_key = (artist_name.strip().lower(), track_title.strip().lower())
    if cache_key in _lastfm_track_cache:
        return _lastfm_track_cache[cache_key]
    try:
        resp = _requests.get(
            "http://ws.audioscrobbler.com/2.0/",
            params={
                "method": "track.getInfo",
                "artist": artist_name,
                "track":  track_title,
                "api_key": LASTFM_API_KEY,
                "format": "json",
            },
            timeout=3,
        )
        count = int(resp.json().get("track", {}).get("playcount", 0) or 0)
    except Exception:
        count = 0
    _lastfm_track_cache[cache_key] = count
    return count


# ===========================================================================
# K-means Helpers (numpy required)
# ===========================================================================

def _kmeans_pp_init(X, k, rng):
    """K-means++ centroid seeding."""
    n = len(X)
    idx = rng.integers(0, n)
    centroids = [X[idx]]
    for _ in range(k - 1):
        dists = np.array([min(float(np.linalg.norm(x - c)) ** 2 for c in centroids) for x in X])
        total = dists.sum()
        if total == 0:
            idx = rng.integers(0, n)
        else:
            idx = rng.choice(n, p=dists / total)
        centroids.append(X[idx])
    return np.array(centroids)


def _kmeans_fit(X, k, seed=42, max_iter=50):
    """
    K-means with k-means++ seeding. Returns (centroids, labels).
    Deterministic given the same seed.
    """
    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(X, k, rng)
    labels = np.zeros(len(X), dtype=int)

    for _ in range(max_iter):
        # Assignment: distance from each point to each centroid
        dists = np.stack([np.linalg.norm(X - c, axis=1) for c in centroids], axis=1)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            new_centroids[j] = X[mask].mean(axis=0) if mask.any() else centroids[j]
        centroids = new_centroids

    return centroids, labels


# ===========================================================================
# Cover Generation
# ===========================================================================

def _make_gradient_image(w, h, color_top, color_bottom, diagonal=False):
    """
    RGBA gradient image. Colors are 3-tuples (RGB) or 4-tuples (RGBA).
    diagonal=True: blends top-left → bottom-right (requires numpy).
    diagonal=False: simple vertical top → bottom gradient.
    """
    at = color_top[3]    if len(color_top)    == 4 else 255
    ab = color_bottom[3] if len(color_bottom) == 4 else 255

    if diagonal and _NUMPY_AVAILABLE:
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        t = (x[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        def _ch(a, b):
            return np.clip(a + t * (b - a), 0, 255).astype(np.uint8)
        arr = np.stack([
            _ch(color_top[0], color_bottom[0]),
            _ch(color_top[1], color_bottom[1]),
            _ch(color_top[2], color_bottom[2]),
            _ch(at, ab),
        ], axis=2)
        return Image.fromarray(arr, "RGBA")

    img  = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(img)
    for y_coord in range(h):
        t = y_coord / h
        draw.line([(0, y_coord), (w, y_coord)], fill=(
            int(color_top[0] + t * (color_bottom[0] - color_top[0])),
            int(color_top[1] + t * (color_bottom[1] - color_top[1])),
            int(color_top[2] + t * (color_bottom[2] - color_top[2])),
            int(at           + t * (ab              - at)),
        ))
    return img


def _rotated_rect_points(cx, cy, w, h, angle_deg):
    """4 corner points of a rectangle centred at (cx, cy) rotated by angle_deg."""
    import math
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca) for dx, dy in corners]


def _add_bottom_vignette(img, strength=175, coverage=0.55):
    """Quadratic dark gradient over the bottom `coverage` fraction — improves text contrast."""
    W, H = img.size
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vignette)
    start_y = int(H * (1.0 - coverage))
    for y in range(start_y, H):
        t = (y - start_y) / (H - start_y)
        draw.line([(0, y), (W, y)], fill=(0, 0, 0, int(t * t * strength)))
    return Image.alpha_composite(img, vignette)


def _make_geometric_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Diagonal tilted strips. v selects strip angle/count/arrangement variant."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # Each variant: list of (cx_f, cy_f, wf, hf, angle, alpha, color)
    variants = [
        # v=0 default — 5 strips at 22° (party/misc)
        [(0.88,0.10,0.78,1.90,22,55,L),(0.18,0.90,0.72,1.80,22,50,D),
         (0.52,0.48,0.60,1.65,22,35,M),(0.10,0.22,0.52,1.45,-18,32,L),(0.82,0.80,0.55,1.35,-18,28,D)],
        # v=1 running — shallower 12° speed strips
        [(0.90,0.08,0.85,1.90,12,52,L),(0.12,0.92,0.80,1.85,12,47,D),
         (0.50,0.50,0.70,1.70,12,32,M),(0.25,0.30,0.55,1.50,-10,28,L),(0.75,0.72,0.50,1.40,-10,24,D)],
        # v=2 confidence_boost — 3 bold strips at 38°
        [(0.85,0.12,0.90,2.00,38,58,L),(0.15,0.88,0.85,1.95,38,52,D),(0.50,0.50,0.65,1.75,38,38,M)],
        # v=3 folk_acoustic — simple 3 strips, warm angle 20°
        [(0.80,0.15,0.75,1.85,20,48,L),(0.20,0.85,0.70,1.80,20,44,D),(0.50,0.50,0.50,1.60,20,30,M)],
        # v=4 brunch_mix — 4 airy light strips at 25°
        [(0.88,0.10,0.78,1.90,25,45,L),(0.18,0.90,0.72,1.80,25,40,D),
         (0.55,0.42,0.55,1.60,25,28,M),(0.30,0.65,0.48,1.55,-20,22,L)],
        # v=5 acoustic_romance — 2 minimal gentle strips
        [(0.82,0.20,0.78,1.90,20,42,L),(0.18,0.80,0.72,1.85,20,38,D)],
        # v=6 commute_mix — 6 narrow regular strips at 15°
        [(0.92,0.08,0.60,1.90,15,44,L),(0.08,0.92,0.58,1.88,15,40,D),
         (0.55,0.30,0.50,1.70,15,30,M),(0.45,0.70,0.48,1.68,15,28,L),
         (0.20,0.20,0.42,1.55,-12,24,D),(0.80,0.80,0.40,1.52,-12,20,M)],
        # v=7 moody_mix — 4 heavy dark strips at 30°
        [(0.88,0.08,0.82,1.95,30,60,L),(0.12,0.92,0.78,1.90,30,55,D),
         (0.52,0.45,0.62,1.72,30,42,M),(0.48,0.55,0.58,1.68,-25,36,D)],
        # v=8 synthpop_romance — 4 steep 80s-style strips at 52°
        [(0.88,0.08,0.78,1.95,52,58,L),(0.12,0.92,0.72,1.90,52,52,D),
         (0.55,0.38,0.62,1.72,52,40,M),(0.45,0.62,0.55,1.65,-45,34,L)],
    ]
    strips = variants[min(v, len(variants) - 1)]
    angle_jitter = rng.uniform(-4, 4) if rng else 0
    for cx_f, cy_f, wf, hf, angle, alpha, color in strips:
        pts = _rotated_rect_points(cx_f * w, cy_f * h, wf * w, hf * h, angle + angle_jitter)
        r, g, b = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(r, g, b, alpha))
    return img


def _make_concentric_circles_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Concentric rings. v shifts centre position and ring count/density."""
    # (cx_frac, cy_frac, n_rings, r_max_frac)
    configs = [
        (0.50, 0.52, 10, 0.90),  # v=0 time_capsule — centred
        (0.50, 0.50, 14, 0.92),  # v=1 focus — more rings, tight
        (0.35, 0.48, 10, 0.88),  # v=2 late_night_romance — off-centre left
        (0.50, 0.32,  7, 0.85),  # v=3 winter_mix — fewer, from upper centre
        (0.50, 0.46, 12, 0.88),  # v=4 deep_work — tight focus, slightly up
        (0.42, 0.56, 10, 0.88),  # v=5 nostalgia_mix — slightly off-centre
        (0.62, 0.46, 10, 0.88),  # v=6 party_throwback — off-centre right
    ]
    cx_f, cy_f, n_rings, r_frac = configs[min(v, len(configs) - 1)]
    img  = _make_gradient_image(w, h, color_bottom, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    cx, cy, r_max = int(cx_f * w), int(cy_f * h), int(r_frac * w)
    if rng:
        cx += int(rng.uniform(-0.04, 0.04) * w)
        cy += int(rng.uniform(-0.04, 0.04) * h)
    for i in range(n_rings, 0, -1):
        r = int(r_max * i / n_rings)
        t = 1.0 - i / n_rings
        col = tuple(int(color_bottom[k] + t * (color_top[k] - color_bottom[k])) for k in range(3))
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                     outline=(*col, 200), width=max(5, r // 7))
    return img


def _make_radial_glow_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Soft radial glow. v shifts the glow centre (sunrise, spotlight, candlelight, etc.)."""
    # (cx_frac, cy_frac, gamma)
    configs = [
        (0.50, 0.40, 0.65),  # v=0 discover_weekly — upper-centre
        (0.50, 0.65, 0.70),  # v=1 sunday_morning  — warm lower glow + sunrise rings
        (0.50, 0.80, 0.38),  # v=2 sunset_mix       — horizon glow, sharp falloff + corona rings
        (0.50, 0.50, 0.65),  # v=3 piano_romance    — centred spotlight + stage rings
        (0.50, 0.52, 0.28),  # v=4 candlelight      — tight hot-spot + halo rings
        (0.32, 0.35, 0.60),  # v=5 late_night       — off-centre dark ambient
        (0.50, 0.22, 0.42),  # v=6 rainy_day        — overcast top glow + rings
    ]
    # Concentric ring overlays: {v: [(radius_norm, half_width_norm, brightness_delta), ...]}
    # Rings are painted onto the gradient array after it is built, using the
    # normalised dist array (same units as the gamma falloff).
    _rings = {
        1: [(0.22, 0.008, 38), (0.40, 0.007, 26)],                                          # sunrise horizon bands
        2: [(0.16, 0.007, 55), (0.28, 0.006, 42), (0.42, 0.005, 30), (0.58, 0.005, 18)],
        3: [(0.18, 0.009, 52), (0.32, 0.008, 38), (0.48, 0.007, 26)],                       # stage spotlight rings
        4: [(0.07, 0.012, 100), (0.17, 0.010, 75), (0.30, 0.008, 52), (0.46, 0.007, 32)],
        6: [(0.20, 0.009, 48), (0.38, 0.008, 35), (0.57, 0.007, 24)],
    }
    cx_f, cy_f, gamma = configs[min(v, len(configs) - 1)]
    if not _NUMPY_AVAILABLE:
        return _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    cx, cy = w * cx_f, h * cy_f
    if rng:
        cx += rng.uniform(-0.03, 0.03) * w
        cy += rng.uniform(-0.03, 0.03) * h
    x  = np.linspace(0, w, w, dtype=np.float32)
    y  = np.linspace(0, h, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(((xx - cx) / (w * 0.62)) ** 2 + ((yy - cy) / (h * 0.62)) ** 2)
    t    = np.clip(dist, 0.0, 1.0) ** gamma
    def _ch(a, b): return np.clip(a + t * (b - a), 0, 255).astype(np.uint8)
    arr = np.stack([_ch(color_top[0], color_bottom[0]),
                    _ch(color_top[1], color_bottom[1]),
                    _ch(color_top[2], color_bottom[2]),
                    np.full((h, w), 255, np.uint8)], axis=2)
    for ring_r, ring_w, bright in _rings.get(v, []):
        mask = np.abs(dist - ring_r) < ring_w
        arr[mask, :3] = np.clip(arr[mask, :3].astype(np.int32) + bright, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _make_ripples_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Overlapping concentric ripple rings — like raindrops landing on water. Flattened
    ellipses give a gentle on-the-surface perspective. Used for rainy themes so the
    background echoes the falling-drop icon."""
    rng = rng or random
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    def _clamp(x): return max(0, min(255, x))
    light = tuple(_clamp(c + 64) for c in color_top[:3])
    dark  = tuple(int(c * 0.6) for c in color_top[:3])
    # WHY: ripple centres are confined to the LOWER band (water sits *below* the falling rain) —
    # previously they scattered from y≈0.08 and landed level with / above the rain-cloud glyph,
    # which read as nonsense. Flatter squish (0.42) reads as drops on a surface, not floating rings.
    for _ in range(5 + (v % 3)):
        rx    = rng.randint(int(w * 0.10), int(w * 0.90))
        ry    = rng.randint(int(h * 0.50), int(h * 0.74))
        rings = rng.randint(3, 5)
        gap   = rng.randint(24, 40)
        for k in range(1, rings + 1):
            rr    = k * gap
            alpha = max(16, 110 - k * 20)
            tone  = light if k % 2 else dark
            draw.ellipse([rx - rr, ry - rr * 0.42, rx + rr, ry + rr * 0.42],
                         outline=(*tone, alpha), width=3)
    return Image.alpha_composite(img, overlay)


def _make_waves_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Sinusoidal wave bands. v controls band count, amplitude and phase for unique looks."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=45): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.55): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # Each variant: list of (y_frac, amp_frac, freq, phase, band_h_frac, alpha, color)
    variants = [
        # v=0 default extras (deep_cuts etc) — 4 medium bands
        [(0.22,0.07,0.9,0.0,0.20,55,L),(0.48,0.08,1.2,1.1,0.22,48,D),
         (0.65,0.06,1.0,2.3,0.18,42,M),(0.82,0.07,1.4,0.6,0.24,38,L)],
        # v=1 chill — 3 wide gentle rolling waves, high amplitude
        [(0.25,0.10,0.7,0.0,0.26,52,L),(0.55,0.12,0.8,1.6,0.28,46,D),(0.80,0.09,0.6,3.0,0.24,40,M)],
        # v=2 sleep — 2 barely-visible ultra-flat bands
        [(0.35,0.03,0.5,0.0,0.18,38,L),(0.70,0.03,0.6,2.0,0.20,32,D)],
        # v=3 melancholy — 6 narrow tight compressed waves
        [(0.15,0.05,1.6,0.0,0.12,52,D),(0.28,0.04,1.8,0.8,0.11,46,L),
         (0.42,0.05,1.4,1.7,0.12,42,D),(0.56,0.04,2.0,0.4,0.11,38,M),
         (0.70,0.05,1.5,2.5,0.12,35,D),(0.84,0.04,1.7,1.2,0.11,30,L)],
        # v=4 love_songs — 5 lyrical flowing, medium
        [(0.18,0.08,0.8,0.0,0.18,52,L),(0.35,0.09,1.1,1.3,0.20,47,D),
         (0.52,0.07,0.9,2.6,0.17,42,M),(0.68,0.08,1.0,0.9,0.19,38,L),(0.84,0.07,1.2,2.0,0.17,34,D)],
        # v=5 walking_mix — 5 steady even-spaced, slight diagonal feel
        [(0.18,0.06,1.0,0.3,0.16,50,L),(0.34,0.07,1.0,1.6,0.17,45,D),
         (0.50,0.06,1.0,3.0,0.16,40,M),(0.66,0.07,1.0,1.0,0.17,36,L),(0.82,0.06,1.0,2.3,0.16,32,D)],
        # v=6 bittersweet — 4 irregular, asymmetric mixed-feeling waves
        [(0.20,0.09,0.7,0.0,0.22,54,L),(0.45,0.06,1.5,0.5,0.14,46,D),
         (0.62,0.10,0.8,2.8,0.23,40,M),(0.83,0.05,1.8,1.4,0.13,35,L)],
        # v=7 cosy — 3 enveloping warm waves (different phase from chill)
        [(0.22,0.10,0.7,1.8,0.26,52,M),(0.52,0.11,0.8,0.3,0.28,46,L),(0.78,0.09,0.6,2.5,0.24,40,D)],
        # v=8 beach_vibes — 4 rolling ocean waves, higher amplitude
        [(0.20,0.11,0.8,0.0,0.22,54,L),(0.44,0.13,1.0,1.4,0.24,48,D),
         (0.66,0.10,0.9,2.8,0.21,43,M),(0.85,0.12,1.1,0.7,0.23,38,L)],
        # v=9 cool_down — 4 measured, slightly flatter cooling waves
        [(0.22,0.07,0.8,0.6,0.19,50,L),(0.44,0.08,1.0,2.0,0.20,44,D),
         (0.64,0.07,0.9,3.5,0.18,39,M),(0.83,0.08,1.1,1.2,0.20,34,L)],
    ]
    wave_defs = variants[min(v, len(variants) - 1)]
    steps = w + 1
    for y_frac, amp_frac, freq, phase, bh_frac, alpha, color in wave_defs:
        if rng:
            phase = phase + rng.uniform(-0.5, 0.5)
        cy  = y_frac * h; amp = amp_frac * h; bh = bh_frac * h
        top = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase)) for x in range(steps)]
        bot = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase) + bh) for x in range(steps - 1, -1, -1)]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(top + bot, fill=(rc, gc, bc, alpha))
    return img


def _make_floating_circles_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Overlapping semi-transparent spheres. v changes count, size and arrangement."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # Each variant: list of (cx_frac, cy_frac, rad_frac, alpha, color)
    variants = [
        # v=0 happy — 7 circles, classic joyful arrangement
        [(0.75,0.22,0.42,48,L),(0.15,0.68,0.34,44,D),(0.90,0.80,0.28,40,M),
         (0.42,0.12,0.20,36,D),(0.12,0.20,0.16,32,L),(0.62,0.70,0.14,28,L),(0.88,0.08,0.11,24,M)],
        # v=1 celebration — many smaller scattered circles (confetti)
        [(0.80,0.18,0.22,50,L),(0.20,0.72,0.20,46,D),(0.55,0.35,0.18,42,M),
         (0.10,0.40,0.16,40,L),(0.88,0.55,0.15,38,D),(0.40,0.80,0.14,35,M),
         (0.65,0.10,0.13,32,L),(0.30,0.25,0.12,28,D),(0.75,0.68,0.11,26,L),(0.50,0.58,0.10,22,M)],
        # v=2 lazy_sunday — 4 large slow circles, widely spaced
        [(0.72,0.25,0.52,44,L),(0.18,0.72,0.48,40,D),(0.88,0.78,0.34,36,M),(0.38,0.10,0.28,32,L)],
        # v=3 spring_mix — many small clustered buds, upper half
        [(0.60,0.20,0.18,50,L),(0.35,0.15,0.16,46,D),(0.80,0.12,0.15,42,M),
         (0.20,0.30,0.14,40,L),(0.70,0.35,0.13,37,D),(0.45,0.28,0.12,34,M),
         (0.90,0.30,0.11,30,L),(0.15,0.48,0.10,26,D),(0.55,0.42,0.09,22,M)],
        # v=4 first_date — nervous scattered, uneven sizes
        [(0.68,0.18,0.30,46,L),(0.22,0.62,0.25,42,D),(0.85,0.72,0.20,38,M),
         (0.38,0.08,0.16,36,L),(0.08,0.18,0.13,32,D),(0.58,0.82,0.12,28,M),(0.92,0.08,0.09,24,L)],
        # v=5 driving_singalong — carefree arrangement, weighted to right
        [(0.82,0.20,0.38,48,L),(0.55,0.70,0.30,44,D),(0.95,0.60,0.24,40,M),
         (0.68,0.10,0.18,36,L),(0.38,0.30,0.14,32,D),(0.78,0.85,0.12,28,L)],
        # v=6 pre_party — building up, weighted to upper-right corner
        [(0.88,0.12,0.40,50,L),(0.70,0.30,0.28,46,D),(0.92,0.45,0.22,42,M),
         (0.55,0.15,0.18,38,L),(0.78,0.62,0.15,34,D),(0.60,0.50,0.11,28,M),(0.42,0.08,0.09,24,L)],
    ]
    circle_defs = variants[min(v, len(variants) - 1)]
    for cx_f, cy_f, rf, alpha, color in circle_defs:
        if rng:
            cx_f = cx_f + rng.uniform(-0.04, 0.04)
            cy_f = cy_f + rng.uniform(-0.04, 0.04)
            rf   = max(0.04, rf + rng.uniform(-0.02, 0.02))
        cx  = int(cx_f * w); cy = int(cy_f * h); rad = int(rf * min(w, h))
        rc, gc, bc = (_clamp(c) for c in color)
        draw.ellipse([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=(rc, gc, bc, alpha))
    return img


def _make_rays_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Triangular rays. v changes origin point, ray count and angular spread."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    # (ox_frac, oy_frac, n_rays, start_deg, spread_deg)
    configs = [
        (0.06, 0.06, 9,   -8,  100),  # v=0 default — upper-left, 9 rays
        (0.50, 1.05, 11, -125,  110),  # v=1 empowering — from bottom-centre, upward fan
        (0.06, 0.06, 6,   -5,   60),  # v=2 angst_mix — top-left, tight aggressive
        (0.50,-0.05, 8,   -45,  130),  # v=3 main_character — from top-centre, spotlight
        (0.06, 0.06, 10,  -10,   90),  # v=4 friday_night — top-left, medium spread
        (-0.05,0.50, 7,   -55,   70),  # v=5 night_drive — from left edge, horizontal fan
    ]
    ox_f, oy_f, n_rays, start_deg, spread_deg = configs[min(v, len(configs) - 1)]
    if rng:
        start_deg += rng.uniform(-12, 12)
    ox, oy = ox_f * w, oy_f * h
    dist   = max(w, h) * 2.4
    for i in range(n_rays):
        a1 = math.radians(start_deg + i * spread_deg / n_rays)
        a2 = math.radians(start_deg + (i + 1) * spread_deg / n_rays)
        color = L if i % 2 == 0 else D
        alpha = 52 if i % 2 == 0 else 36
        pts = [(ox, oy),
               (ox + dist * math.cos(a1), oy + dist * math.sin(a1)),
               (ox + dist * math.cos(a2), oy + dist * math.sin(a2))]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_arc_sweep_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Large circular arcs from off-canvas centres. v shifts arc origins for different sweep directions."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=50): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.55): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    n_steps = 90
    # Each variant: list of (cx_frac, cy_frac, r_frac, a_start_deg, a_end_deg, alpha, color)
    variants = [
        # v=0 romantic_mix — arcs sweeping in from right
        [(1.35,0.50,1.20,145,215,55,L),(-0.30,0.45,1.00,-35,55,48,D),
         (0.50,1.45,1.05,218,322,42,M),(0.50,-0.40,0.90,38,142,36,L)],
        # v=1 evening_unwind — arcs from left (mirror, unwinding feel)
        [(-0.35,0.50,1.20,-35,35,55,L),(1.30,0.45,1.00,145,215,48,D),
         (0.50,1.45,1.05,218,322,42,M),(0.50,-0.40,0.90,38,142,36,L)],
        # v=2 weekend_mix — arcs from top and bottom (horizontal sweeps)
        [(0.50,-0.40,1.15,28,152,55,L),(0.50,1.40,1.10,212,332,48,D),
         (1.35,0.50,0.95,145,215,40,M),(-0.30,0.45,0.88,-35,55,35,L)],
        # v=3 on_repeat (non-mood extras default) — tighter, more circular
        [(1.25,0.50,1.05,148,212,52,L),(-0.25,0.45,0.95,-32,52,46,D),
         (0.50,1.35,0.98,220,320,40,M),(0.50,-0.35,0.85,40,140,34,L)],
        # v=4 romantic_jazz — smooth bottom curves, jazzier flow
        [(0.50,1.50,1.20,210,330,55,L),(1.30,0.40,1.05,148,212,48,D),
         (-0.30,0.55,0.95,-30,50,42,M),(0.50,-0.40,0.85,40,140,36,L)],
    ]
    arc_defs = variants[min(v, len(variants) - 1)]
    for cx_f, cy_f, rf, a0, a1, alpha, color in arc_defs:
        if rng:
            aj = rng.uniform(-8, 8)
            a0, a1 = a0 + aj, a1 + aj
        cx  = cx_f * w; cy = cy_f * h; rad = rf * max(w, h)
        pts = [(cx + rad * math.cos(math.radians(a0 + s * (a1 - a0) / n_steps)),
                cy + rad * math.sin(math.radians(a0 + s * (a1 - a0) / n_steps)))
               for s in range(n_steps + 1)]
        pts.append((cx, cy))
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_aurora_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Narrow shimmering ribbon waves. v controls band count, frequency and pacing."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=40): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.60): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    variants = [
        # v=0 default — 6 ribbons, medium freq
        [(0.18,0.04,1.8,0.0,0.10,55,L),(0.32,0.05,1.3,1.3,0.12,50,D),(0.46,0.04,2.1,2.7,0.09,45,M),
         (0.58,0.05,1.6,0.8,0.11,42,L),(0.70,0.04,1.1,1.9,0.10,38,D),(0.82,0.05,1.9,3.2,0.13,35,M)],
        # v=1 daydreaming — 5 slower ribbons, lower frequency, wider bands
        [(0.20,0.05,0.9,0.0,0.14,52,L),(0.36,0.06,0.7,1.8,0.15,47,D),(0.52,0.05,1.0,3.2,0.13,43,M),
         (0.66,0.06,0.8,0.6,0.14,38,L),(0.80,0.05,0.6,2.4,0.15,34,D)],
        # v=2 emotional — 8 intense, faster ribbons
        [(0.14,0.04,2.2,0.0,0.09,56,L),(0.24,0.05,1.8,0.9,0.10,51,D),(0.34,0.04,2.5,2.1,0.08,47,M),
         (0.44,0.05,1.6,1.4,0.10,43,L),(0.54,0.04,2.0,3.0,0.09,39,D),(0.64,0.05,1.9,0.5,0.10,36,M),
         (0.74,0.04,2.3,1.8,0.09,32,L),(0.84,0.05,1.7,2.7,0.10,28,D)],
        # v=3 summer_evening — 4 wide slow warm ribbons
        [(0.22,0.06,0.6,0.0,0.16,52,L),(0.42,0.07,0.5,2.0,0.18,47,D),
         (0.62,0.06,0.7,1.0,0.16,42,M),(0.80,0.07,0.4,3.0,0.18,37,L)],
        # v=4 strings_romance — 6 ribbons, different phases (classical)
        [(0.18,0.04,1.6,0.5,0.10,54,L),(0.32,0.05,1.2,2.0,0.12,49,D),(0.46,0.04,1.9,0.0,0.09,44,M),
         (0.58,0.05,1.4,2.8,0.11,41,L),(0.70,0.04,1.0,1.5,0.10,37,D),(0.82,0.05,1.7,0.3,0.12,33,M)],
        # v=5 golden_hour — 4 wide, very slow, warm horizon bands
        [(0.25,0.07,0.5,0.0,0.18,54,L),(0.48,0.08,0.4,2.5,0.20,48,D),
         (0.68,0.07,0.6,1.2,0.17,43,M),(0.85,0.08,0.3,3.8,0.19,38,L)],
        # v=6 romantic_dinner — 5 warm atmospheric ribbons
        [(0.20,0.05,1.0,0.8,0.12,53,M),(0.36,0.06,0.8,2.2,0.13,48,L),(0.52,0.05,1.2,0.0,0.11,43,D),
         (0.67,0.06,0.9,1.6,0.12,38,M),(0.82,0.05,1.1,3.0,0.13,34,L)],
        # v=7 dreamy_mix — 7 rapid dreamy ribbons, high freq
        [(0.16,0.04,2.4,0.0,0.09,55,L),(0.27,0.05,2.0,1.2,0.10,50,D),(0.38,0.04,2.8,2.5,0.08,46,M),
         (0.49,0.05,2.2,0.7,0.10,42,L),(0.60,0.04,2.6,1.9,0.09,38,D),(0.71,0.05,2.1,3.1,0.10,34,M),
         (0.82,0.04,2.4,0.4,0.09,30,L)],
    ]
    ribbon_defs = variants[min(v, len(variants) - 1)]
    steps = w + 1
    for y_frac, amp_frac, freq, phase, bh_frac, alpha, color in ribbon_defs:
        if rng:
            y_frac = y_frac + rng.uniform(-0.015, 0.015)
            phase  = phase  + rng.uniform(-0.4,   0.4)
        cy = y_frac * h; amp = amp_frac * h; bh = bh_frac * h
        top = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase)) for x in range(steps)]
        bot = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase) + bh) for x in range(steps - 1, -1, -1)]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(top + bot, fill=(rc, gc, bc, alpha))
    return img


def _make_triangles_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Bold geometric triangles. v changes orientation, count and arrangement."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # Each variant: list of ([(xf,yf),...], color, alpha)
    variants = [
        # v=0 default — 4 triangles, diagonal mix (driving/commute)
        [([(1.05,-0.05),(0.30,0.52),(1.05,1.05)],L,55),
         ([(-0.05,0.15),(0.70,0.52),(-0.05,1.05)],D,50),
         ([(0.15,-0.05),(0.85,-0.05),(0.50,0.70)],M,40),
         ([(-0.05,-0.05),(0.38,-0.05),(-0.05,0.50)],L,30)],
        # v=1 cathartic — downward-pointing triangles (emotional release, pouring down)
        [([(0.15,-0.05),(0.85,-0.05),(0.50,0.80)],L,56),
         ([(0.50,-0.05),(1.05,-0.05),(1.05,0.60)],D,50),
         ([(-0.05,-0.05),(0.40,-0.05),(-0.05,0.65)],M,44),
         ([(0.25,0.20),(0.75,0.20),(0.50,0.95)],L,36)],
        # v=2 indie_romance — 3 asymmetric alternative-feel triangles
        [([(1.05,0.08),(0.22,0.45),(1.05,0.88)],L,54),
         ([(-0.05,0.28),(0.65,0.62),(-0.05,1.05)],D,48),
         ([(0.10,-0.05),(1.05,-0.05),(0.62,0.55)],M,38)],
        # v=3 autumn_mix — falling leaf arrangement, overlapping across canvas
        [([(0.00,0.00),(0.50,0.00),(0.20,0.55)],L,52),
         ([(0.50,0.00),(1.05,0.00),(0.80,0.60)],D,48),
         ([(0.10,0.45),(0.60,0.45),(0.35,1.05)],M,42),
         ([(0.55,0.40),(1.05,0.40),(0.82,1.05)],L,36),
         ([(-0.05,0.20),(0.28,0.20),(-0.05,0.75)],D,28)],
        # v=4 heartbreak — many sharp small triangles, shattered feel
        [([(0.10,0.00),(0.45,0.00),(0.25,0.38)],L,54),
         ([(0.55,0.00),(0.90,0.00),(0.72,0.40)],D,50),
         ([(0.00,0.42),(0.32,0.42),(0.15,0.80)],M,46),
         ([(0.35,0.38),(0.65,0.38),(0.50,0.75)],L,40),
         ([(0.68,0.44),(1.00,0.44),(0.85,0.82)],D,36),
         ([(0.20,0.78),(0.55,0.78),(0.38,1.05)],M,30)],
        # v=5 driving_mix — forward-pointing right, motion arrows
        [([(-0.05,0.10),(0.65,0.50),(-0.05,0.90)],L,55),
         ([(0.10,0.00),(0.80,0.50),(0.10,1.00)],D,48),
         ([(0.40,0.05),(1.05,0.50),(0.40,0.95)],M,40),
         ([(0.70,0.15),(1.05,0.50),(0.70,0.85)],L,32)],
        # v=6 after_work — sharp vertical slash, transition feel
        [([(0.42,-0.05),(0.60,-0.05),(0.52,1.05)],L,55),
         ([(0.60,-0.05),(1.05,-0.05),(1.05,0.55)],D,50),
         ([(-0.05,0.45),( 0.42,-0.05),(-0.05,-0.05)],M,44),
         ([(0.18,-0.05),(0.42,-0.05),(0.30,0.55)],L,36)],
    ]
    # Fix negative literal syntax issue in v=5
    if v == 5:
        tri_defs = [
            ([(0.00,0.10),(0.65,0.50),(0.00,0.90)],L,55),
            ([(0.10,0.00),(0.80,0.50),(0.10,1.00)],D,48),
            ([(0.40,0.05),(1.05,0.50),(0.40,0.95)],M,40),
            ([(0.70,0.15),(1.05,0.50),(0.70,0.85)],L,32),
        ]
    else:
        tri_defs = variants[min(v, len(variants) - 1)]
    for pts_frac, color, alpha in tri_defs:
        if rng:
            pts = [((xf + rng.uniform(-0.04, 0.04)) * w, (yf + rng.uniform(-0.04, 0.04)) * h) for xf, yf in pts_frac]
        else:
            pts = [(xf * w, yf * h) for xf, yf in pts_frac]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_diamond_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Overlapping diamond/rhombus shapes. v changes arrangement, rotation and size."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=45): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.55): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    def _diamond_pts(cx, cy, rw, rh, rot_deg=0):
        """Diamond (rhombus) optionally rotated by rot_deg around its centre."""
        r = math.radians(rot_deg)
        base = [(0, -rh), (rw, 0), (0, rh), (-rw, 0)]
        return [(cx + bx*math.cos(r) - by*math.sin(r),
                 cy + bx*math.sin(r) + by*math.cos(r)) for bx, by in base]
    # Each variant: list of (cx_f, cy_f, rw_f, rh_f, rot_deg, alpha, color)
    variants = [
        # v=0 dinner — default 6 diamonds, classic
        [(0.82,0.22,0.60,0.48,0,52,L),(0.20,0.78,0.55,0.44,0,47,D),(0.52,0.52,0.34,0.28,0,40,M),
         (0.08,0.14,0.26,0.22,0,33,L),(0.92,0.88,0.28,0.24,0,28,D),(0.50,0.16,0.16,0.14,0,24,M)],
        # v=1 string_quartet — rotated 18° (formal, tilted elegance)
        [(0.80,0.20,0.62,0.50,18,52,L),(0.22,0.80,0.56,0.46,18,47,D),(0.52,0.52,0.36,0.30,18,40,M),
         (0.10,0.12,0.28,0.24,18,33,L),(0.90,0.88,0.30,0.26,18,28,D),(0.50,0.18,0.18,0.16,18,24,M)],
        # v=2 date_night — tighter, smaller, more numerous
        [(0.78,0.20,0.42,0.34,0,50,L),(0.22,0.78,0.40,0.32,0,46,D),(0.52,0.52,0.28,0.22,0,42,M),
         (0.08,0.12,0.22,0.18,0,36,L),(0.92,0.88,0.24,0.20,0,30,D),
         (0.50,0.14,0.14,0.12,0,26,M),(0.30,0.42,0.18,0.15,0,22,L),(0.72,0.60,0.16,0.14,0,18,D)],
        # v=3 after_dark — 3 large, bold, club aesthetic
        [(0.78,0.22,0.75,0.60,8,56,L),(0.22,0.78,0.70,0.56,8,50,D),(0.50,0.50,0.45,0.36,8,40,M)],
    ]
    diamond_defs = variants[min(v, len(variants) - 1)]
    for cx_f, cy_f, rw_f, rh_f, rot, alpha, color in diamond_defs:
        if rng:
            cx_f = cx_f + rng.uniform(-0.04, 0.04)
            cy_f = cy_f + rng.uniform(-0.04, 0.04)
            rot  = rot  + rng.uniform(-8, 8)
        pts = _diamond_pts(cx_f * w, cy_f * h, rw_f * w, rh_f * h, rot)
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_starburst_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Rays emanating from the exact centre — sunrise, euphoria, party burst."""
    # (n_rays, inner_r_frac)
    configs = [(14,0.06),(18,0.10),(16,0.14),(10,0.18),(22,0.04)]
    n_rays, inner_r_frac = configs[min(v, len(configs) - 1)]
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    cx, cy = w // 2, h // 2
    outer  = max(w, h) * 1.25
    inner  = inner_r_frac * min(w, h)
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rot_offset = rng.uniform(-15, 15) if rng else 0
    for i in range(n_rays):
        a_mid  = math.radians(i * 360 / n_rays - 90 + rot_offset)
        a_half = math.radians(180 / n_rays * 0.52)
        a0, a1 = a_mid - a_half, a_mid + a_half
        color = L if i % 2 == 0 else D
        alpha = 55 if i % 2 == 0 else 34
        pts = [
            (cx + inner * math.cos(a0), cy + inner * math.sin(a0)),
            (cx + outer * math.cos(a0), cy + outer * math.sin(a0)),
            (cx + outer * math.cos(a1), cy + outer * math.sin(a1)),
            (cx + inner * math.cos(a1), cy + inner * math.sin(a1)),
        ]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_chevrons_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Right-pointing V-chevron bands stacked vertically — forward momentum."""
    # (n_chevrons, peak_x_frac) — peak is the rightward point of each V
    configs = [(4,0.62),(6,0.58),(8,0.54)]
    n_chevs, peak_f = configs[min(v, len(configs) - 1)]
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=50): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    spacing = h / n_chevs
    half_h  = spacing * 0.44
    peak_x  = peak_f * w + (rng.uniform(-0.04, 0.04) * w if rng else 0)
    for i in range(n_chevs):
        mid_y = spacing * (i + 0.5)
        pts = [
            (-0.05 * w, mid_y - half_h),
            (peak_x,    mid_y),
            (-0.05 * w, mid_y + half_h),
        ]
        color = L if i % 2 == 0 else D
        alpha = 54 if i % 2 == 0 else 38
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_spiral_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Concentric arc-bands offset progressively to create a spiral impression."""
    # (n_arcs, arc_span_deg, direction, base_alpha)
    configs = [(9, 260, 1, 50),(8, 250, -1, 48),(12, 240, 1, 45)]
    n_arcs, span, direction, base_alpha = configs[min(v, len(configs) - 1)]
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    cx, cy  = w // 2, h // 2
    max_r   = min(w, h) * 0.52
    n_pts   = 100
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=45): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.55): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    start_offset = rng.uniform(-20, 20) if rng else 0
    for j in range(n_arcs):
        r_outer = max_r * (j + 1) / n_arcs
        r_inner = max_r * j / n_arcs * 0.90
        start_deg = direction * j * (360 / n_arcs) * 0.65 - 90 + start_offset
        end_deg   = start_deg + direction * span
        alpha = max(18, base_alpha - j * 2)
        color = L if j % 2 == 0 else D
        outer_pts = [(cx + r_outer * math.cos(math.radians(start_deg + k * (end_deg - start_deg) / n_pts)),
                      cy + r_outer * math.sin(math.radians(start_deg + k * (end_deg - start_deg) / n_pts)))
                     for k in range(n_pts + 1)]
        inner_pts = [(cx + r_inner * math.cos(math.radians(start_deg + k * (end_deg - start_deg) / n_pts)),
                      cy + r_inner * math.sin(math.radians(start_deg + k * (end_deg - start_deg) / n_pts)))
                     for k in range(n_pts, -1, -1)]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(outer_pts + inner_pts, fill=(rc, gc, bc, alpha))
    return img


# === NEW genre/mood generators ================================================
# Genre/mood-evocative backgrounds: the SHAPE signals the music so glyph-less covers still
# read right. Same contract as the originals — (w, h, color_top, color_bottom, v, rng) -> RGBA;
# translucent shapes over the gradient; rng jitters each element a few %. The bottom ~22% (title
# bar) and the top-left badge zone are kept calm.

def _make_equalizer_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Vertical EQ / spectrum bars rising from a baseline — electronic, dance, pop, hip-hop, bass."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=72): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # (n_bars, mirror, gap, max_h, min_h) — n_bars monotonic in i so all variants are distinct.
    configs = [(12 + i, i % 4 == 3, 0.18 + (i % 4) * 0.06,
                0.30 if i % 4 == 3 else 0.50 + (i % 3) * 0.05, 0.07 + (i % 3) * 0.03) for i in range(10)]
    n, mirror, gapf, maxh, minh = configs[min(v, len(configs) - 1)]
    base_y = h * 0.72; mid_y = h * 0.42
    seg = w / n; bw = seg * (1 - gapf)
    # WHY: palette-derived L/M/D bars washed out over the gradient (decade_10s feedback). Use a
    # vibrant fixed spectrum instead so every bar reads with strong contrast on any background.
    HUES = [(64, 220, 232), (236, 72, 170), (250, 208, 64), (96, 220, 120),
            (255, 138, 64), (120, 130, 245), (240, 96, 110)]
    for i in range(n):
        ph = abs(0.5 + 0.5 * math.sin(i * 0.9 + v) * math.cos(i * 0.37 + v * 0.5))
        hh = (minh + (maxh - minh) * ph) * h * (1 + (rng.uniform(-0.05, 0.05) if rng else 0))  # low jitter = clean bars
        x0 = i * seg + (seg - bw) / 2
        col = HUES[(i + v) % len(HUES)]      # cycle hues, offset per variant so each cover differs
        alpha = 215 if i % 2 == 0 else 185   # bright + near-opaque so bars pop
        if mirror:
            draw.rectangle([x0, mid_y - hh / 2, x0 + bw, mid_y + hh / 2], fill=(*col, alpha))
        else:
            draw.rectangle([x0, base_y - hh, x0 + bw, base_y], fill=(*col, alpha))
    return img


def _make_grid_perspective_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Vanishing-point wireframe floor + horizon band — synthwave, vaporwave, techno, 80s."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=85): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    # (n_verticals, horizon_frac, line_alpha)
    configs = [(13, 0.46, 70), (17, 0.42, 60), (11, 0.50, 78), (21, 0.40, 54)]
    n_v, hz_f, alpha = configs[min(v, len(configs) - 1)]
    vp_x = w * 0.5 + (rng.uniform(-0.05, 0.05) * w if rng else 0)
    hz_y = h * hz_f
    lw   = max(2, w // 360)
    for k in range(6):                                   # soft horizon glow
        a = int(alpha * (0.5 - k * 0.07))
        if a > 0:
            draw.rectangle([0, hz_y - k * 4, w, hz_y + k * 4], fill=(*L, a))
    for i in range(-n_v, n_v + 1):                       # converging verticals (floor)
        x_bottom = vp_x + i * (w / n_v) * 1.5
        draw.line([(vp_x, hz_y), (x_bottom, h)], fill=(*L, alpha), width=lw)
    y, step = hz_y + 6, (h - hz_y) * 0.06                # perspective horizontals (floor)
    while y < h:
        draw.line([(0, y), (w, y)], fill=(*L, int(alpha * 0.8)), width=lw)
        y += step; step *= 1.32
    # mirror a ceiling grid UPWARD so the detail fills higher up the cover
    for i in range(-n_v, n_v + 1):
        draw.line([(vp_x, hz_y), (vp_x + i * (w / n_v) * 1.5, 0)], fill=(*L, int(alpha * 0.62)), width=lw)
    y, step = hz_y - 6, hz_y * 0.06
    while y > 0:
        draw.line([(0, y), (w, y)], fill=(*L, int(alpha * 0.5)), width=lw)
        y -= step; step *= 1.32
    return img


def _make_circuit_background(w, h, color_top, color_bottom, v=0, rng=None):
    """PCB traces with right-angle bends + node pads — chiptune, gaming, industrial, IDM, dnb."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=82): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    rng = rng or random.Random(v)
    configs = [(7, 12), (10, 14), (5, 10), (13, 16)]     # (n_traces, grid)
    n_traces, grid = configs[min(v, len(configs) - 1)]
    cell = w / grid
    lw = max(2, w // 320)
    node = lambda gx, gy: (gx * cell + cell / 2, gy * cell + cell / 2)
    for t in range(n_traces):
        gx, gy = rng.randint(0, grid - 1), rng.randint(0, grid - 1)
        pts = [node(gx, gy)]
        for _ in range(rng.randint(3, 6)):
            if rng.random() < 0.5:
                gx = max(0, min(grid - 1, gx + rng.choice([-3, -2, 2, 3])))
            else:
                gy = max(0, min(grid - 1, gy + rng.choice([-3, -2, 2, 3])))
            x2, y2 = node(gx, gy)
            pts.append((x2, pts[-1][1])); pts.append((x2, y2))    # right-angle bend
        a = 46 + (t % 2) * 16
        draw.line(pts, fill=(*L, a), width=lw, joint="curve")
        for (px, py) in pts[::2]:
            r = cell * 0.16
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(*L, min(255, a + 22)))
    return img


def _make_waveform_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Centred oscilloscope waveform lines, edge-tapered — electronic, ambient, downtempo, synth."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=88): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.55): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    configs = [(5, 0.14, 2.0), (7, 0.11, 3.0), (4, 0.18, 1.4), (6, 0.13, 4.5)]  # (n_lines, amp, freq) — denser
    n_lines, ampf, freq = configs[min(v, len(configs) - 1)]
    mid = h * 0.42; lw = max(3, w // 260)
    ph0 = rng.uniform(0, 6.28) if rng else 0
    for li in range(n_lines):
        amp = ampf * h * (1 - li * 0.12); ph = ph0 + li * 0.8
        col = L if li % 2 == 0 else D
        a = 72 if li % 2 == 0 else 46
        pts = []
        for px in range(0, w + 1, 8):
            t = px / w
            y = mid + amp * math.sin(math.pi * t) * math.sin(freq * 2 * math.pi * t + ph)
            pts.append((px, y))
        draw.line(pts, fill=(*col, a), width=lw, joint="curve")
    return img


def _make_laser_fan_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Thin bright beams fanning up from the lower edge — club lasers; EDM, trance, party, garage."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=96): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    configs = [(9 + i, [0.50, 0.22, 0.78, 0.40, 0.60][i % 5],
                [0.96, 0.92, 0.94][i % 3]) for i in range(8)]   # (n,ox,oy), n monotonic
    n, ox, oy = configs[min(v, len(configs) - 1)]
    cx, cy = w * ox, h * oy
    spread = math.radians(150)
    rot = rng.uniform(-0.12, 0.12) if rng else 0
    outer = max(w, h) * 1.4
    for i in range(n):
        a_mid = -math.pi / 2 + (i / (n - 1) - 0.5) * spread + rot
        bw = math.radians(1.7)
        a0, a1 = a_mid - bw, a_mid + bw
        pts = [(cx, cy),
               (cx + outer * math.cos(a0), cy + outer * math.sin(a0)),
               (cx + outer * math.cos(a1), cy + outer * math.sin(a1))]
        draw.polygon(pts, fill=(*L, 60 if i % 2 == 0 else 38))
    r = w * 0.03
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*L, 110))
    return img


def _make_concentric_pulse_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Many thin concentric rings — a sound/sonar pulse; deep house, bass, techno, ambient.
    Distinct from `circles` (few thick outline rings): many thin rings fading outward."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=82): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    configs = [(14 + i, [0.50, 0.32, 0.68, 0.42, 0.58][i % 5],
                [0.44, 0.40, 0.50, 0.46][i % 4]) for i in range(12)]   # (n,cx,cy), n monotonic
    n, cxf, cyf = configs[min(v, len(configs) - 1)]
    cx = w * cxf + (rng.uniform(-0.03, 0.03) * w if rng else 0); cy = h * cyf
    max_r = max(w, h) * 0.95; lw = max(2, w // 300)
    for i in range(1, n + 1):
        r = max_r * i / n
        a = max(14, int(90 * (1 - i / n)))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*L, a), width=lw)
    return img


def _make_low_poly_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Tessellated faceted-gem triangle mesh, per-facet tone — indie, alt-rock, modern electronic.
    WHY: gives alt/indie its own faceted family so the old recoloured-triangle twins diverge."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _c01(x): return max(0.0, min(1.0, x))
    def _lighten(c, amt=74): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.46): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    configs = [(5 + (i % 6), 5 + (i % 6) + (i // 6)) for i in range(10)]   # (cols, rows), all distinct
    cols, rows = configs[min(v, len(configs) - 1)]
    rng = rng or random.Random(v)
    cw, ch = w / cols, h / rows
    pts = [[(c * cw + (0 if c in (0, cols) else rng.uniform(-0.4, 0.4) * cw),
             r * ch + (0 if r in (0, rows) else rng.uniform(-0.4, 0.4) * ch))
            for c in range(cols + 1)] for r in range(rows + 1)]
    def tone(fr):
        fr = _c01(fr)
        return tuple(int(D[k] + (L[k] - D[k]) * fr) for k in range(3))
    for r in range(rows):
        for c in range(cols):
            p00, p10 = pts[r][c], pts[r][c + 1]
            p01, p11 = pts[r + 1][c], pts[r + 1][c + 1]
            f = (r + c) / (rows + cols)
            draw.polygon([p00, p10, p11], fill=(*tone(f + rng.uniform(-0.14, 0.14)), 172))
            draw.polygon([p00, p11, p01], fill=(*tone(f + rng.uniform(-0.14, 0.14)), 172))
    return img


def _make_vinyl_grooves_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Concentric record grooves + off-centre label disc + sheen arc — soul, funk, disco, Motown, R&B, gospel."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=72): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    configs = [(0.62, 0.46, 0.64), (0.50, 0.42, 0.68), (0.38, 0.50, 0.60), (0.66, 0.40, 0.62)]
    cxf, cyf, rmf = configs[min(v, len(configs) - 1)]
    cx = w * cxf + (rng.uniform(-0.02, 0.02) * w if rng else 0); cy = h * cyf
    rmax = min(w, h) * rmf; label_r = rmax * 0.27
    n_g = 26; lw = max(2, int((rmax - label_r) / n_g * 0.6))
    for i in range(n_g):
        r = label_r + (rmax - label_r) * i / n_g
        tone = D if i % 2 == 0 else L
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*tone, 70 if i % 2 == 0 else 30), width=lw)
    draw.ellipse([cx - label_r, cy - label_r, cx + label_r, cy + label_r], fill=(*M, 210))   # label
    hr = label_r * 0.12
    draw.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], fill=(*D, 230))                       # spindle hole
    draw.arc([cx - rmax * 0.92, cy - rmax * 0.92, cx + rmax * 0.92, cy + rmax * 0.92],
             200, 250, fill=(*L, 120), width=max(3, int(rmax * 0.04)))                       # sheen
    return img


def _make_halftone_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Pop-art halftone dot screen, dot size grows across an axis — pop, hyperpop, indie-pop, britpop, ska, 90s."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=88): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    configs = [(12 + i * 2, ['y', 'x', 'radial', 'yinv'][i % 4]) for i in range(8)]   # (cols, axis), cols monotonic
    cols, axis = configs[min(v, len(configs) - 1)]
    sp = w / cols; rows = int(h / sp) + 1
    for j in range(rows + 1):
        for i in range(cols + 1):
            cx, cy = i * sp, j * sp
            if axis == 'y':      p = cy / h
            elif axis == 'yinv': p = 1 - cy / h
            elif axis == 'x':    p = cx / w
            else:                p = 1 - math.hypot(cx - w / 2, cy - h / 2) / math.hypot(w / 2, h / 2)
            rad = sp * (0.10 + 0.42 * max(0.0, min(1.0, p)))
            if rad >= 1:
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(*L, 72))
    return img


def _make_brushstrokes_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Broad painterly diagonal sweeps with rounded ends — jazz, blues, expressive/emotional moods."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=66): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # WHY procedural: many distinct stroke arrangements (one per v) without a huge literal table.
    # (cx_f, cy_f, len_f, thick_f, angle, alpha, color) generated from v.
    tones = [L, M, D]
    n_strokes = 2 + (v % 4)
    strokes = []
    for s in range(n_strokes):
        strokes.append((0.42 + 0.06 * ((v + s) % 3),
                        0.26 + (0.50 / max(1, n_strokes)) * s + 0.04 * (v % 3),
                        1.30 - 0.06 * s, 0.20 - 0.025 * s,
                        -16 - ((v + s) % 5) * 4, max(30, 60 - s * 8), tones[s % 3]))
    jit = rng.uniform(-3, 3) if rng else 0
    for cx_f, cy_f, lenf, thf, ang, a, col in strokes:
        cx, cy = cx_f * w, cy_f * h
        length, thick = lenf * w, thf * h
        draw.polygon(_rotated_rect_points(cx, cy, length, thick, ang + jit), fill=(*col, a))
        rad = thick / 2; ra = math.radians(ang + jit)
        ex, ey = math.cos(ra) * length / 2, math.sin(ra) * length / 2
        for sgn in (-1, 1):
            ecx, ecy = cx + sgn * ex, cy + sgn * ey
            draw.ellipse([ecx - rad, ecy - rad, ecx + rad, ecy + rad], fill=(*col, a))
    return img


def _make_smoke_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Wide vertical haze ribbons, gaussian-softened — psych, stoner, trip-hop, lofi, late-night, jazz-dinner.
    Distinct from `aurora` (horizontal ribbons): tall vertical columns that waver."""
    img     = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=60): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    configs = [(2 + i % 5, 0.07 + (i % 4) * 0.02, 1.0 + i * 0.25) for i in range(10)]  # (n,amp,freq), freq monotonic
    n, ampf, freq = configs[min(v, len(configs) - 1)]
    ph0 = rng.uniform(0, 6.28) if rng else 0
    for ri in range(n):
        cx = (ri + 0.5) / n * w; amp = ampf * w
        halfw = w * (0.10 + 0.03 * (ri % 2)); ph = ph0 + ri * 1.1
        left, right = [], []
        for py in range(0, h + 1, 10):
            off = amp * math.sin(freq * 2 * math.pi * (py / h) + ph)
            left.append((cx + off - halfw, py)); right.append((cx + off + halfw, py))
        draw.polygon(left + right[::-1], fill=(*L, 40 if ri % 2 == 0 else 28))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=14))
    return Image.alpha_composite(img, overlay)


# _make_marquee_lights_background retired (round 4) — family removed; blues_bar reassigned to smoke.


def _make_staff_lines_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Musical staff lines with a few note heads — classical, strings, orchestral, chamber."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=82): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    rng = rng or random.Random(v)
    configs = [(2, 5), (3, 5), (1, 5), (2, 4)]                # (n_staves, lines_per)
    n_staves, lpl = configs[min(v, len(configs) - 1)]
    lw = max(2, w // 400)
    region_top, region_bot = h * 0.12, h * 0.70
    stave_gap = (region_bot - region_top) / n_staves
    line_gap = stave_gap * 0.16
    for s in range(n_staves):
        top = region_top + s * stave_gap + stave_gap * 0.2
        ys = [top + k * line_gap for k in range(lpl)]
        for y in ys:
            draw.line([(w * 0.04, y), (w * 0.96, y)], fill=(*L, 64), width=lw)
        # note heads SNAPPED to the line/space grid (not floating) + a stem, so they read as notation
        grid = ys + [(ys[k] + ys[k + 1]) / 2 for k in range(len(ys) - 1)]
        nr = line_gap * 0.58
        placed = []
        tries = 0
        while len(placed) < rng.randint(3, 5) and tries < 60:
            tries += 1
            nx = rng.uniform(w * 0.14, w * 0.88); ny = rng.choice(grid)
            if any(abs(nx - px) < nr * 4 and abs(ny - py) < nr * 2.2 for px, py in placed):
                continue                                  # keep note-heads from overlapping
            placed.append((nx, ny))
            draw.ellipse([nx - nr * 1.25, ny - nr * 0.9, nx + nr * 1.25, ny + nr * 0.9], fill=(*L, 175))
            draw.line([(nx + nr * 1.15, ny), (nx + nr * 1.15, ny - line_gap * 3.0)], fill=(*L, 150), width=max(2, lw))
    return img


def _make_mountains_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Layered mountain-range silhouettes receding back→front — folk, acoustic, country, celtic, outdoors."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _c01(x): return max(0.0, min(1.0, x))
    def _lighten(c, amt=70): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    configs = [(3 + i % 3, 2 + i % 4) for i in range(6)]   # (n_layers, base_peaks), distinct
    n_layers, base_pk = configs[min(v, len(configs) - 1)]
    def tone(fr):
        fr = _c01(fr); return tuple(int(L[k] + (D[k] - L[k]) * fr) for k in range(3))
    for li in range(n_layers):
        frac = li / (n_layers - 1) if n_layers > 1 else 0
        base_y = h * (0.40 + 0.34 * frac); peak_h = h * (0.12 + 0.16 * (1 - frac))
        npk = base_pk + li; seg = w / npk
        pts = [(0, h)]
        for k in range(npk + 1):
            ph = peak_h * (0.5 + 0.5 * abs(math.sin(k * 1.3 + li))) * (1 + rng.uniform(-0.15, 0.15))
            pts.append((k * seg, base_y - ph))
        pts.append((w, h))
        draw.polygon(pts, fill=(*tone(frac), min(230, 210 if li == n_layers - 1 else 150 + li * 10)))
    return img


def _make_starfield_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Scattered stars (+ optional nebula) on a deep gradient — ambient, sleep, midnight, dreamy, space."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=120): return tuple(_clamp(x + amt) for x in c[:3])
    L = _lighten(color_top)
    rng = rng or random.Random(v)
    configs = [(80 + i * 14, i % 3 != 0) for i in range(12)]   # (n_stars, nebula), n monotonic
    n_stars, nebula = configs[min(v, len(configs) - 1)]
    if nebula:
        nb = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        nd = ImageDraw.Draw(nb)
        ncx, ncy, nr = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.2, 0.4), w * 0.34
        nd.ellipse([ncx - nr, ncy - nr * 0.6, ncx + nr, ncy + nr * 0.6], fill=(*L, 40))
        img = Image.alpha_composite(img, nb.filter(ImageFilter.GaussianBlur(radius=60)))
    draw = ImageDraw.Draw(img, 'RGBA')
    bar_top = h * 0.74
    for _ in range(n_stars):
        x, y = rng.uniform(0, w), rng.uniform(0, bar_top)
        s = rng.random(); r = 1 if s < 0.70 else (2 if s < 0.93 else 3)
        a = rng.randint(120, 235)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*L, a))
        if r == 3:
            draw.line([(x - 5, y), (x + 5, y)], fill=(*L, max(0, a - 60)), width=1)
            draw.line([(x, y - 5), (x, y + 5)], fill=(*L, max(0, a - 60)), width=1)
    return img


def _make_palm_sunburst_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Retro sun low on the horizon + palm-tree silhouettes — latin, tropical, reggae, afrobeat, beach.
    WHY redrawn: the old corner 'fronds' rendered as stray lines top-left; now real palm-tree shapes."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=92): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.30): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    configs = [(0.50, 0.70, 0.26, 1), (0.62, 0.68, 0.24, 2),
               (0.40, 0.72, 0.28, 1), (0.55, 0.66, 0.22, 2)]   # (sun x,y,r ; n_trees)
    sxf, syf, srf, ntrees = configs[min(v, len(configs) - 1)]
    scx, scy, sr = w * sxf, h * syf, w * srf
    draw.ellipse([scx - sr, scy - sr, scx + sr, scy + sr], fill=(*L, 170))   # retro sun
    for i in range(1, 6):                                                    # slats across the sun
        yy = scy - sr * 0.5 + i * (sr * 0.28)
        draw.rectangle([scx - sr, yy, scx + sr, yy + max(3, sr * 0.06)], fill=(*D, 110))
    for tx, lean in [(w * 0.16, -1), (w * 0.86, 1)][:ntrees]:                # palm-tree silhouettes
        base_y, top_y = h * 0.80, h * 0.42
        crown = (tx + lean * w * 0.05, top_y)
        draw.line([(tx, base_y), ((tx + crown[0]) / 2, (base_y + top_y) / 2), crown],
                  fill=(*D, 185), width=max(4, int(w * 0.013)), joint="curve")
        for k in range(6):                                                   # drooping fronds
            a = math.radians(202 + k * 27)
            fx2, fy2 = crown[0] + math.cos(a) * w * 0.11, crown[1] - math.sin(a) * h * 0.07
            draw.line([crown, (fx2, fy2)], fill=(*D, 175), width=max(3, int(w * 0.006)))
    return img


def _make_sun_horizon_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Sun on a striped horizon — warm retro sunset; country, road-trip, summer, golden-hour."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=90): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    M = _mid(color_top, color_bottom)
    # (sun x, horizon y [LOWER on page], sun r, stripe count) — horizon is lower than before.
    configs = [([0.50, 0.40, 0.60, 0.45, 0.55][i % 5], 0.64 + (i % 4) * 0.025,
                0.20 + (i % 3) * 0.03, 8 + i) for i in range(10)]
    sxf, syf, srf, nb = configs[min(v, len(configs) - 1)]
    hz = h * syf; scx, sr = w * sxf, w * srf
    # Warm horizon glow (soft light rising from the horizon) → a multi-colour sunrise/sunset sky even
    # when the palette top is cool. WHY: the sun was painted with the lightened palette top, so a violet
    # sunrise sky gave a PURPLE sun; the sun is now always warm and a warm glow enriches the gradient.
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gld = ImageDraw.Draw(glow)
    gld.ellipse([scx - sr * 3.4, hz - sr * 2.1, scx + sr * 3.4, hz + sr * 1.3], fill=(255, 168, 92, 130))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=64)))
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.ellipse([scx - sr, hz - sr, scx + sr, hz + sr], fill=(255, 232, 178, 235))            # warm sun
    draw.ellipse([scx - sr * 0.66, hz - sr * 0.66, scx + sr * 0.66, hz + sr * 0.66], fill=(255, 247, 218, 205))  # bright core
    # WHY: uniform full-width stripes from top to horizon, same y-grid across sky AND sun, so the
    # sun's slats line up with the sky stripes (the old per-sun venetian gap looked misaligned).
    slot = hz / nb
    for i in range(nb):
        yb = i * slot
        col = L if i % 2 == 0 else M
        draw.rectangle([0, yb, w, yb + slot * 0.55], fill=(*col, 64 if i % 2 == 0 else 46))
    draw.rectangle([0, hz, w, h], fill=(*D, 120))             # darker ground band
    return img


def _make_woodgrain_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Flowing woodgrain lines + a knot — warm acoustic, cosy, folk, campfire, singer-songwriter."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=58): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    configs = [10 + i for i in range(12)]   # n_lines, monotonic (WHY: dropped the 'knot' ovals —
    n_lines = configs[min(v, len(configs) - 1)]                       # they read as stray little circles)
    for i in range(n_lines):
        base_y = h * (i + 0.5) / n_lines; amp = h * 0.02 * (1 + i % 3); ph = rng.uniform(0, 6.28)
        pts = [(x, base_y + amp * math.sin(x * 0.012 + ph)) for x in range(0, w + 1, 12)]
        draw.line(pts, fill=(*(L if i % 2 else D), 40), width=max(2, h // 240))
    return img


def _make_amp_stack_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A backline of guitar AMP STACKS — head (with knobs) + speaker cabinet(s) with cone grilles.
    WHY redrawn: the old concentric cones read as gas hob-burners from above; this is a literal amp."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=60): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.42): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # (centre_x, n_cabs, cab_width_frac) — a Hughes & Kettner-style stack: a head NARROWER than the 4×12
    # cab(s) below it, each cab exactly 2×2 = 4 speaker cones. WHY 8 configs: amp_stack is _BG_SYMMETRIC.
    configs = [(0.50, 2, 0.44), (0.50, 1, 0.48), (0.46, 2, 0.40), (0.54, 1, 0.44),
               (0.50, 2, 0.48), (0.45, 1, 0.42), (0.55, 2, 0.42), (0.50, 1, 0.50)]
    cxf, cabs, wf = configs[min(v, len(configs) - 1)]
    cab_w = w * wf; cx = w * cxf; sx = cx - cab_w / 2
    cab_dark = _darken(color_bottom, 0.26)            # near-black cabinet vinyl
    grille   = _darken(color_bottom, 0.40)            # grille cloth (a touch lighter)
    metal    = _lighten(color_top, 70)                # head control panel / piping
    hk_blue  = (60, 140, 235)                         # H&K signature power glow
    top_y, bot_y = h * 0.10, h * 0.74
    span = bot_y - top_y
    head_hf, gpf = 0.28, 0.045                                 # head height & gaps as fractions of cab_w
    # WHY: keep each cab SQUARE (a real 4×12 is ~square) so the 2×2 cones fill the MAJORITY of the face.
    # Scale cab_w down if a tall (2-cab) stack wouldn't fit the vertical span — never leave a wide/short cab.
    denom = head_hf + gpf + cabs + (cabs - 1) * gpf
    cab_w = min(w * wf, span / denom)
    cab_h = cab_w
    head_h = cab_w * head_hf; gp = cab_w * gpf
    stack_h = head_h + gp + cabs * cab_h + (cabs - 1) * gp
    sx = cx - cab_w / 2
    top = top_y + (span - stack_h) / 2                         # centre the stack vertically
    rad = max(5, int(cab_w * 0.035)); lwd = max(2, int(w * 0.004))
    def rrect(box, fill, outline=None):
        try:
            draw.rounded_rectangle(box, radius=rad, fill=fill, outline=outline, width=lwd)
        except Exception:
            draw.rectangle(box, fill=fill, outline=outline, width=lwd)
    draw.ellipse([sx - cab_w * 0.08, top + stack_h - 6, sx + cab_w * 1.08, top + stack_h + 22], fill=(0, 0, 0, 70))  # floor shadow
    # --- amp head (narrower than the cab) ---
    hw = cab_w * 0.80; hx = cx - hw / 2
    rrect([hx, top, hx + hw, top + head_h], (*cab_dark, 235), (*metal, 150))
    draw.rectangle([hx + hw * 0.10, top + head_h * 0.20, hx + hw * 0.90, top + head_h * 0.52], fill=(*metal, 205))  # control panel
    py = top + head_h * 0.36
    for k in range(6):                                                           # knobs on the panel
        kx = hx + hw * (0.17 + 0.66 * k / 5); kr = head_h * 0.11
        draw.ellipse([kx - kr, py - kr, kx + kr, py + kr], fill=(*cab_dark, 235), outline=(*metal, 220))
    draw.rectangle([hx + hw * 0.10, top + head_h * 0.66, hx + hw * 0.32, top + head_h * 0.82], fill=(*hk_blue, 190))  # power glow
    # --- cabinet(s): each a 2×2 (= 4 cone) 4×12 ---
    cy0 = top + head_h + gp
    for cab in range(cabs):
        cby0 = cy0 + cab * (cab_h + gp); cby1 = cby0 + cab_h
        rrect([sx, cby0, sx + cab_w, cby1], (*cab_dark, 235), (*metal, 130))
        inset = cab_w * 0.04
        gx0, gy0, gx1, gy1 = sx + inset, cby0 + inset, sx + cab_w - inset, cby1 - inset
        draw.rectangle([gx0, gy0, gx1, gy1], fill=(*grille, 225))                                      # grille cloth
        for px, py2 in [(sx, cby0), (sx + cab_w, cby0), (sx, cby1), (sx + cab_w, cby1)]:               # small corner caps
            cs = cab_w * 0.03
            draw.rectangle([px - cs, py2 - cs, px + cs, py2 + cs], fill=(*metal, 165))
        cellw, cellh = (gx1 - gx0) / 2, (gy1 - gy0) / 2
        cr = min(cellw, cellh) * 0.49                                                                  # 12" speakers fill the majority of the cab
        cone, cone_d, cap = _darken(color_bottom, 0.20), _darken(color_bottom, 0.10), _darken(color_bottom, 0.30)
        for rr in range(2):
            for cc in range(2):
                ccx, ccy = gx0 + cellw * (cc + 0.5), gy0 + cellh * (rr + 0.5)
                draw.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=(*cone_d, 245), outline=(*metal, 150), width=max(2, int(cr * 0.08)))  # frame
                for sd in range(8):                                                                    # mounting screws round the frame
                    sa = math.radians(sd * 45 + 22)
                    sxx, syy = ccx + math.cos(sa) * cr * 0.9, ccy + math.sin(sa) * cr * 0.9
                    draw.ellipse([sxx - cr * 0.05, syy - cr * 0.05, sxx + cr * 0.05, syy + cr * 0.05], fill=(*metal, 175))
                draw.ellipse([ccx - cr * 0.82, ccy - cr * 0.82, ccx + cr * 0.82, ccy + cr * 0.82], fill=(*cone, 245))                            # cone
                draw.ellipse([ccx - cr * 0.30, ccy - cr * 0.30, ccx + cr * 0.30, ccy + cr * 0.30], fill=(*cap, 245), outline=(*metal, 120))      # dust cap
                draw.ellipse([ccx - cr * 0.26, ccy - cr * 0.34, ccx - cr * 0.02, ccy - cr * 0.10], fill=(255, 255, 255, 45))                     # highlight
    return img


def _make_shards_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Sharp angular shards radiating from a point — aggressive; punk, metal, industrial, hardcore."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=70): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.46): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    configs = [(7 + i, [0.50, 0.30, 0.70, 0.42][i % 4],
                [0.30, 0.25, 0.35, 0.28][i % 4]) for i in range(8)]   # (n,ox,oy), n monotonic
    n, oxf, oyf = configs[min(v, len(configs) - 1)]
    ox, oy = w * oxf, h * oyf; outer = max(w, h) * 1.3
    for i in range(n):
        a_mid = i * 2 * math.pi / n + rng.uniform(-0.10, 0.10)
        spread = rng.uniform(0.04, 0.12); ln = outer * rng.uniform(0.6, 1.0)
        a0, a1 = a_mid - spread, a_mid + spread
        pts = [(ox, oy), (ox + ln * math.cos(a0), oy + ln * math.sin(a0)),
               (ox + ln * math.cos(a1), oy + ln * math.sin(a1))]
        draw.polygon(pts, fill=(*(L if i % 2 else D), 55 if i % 2 else 38))
    return img


def _make_confetti_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Scattered confetti bits (rotated quads + triangles) — party, celebration, festive, cookout."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=84): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.50): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    rng = rng or random.Random(v)
    n = [28, 40, 52, 64, 76, 88, 34, 58][min(v, 7)]
    bar_top = h * 0.74; tones = [L, D, M]
    for _ in range(n):
        x, y = rng.uniform(0, w), rng.uniform(0, bar_top)
        s = rng.uniform(w * 0.012, w * 0.030); ang = rng.uniform(0, 360)
        col = rng.choice(tones); a = rng.randint(60, 150)
        if rng.random() < 0.5:
            pts = _rotated_rect_points(x, y, s * 1.8, s * 0.7, ang)
        else:
            ra = math.radians(ang)
            pts = [(x + s * math.cos(ra + k * 2.094), y + s * math.sin(ra + k * 2.094)) for k in range(3)]
        draw.polygon(pts, fill=(*col, a))
    return img


def _make_tartan_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Real tartan sett — a repeating colour sequence drawn as semi-transparent warp + weft bands so
    the crossings blend into the classic plaid, with red + cream over-stripes. Scotland scene, celtic."""
    img     = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _darken(c, fac): return tuple(int(x * fac) for x in c[:3])
    def _lighten(c, amt): return tuple(_clamp(x + amt) for x in c[:3])
    # Irn-Bru tartan (Scottish Register, Kinloch Anderson) — matched to the real sett: ORANGE-dominant
    # ground, azure-blue bands, white tramline pinstripes with a thin navy line. Hard-pinned (palette-free).
    orange = (200, 76, 24)
    blue   = (26, 118, 176)
    white  = (240, 240, 242)
    navy   = (20, 28, 56)
    unit = w * [0.155, 0.13, 0.185, 0.14][min(v, 3)]
    # sett (width × unit, colour, alpha) — big orange block, W-navy-W tramline, orange, BLUE band with a
    # W-orange-W centre tramline, orange, tramline → repeats. WHY orange is low-alpha: the bg gradient is
    # already orange (so orange BANDS are skipped, alpha 0); the blue/white/navy draw as a GRID in BOTH
    # directions. WHY skip orange + asymmetric warp/weft alpha: drawing opaque orange weft erased the
    # vertical blue warp → horizontal-stripe bug. Now warp (vertical) is full and weft (horizontal) is
    # lighter, so BOTH axes read and the blue∩blue crossings build solid blue squares = a real plaid.
    sett = [(1.00, orange, 0),
            (0.05, white, 235), (0.035, navy, 220), (0.05, white, 235),
            (0.42, orange, 0),
            (0.28, blue, 215), (0.05, white, 235), (0.035, orange, 170), (0.05, white, 235), (0.28, blue, 215),
            (0.42, orange, 0),
            (0.05, white, 235), (0.035, navy, 220), (0.05, white, 235)]
    def stripes(horizontal, amul):
        pos = 0; limit = h if horizontal else w
        while pos < limit:
            for wf, col, a in sett:
                bw = unit * wf
                if a > 0:
                    aa = max(1, int(a * amul))
                    if horizontal:
                        draw.rectangle([0, pos, w, pos + bw], fill=(*col, aa))
                    else:
                        draw.rectangle([pos, 0, pos + bw, h], fill=(*col, aa))
                pos += bw
    stripes(False, 1.0)      # warp (vertical) full strength
    stripes(True, 0.78)      # weft (horizontal) lighter → both directions read; crossings blend = plaid
    return Image.alpha_composite(img, overlay)


def _make_cityscape_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A dusk skyline — a DISTINCT sky (palette gradient + a warm city-lights glow on the horizon) with
    near-black building silhouettes and warm lit windows. London/Australia scenes, urban genres.
    WHY redrawn: mid-tone buildings blended into the sky; dark silhouettes + a glow read as a real city."""
    img  = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=110): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top)
    rng = rng or random.Random(v)
    n = [12, 16, 9, 14][min(v, 3)]
    base_y = h * 0.72
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))                                    # warm city-lights haze
    ImageDraw.Draw(glow).ellipse([-w * 0.1, base_y - h * 0.20, w * 1.1, base_y + h * 0.06], fill=(*L, 95))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=55)))
    draw = ImageDraw.Draw(img, 'RGBA')
    sil = _darken(color_bottom, 0.20)                                                 # near-black buildings
    win = (255, 214, 130)                                                             # warm lit windows
    bw = w / n; wsz = max(3, int(bw * 0.12))
    for i in range(n):
        bh = min(h * 0.58, h * (0.14 + 0.46 * abs(math.sin(i * 1.7 + v))))
        x0, x1 = i * bw, i * bw + bw * 0.94; top = base_y - bh
        draw.rectangle([x0, top, x1, base_y], fill=(*sil, 245))
        if rng.random() < 0.4:                                                        # some have a roof spire/tower
            sx = (x0 + x1) / 2
            draw.rectangle([sx - wsz * 0.4, top - bh * 0.10, sx + wsz * 0.4, top], fill=(*sil, 245))
        wy = top + wsz * 1.8
        while wy < base_y - wsz:
            wx = x0 + wsz
            while wx < x1 - wsz:
                if rng.random() < 0.5:
                    draw.rectangle([wx, wy, wx + wsz, wy + wsz], fill=(*win, rng.randint(120, 220)))
                wx += wsz * 2.0
            wy += wsz * 2.2
    draw.rectangle([0, base_y, w, h], fill=(*_darken(color_bottom, 0.14), 235))       # dark street
    return img


# === NEW families, round 2 (genre / mood / season) ============================

def _make_mod_target_background(w, h, color_top, color_bottom, v=0, rng=None):
    """RAF-roundel / mod target — concentric bullseye (Mod, Britpop, Madchester)."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=95): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    configs = [(0.50, 0.42, 0.46), (0.62, 0.40, 0.42), (0.38, 0.46, 0.44), (0.50, 0.38, 0.40)]
    cxf, cyf, rf = configs[min(v, len(configs) - 1)]
    cx, cy, R = w * cxf, h * cyf, min(w, h) * rf
    # Real RAF roundel: EXACTLY three concentric rings — outer blue, white, inner red. Nothing else.
    # WHY: the previous 4th (white centre dot) made it read as a dartboard, not the RAF/Mod target.
    blue, white, red = (24, 64, 156), (238, 238, 242), (206, 42, 52)
    for frac, col in [(1.0, blue), (0.62, white), (0.30, red)]:
        rr = R * frac
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(*col, 240))
    return img


def _make_disco_ball_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Faceted mirror-ball + sparkles — disco, funk, Motown, dance, party."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=95): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    rng = rng or random.Random(v)
    configs = [(0.50, 0.40, 0.30), (0.62, 0.38, 0.26), (0.40, 0.42, 0.28), (0.50, 0.36, 0.32)]
    cxf, cyf, rf = configs[min(v, len(configs) - 1)]
    cx, cy, R = w * cxf, h * cyf, min(w, h) * rf
    draw.line([(cx, cy - R), (cx, h * 0.02)], fill=(*L, 120), width=max(2, int(R * 0.02)))    # hang wire
    draw.ellipse([cx - R * 0.06, cy - R - R * 0.10, cx + R * 0.06, cy - R + R * 0.02], fill=(*L, 190))  # mount
    # WHY rewrite: square glitter tiles with random alpha read as flat noise. Shade each mirror tile as a
    # point on a sphere lit from the upper-left (silver ramp + a few coloured-light reflections) → a real ball.
    ball = Image.new("RGBA", (w, h), (0, 0, 0, 0)); bd = ImageDraw.Draw(ball, 'RGBA')
    bd.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(24, 26, 40, 255))          # dark grout between tiles
    SIL = [(40, 44, 60), (110, 122, 144), (188, 200, 220), (248, 250, 255)]       # shadow→highlight silver
    casts = [L, M, (236, 96, 150), (96, 200, 220), (250, 210, 90)]                # coloured-light reflections
    lx, ly, lz = -0.50, -0.55, 0.67                                               # light from upper-left
    cell = R * 0.135
    row, yy = 0, cy - R - cell
    while yy < cy + R + cell:
        off = (cell * 0.5) if row % 2 else 0.0                                    # brick-offset rows = real mirror tiles
        xx = cx - R - cell + off
        while xx < cx + R + cell:
            mx, my = xx + cell / 2, yy + cell / 2
            dx, dy = (mx - cx) / R, (my - cy) / R
            r2 = dx * dx + dy * dy
            if r2 < 1.10:                                                         # draw past the rim; the mask clips it
                z = math.sqrt(max(0.0, 1.0 - min(1.0, r2)))
                b = 0.15 + 0.85 * max(0.0, dx * lx + dy * ly + z * lz)            # sphere brightness
                base = SIL[0] if b < 0.32 else SIL[1] if b < 0.58 else SIL[2] if b < 0.84 else SIL[3]
                if rng.random() < 0.18:
                    cc = rng.choice(casts); base = tuple((base[k] + cc[k]) // 2 for k in range(3))
                if rng.random() < 0.07 and b > 0.45:
                    base = (252, 253, 255)                                        # occasional bright flash
                bd.rectangle([xx + 0.6, yy + 0.6, xx + cell - 1.0, yy + cell - 1.0], fill=(*base, 255))
            xx += cell
        yy += cell; row += 1
    cmask = Image.new("L", (w, h), 0)                                             # clip tiles to the EXACT rim
    ImageDraw.Draw(cmask).ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)     # → tiles reach the edge, no gap
    ball.putalpha(cmask)
    img = Image.alpha_composite(img, ball)
    spec = Image.new("RGBA", (w, h), (0, 0, 0, 0)); sd = ImageDraw.Draw(spec)
    hx, hy = cx - R * 0.34, cy - R * 0.36; hr = R * 0.24
    sd.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(255, 255, 255, 150))   # specular hotspot
    img = Image.alpha_composite(img, spec.filter(ImageFilter.GaussianBlur(radius=max(2, int(R * 0.10)))))
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.arc([cx - R, cy - R, cx + R, cy + R], 20, 150, fill=(8, 8, 18, 120), width=max(2, int(R * 0.05)))  # shaded underside
    for _ in range(5):                                                            # light-ray sparkles around the ball
        a = rng.uniform(0, 6.28); rr = R * rng.uniform(1.06, 1.5)
        sx, sy = cx + math.cos(a) * rr, cy + math.sin(a) * rr
        if 0 < sx < w and 0 < sy < h * 0.7:
            gl = R * rng.uniform(0.07, 0.13)
            draw.line([(sx - gl, sy), (sx + gl, sy)], fill=(255, 255, 255, 150), width=2)
            draw.line([(sx, sy - gl), (sx, sy + gl)], fill=(255, 255, 255, 150), width=2)
    return img


def _make_cassette_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Cassette tape — body, reels, tape window — retro mixtape (hip-hop, throwback, lofi)."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=80): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    # WHY 8 configs: cassette is _BG_SYMMETRIC (no h-flip doubling) and now also carries swagger — needs ≥7.
    configs = [(0.50, 0.40, 1.0), (0.50, 0.42, 0.86), (0.46, 0.38, 1.05), (0.54, 0.40, 0.92),
               (0.48, 0.44, 0.95), (0.52, 0.36, 1.0), (0.50, 0.38, 0.90), (0.50, 0.43, 1.02)]
    cxf, cyf, sc = configs[min(v, len(configs) - 1)]
    ACCENTS = [(228, 96, 72), (236, 188, 72), (96, 188, 220), (210, 110, 196),
               (120, 200, 130), (240, 150, 90), (150, 150, 240), (236, 110, 130)]
    accent = ACCENTS[v % len(ACCENTS)]      # a coloured label so mixtapes differ at a glance
    cx, cy = w * cxf, h * cyf
    bw, bh = w * 0.62 * sc, w * 0.40 * sc
    x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    lwd = max(2, int(w * 0.004))
    def rrect(box, fill, outline=None):
        try:
            draw.rounded_rectangle(box, radius=max(4, int(bw * 0.04)), fill=fill, outline=outline, width=lwd)
        except Exception:
            draw.rectangle(box, fill=fill, outline=outline, width=lwd)
    rrect([x0, y0, x1, y1], (*M, 175), (*L, 190))                                   # body
    rrect([x0 + bw * 0.10, y0 + bh * 0.12, x1 - bw * 0.10, y0 + bh * 0.40], (*accent, 180), (*L, 150))  # label
    wy0, wy1 = cy + bh * 0.02, cy + bh * 0.30
    draw.rectangle([x0 + bw * 0.16, wy0, x1 - bw * 0.16, wy1], fill=(*D, 175))      # tape window
    for side in (-1, 1):
        rx, ry, rr = cx + side * bw * 0.18, (wy0 + wy1) / 2, bh * 0.11
        draw.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=(*L, 195))          # reel
        for k in range(6):
            a = math.radians(k * 60)
            draw.line([(rx, ry), (rx + math.cos(a) * rr * 0.8, ry + math.sin(a) * rr * 0.8)], fill=(*D, 170), width=2)
        hr = rr * 0.3
        draw.ellipse([rx - hr, ry - hr, rx + hr, ry + hr], fill=(*D, 205))
    return img


def _make_pixel_grid_background(w, h, color_top, color_bottom, v=0, rng=None):
    """8-bit pixel blocks + a space-invader motif — chiptune, gaming, hyperpop."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=80): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.45): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    rng = rng or random.Random(v)
    cols = [16, 22, 12, 28][min(v, 3)]
    px = w / cols
    # 8-bit pixel field — clusters of lit blocks (reads as pixel-art/chiptune without a literal invader).
    for r in range(int(h * 0.74 / px)):
        for c in range(cols):
            p = rng.random()
            if p < 0.34:
                tone = L if p < 0.12 else (M if p < 0.24 else D)
                draw.rectangle([c * px + 1, r * px + 1, c * px + px - 1, r * px + px - 1],
                               fill=(*tone, rng.randint(55, 150)))
    return img


# _make_flames_background retired (round 4) — family removed; party→laser_fan, punk_energy→guitar, slow_burn→smoke.


def _make_columns_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Neoclassical fluted columns + capitals — classical, strings, orchestral."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=72): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.46): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    n = [4, 5, 3, 6][min(v, 3)]
    top_y, bot_y = h * 0.16, h * 0.72
    seg = w / n; colw = seg * 0.5; cap = (bot_y - top_y) * 0.06
    for i in range(n):
        cx = i * seg + seg / 2; x0, x1 = cx - colw / 2, cx + colw / 2
        draw.rectangle([x0 - colw * 0.14, top_y, x1 + colw * 0.14, top_y + cap], fill=(*L, 155))          # capital
        draw.rectangle([x0 - colw * 0.14, bot_y - cap, x1 + colw * 0.14, bot_y], fill=(*L, 155))          # base
        draw.rectangle([x0, top_y, x1, bot_y], fill=(*L, 95))                                             # shaft
        for f in range(4):
            fx = x0 + colw * (0.2 + 0.2 * f)
            draw.line([(fx, top_y + cap), (fx, bot_y - cap)], fill=(*D, 90), width=max(1, int(colw * 0.05)))  # flute
    return img


def _make_film_strip_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Realistic 35mm film strip(s) — dark film base, two rows of rounded Kodak perforations, and lit
    image frames between them divided by frame lines. Soundtracks, cinematic, main character."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac): return tuple(int(x * fac) for x in c[:3])
    rng = rng or random.Random(v)
    film = (18, 18, 22)                                            # film base (near-black)
    perf = (238, 238, 232)                                        # clear perforations
    frame_tone = _lighten(color_top, 45)                          # the projected image area
    scene = _lighten(color_bottom, 40)
    nst, frames = [(2, 6), (3, 5), (1, 7), (2, 8)][min(v, 3)]
    strip_h = h * (0.26 if nst <= 2 else 0.205)
    centres = {1: [0.50], 2: [0.32, 0.70], 3: [0.22, 0.50, 0.78]}[nst]
    for cyf in centres:
        cy = h * cyf; y0, y1 = cy - strip_h / 2, cy + strip_h / 2
        draw.rectangle([-20, y0, w + 20, y1], fill=(*film, 240))                              # film base
        np = frames * 2
        pw, ph, pr = w / np * 0.42, strip_h * 0.15, max(2, int(strip_h * 0.04))
        for k in range(np):                                                                   # two perforation rows
            px = (k + 0.5) * w / np
            for py in (y0 + strip_h * 0.12, y1 - strip_h * 0.12):
                box = [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2]
                try:
                    draw.rounded_rectangle(box, radius=pr, fill=(*perf, 225))
                except Exception:
                    draw.rectangle(box, fill=(*perf, 225))
        fy0, fy1 = y0 + strip_h * 0.27, y1 - strip_h * 0.27                                   # image frames between perfs
        fw = w / frames
        for k in range(frames):
            fx0, fx1 = k * fw + fw * 0.05, (k + 1) * fw - fw * 0.05
            draw.rectangle([fx0, fy0, fx1, fy1], fill=(*frame_tone, 165))                     # projected frame
            draw.line([(fx0, (fy0 + fy1) / 2), (fx1, (fy0 + fy1) / 2)], fill=(*scene, 90), width=1)  # faint horizon in-frame
    return img


def _make_stained_glass_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Leaded stained-glass panes + arches — gospel & choir."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=85): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    # Rich stained-glass jewel tones (cobalt, ruby, emerald, gold, violet, amber, teal).
    JEWELS = [(42, 78, 178), (172, 36, 64), (34, 142, 92), (224, 172, 52),
              (120, 54, 168), (220, 110, 44), (32, 132, 146)]
    cols, rows = [(4, 5), (5, 6), (3, 4), (6, 7)][min(v, 3)]
    cw, chh = w / cols, h * 0.74 / rows
    lead = max(2, int(w * 0.006))
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw, r * chh
            draw.rectangle([x0, y0, x0 + cw, y0 + chh], fill=(*rng.choice(JEWELS), rng.randint(120, 180)),
                           outline=(*D, 210), width=lead)               # dark leading between panes
    for c in range(cols):                                   # arched tops
        x0 = c * cw
        draw.arc([x0, -chh, x0 + cw, chh], 180, 360, fill=(*L, 170), width=lead)
    return img


# _make_neon_background retired (round 4) — family removed; swagger→cassette, synthwave/vaporwave→grid_perspective.


def _make_candle_glow_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Candle flame(s) + warm halo — candlelight, romantic/jazz dinner, date night."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    flame = Image.new("RGBA", (w, h), (0, 0, 0, 0)); fd = ImageDraw.Draw(flame)
    def _clamp(x): return max(0, min(255, x))
    warm = (255, 220, 150)                                # candlelight is always warm
    L = tuple(_clamp(x + 90) for x in color_top[:3])
    rng = rng or random.Random(v)
    n = [3, 2, 4, 1][min(v, 3)]
    seg = w / (n + 1)
    for i in range(1, n + 1):
        cx = seg * i + rng.uniform(-0.04, 0.04) * w
        cy = h * rng.uniform(0.40, 0.52)
        R = w * 0.16
        gd.ellipse([cx - R, cy - R, cx + R, cy + R], fill=(*warm, 80))            # halo
        fh = h * 0.10
        fd.polygon([(cx, cy - fh), (cx - fh * 0.28, cy - fh * 0.2), (cx, cy + fh * 0.15),
                    (cx + fh * 0.28, cy - fh * 0.2)], fill=(*warm, 230))           # flame
        fd.ellipse([cx - fh * 0.14, cy - fh * 0.4, cx + fh * 0.14, cy], fill=(255, 250, 225, 230))
        cw_ = w * 0.018
        fd.rectangle([cx - cw_, cy + fh * 0.1, cx + cw_, cy + h * 0.22], fill=(*L, 175))  # candle stick
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=40)))
    return Image.alpha_composite(img, flame)


def _make_bokeh_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Soft defocused light orbs — glam, love songs, late-night romance."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    orbs = Image.new("RGBA", (w, h), (0, 0, 0, 0)); od = ImageDraw.Draw(orbs)
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 95) for x in color_top[:3])
    rng = rng or random.Random(v)
    for _ in range([10, 14, 7, 18][min(v, 3)]):
        r = w * rng.uniform(0.04, 0.14)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.74)
        od.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*L, rng.randint(40, 115)))
    orbs = orbs.filter(ImageFilter.GaussianBlur(radius=12))
    od2 = ImageDraw.Draw(orbs)
    for _ in range([4, 5, 3, 6][min(v, 3)]):                                       # a few crisp orbs
        r = w * rng.uniform(0.03, 0.07)
        cx, cy = rng.uniform(0, w), rng.uniform(0, h * 0.7)
        od2.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*L, 150), width=max(2, int(r * 0.18)))
    return Image.alpha_composite(img, orbs)


def _make_clouds_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Soft layered cloud puffs — dreamy, daydreaming, overcast, grey skies."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0)); ld = ImageDraw.Draw(layer)
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 70) for x in color_top[:3])
    rng = rng or random.Random(v)
    for i in range([3, 4, 2, 5, 3, 4][min(v, 5)]):
        cy = h * (0.18 + 0.15 * i) + rng.uniform(-0.03, 0.03) * h
        cx = w * rng.uniform(0.3, 0.7); cw_ = w * rng.uniform(0.28, 0.42); a = rng.randint(50, 90)
        for k in range(5):
            ex = cx - cw_ / 2 + cw_ * k / 4; er = cw_ * (0.18 + 0.10 * abs(math.sin(k)))
            ld.ellipse([ex - er, cy - er * 0.8, ex + er, cy + er * 0.8], fill=(*L, a))
        ld.ellipse([cx - cw_ * 0.5, cy - cw_ * 0.04, cx + cw_ * 0.5, cy + cw_ * 0.20], fill=(*L, a))
    return Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(radius=7)))


def _make_zen_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Raked-sand ripples + stacked stones — spa, wind down, meditation, yoga."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=70): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.46): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    cxf, cyf = [(0.50, 0.50), (0.38, 0.46), (0.62, 0.52), (0.50, 0.44)][min(v, 3)]
    cx, cy = w * cxf, h * cyf
    for i in range(11, 0, -1):                                  # raked sand rings around the stones
        rr = min(w, h) * 0.52 * i / 11
        draw.ellipse([cx - rr, cy - rr * 0.5, cx + rr, cy + rr * 0.5],
                     outline=(*L, max(18, 64 - i * 3)), width=max(2, w // 360))
    sy = cy
    for sw_ in (0.11, 0.085, 0.06):                            # stacked stones
        sr = w * sw_
        draw.ellipse([cx - sr, sy - sr * 0.55, cx + sr, sy + sr * 0.55], fill=(*D, 185))
        sy -= sr * 0.95
    return img


def _make_grid_paper_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Graph / dotted notebook grid — study, focus, deep work, deep reading."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 80) for x in color_top[:3])
    cols = [16, 22, 12, 26][min(v, 3)]; sp = w / cols
    style = 'dots' if v % 2 else 'lines'; lw = max(1, w // 500); reg = h * 0.74
    if style == 'lines':
        x = 0
        while x <= w:
            draw.line([(x, 0), (x, reg)], fill=(*L, 38), width=lw); x += sp
        y = 0
        while y <= reg:
            draw.line([(0, y), (w, y)], fill=(*L, 38), width=lw); y += sp
    else:
        y = sp / 2
        while y < reg:
            x = sp / 2
            while x < w:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(*L, 90)); x += sp
            y += sp
    return img


def _make_moonlight_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Moon + craters + soft stars — sleep, midnight, clear night, 3am."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 130) for x in color_top[:3])
    D = tuple(int(x * 0.7) for x in L)
    rng = rng or random.Random(v)
    mxf, myf, mrf = [(0.68, 0.30, 0.16), (0.50, 0.28, 0.18), (0.32, 0.32, 0.14), (0.72, 0.26, 0.20)][min(v, 3)]
    mx, my, mr = w * mxf, h * myf, w * mrf
    bar = h * 0.74
    for _ in range(70):                                        # stars
        x, y = rng.uniform(0, w), rng.uniform(0, bar)
        r = 1 if rng.random() < 0.8 else 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*L, rng.randint(110, 220)))
    draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=(*L, 205))             # moon
    for _ in range(4):                                         # craters
        a = rng.uniform(0, 6.28); rr = mr * rng.uniform(0.1, 0.7)
        cxx, cyy = mx + math.cos(a) * mr * 0.5, my + math.sin(a) * mr * 0.5
        draw.ellipse([cxx - rr * 0.3, cyy - rr * 0.3, cxx + rr * 0.3, cyy + rr * 0.3], fill=(*D, 120))
    return img


def _make_cosmos_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Ringed planet + orbit + stars — ambient drift, awe & wonder, cosmic."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=110): return tuple(_clamp(x + amt) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); M = _mid(color_top, color_bottom)
    rng = rng or random.Random(v)
    for _ in range(70):                                        # stars
        x, y = rng.uniform(0, w), rng.uniform(0, h * 0.74)
        r = 1 if rng.random() < 0.85 else 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*L, rng.randint(90, 200)))
    pxf, pyf, prf = [(0.62, 0.36, 0.16), (0.40, 0.34, 0.18), (0.50, 0.30, 0.14), (0.66, 0.40, 0.20)][min(v, 3)]
    px, py, pr = w * pxf, h * pyf, w * prf
    rw = pr * 1.9
    ring_box = [px - rw, py - rw * 0.30, px + rw, py + rw * 0.30]
    lwd = max(4, w // 200)
    # WHY split arcs: the ring must pass BEHIND the planet at the back and IN FRONT at the front.
    draw.arc(ring_box, 180, 360, fill=(*L, 140), width=lwd)                        # back half (behind)
    draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(*M, 215))             # planet
    draw.ellipse([px - pr * 0.7, py - pr * 0.7, px + pr * 0.25, py + pr * 0.25], fill=(*L, 80))  # highlight
    draw.arc(ring_box, 0, 180, fill=(*L, 185), width=lwd)                          # front half (in front)
    return img


def _make_lightning_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Jagged electric bolts (glow) — empowering, confidence, motivation, beast mode."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 120) for x in color_top[:3])
    rng = rng or random.Random(v)
    n = [3, 4, 2, 5][min(v, 3)]; lw = max(3, int(w * 0.01))
    for i in range(n):
        x = w * (i + 0.5) / n + rng.uniform(-0.05, 0.05) * w
        pts = [(x, 0)]; y = 0
        while y < h * 0.7:
            y += h * rng.uniform(0.08, 0.16); x += rng.uniform(-0.08, 0.08) * w
            pts.append((x, y))
        gd.line(pts, fill=(*L, 220), width=lw, joint="curve")
        if len(pts) > 2:                                       # a branch
            bx, by = pts[len(pts) // 2]
            gd.line([(bx, by), (bx + rng.uniform(-0.12, 0.12) * w, by + h * 0.16)], fill=(*L, 180), width=max(2, lw - 2))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=8)))
    return Image.alpha_composite(img, glow)


def _make_blossom_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Falling cherry-blossom petals + a few blooms — spring mixes."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    rng = rng or random.Random(v)
    pinks = [(255, 200, 215), (250, 180, 205), (240, 212, 226), (255, 224, 235)]
    bar = h * 0.74
    for _ in range([30, 45, 22, 60][min(v, 3)]):                  # drifting petals
        x, y = rng.uniform(0, w), rng.uniform(0, bar)
        s = w * rng.uniform(0.012, 0.026)
        draw.ellipse([x - s, y - s * 0.6, x + s, y + s * 0.6], fill=(*rng.choice(pinks), rng.randint(120, 200)))
    for _ in range([4, 6, 3, 8][min(v, 3)]):                      # 5-petal blossoms
        fx, fy = rng.uniform(w * 0.1, w * 0.9), rng.uniform(0, bar * 0.85)
        fr = w * rng.uniform(0.02, 0.038); col = rng.choice(pinks)
        for k in range(5):
            a = math.radians(k * 72)
            px, py = fx + math.cos(a) * fr, fy + math.sin(a) * fr
            draw.ellipse([px - fr * 0.6, py - fr * 0.6, px + fr * 0.6, py + fr * 0.6], fill=(*col, 200))
        draw.ellipse([fx - fr * 0.3, fy - fr * 0.3, fx + fr * 0.3, fy + fr * 0.3], fill=(255, 235, 170, 220))
    return img


def _make_snowfall_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Drifting snow + low drifts — snow day, frosty, winter frost/nights."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 120) for x in color_top[:3])
    rng = rng or random.Random(v)
    bar = h * 0.80
    draw.ellipse([-w * 0.2, h * 0.66, w * 0.62, h * 0.98], fill=(*L, 55))         # drifts
    draw.ellipse([w * 0.42, h * 0.70, w * 1.2, h * 1.02], fill=(*L, 50))
    for _ in range([60, 90, 40, 120][min(v, 3)]):                # flakes
        x, y = rng.uniform(0, w), rng.uniform(0, bar)
        r = rng.choice([1, 2, 2, 3])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*L, rng.randint(120, 230)))
    return img


def _make_rainfall_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Diagonal rain streaks — rain weather + sad moods (heartbreak, grief, melancholy)."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 70) for x in color_top[:3])
    rng = rng or random.Random(v)
    ang = math.radians(74); dx, dy = math.cos(ang), math.sin(ang)
    ln = h * 0.06; lw = max(1, w // 420)
    for _ in range([120, 170, 80, 220][min(v, 3)]):
        x, y = rng.uniform(0, w), rng.uniform(-h * 0.05, h * 0.76)
        draw.line([(x, y), (x + dx * ln, y + dy * ln)], fill=(*L, rng.randint(60, 145)), width=lw)
    return img


def _make_pine_forest_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Rows of evergreen pines receding — folk, celtic, campfire, cosy, winter."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _c01(x): return max(0.0, min(1.0, x))
    def _lighten(c, amt=66): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rows = [3, 4, 2, 5][min(v, 3)]
    def tone(fr): fr = _c01(fr); return tuple(int(L[k] + (D[k] - L[k]) * fr) for k in range(3))
    for layer in range(rows):
        frac = layer / (rows - 1) if rows > 1 else 0
        base_y = h * (0.46 + 0.26 * frac); size = w * (0.05 + 0.03 * frac)
        n = 5 + layer * 2; th = size * 3.2
        col = tone(frac)
        for i in range(n + 1):
            tx = i * (w / n)
            top = base_y - th
            draw.polygon([(tx - size, base_y), (tx + size, base_y), (tx, top)], fill=(*col, 205))
            draw.polygon([(tx - size * 0.8, base_y - th * 0.42), (tx + size * 0.8, base_y - th * 0.42), (tx, top - th * 0.18)], fill=(*col, 205))
    return img


def _make_tropical_leaves_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Big tropical leaf silhouettes from the corners — tropical, reggae, afrobeat, beach."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=55): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.40): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    anchors = [[(0.04, 0.06, 35)], [(0.96, 0.06, 145), (0.05, 0.95, -55)],
               [(0.05, 0.95, -50)], [(0.96, 0.95, 215), (0.04, 0.08, 40)]][min(v, 3)]
    for axf, ayf, base_deg in anchors:
        ax, ay = w * axf, h * ayf
        for k in range(3):
            ang = math.radians(base_deg + (k - 1) * 26)
            ln, wd = w * 0.52, w * 0.10
            tip = (ax + math.cos(ang) * ln, ay + math.sin(ang) * ln)
            perp = ang + math.pi / 2
            m1 = (ax + math.cos(ang) * ln * 0.5 + math.cos(perp) * wd, ay + math.sin(ang) * ln * 0.5 + math.sin(perp) * wd)
            m2 = (ax + math.cos(ang) * ln * 0.5 - math.cos(perp) * wd, ay + math.sin(ang) * ln * 0.5 - math.sin(perp) * wd)
            draw.polygon([(ax, ay), m1, tip, m2], fill=(*D, 150))
            draw.line([(ax, ay), tip], fill=(*L, 110), width=max(2, int(w * 0.004)))   # midrib
            for t in (0.35, 0.55, 0.75):
                vx, vy = ax + math.cos(ang) * ln * t, ay + math.sin(ang) * ln * t
                draw.line([(vx, vy), (vx + math.cos(perp) * wd * 0.7, vy + math.sin(perp) * wd * 0.7)], fill=(*L, 90), width=max(1, int(w * 0.002)))
                draw.line([(vx, vy), (vx - math.cos(perp) * wd * 0.7, vy - math.sin(perp) * wd * 0.7)], fill=(*L, 90), width=max(1, int(w * 0.002)))
    return img


def _make_gingham_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Even picnic gingham check (overlaps darken) — cookout, brunch, cooking, dinner party."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 60) for x in color_top[:3])
    cell = w * [0.10, 0.08, 0.13, 0.07][min(v, 3)]; a = 66
    x = 0
    while x < w:
        draw.rectangle([x, 0, x + cell, h], fill=(*L, a)); x += cell * 2
    y = 0
    while y < h:
        draw.rectangle([0, y, w, y + cell], fill=(*L, a)); y += cell * 2
    return Image.alpha_composite(img, overlay)


def _make_open_road_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A road receding to a vanishing point on a DISTINCT horizon — sky + sun glow above, ground below,
    asphalt road with a dashed centre line. Road trip, summer roadtrip, driving (Windows Down).
    WHY redrawn: the old road just faded into the gradient with no sky/horizon."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=95): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.42): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    hz  = h * [0.42, 0.40, 0.44, 0.41][min(v, 3)]                 # horizon line
    vpx = w * [0.50, 0.42, 0.58, 0.50][min(v, 3)]                 # vanishing point ON the horizon
    draw.rectangle([0, 0, w, hz], fill=(*L, 70))                  # distinct sky band above the horizon
    draw.ellipse([vpx - w * 0.17, hz - w * 0.17, vpx + w * 0.17, hz + w * 0.17], fill=(*L, 70))  # low sun glow
    draw.rectangle([0, hz, w, h], fill=(*D, 150))                 # ground below the horizon
    by = h * 0.99; bw = w * 0.62                                  # asphalt road → vanishing point
    draw.polygon([(w * 0.5 - bw / 2, by), (w * 0.5 + bw / 2, by), (vpx + w * 0.018, hz), (vpx - w * 0.018, hz)], fill=(44, 44, 50, 240))
    draw.line([(w * 0.5 - bw / 2, by), (vpx - w * 0.018, hz)], fill=(*L, 150), width=max(2, int(w * 0.004)))  # edge lines
    draw.line([(w * 0.5 + bw / 2, by), (vpx + w * 0.018, hz)], fill=(*L, 150), width=max(2, int(w * 0.004)))
    n = 7
    lerp = lambda t: (w * 0.5 + (vpx - w * 0.5) * t, by + (hz - by) * t)
    for i in range(n):                                            # dashed yellow centre line
        x0, y0 = lerp(i / n); x1, y1 = lerp((i + 0.5) / n)
        draw.line([(x0, y0), (x1, y1)], fill=(245, 220, 90, 235), width=max(2, int((1 - i / n) * w * 0.02)))
    return img


def _make_holiday_lights_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Strings of festive fairy-lights / baubles — festive."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 50) for x in color_top[:3])
    bulbs = [(235, 70, 70), (80, 200, 95), (245, 205, 90), (130, 170, 255)]
    nstr = [3, 4, 2, 5][min(v, 3)]
    wire_y = lambda x, y0: y0 + math.sin(x * 0.012) * h * 0.045
    strings = [h * (0.12 + s * 0.15) for s in range(nstr)]
    for y0 in strings:
        for k in range(11):
            bx = w * (k + 0.5) / 11; by = wire_y(bx, y0) + h * 0.018; br = w * 0.013
            gd.ellipse([bx - br * 2.4, by - br * 2.4, bx + br * 2.4, by + br * 2.4], fill=(*bulbs[k % 4], 110))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=9)))
    draw = ImageDraw.Draw(img, 'RGBA')
    for y0 in strings:
        draw.line([(x, wire_y(x, y0)) for x in range(0, w + 1, 10)], fill=(*L, 130), width=max(2, w // 400))
        for k in range(11):
            bx = w * (k + 0.5) / 11; by = wire_y(bx, y0) + h * 0.018; br = w * 0.013
            draw.line([(bx, wire_y(bx, y0)), (bx, by)], fill=(*L, 120), width=max(1, w // 650))
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(*bulbs[k % 4], 235))
    return img


# === NEW families, round 3 (genre-true, album-cover quality) ===================

def _make_jazz_club_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A lively jazz combo under crossed spotlights — a warm-wood double bass + a brass trumpet + a
    scatter of real music-note glyphs. WHY redrawn: flat silhouettes + hand-drawn notes read as crude."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    rng = rng or random.Random(v)
    warm = (255, 206, 128)
    sp = Image.new("RGBA", (w, h), (0, 0, 0, 0)); sd = ImageDraw.Draw(sp)
    sd.polygon([(w * 0.16, -h * 0.05), (w * 0.60, h * 0.95), (w * 0.28, h * 0.95)], fill=(*warm, 30))  # crossed
    sd.polygon([(w * 0.84, -h * 0.05), (w * 0.72, h * 0.95), (w * 0.42, h * 0.95)], fill=(*warm, 26))  # spotlights
    img = Image.alpha_composite(img, sp.filter(ImageFilter.GaussianBlur(radius=36)))
    draw = ImageDraw.Draw(img, 'RGBA')
    # WHY 10 entries + _BG_SYMMETRIC (no h-flip): the h-mirror was flipping the music-note glyphs backwards.
    flip = [0, -1, 1, 0, -1, 1, 0.5, -0.5, 1.5, -1.5][min(v, 9)]                              # per-variant nudge
    floor = h * 0.74; lwd = max(2, int(w * 0.005))
    draw.line([(0, floor), (w, floor)], fill=(40, 30, 24, 90), width=max(2, int(h * 0.006)))  # stage edge
    # --- double bass (left), warm wood standing on an endpin ---
    bx = w * (0.28 + 0.02 * flip); wood, wood_d, strc = (150, 88, 42), (94, 52, 24), (232, 224, 200)
    uy, ly = h * 0.46, h * 0.60; upr, lwr = w * 0.10, w * 0.135
    draw.line([(bx, ly + lwr * 0.92), (bx, floor + h * 0.02)], fill=(*wood_d, 235), width=max(3, int(w * 0.006)))  # endpin
    draw.ellipse([bx - lwr, ly - lwr * 0.86, bx + lwr, ly + lwr * 0.96], fill=(*wood, 235))   # lower bout
    draw.ellipse([bx - upr, uy - upr * 0.80, bx + upr, uy + upr * 0.90], fill=(*wood, 235))   # upper bout
    draw.polygon([(bx - upr * 0.82, uy), (bx + upr * 0.82, uy),
                  (bx + lwr * 0.82, ly), (bx - lwr * 0.82, ly)], fill=(*wood, 235))           # waist
    shoulder = uy - upr * 0.78; nw = w * 0.026
    draw.rectangle([bx - nw * 0.5, h * 0.155, bx + nw * 0.5, shoulder], fill=(*wood_d, 235))  # neck
    sr = w * 0.018
    draw.arc([bx - sr, h * 0.13, bx + sr, h * 0.172], 0, 330, fill=(*wood, 235), width=max(3, int(w * 0.01)))  # scroll
    bjy = ly + lwr * 0.20
    for s in (-1, 1):                                                                         # f-holes
        draw.line([(bx + s * upr * 0.46, bjy - h * 0.05), (bx + s * upr * 0.46, bjy + h * 0.04)], fill=(*wood_d, 200), width=lwd)
    for s in (-1.5, -0.5, 0.5, 1.5):                                                          # 4 strings
        draw.line([(bx + s * nw * 0.2, ly + lwr * 0.7), (bx + s * nw * 0.2, h * 0.16)], fill=(*strc, 175), width=2)
    # --- trumpet (right), brass ---
    tx, ty = w * (0.65 + 0.02 * flip), h * 0.42; tw = max(8, int(w * 0.022))
    brass, brass_h, brass_d = (212, 172, 60), (244, 216, 124), (150, 116, 30)
    draw.line([(tx - w * 0.14, ty), (tx + w * 0.06, ty)], fill=(*brass, 255), width=tw)       # leadpipe
    draw.line([(tx - w * 0.14, ty - tw * 0.28), (tx + w * 0.06, ty - tw * 0.28)], fill=(*brass_h, 255), width=max(2, tw // 3))  # highlight
    draw.polygon([(tx + w * 0.06, ty - h * 0.02), (tx + w * 0.06, ty + h * 0.02),
                  (tx + w * 0.20, ty + h * 0.085), (tx + w * 0.20, ty - h * 0.085)], fill=(*brass, 255))  # bell flare
    draw.ellipse([tx + w * 0.185, ty - h * 0.085, tx + w * 0.215, ty + h * 0.085], fill=(*brass_h, 255))  # bell rim
    draw.ellipse([tx - w * 0.165, ty - tw * 0.7, tx - w * 0.13, ty + tw * 0.7], fill=(*brass_d, 255))     # mouthpiece
    for k in range(3):                                                                        # 3 valves + caps
        vx = tx - w * 0.04 + k * w * 0.035
        draw.rectangle([vx - tw * 0.32, ty - h * 0.06, vx + tw * 0.32, ty], fill=(*brass_d, 255))
        draw.ellipse([vx - tw * 0.42, ty - h * 0.078, vx + tw * 0.42, ty - h * 0.05], fill=(*brass_h, 255))
    # --- scattered music-note GLYPHS (non-overlapping), warm gold ---
    notes = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gold = (250, 210, 120)
    names = ["music_note", "music_note_2"]; placed = []; target = rng.randint(3, 5); tries = 0
    while len(placed) < target and tries < 60:
        tries += 1
        nx, ny = rng.uniform(w * 0.42, w * 0.84), rng.uniform(h * 0.10, h * 0.32)
        nsz = w * rng.uniform(0.05, 0.075); r = nsz * 0.55
        if any(math.hypot(nx - px, ny - py) < r + pr for px, py, pr in placed):
            continue
        placed.append((nx, ny, r))
        _draw_glyph(notes, names[len(placed) % 2], nx, ny, nsz, (*gold, 235), tilt=rng.uniform(-16, 16))
    return Image.alpha_composite(img, notes)


def _make_strings_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A cello in warm wood — classical, string quartet, neoclassical, strings & romance.
    WHY redrawn: the old neck floated off the bouts and strings fanned out. Here a heel joins the neck to
    the body and the neck, fingerboard, bridge and 4 parallel strings all share the centreline = connected."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    wood, wood_d, fb, strc = (150, 86, 40), (96, 52, 22), (44, 30, 22), (232, 224, 200)
    configs = [(0.50, 1.00), (0.42, 0.94), (0.58, 1.06), (0.50, 0.86)]    # (centre_x, scale) — all distinct
    cxf, scl = configs[min(v, len(configs) - 1)]
    cx = w * cxf
    uy, ly = h * 0.46, h * 0.62
    upr, lwr = w * 0.105 * scl, w * 0.140 * scl
    lwd = max(2, int(w * 0.006))
    draw.ellipse([cx - lwr, ly - lwr * 0.86, cx + lwr, ly + lwr * 0.96], fill=(*wood, 235))   # lower bout
    draw.ellipse([cx - upr, uy - upr * 0.80, cx + upr, uy + upr * 0.90], fill=(*wood, 235))   # upper bout
    draw.polygon([(cx - upr * 0.82, uy), (cx + upr * 0.82, uy),
                  (cx + lwr * 0.82, ly), (cx - lwr * 0.82, ly)], fill=(*wood, 235))           # waist
    draw.arc([cx - lwr, ly - lwr * 0.86, cx + lwr, ly + lwr * 0.96], 300, 80, fill=(*wood_d, 130), width=lwd)  # edge sheen
    shoulder = uy - upr * 0.78; nw = w * 0.034 * scl; neck_top = h * 0.16
    draw.polygon([(cx - upr * 0.34, shoulder), (cx + upr * 0.34, shoulder),
                  (cx + nw * 0.6, shoulder - h * 0.04), (cx - nw * 0.6, shoulder - h * 0.04)], fill=(*wood, 235))  # heel
    draw.rectangle([cx - nw * 0.6, neck_top, cx + nw * 0.6, shoulder - h * 0.03], fill=(*wood_d, 235))   # neck
    draw.rectangle([cx - nw * 0.42, neck_top, cx + nw * 0.42, shoulder - h * 0.03], fill=(*fb, 235))     # fingerboard
    draw.rectangle([cx - nw * 0.5, neck_top - h * 0.03, cx + nw * 0.5, neck_top], fill=(*wood, 235))     # pegbox
    sr = w * 0.02
    draw.arc([cx - sr, neck_top - h * 0.058, cx + sr, neck_top - h * 0.012], 0, 330, fill=(*wood, 235), width=max(3, int(w * 0.01)))  # scroll
    by = ly + lwr * 0.18
    draw.line([(cx - upr * 0.4, by), (cx + upr * 0.4, by)], fill=(*wood_d, 220), width=lwd)   # bridge
    for s in (-1, 1):                                                                         # f-holes
        fx = cx + s * upr * 0.5
        draw.line([(fx, by - h * 0.06), (fx, by + h * 0.04)], fill=(*wood_d, 200), width=lwd)
    tail_y = ly + lwr * 0.72
    for s in (-1.5, -0.5, 0.5, 1.5):                                                          # 4 aligned strings
        ox = s * nw * 0.22
        draw.line([(cx + ox, tail_y), (cx + ox, neck_top)], fill=(*strc, 175), width=max(1, int(w * 0.0035)))
    return img


# Gibson Les Paul — a 480-point outline of the WHOLE guitar (body + neck + headstock) traced off a real
# LP photo: the transparent-PNG alpha was boundary-traced (5000 px), PCA-rotated so the neck lies
# horizontal, flipped so the single cutaway sits at the BOTTOM (treble side), and normalised to body
# half-width 0.70 with the body centred on the origin. nx<=1.149 = body, 1.149..2.83 = neck, >=2.83 = headstock.
# WHY a traced outline: hand-built body polygons read as a lumpy blob; the real contour is unmistakably an LP.
_GUITAR_CP = [
    (-0.2041, -0.7091), (-0.179, -0.7069), (-0.1539, -0.7047), (-0.1288, -0.7025), (-0.1037, -0.7003), (-0.0786, -0.6981),
    (-0.0546, -0.6931), (-0.0313, -0.6865), (-0.008, -0.6802), (0.0146, -0.6724), (0.0374, -0.6645), (0.0595, -0.6555),
    (0.0814, -0.646), (0.1029, -0.6352), (0.1238, -0.6238), (0.1445, -0.6121), (0.1645, -0.5981), (0.1852, -0.5861),
    (0.2051, -0.5721), (0.2236, -0.5564), (0.2434, -0.5425), (0.2608, -0.525), (0.2794, -0.5093), (0.2971, -0.4921),
    (0.3153, -0.4762), (0.3326, -0.4588), (0.3511, -0.4431), (0.3709, -0.4291), (0.3912, -0.4163), (0.4125, -0.4059),
    (0.4364, -0.4007), (0.4604, -0.3957), (0.4847, -0.3965), (0.5084, -0.4003), (0.5317, -0.4062), (0.5546, -0.4148),
    (0.5765, -0.4249), (0.5979, -0.4364), (0.6199, -0.4463), (0.6414, -0.4578), (0.6641, -0.4664), (0.686, -0.4757),
    (0.7084, -0.4843), (0.7318, -0.4905), (0.7551, -0.496), (0.7786, -0.4999), (0.8031, -0.5007), (0.8274, -0.5015),
    (0.8512, -0.4965), (0.8752, -0.4915), (0.8978, -0.4837), (0.9194, -0.4736), (0.9403, -0.4623), (0.9602, -0.4489),
    (0.9795, -0.4341), (0.9981, -0.4183), (1.0143, -0.399), (1.0305, -0.3797), (1.0452, -0.3597), (1.059, -0.3393),
    (1.0763, -0.3498), (1.0868, -0.3585), (1.1015, -0.3396), (1.1057, -0.3158), (1.0867, -0.3175), (1.0753, -0.3035),
    (1.0822, -0.2806), (1.0889, -0.2574), (1.0953, -0.2345), (1.0995, -0.211), (1.101, -0.1867), (1.1041, -0.1628),
    (1.1037, -0.1381), (1.1206, -0.1266), (1.145, -0.1275), (1.1701, -0.1253), (1.1944, -0.1261), (1.2188, -0.1269),
    (1.2439, -0.1247), (1.2683, -0.1255), (1.2934, -0.1233), (1.3177, -0.1241), (1.3421, -0.1249), (1.3672, -0.1227),
    (1.3915, -0.1235), (1.4166, -0.1213), (1.4409, -0.1221), (1.4657, -0.1213), (1.4903, -0.1208), (1.5149, -0.1207),
    (1.5385, -0.1195), (1.5629, -0.1203), (1.588, -0.1181), (1.6124, -0.1189), (1.6375, -0.1167), (1.6619, -0.1175),
    (1.6864, -0.1174), (1.7113, -0.1161), (1.7364, -0.1139), (1.7607, -0.1147), (1.7852, -0.1155), (1.8103, -0.1133),
    (1.8346, -0.1141), (1.8597, -0.1119), (1.884, -0.1128), (1.9086, -0.1124), (1.9334, -0.1114), (1.9578, -0.1122),
    (1.9829, -0.11), (2.0072, -0.1108), (2.0323, -0.1086), (2.0567, -0.1094), (2.0811, -0.1102), (2.1062, -0.108),
    (2.1306, -0.1088), (2.1557, -0.1066), (2.1801, -0.1074), (2.2052, -0.1052), (2.2295, -0.106), (2.2546, -0.1041),
    (2.2791, -0.1046), (2.3035, -0.1054), (2.3286, -0.1033), (2.3529, -0.1041), (2.3775, -0.1037), (2.4023, -0.1027),
    (2.427, -0.1022), (2.4519, -0.1013), (2.4763, -0.1021), (2.5014, -0.0999), (2.5258, -0.1007), (2.5509, -0.0985),
    (2.5752, -0.0993), (2.5998, -0.0987), (2.6246, -0.0979), (2.6497, -0.0957), (2.6741, -0.0965), (2.6992, -0.0943),
    (2.7228, -0.0981), (2.7457, -0.1049), (2.7674, -0.1152), (2.7875, -0.1294), (2.8042, -0.146), (2.8284, -0.1476),
    (2.8436, -0.1586), (2.8317, -0.1796), (2.8273, -0.203), (2.8254, -0.226), (2.8492, -0.2297), (2.8732, -0.2316),
    (2.8971, -0.2264), (2.8932, -0.2019), (2.887, -0.1787), (2.8736, -0.1591), (2.8871, -0.1454), (2.9115, -0.1461),
    (2.9366, -0.144), (2.9611, -0.1448), (2.9662, -0.1636), (2.955, -0.1839), (2.951, -0.2075), (2.9543, -0.2261),
    (2.978, -0.2299), (3.0026, -0.2292), (3.0228, -0.221), (3.0178, -0.1972), (3.0126, -0.1734), (2.9993, -0.1535),
    (3.0196, -0.1456), (3.044, -0.1464), (3.068, -0.1488), (3.092, -0.1511), (3.0894, -0.1712), (3.0796, -0.1931),
    (3.0752, -0.2167), (3.0843, -0.2327), (3.1078, -0.2376), (3.1329, -0.2355), (3.1492, -0.2236), (3.1442, -0.1996),
    (3.1386, -0.176), (3.1258, -0.1577), (3.1481, -0.1549), (3.1725, -0.1559), (3.1963, -0.1595), (3.2205, -0.161),
    (3.2443, -0.1641), (3.2686, -0.1649), (3.2707, -0.1435), (3.2693, -0.1202), (3.273, -0.0965), (3.2791, -0.0735),
    (3.2867, -0.0506), (3.2905, -0.0268), (3.2883, -0.0017), (3.2921, 0.0219), (3.2863, 0.0455), (3.2769, 0.0674),
    (3.2714, 0.0912), (3.2675, 0.115), (3.2702, 0.1387), (3.2694, 0.1599), (3.2444, 0.1588), (3.2193, 0.1566),
    (3.1955, 0.1515), (3.1704, 0.1493), (3.1453, 0.1472), (3.1269, 0.1565), (3.1411, 0.176), (3.1475, 0.1991),
    (3.1513, 0.2228), (3.1296, 0.2308), (3.1051, 0.2317), (3.0813, 0.2267), (3.0795, 0.2042), (3.0846, 0.1803),
    (3.0962, 0.1594), (3.0847, 0.1448), (3.0596, 0.1426), (3.0345, 0.1404), (3.0101, 0.1412), (3.0039, 0.1575),
    (3.0174, 0.1767), (3.0229, 0.1999), (3.0202, 0.2206), (2.9966, 0.2251), (2.9735, 0.2231), (2.9506, 0.2165),
    (2.9543, 0.193), (2.9593, 0.169), (2.9722, 0.1494), (2.953, 0.1392), (2.9287, 0.14), (2.9042, 0.1408),
    (2.8798, 0.1416), (2.8821, 0.1606), (2.8907, 0.1827), (2.8974, 0.2055), (2.8883, 0.2233), (2.8645, 0.2271),
    (2.8407, 0.2239), (2.8264, 0.2122), (2.8286, 0.1871), (2.8362, 0.1644), (2.8402, 0.144), (2.8151, 0.1418),
    (2.7936, 0.1323), (2.7762, 0.1149), (2.7566, 0.1008), (2.735, 0.0907), (2.711, 0.0857), (2.6859, 0.0835),
    (2.6615, 0.0843), (2.6372, 0.0851), (2.6128, 0.0859), (2.5884, 0.0867), (2.564, 0.0875), (2.5396, 0.0884),
    (2.5145, 0.0862), (2.4902, 0.087), (2.4662, 0.089), (2.4416, 0.0886), (2.4172, 0.0894), (2.3928, 0.0902),
    (2.3685, 0.091), (2.344, 0.0918), (2.3193, 0.0909), (2.2946, 0.0906), (2.2703, 0.0912), (2.2458, 0.092),
    (2.2215, 0.0928), (2.197, 0.0936), (2.1726, 0.0944), (2.148, 0.0941), (2.1238, 0.0955), (2.099, 0.0949),
    (2.0745, 0.0947), (2.05, 0.0955), (2.0256, 0.0963), (2.0013, 0.0971), (1.977, 0.0979), (1.9519, 0.0957),
    (1.9274, 0.0965), (1.903, 0.0973), (1.8786, 0.0981), (1.8543, 0.0989), (1.8312, 0.0969), (1.8067, 0.0977),
    (1.7824, 0.0985), (1.7598, 0.0994), (1.7354, 0.1002), (1.7103, 0.0981), (1.6859, 0.0988), (1.6614, 0.0996),
    (1.6369, 0.1004), (1.6126, 0.1012), (1.5881, 0.102), (1.5637, 0.1028), (1.5392, 0.1036), (1.5144, 0.1025),
    (1.4898, 0.1024), (1.4654, 0.1031), (1.441, 0.1039), (1.4165, 0.1047), (1.3921, 0.1049), (1.3671, 0.1033),
    (1.3428, 0.1041), (1.3185, 0.1049), (1.294, 0.1057), (1.2696, 0.1065), (1.2445, 0.1043), (1.2201, 0.1051),
    (1.1956, 0.1059), (1.1712, 0.1067), (1.1467, 0.1075), (1.1217, 0.1058), (1.0973, 0.1061), (1.0728, 0.1069),
    (1.0483, 0.1077), (1.0238, 0.1085), (0.9995, 0.1094), (0.9754, 0.1121), (0.9512, 0.1141), (0.9279, 0.1195),
    (0.905, 0.1273), (0.8835, 0.138), (0.8626, 0.1501), (0.845, 0.1675), (0.8294, 0.1861), (0.8189, 0.2078),
    (0.8114, 0.2307), (0.8064, 0.2546), (0.8072, 0.2789), (0.8122, 0.3025), (0.8205, 0.3252), (0.8326, 0.3461),
    (0.848, 0.3644), (0.8666, 0.3801), (0.8855, 0.3957), (0.9018, 0.4124), (0.908, 0.4353), (0.8985, 0.4574),
    (0.8774, 0.4696), (0.8544, 0.4752), (0.8306, 0.479), (0.8063, 0.4798), (0.7819, 0.4806), (0.7568, 0.4784),
    (0.7329, 0.4734), (0.7089, 0.4684), (0.6861, 0.461), (0.6635, 0.4524), (0.6418, 0.4419), (0.6201, 0.4315),
    (0.5986, 0.4202), (0.5774, 0.4086), (0.5566, 0.3969), (0.5341, 0.3885), (0.5102, 0.3835), (0.4863, 0.3785),
    (0.4612, 0.3763), (0.4374, 0.3796), (0.4139, 0.384), (0.392, 0.3939), (0.3709, 0.406), (0.3507, 0.4204),
    (0.3314, 0.4366), (0.3159, 0.4551), (0.2973, 0.4717), (0.2792, 0.4886), (0.2599, 0.5048), (0.2415, 0.5202),
    (0.2232, 0.5356), (0.2031, 0.5497), (0.183, 0.5639), (0.1628, 0.5781), (0.1416, 0.5905), (0.1214, 0.6046),
    (0.0995, 0.6151), (0.0776, 0.6254), (0.0557, 0.6357), (0.0329, 0.6445), (0.0108, 0.6538), (-0.0124, 0.661),
    (-0.0356, 0.6676), (-0.0588, 0.6742), (-0.0826, 0.678), (-0.1061, 0.6826), (-0.1299, 0.6856), (-0.1543, 0.6864),
    (-0.1787, 0.6872), (-0.2024, 0.6902), (-0.2273, 0.6889), (-0.2524, 0.6867), (-0.2775, 0.6845), (-0.3016, 0.6801),
    (-0.3264, 0.6773), (-0.3497, 0.6707), (-0.3729, 0.6644), (-0.3956, 0.6565), (-0.4183, 0.6487), (-0.4405, 0.6398),
    (-0.4623, 0.6302), (-0.485, 0.6221), (-0.5039, 0.6065), (-0.527, 0.604), (-0.549, 0.5941), (-0.5627, 0.5736),
    (-0.5753, 0.5557), (-0.5904, 0.5383), (-0.6075, 0.5206), (-0.6239, 0.5016), (-0.6401, 0.4823), (-0.6556, 0.4627),
    (-0.6686, 0.4419), (-0.6829, 0.4216), (-0.6952, 0.4006), (-0.7069, 0.3794), (-0.7176, 0.3578), (-0.7279, 0.3359),
    (-0.736, 0.3132), (-0.7433, 0.2901), (-0.7508, 0.2672), (-0.7564, 0.244), (-0.761, 0.2202), (-0.7662, 0.197),
    (-0.7699, 0.1733), (-0.7722, 0.1493), (-0.776, 0.1255), (-0.7763, 0.101), (-0.7758, 0.0762), (-0.7784, 0.0523),
    (-0.7792, 0.0279), (-0.7822, 0.0043), (-0.8011, 0.0099), (-0.8161, 0.0093), (-0.8225, -0.0136), (-0.8145, -0.0364),
    (-0.7987, -0.0315), (-0.78, -0.03), (-0.7778, -0.0551), (-0.7756, -0.0802), (-0.7734, -0.1053), (-0.7715, -0.1285),
    (-0.7692, -0.1536), (-0.767, -0.1787), (-0.762, -0.2026), (-0.7572, -0.2266), (-0.7519, -0.2503), (-0.7469, -0.2742),
    (-0.7391, -0.2969), (-0.7312, -0.3194), (-0.7244, -0.3426), (-0.7155, -0.3649), (-0.7067, -0.387), (-0.6967, -0.4086),
    (-0.6863, -0.43), (-0.6735, -0.4503), (-0.6613, -0.471), (-0.6477, -0.4907), (-0.6325, -0.5097), (-0.6162, -0.5278),
    (-0.5988, -0.5451), (-0.5813, -0.5626), (-0.562, -0.5788), (-0.5427, -0.595), (-0.5226, -0.6091), (-0.5016, -0.6213),
    (-0.4807, -0.6339), (-0.4596, -0.6458), (-0.4378, -0.6564), (-0.4153, -0.6658), (-0.3931, -0.6746), (-0.3701, -0.6823),
    (-0.3469, -0.6893), (-0.3236, -0.6951), (-0.2999, -0.6988), (-0.2762, -0.7025), (-0.2521, -0.7047), (-0.2284, -0.7083),
]


def _make_guitar_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A Gibson Les Paul — indie, rock, grunge, punk, blues, pub rock. The whole-guitar silhouette is a
    480-point trace of a real LP (see _GUITAR_CP); on top sit a bound rosewood board with trapezoid inlays,
    2 humbuckers (width ~= the string spread), tune-o-matic + stopbar, 4 amber knobs + toggle, and a black
    3+3 headstock (its own traced region). Drawn flat (neck to the right) then rotated to lean up-right."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    # 10 real finishes — the guitar is NEVER h-mirrored (a flipped Les Paul reads left-handed/wrong), so
    # extra variants are added here as genuine colours instead of flip-twins (guitar is in _BG_SYMMETRIC).
    variants = [(16, (38, 22, 14), (190, 118, 46)),    # tobacco / heritage cherry sunburst
                (14, (18, 18, 22), (178, 180, 188)),   # silverburst
                (18, (168, 126, 40), (230, 192, 96)),  # goldtop (near-solid gold)
                (15, (58, 12, 20), (192, 42, 50)),     # cherry / wine-red burst
                (12, (15, 15, 18), (46, 46, 52)),      # ebony (near-solid black)
                (17, (120, 70, 22), (232, 180, 76)),   # honey / lemon burst
                (16, (16, 30, 72), (76, 138, 216)),    # blue burst
                (15, (10, 40, 24), (52, 142, 80)),     # emerald-green burst
                (17, (32, 16, 50), (126, 76, 182)),    # purple burst
                (13, (34, 92, 118), (128, 186, 208))]  # (lean deg, edge, centre) — pelham/teal blue
    ang, edge, centre = variants[min(v, len(variants) - 1)]
    binding, fboard, fret, inlay = (230, 224, 198), (74, 48, 34), (198, 198, 202), (236, 234, 212)
    pg_cream, pu_sur, pu_body, gold = (234, 228, 200), (232, 226, 200), (40, 36, 34), (214, 180, 90)
    amber, chrome, hs_blk = (214, 150, 60), (208, 212, 218), (24, 22, 26)
    # supersample: draw the whole layer at 2x and LANCZOS-downsample at the very end. WHY: PIL polygons are
    # aliased, so a 2 px binding looks fuzzy at 1x; supersampling antialiases every edge cleanly WITHOUT
    # smoothing/altering the traced body shape (point-smoothing pulled in the tail + upper bout — never do that).
    ss = 2; W, H = w * ss, h * ss
    gl = Image.new("RGBA", (W, H), (0, 0, 0, 0)); g = ImageDraw.Draw(gl, 'RGBA')
    def rr(box, fill, outline=None):
        try:
            g.rounded_rectangle(box, radius=int(W * 0.01), fill=fill, outline=outline)
        except Exception:
            g.rectangle(box, fill=fill, outline=outline)
    S = W * 0.205; bx, by = W * 0.30, H * 0.545
    scy = by                                                              # neck centreline (trace neck centred on it)
    def P(nx, ny): return (bx + nx * S, by + ny * S)
    cp = _GUITAR_CP                                                       # the EXACT approved trace — never smoothed
    NECK_X, HS_X = 1.149, 2.83                                            # body->neck pocket, neck->headstock (from the trace)
    # --- whole-guitar silhouette: cream binding (full outline) ---
    g.polygon([P(nx, ny) for nx, ny in cp], fill=(*binding, 255))
    # --- body: a real sunburst — radial gradient (bright flamed centre -> dark burst edge) clipped to
    # the body, so the cream binding shows as a thin rim. Base-fill the body first so the irregular
    # corners (cutaway horn, bouts) stay edge-dark where the ellipses don't reach. ---
    body = [(nx, ny) for nx, ny in cp if nx <= NECK_X]
    burst = Image.new("RGBA", (W, H), (0, 0, 0, 0)); bd = ImageDraw.Draw(burst)
    bd.polygon([(bx + nx * S, by + ny * S) for nx, ny in body], fill=(*edge, 255))
    cxb, cyb = bx + 0.14 * S, by
    for i in range(26, -1, -1):
        t = i / 26.0                                                      # 1 = edge .. 0 = centre
        cc = tuple(int(centre[k] + (edge[k] - centre[k]) * t) for k in range(3))
        rx, ry = (0.10 + 0.92 * t) * S, (0.08 + 0.66 * t) * S
        bd.ellipse([cxb - rx, cyb - ry, cxb + rx, cyb + ry], fill=(*cc, 255))
    # uniform, barely-there binding: erode the body mask by a CONSTANT pixel width (a thick boundary line
    # eats the same amount all the way round) rather than scaling toward the centre (which is non-uniform)
    bind_px = max(2, int(round(W * 0.0015)))                             # ~1.5 px (final) — barely, barely there
    mask = Image.new("L", (W, H), 0); mdr = ImageDraw.Draw(mask)
    bodypx = [(bx + nx * S, by + ny * S) for nx, ny in body]
    mdr.polygon(bodypx, fill=255)
    mdr.line(bodypx + [bodypx[0]], fill=0, width=bind_px * 2, joint="curve")
    gl.paste(burst, (0, 0), mask)
    # --- HARDWARE — every part's position & size measured directly off lespaul_ref2.png (warped into
    # this exact frame): bridge pickup (0.22,-0.01) 0.165x0.296, neck pickup (0.66,-0.01), tune-o-matic
    # (0.054), stopbar (-0.092), the cream pickguard contour, the 4-knob parallelogram and the toggle. ---
    cy0 = scy - 0.012 * S                                                 # bridge/pickups/tailpiece centreline
    # cream pickguard (traced contour: flat top along the strings, point past the neck pickup, wide tail)
    pg = [(0.10, 0.122), (0.857, 0.118), (0.74, 0.24), (0.647, 0.267), (0.557, 0.30), (0.467, 0.337),
          (0.377, 0.375), (0.287, 0.412), (0.197, 0.45), (0.137, 0.455), (0.105, 0.40)]
    g.polygon([P(nx, ny) for nx, ny in pg], fill=(*pg_cream, 245))
    # stopbar tailpiece + tune-o-matic bridge — slim chrome bars (NOT solid blocks): a stopbar with
    # string anchors + end studs, and a saddle bar (kept shorter than the pickups) + 6 saddles + 2 posts
    chrome_d = (150, 154, 162)
    rr([bx - 0.114 * S, cy0 - 0.140 * S, bx - 0.066 * S, cy0 + 0.140 * S], (*chrome, 255))            # stopbar bar
    for s in range(6):                                                                                # string anchor holes
        yy = cy0 - 0.100 * S + s * (0.20 * S / 5)
        g.ellipse([bx - 0.099 * S, yy - 0.008 * S, bx - 0.081 * S, yy + 0.008 * S], fill=(*chrome_d, 255))
    for sy in (cy0 - 0.150 * S, cy0 + 0.150 * S):                                                     # 2 anchor studs
        g.ellipse([bx - 0.102 * S, sy - 0.016 * S, bx - 0.078 * S, sy + 0.016 * S], fill=(*chrome, 255), outline=(*chrome_d, 255))
    rr([bx + 0.028 * S, cy0 - 0.120 * S, bx + 0.080 * S, cy0 + 0.120 * S], (*chrome, 255))            # tune-o-matic saddle bar
    for s in range(6):                                                                                # 6 saddles
        yy = cy0 - 0.095 * S + s * (0.19 * S / 5)
        g.rectangle([bx + 0.036 * S, yy - 0.010 * S, bx + 0.072 * S, yy + 0.010 * S], fill=(*chrome_d, 255))
    for sy in (cy0 - 0.158 * S, cy0 + 0.158 * S):                                                     # 2 thumbwheel posts
        g.ellipse([bx + 0.040 * S, sy - 0.018 * S, bx + 0.068 * S, sy + 0.018 * S], fill=(*chrome, 255), outline=(*chrome_d, 255))
    # 2 humbuckers — exact footprint 0.165w x 0.296h: cream mounting ring, twin dark bobbins, gold poles
    for cxn in (0.220, 0.664):
        cxh = bx + cxn * S
        rr([cxh - 0.082 * S, cy0 - 0.148 * S, cxh + 0.082 * S, cy0 + 0.148 * S], (*pu_sur, 255))   # mounting ring
        g.rectangle([cxh - 0.062 * S, cy0 - 0.126 * S, cxh + 0.062 * S, cy0 + 0.126 * S], fill=(*pu_body, 255))  # bobbins
        for rowx in (-0.030, 0.030):                                      # twin coils; poles span ~ the strings
            for p in range(6):
                yy = cy0 - 0.100 * S + p * (0.20 * S / 5)
                g.ellipse([cxh + rowx * S - 0.010 * S, yy - 0.010 * S, cxh + rowx * S + 0.010 * S, yy + 0.010 * S], fill=(*gold, 255))
    # 4 amber top-hat knobs — the measured parallelogram on the lower bout
    for kx_n, ky_n in [(-0.41, 0.30), (-0.17, 0.31), (-0.30, 0.48), (-0.07, 0.49)]:
        kx, ky = P(kx_n, ky_n); kr = 0.055 * S
        g.ellipse([kx - kr, ky - kr, kx + kr, ky + kr], fill=(*amber, 255), outline=(*hs_blk, 170))
        g.ellipse([kx - kr * 0.55, ky - kr * 0.55, kx + kr * 0.55, ky + kr * 0.55], outline=(150, 104, 38, 220))
    # 3-way toggle (upper bout by the cutaway shoulder): cream poker-chip + chrome nut + black tip
    tgx, tgy = P(0.83, -0.29); tr = 0.052 * S
    g.ellipse([tgx - tr, tgy - tr, tgx + tr, tgy + tr], fill=(*pg_cream, 255))            # poker-chip washer
    g.ellipse([tgx - tr * 0.5, tgy - tr * 0.5, tgx + tr * 0.5, tgy + tr * 0.5], fill=(*chrome, 255))
    g.ellipse([tgx - tr * 0.22, tgy - tr * 0.22, tgx + tr * 0.22, tgy + tr * 0.22], fill=(*hs_blk, 255))
    # --- bound rosewood fretboard — runs from the NUT right down to the neck pickup (overhangs the body,
    # per the photo). Les Paul 24.75" scale: fret n at NUT_X - SCALE*(1-2^(-n/12)); trapezoid inlays at
    # 3,5,7,9,12,15,17,19,21 (centred in the space). Tapers narrow at the nut -> wide at the body. ---
    NUT_X, FB_END = 2.67, 0.76; SCALE = NUT_X - 0.054                      # nut -> bridge saddle
    def fb_hw(nx): return (0.088 + 0.020 * (NUT_X - nx) / (NUT_X - FB_END)) * S   # board half-width
    def fnx(n): return NUT_X - SCALE * (1 - 2 ** (-n / 12.0))              # fret position (n=0 nut)
    def board_quad(pad):
        return [(P(FB_END, 0)[0], scy - fb_hw(FB_END) - pad), (P(NUT_X, 0)[0], scy - fb_hw(NUT_X) - pad),
                (P(NUT_X, 0)[0], scy + fb_hw(NUT_X) + pad), (P(FB_END, 0)[0], scy + fb_hw(FB_END) + pad)]
    g.polygon(board_quad(0.012 * S), fill=(*binding, 255))                 # cream binding
    g.polygon(board_quad(0.0), fill=(*fboard, 255))                        # rosewood
    for n in range(1, 23):                                                 # 22 frets (closer toward the body)
        fx = P(fnx(n), 0)[0]; hw = fb_hw(fnx(n))
        g.line([(fx, scy - hw), (fx, scy + hw)], fill=(*fret, 255), width=max(1, int(W * 0.0022)))
    for n in (3, 5, 7, 9, 12, 15, 17, 19, 21):                             # trapezoid inlays
        mx = (fnx(n - 1) + fnx(n)) / 2; cxi = P(mx, 0)[0]; hw = fb_hw(mx)
        iw = (P(fnx(n - 1), 0)[0] - P(fnx(n), 0)[0]) * 0.30; ih = hw * 0.60
        g.polygon([(cxi - iw, scy - ih * 0.82), (cxi + iw, scy - ih), (cxi + iw, scy + ih), (cxi - iw, scy + ih * 0.82)], fill=(*inlay, 255))
    nx_n = P(NUT_X, 0)[0]; nhw = fb_hw(NUT_X) + 0.012 * S                   # bone nut
    g.rectangle([nx_n - 0.012 * S, scy - nhw, nx_n + 0.014 * S, scy + nhw], fill=(238, 232, 214, 255))
    # --- headstock: trace region (from just past the nut) filled black; truss-rod, tuners, logo on top ---
    hs = [P(nx, ny) for nx, ny in cp if nx >= NUT_X + 0.02]
    if len(hs) >= 3:
        g.polygon(hs, fill=(*hs_blk, 255))
    g.polygon([P(2.71, -0.050), P(2.795, -0.060), P(2.83, 0), P(2.795, 0.060), P(2.71, 0.050)],
              fill=(40, 38, 42, 255), outline=(120, 118, 122, 200))        # bell truss-rod cover
    for txn in (2.865, 2.991, 3.118):                                      # 3 + 3 tuners
        for sgn in (-1, 1):
            px, py = P(txn, sgn * 0.085)                                   # face post (string hole)
            g.ellipse([px - 0.026 * S, py - 0.026 * S, px + 0.026 * S, py + 0.026 * S], fill=(*chrome, 255), outline=(*chrome_d, 255))
            g.ellipse([px - 0.009 * S, py - 0.009 * S, px + 0.009 * S, py + 0.009 * S], fill=(118, 120, 126, 255))
            ey, ty = sgn * 0.145, sgn * 0.235                              # keystone button at the edge, outward
            g.polygon([P(txn - 0.030, ey), P(txn + 0.030, ey), P(txn + 0.020, ty), P(txn - 0.020, ty)],
                      fill=(*chrome, 255), outline=(*chrome_d, 220))
    # --- strings: stopbar -> nut, then fanning to the 6 tuner posts ---
    posts = ([P(txn, -0.085) for txn in (2.865, 2.991, 3.118)] +        # top row: outer string -> nearest peg
             [P(txn, 0.085) for txn in (3.118, 2.991, 2.865)])           # bottom row REVERSED -> mirrors the top
    for s in range(6):
        sy = cy0 - 0.095 * S + s * (0.19 * S / 5)
        ny_nut = scy - fb_hw(NUT_X) * 0.78 + s * (fb_hw(NUT_X) * 1.56 / 5)
        g.line([(bx - 0.092 * S, sy), (nx_n, ny_nut)], fill=(232, 232, 236, 175), width=max(1, int(W * 0.0013)))
        g.line([(nx_n, ny_nut), posts[s]], fill=(214, 214, 220, 150), width=max(1, int(W * 0.0013)))
    gl = gl.rotate(ang, center=(bx, by), resample=Image.BICUBIC).resize((w, h), Image.LANCZOS)  # downsample -> AA
    return Image.alpha_composite(img, gl)


def _make_acoustic_guitar_background(w, h, color_top, color_bottom, v=0, rng=None):
    """An acoustic (dreadnought) guitar — folk & acoustic. Warm spruce top, soundhole + rosette.
    Drawn flat (neck to the right) on a layer then rotated up-right; the acoustic counterpart to guitar."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    variants = [(20, 1.00), (15, 0.94), (24, 1.06)]       # (lean°, scale)
    ang, scl = variants[min(v, len(variants) - 1)]
    spruce, bind = (224, 192, 138), (150, 112, 60)
    rose, hole = (196, 150, 70), (40, 28, 18)
    neck_c, fboard, fret, dot = (140, 96, 54), (58, 40, 28), (180, 180, 184), (230, 230, 234)
    strc = (236, 230, 210)
    gl = Image.new("RGBA", (w, h), (0, 0, 0, 0)); g = ImageDraw.Draw(gl, 'RGBA')
    bx, by = w * 0.38, h * 0.58; BL = w * 0.30 * scl; BH = w * 0.32 * scl
    lb = (bx - BL * 0.18, by, BL * 0.64, BH * 1.00)        # lower bout
    ub = (bx + BL * 0.16, by, BL * 0.54, BH * 0.80)        # upper bout
    for ex, ey, ew, eh in (lb, ub):
        g.ellipse([ex - ew / 2, ey - eh / 2, ex + ew / 2, ey + eh / 2], fill=(*bind, 255))         # binding
        g.ellipse([ex - ew / 2 + 3, ey - eh / 2 + 3, ex + ew / 2 - 3, ey + eh / 2 - 3], fill=(*spruce, 255))
    g.polygon([(bx - BL * 0.18, by - BH * 0.42), (bx + BL * 0.16, by - BH * 0.34),
               (bx + BL * 0.16, by + BH * 0.34), (bx - BL * 0.18, by + BH * 0.42)], fill=(*spruce, 255))  # waist
    hx, hy = bx + BL * 0.10, by; hr = BH * 0.17           # soundhole + rosette
    g.ellipse([hx - hr - 4, hy - hr - 4, hx + hr + 4, hy + hr + 4], outline=(*rose, 255), width=max(2, int(w * 0.006)))
    g.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(*hole, 255))
    g.rectangle([bx - BL * 0.10, by - BH * 0.10, bx - BL * 0.02, by + BH * 0.10], fill=(*neck_c, 255))   # bridge
    nx0 = bx + BL * 0.40; NL = w * 0.32; NW = w * 0.044
    g.rectangle([nx0, by - NW / 2, nx0 + NL, by + NW / 2], fill=(*neck_c, 255))
    g.rectangle([nx0, by - NW * 0.42, nx0 + NL, by + NW * 0.42], fill=(*fboard, 255))
    for k in range(1, 8):
        fxx = nx0 + NL * k / 8
        g.line([(fxx, by - NW * 0.42), (fxx, by + NW * 0.42)], fill=(*fret, 255), width=max(1, int(w * 0.002)))
    for k in (3, 5):
        cxk = nx0 + NL * k / 8
        g.ellipse([cxk - w * 0.005, by - w * 0.005, cxk + w * 0.005, by + w * 0.005], fill=(*dot, 255))
    hx0 = nx0 + NL                                          # acoustic headstock + 3+3 tuners
    try:
        g.rounded_rectangle([hx0, by - NW * 0.7, hx0 + w * 0.075, by + NW * 0.7], radius=int(w * 0.01), fill=(*neck_c, 255))
    except Exception:
        g.rectangle([hx0, by - NW * 0.7, hx0 + w * 0.075, by + NW * 0.7], fill=(*neck_c, 255))
    for k in range(3):
        tx = hx0 + w * 0.016 + k * w * 0.022
        g.ellipse([tx - w * 0.005, by - NW * 0.85, tx + w * 0.005, by - NW * 0.6], fill=(220, 220, 224, 255))
        g.ellipse([tx - w * 0.005, by + NW * 0.6, tx + w * 0.005, by + NW * 0.85], fill=(220, 220, 224, 255))
    for s in range(6):                                      # strings: bridge → headstock
        sy = by - NW * 0.32 + s * (NW * 0.64 / 5)
        g.line([(bx - BL * 0.06, sy), (hx0 + w * 0.02, sy)], fill=(*strc, 180), width=1)
    gl = gl.rotate(ang, center=(bx, by), resample=Image.BICUBIC)
    return Image.alpha_composite(img, gl)


def _make_checkerboard_background(w, h, color_top, color_bottom, v=0, rng=None):
    """2-tone black/white check — ska, 2-tone, mod-adjacent."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0)); draw = ImageDraw.Draw(overlay, 'RGBA')
    cols = [8, 6, 10, 7][min(v, 3)]
    cell = w / cols
    black, white = (24, 24, 28), (240, 240, 244)
    rows = int(h * 0.74 / cell) + 1
    for r in range(rows):
        for c in range(cols):
            col = white if (r + c) % 2 == 0 else black
            draw.rectangle([c * cell, r * cell, c * cell + cell, r * cell + cell], fill=(*col, 150))
    return Image.alpha_composite(img, overlay)


def _make_desert_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Rolling sand dunes + low sun — stoner/desert rock, outlaw country."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _c01(x): return max(0.0, min(1.0, x))
    def _lighten(c, amt=85): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.42): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    sx = w * [0.5, 0.36, 0.64, 0.46][min(v, 3)]; sr = w * 0.16; sy = h * 0.40
    draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(*L, 180))                          # sun
    layers = 4
    def tone(fr): fr = _c01(fr); return tuple(int(L[k] + (D[k] - L[k]) * fr) for k in range(3))
    for li in range(layers):                                                                    # smooth dunes
        frac = li / (layers - 1); base = h * (0.46 + 0.12 * li)
        pts = [(0, h)]
        for x in range(0, w + 1, 24):
            pts.append((x, base + math.sin(x * 0.004 + li * 1.7) * h * 0.05))
        pts.append((w, h))
        draw.polygon(pts, fill=(*tone(frac), 210))
    return img


def _make_motion_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Dynamic speed streaks — running, walking, workout, high-energy activity."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=95): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.5): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    n = [22, 30, 16, 36][min(v, 3)]
    ang = math.radians([12, 8, 16, 10][min(v, 3)])
    dx, dy = math.cos(ang), math.sin(ang)
    for _ in range(n):
        y = rng.uniform(0, h * 0.74); x0 = rng.uniform(-w * 0.1, w * 0.6)
        ln = w * rng.uniform(0.18, 0.5); lw = max(2, int(h * rng.uniform(0.004, 0.016)))
        col = L if rng.random() < 0.5 else D
        draw.line([(x0, y), (x0 + dx * ln, y + dy * ln)], fill=(*col, rng.randint(70, 150)), width=lw)
    return img


def _make_traffic_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Car light-trails on a night road (long exposure) — commute / beat the traffic."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    rng = rng or random.Random(v)
    red, white = (235, 70, 60), (250, 240, 210)
    n = [6, 8, 5, 9][min(v, 3)]
    for i in range(n):
        y = h * (0.16 + 0.62 * i / max(1, n - 1)) + rng.uniform(-0.02, 0.02) * h
        col = red if i % 2 else white
        amp = h * 0.02
        pts = [(x, y + math.sin(x * 0.006 + i) * amp) for x in range(0, w + 1, 12)]
        draw.line(pts, fill=(*col, rng.randint(90, 150)), width=max(3, int(h * 0.012)))
    return img


def _make_beach_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Sea horizon + sun + rolling waves — beach, summer, tropical."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=90): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.5): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    hz = h * [0.50, 0.46, 0.54, 0.48][min(v, 3)]
    sx = w * [0.5, 0.36, 0.64, 0.5][min(v, 3)]; sr = w * 0.15
    draw.ellipse([sx - sr, hz - sr, sx + sr, hz + sr], fill=(*L, 200))                          # sun
    draw.rectangle([0, hz, w, h], fill=(*D, 90))                                                # sea
    for i in range(5):                                                                          # wave lines
        wy = hz + (h - hz) * (0.12 + 0.18 * i)
        pts = [(x, wy + math.sin(x * 0.02 + i) * h * 0.012) for x in range(0, w + 1, 12)]
        draw.line(pts, fill=(*L, 110 - i * 12), width=max(2, int(h * 0.006)))
    return img


def _make_wedding_rings_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Two interlocking gold rings + petals/sparkles — wedding day."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    rng = rng or random.Random(v)
    gold = (236, 206, 120)
    cx, cy = w * 0.50, h * 0.40
    r = w * [0.12, 0.10, 0.13, 0.11][min(v, 3)]; off = r * 0.72
    lw = max(6, int(w * 0.022))
    draw.ellipse([cx - off - r, cy - r, cx - off + r, cy + r], outline=(*gold, 235), width=lw)
    draw.ellipse([cx + off - r, cy - r, cx + off + r, cy + r], outline=(*gold, 235), width=lw)
    draw.arc([cx + off - r, cy - r, cx + off + r, cy + r], 150, 250, fill=(*gold, 120), width=lw)  # interlock hint
    pinks = [(255, 200, 215), (250, 182, 205), (255, 226, 234)]
    bar = h * 0.74
    for _ in range([16, 22, 12, 28][min(v, 3)]):
        x, y = rng.uniform(0, w), rng.uniform(0, bar); s = w * rng.uniform(0.010, 0.022)
        draw.ellipse([x - s, y - s * 0.6, x + s, y + s * 0.6], fill=(*rng.choice(pinks), rng.randint(110, 180)))
    return img


def _make_lounge_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Cocktail glass + warm glow + bokeh — after-hours r&b, late-night lounge (NOT a skyline)."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False).convert("RGBA")
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    warm = tuple(max(0, min(255, c + 95)) for c in color_top[:3])
    rng = rng or random.Random(v)
    for _ in range([9, 12, 7, 14][min(v, 3)]):                     # bokeh
        r = w * rng.uniform(0.04, 0.11); bx, by = rng.uniform(0, w), rng.uniform(0, h * 0.72)
        gd.ellipse([bx - r, by - r, bx + r, by + r], fill=(*warm, rng.randint(45, 95)))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(radius=16)))
    draw = ImageDraw.Draw(img, 'RGBA')
    D = tuple(int(x * 0.30) for x in color_bottom[:3])
    cx, cy = w * [0.50, 0.42, 0.58, 0.50][min(v, 3)], h * 0.40
    gw = w * 0.14; lw = max(4, int(w * 0.012))
    draw.line([(cx - gw, cy - gw * 0.85), (cx + gw, cy - gw * 0.85)], fill=(*D, 220), width=lw)   # rim
    draw.polygon([(cx - gw, cy - gw * 0.85), (cx + gw, cy - gw * 0.85), (cx, cy + gw * 0.15)], outline=(*D, 230), fill=(*warm, 70))  # bowl
    draw.line([(cx, cy + gw * 0.15), (cx, cy + gw * 1.05)], fill=(*D, 220), width=lw)             # stem
    draw.line([(cx - gw * 0.5, cy + gw * 1.05), (cx + gw * 0.5, cy + gw * 1.05)], fill=(*D, 220), width=lw)  # base
    draw.line([(cx + gw * 0.3, cy - gw * 0.5), (cx + gw * 0.55, cy - gw * 0.9)], fill=(*D, 200), width=max(2, lw // 2))  # pick
    draw.ellipse([cx + gw * 0.48, cy - gw * 1.0, cx + gw * 0.62, cy - gw * 0.86], fill=(120, 160, 70, 220))  # olive
    return img


def _make_prism_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Prism splitting a white beam into a spectrum fan — prog & art rock, art-pop."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    L = tuple(_clamp(x + 95) for x in color_top[:3])
    cx, cy = w * [0.44, 0.38, 0.50, 0.42][min(v, 3)], h * 0.40
    tri = w * 0.13
    spectrum = [(232, 44, 44), (240, 140, 44), (242, 220, 64), (60, 200, 96), (60, 140, 232), (146, 72, 210)]
    ox, oy = cx + tri * 0.55, cy + tri * 0.25
    for i, col in enumerate(spectrum):                            # rainbow fan out
        a = math.radians(-22 + i * 9)
        draw.line([(ox, oy), (ox + math.cos(a) * w * 0.55, oy + math.sin(a) * w * 0.55)],
                  fill=(*col, 150), width=max(3, int(h * 0.012)))
    draw.line([(0, cy - tri * 0.1), (cx - tri * 0.5, cy - tri * 0.1)], fill=(*L, 180), width=max(3, int(h * 0.01)))  # white beam in
    draw.polygon([(cx, cy - tri), (cx - tri * 0.9, cy + tri * 0.7), (cx + tri * 0.9, cy + tri * 0.7)],
                 fill=(*L, 70), outline=(*L, 200))               # prism
    return img


def _make_crescendo_background(w, h, color_top, color_bottom, v=0, rng=None):
    """A building sound-swell rising left→right — post-rock, cinematic build."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _lighten(c, amt=70): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.5): return tuple(int(x * fac) for x in c[:3])
    def _mid(c1, c2): return tuple((a + b) // 2 for a, b in zip(c1[:3], c2[:3]))
    L = _lighten(color_top); D = _darken(color_bottom); M = _mid(color_top, color_bottom)
    base_y = h * 0.74
    freq = [1.6, 2.2, 1.2, 2.6][min(v, 3)]
    for layer, (col, a) in enumerate([(D, 150), (M, 150), (L, 150)]):
        pts = [(0, base_y)]
        for x in range(0, w + 1, 10):
            t = x / w
            amp = h * (0.04 + 0.46 * t) * (1 - layer * 0.18)     # amplitude grows toward the right = crescendo
            pts.append((x, base_y - amp * (0.55 + 0.45 * math.sin(x * 0.012 * freq + layer))))
        pts += [(w, base_y)]
        draw.polygon(pts, fill=(*col, a))
    return img


def _make_meadow_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Rolling green hills + wildflowers — spring acoustic, spring, pastoral folk."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    draw = ImageDraw.Draw(img, 'RGBA')
    def _clamp(x): return max(0, min(255, x))
    def _c01(x): return max(0.0, min(1.0, x))
    def _lighten(c, amt=80): return tuple(_clamp(x + amt) for x in c[:3])
    def _darken(c, fac=0.5): return tuple(int(x * fac) for x in c[:3])
    L = _lighten(color_top); D = _darken(color_bottom)
    rng = rng or random.Random(v)
    def tone(fr): fr = _c01(fr); return tuple(int(L[k] + (D[k] - L[k]) * fr) for k in range(3))
    hills = 3
    tops = []
    for li in range(hills):
        frac = li / (hills - 1); base = h * (0.50 + 0.12 * li)
        pts = [(0, h)]
        for x in range(0, w + 1, 24):
            pts.append((x, base + math.sin(x * 0.004 + li * 2.1) * h * 0.05))
        pts += [(w, h)]
        draw.polygon(pts, fill=(*tone(frac), 215)); tops.append((base, frac))
    flowers = [(255, 210, 90), (250, 130, 170), (250, 250, 250), (200, 120, 230)]
    for _ in range([18, 26, 12, 32][min(v, 3)]):                  # wildflowers on the front hills
        x = rng.uniform(0, w); y = rng.uniform(h * 0.58, h * 0.72); fr = w * rng.uniform(0.008, 0.016)
        col = rng.choice(flowers)
        for k in range(5):
            a = math.radians(k * 72)
            draw.ellipse([x + math.cos(a) * fr - fr * 0.5, y + math.sin(a) * fr - fr * 0.5,
                          x + math.cos(a) * fr + fr * 0.5, y + math.sin(a) * fr + fr * 0.5], fill=(*col, 200))
    return img


# _make_papel_picado_background retired (round 4) — family removed; latin_heat→palm_sunburst, summer_heat→sun_horizon.


def _make_kente_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Bold woven kente strips with geometric motifs — afrobeat, African."""
    img = _make_gradient_image(w, h, color_top, color_bottom, diagonal=False)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0)); draw = ImageDraw.Draw(overlay, 'RGBA')
    kente = [(34, 34, 40), (224, 172, 44), (208, 56, 52), (40, 150, 92)]
    n = [7, 9, 6, 10][min(v, 3)]
    strip = h / n
    for r in range(n):
        y0 = r * strip; base = kente[r % 4]
        draw.rectangle([0, y0, w, y0 + strip], fill=(*base, 130))
        motif = kente[(r + 2) % 4]; blocks = 8
        for c in range(blocks):
            bx = w * (c + 0.5) / blocks
            if (r + c) % 2 == 0:                                  # alternating squares
                draw.rectangle([bx - strip * 0.28, y0 + strip * 0.22, bx + strip * 0.28, y0 + strip * 0.78], fill=(*motif, 150))
            else:                                                 # diamonds
                draw.polygon([(bx, y0 + strip * 0.18), (bx + strip * 0.3, y0 + strip * 0.5),
                              (bx, y0 + strip * 0.82), (bx - strip * 0.3, y0 + strip * 0.5)], fill=(*motif, 150))
    return Image.alpha_composite(img, overlay)


# ── Background generator registry ─────────────────────────────────────────────
# Maps a _COVER_BG_STYLES style name -> its generator fn. Adding a new pattern is one
# line here plus the _make_<name>_background above — no edits to the cover dispatcher.
# WHY a registry (not an if/elif chain): the redesign adds ~25 genre/mood-evocative
# families, and validator/contact-sheet import this as the single source of "known styles".
# NEW generators are defined just above this dict (search "# === NEW genre/mood generators").
def _make_union_jack_background(w, h, color_top, color_bottom, v=0, rng=None):
    """Stylised Union Jack — blue ground, white + red diagonal saltires, white + red St George cross on top.
    Hard-coded flag colours (ignores the gradient). Left-right symmetric -> listed in _BG_SYMMETRIC (no h-flip twin)."""
    blue, white, red = (12, 48, 135), (244, 244, 248), (200, 16, 47)
    # (cross half-width frac, saltire width frac, red/white ratio) — subtle proportion shifts per variant
    configs = [(0.115, 0.16, 0.46), (0.10, 0.14, 0.50), (0.13, 0.18, 0.42), (0.105, 0.15, 0.48)]
    cw, sw, ratio = configs[min(v, len(configs) - 1)]
    img  = _make_gradient_image(w, h, blue, (8, 30, 90), diagonal=False)   # blue ground — RGBA, like every other generator
    draw = ImageDraw.Draw(img, 'RGBA')
    wsw = max(2, int(w * sw)); rsw = max(2, int(wsw * ratio))           # white then narrower red saltire (diagonals)
    for col, width in ((white, wsw), (red, rsw)):
        draw.line([(0, 0), (w, h)], fill=(*col, 255), width=width)
        draw.line([(0, h), (w, 0)], fill=(*col, 255), width=width)
    wcw = max(2, int(w * cw)); rcw = max(2, int(wcw * 0.58))            # white then narrower red St George cross, on top
    cx, cy = w // 2, h // 2
    draw.rectangle([cx - wcw, 0, cx + wcw, h], fill=(*white, 255))
    draw.rectangle([0, cy - wcw, w, cy + wcw], fill=(*white, 255))
    draw.rectangle([cx - rcw, 0, cx + rcw, h], fill=(*red, 255))
    draw.rectangle([0, cy - rcw, w, cy + rcw], fill=(*red, 255))
    return img

_BG_GENERATORS = {
    "union_jack":       _make_union_jack_background,
    "geometric":        _make_geometric_background,
    "circles":          _make_concentric_circles_background,
    "radial":           _make_radial_glow_background,
    "ripples":          _make_ripples_background,
    "waves":            _make_waves_background,
    "floating_circles": _make_floating_circles_background,
    "rays":             _make_rays_background,
    "arc_sweep":        _make_arc_sweep_background,
    "aurora":           _make_aurora_background,
    "triangles":        _make_triangles_background,
    "diamond":          _make_diamond_background,
    "starburst":        _make_starburst_background,
    "chevrons":         _make_chevrons_background,
    "spiral":           _make_spiral_background,
    # --- new: electronic / dance / rock-indie cluster ---
    "equalizer":        _make_equalizer_background,
    "grid_perspective": _make_grid_perspective_background,
    "circuit":          _make_circuit_background,
    "waveform":         _make_waveform_background,
    "laser_fan":        _make_laser_fan_background,
    "concentric_pulse": _make_concentric_pulse_background,
    "low_poly":         _make_low_poly_background,
    # --- new: soul/funk · pop · jazz/classical · folk · ambient ---
    "vinyl_grooves":    _make_vinyl_grooves_background,
    "halftone":         _make_halftone_background,
    "brushstrokes":     _make_brushstrokes_background,
    "smoke":            _make_smoke_background,
    "staff_lines":      _make_staff_lines_background,
    "mountains":        _make_mountains_background,
    "starfield":        _make_starfield_background,
    # --- new: tropical/sunset · wood · rock-amp · aggressive · party · place ---
    "palm_sunburst":    _make_palm_sunburst_background,
    "sun_horizon":      _make_sun_horizon_background,
    "woodgrain":        _make_woodgrain_background,
    "amp_stack":        _make_amp_stack_background,
    "shards":           _make_shards_background,
    "confetti":         _make_confetti_background,
    "tartan":           _make_tartan_background,
    "cityscape":        _make_cityscape_background,
    # --- round 2: genre families ---
    "mod_target":       _make_mod_target_background,
    "disco_ball":       _make_disco_ball_background,
    "cassette":         _make_cassette_background,
    "pixel_grid":       _make_pixel_grid_background,
    "columns":          _make_columns_background,
    "film_strip":       _make_film_strip_background,
    "stained_glass":    _make_stained_glass_background,
    # --- round 2: mood / romance / calm / night ---
    "candle_glow":      _make_candle_glow_background,
    "bokeh":            _make_bokeh_background,
    "clouds":           _make_clouds_background,
    "zen":              _make_zen_background,
    "grid_paper":       _make_grid_paper_background,
    "moonlight":        _make_moonlight_background,
    "cosmos":           _make_cosmos_background,
    "lightning":        _make_lightning_background,
    # --- round 2: season / weather / place / activity ---
    "blossom":          _make_blossom_background,
    "snowfall":         _make_snowfall_background,
    "rainfall":         _make_rainfall_background,
    "pine_forest":      _make_pine_forest_background,
    "tropical_leaves":  _make_tropical_leaves_background,
    "gingham":          _make_gingham_background,
    "open_road":        _make_open_road_background,
    "holiday_lights":   _make_holiday_lights_background,
    # --- round 3: genre-true families ---
    "jazz_club":        _make_jazz_club_background,
    "strings":          _make_strings_background,
    "guitar":           _make_guitar_background,
    "acoustic_guitar":  _make_acoustic_guitar_background,
    "checkerboard":     _make_checkerboard_background,
    "desert":           _make_desert_background,
    "motion":           _make_motion_background,
    "traffic":          _make_traffic_background,
    "beach":            _make_beach_background,
    "wedding_rings":    _make_wedding_rings_background,
    "lounge":           _make_lounge_background,
    "prism":            _make_prism_background,
    "crescendo":        _make_crescendo_background,
    "meadow":           _make_meadow_background,
    "kente":            _make_kente_background,
}


def _bg_variant_count(fn):
    """How many variants a generator defines = length of its first `configs`/`variants` list literal.
    WHY: a _COVER_BG_STYLES entry with v >= this count silently clamps to the last variant
    (the root of the old indie_rock/rap_rock identical-twin bug); cover_validator uses this to flag it."""
    import ast, inspect, re
    try:
        src = inspect.getsource(fn)
    except Exception:
        return None
    try:
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("configs", "variants")):
                continue
            val = node.value
            if isinstance(val, ast.List):                         # literal table
                return len(val.elts)
            if isinstance(val, ast.ListComp) and len(val.generators) == 1:   # [.. for i in range(N)]
                it = val.generators[0].iter
                if (isinstance(it, ast.Call) and isinstance(getattr(it, "func", None), ast.Name)
                        and it.func.id == "range" and it.args
                        and all(isinstance(a, ast.Constant) for a in it.args)):
                    args = [a.value for a in it.args]
                    return args[0] if len(args) == 1 else args[1] - args[0]
    except Exception:
        pass
    # inline pattern `[...][min(v, K)]` (many round-2 generators) → K + 1 variants
    ks = [int(m) for m in re.findall(r"min\(v,\s*(\d+)\)", src)]
    return max(ks) + 1 if ks else None

# Generators whose variant table is an inline literal / procedural (no named configs list) — set by hand.
_BG_VARIANT_COUNTS_OVERRIDE = {"ripples": 6, "confetti": 8, "tartan": 4, "cityscape": 4, "brushstrokes": 10}
_BG_BASE_COUNTS = {name: (_BG_VARIANT_COUNTS_OVERRIDE.get(name) or _bg_variant_count(fn) or 1)
                   for name, fn in _BG_GENERATORS.items()}

# Families whose horizontal mirror looks identical (centred/symmetric) — these get NO flip-doubling.
# WHY: an h-flip of an off-centre/asymmetric pattern is a genuinely different cover (and the per-key
# rng means even the same base variant renders differently per playlist), so asymmetric families
# offer 2× distinct variants for free — enough headroom for ~269 unique covers without huge tables.
_BG_SYMMETRIC = {"union_jack", "starburst", "amp_stack", "staff_lines", "vinyl_grooves", "tartan",
                 # round-2 deterministic/centred families whose h-mirror would be a redundant twin:
                 "mod_target", "columns", "film_strip", "zen", "grid_paper", "pine_forest",
                 "tropical_leaves", "gingham", "cassette", "open_road", "holiday_lights",
                 # round 3 centred/symmetric families:
                 "strings", "checkerboard", "wedding_rings", "lounge", "kente", "jazz_club",
                 # guitar: a flipped Les Paul reads as a different (left-handed) instrument — NEVER mirror it.
                 # Its extra variants are real colour finishes (10), not flip-twins. WHY: user requirement.
                 "guitar"}
_BG_VARIANT_COUNTS = {n: _BG_BASE_COUNTS[n] * (1 if n in _BG_SYMMETRIC else 2) for n in _BG_GENERATORS}


def _render_bg(style, color_top, color_bottom, v, rng):
    """Render background `style` at variant `v`. Variants beyond a generator's base table are the
    base pattern horizontally mirrored (asymmetric families only) — a free 2× distinct-variant space."""
    gen  = _BG_GENERATORS.get(style, _make_geometric_background)
    base = _BG_BASE_COUNTS.get(style, 1)
    img  = gen(1000, 1000, color_top, color_bottom, v % base, rng=rng)
    if (v // base) % 2 and style not in _BG_SYMMETRIC:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# ── Cover icon overlays ──────────────────────────────────────────────────────

def _icon_light(c, f=0.45, alpha=200):
    """Lightened, slightly warm variant of colour c for icon overlays."""
    return (max(0, min(255, int(c[0] + (255 - c[0]) * f))),
            max(0, min(255, int(c[1] + (255 - c[1]) * f * 0.95))),
            max(0, min(255, int(c[2] + (255 - c[2]) * f * 0.85))),
            alpha)


# --- Material Symbols icon font (variable; rendered by codepoint) ---------------
MS_ICON_FONT = os.path.join(
    _BASE_DIR, "assets", "Material_Symbols_Rounded",
    "MaterialSymbolsRounded-VariableFont_FILL,GRAD,opsz,wght.ttf")

# Material Symbol name → codepoint (extracted from the font's cmap). Extend as needed.
_MS_CODEPOINTS = {
    "ac_unit": 0xEB3B,
    "air": 0xEFD8,
    "foggy": 0xE818,
    "bedtime": 0xF159,
    "hot_tub": 0xEB46,
    "menu_book": 0xEA19,
    "potted_plant": 0xF8AA,
    "self_improvement": 0xEA78,
    "wb_twilight": 0xE1C6,
    "headphones": 0xF01F,
    "mic": 0xE31D,
    "movie": 0xE684,
    "stadia_controller": 0xF135,
    "favorite": 0xE87E, "heart_broken": 0xEAC2, "candle": 0xF588, "music_note": 0xE405,
    "music_note_2": 0xFFFD8, "piano": 0xE521, "graphic_eq": 0xE1B8, "bedtime": 0xF159, "sailing": 0xE502,
    "dark_mode": 0xE51C, "wb_sunny": 0xE430, "clear_day": 0xF157, "beach_access": 0xEB3E,
    "rainy": 0xF176, "water_drop": 0xE798, "local_cafe": 0xEB44, "restaurant": 0xE56C,
    "wine_bar": 0xF1E8, "skillet": 0xF543, "fitness_center": 0xEB43, "directions_run": 0xE566,
    "local_fire_department": 0xEF55, "trending_up": 0xE8E5, "whatshot": 0xE80E,
    "thunderstorm": 0xEBDB, "celebration": 0xEA65, "nightlife": 0xEA62, "festival": 0xEA68,
    "mood": 0xEA22, "snowflake": 0xED5B, "forest": 0xEA99, "local_florist": 0xE545,
    "center_focus_strong": 0xE3B4, "self_improvement": 0xEA78, "spa": 0xEB4C, "cloud": 0xF15C,
    "directions_car": 0xEFF7, "route": 0xEACD, "directions_walk": 0xE536, "star_shine": 0xF31D,
    "weekend": 0xE16B, "repeat": 0xE040, "replay": 0xE042, "radar": 0xF04E,
    "travel_explore": 0xE2DB, "explore": 0xE87A, "history": 0xE8B3,
    "moon_stars": 0xF34F, "partly_cloudy_night": 0xEA46, "local_bar": 0xE540,
    "brunch_dining": 0xEA73, "bakery_dining": 0xEA53, "egg_alt": 0xEAC8, "flare": 0xE3E4,
    # --- added in the cover redesign (codepoints extracted from the font via ms_glyphs.py) ---
    "eco": 0xEA35, "skillet": 0xF543, "bolt": 0xEA0B, "palette": 0xE3B7, "casino": 0xEB40,
    "church": 0xEAAE, "thermostat": 0xF076, "album": 0xE019, "redeem": 0xE8B1, "trophy": 0xE71A,
    "star": 0xE838,
    "radio": 0xE03E,
    "outdoor_grill": 0xEA47,
}

# Two-tone glyphs: drawn as a filled duotone (see _draw_glyph) — a solid FILL=1 body in the glyph
# colour + a darker-tone FILL=0 interior/outline detail (a screen, pages, a cup, a glass…) on top.
# Solid symbols (heart, note, star, sun, drop) stay single-tone FILL=1.
_TWO_TONE_GLYPHS = {
    "movie", "menu_book", "local_cafe", "restaurant", "stadia_controller", "headphones",
    "wine_bar", "local_bar", "brunch_dining", "bakery_dining", "egg_alt", "piano", "weekend",
    "hot_tub",
}

_MS_FONT_CACHE = {}


def _load_ms_font(size, fill=1):
    """Material Symbols variable font at `size` px em, heavy / opsz 48. `fill` selects the
    FILL axis (1 = solid, 0 = outlined two-tone). Cached by (size, fill)."""
    key = (size, fill)
    font = _MS_FONT_CACHE.get(key)
    if font is None:
        font = ImageFont.truetype(MS_ICON_FONT, size)
        try:
            font.set_variation_by_axes([fill, 0, 48, 700])  # FILL, GRAD, opsz, wght
        except Exception:
            pass
        _MS_FONT_CACHE[key] = font
    return font


def _draw_glyph(layer, icon_name, cx, cy, size, fill, tilt=0, flip=False,
                stroke_width=0, stroke_fill=None, fill2=None):
    """Render a Material Symbol glyph centred at (cx, cy) at `size` px em, optionally tilted,
    mirrored (flip = horizontal) and outlined (stroke), compositing onto RGBA `layer` via its
    own tile (so it can rotate/mirror freely). A two-tone glyph with `fill2` is drawn as a filled
    duotone: a solid FILL=1 body in `fill` + a FILL=0 interior/outline detail in `fill2` on top."""
    cp = _MS_CODEPOINTS.get(icon_name)
    if cp is None or not _PIL_AVAILABLE:
        return
    ch       = chr(cp)
    sz       = max(8, int(round(size)))
    two_tone = icon_name in _TWO_TONE_GLYPHS and fill2 is not None
    font = _load_ms_font(sz, 1 if two_tone else (0 if icon_name in _TWO_TONE_GLYPHS else 1))
    l, t, r, b = font.getbbox(ch, stroke_width=stroke_width)     # ink bounds, to centre
    gw, gh = max(1, r - l), max(1, b - t)
    pad = max(8, int(size * 0.10) + stroke_width)
    tile = Image.new("RGBA", (gw + 2 * pad, gh + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    draw.text((pad - l, pad - t), ch, font=font, fill=fill,
              stroke_width=stroke_width, stroke_fill=stroke_fill)
    if two_tone:                                                 # darker interior/outline detail over the body
        draw.text((pad - l, pad - t), ch, font=_load_ms_font(sz, 0), fill=fill2)
    if flip:
        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
    if tilt:
        tile = tile.rotate(tilt, resample=Image.BICUBIC, expand=True)
    layer.alpha_composite(tile, (int(round(cx - tile.width / 2)),
                                 int(round(cy - tile.height / 2))))


# When set to a list (by cover_validator), the cluster/scatter helpers append every placed glyph's
# (x, y, radius) so the validator can assert no two glyphs on a cover overlap. None in normal use.
_CLUSTER_FOOTPRINTS = None
_GLYPH_RADIUS_FRAC = 0.46                     # Material Symbol visual radius ≈ this × em size


def _rec_footprint(x, y, r):
    if _CLUSTER_FOOTPRINTS is not None:
        _CLUSTER_FOOTPRINTS.append((x, y, r))


def _draw_glyph_cluster(layer, names, cx, cy, anchor_size, rng, fill, n=3, stroke=None,
                        ring_mult=0.50, sat_lo=0.50, sat_hi=0.62):
    """Composed constellation: one anchor glyph + (n-1) satellites on a ring, sized and spaced
    so NO two glyphs overlap. WHY computed (not fixed) ring: the old ring (0.50×anchor) was
    smaller than anchor_radius+satellite_radius, so satellites always overlapped the anchor and
    relied on outlines to stay legible. Now the ring is the max of (a) clear-the-anchor, (b)
    clear-neighbouring-satellites, (c) an explicit `ring_mult` lower bound. `names` is cycled
    across glyphs (["music_note","music_note_2"]) or repeated (["favorite"], ["flare"])."""
    sw, sf = stroke or (0, None)

    def g(name, gx, gy, gsize, gtilt):
        _draw_glyph(layer, name, gx, gy, gsize, fill, tilt=gtilt,
                    stroke_width=int(round(gsize * sw)) if sw else 0, stroke_fill=sf)
        _rec_footprint(gx, gy, gsize * _GLYPH_RADIUS_FRAC)

    if n <= 1:
        g(names[0], cx, cy, anchor_size, rng.uniform(-8, 8))
        return

    GR = 0.46                                       # Material Symbol visual radius ≈ 0.46 × em size
    margin    = anchor_size * 0.05
    sat_sizes = [anchor_size * rng.uniform(sat_lo, sat_hi) for _ in range(n - 1)]
    rs_max    = max(sat_sizes) * GR
    ring      = anchor_size * GR + rs_max + margin             # (a) clear the anchor
    if n - 1 >= 2:                                             # (b) clear neighbouring satellites
        ring = max(ring, (rs_max + margin * 0.5) / math.sin(math.pi / (n - 1)))
    ring = max(ring, anchor_size * ring_mult)                  # (c) honour an explicit wider spread

    g(names[0], cx, cy, anchor_size, rng.uniform(-8, 8))
    base_a = rng.uniform(0, 2 * math.pi)
    for i in range(1, n):
        ang = base_a + 2 * math.pi * (i - 1) / (n - 1) + rng.uniform(-0.10, 0.10)
        g(names[i % len(names)], cx + ring * math.cos(ang), cy + ring * math.sin(ang),
          sat_sizes[i - 1], rng.uniform(-15, 15))


def _draw_falling_cluster(layer, cloud_name, flake_name, cx, cy, size, rng, fill, stroke=None):
    """A cloud with several flakes falling below it (Winter). Flakes are drawn FIRST so the cloud
    composites ON TOP, and each flake is placed with a min-separation (reject-and-retry) so no two
    flakes overlap each other or sit under the cloud. `size` is the cloud's em size."""
    sw, sf = stroke or (0, None)

    def g(name, gx, gy, gsize, gtilt):
        _draw_glyph(layer, name, gx, gy, gsize, fill, tilt=gtilt,
                    stroke_width=int(round(gsize * sw)) if sw else 0, stroke_fill=sf)
        _rec_footprint(gx, gy, gsize * _GLYPH_RADIUS_FRAC)

    GR = 0.46
    cloud_y = cy - size * 0.70
    placed  = [(cx, cloud_y, size * GR * 0.92)]          # reserve the cloud footprint
    target, tries = rng.randint(6, 7), 0
    while len(placed) - 1 < target and tries < 200:
        tries += 1
        fsize = size * rng.uniform(0.24, 0.38)
        fr    = fsize * GR
        fx    = cx + rng.uniform(-1.0, 1.0) * size
        fy    = cloud_y + size * rng.uniform(0.60, 2.10)            # below the cloud
        if all((fx - px) ** 2 + (fy - py) ** 2 > (fr + pr) ** 2 for px, py, pr in placed):
            g(flake_name, fx, fy, fsize, rng.uniform(-22, 22))      # flakes first
            placed.append((fx, fy, fr))
    g(cloud_name, cx, cloud_y, size, rng.uniform(-6, 6))           # cloud composited on top


def _icon_dark(c, f=0.42):
    """Deepened (darkened) variant of colour c — used for icons on bright backgrounds."""
    return tuple(max(8, int(x * f)) for x in c[:3])


def _ensure_icon_contrast(rgb, bg_lum, min_gap=72):
    """Keep an icon colour's hue but nudge it lighter or darker only as much as needed to
    hold a minimum luminance gap from the background under it (pushing whichever way the
    colour already leans relative to the background)."""
    il  = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    gap = abs(il - bg_lum)
    if gap >= min_gap:
        return tuple(int(c) for c in rgb[:3])
    t = min(0.80, (min_gap - gap) / 150.0)
    if il >= bg_lum:                                          # already lighter → lighten
        return tuple(int(c + (255 - c) * t) for c in rgb[:3])
    return tuple(int(c * (1 - t)) for c in rgb[:3])          # darker → darken


def _bg_luminance(base, cx, cy, R):
    """Mean perceived luminance (0–255) of `base` (RGBA) under the icon footprint."""
    W, H = base.size
    x0, x1 = max(0, int(cx - R)), min(W, int(cx + R))
    y0, y1 = max(0, int(cy - R)), min(H, int(cy + R))
    if x1 <= x0 or y1 <= y0:
        return 128.0
    region = base.crop((x0, y0, x1, y1)).convert("RGB")
    if _NUMPY_AVAILABLE:
        arr = np.asarray(region, dtype=np.float32)
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        return float(lum.mean())
    px = region.load()
    rw, rh = region.size
    step = max(1, min(rw, rh) // 12)
    total, n = 0.0, 0
    for yy in range(0, rh, step):
        for xx in range(0, rw, step):
            r, g, b = px[xx, yy]
            total += 0.299 * r + 0.587 * g + 0.114 * b
            n += 1
    return total / max(1, n)


def _place_icon(base, overlay, cx, cy, scale=1.0, angle=0.0, shadow=True, bg_lum=None):
    """Rotate + scale `overlay` about (cx, cy), add a contrast shadow that hugs the icon
    shape, then composite onto `base`. Returns a new RGBA image."""
    W, H = base.size
    if angle:
        overlay = overlay.rotate(angle, center=(cx, cy), resample=Image.BICUBIC)
    if scale and abs(scale - 1.0) > 1e-3:
        sw, sh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
        scaled = overlay.resize((sw, sh), resample=Image.BICUBIC)
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        # paste (not alpha_composite) so negative offsets from scale>1 crop correctly
        layer.paste(scaled, (int(round(cx - cx * scale)), int(round(cy - cy * scale))), scaled)
        overlay = layer
    if shadow:
        # Soft dark drop shadow (down-right) that hugs the icon shape — lifts it off the
        # background for separation without the neon halo a light glow would produce.
        sh_alpha = overlay.getchannel("A").point(lambda v: int(v * 0.5))
        tint = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        tint.putalpha(sh_alpha)
        tint = tint.filter(ImageFilter.GaussianBlur(radius=11))
        drop = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        drop.alpha_composite(tint, (4, 6))
        base = Image.alpha_composite(base, drop)
    return Image.alpha_composite(base, overlay)


def _draw_ripples_below(layer, cx, cy, size, rgb, alpha):
    """A few small ripples CLEARLY below a rain-cloud icon (Rainy Day) — not overlapping the cloud.
    WHY max(): anchor the ripples in the lower band (≥60% down) regardless of the cloud's size, so a
    big cloud can't push the rain into them — the previous fixed cy-offset still collided."""
    draw = ImageDraw.Draw(layer)
    H    = layer.size[1]
    ry   = max(cy + size * 0.88, H * 0.60)              # well below the cloud, in the lower band
    a0   = max(40, int(alpha * 0.55))
    lw   = max(2, int(size * 0.012))
    for k, rr in enumerate((size * 0.12, size * 0.20, size * 0.28)):
        draw.ellipse([cx - rr, ry - rr * 0.32, cx + rr, ry + rr * 0.32],
                     outline=(*rgb, max(28, a0 - k * 16)), width=lw)


def _draw_scattered_notes(layer, W, H, avoid, rng, fill, stroke, note_size, target=4):
    """Scatter a few music_note / music_note_2 glyphs at non-overlapping positions, clear of
    `avoid` ((x, y, r) footprints), the title bar and the top-left Meloday+ badge — used to
    dress the Friday Night cover around its hero glyph."""
    sw, sf = stroke or (0, None)
    placed  = list(avoid)
    for _fp in avoid:                            # record the hero footprint(s) for overlap validation
        _rec_footprint(*_fp)
    names   = ["music_note", "music_note_2"]
    margin  = 60
    bar_top = int(H * 0.78)
    k = tries = 0
    while k < target and tries < 240:
        tries += 1
        x = rng.uniform(margin + note_size * 0.5, W - margin - note_size * 0.5)
        y = rng.uniform(150, bar_top - 30 - note_size * 0.5)
        r = note_size * 0.62
        if x - r < 250 and y - r < 150:                  # top-left Meloday+ badge
            continue
        if all((x - px) ** 2 + (y - py) ** 2 > (r + pr) ** 2 for px, py, pr in placed):
            _draw_glyph(layer, names[k % 2], x, y, note_size, fill, tilt=rng.uniform(-18, 18),
                        stroke_width=int(round(note_size * sw)) if sw else 0, stroke_fill=sf)
            placed.append((x, y, r))
            _rec_footprint(x, y, r)
            k += 1


# Real fallen leaves are a MIX of autumnal colours, never one uniform tone.
_AUTUMN_FILLS = [(214, 120, 40), (190, 78, 32), (228, 168, 58), (162, 74, 40), (205, 108, 48), (178, 138, 52)]


def _draw_falling_leaves(layer, W, H, leaf_name, rng, fills, size):
    """Scatter leaf glyphs across the upper canvas in MIXED autumnal colours, non-overlapping —
    some high (still falling), some settling low. Each leaf a different colour from `fills`."""
    placed, margin, bar_top = [], 50, int(H * 0.74)
    target, tries = 12, 0
    while len(placed) < target and tries < 400:
        tries += 1
        s = size * rng.uniform(0.5, 1.1); r = s * 0.42
        x = rng.uniform(margin, W - margin); y = rng.uniform(margin + 60, bar_top - 20)
        if x - r < 250 and y - r < 150:                  # keep clear of the top-left badge
            continue
        if all((x - px) ** 2 + (y - py) ** 2 > (r + pr) ** 2 for px, py, pr in placed):
            _draw_glyph(layer, leaf_name, x, y, s, (*rng.choice(fills), 235), tilt=rng.uniform(-55, 55))
            _rec_footprint(x, y, r)
            placed.append((x, y, r))


def _draw_icon_overlay(img, key, color_top, color_bottom, rng):
    """Composite a profile-specific Material Symbol glyph onto img (RGBA). Returns RGBA.

    The glyph is the topmost art below the title text: rendered at a prominent hero size,
    placed off-centre and gently tilted per profile (seeded by the weekly rng) within a
    safe zone clear of the title bar and badge, with a contrast drop-shadow so it reads on
    any background. Some profiles render a composed cluster (anchor+satellites, or falling).
    """
    icon_name = _PROFILE_ICON.get(key)
    if not icon_name or not _PIL_AVAILABLE:
        return img
    if key in _PROFILE_ICON_ROTATE:                      # weekly rotation (e.g. date_night)
        icon_name = rng.choice(_PROFILE_ICON_ROTATE[key])

    W, H = img.size
    base = img.convert("RGBA")

    meta = dict(_ICON_DEFAULT_META)
    meta.update(_ICON_META.get(icon_name, {}))
    meta.update(_ICON_PROFILE_OVERRIDE.get(key, {}))

    # Cluster dispatch — romance hearts / music notes / celebration flares (ring), winter (falling).
    heart_mode = _HEART_MODE.get(key) if key in _HEART_MODE else None
    cluster_mode, cluster_names, cluster_n = None, None, 0
    if heart_mode and heart_mode != "solitary":
        cluster_mode, cluster_names = "ring", ["favorite"]
        cluster_n = {"pair": 2, "trio": 3, "cluster": 5}.get(heart_mode, 3)
    elif icon_name == "music_note":
        cluster_mode, cluster_names, cluster_n = "ring", ["music_note", "music_note_2"], rng.randint(3, 4)
    elif icon_name == "flare":
        cluster_mode, cluster_names, cluster_n = "ring", ["flare"], rng.randint(4, 5)
    elif key == "winter_mix":
        cluster_mode = "falling"
    elif key == "autumn_leaves":
        cluster_mode = "leaves"
    kind = "cluster" if cluster_mode else "single"

    scale = meta["base_scale"] * rng.uniform(0.92, 1.12)

    # Off-centre placement, clamped clear of the bottom title bar (22%) and the badge.
    R       = meta["extent"] * scale
    R_clamp = R * (1.35 if kind == "cluster" else 1.0)   # clusters spread wider (non-overlap rings)
    margin  = 40
    bar_top = int(H * 0.78)
    ax, ay  = meta["anchor"]
    cx = ax * W + rng.uniform(-0.12 * W, 0.12 * W)
    cy = ay * H + rng.uniform(-0.06 * H, 0.06 * H)
    cx = max(margin + R_clamp, min(W - margin - R_clamp, cx))
    cy = max(margin + R_clamp, min(bar_top - 18 - R_clamp, cy))
    if cy - R_clamp < 100 and cx - R_clamp < 240:        # avoid the top-left Meloday+ badge
        cx = min(W - margin - R_clamp, 240 + R_clamp)
    cx, cy = int(round(cx)), int(round(cy))
    bg_lum = _bg_luminance(base, cx, cy, R)

    # Glyph colour: the icon's own mood colour where defined, else an in-hue tint; guard contrast.
    base_rgb = _PROFILE_ICON_COLOR.get(key) or _ICON_COLOR.get(icon_name)
    if base_rgb is None:
        base_rgb = _icon_dark(color_top) if bg_lum >= 150 else _icon_light(color_top, f=0.50)[:3]
    rgb   = _ensure_icon_contrast(base_rgb, bg_lum)
    alpha = meta.get("alpha", 240)
    fill  = (*rgb, alpha)
    fill2 = None
    if icon_name in _TWO_TONE_GLYPHS:                # darker (or lighter) in-hue detail over the filled body
        fl = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        detail = tuple(int(c * 0.50) for c in rgb) if fl >= 110 \
            else tuple(min(255, int(c + (255 - c) * 0.55)) for c in rgb)
        fill2 = (*detail, alpha)
    size  = max(40, int(round(2 * R)))               # font em size → prominent hero proportion

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if cluster_mode:
        # Outline so overlapping shapes stay legible — contrast vs. the fill.
        fl = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        stroke_rgb = tuple(int(c * 0.28) for c in rgb) if fl >= 110 \
            else tuple(min(255, int(c + (255 - c) * 0.75)) for c in rgb)
        stroke = (0.045, (*stroke_rgb, alpha))
        if cluster_mode == "leaves":                 # autumn: scattered multicolour falling leaves
            _draw_falling_leaves(overlay, W, H, icon_name, rng, _AUTUMN_FILLS, size * 0.5)
        elif cluster_mode == "falling":
            _draw_falling_cluster(overlay, "cloud", "snowflake", cx, cy, size * 0.46, rng, fill, stroke=stroke)
        elif icon_name == "flare":                   # spread the sparks so none overlap
            _draw_glyph_cluster(overlay, cluster_names, cx, cy, anchor_size=size * 0.44,
                                rng=rng, fill=fill, n=cluster_n, stroke=stroke,
                                ring_mult=0.98, sat_lo=0.52, sat_hi=0.66)
        else:
            _draw_glyph_cluster(overlay, cluster_names, cx, cy, anchor_size=size * 0.60,
                                rng=rng, fill=fill, n=cluster_n, stroke=stroke)
    else:
        angle = meta.get("tilt_bias", 0) + rng.uniform(-meta["tilt"], meta["tilt"])
        _draw_glyph(overlay, icon_name, cx, cy, size, fill, tilt=angle, flip=meta.get("flip", False), fill2=fill2)
        if key == "rainy_day":                       # little ripples landing below the cloud
            _draw_ripples_below(overlay, cx, cy, size, rgb, alpha)
        elif key == "friday_night":                  # scatter non-overlapping music notes around it
            _draw_scattered_notes(overlay, W, H, [(cx, cy, R)], rng, fill, stroke=None,
                                  note_size=size * 0.32)

    # The glyph is the topmost art (below the title text); _place_icon adds the drop-shadow.
    return _place_icon(base, overlay, cx, cy, scale=1.0, angle=0.0,
                       shadow=meta.get("shadow", True), bg_lum=bg_lum)


def _brighten_accent(rgb, min_v=0.93, max_s=0.78):
    """Lift an accent colour to a luminance floor so the accent title word reads on the cover's darkened
    bottom — WHY: the geo-radio "Now" accent is the gradient's top colour, and a mid-blue (Scotland/UK) on a
    blue background was low-contrast. Keeps the identity hue; just brighter and a touch less saturated."""
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    r, g, b = colorsys.hsv_to_rgb(h, min(s, max_s), max(v, min_v))
    return (round(r * 255), round(g * 255), round(b * 255))


def _apply_cover_text(img, title, subtitle=None, accent_color=None, text_style="default"):
    """
    Composite title + optional subtitle + Meloday+ badge onto img (RGBA).

    text_style="default":
      Meloday+ pill badge top-left. Title near bottom, split into lines:
        2-word title  → word-per-line; second line in accent_color
        "Daily Mix N" → "Daily Mix" label (90px) + large N (180px) in accent_color
        "X … YYYY"    → title line white + year at 220px in accent_color (Spotify Top Songs)
        other         → wrap_text fallback, all white
      Subtitle in light font below.

    text_style="bar":
      Meloday+ pill badge top-left. Solid dark bar across bottom 22% of canvas.
      Left accent stripe (10px) in accent_color. Title text white inside bar (single line).
    """
    W, H = img.size
    shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text_layer   = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_draw  = ImageDraw.Draw(shadow_layer)
    text_draw    = ImageDraw.Draw(text_layer)
    margin = 60

    _accent_rgba = (*(accent_color or (255, 255, 255)), 255)

    # --- Meloday+ pill badge (top-left, both styles) ---
    try:
        badge_font = ImageFont.truetype(FONT_MELODAY_PATH, size=30)
        badge_text = "Meloday+"
        bb   = text_draw.textbbox((0, 0), badge_text, font=badge_font)
        bw   = bb[2] - bb[0] + 28
        bh   = bb[3] - bb[1] + 16
        bx, by = margin, 44
        badge_draw = ImageDraw.Draw(text_layer)
        try:
            badge_draw.rounded_rectangle(
                [(bx, by), (bx + bw, by + bh)],
                radius=bh // 2, fill=(255, 255, 255, 230))
        except AttributeError:
            badge_draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=(255, 255, 255, 230))
        badge_draw.text((bx + 14, by + 8), badge_text, font=badge_font, fill=(20, 20, 20, 255))
    except Exception:
        pass

    # ======================================================== BAR STYLE
    if text_style == "bar":
        # Solid dark bar at bottom with left accent stripe — Spotify Niche/Mood Mix look.
        BAR_H    = int(H * 0.22)
        STRIPE_W = 10
        bar_y    = H - BAR_H
        bar_draw = ImageDraw.Draw(text_layer)
        bar_draw.rectangle([(0, bar_y), (W, H)], fill=(0, 0, 0, 200))
        if accent_color:
            bar_draw.rectangle([(0, bar_y), (STRIPE_W, H)], fill=(*accent_color, 255))

        bar_size = 76
        try:
            font_bar = ImageFont.truetype(FONT_MAIN_PATH, size=bar_size)
        except (IOError, OSError):
            font_bar = ImageFont.load_default()
        text_x = STRIPE_W + 24
        while bar_size > 48:
            try:
                font_bar = ImageFont.truetype(FONT_MAIN_PATH, size=bar_size)
            except (IOError, OSError):
                break
            tb = text_draw.textbbox((0, 0), title, font=font_bar)
            if (tb[2] - tb[0]) <= W - text_x - margin:
                break
            bar_size -= 3

        tb = text_draw.textbbox((0, 0), title, font=font_bar)
        ty = bar_y + (BAR_H - (tb[3] - tb[1])) // 2
        shadow_draw.text((text_x, ty), title, font=font_bar, fill=(0, 0, 0, 120))
        text_draw.text((text_x, ty),   title, font=font_bar, fill=(255, 255, 255, 255))
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=20))
        result = Image.alpha_composite(img, shadow_layer)
        return Image.alpha_composite(result, text_layer)

    # ======================================================== DEFAULT STYLE
    def _load_font(size):
        try:
            return ImageFont.truetype(FONT_MAIN_PATH, size=size)
        except (IOError, OSError):
            return ImageFont.load_default()

    font_title = _load_font(108)
    font_year  = _load_font(220)

    # --- Auto-size subtitle ---
    font_sub = None
    if subtitle:
        sub_size = 54
        try:
            font_sub = ImageFont.truetype(FONT_LIGHT_PATH, size=sub_size)
            while sub_size > 38:
                sb = text_draw.textbbox((0, 0), subtitle, font=font_sub)
                if (sb[2] - sb[0]) <= W - margin * 2:
                    break
                sub_size -= 3
                font_sub = ImageFont.truetype(FONT_LIGHT_PATH, size=sub_size)
        except (IOError, OSError):
            font_sub = ImageFont.load_default()

    # --- Smart title segmentation — each segment: (text, color_rgba, font) ---
    # "Daily Mix N" intentionally falls through to the wrap_text path so the full
    # title renders as one large white line — matching the Spotify Daily Mix layout
    # where the album art collage and tint carry the visual identity, not the number.
    m_year = re.match(r'^(.+?)\s+(\d{4})$', title.strip())
    words  = title.split()

    if m_year:
        # "Top Songs 2024" → white label + giant accent year (Spotify "Your Top Songs" style)
        segments = [
            (m_year.group(1), (255, 255, 255, 255), font_title),
            (m_year.group(2), _accent_rgba,          font_year),
        ]
    elif len(words) == 2:
        # Two-word title: word-per-line; second word in accent color
        segments = [
            (words[0], (255, 255, 255, 255), font_title),
            (words[1], _accent_rgba,          font_title),
        ]
    else:
        # Fallback: standard text wrap, all white
        lines = wrap_text(title, font_title, text_draw, W - margin * 2)
        segments = [(line, (255, 255, 255, 255), font_title) for line in lines]

    # --- Measure total title block height ---
    LINE_GAP = 12
    total_h = sum(
        (text_draw.textbbox((0, 0), t, font=f)[3] - text_draw.textbbox((0, 0), t, font=f)[1]) + LINE_GAP
        for t, _, f in segments
    )

    SUBTITLE_RESERVE = 78
    y_pos = H - 72 - SUBTITLE_RESERVE - total_h

    # --- Draw title segments ---
    for text, color, font in segments:
        b  = text_draw.textbbox((0, 0), text, font=font)
        lh = b[3] - b[1]
        shadow_draw.text((margin, y_pos), text, font=font, fill=(0, 0, 0, 160))
        text_draw.text((margin, y_pos),   text, font=font, fill=color)
        y_pos += lh + LINE_GAP

    # --- Subtitle ---
    if subtitle and font_sub:
        sub_y = H - 72 - SUBTITLE_RESERVE + 20
        shadow_draw.text((margin, sub_y), subtitle, font=font_sub, fill=(0, 0, 0, 120))
        text_draw.text((margin, sub_y),   subtitle, font=font_sub, fill=(255, 255, 255, 200))

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=30))
    result = Image.alpha_composite(img, shadow_layer)
    return Image.alpha_composite(result, text_layer)


def _extract_dominant_color(image):
    """
    Extract the most vibrant dominant color from a PIL Image.
    Uses a saturation × brightness weighted average to find the prominent hue,
    ignoring near-black and near-white pixels.
    Returns an (R, G, B) tuple.
    """
    small   = image.resize((30, 30)).convert("RGB")
    pixels  = list(small.getdata())
    wr = wg = wb = total = 0.0
    for r, g, b in pixels:
        _, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        if v < 0.20 or v > 0.95:     # skip near-black and near-white
            continue
        w = s * v                      # weight: favour saturated, bright colours
        wr += r * w
        wg += g * w
        wb += b * w
        total += w
    if total < 0.01:
        return (80, 100, 200)          # fallback if no saturated pixels found
    return (int(wr / total), int(wg / total), int(wb / total))


def _generate_extras_cover(key, title, subtitle=None):
    """
    Styled cover for non-Daily-Mix playlists.
    Background style per key (geometric / circles / radial).
    Text style: "bar" for mood profiles (Spotify Niche Mix bottom-bar look),
                "default" for all other extras (word-per-line split + accent last line).
    Vignette + Meloday+ badge applied to all styles.
    Returns saved .webp path or None on failure.
    """
    if not _PIL_AVAILABLE:
        xlog("[WARN] PIL not available — cover generation skipped.")
        return None

    # Year-specific Top Songs covers: key = "top_songs_YYYY"
    if key.startswith("top_songs_") and len(key) == 14:
        try:
            year     = int(key[-4:])
            # Pick from the 20 curated, mutually-distinct covers by `year % 20` so each year is a
            # bold, stable, never-colliding "year in review" look (no weekly shift like mood mixes).
            (color_top, color_bottom), bg_style, bg_v = _TOP_SONGS_COVERS[year % len(_TOP_SONGS_COVERS)]
            _rng                     = random.Random(year)   # stable per-year geometry/jitter
            _jitter_range            = 10                     # subtle — keep the curated palette intact
        except (ValueError, IndexError):
            color_top, color_bottom = _EXTRAS_COVER_COLORS.get("top_songs", ((200, 160, 30), (140, 90, 10)))
            bg_style, bg_v = "geometric", 0
            _iso = datetime.now().isocalendar()
            _rng = random.Random(hash((key, int(_iso[0]), int(_iso[1]))))
            _jitter_range = 18
    else:
        color_top, color_bottom = _EXTRAS_COVER_COLORS.get(key, ((50, 65, 110), (20, 30, 65)))
        style_entry = _COVER_BG_STYLES.get(key, ("geometric", 0))
        if isinstance(style_entry, tuple):
            bg_style, bg_v = style_entry
        else:
            bg_style, bg_v = style_entry, 0
        # Weekly colour jitter — seeded by (key, ISO year, ISO week) so the cover is
        # stable within a week and shifts naturally on Monday with profile reselection.
        _iso = datetime.now().isocalendar()
        _rng = random.Random(hash((key, int(_iso[0]), int(_iso[1]))))
        _jitter_range = 18

    def _jitter(c):
        return tuple(max(0, min(255, ch + _rng.randint(-_jitter_range, _jitter_range))) for ch in c)
    color_top    = _jitter(color_top)
    color_bottom = _jitter(color_bottom)

    # Geo-radio covers ("X Now") use the word-per-line "default" title (accent-coloured last word, no bottom
    # bar) — WHY: chosen look for the radio family; the mood "bar" style is kept for all other mood mixes.
    text_style = "default" if key in _GEO_RADIO_PROFILES else ("bar" if key in _MOOD_PROFILE_KEYS else "default")
    # Geo-radio uses an accent-coloured last word ("Now"); brighten it so a mid-blue identity colour still
    # reads on the darkened bottom (Scotland/UK were blue-on-blue). Other covers keep the raw accent.
    accent = _brighten_accent(color_top) if key in _GEO_RADIO_PROFILES else color_top
    img    = _render_bg(bg_style, color_top, color_bottom, bg_v, _rng)
    img    = _add_bottom_vignette(img)
    img    = _draw_icon_overlay(img, key, color_top, color_bottom, _rng)
    result = _apply_cover_text(img, title, subtitle, accent_color=accent, text_style=text_style)
    out_path = os.path.join(COVER_IMAGE_DIR, f"extras_{key}.webp")
    try:
        result.convert("RGB").save(out_path)
        return out_path
    except Exception as e:
        xlog(f"[WARN] Cover save failed for '{key}': {e}")
        return None


def _generate_daily_mix_cover(plex, tracks, mix_key, title, subtitle=None):
    """
    Daily Mix cover: 2×2 album-art collage with a colour-gradient tint, then text overlay.
    Fetches album thumbnails from Plex for the first 4 distinct albums in the mix.
    Falls back to a plain gradient if fewer than 2 thumbnails are available.
    """
    if not _PIL_AVAILABLE:
        return None

    W, H = 1000, 1000
    half = W // 2
    color_top, color_bottom = _EXTRAS_COVER_COLORS.get(mix_key, ((50, 65, 110), (20, 30, 65)))

    # Collect up to 4 distinct album thumbnails
    seen_albums = set()
    thumb_images = []
    for t in tracks:
        album_rk = str(getattr(t, "parentRatingKey", "") or "")
        if album_rk in seen_albums:
            continue
        seen_albums.add(album_rk)
        thumb_url = getattr(t, "thumb", None) or getattr(t, "parentThumb", None)
        if not thumb_url:
            continue
        try:
            url = plex.url(thumb_url, includeToken=True)
            resp = plex._session.get(url, timeout=8)
            resp.raise_for_status()
            thumb_images.append(Image.open(io.BytesIO(resp.content)).convert("RGBA"))
        except Exception:
            pass
        if len(thumb_images) >= 4:
            break

    if len(thumb_images) < 2:
        return _generate_extras_cover(mix_key, title, subtitle)

    # Pad to exactly 4 by repeating the last image
    while len(thumb_images) < 4:
        thumb_images.append(thumb_images[-1].copy())

    # Build 2×2 collage
    collage = Image.new("RGBA", (W, H))
    for thumb, (px, py) in zip(thumb_images, [(0, 0), (half, 0), (0, half), (half, half)]):
        collage.paste(thumb.resize((half, half), Image.LANCZOS), (px, py))

    # Extract dominant colour from the collage (Spotify-style dynamic tinting)
    dominant = _extract_dominant_color(collage.convert("RGB"))
    r, g, b  = dominant
    # Top-left: lighter, more vibrant; bottom-right: darker
    tint_top    = (min(255, int(r * 1.25)), min(255, int(g * 1.25)), min(255, int(b * 1.25)), 145)
    tint_bottom = (int(r * 0.55), int(g * 0.55), int(b * 0.55), 165)
    tint = _make_gradient_image(W, H, tint_top, tint_bottom, diagonal=True)
    collage = Image.alpha_composite(collage, tint)
    collage = _add_bottom_vignette(collage)

    accent_rgb = tint_top[:3]   # dominant-colour-derived accent for the mix number
    result = _apply_cover_text(collage, title, subtitle, accent_color=accent_rgb)
    out_path = os.path.join(COVER_IMAGE_DIR, f"extras_{mix_key}.webp")
    try:
        result.convert("RGB").save(out_path)
        return out_path
    except Exception as e:
        xlog(f"[WARN] Cover save failed for '{mix_key}': {e}")
        return None


# ===========================================================================
# Playlist Builders
# ===========================================================================

# --- 1. On Repeat ---

# skipCount deprioritisation for the personal play-history builders (On Repeat / Repeat Rewind / Top Songs /
# All-Time Favourites / Rediscovery). Plex exposes a per-track skipCount; tracks you keep skipping shouldn't
# dominate "your favourites". PROD-live only — dev has no live Plex, so getattr(...) → 0 → a graceful no-op
# (mirrors the existing getattr(s, "lastViewedAt", None) pattern). _skip_factor multiplies a builder's positive
# rank score (≤1, scale-free so it composes with plays/viewCount of any magnitude); _SKIP_HEAVY flags tracks to
# demote where the sort key is a time, not a score (Rediscovery).
_SKIP_K     = 0.15   # each skip shrinks the rank score; capped
_SKIP_CAP   = 12
_SKIP_HEAVY = 5      # >= this many skips → "you actively dislike this" (Rediscovery demotes to the back)

def _skip_factor(track):
    """Multiplicative rank deprioritisation by Plex skipCount: 1.0 (no skips / dev / missing attr) down toward
    a floor as skips rise. Scale-free, so it composes with any builder's positive ranking score."""
    s = min(int(getattr(track, "skipCount", 0) or 0), _SKIP_CAP)
    return 1.0 / (1.0 + s * _SKIP_K)

# ── build_on_repeat → "On Repeat" ─────────────────────────────────────────────────────────
# Theme:    What you're hammering right now — genuine current obsession.
# Sound:    N/A (personalisation, not acoustic).
# Era/Geo:  Any era · your library.
# Music:    Songs played ≥2× in the last 30 days, your top-rated copies.
# Criteria: 30-day window (never expands) · ≥2 plays · drop userRating≤4 · ≤3/artist · recency×rating score · no listener floor.
# Flow:     Score-ranked: recency × rating, top first.
# Enhance:  skipCount deprioritisation (applied).
def build_on_repeat(plex, history_entries, excluded_album_keys, target=30):
    """
    Tracks you can't stop playing right now. Fixed 30-day window — Spotify never
    expands the window to pad the count; fewer tracks is the correct result for
    a light-listening week. Only tracks played at least twice qualify (genuine
    repetition, not a casual single listen). Artist cap of 3 — Spotify allows
    multiple tracks from the same artist when they genuinely dominate.
    """
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=30)
    artist_limit = 3

    # Count plays per track first, then score only those with >= 2 plays
    play_counts_30d = Counter()
    for e in history_entries:
        if not e.viewedAt or e.viewedAt < cutoff:
            continue
        pk = str(getattr(e, "parentRatingKey", "") or "")
        if pk in excluded_album_keys:
            continue
        ur = getattr(e, "userRating", None)
        if ur is not None and ur <= 4:
            continue
        play_counts_30d[str(e.ratingKey)] += 1

    scores = defaultdict(float)
    for e in history_entries:
        if not e.viewedAt or e.viewedAt < cutoff:
            continue
        rk = str(e.ratingKey)
        if play_counts_30d[rk] < 2:  # must have been played at least twice
            continue
        days_ago = (now - e.viewedAt).total_seconds() / 86400
        ur = getattr(e, "userRating", None)
        scores[rk] += (1.0 + (1.0 - days_ago / 30.0)) * _rating_multiplier(ur)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    track_map = resolve_tracks_by_keys(plex, [rk for rk, _ in ranked[:max(400, target * 10)]])
    ranked.sort(key=lambda x: x[1] * _skip_factor(track_map.get(x[0])), reverse=True)  # demote tracks you skip

    artist_count = Counter()
    seen_songs   = set()
    result = []
    for rk, _ in ranked:
        t = track_map.get(rk)
        if not t or is_low_rated(t):
            continue
        sk = _song_key(t)
        if sk in seen_songs:
            continue
        ak = _artist_key(t)
        if artist_count[ak] >= artist_limit:
            continue
        seen_songs.add(sk)
        artist_count[ak] += 1
        result.append(t)
        if len(result) >= target:
            break
    return result


# --- 2. Repeat Rewind ---

# ── build_repeat_rewind → "Repeat Rewind" ─────────────────────────────────────────────────
# Theme:    Last month's obsessions you've since gone quiet on.
# Sound:    N/A (personalisation).
# Era/Geo:  Any era · your library.
# Music:    Tracks heavy 4–10 weeks ago, not played in the last 30 days.
# Criteria: plays 30–70d ago · NOT in last 30d · ≥2 (rating-weighted) · ≤2/artist · no listener floor.
# Flow:     Ranked by peak-window play frequency.
# Enhance:  skipCount deprioritisation (applied).
def build_repeat_rewind(plex, history_entries, excluded_album_keys, target=30):
    """
    Tracks that were in heavy rotation 4–10 weeks ago but haven't been played
    in the last 30 days — the previous month's On Repeat.
    """
    now = datetime.now(tz=timezone.utc)
    silence_cutoff = now - timedelta(days=30)
    peak_start    = now - timedelta(days=70)
    artist_limit  = min(2, max(1, int(target * 0.10)))

    recent_keys = {
        str(e.ratingKey)
        for e in history_entries
        if e.viewedAt and e.viewedAt >= silence_cutoff
    }

    peak_counts = Counter()
    for e in history_entries:
        if not e.viewedAt:
            continue
        rk = str(e.ratingKey)
        if rk in recent_keys:
            continue
        if e.viewedAt < peak_start or e.viewedAt >= silence_cutoff:
            continue
        pk = str(getattr(e, "parentRatingKey", "") or "")
        if pk in excluded_album_keys:
            continue
        ur = getattr(e, "userRating", None)
        if ur is not None and ur <= 4:
            continue
        peak_counts[rk] += _rating_multiplier(ur)  # float counts; threshold still applied below

    # Only tracks played at least twice (equiv. score ≥ 2) qualify.
    # Using float counts means a 5-star track played twice scores higher than
    # an unrated track played twice.
    ranked = sorted(
        [(rk, c) for rk, c in peak_counts.items() if c >= 2],
        key=lambda x: x[1], reverse=True,
    )
    track_map = resolve_tracks_by_keys(plex, [rk for rk, _ in ranked[:max(200, target * 5)]])
    ranked.sort(key=lambda x: x[1] * _skip_factor(track_map.get(x[0])), reverse=True)  # demote tracks you skip

    artist_count = Counter()
    seen_songs   = set()
    result = []
    for rk, _ in ranked:
        t = track_map.get(rk)
        if not t or is_low_rated(t):
            continue
        sk = _song_key(t)
        if sk in seen_songs:
            continue
        ak = _artist_key(t)
        if artist_count[ak] >= artist_limit:
            continue
        seen_songs.add(sk)
        artist_count[ak] += 1
        result.append(t)
        if len(result) >= target:
            break
    return result


# --- 3. Release Radar ---

_album_tracks_cache = {}  # album ratingKey -> list[Track]


def _cached_album_tracks(album):
    rk = str(album.ratingKey)
    if rk not in _album_tracks_cache:
        try:
            _album_tracks_cache[rk] = album.tracks()
        except Exception:
            _album_tracks_cache[rk] = []
    return _album_tracks_cache[rk]


def _album_release_date(album):
    """Extract the release date from a Plex album object. Returns datetime.date or None."""
    oaa = getattr(album, "originallyAvailableAt", None)
    if oaa is not None:
        # Check datetime before date — datetime is a subclass of date, so isinstance(oaa, date)
        # would pass for both, returning the datetime object un-converted.
        if isinstance(oaa, datetime):
            return oaa.date()
        if isinstance(oaa, date):
            return oaa
    yr = getattr(album, "year", None)
    if yr:
        return date(yr, 1, 1)
    return None


def _select_album_reps(album, tracks, essentia_cache):
    """
    Pick 1–2 representative tracks from an album.

    Primary:   most globally-played track per Last.fm track.getInfo — real
               play-count data scoped to this specific track, not the artist's
               all-time chart.
    Secondary: most acoustically central track from the remaining tracks —
               complements the popularity pick with a sonically representative
               choice rather than simply taking track 2.

    Falls back to pure acoustic centrality (n=2) when Last.fm is unavailable
    or all tracks return zero plays (brand-new release not yet indexed).
    """
    if not tracks:
        return []

    artist_name = getattr(album, "parentTitle", "") or ""
    is_va = artist_name.strip().casefold() in {"various artists", "various"}

    primary = None

    if LASTFM_API_KEY and artist_name and not is_va:
        scored = [(_lastfm_track_playcount(t.title or "", artist_name), t)
                  for t in tracks]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > 0:
            primary = scored[0][1]

    if primary is None:
        return _pick_representative_tracks(tracks, essentia_cache, n=2)

    # Secondary: most acoustically central track excluding the primary
    remaining = [t for t in tracks if str(t.ratingKey) != str(primary.ratingKey)]
    if not remaining:
        return [primary]
    acoustic = _pick_representative_tracks(remaining, essentia_cache, n=1)
    return [primary] + (acoustic if acoustic else [])


# ── build_release_radar → "Release Radar" ─────────────────────────────────────────────────
# Theme:    New releases that match your taste.
# Sound:    Matches your listening centroid (acoustic + Last.fm-tag affinity).
# Era/Geo:  Released in the last 14–90 days · any origin.
# Music:    1–2 reps per new album (Last.fm playcount + acoustic centrality), known-artists first.
# Criteria: expanding 14–90d window · 0.65 acoustic + 0.35 tag affinity · album-dedup · no listener floor.
# Flow:     Newest release first (known-artists prioritised within each week).
# Enhance:  emb sounds-like blend (applied); release_types album-vs-single lean (roadmap).
def build_release_radar(plex, music, essentia_cache, centroid, excluded_album_keys,
                        history_entries=None):
    """
    Tracks from recently released albums, ranked by recency then affinity.

    Releases from artists in listening history are prioritised within the same
    recency group so you never miss a new release from an artist you actually play.

    Pass 1: one track per album (most popular per Last.fm, or most central).
    Pass 2: if pass 1 yields fewer than TARGET tracks, fill with a second
            representative track from the highest-ranked albums.

    Expands the window in 7-day steps until TARGET unique-artist releases are
    available or RELEASE_RADAR_MAX_DAYS is reached.
    """
    today  = date.today()
    TARGET = RELEASE_RADAR_MIN_TRACKS  # default 50

    # Build known-artist set from listening history — used to prioritise releases
    # from artists the user actually plays over library-only artists.
    known_artists = set()
    if history_entries:
        for e in history_entries:
            name = (getattr(e, "grandparentTitle", "") or
                    getattr(e, "originalTitle", "") or "")
            if name:
                known_artists.add(norm_text(primary_artist(name)))

    try:
        all_albums = music.search(libtype="album", container_size=5000)
    except Exception as e:
        xlog(f"[ERROR] release_radar: album fetch failed: {e}")
        return []

    def _sort_key(item):
        rel, affinity, _, art_k, _ = item
        week     = rel.isocalendar()[1]
        is_known = 1 if art_k in known_artists else 0
        # Sort descending: newest week first; within week, known artists before
        # library-only; within that group, higher acoustic affinity wins.
        return (rel.year, week, is_known, affinity)

    window_days = RELEASE_RADAR_START_DAYS
    album_data  = []

    while window_days <= RELEASE_RADAR_MAX_DAYS:
        cutoff_date = today - timedelta(days=window_days)
        qualifying  = []
        for album in all_albums:
            if str(album.ratingKey) in excluded_album_keys:
                continue
            rel = _album_release_date(album)
            if rel and rel >= cutoff_date:
                qualifying.append((rel, album))

        album_data = []
        for rel, album in qualifying:
            tracks = _cached_album_tracks(album)
            if not tracks:
                continue
            rks          = [str(t.ratingKey) for t in tracks]
            alb_centroid = _album_acoustic_centroid(rks, essentia_cache)
            ac_aff       = 1.0 - _acoustic_distance_to_centroid(alb_centroid, centroid)
            # taste: how well the album's tags (incl. the artist's established Last.fm tags, present
            # even on a brand-new track) match the listening profile — the best-aligned track stands in.
            tag_aff      = max((_tag_overlap_score(essentia_cache.get(rk, {}), centroid) for rk in rks), default=0.0)
            affinity     = 0.65 * ac_aff + 0.35 * tag_aff
            alb_emb = _emb_centroid(rks, essentia_cache)                        # does the album "sound like" the listening profile?
            cen_emb = centroid.get("emb") if isinstance(centroid, dict) else None
            if alb_emb is not None and cen_emb is not None:
                affinity = 0.5 * affinity + 0.5 * (0.5 + 0.5 * _emb_cosine(alb_emb, cen_emb))
            reps         = _select_album_reps(album, tracks, essentia_cache)
            artist_key   = _artist_key(reps[0]) if reps else ""
            album_data.append((rel, affinity, album.ratingKey, artist_key, reps))

        album_data.sort(key=_sort_key, reverse=True)

        # Count total available tracks (first + second per album) to see if we can
        # reach TARGET. Counting only pass-1 tracks misses the case where the window
        # has enough albums but many are singles — second tracks from multi-track
        # albums fill the gap without requiring 50 distinct artists.
        seen     = set()
        p1_count = 0
        p2_count = 0
        for _, _, _, art_k, reps in album_data:
            if art_k and art_k in seen:
                continue
            if art_k:
                seen.add(art_k)
            if reps and not is_low_rated(reps[0]):
                p1_count += 1
            if len(reps) > 1 and not is_low_rated(reps[1]):
                p2_count += 1

        if p1_count + p2_count >= TARGET:
            break
        window_days += RELEASE_RADAR_STEP_DAYS

    # Pass 1: one track per album (most recent release per artist)
    seen_artists  = set()
    result        = []
    second_tracks = []
    for rel, affinity, album_rk, art_k, reps in album_data:
        if art_k and art_k in seen_artists:
            continue
        if art_k:
            seen_artists.add(art_k)
        if not reps:
            continue
        if not is_low_rated(reps[0]):
            result.append((rel, reps[0]))
        if len(reps) > 1 and not is_low_rated(reps[1]):
            second_tracks.append((rel, reps[1]))

    # Pass 2: fill to TARGET with second representative tracks if needed
    for rel, t in second_tracks:
        if len(result) >= TARGET:
            break
        result.append((rel, t))

    result.sort(key=lambda x: x[0], reverse=True)
    return _dedup_filter([t for _, t in result[:TARGET]], essentia_cache)


# --- 4. Discover Weekly ---

# ── build_discover_weekly → "Discover Weekly" ─────────────────────────────────────────────
# Theme:    Your weekly mixtape of fresh, never-played music.
# Sound:    Acoustically adjacent to your taste, in a discovery sweet-spot (not mega-hits, not noise).
# Era/Geo:  Any era · any origin.
# Music:    70% new-artist / 30% familiar, 20% stretch picks; never-played; 1/artist + 1/album.
# Criteria: centroid affinity + discovery sweet-spot · viewCount=0 · interleaved · day-seeded · no listener floor.
# Flow:     Round-robin interleave: safe-new -> familiar -> stretch.
# Enhance:  emb_effnet adjacency blend (applied).
def build_discover_weekly(plex, history_entries, essentia_cache, centroid,
                          excluded_album_keys, target=30):
    """
    Your weekly mixtape of fresh music.

    Mirrors Spotify's Discover Weekly structure:
    - ~70% from artists you've NEVER played (new artist discovery)
    - ~30% from familiar artists, tracks you haven't heard
    - 20% of the new-artist portion are 'stretch' picks — acoustically adjacent
      but less predictable, drawn from rank 300+ rather than the top matches
    - Max 1 track per artist and 1 per album across the whole playlist
    - New-artist and familiar tracks interleaved throughout (not grouped)
    - Exploration slice is seeded by date — stable within the same day
    """
    played_keys = {str(e.ratingKey) for e in history_entries}

    # Artists the user has actually played (from cache entries in their history)
    played_artists = {
        norm_text(primary_artist(essentia_cache.get(rk, {}).get("artist", "") or ""))
        for rk in played_keys
        if essentia_cache.get(rk, {}).get("artist")
    }
    played_artists.discard("")

    # Score all unplayed entries and split by artist familiarity
    new_artist_scored = []
    familiar_artist_scored = []
    for rk, entry in essentia_cache.items():
        if rk in played_keys:
            continue
        aff  = acoustic_affinity(rk, centroid, essentia_cache)   # taste: acoustic + Last.fm/AllMusic tags
        lis  = entry.get("lastfm_listeners") or 0
        pop  = min(1.0, math.log10(lis + 1) / 6.5)
        # discovery sweet spot: reward lesser-known-but-REAL (peak ~a few-k to tens-of-k listeners),
        # penalising BOTH the obvious mega-hits AND near-zero-listener noise.
        disc = max(0.0, 1.0 - abs(pop - 0.6) * 1.4)
        score = 0.8 * aff + 0.2 * disc
        artist = norm_text(primary_artist(entry.get("artist", "") or ""))
        if artist and artist not in played_artists:
            new_artist_scored.append((score, rk))
        else:
            familiar_artist_scored.append((score, rk))

    new_artist_scored.sort(reverse=True)
    familiar_artist_scored.sort(reverse=True)

    n_new      = int(target * 0.70)   # ~21 new-artist tracks
    n_familiar = target - n_new        # ~9 familiar-artist tracks
    n_new_safe    = int(n_new * 0.80)  # top-affinity new artists
    n_new_explore = n_new - n_new_safe  # stretch picks

    pool_size = max(300, target * 8)
    explore_start = pool_size
    explore_end   = pool_size + target * 4

    # Resolve all candidates in one shot
    all_candidate_keys = list(dict.fromkeys(
        [rk for _, rk in new_artist_scored[:explore_end]] +
        [rk for _, rk in familiar_artist_scored[:pool_size]]
    ))
    track_map = resolve_tracks_by_keys(plex, all_candidate_keys)

    artist_count = Counter()
    album_count  = Counter()
    seen_songs   = set()

    def _eligible(rk, t):
        if not t or is_low_rated(t):
            return False
        if (getattr(t, "viewCount", None) or 0) > 0:
            return False
        pk = str(getattr(t, "parentRatingKey", "") or "")
        if pk in excluded_album_keys or album_count[pk] >= 1:
            return False
        if artist_count[_artist_key(t)] >= 1:
            return False
        if _song_key(t) in seen_songs:
            return False
        return True

    def _accept(t):
        pk = str(getattr(t, "parentRatingKey", "") or "")
        artist_count[_artist_key(t)] += 1
        album_count[pk] += 1
        seen_songs.add(_song_key(t))

    # 1. Safe new-artist picks (top affinity)
    new_safe_tracks = []
    for _, rk in new_artist_scored[:pool_size]:
        if len(new_safe_tracks) >= n_new_safe:
            break
        t = track_map.get(rk)
        if not _eligible(rk, t):
            continue
        _accept(t)
        new_safe_tracks.append(t)

    # 2. Exploration picks — random sample from rank pool_size→explore_end
    #    Seeded by date so the same tracks appear if re-run on the same day.
    day_seed = int(datetime.now(tz=timezone.utc).strftime("%Y%m%d"))
    explore_candidates = [
        (rk, track_map[rk])
        for _, rk in new_artist_scored[explore_start:explore_end]
        if rk in track_map and _eligible(rk, track_map[rk])
    ]
    random.Random(day_seed).shuffle(explore_candidates)
    explore_tracks = []
    for rk, t in explore_candidates:
        if len(explore_tracks) >= n_new_explore:
            break
        if not _eligible(rk, t):  # re-check after other picks may have consumed artist/album
            continue
        _accept(t)
        explore_tracks.append(t)

    # 3. Familiar-artist picks
    familiar_tracks = []
    for _, rk in familiar_artist_scored[:pool_size]:
        if len(familiar_tracks) >= n_familiar:
            break
        t = track_map.get(rk)
        if not _eligible(rk, t):
            continue
        _accept(t)
        familiar_tracks.append(t)

    # Interleave: safe new → familiar → stretch, cycling through all three
    # so the playlist flows as one curated mix rather than grouped sections.
    return _round_robin_interleave([new_safe_tracks, familiar_tracks, explore_tracks], target)


# --- 5. Daily Mixes ---

# ── build_daily_mixes → "Daily Mix 1–6" ───────────────────────────────────────────────────
# Theme:    Spotify-style clustered mixes of your library.
# Sound:    Each mix = one k-means acoustic cluster (internally cohesive).
# Era/Geo:  Any era · your library.
# Music:    40% history + 60% library closest to each cluster centroid.
# Criteria: k-means++ on ~10 acoustic dims (numpy) · centroid-ranked · per-cluster proportional · no listener floor.
# Flow:     Centroid-distance ranked within each k-means cluster (closest first).
# Enhance:  emb_effnet k-means clusters (applied).
def build_daily_mixes(plex, history_entries, essentia_cache, excluded_album_keys,
                      n_mixes=6, mix_size=50):
    """
    N Daily Mixes built via k-means clustering on audio embeddings (acoustic 4-d fallback). Requires numpy.
    Each mix: 40% history tracks from the cluster, 60% library tracks closest to centroid.
    Returns list of (playlist_name, description, tracks) tuples.
    """
    if not _NUMPY_AVAILABLE:
        xlog("[WARN] daily_mixes: numpy not available — skipping.")
        return []

    play_counts = Counter(str(e.ratingKey) for e in history_entries)

    # Build normalised feature matrix for all acoustically-complete cache entries
    cache_rks = []
    feature_rows = []
    for rk, entry in essentia_cache.items():
        bpm   = entry.get("bpm")
        energy = entry.get("energy")
        dance  = entry.get("danceability")
        bright = entry.get("brightness")
        if any(v is None for v in (bpm, energy, dance, bright)):
            continue
        cache_rks.append(rk)
        feature_rows.append([
            bpm / 200.0,
            (energy + 23.0) / 23.0,
            float(dance),
            float(bright),
        ])

    if len(cache_rks) < n_mixes:
        xlog(f"[WARN] daily_mixes: only {len(cache_rks)} acoustically-complete tracks — need at least {n_mixes}.")
        return []

    X = np.array(feature_rows, dtype=np.float32)

    # Cluster on audio EMBEDDINGS ("sounds-like") where available, assigning the rest by acoustic nearest-
    # centroid; fall back entirely to acoustic 4-d clustering when embeddings are too sparse. WHY: richer,
    # more coherent clusters than bpm/energy/dance/brightness alone (audit Tier-4). centroids stay in
    # acoustic-4d space so the downstream history/library/distance code is unchanged.
    emb_idx, emb_mat = [], []
    for _i, _rk in enumerate(cache_rks):
        _v = _track_emb(essentia_cache.get(_rk, {}))
        if _v is not None:
            emb_idx.append(_i); emb_mat.append(_v)
    if len(emb_idx) >= n_mixes * 50:
        _, emb_labels = _kmeans_fit(np.array(emb_mat, dtype=np.float32), k=n_mixes)
        labels = np.full(len(cache_rks), -1, dtype=int)
        for _j, _i in enumerate(emb_idx):
            labels[_i] = int(emb_labels[_j])
        centroids = np.zeros((n_mixes, X.shape[1]), dtype=np.float32)   # acoustic-4d mean of each cluster's embedded members
        for _c in range(n_mixes):
            _m = labels == _c
            centroids[_c] = X[_m].mean(axis=0) if _m.any() else X.mean(axis=0)
        _miss = np.where(labels == -1)[0]                              # no-embedding tracks -> nearest cluster acoustically
        if len(_miss):
            _d = np.stack([np.linalg.norm(X[_miss] - centroids[_c], axis=1) for _c in range(n_mixes)], axis=1)
            labels[_miss] = np.argmin(_d, axis=1)
        xlog(f"[INFO] daily_mixes: embedding-clustered ({len(emb_idx)}/{len(cache_rks)} embedded)")
    else:
        centroids, labels = _kmeans_fit(X, k=n_mixes)
        xlog(f"[INFO] daily_mixes: acoustic-clustered (only {len(emb_idx)} embedded — too sparse for embeddings)")

    # Build lookup: rk -> (array_index, cluster_id)
    rk_to_idx = {rk: i for i, rk in enumerate(cache_rks)}

    # Sort cluster IDs by history volume descending → Mix 1 = most-played cluster
    cluster_history_vol = Counter()
    for rk, count in play_counts.items():
        idx = rk_to_idx.get(rk)
        if idx is not None:
            cluster_history_vol[int(labels[idx])] += count

    sorted_cluster_ids = sorted(range(n_mixes), key=lambda c: cluster_history_vol[c], reverse=True)

    mixes = []
    for mix_num, cluster_id in enumerate(sorted_cluster_ids, start=1):
        centroid_vec = centroids[cluster_id]

        # Convert normalised centroid back to original units for distance functions
        cluster_centroid = {
            "bpm":          float(centroid_vec[0] * 200.0),
            "energy":       float(centroid_vec[1] * 23.0 - 23.0),
            "danceability": float(centroid_vec[2]),
            "brightness":   float(centroid_vec[3]),
        }

        # Cluster membership
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        cluster_rks = [cache_rks[i] for i in cluster_indices]

        # Dominant styles from history tracks in this cluster (for description)
        style_counts = Counter()
        for rk in cluster_rks:
            if rk in play_counts:
                for s in (essentia_cache[rk].get("styles") or []):
                    style_counts[s] += play_counts[rk]
        top_styles = [s for s, _ in style_counts.most_common(3)]
        description = (" · ".join(top_styles) + " · Updated daily") if top_styles else "Updated daily"

        # History candidates: most-played first
        history_rks = sorted(
            [rk for rk in cluster_rks if rk in play_counts],
            key=lambda rk: play_counts[rk], reverse=True,
        )

        # Library candidates: closest to cluster centroid (vectorised)
        dists = np.linalg.norm(X[cluster_indices] - centroid_vec, axis=1)
        order = np.argsort(dists)
        library_rks = [cache_rks[cluster_indices[i]] for i in order
                       if cache_rks[cluster_indices[i]] not in play_counts
                       or play_counts[cache_rks[cluster_indices[i]]] <= 1]

        n_history = int(mix_size * 0.4)
        n_library = mix_size - n_history

        candidate_rks = list(dict.fromkeys(history_rks[:n_history * 3] + library_rks[:n_library * 3]))
        track_map = resolve_tracks_by_keys(plex, candidate_rks)

        artist_limit = max(2, int(mix_size * ARTIST_RATIO))
        artist_count = Counter()

        history_tracks = []
        for rk in history_rks:
            if len(history_tracks) >= n_history:
                break
            t = track_map.get(rk)
            if not t or is_low_rated(t):
                continue
            if str(getattr(t, "parentRatingKey", "")) in excluded_album_keys:
                continue
            ak = _artist_key(t)
            if artist_count[ak] >= artist_limit:
                continue
            artist_count[ak] += 1
            history_tracks.append(t)

        library_tracks = []
        for rk in library_rks:
            if len(library_tracks) >= n_library:
                break
            t = track_map.get(rk)
            if not t or is_low_rated(t):
                continue
            if str(getattr(t, "parentRatingKey", "")) in excluded_album_keys:
                continue
            ak = _artist_key(t)
            if artist_count[ak] >= artist_limit:
                continue
            artist_count[ak] += 1
            library_tracks.append(t)

        # Sort combined tracks by distance to cluster centroid — creates acoustic
        # flow within each mix rather than alternating history/library blocks.
        # dedup to one canonical copy per song (the same track can be in BOTH pools — no dedup before)
        combined = _dedup_canonical(history_tracks + library_tracks, essentia_cache)
        combined.sort(key=lambda t: (
            float(np.linalg.norm(X[rk_to_idx[str(t.ratingKey)]] - centroid_vec))
            if str(t.ratingKey) in rk_to_idx else 1.0
        ) - _rating_dist_bonus(getattr(t, "userRating", None)))
        tracks = combined[:mix_size]
        styles_subtitle = " · ".join(top_styles) if top_styles else ""
        mixes.append((f"Daily Mix {mix_num} • Meloday+", description, tracks, styles_subtitle))

    return mixes


# --- 6. Rediscovery Mix ---

# ── build_rediscovery → "Rediscovery" ─────────────────────────────────────────────────────
# Theme:    Tracks you loved but haven't touched in 6–24 months.
# Sound:    N/A (personalisation).
# Era/Geo:  Any era · your library.
# Music:    Rating≥7★ OR ≥3 plays, last played 6–24 months ago, longest-neglected first.
# Criteria: last-play 6–24mo · loved (≥7★ or ≥3 plays) · ≤cap/artist · no listener floor.
# Flow:     Longest-neglected first (oldest last-play).
# Enhance:  skipCount deprioritisation (applied).
def build_rediscovery(plex, history_entries, excluded_album_keys, target=40):
    """
    Tracks you loved (rated ≥ 7 or played ≥ 3 times) but haven't touched
    in 6–24 months. Sorted with the longest-neglected first.
    """
    now = datetime.now(tz=timezone.utc)
    silence_floor   = now - timedelta(days=6 * 30)
    silence_ceiling = now - timedelta(days=24 * 30)
    artist_limit    = max(1, int(target * 0.10))

    last_played = {}
    all_time_plays = Counter()
    for e in history_entries:
        if not e.viewedAt:
            continue
        rk = str(e.ratingKey)
        pk = str(getattr(e, "parentRatingKey", "") or "")
        if pk in excluded_album_keys:
            continue
        all_time_plays[rk] += 1
        if rk not in last_played or e.viewedAt > last_played[rk]:
            last_played[rk] = e.viewedAt

    candidates = sorted(
        [(rk, lp, all_time_plays[rk])
         for rk, lp in last_played.items()
         if silence_ceiling <= lp < silence_floor],
        key=lambda x: x[2], reverse=True,  # most-played first — most likely to qualify
    )[:target * 8]

    track_map = resolve_tracks_by_keys(plex, [rk for rk, _, _ in candidates])

    eligible = []
    for rk, lp, plays in candidates:
        t = track_map.get(rk)
        if not t or is_low_rated(t):
            continue
        ur = getattr(t, "userRating", None)
        if not ((ur is not None and ur >= 7) or plays >= 3):
            continue
        eligible.append((lp, t))

    # longest-neglected first, but push tracks you actively skip (>= _SKIP_HEAVY) to the back so they don't
    # resurface (skipCount is prod-live; 0 on dev → all in the first bucket → unchanged ordering).
    eligible.sort(key=lambda x: ((getattr(x[1], "skipCount", 0) or 0) >= _SKIP_HEAVY, x[0]))

    artist_count = Counter()
    seen_songs   = set()
    result = []
    for _, t in eligible:
        sk = _song_key(t)
        if sk in seen_songs:
            continue
        ak = _artist_key(t)
        if artist_count[ak] >= artist_limit:
            continue
        seen_songs.add(sk)
        artist_count[ak] += 1
        result.append(t)
        if len(result) >= target:
            break
    return result


# --- 7. Time Capsule ---

# ── build_time_capsule → "Time Capsule" ───────────────────────────────────────────────────
# Theme:    A snapshot of your standout earlier PLAY-years (not release years).
# Sound:    N/A (personalisation).
# Era/Geo:  Your top play-years (≥2 apart, before the last 3) · your library.
# Music:    The tracks you played most in those years, now gone quiet.
# Criteria: top-3 play-years via Plex API · gone-quiet 90d · interleaved · canonical-dedup · no listener floor.
# Flow:     Round-robin interleave across your standout play-years.
# Enhance:  —
def build_time_capsule(plex, music, history_entries, essentia_cache, excluded_album_keys, target=30):
    """Nostalgia mix from your STANDOUT EARLIER YEARS — the tracks you actually PLAYED during those years
    (not what was released then), that you've since gone quiet on. The years + per-year play counts come from
    the Plex API (history windows via `fetch_history_window`); `history_entries` is used only for the recent
    gone-quiet filter. Interleaved across years, canonical-deduped, artist-capped."""
    _RECENT_GAP, _ERAS, _SPACING = 3, 3, 2   # skip the recent (non-nostalgic) years; 3 years, ≥2 apart
    now = datetime.now(tz=timezone.utc)
    silence_cutoff = now - timedelta(days=90)

    # Tracks heard in the last 90 days → excluded (a time capsule is what you've drifted FROM).
    recently = {str(e.ratingKey) for e in history_entries
                if e.viewedAt and e.viewedAt >= silence_cutoff}

    # Standout EARLIER play-years: count each candidate year cheaply, take the top few spaced apart.
    counts = []
    for yr in range(now.year - _RECENT_GAP, 2004, -1):
        n = _year_play_count(music, yr)
        if n >= target:
            counts.append((n, yr))
    counts.sort(reverse=True)
    peak_years = []
    for _, yr in counts:
        if all(abs(yr - py) >= _SPACING for py in peak_years):
            peak_years.append(yr)
        if len(peak_years) >= _ERAS:
            break
    if not peak_years:
        xlog("[WARN] time_capsule: no qualifying earlier years with play history.")
        return [], None
    era_label = " · ".join(str(y) for y in sorted(peak_years))
    xlog(f"[INFO] time_capsule: standout earlier years {sorted(peak_years)}")

    # Per-year pool: the tracks you PLAYED most that year, minus anything you've played recently.
    era_pools = []
    for yr in sorted(peak_years, reverse=True):
        plays = Counter()
        for e in fetch_history_window(music, datetime(yr, 1, 1, tzinfo=timezone.utc),
                                      datetime(yr + 1, 1, 1, tzinfo=timezone.utc)):
            rk = str(e.ratingKey)
            if rk not in recently:
                plays[rk] += 1
        era_pools.append([rk for rk, _ in plays.most_common(target)])

    if not any(era_pools):
        xlog("[WARN] time_capsule: no qualifying tracks in those years.")
        return [], era_label

    all_candidate_keys = list(dict.fromkeys(rk for pool in era_pools for rk in pool))
    track_map = resolve_tracks_by_keys(plex, all_candidate_keys)

    artist_limit = max(1, int(target * 0.10))
    artist_count = Counter()

    filtered_pools = []
    for pool in era_pools:
        filtered = []
        for rk in pool:
            t = track_map.get(rk)
            if not t or is_low_rated(t):
                continue
            if str(getattr(t, "parentRatingKey", "")) in excluded_album_keys:
                continue
            filtered.append(t)
        filtered_pools.append(filtered)

    filtered_pools = [_dedup_canonical(p, essentia_cache) for p in filtered_pools]  # canonical copy per song
    # Round-robin across eras with artist cap and inline dedup
    result      = []
    seen_songs  = set()
    queues = [list(p) for p in filtered_pools if p]
    while queues and len(result) < target:
        next_queues = []
        for q in queues:
            while q:
                t = q.pop(0)
                sk = _song_key(t)
                if sk in seen_songs:
                    continue
                ak = _artist_key(t)
                if artist_count[ak] < artist_limit:
                    seen_songs.add(sk)
                    artist_count[ak] += 1
                    result.append(t)
                    break  # take one and move to next queue
            if q:
                next_queues.append(q)
        queues = next_queues

    return result, era_label


# Matches a rotating Time Machine title ("June 2019 • Meloday+") for stale-playlist cleanup.
_TIME_MACHINE_TITLE_RE = re.compile(r"^[A-Z][a-z]{2,8} \d{4} • Meloday\+$")


# ── build_time_machine → "“Month YYYY” (rotating)" ────────────────────────────────────────
# Theme:    This time, years ago — what you played around today's date in a past year.
# Sound:    N/A (personalisation).
# Era/Geo:  ±21 days around today in a rotating past year · your library.
# Music:    Your most-played tracks in that window; featured year rotates ~weekly.
# Criteria: ±21d window · gone-quiet 90d · weekly year rotation · canonical-dedup · no listener floor.
# Flow:     Most-played-in-window first.
# Enhance:  —
def build_time_machine(plex, music, history_entries, essentia_cache, excluded_album_keys, target=30):
    """'This time, years ago' — the tracks you were playing around today's date in a single past year
    (e.g. 'June 2019'), rotating the featured year ~weekly. Returns (tracks, 'Month YYYY')."""
    _WINDOW_DAYS, _MIN_TRACKS = 21, 8
    now = datetime.now(tz=timezone.utc)
    silence_cutoff = now - timedelta(days=90)
    recently = {str(e.ratingKey) for e in history_entries
                if e.viewedAt and e.viewedAt >= silence_cutoff}

    def _window(yr):
        try:
            anchor = now.replace(year=yr)
        except ValueError:                      # today is Feb 29 and yr isn't a leap year
            anchor = now.replace(year=yr, day=28)
        return anchor - timedelta(days=_WINDOW_DAYS), anchor + timedelta(days=_WINDOW_DAYS)

    # Eligible past years (enough in-window, gone-quiet plays); rotate weekly through them.
    candidates = []
    for yr in range(now.year - 1, 2004, -1):
        start, end = _window(yr)
        plays = Counter(str(e.ratingKey) for e in fetch_history_window(music, start, end)
                        if str(e.ratingKey) not in recently)
        if len(plays) >= _MIN_TRACKS:
            candidates.append((yr, plays))
        if len(candidates) >= 8:
            break
    if not candidates:
        xlog("[WARN] time_machine: no past year has enough plays around this date.")
        return [], None

    years = [y for y, _ in candidates]
    yr, plays = candidates[(now.isocalendar()[1] + now.year) % len(years)]
    label = f"{now.strftime('%B')} {yr}"
    xlog(f"[INFO] time_machine: {label} ({len(plays)} candidate tracks)")

    pool_rks  = [rk for rk, _ in plays.most_common(target * 3)]
    track_map = resolve_tracks_by_keys(plex, pool_rks)
    tracks = [track_map[rk] for rk in pool_rks
              if rk in track_map and not is_low_rated(track_map[rk])
              and str(getattr(track_map[rk], "parentRatingKey", "")) not in excluded_album_keys]
    tracks = _dedup_canonical(tracks, essentia_cache)
    artist_limit, artist_count, result = max(1, int(target * 0.10)), Counter(), []
    for t in tracks:
        if len(result) >= target:
            break
        ak = _artist_key(t)
        if artist_count[ak] < artist_limit:
            artist_count[ak] += 1
            result.append(t)
    return result, label


# --- 8. Artist Deep Cuts ---

# ── build_deep_cuts → "Deep Cuts" ─────────────────────────────────────────────────────────
# Theme:    Unheard album tracks from the artists you love most.
# Sound:    Ranked to each artist's own well-played centroid.
# Era/Geo:  Any era · your library.
# Music:    ≤2-play tracks from your top-15 artists (last 6mo), rating-bonused.
# Criteria: top-15 artists (6mo) · ≤2 plays · artist-centroid distance + rating bonus · round-robin · no listener floor.
# Flow:     Round-robin interleave across your top-15 artists.
# Enhance:  emb_effnet cohesion to the artist's well-played sound (applied).
def build_deep_cuts(plex, history_entries, essentia_cache, excluded_album_keys,
                    target=40, top_artists_n=15, tracks_per_artist=3):
    """
    Unheard or rarely-played tracks from the top-15 most-listened artists
    in the past 6 months. Ranked per artist by acoustic similarity to that
    artist's own well-played centroid. Round-robin interleaved.
    """
    now = datetime.now(tz=timezone.utc)
    window_cutoff = now - timedelta(days=180)

    # Top artists by recent plays (from history entries)
    artist_recent_plays = Counter()
    for e in history_entries:
        if not e.viewedAt or e.viewedAt < window_cutoff:
            continue
        raw = str(getattr(e, "grandparentTitle", "") or "")
        ak = norm_text(primary_artist(raw))
        if ak:
            artist_recent_plays[ak] += 1

    top_artists = [name for name, _ in artist_recent_plays.most_common(top_artists_n)]
    if not top_artists:
        xlog("[INFO] deep_cuts: no recent artist history.")
        return []

    all_time_plays = Counter(str(e.ratingKey) for e in history_entries)
    top_artist_set = set(top_artists)

    # Use essentia cache to find tracks per artist (avoids a full music.search())
    tracks_by_artist_rks = defaultdict(list)
    for rk, entry in essentia_cache.items():
        ak = norm_text(primary_artist(entry.get("artist", "") or ""))
        if ak in top_artist_set:
            tracks_by_artist_rks[ak].append(rk)

    artist_deep_cuts = {}
    seen_songs = set()  # shared across all artists — prevents same song appearing twice
    for artist_key in top_artists:
        artist_rks = tracks_by_artist_rks.get(artist_key, [])
        if not artist_rks:
            continue

        well_played_rks = [rk for rk in artist_rks if all_time_plays.get(rk, 0) > 2]
        deep_cut_rks    = [rk for rk in artist_rks if all_time_plays.get(rk, 0) <= 2]
        if not deep_cut_rks:
            continue

        artist_centroid = _album_acoustic_centroid(well_played_rks, essentia_cache) if well_played_rks else {}
        artist_emb      = _emb_centroid(well_played_rks, essentia_cache) if well_played_rks else None  # "sounds-like" the artist's well-played cuts

        # Score deep cuts by distance to the artist's well-played centroid (acoustic + embedding "sounds-like").
        # Resolve a larger pool so rating bonus can surface high-rated tracks
        # that sit slightly further from the centroid.
        has_acoustic = any(artist_centroid.get(f) for f in ("bpm", "energy"))
        scored = []
        for rk in deep_cut_rks:
            entry = essentia_cache.get(rk, {})
            score = (1.0 - _acoustic_distance_to_centroid(entry, artist_centroid)
                     if has_acoustic else 0.5)
            if artist_emb is not None:
                v = _track_emb(entry)
                if v is not None:
                    score = 0.5 * score + 0.5 * (0.5 + 0.5 * _emb_cosine(v, artist_emb))
            scored.append((score, rk))
        scored.sort(reverse=True)

        top_rks = [rk for _, rk in scored[:tracks_per_artist * 4]]
        track_map = resolve_tracks_by_keys(plex, top_rks)

        # Re-sort resolved tracks with rating bonus applied
        resolved_scored = []
        for score, rk in scored:
            t = track_map.get(rk)
            if not t:
                continue
            adjusted = score + _rating_dist_bonus(getattr(t, "userRating", None))
            resolved_scored.append((adjusted, t))
        resolved_scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        for _, t in resolved_scored:
            if is_low_rated(t):
                continue
            if str(getattr(t, "parentRatingKey", "")) in excluded_album_keys:
                continue
            sk = _song_key(t)
            if sk in seen_songs:
                continue
            seen_songs.add(sk)
            selected.append(t)
            if len(selected) >= tracks_per_artist:
                break

        if selected:
            artist_deep_cuts[artist_key] = _dedup_canonical(selected, essentia_cache)

    return _round_robin_interleave(
        [artist_deep_cuts[a] for a in top_artists if a in artist_deep_cuts],
        cap=target,
    )


# --- 9. Top Songs by Year ---

# ── build_top_songs → "Top Songs YYYY" ────────────────────────────────────────────────────
# Theme:    Your most-played tracks of each calendar year.
# Sound:    N/A (personalisation).
# Era/Geo:  Per calendar year · your library.
# Music:    That year's most-played, ≤5/artist.
# Criteria: per-year top plays · ≥20 distinct · drop userRating≤4 · past years immutable (+prev-year 60d grace) · no listener floor.
# Flow:     Most-played first (per calendar year).
# Enhance:  skipCount deprioritisation (applied).
def build_top_songs(plex, history_entries, excluded_album_keys,
                    target=100, min_distinct=20, existing_playlists=None):
    """
    Your most-played tracks during each calendar year you have history for.
    Returns list of (year, [tracks]) tuples sorted by year descending.
    Past years whose playlists already exist in Plex are skipped — they don't
    change. The current year is always regenerated.
    """
    current_year = datetime.now(tz=timezone.utc).year
    start_year   = int(_extras.get("top_songs_start_year", current_year - 5))
    existing     = existing_playlists or {}

    # Regenerate the previous year during a 60-day grace period so that plays
    # from the final days of December (after the last run of the year) are captured.
    # After the grace period the previous year is treated as immutable.
    _grace_cutoff = date(current_year, 1, 1) + timedelta(days=60)
    _in_grace     = date.today() < _grace_cutoff
    prev_year     = current_year - 1

    year_plays = defaultdict(Counter)
    for e in history_entries:
        if not e.viewedAt:
            continue
        if e.viewedAt.year < start_year:
            continue
        pk = str(getattr(e, "parentRatingKey", "") or "")
        if pk in excluded_album_keys:
            continue
        ur = getattr(e, "userRating", None)
        if ur is not None and ur <= 4:
            continue
        year_plays[e.viewedAt.year][str(e.ratingKey)] += _rating_multiplier(ur)

    results = []
    for year in sorted(year_plays.keys(), reverse=True):
        # Current year always regenerated; previous year regenerated during the
        # 60-day grace period to capture plays from the final days of December;
        # all other past years skipped once their playlist exists.
        if year != current_year and f"Top Songs {year} • Meloday+" in existing:
            if year == prev_year and _in_grace:
                xlog(f"[INFO] top_songs: regenerating {year} (year-end grace — capturing final days)")
            else:
                xlog(f"[INFO] top_songs: skipping {year} (playlist already exists)")
                continue
        counts = year_plays[year]
        if len(counts) < min_distinct:
            continue
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        candidate_keys = [rk for rk, _ in ranked[:max(400, target * 4)]]
        track_map = resolve_tracks_by_keys(plex, candidate_keys)
        ranked.sort(key=lambda x: x[1] * _skip_factor(track_map.get(x[0])), reverse=True)  # demote tracks you skip

        artist_count = Counter()
        seen_songs   = set()
        tracks = []
        for rk, _ in ranked:
            t = track_map.get(rk)
            if not t or is_low_rated(t):
                continue
            pk = str(getattr(t, "parentRatingKey", "") or "")
            if pk in excluded_album_keys:
                continue
            sk = _song_key(t)
            if sk in seen_songs:
                continue
            ak = _artist_key(t)
            if artist_count[ak] >= 5:
                continue
            seen_songs.add(sk)
            artist_count[ak] += 1
            tracks.append(t)
            if len(tracks) >= target:
                break

        if tracks:
            results.append((year, tracks))

    return results


# --- 10. All-Time Favourites ---

# ── build_all_time_favourites → "All-Time Favourites" ─────────────────────────────────────
# Theme:    Your lifetime most-played tracks.
# Sound:    N/A (personalisation).
# Era/Geo:  Any era · your library.
# Music:    Highest viewCount; your most-played copy of each song.
# Criteria: viewCount:desc · stop at 0-play · inline (not canonical) dedup · no listener floor.
# Flow:     Most-played first (viewCount).
# Enhance:  skipCount deprioritisation (applied).
def build_all_time_favourites(music, excluded_album_keys, target=100):
    """
    All-time most-played tracks using track.viewCount — no history fetch needed.
    viewCount covers the full lifetime of your library, not just a history window.
    Deduplication is inline so we always hit the target count of unique songs
    regardless of how many compilation copies exist for each track.
    """
    try:
        candidates = music.search(
            libtype="track",
            sort="viewCount:desc",
            container_size=max(1000, target * 8),
        )
    except Exception as e:
        xlog(f"[ERROR] all_time_favourites: track fetch failed: {e}")
        return []

    # NB: NO canonical dedup here — this list is view-count-sorted (your plays) and the loop stops at
    # the first 0-play track. Swapping in a studio copy you've never played would have 0 plays and cut
    # the playlist short. The inline dedup below keeps your most-played copy of each song (correct).
    # Re-order by viewCount × skip-factor so heavily-skipped favourites sink (0-play tracks keep factor 1.0 and
    # stay last, so the "stop at vc==0" below still holds); prod-live, a no-op on dev (skipCount → 0).
    candidates = sorted(candidates, key=lambda t: (getattr(t, "viewCount", None) or 0) * _skip_factor(t), reverse=True)
    artist_count = Counter()
    seen_songs   = set()
    result = []
    for t in candidates:
        vc = getattr(t, "viewCount", None) or 0
        if vc == 0:
            break  # server-sorted; no point continuing past zero-play tracks
        if is_low_rated(t):
            continue
        pk = str(getattr(t, "parentRatingKey", "") or "")
        if pk in excluded_album_keys:
            continue
        # Dedup inline so compilation copies don't count towards the target
        sk = _song_key(t)
        if sk in seen_songs:
            continue
        ak = _artist_key(t)
        if artist_count[ak] >= 5:
            continue
        seen_songs.add(sk)
        artist_count[ak] += 1
        result.append(t)
        if len(result) >= target:
            break
    return result


# --- 11. Mood / Activity Mixes ---


# Two mood mixes whose (valence-weighted) centroids are closer than this select overly
# similar tracks — don't surface them at the same time. Re-tuned against the now-calibrated
# centroids (real TF valence/arousal/vocal): tightened to catch only true near-duplicate
# profiles — the city/genre twins (uk_garage≈london_garage, melbourne_techno≈glasgow_underground
# at ~0.00) — while keeping a full ~12/13-category slate. Above this, sonic adjacency is fine.
_SIM_GUARD_DISTANCE = 0.06


def _geo_group(key):
    """City mixes -> their city (glasgow/london/melbourne); everything else -> None. A SECOND diversity
    axis: city mixes keep their genre category AND a city group, so a slate avoids two same-city mixes
    even when their genres differ. The pinned scene showcases (_PROFILE_GEO_GATE) are handled separately
    and carry no city group here. WHY: the user wants city mixes counted on both vibe and geo."""
    if key in _PROFILE_GEO_GATE:
        return None
    for pre, city in (("glasgow_", "glasgow"), ("london_", "london"), ("melbourne_", "melbourne")):
        if key.startswith(pre):
            return city
    return None


def _select_diverse_profiles(scored_profiles, n_active, max_per_category=2, max_per_geo=1):
    """
    Greedy diversity selection.
    `scored_profiles` is sorted (score, profile_key) ascending (lower = better fit).
    Picks n_active profiles ensuring (a) at most max_per_category from the same vibe category,
    (b) at most max_per_geo from the same city (the city-mix geo axis — so a slate isn't two
    Glasgow mixes even when their genres differ), and (c) no two picks whose centroids are within
    _SIM_GUARD_DISTANCE. Deferred candidates backfill any remaining slots — preferring still-distinct
    ones, relaxing the caps only in the final pass — so the slate is never left short.
    """
    cat_count = Counter()
    geo_count = Counter()
    selected  = []
    deferred  = []

    def _too_similar(key):
        # Decade mixes aren't sound-based (hits of the period) — never sim-gate them by centroid, in either
        # direction. The max_per_category cap still limits how many decades share a slate.
        if _PROFILE_CATEGORY.get(key) == "era":
            return False
        return any(
            _acoustic_distance_to_centroid(_MOOD_PROFILES[key], _MOOD_PROFILES[s]) < _SIM_GUARD_DISTANCE
            for s in selected if s in _MOOD_PROFILES and _PROFILE_CATEGORY.get(s) != "era"
        )

    def _take(key):
        selected.append(key)
        cat_count[_PROFILE_CATEGORY.get(key, "other")] += 1
        g = _geo_group(key)
        if g:
            geo_count[g] += 1

    for _, key in scored_profiles:
        if len(selected) >= n_active:
            break
        cat = _PROFILE_CATEGORY.get(key, "other")
        g = _geo_group(key)
        # Hard caps: per vibe category AND per city (city mixes only), plus the near-duplicate guard.
        # With calibrated centroids the similarity guard is trustworthy, so it blocks a near-duplicate
        # twin even across categories; deferred picks backfill the rest.
        if (cat_count[cat] >= max_per_category
                or (g and geo_count[g] >= max_per_geo)
                or (key in _MOOD_PROFILES and _too_similar(key))):
            deferred.append(key)
            continue
        _take(key)

    # Backfill: first with deferred profiles that are still distinct (category cap already met, but keep
    # the geo + similarity guards), then with any — so length always wins over the soft preferences.
    for require_distinct in (True, False):
        for key in deferred:
            if len(selected) >= n_active:
                break
            if key in selected:
                continue
            if require_distinct:
                g = _geo_group(key)
                if (g and geo_count[g] >= max_per_geo) or (key in _MOOD_PROFILES and _too_similar(key)):
                    continue
            _take(key)
        if len(selected) >= n_active:
            break
    return selected[:n_active]


def _external_used_rks(existing_playlists, building_names):
    """Rating keys already used by EXISTING dedup-eligible Meloday+ mixes built by OTHER cron runs,
    so the no-repeat invariant holds across runs (a track in the Morning mix won't be reused by a
    mood mix built this run, and vice versa). EXEMPT: decade/era mixes (category 'era') and the
    separate stats/discovery playlists (On Repeat, Top Songs, …) which aren't in _MOOD_MIX_NAMES.
    Mixes being rebuilt/removed this run (`building_names`) are skipped — their tracks are transient."""
    used = set()
    nk = {v: k for k, v in _MOOD_MIX_NAMES.items()}
    for name, pl in (existing_playlists or {}).items():
        key = nk.get(name)
        if key is None or _PROFILE_CATEGORY.get(key) in ("era", "geo_scene") or name in building_names:
            continue
        try:
            used.update(str(t.ratingKey) for t in pl.items())
        except Exception:
            pass
    return used


# ---------------------------------------------------------------------------
# Surface rotation — per-profile {last-on-slate gslot, last-built ordinal}. Lets the slate (a) guarantee
# coverage via an uncapped overdue bonus (the longest-unseen mix always cycles in → nothing is never
# surfaced) and (b) rebuild a surviving playlist only once a day (sub-daily runs build only NEW entries).
# ---------------------------------------------------------------------------
_ANYTIME_OVERDUE_RATE = 0.10   # per-slot bonus for an unseen profile (uncapped → everyone returns)

def _surface_rotation_path():
    return os.path.join(_BASE_DIR, "assets", "mood_surface_rotation.json")

def _load_surface_rotation():
    """{profile_key: {"s": last-on-slate gslot}} (+ "__slot__": last-rotated gslot). {} if missing/corrupt."""
    try:
        with open(_surface_rotation_path()) as f:
            return json.load(f).get("rotation", {})
    except (OSError, ValueError):
        return {}

def _save_surface_rotation(rotation):
    """Atomic best-effort write (never fatal to a build), mirroring _save_geo_rotation."""
    try:
        os.makedirs(os.path.dirname(_surface_rotation_path()), exist_ok=True)
        tmp = _surface_rotation_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "rotation": rotation}, f)
        os.replace(tmp, _surface_rotation_path())
    except OSError as e:
        xlog(f"[WARN] mood_mixes: could not persist surface rotation: {e}")

def _surface_overdue(last_gslot, cur_gslot):
    """Uncapped overdue bonus: rises with slots since the profile was last on the slate; never-seen gets
    top priority. Mirrors _geo_overdue_bonus — guarantees every rotating mix cycles in over time."""
    if last_gslot is None:
        return _ANYTIME_OVERDUE_RATE * 10_000
    return _ANYTIME_OVERDUE_RATE * max(0, cur_gslot - last_gslot)


def build_mood_mixes(plex, history_entries, essentia_cache, excluded_album_keys,
                     n_active=5, mix_size=50, reselect=False, time_context=False,
                     weather_context=False, existing_playlists=None):
    """
    Operating modes:

    time_context=True  (boundary cron — 5am, noon, 5pm, 9pm, 10pm, 2am, 4am):
        Only adds/removes hard time-of-day playlists (Morning, Dinner, Late Night, Sleep).
        Does NOT touch general, weather, or seasonal mixes.

    weather_context=True  (hourly cron):
        Only adds/removes weather playlists to match the current conditions. Cheap to run
        often so weather mixes track the weather through the day. Does NOT touch the rest.

    default (general — run HOURLY so the slate tracks the schedule):
        Builds the three-tier slate: CONTEXT (every scheduled, in-window profile — uncapped) +
        ANYTIME (a rotating, time-ranked, overdue-covered set of size mood_mix_count) + GEO (>=1 city
        mix per city) + weather + seasonal + the 3 pinned scenes. Slot-stable diffing only builds new
        entries (+ a once-a-day content refresh) and removes rotated-out ones, so hourly runs are cheap.
        (The `reselect` flag is retained for CLI compatibility but is now a no-op — the overdue rotation
        continuously refreshes the anytime tier, so there is no separate weekly "reselection".)

    Returns (mixes, profiles_to_delete) where mixes is a list of
    (playlist_name, profile_key, tracks) tuples.
    """
    name_to_key = {v: k for k, v in _MOOD_MIX_NAMES.items()}
    existing     = existing_playlists or {}

    # Recently-played exclusion for variety (configurable; 0 disables). Tracks heard in the
    # last N days are kept out of every mix this run, so the slate keeps turning over instead
    # of resurfacing the same songs.
    _excl_days = int(_extras.get("mood_mix_exclude_played_days", 3))
    recent_rks = set()
    if _excl_days > 0 and history_entries:
        _cut = datetime.now(tz=timezone.utc) - timedelta(days=_excl_days)
        recent_rks = {str(e.ratingKey) for e in history_entries
                      if e.viewedAt and e.viewedAt >= _cut}

    # Listening-hour affinity (build-time): the share of each track's plays that fall within
    # ~2h of the current local hour, so the rotating mixes lean toward what you actually play now.
    _HOUR_AFFINITY.clear()
    if history_entries:
        _now_h = datetime.now().hour
        _hp = {}
        for e in history_entries:
            if e.viewedAt:
                _hp.setdefault(str(e.ratingKey), []).append(e.viewedAt.astimezone().hour)
        for _rk, _hours in _hp.items():
            _near = sum(1 for h in _hours if min((h - _now_h) % 24, (_now_h - h) % 24) <= 2)
            if _near:
                _HOUR_AFFINITY[_rk] = _near / len(_hours)

    # ------------------------------------------------------------------
    # MODE 1: Time-context — only manage hard time-of-day mixes
    # ------------------------------------------------------------------
    if time_context:
        current_hour = _get_active_hour()
        should_be_active = {
            k for k in _TIME_PROFILES
            if _in_time_window(current_hour, _TIME_BIASED_PROFILES[k])
        }
        currently_active = {
            name_to_key[name]
            for name in existing
            if name in name_to_key and name_to_key[name] in _TIME_PROFILES
        }
        to_add    = should_be_active - currently_active
        to_remove = currently_active - should_be_active

        xlog(f"[INFO] mood_mixes (time): hour={current_hour} "
             f"add={sorted(to_add)} remove={sorted(to_remove)}")

        building = {_MOOD_MIX_NAMES[k] for k in (to_add | to_remove)}
        seen_rks = set(recent_rks) | _external_used_rks(existing, building)   # cross-run dedup
        mixes = []
        for profile_key in sorted(to_add):
            tracks = _build_mix_tracks(
                profile_key, essentia_cache, history_entries,
                excluded_album_keys, mix_size, plex, hard_exclude_rks=seen_rks)
            seen_rks.update(str(t.ratingKey) for t in tracks)
            mixes.append((_MOOD_MIX_NAMES[profile_key], profile_key, tracks))

        return mixes, list(to_remove)

    # ------------------------------------------------------------------
    # MODE 0: Weather-context — only manage weather mixes (hourly cron)
    # ------------------------------------------------------------------
    if weather_context:
        loc     = _extras.get("weather_location")
        weather = _get_weather(loc) if loc else None
        active  = {k for k in _WEATHER_PROFILES if _weather_boost(k, weather) < 0}
        current = {name_to_key[name] for name in existing
                   if name in name_to_key and name_to_key[name] in _WEATHER_PROFILES}
        to_add, to_remove = active - current, current - active
        cond = weather["condition"] if weather else "n/a"
        xlog(f"[INFO] mood_mixes (weather): {cond} → add={sorted(to_add)} remove={sorted(to_remove)}")
        building = {_MOOD_MIX_NAMES[k] for k in (to_add | to_remove)}
        seen_rks = set(recent_rks) | _external_used_rks(existing, building)   # cross-run dedup
        mixes = []
        for profile_key in sorted(to_add):
            tracks = _build_mix_tracks(
                profile_key, essentia_cache, history_entries,
                excluded_album_keys, mix_size, plex, hard_exclude_rks=seen_rks)
            seen_rks.update(str(t.ratingKey) for t in tracks)
            mixes.append((_MOOD_MIX_NAMES[profile_key], profile_key, tracks))
        return mixes, list(to_remove)

    # ------------------------------------------------------------------
    # MODE 2 / 3: General + weather + seasonal mixes
    # ------------------------------------------------------------------
    weather_location = _extras.get("weather_location")
    weather = _get_weather(weather_location) if weather_location else None
    if weather:
        xlog(f"[INFO] mood_mixes: weather = {weather['condition']}, {weather['temp_c']}°C")

    lat = weather.get("lat", 0.0) if weather else 0.0

    # Determine which general profiles are active
    today_wd     = _get_active_weekday()
    current_hour = _get_active_hour()
    cur_ord      = _get_active_date().toordinal()         # travel-aware, consistent with hour/weekday above
    rot_per_day  = max(1, int(_extras.get("mood_mix_rotations_per_day", 6)))
    slot         = (current_hour * rot_per_day) // 24
    cur_gslot    = cur_ord * rot_per_day + slot          # monotonic global slot index
    _rotation    = _load_surface_rotation()              # {key: {"s": last-on-slate gslot}} + "__slot__": last-rotated gslot
    _new_slot    = _rotation.get("__slot__") != cur_gslot   # rotate anytime/city ONLY when the slot advances
    _cur_general = {name_to_key[n] for n in (existing or {})   # general mixes currently on the shelf (reuse within a slot)
                    if n in name_to_key and name_to_key[n] in _GENERAL_PROFILES}

    # ---- Three tiers: CONTEXT (scheduled & in-window, UNCAPPED) + ANYTIME (rotating, time-ranked,
    # overdue-covered) + GEO (>=1 per city). Surfaces every right-now playlist with no cap, rotates the
    # rest across the day, and guarantees every mix surfaces over time (uncapped overdue bonus). ----
    now             = datetime.now(tz=timezone.utc)
    recent_entries  = [e for e in history_entries
                       if e.viewedAt and e.viewedAt >= now - timedelta(days=30)]
    recent_centroid = compute_listening_centroid(recent_entries, essentia_cache, top_n=100)
    _has_centroid   = bool(recent_centroid.get("bpm"))
    n_cats          = len(set(_PROFILE_CATEGORY.values()))

    def _fillable(ks):
        keep = [k for k in ks if _profile_yield(k, essentia_cache) >= _MIN_PROFILE_YIELD]
        if len(keep) < len(ks):
            xlog(f"[INFO] mood_mixes: skipped {len(ks) - len(keep)} unfillable: {sorted(set(ks) - set(keep))}")
        return keep

    def _adist(k):
        # Decade mixes are hits-of-the-period regardless of sound — never rank them by acoustic fit to recent
        # listening; a neutral constant lets them surface via the overdue rotation (fair, sound-independent).
        if _PROFILE_CATEGORY.get(k) == "era":
            return 0.5
        return _acoustic_distance_to_centroid(_MOOD_PROFILES[k], recent_centroid) if _has_centroid else 0.0

    def _stratified(scored):
        # best-fitting few per category (era/geo uncapped) so the pool is vibe-balanced, not near-clones
        per_cat = max(2, math.ceil(n_active * 4 / max(1, n_cats)))
        seen, pool = Counter(), []
        for _, k in scored:
            cat = _PROFILE_CATEGORY.get(k, "other")
            cap = 999 if cat in ("era", "geo_scene") else per_cat
            if seen[cat] < cap:
                seen[cat] += 1
                pool.append(k)
        return pool

    def _overdue(k):
        return _surface_overdue((_rotation.get(k) or {}).get("s"), cur_gslot)

    # CONTEXT — every scheduled profile whose window matches now, UNCAPPED (no _select_diverse_profiles).
    # Excludes city mixes (geo tier). The "right playlist at the right time"; rotates as windows change.
    active_context = _fillable([k for k in _GENERAL_PROFILES
                               if _is_scheduled(k) and not _geo_group(k)
                               and _in_schedule(k, current_hour, today_wd)
                               and _profile_season_ok(k, lat)])

    # ANYTIME — unscheduled, non-city vibe/genre/mood mixes: a rotating, time-ranked set ADDED ON TOP.
    # Ranked by acoustic fit + soft time/day boost minus an uncapped overdue bonus, so time-fitting moods
    # surface now (romance in the evening, ambient late) AND the longest-unseen always cycles in (coverage).
    anytime_pool = _fillable([k for k in _GENERAL_PROFILES
                              if not _is_scheduled(k) and not _geo_group(k)
                              and _profile_season_ok(k, lat)])
    _cur_anytime = [k for k in _cur_general
                    if not _is_scheduled(k) and not _geo_group(k) and k in anytime_pool]
    if _new_slot or not _cur_anytime:                    # rotate to a fresh set when the slot advances
        a_scored = sorted((_mood_rotation_score(k, _adist(k), current_hour, weather) - _overdue(k), k)
                          for k in anytime_pool)
        a_pool   = _stratified(a_scored) if _has_centroid else [k for _, k in a_scored]
        active_anytime = _select_diverse_profiles([(i, k) for i, k in enumerate(a_pool)],
                                                  n_active, max_per_category=max(2, math.ceil(n_active / n_cats)))
    else:
        active_anytime = _cur_anytime                    # stable within the slot — hourly re-runs are no-ops here

    # GEO — guarantee >=1 city mix per city; pick the most time-fitting + most-overdue so every city mix
    # cycles in over time and night mixes lead at night. (Pinned scenes are handled separately below.)
    _cur_city = [k for k in _cur_general if _geo_group(k)]
    if _new_slot or not _cur_city:                       # rotate the per-city pick when the slot advances
        active_city = []
        for _pre in ("glasgow_", "london_", "melbourne_"):
            _cands = _fillable([k for k in _GENERAL_PROFILES
                                if k.startswith(_pre) and k not in _PINNED_PROFILES and _profile_season_ok(k, lat)])
            if _cands:
                active_city.append(min(_cands, key=lambda k:
                    _mood_rotation_score(k, _adist(k), current_hour, weather) - _overdue(k)))
    else:
        active_city = _cur_city                          # stable within the slot

    active_general = (active_context
                      + [k for k in active_anytime if k not in active_context]
                      + [k for k in active_city if k not in active_context and k not in active_anytime])
    xlog(f"[INFO] mood_mixes: context{active_context} + anytime{active_anytime} + city{active_city}")
    # Validation hook: force specific mixes onto the slate regardless of rotation, e.g.
    #   MELODAY_FORCE_MIXES=rave_cave   (comma/space-separated profile keys). No-op when unset.
    _forced = [k for k in os.environ.get("MELODAY_FORCE_MIXES", "").replace(",", " ").split()
               if k in _MOOD_MIX_NAMES and k not in active_general]
    if _forced:
        active_general += _forced
        xlog(f"[INFO] mood_mixes: FORCED onto slate via MELODAY_FORCE_MIXES → {_forced}")

    # Weather + season tiers (condition / calendar gated)
    active_weather  = [k for k in _WEATHER_PROFILES if _weather_boost(k, weather) < 0]
    active_seasonal = [k for k in _SEASONAL_PROFILES if _season_active(k, lat)]

    # Pinned geo scenes — present every run; content refreshes ONCE A DAY (skip any whose artist-rotation
    # already advanced today, so the sub-daily cron doesn't churn them). Never auto-removed.
    _geo_rot = _load_geo_rotation()
    def _scene_done_today(scene):
        return any((v or {}).get("last") == cur_ord for v in (_geo_rot.get(scene) or {}).values())
    active_pinned = [k for k in ("scotland_scene", "australia_scene", "london_scene",
                                 "scottish_hits", "australian_hits", "london_hits",
                                 "uk_scene", "uk_hits", "scotland_now", "london_now", "uk_now", "australia_now")
                     if k in _MOOD_MIX_NAMES and not _scene_done_today(k)]

    active_profiles = active_general + active_weather + active_seasonal + active_pinned

    # Churn control (slot-stable diffing): BUILD only genuinely new entries (+ the daily-refreshed pinned
    # scenes); REMOVE only profiles that rotated out. A surviving playlist is NEVER rebuilt while it stays
    # on the slate — WHY: its content must stay consistent until it actually rotates out and back in;
    # rebuilding in place silently reshuffled a playlist the user was still looking at. Managed = the
    # rotating/conditional tiers; pinned scenes and the cron mixes (_TIME_PROFILES) aren't auto-removed.
    managed         = _GENERAL_PROFILES | _WEATHER_PROFILES | _SEASONAL_PROFILES
    desired_managed = set(active_general) | set(active_weather) | set(active_seasonal)
    currently = {name_to_key[name] for name in existing
                 if name in name_to_key and name_to_key[name] in managed}
    to_remove = sorted(currently - desired_managed)
    to_build = [k for k in active_profiles                              # survivors keep their tracks untouched;
                if k in active_pinned or k not in currently]           # only brand-new entries + pinned (re)build
    if _new_slot:                                                        # advance the rotation once per slot
        for k in active_profiles:
            _rotation.setdefault(k, {})["s"] = cur_gslot                 # "on the slate this slot" → overdue
        _rotation["__slot__"] = cur_gslot
    xlog(f"[INFO] mood_mixes: add={sorted(set(to_build))} remove={to_remove} "
         f"weather{active_weather} season{active_seasonal} pinned{active_pinned}")
    # Cross-run dedup: exclude tracks already used by dedup-eligible mixes built by OTHER runs (e.g. the
    # time-of-day mixes) AND the held-over anytime core, but NOT the mixes we rebuild/remove now.
    building = ({_MOOD_MIX_NAMES[k] for k in to_build}
                | {_MOOD_MIX_NAMES[k] for k in to_remove})
    seen_rks = set(recent_rks) | _external_used_rks(existing, building)
    mixes = []
    for profile_key in to_build:
        is_showcase = _PROFILE_CATEGORY.get(profile_key) in ("era", "geo_scene")  # decade + geo: EXEMPT from cross-mix dedup
        # WHY: isolate each mix's build. Before, one mix raising aborted the whole build_mood_mixes; the caller's
        # broad except then applied NO adds/removes, freezing the slate at the previous (e.g. overnight) state.
        # Now a single failure logs + skips, and to_remove (computed above) still applies, so the slate still rotates.
        try:
            tracks = _build_mix_tracks(
                profile_key, essentia_cache, history_entries,
                excluded_album_keys, mix_size, plex,
                hard_exclude_rks=(recent_rks if is_showcase else seen_rks))
        except Exception:
            xlog(f"[ERROR] mood_mixes: build '{profile_key}' failed, skipping:\n{traceback.format_exc()}")
            continue
        if not tracks:
            continue                                            # nothing fit — don't upsert an empty playlist
        if not is_showcase:
            seen_rks.update(str(t.ratingKey) for t in tracks)   # no repeats across dedup-eligible mixes
        mixes.append((_MOOD_MIX_NAMES[profile_key], profile_key, tracks))
    _save_surface_rotation(_rotation)
    return mixes, to_remove


_SONG_MIN_YEAR_CACHE = {}
_SONG_ANYYEAR_CACHE = {}
_ERA_MIN_LISTENERS = 300_000   # decade mix: a track must clear this Last.fm listener floor to count as
                               # part of the decade's recognisable canon — so older/smaller decades
                               # self-size below it while the modern ones have far more above it
_ERA_RESCUE_TOP_N = 5          # decade ANTHEM-RESCUE: also keep each artist's own Last.fm top-N signature
_ERA_RESCUE_MIN   = 150_000    # songs (>= this many listeners) that fell below the top-150 cut, so genuine
                               # classics the tight cap drops (The Twist, Get Ready, Jailbreak) aren't lost
_GEO_HITS_POOL = 200           # geo HITS mixes (Scottish/Australian/London Hits): keep the top-N songs by
                               # global Last.fm listeners per origin. N IS the auto per-location floor — the
                               # Nth-song listener count varies by place (London's bar ~2.5x Scotland's) and
                               # re-derives as the library grows. Tight by design: genuine hits, repeats are OK.
_GEO_HITS_MIN_LISTENERS = 50_000   # absolute backstop only — guards a hypothetical tiny/new origin from
                                   # surfacing obscure tracks (inert for these three, whose top-N all clear ~267k)
_GEO_HITS_HYBRID = {           # per-profile selection for the geo HITS mixes: a base pool + an anthem-rescue.
    # base = top-`cap` by listeners (place hits) or >=`floor` (UK — too deep for a cap). rescue = the artist's OWN
    # Last.fm top_n (>= rescue listeners) ADDED on top, so genuine signatures Last.fm UNDER-COUNTS on split/quoted
    # pages aren't dropped (Bowie "Heroes" 248k = his #2; Fleetwood Mac "The Chain" 628k sits below London's ~648k
    # top-200 cut). Rescue only adds — the "biggest hits" base is unchanged. top_n/rescue shared + tunable.
    "scottish_hits":   {"cap": _GEO_HITS_POOL, "top_n": 5, "rescue": 150_000},
    "australian_hits": {"cap": _GEO_HITS_POOL, "top_n": 5, "rescue": 150_000},
    "london_hits":     {"cap": _GEO_HITS_POOL, "top_n": 5, "rescue": 150_000},
    "uk_hits":         {"floor": 400_000,      "top_n": 5, "rescue": 150_000},
}
_GEO_RADIO = {                 # geo RADIO mixes: ~(1-throwback_frac) most-popular CONTEMPORARY songs (released in
    # the last `recent_years`) + ~throwback_frac throwback classics, 1-per-artist — like a place-only radio
    # station. recent/throwback_pool cap the distinct-artist candidate windows the daily pick draws from.
    # `recency_base` (>1) leans the CONTEMPORARY bucket toward newer releases like a real "currents" rotation —
    # score = listeners * recency_base ** (year - cut) — and the recent pick becomes a weighted sample so the
    # freshest/biggest currents heavy-rotate (throwback stays listener-ranked; classics are timeless). 1.0 = off.
    # Routed via _is_geo_radio in _build_mix_tracks.
    "scotland_now": {"recent_years": 5, "throwback_frac": 0.10, "recent_pool": 55, "throwback_pool": 30, "recency_base": 1.6},
    "london_now":   {"recent_years": 5, "throwback_frac": 0.10, "recent_pool": 55, "throwback_pool": 30, "recency_base": 1.6},
    "uk_now":       {"recent_years": 5, "throwback_frac": 0.10, "recent_pool": 80, "throwback_pool": 40, "recency_base": 1.6},
    "australia_now": {"recent_years": 5, "throwback_frac": 0.10, "recent_pool": 55, "throwback_pool": 30, "recency_base": 1.6},
}
_GEO_RADIO_PROFILES = set(_GEO_RADIO)
# Geo-radio variety: a track the station featured in the last _RADIO_SURFACE_WINDOW days is SOFT-deprioritised
# (weight × down to _RADIO_SURFACE_FLOOR, never hard-banned) so each artist's REPRESENTATIVE track rotates
# across their catalogue day-to-day. The artist ROSTER stays popularity-stable (the giants recur — power
# rotation), only the track per artist cycles. Per-mix log lives in the geo-rotation file: "surfaced": {rk: ord}.
_RADIO_SURFACE_WINDOW = 14
_RADIO_SURFACE_FLOOR  = 0.3
# A title that marks a re-record / remaster / re-version / reissue — i.e. NOT new material, however recent its
# release year. Used by the geo-radio split to keep these out of "contemporary" (a radio station treats them as
# the OLD songs they are). KEEPS "radio edit" / "single version" / "album version" — those ARE the radio cut.
_REISSUE_RE = re.compile(
    r"\bre-?record(?:ed|ing)?\b|\bnew\s+(?:version|recording|mix|cut|edit)\b|\bre-?master(?:ed)?\b"
    r"|\banniversary\b|\b(?:19|20)\d{2}\s+(?:version|mix|edit|re-?master(?:ed)?|recording|cut)\b"
    r"|\b(?:version|mix|edit|re-?master(?:ed)?|recording|cut)\s+(?:19|20)\d{2}\b"
    r"|['’]\d{2}\s+(?:version|mix|edit|re-?master(?:ed)?|recording|cut)\b", re.I)

# Decade/era mixes drop compilation albums (Various-Artists comps + single-artist Greatest Hits/Best Of).
# Authoritative source = the MusicBrainz `release_types` read from each file's tags (cached); for the ~1%
# of tracks not yet tagged, fall back to the file-path folder convention.
_EXCLUDE_COMPS_FROM_ERA     = bool(_extras.get("exclude_compilations_from_era", True))
_ERA_EXCLUDED_RELEASE_TYPES = {str(t).lower() for t in
                               (_extras.get("era_excluded_release_types") or ["compilation"])}
_COMP_FOLDER_CUES = ("greatest hits", "best of", "the best of", "very best of", "anthology",
                     "the collection", "the essential", "compilation", "superhits", "super hits")


def _comp_folder_fallback(file_path):
    """Heuristic compilation check from the file path, for tracks whose MB `release_types` isn't cached
    yet: the user's `/Various Artists [add compilations to this artist]/` parent folder (parts[-3]) or a
    greatest-hits/best-of style album folder (parts[-2])."""
    if not file_path:
        return False
    parts = file_path.split("/")
    artist_folder = parts[-3].lower() if len(parts) >= 3 else ""
    album_folder  = parts[-2].lower() if len(parts) >= 2 else ""
    if "various artist" in artist_folder:
        return True
    return any(cue in album_folder for cue in _COMP_FOLDER_CUES)


def _is_compilation(entry):
    """True if a track's release is a compilation. PRIMARY signal: the cached MB `release_types` (Picard
    tags); FALLBACK when those are absent (NULL/empty): the file-path folder heuristic (Various Artists /
    hits-comp album name). Used to keep mixes on the studio original AND to drop comps from decade mixes."""
    rt = entry.get("release_types")
    if rt:
        return bool({str(t).lower() for t in rt} & _ERA_EXCLUDED_RELEASE_TYPES)
    return _comp_folder_fallback(entry.get("file_path"))
def _era_is_comp(entry):
    """Compilation check for the DECADE mixes specifically — `_is_compilation` PLUS any track filed under a
    Various-Artists folder. WHY: VA soundtracks (release_types ['album','soundtrack']) slip _is_compilation,
    and a song owned ONLY on a VA comp/soundtrack has no reliable decade and shouldn't anchor one (e.g.
    'God Only Knows'/'Heroes', owned only on VA releases). Era-scoped — canonical selection keeps plain
    _is_compilation."""
    return _is_compilation(entry) or "/various artists" in (entry.get("file_path") or "").lower()


def _entry_song_key(entry):
    # WHY: lastfm_query_title (not clean_title) — a remix/live/acoustic keys as its OWN song (own listener
    # count, its own decade-collapse entry, not borrowing the original's popularity), reissues still collapse.
    return (norm_text(primary_artist(entry.get("artist") or "")),
            norm_text(lastfm_query_title(entry.get("title") or "")))
def _song_min_year_map(essentia_cache):
    """Earliest cached year per song (artist+title) — so decade/era gating uses a track's ORIGINAL
    release rather than a reissue/compilation album year (Plex `year` is the album's, which is wrong
    for compilations). Computed once per cache object."""
    cid = id(essentia_cache)
    m = _SONG_MIN_YEAR_CACHE.get(cid)
    if m is None:
        m = {}
        for e in essentia_cache.values():
            y = e.get("year")
            if not y or _era_is_comp(e):        # WHY skip comps/VA: their album-year is the comp/reissue year,
                continue                        # not the original, and it was mis-dating songs (a 1990 comp set
            k = _entry_song_key(e)              # "God Only Knows" to the 90s). Studio-copy songs are unaffected —
            if k[1] and y < m.get(k, 9999):     # their min already comes from the studio copy.
                m[k] = y
        _SONG_MIN_YEAR_CACHE[cid] = m
    return m
def _song_year_anycopy_map(essentia_cache):
    """Earliest cached year per song over ALL copies — INCLUDING compilations (unlike _song_min_year_map,
    which excludes them). For the geo-radio contemporary test the comp signal is wanted: if a song ever
    appeared on a pre-cutoff comp/Greatest-Hits it is NOT brand-new (you don't comp a last-5-years song),
    so a clean-titled reissue (its only non-comp copy being a recent re-release) is still dated as old."""
    cid = id(essentia_cache)
    m = _SONG_ANYYEAR_CACHE.get(cid)
    if m is None:
        m = {}
        for e in essentia_cache.values():
            y = e.get("year")
            if not y:
                continue
            k = _entry_song_key(e)
            if k[1] and y < m.get(k, 9999):
                m[k] = y
        _SONG_ANYYEAR_CACHE[cid] = m
    return m


_HISTORY_PLAY_COUNTS_CACHE = {}   # id(history_entries) -> Counter of plays per rating_key
def _history_play_counts(history_entries):
    """Per-track play Counter, memoised on the history_entries object so the several mix builds in
    one run share a single pass over the (large) play history instead of rebuilding it each time."""
    cid = id(history_entries)
    pc = _HISTORY_PLAY_COUNTS_CACHE.get(cid)
    if pc is None:
        pc = Counter(str(e.ratingKey) for e in history_entries)
        _HISTORY_PLAY_COUNTS_CACHE.clear()   # only the current run's history object is ever needed
        _HISTORY_PLAY_COUNTS_CACHE[cid] = pc
    return pc


_NONCANON_SUBSTR = ("instrumental", "unplugged", "karaoke", "remix", "acoustic", "rehearsal",
                    "acapella", "a cappella", "re-recorded", "rerecorded", "sessions")
# In the TITLE, only treat "live"/"demo" as a version marker when it's delimited ("(Live)", "- Live",
# "Live at…") — so song titles like "Live and Let Die" aren't mistaken for live recordings.
_TITLE_LIVE_DEMO = re.compile(
    r"[\(\[\-]\s*live\b|\blive (?:at|in|from|around|session|recording|concert|version)\b"
    r"|\blive\s*[\)\]]|[\(\[]\s*demo\b|\bdemo\s*[\)\]]", re.I)
_ALBUM_LIVE = re.compile(r"\blive\b", re.I)   # a bare "Live" in the ALBUM folder = a live album
def _version_penalty(title, album):
    """Lower = more canonical. Penalises different-recording copies (live / remix / demo /
    instrumental / acoustic / orchestral-"sessions" re-recordings — from the track title + its album
    name) and, mildly, compilations. Ties then break to the earliest (original) release year."""
    title = (title or "").lower(); album = (album or "").lower()
    pen = sum(4 for kw in _NONCANON_SUBSTR if kw in title or kw in album)
    if _TITLE_LIVE_DEMO.search(title) or _ALBUM_LIVE.search(album):
        pen += 4
    if "various artists" in album:
        pen += 2
    if any(cue in album for cue in _COMP_FOLDER_CUES):   # hits-comp album name (name fallback)
        pen += 1
    return pen
_NONCANON_RELEASE_TYPES = {"live", "demo", "dj-mix"}   # non-studio MB release types -> penalise the copy
def _canonical_penalty(entry):
    """Version penalty for a CACHE entry (album = the file-path's album folder). Non-studio releases —
    compilations AND live/demo/dj-mix albums — are pushed down via the MB release-type signal (reliable
    even when the album name gives nothing away, e.g. the live album "Bullet In A Bible") so the studio
    original wins the canonical pick. Lower = more canonical."""
    parts = (entry.get("file_path") or "").split("/")
    album = parts[-2] if len(parts) >= 2 else (entry.get("file_path") or "")
    pen = _version_penalty(entry.get("title"), album)
    if _is_compilation(entry):
        pen += 2
    if {str(t).lower() for t in (entry.get("release_types") or [])} & _NONCANON_RELEASE_TYPES:
        pen += 4
    return pen
def _canonical_penalty_track(t):
    """Version penalty for a Plex Track (album = parentTitle)."""
    return _version_penalty(getattr(t, "title", ""), getattr(t, "parentTitle", ""))


# ---------------------------------------------------------------------------
# Geo "scene" mixes — equal-weight artist-coverage rotation
# Every eligible artist from the place is weighted EQUALLY: rotate through ALL of them over a few days
# (longest-unseen first) for even, fair coverage — no own-depth / global-fame lean — surfacing a RANDOM
# song from each chosen artist's top-10 (by global Last.fm popularity) eligible tracks. See _is_geo below.
# WHY equal weight: a "Sounds of X" showcase should represent the whole local scene evenly, not skew to
# the artists you own most or the globally-famous few.
# ---------------------------------------------------------------------------
_GEO_SONGS_PER_ARTIST = 10        # rotate a random song from each artist's top-N (after recently-played)
_GEO_OVERDUE_RATE     = 0.10      # per-day rotation bonus for an unshown artist; tuned so the longest-
                                  # unseen always cycle in within ~ceil(artists/mix) days — even, fair
                                  # coverage of every eligible artist (all artists weighted equally)

# Score/classical-dominated catalogues (film/soundtrack/game composers) aren't a band/song "scene".
# Deliberately TIGHTER than _INSTRUMENTAL_CUES (no post-rock/ambient/math-rock) so legit post-rock bands
# like Mogwai stay in.
_SCORE_CLASSICAL_TAGS = ("soundtrack", "film score", "original score", "score", "stage & screen",
    "classical", "contemporary classical", "modern classical", "neo-classical", "orchestral", "opera",
    "video game", "library music")


def _is_score_classical_artist(entries):
    """True if a MAJORITY of an artist's tracks carry a score/classical tag — so film/soundtrack/classical
    catalogues (Lorne Balfe, Craig Armstrong) stay out of a band/song scene mix, while a band with the odd
    orchestral track stays in."""
    if not entries:
        return False
    hits = sum(1 for e in entries
               if any(s in t for t in _track_style_tags(e) for s in _SCORE_CLASSICAL_TAGS))
    return hits * 2 > len(entries)


def _geo_rotation_path():
    return os.path.join(_BASE_DIR, "assets", "geo_scene_rotation.json")


def _load_geo_rotation():
    """Per-scene rotation state {profile_key: {artist: {"last": ordinal, "songs": [song-keys]}}}. {} if missing."""
    try:
        with open(_geo_rotation_path()) as f:
            return json.load(f).get("scenes", {})
    except (OSError, ValueError):
        return {}


def _save_geo_rotation(scenes):
    """Atomic best-effort write (never fatal to a build), mirroring _save_geo_rotation."""
    try:
        os.makedirs(os.path.dirname(_geo_rotation_path()), exist_ok=True)
        tmp = _geo_rotation_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"version": 1, "scenes": scenes}, f)
        os.replace(tmp, _geo_rotation_path())
    except OSError as e:
        xlog(f"[WARN] geo_scene: could not persist rotation state: {e}")


def _geo_overdue_bonus(last_ord, cur_ord):
    """Rotation bonus that rises (uncapped) with days since an artist was last shown. With all artists
    weighted equally this is the sole ranking signal (a tiny random jitter only breaks ties), so the
    longest-unseen always cycle in within ~a cycle — even coverage of the whole scene. Never-shown artists
    get top priority, so newly-added local artists surface promptly. Uncapped is deliberate: it guarantees
    every artist returns rather than letting a lucky early streak permanently starve the rest."""
    if last_ord is None:
        return _GEO_OVERDUE_RATE * 10_000
    return _GEO_OVERDUE_RATE * max(0, cur_ord - last_ord)


def _dedup_canonical(tracks, essentia_cache=None):
    """Collapse same-song duplicate Track objects to the most CANONICAL copy (studio/original — not
    live / remix / demo / instrumental / compilation), keeping the song at its best (first / highest-
    scored) position in the list. Uses the cache entry (by rating key) for the version penalty. NOTE:
    the play-history playlists (On Repeat etc.) rank by plays first, then swap to the canonical copy via
    _canonicalize_tracks — so the canonical preference is applied everywhere, not skipped here."""
    best, order = {}, []
    for t in tracks:
        sk = _song_key(t)
        e  = (essentia_cache or {}).get(str(t.ratingKey))
        if e:
            cs = (_canonical_penalty(e), e.get("year") or 9999, str(t.ratingKey))
        else:                                   # not in cache -> score from the Track itself
            cs = (_canonical_penalty_track(t), getattr(t, "year", None) or 9999, str(t.ratingKey))
        if sk not in best:
            best[sk] = (cs, t); order.append(sk)
        elif cs < best[sk][0]:
            best[sk] = (cs, t)
    return [best[sk][1] for sk in order]


_CANONICAL_RK_CACHE = {}   # id(essentia_cache) -> {song_key -> canonical rk}
def _canonical_rk_map(essentia_cache):
    """Map each song to the rating-key of its CANONICAL copy (studio original over live/remix/comp),
    computed once per cache (mirrors _song_min_year_map). Meaningfully-different recordings (remix /
    extended / live / acoustic) carry their own song key via lastfm_query_title, so each is its own
    canonical copy; a song with a single copy maps to itself."""
    cid = id(essentia_cache)
    m = _CANONICAL_RK_CACHE.get(cid)
    if m is None:
        best = {}                                   # song_key -> (penalty, year, rk)
        for rk, e in essentia_cache.items():
            sk = _entry_song_key(e)
            if not sk[1]:                            # no title -> can't group
                continue
            cand = (_canonical_penalty(e), e.get("year") or 9999, rk)
            if sk not in best or cand < best[sk]:
                best[sk] = cand
        m = {sk: cand[2] for sk, cand in best.items()}
        _CANONICAL_RK_CACHE.clear()                 # only the current run's cache is ever needed
        _CANONICAL_RK_CACHE[cid] = m
    return m


def _canonicalize_tracks(plex, tracks, essentia_cache):
    """Swap each track for the canonical copy of its song (studio original over live/remix/compilation),
    preserving order and length. Selection/ranking is already done — this only changes WHICH copy shows,
    so it's safe even on play-ranked playlists (On Repeat etc.). Meaningfully-different recordings and
    single-copy songs map to themselves (no-op). Resolves only the copies that actually change."""
    if not tracks or not essentia_cache:
        return tracks
    cmap = _canonical_rk_map(essentia_cache)
    targets = []                                    # (original track, canonical rk)
    for t in tracks:
        rk = str(t.ratingKey)
        e  = essentia_cache.get(rk)
        sk = _entry_song_key(e) if e else _song_key(t)
        targets.append((t, cmap.get(sk, rk)))
    need = {cr for t, cr in targets if cr != str(t.ratingKey)}
    swapped = resolve_tracks_by_keys(plex, list(need)) if need else {}
    return [swapped.get(cr) or t for t, cr in targets]   # fall back to the original if it can't resolve


# ---------------------------------------------------------------------------
# DJ-style flow ordering — re-sequence a SELECTED mix so adjacent tracks mix
# (beatmatch + harmonic key + energy). Beat-driven mixes also get a set ARC.
# Pure-acoustic (cached bpm/key/energy); no per-track Plex calls.
# ---------------------------------------------------------------------------
_DJ_ARC_PROFILES = {
    "rave_cave", "techno", "trance", "deep_house", "house_party", "dnb", "bass_drop",
    "uk_garage", "festival_edm", "dance_pop", "funk_disco",
    "glasgow_house", "glasgow_underground", "glasgow_bass",
    "london_garage", "london_grime", "london_dubstep", "london_jungle",
    "melbourne_club", "melbourne_techno",
    # High-energy electronic that isn't club/city-tagged but is just as beat-driven — give it the same
    # build->peak->wind-down set arc. WHY: any electronic-based mix should flow like a DJ set.
    "industrial", "hyperpop", "chiptune",
}
_DJ_ARC_WEIGHT = 2.0   # how strongly the energy ARC pulls vs adjacent-transition smoothness


def _dj_transition(ea, eb, emb_a=None, emb_b=None):
    """DJ transition cost between two cache entries — beatmatch + harmonic key + energy + a light embedding
    "sounds-like" term (the DJ essentials), offline. Lower = smoother mix. Same-artist back-to-back is
    heavily penalised. emb_a/emb_b are precomputed normalised embeddings (None -> the emb term is skipped)."""
    d  = 2.0 * get_bpm_distance(ea.get("bpm"), eb.get("bpm"))        # octave-aware tempo (reused)
    d += 1.5 * get_harmonic_distance(ea.get("key"), eb.get("key"))   # Camelot-wheel key (reused)
    en_a, en_b = ea.get("energy"), eb.get("energy")
    if en_a is not None and en_b is not None:
        d += min(abs(en_a - en_b) / 10.0, 1.0)
    if ea.get("artist") and ea.get("artist") == eb.get("artist"):
        d += 2.0
    if emb_a is not None and emb_b is not None:
        d += 0.5 * (1.0 - _emb_cosine(emb_a, emb_b))   # "sounds-like" tiebreak; skipped when either lacks an embedding
    return d


def _dj_order(tracks, essentia_cache, arc=False):
    """Sequence already-selected tracks like a DJ: smooth tempo/key/energy transitions (greedy + 2-opt);
    arc=True also shapes the energy (ease-in -> build -> peak ~75% -> wind-down). Re-orders only — the
    selection is unchanged. Pure-acoustic, no Plex calls."""
    if len(tracks) < 4:
        return tracks
    ent  = {str(t.ratingKey): essentia_cache.get(str(t.ratingKey), {}) for t in tracks}
    embs = {rk: _track_emb(e) for rk, e in ent.items()}   # precompute once — _dj_transition is called O(n²)
    def _c(a, b):
        ka, kb = str(a.ratingKey), str(b.ratingKey)
        return _dj_transition(ent[ka], ent[kb], embs[ka], embs[kb])

    if not arc:
        rem = tracks[1:]; order = [tracks[0]]; cur = order[0]
        while rem:                                                  # greedy nearest-neighbour seed
            nxt = min(rem, key=lambda t: _c(cur, t)); rem.remove(nxt); order.append(nxt); cur = nxt
        def _edge(a, b):
            d = _c(a, b); return (d ** 3) * 20 if d > 0.4 else d     # cubic penalty on hard jumps
        improved, passes = True, 0
        while improved and passes < max(20, len(order)):            # 2-opt refinement
            improved, passes = False, passes + 1
            for i in range(1, len(order) - 1):
                for j in range(i + 1, len(order)):
                    a, b, c = order[i - 1], order[i], order[j]
                    d2 = order[j + 1] if j + 1 < len(order) else None
                    before = _edge(a, b) + (_edge(c, d2) if d2 else 0.0)
                    after  = _edge(a, c) + (_edge(b, d2) if d2 else 0.0)
                    if after + 1e-9 < before:
                        order[i:j + 1] = order[i:j + 1][::-1]; improved = True
        return order

    # ARC mode: energy ease-in -> peak ~75% -> wind-down, with beatmatched/harmonic transitions
    def _intensity(e):
        comp = w = 0.0
        if e.get("arousal")      is not None: comp += 0.30 * e["arousal"];                               w += 0.30
        if e.get("danceability") is not None: comp += 0.25 * e["danceability"];                          w += 0.25
        if e.get("bpm")          is not None: comp += 0.35 * min(max((e["bpm"] - 90) / 80.0, 0.0), 1.0); w += 0.35  # tempo = the dance-energy signal
        if e.get("energy")       is not None: comp += 0.10 * min(max((e["energy"] + 22) / 16.0, 0.0), 1.0); w += 0.10
        return comp / w if w else 0.5
    inten = {rk: _intensity(e) for rk, e in ent.items()}
    lo, hi = min(inten.values()), max(inten.values()); rng = (hi - lo) or 1.0
    norm = {rk: (v - lo) / rng for rk, v in inten.items()}
    n = len(tracks)
    def _target(i):
        p = i / (n - 1)
        return 0.45 + 0.55 * (p / 0.75) ** 0.85 if p <= 0.75 else 1.0 - 0.45 * ((p - 0.75) / 0.25)
    rem = tracks[:]
    first = min(rem, key=lambda t: norm[str(t.ratingKey)]); rem.remove(first); order = [first]
    for i in range(1, n):
        T, prev = _target(i), order[-1]
        nxt = min(rem, key=lambda t: _DJ_ARC_WEIGHT * abs(norm[str(t.ratingKey)] - T) + _c(prev, t))
        rem.remove(nxt); order.append(nxt)
    return order


# ---------------------------------------------------------------------------
# Seed-artist centroids — a few mixes are defined by SOUND, not by any genre tag. Their target acoustic
# fingerprint is the mean vector of artists we know belong, computed from the cache at build time (already
# in raw entry space, so directly comparable in _acoustic_distance_to_centroid — no calibration needed).
# acoustic_romance: gentle acoustic singer-songwriters à la Jack Johnson; the romantic lyric themes
# (_PROFILE_LYRIC_THEMES) then pull the love songs to the top within that sound.
# ---------------------------------------------------------------------------
_SEED_ARTISTS = {
    "acoustic_romance": ["jack johnson", "jason mraz", "donavon frankenreiter", "matt costa", "amos lee",
                         "ben howard", "angus & julia stone", "city and colour", "xavier rudd",
                         "ziggy alberts", "jose gonzalez", "josé gonzález", "iron & wine", "damien rice"],
}
_SEED_DIMS = ("bpm", "energy", "danceability", "brightness", "beat_confidence", "onset_rate",
              "dynamic_complexity", "arousal", "valence", "vocal_presence")
_SEED_CENTROID_CACHE = {}

def _seed_centroid(profile_key, essentia_cache):
    """Mean acoustic vector of a mix's seed artists (memoised). None if too few seed tracks are present."""
    if profile_key in _SEED_CENTROID_CACHE:
        return _SEED_CENTROID_CACHE[profile_key]
    seeds = _SEED_ARTISTS.get(profile_key) or []
    ents = [e for e in essentia_cache.values()
            if any(sd in (e.get("artist") or "").lower() for sd in seeds)]
    cen = None
    if len(ents) >= 20:
        cen = {}
        for d in _SEED_DIMS:
            vals = [e[d] for e in ents if e.get(d) is not None]
            if vals:
                cen[d] = sum(vals) / len(vals)
    _SEED_CENTROID_CACHE[profile_key] = cen
    return cen


# Embedding seed-artists — like _SEED_ARTISTS but used ONLY for the embedding "sounds-like" pull in
# _combined_score (they do NOT swap the acoustic centroid the way _SEED_ARTISTS does), so they sharpen the
# audit's broad/fuzzy mixes without disturbing their hand-tuned centroid or gate. acoustic_romance reuses
# its _SEED_ARTISTS list. WHY: emb_effnet was collected but unused (audit Tier-4: seed & fuzzy mixes).
_EMB_SEEDS = {
    "acoustic_romance": _SEED_ARTISTS["acoustic_romance"],
    # situationship: yearning bedroom-pop / alt-R&B about undefined relationships
    "situationship": ["clairo", "steve lacy", "the marias", "omar apollo", "snail mail", "beabadoobee",
                      "gracie abrams", "holly humberstone", "phoebe bridgers", "boygenius"],
    # indie_rock: big indie / alt-rock anthems
    "indie_rock": ["the killers", "arctic monkeys", "kings of leon", "the strokes", "interpol", "foals",
                   "two door cinema club", "phoenix", "vampire weekend", "bloc party", "franz ferdinand"],
}
_SEED_EMB_CENTROID_CACHE = {}

def _seed_emb_centroid(profile_key, essentia_cache):
    """Mean normalised embedding of a profile's _EMB_SEEDS artists (memoised). None if <5 present / no numpy."""
    if profile_key in _SEED_EMB_CENTROID_CACHE:
        return _SEED_EMB_CENTROID_CACHE[profile_key]
    seeds = _EMB_SEEDS.get(profile_key) or []
    cen = None
    if seeds and _NUMPY_AVAILABLE:
        rks = [rk for rk, e in essentia_cache.items()
               if any(sd in (e.get("artist") or "").lower() for sd in seeds)]
        cen = _emb_centroid(rks, essentia_cache)
    _SEED_EMB_CENTROID_CACHE[profile_key] = cen
    return cen

_MIX_EMB_CORE_N = 40   # build the cohesion centroid from the N candidates nearest the acoustic centroid

def _mix_emb_centroid(cand_keys, essentia_cache, target, which):
    """Two-pass "sounds-like" cohesion centroid: take the _MIX_EMB_CORE_N candidates closest to the mix's
    ACOUSTIC centroid (its core sound), return the mean normalised `which` embedding of those. None (no-op)
    if numpy is absent or <5 of the core carry the vector — graceful under partial coverage. emb_musicnn
    (vibe, cross-genre) for fuzzy mood mixes; emb_effnet (sub-style) for genre-gated mixes."""
    if not _NUMPY_AVAILABLE:
        return None
    core = heapq.nsmallest(_MIX_EMB_CORE_N, cand_keys,
                           key=lambda rk: _acoustic_distance_to_centroid(essentia_cache.get(rk, {}), target))
    return _emb_centroid(core, essentia_cache, which)

def _embedding_boost(entry, cen, which):
    """Pull a track toward the mix's embedding-cohesion centroid ("sounds like the mix's core"). `cen`/`which`
    are computed once per build in _build_mix_tracks (a profile's seed-artist centroid for _EMB_SEEDS profiles,
    else self-cohesion). Graceful 0.0 when there's no centroid or the track lacks the embedding."""
    if cen is None:
        return 0.0
    v = _track_emb(entry, which)
    if v is None:
        return 0.0
    return -_EMB_WEIGHT * _emb_cosine(v, cen)   # negative = pull (closer in sound -> better/lower score)


# ---------------------------------------------------------------------------
# Loudness-consistency lean — pacing mixes prefer steady-loudness (low LRA) tracks
# ---------------------------------------------------------------------------
# integrated_loudness holds the EBU R128 loudness RANGE (LRA, in LU): low = steady/compressed dynamics,
# high = big quiet↔loud swings. workout/running/focus/deep_work/sleep flow better with steady loudness
# (cadence, concentration, no jarring jumps), so they lean toward LOW LRA. WHY: integrated_loudness was
# collected but unused — this is the audit's activity-loudness-1. It's a separate axis from `energy`
# (which is LUFS), so it does not double-count with the centroid. (cache LRA median ≈6.4, p75 ≈10.)
_LOUDNESS_WEIGHT  = 0.06
_LOUDNESS_LRA_CAP = 12.0
_PACING_PROFILES  = {"workout", "running", "focus", "deep_work", "sleep"}


def _loudness_consistency_boost(entry, profile_key):
    """Favour steady-loudness (low LRA) tracks in pacing mixes; no-op elsewhere or when data is missing."""
    if profile_key not in _PACING_PROFILES:
        return 0.0
    lra = entry.get("integrated_loudness")
    if lra is None:
        return 0.0
    steadiness = 1.0 - min(1.0, lra / _LOUDNESS_LRA_CAP)   # 1.0 = perfectly steady … 0.0 = very dynamic
    return -_LOUDNESS_WEIGHT * steadiness


# ---------------------------------------------------------------------------
# Danceability co-signal — dance/EDM mixes deprioritise clearly-non-danceable tracks
# ---------------------------------------------------------------------------
# danceability_hl is the MusiCNN high-level danceability classifier (0–1), a second opinion distinct from the
# Essentia danceability already in the centroid. The Discogs style gate can still let through the occasional
# ballad / interlude / ambient cut that carries the genre tag but isn't danceable; for the dance/EDM mixes
# (the DJ-arc set) we softly PUSH DOWN tracks the classifier scores clearly non-danceable. SOFT, not a gate —
# the dance subset already averages ~0.93, so this only nudges the rare misfire and never thins the pool.
# WHY: danceability_hl was collected but unused. (audit danceability-co-gate — implemented as a soft penalty)
_DANCE_PROFILES       = _DJ_ARC_PROFILES        # the 23 dance/electronic mixes (arc-ordered + danceability-penalised)
_DANCE_FLOOR          = 0.50                     # below this, a dance-mix track is nudged down
_DANCE_PENALTY_WEIGHT = 0.30                     # push at danceability_hl=0; scales with the shortfall below the floor


def _danceability_penalty(entry, profile_key):
    """Soft push-down for clearly-non-danceable tracks in dance/EDM mixes; no-op elsewhere / when missing."""
    if profile_key not in _DANCE_PROFILES:
        return 0.0
    dhl = entry.get("danceability_hl")
    if dhl is None or dhl >= _DANCE_FLOOR:
        return 0.0
    return _DANCE_PENALTY_WEIGHT * (_DANCE_FLOOR - dhl)   # positive = worse score (pushed down)


def _build_mix_tracks(profile_key, essentia_cache, history_entries,
                      excluded_album_keys, mix_size, plex, hard_exclude_rks=None):
    """Build the mix_size-track list for a single mix profile.

    `hard_exclude_rks` (tracks played in the last N days + tracks already used by other
    dedup-eligible mixes this run / cross-run) is excluded UNCONDITIONALLY and never re-admitted.
    Length is reached instead by a wide candidate pool and by relaxing only the in-mix artist cap
    (prefer 1 per artist -> 2 -> 3 -> unlimited). A hard-gated genre mix whose eligible pool is
    genuinely too small returns short rather than re-using excluded tracks or breaking its genre.
    """
    target       = _MOOD_PROFILES[profile_key]
    if profile_key in _SEED_ARTISTS:                # centroid-defined mix: target = mean vector of seed artists
        _sc = _seed_centroid(profile_key, essentia_cache)
        if _sc:
            target = _sc
    play_counts  = _history_play_counts(history_entries)
    _is_era      = _PROFILE_CATEGORY.get(profile_key) == "era"
    _is_geo      = _PROFILE_CATEGORY.get(profile_key) == "geo_scene"
    _is_geo_hits = profile_key in _GEO_HITS_PROFILES   # geo HITS variant (category geo_scene, but top-N by listeners)
    _is_geo_radio = profile_key in _GEO_RADIO_PROFILES  # geo RADIO variant (contemporary + throwback split, 1/artist)
    _is_showcase = _is_era or _is_geo          # decade + geo mixes: popularity-ranked, gated, deduped

    def _combined_score(rk):
        """Acoustic distance adjusted by mood/style tag compatibility and play count — EXCEPT the
        SHOWCASE mixes (decade + geo), which rank purely by Last.fm popularity (the canon of the decade
        / scene) within their year/origin gate, rather than matching a target acoustic fingerprint."""
        entry = essentia_cache.get(rk, {})
        if _is_showcase:
            return -(entry.get("lastfm_listeners") or 0)
        score = (
            _acoustic_distance_to_centroid(entry, target)
            + _moodclass_boost(entry, profile_key)
            + _origin_boost(entry, profile_key)
            + _popularity_boost(entry, profile_key)
            + _listening_hour_boost(rk)
            + _loudness_consistency_boost(entry, profile_key)
            + _danceability_penalty(entry, profile_key)
            + _embedding_boost(entry, _emb_cen, _emb_which)
            + _lyric_boost(entry, profile_key)
            + _lyric_lang_penalty(entry, profile_key)
        )
        # Style-GATED mixes already confirmed genre membership via tags in the gate. Re-applying the
        # external metadata-tag boosts here double-counts the same (noisy) Discogs/Plex/Last.fm tags
        # and rewards mis-tagged mainstream (e.g. pop the Discogs-400 classifier sprays "hardstyle"
        # onto) over genuine genre tracks — so badly that a eurodance novelty out-scored Hannah Laing
        # in Rave Cave. So rank these on our OWN audio analysis (centroid distance + TF mood-class)
        # plus the deliberate popularity/origin/rating leans only. Non-gated mood mixes have NO genre
        # gate, so the tag boosts ARE their primary signal — keep them there.
        if profile_key not in _STYLE_DEFINED_PROFILES:
            score += (_mood_tag_boost(entry, profile_key)
                      + _style_tag_boost(entry, profile_key)
                      + _moodtheme_boost(entry, profile_key)
                      + _lastfm_tag_boost(entry, profile_key))
        return score

    # Candidate keys to score. Style-defined mixes (Synth-Pop Romance, Folk & Acoustic, the
    # jazz/classical/indie mixes) keep only tracks whose styles match the genre so the mix stays
    # genre-pure — apply that hard gate BEFORE scoring so _combined_score (6-10 boost fns/track)
    # runs over the eligible subset (often ~1-2% of the library) rather than the whole cache.
    # WHY output-identical: filtering then sorting == sorting then filtering for a deterministic
    # key, and this style gate has no fallback.
    if profile_key in _STYLE_DEFINED_PROFILES:
        _cand_keys = [rk for rk in essentia_cache
                      if _has_required_style(essentia_cache.get(rk, {}), profile_key)]
    else:
        _cand_keys = list(essentia_cache)

    # Embedding "sounds-like" cohesion: compute the mix's core embedding centroid ONCE, then
    # _combined_score pulls candidates toward it. emb_effnet (sub-style) for genre-gated mixes, emb_musicnn
    # (vibe, cross-genre) for fuzzy mood mixes; a profile's _EMB_SEEDS seed-artist centroid (effnet) overrides.
    # Skipped for showcase mixes (they rank by popularity, not _combined_score). Graceful: None -> no-op.
    _emb_which = "effnet" if profile_key in _STYLE_DEFINED_PROFILES else "musicnn"
    _emb_cen   = _seed_emb_centroid(profile_key, essentia_cache)
    if _emb_cen is not None:
        _emb_which = "effnet"                                   # seed centroids are effnet
    elif not _is_showcase:
        _emb_cen = _mix_emb_centroid(_cand_keys, essentia_cache, target, _emb_which)

    history_rks = sorted(
        [rk for rk in _cand_keys if rk in play_counts],
        key=lambda rk: (_combined_score(rk), -play_counts.get(rk, 0))
    )
    library_rks = sorted(
        [rk for rk in _cand_keys if not play_counts.get(rk)],
        key=_combined_score
    )

    # Decade/era + nostalgia mixes: keep only era-appropriate tracks, gating on the song's EARLIEST
    # year across the library (its original release) rather than the album year — so reissues and
    # compilations don't leak a track into the wrong decade. Fall back if too thin.
    _yw = _PROFILE_YEAR_WINDOW.get(profile_key)
    if _yw:
        lo, hi = _yw
        _smy = _song_min_year_map(essentia_cache)
        def _orig_in(rk):
            e  = essentia_cache.get(rk, {})
            oy = _smy.get(_entry_song_key(e)) or e.get("year")
            return oy is not None and (lo is None or oy >= lo) and (hi is None or oy <= hi)
        _h = [rk for rk in history_rks if _orig_in(rk)]
        _l = [rk for rk in library_rks if _orig_in(rk)]
        history_rks = _h or history_rks
        library_rks = _l or library_rks

    # Anthem gate (Throwback Anthems): keep only an artist's OWN Last.fm top-N tracks that ALSO clear the
    # global-listener floor.
    # WHY: the mix ranks by acoustic vibe with no popularity floor, so it was surfacing deep cuts that fit the
    # vibe over recognisable signature hits — this restricts it to each artist's actual famous tracks.
    # Cheap floor first, then the top-N rank lookup (so the rank map is only consulted for already-loud tracks).
    # Graceful ladder so it never empties the mix: combined gate -> floor-only -> the (year-windowed) pool.
    # Selection only; _dj_order still sequences the chosen tracks sonically.
    _ag = _PROFILE_ANTHEM_GATE.get(profile_key)
    if _ag:
        _floor_n, _topn = _ag["min_listeners"], _ag["top_n"]
        def _floor_ok(rk):
            return (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _floor_n
        def _anthem(rk):
            if not _floor_ok(rk):
                return False
            r = _artist_top_rank(essentia_cache.get(rk, {}))
            return r is not None and r <= _topn
        for _pred in (_anthem, _floor_ok):              # strictest tier that can fill; else the year pool
            _h = [rk for rk in history_rks if _pred(rk)]
            _l = [rk for rk in library_rks if _pred(rk)]
            if len(_h) + len(_l) >= mix_size:
                history_rks, library_rks = _h, _l
                break

    # Geo showcase mixes: HARD origin gate — keep only tracks whose artist is from the place.
    _geo = _PROFILE_GEO_GATE.get(profile_key)
    if _geo:
        _h = [rk for rk in history_rks if _origin_match(essentia_cache.get(rk, {}), _geo)]
        _l = [rk for rk in library_rks if _origin_match(essentia_cache.get(rk, {}), _geo)]
        history_rks = _h or history_rks
        library_rks = _l or library_rks

    # City STYLE mixes (glasgow_/london_/melbourne_*): TIERED, country-bounded SELECTION. A track must be
    # IN-COUNTRY to qualify at all (matches the NATION tier) — a HARD bound with NO full-pool fallback. The
    # nation gate also kills cross-country place-name collisions (Victoria BC, Melbourne FL, London ON,
    # Glasgow KY). Among in-country tracks, rank city (tier 0) -> region (1) -> rest-of-nation (2) and
    # stable-sort each pool so the fill ladder spends city artists first, topping up outward only as needed.
    # Dual-origin artists (e.g. Jimmy Barnes: born Glasgow, based Australia) carry BOTH nations in their place
    # hierarchy, so they correctly qualify for both. Showcases are skipped (guard) so london_scene's own gate
    # is untouched. SELECTION only — _dj_order still sequences the chosen tracks sonically (order unaffected).
    _tiers = None if _is_showcase else _PROFILE_GEO_TIERS.get(profile_key)
    _tier_of = {}
    if _tiers:
        def _track_tier(rk):
            e = essentia_cache.get(rk, {})
            if not _origin_match(e, _tiers[-1]):      # nation gate: out-of-country (or unknown) -> excluded,
                return None                           # and Victoria-BC / Melbourne-FL collisions can't leak in
            for _i in range(len(_tiers) - 1):         # then the finest in-country tier: city (0) or region (1)
                if _origin_match(e, _tiers[_i]):
                    return _i
            return len(_tiers) - 1                     # in-country but neither city nor region -> nation tier
        for rk in history_rks + library_rks:          # disjoint (played vs not), so no double count
            _tt = _track_tier(rk)
            if _tt is not None:
                _tier_of[rk] = _tt
        history_rks = sorted((rk for rk in history_rks if rk in _tier_of), key=_tier_of.get)
        library_rks = sorted((rk for rk in library_rks if rk in _tier_of), key=_tier_of.get)

    # HARD exclusion — recently-played + tracks used by other dedup-eligible mixes. UNconditional:
    # never re-admitted to reach length (we widen the pool / relax the artist cap instead).
    if hard_exclude_rks:
        history_rks = [rk for rk in history_rks if rk not in hard_exclude_rks]
        library_rks = [rk for rk in library_rks if rk not in hard_exclude_rks]

    # Popularity floor — keep only well-known songs for hits-only profiles (the pop mixes), but never
    # return short: fall back to the full score-ranked pool if the floor over-thins it. Generalises the
    # decade mixes' _ERA_MIN_LISTENERS; skipped for the showcase mixes, which do their own popularity gate.
    _floor = _PROFILE_MIN_LISTENERS.get(profile_key)
    if _floor and not _is_showcase:
        def _loud(rk):
            return (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _floor
        _fh = [rk for rk in history_rks if _loud(rk)]
        _fl = [rk for rk in library_rks if _loud(rk)]
        if len(_fh) + len(_fl) >= mix_size * 2:
            history_rks, library_rks = _fh, _fl

    # Showcase mixes (decade + geo): keep the top ~3x most popular tracks of the decade/scene, then
    # pick the slate RANDOMLY (seeded per day — stable within a day, fresh across days) — variety
    # without losing the "recognisable hits" feel. The fill ladder still enforces <=1 artist + excludes.
    if _is_geo_radio:
        # Geo RADIO: a place-only radio station — most-popular CONTEMPORARY songs (released in the last
        # `recent_years`, leaning newer like a "currents" rotation) + a small throwback-classics slice,
        # 1-per-artist. Origin gate already ran above.
        cfg  = _GEO_RADIO[profile_key]
        cut  = _NOW_YEAR - cfg["recent_years"]
        base = cfg.get("recency_base", 1.0)
        _ayr = _song_year_anycopy_map(essentia_cache)   # comp-INCLUSIVE earliest year (proves "not brand-new")
        def _lis(rk): return essentia_cache.get(rk, {}).get("lastfm_listeners") or 0
        def _ay(rk):
            e = essentia_cache.get(rk, {})
            return _ayr.get(_entry_song_key(e)) or e.get("year") or 0
        # contemporary ranking leans newer: listeners * recency_base ** (years-into-window). base=1.0 -> pure pop.
        def _rscore(rk): return (_lis(rk) or 1) * (base ** max(0, _ay(rk) - cut))
        # drop comps/VA, collapse each song to its canonical copy, rank by listeners (like the era/hits branches)
        best = {}
        for rk in dict.fromkeys(history_rks + library_rks):
            e = essentia_cache.get(rk, {})
            if _era_is_comp(e):
                continue
            sk = _entry_song_key(e); sc = (_canonical_penalty(e), e.get("year") or 9999, rk)
            if sk not in best or sc < best[sk][0]:
                best[sk] = (sc, rk)
        uniq = sorted((v[1] for v in best.values()), key=lambda rk: -_lis(rk))
        # exclude score/classical artists (a radio station plays songs, not film cues)
        _by_a = defaultdict(list)
        for rk in uniq:
            _by_a[(essentia_cache.get(rk, {}).get("artist") or "").lower()].append(rk)
        _skip = {a for a, rks in _by_a.items() if _is_score_classical_artist([essentia_cache[r] for r in rks])}
        # Split each artist into a LIST of CONTEMPORARY tracks (re-rolled daily across releases for variety) + a
        # THROWBACK classic. OLD (-> throwback) if comp-inclusive year predates the cut OR the title marks a
        # re-record/reissue (_REISSUE_RE) — a radio station plays those as the old songs they are, never "current";
        # is_alt_recording keeps remix/live/extended out of contemporary too. uniq is listener-desc, so throw_by_a's
        # first-seen per artist is their top classic.
        recent_by_a, throw_by_a = defaultdict(list), {}
        for rk in uniq:
            e = essentia_cache.get(rk, {}); a = (e.get("artist") or "").lower()
            if not a or a in _skip:
                continue
            t = e.get("title") or ""; ay = _ay(rk); is_re = bool(_REISSUE_RE.search(t))
            if ay >= cut and not is_re and not meloday.is_alt_recording(t):
                recent_by_a[a].append(rk)
            elif (0 < ay < cut) or is_re:
                if a not in throw_by_a:
                    throw_by_a[a] = rk
        throw_sorted = sorted(throw_by_a.values(), key=lambda rk: -_lis(rk))           # classics: pure popularity
        cur_ord  = date.today().toordinal()
        # Soft surfacing rotation: a track the station featured in the last _RADIO_SURFACE_WINDOW days is
        # down-weighted (never banned), so the catalogue turns over while the big hits still recur and the pool
        # never thins. State persists in the geo-rotation file under "surfaced": {rk: ordinal} (pruned on load).
        _gr   = _load_geo_rotation()
        _surf = {rk: o for rk, o in (_gr.get(profile_key, {}).get("surfaced") or {}).items()
                 if isinstance(o, int) and 0 <= cur_ord - o < _RADIO_SURFACE_WINDOW}
        def _fresh(rk):   # 1.0 = not shown in the window; ramps down to _RADIO_SURFACE_FLOOR for a just-shown track
            o = _surf.get(rk)
            return 1.0 if o is None else max(_RADIO_SURFACE_FLOOR, min(1.0, (cur_ord - o) / _RADIO_SURFACE_WINDOW))
        def _album(rk):
            p = (essentia_cache.get(rk, {}).get("file_path") or "").split("/")
            return p[-2] if len(p) >= 2 else rk
        def _wchoice(items, weights, rng):
            if not items: return None
            tot = sum(weights)
            if tot <= 0: return rng.choice(items)
            r = rng.random() * tot
            for it, w in zip(items, weights):
                r -= w
                if r <= 0: return it
            return items[-1]
        def _rep_track(a):   # the artist's representative TODAY: pick a RELEASE (coverage + freshness), then a
            rng = random.Random(f"radio-rep-{cur_ord}-{profile_key}-{a}")   # track within it (top-listener, fresh)
            by_alb = defaultdict(list)
            for rk in recent_by_a[a]:
                by_alb[_album(rk)].append(rk)
            albums = list(by_alb)
            alb = _wchoice(albums, [max(_fresh(rk) for rk in by_alb[al]) for al in albums], rng)
            trks = by_alb[alb]
            return _wchoice(trks, [(_lis(rk) or 1) * _fresh(rk) for rk in trks], rng)
        # Rank ARTISTS into the pool by their ceiling score (the same big artists qualify, recency-leaning); the
        # representative TRACK is then re-rolled per day, so prolific artists rotate across all their releases.
        def _ceiling(a): return max(_rscore(rk) for rk in recent_by_a[a])
        recent_artists = sorted(recent_by_a, key=lambda a: -_ceiling(a))
        n_throw  = max(1, round(mix_size * cfg["throwback_frac"]))
        n_recent = mix_size - n_throw
        _used = set()
        def _artist(rk): return (essentia_cache.get(rk, {}).get("artist") or "").lower()
        def _pick_throw(cands, want):     # classics: fresh-weighted daily sample (rotate the classics too)
            cands = [rk for rk in cands if _artist(rk) not in _used]
            rng = random.Random(f"radio-{cur_ord}-{profile_key}-t")
            keyed = sorted(((rng.random() ** (1.0 / max((_lis(rk) or 1) * _fresh(rk), 1e-9)), rk) for rk in cands), reverse=True)
            out = []
            for _, rk in keyed[:want]:
                out.append(rk); _used.add(_artist(rk))
            return out
        def _sample_recent(artists, want):   # A-Res over ARTISTS: weight = ceiling ONLY (roster stays popularity-
            rng = random.Random(f"radio-{cur_ord}-{profile_key}-r")   # stable so the giants recur); the surfacing
            keyed = []                                                # rotation lives inside _rep_track (the track)
            for a in artists:
                if a in _used: continue
                rep = _rep_track(a)
                if rep is None: continue
                keyed.append((rng.random() ** (1.0 / max(_ceiling(a), 1e-9)), a, rep))
            keyed.sort(reverse=True)
            out = []
            for _, a, rep in keyed[:want]:
                out.append(rep); _used.add(a)
            return out
        # throwbacks first (the giants take the classic slots), then contemporary skipping those artists
        pool = _pick_throw(throw_sorted[:cfg["throwback_pool"]], n_throw) + _sample_recent(recent_artists[:cfg["recent_pool"]], n_recent)
        random.Random(f"radio-order-{cur_ord}-{profile_key}").shuffle(pool)
        # Persist: the once-a-day __built__ sentinel (gates _scene_done_today) + the surfacing log (pruned + today's
        # pool stamped at cur_ord), so tomorrow's build down-weights what we just played.
        _surf.update({rk: cur_ord for rk in pool})
        _gr[profile_key] = {"__built__": {"last": cur_ord}, "surfaced": _surf}
        _save_geo_rotation(_gr)
        history_rks, library_rks = [], pool

    elif _is_geo_hits:
        # Geo HITS (Scottish/Australian/London Hits): the hard origin gate already ran above; now select like
        # a decade mix — drop comps, collapse each song to its canonical copy, then keep the top-N by GLOBAL
        # Last.fm listeners. The Nth track's listener count IS the auto per-location floor (London's far higher
        # than Scotland's), so every track is a genuinely massive, "anyone-would-know" hit. Repetition is fine.
        if _EXCLUDE_COMPS_FROM_ERA:
            _kh = [rk for rk in history_rks if not _era_is_comp(essentia_cache.get(rk, {}))]
            _kl = [rk for rk in library_rks if not _era_is_comp(essentia_cache.get(rk, {}))]
            if _kh or _kl:
                history_rks, library_rks = _kh, _kl
        best = {}   # song_key -> (canon_penalty, year, rk) — one canonical copy per song
        for rk in dict.fromkeys(history_rks + library_rks):
            e  = essentia_cache.get(rk, {})
            sk = _entry_song_key(e)
            sc = (_canonical_penalty(e), e.get("year") or 9999, rk)
            if sk not in best or sc < best[sk][0]:
                best[sk] = (sc, rk)
        uniq = sorted((v[1] for v in best.values()),
                      key=lambda rk: -(essentia_cache.get(rk, {}).get("lastfm_listeners") or 0))
        _hyb = _GEO_HITS_HYBRID.get(profile_key) or {}
        # base = the "biggest hits" core: >= floor (deep origins like the UK) or the top-`cap` by listeners.
        if _hyb.get("floor") is not None:
            pool = [rk for rk in uniq if (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _hyb["floor"]]
        else:
            pool = [rk for rk in uniq
                    if (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _GEO_HITS_MIN_LISTENERS][:_hyb.get("cap", _GEO_HITS_POOL)]
        # Anthem-rescue: ADD each artist's OWN Last.fm top-N signatures (>= rescue) that fell below the base cut,
        # so genuine hits Last.fm under-counts on split/quoted pages aren't dropped (Bowie "Heroes"; Fleetwood
        # Mac "The Chain" 628k below London's top-200 cut). Only ADDS — the "biggest hits" base stays intact.
        _topn, _resc = _hyb.get("top_n"), _hyb.get("rescue")
        if _topn and _resc:
            _seen = set(pool)
            for rk in uniq:
                if rk in _seen:
                    continue
                e = essentia_cache.get(rk, {})
                if (e.get("lastfm_listeners") or 0) < _resc:
                    continue
                r = _artist_top_rank(e)
                if r is not None and r <= _topn:
                    pool.append(rk); _seen.add(rk)
        random.Random(f"geohits-{date.today().toordinal()}-{profile_key}").shuffle(pool)
        # Daily-built sentinel so the once-a-day _scene_done_today gate stops the hourly cron rebuilding this
        # pinned mix (mirrors the scenes, which write per-artist 'last' to the same geo-rotation file).
        _gr = _load_geo_rotation()
        _gr[profile_key] = {"__built__": {"last": date.today().toordinal()}}
        _save_geo_rotation(_gr)
        history_rks, library_rks = [], pool

    elif _is_geo:
        # Equal-weight artist-coverage rotation (scotland/australia/london scenes): every gate-matched,
        # non-score/classical artist is weighted EQUALLY and cycles in over ~ceil(artists/mix_size) days
        # (longest-unseen first) — even, fair coverage of the whole local scene, no own-depth / global-fame
        # lean — surfacing a RANDOM song from each chosen artist's top-10 by global Last.fm popularity
        # (after recently-played removal by SONG identity, so all releases/comps of a song go together).
        # WHY equal weight: a "Sounds of X" showcase should represent the local scene evenly, not skew to
        # the artists you own most or the globally-famous few.
        _sk = lambda e: "\t".join(_entry_song_key(e))
        _excl_songs = {_sk(essentia_cache[rk]) for rk in (hard_exclude_rks or ()) if rk in essentia_cache}
        by_artist   = defaultdict(list)
        for rk in dict.fromkeys(history_rks + library_rks):
            a = (essentia_cache.get(rk, {}).get("artist") or "").strip().lower()
            if a:
                by_artist[a].append(rk)

        artist_pool = {}                         # artist -> [(song_key, rk), …] top-10 by Last.fm popularity
        for a, rks in by_artist.items():
            if _is_score_classical_artist([essentia_cache[rk] for rk in rks]):
                continue
            best = {}                            # song_key -> (canon_penalty, year, rk, listeners)
            for rk in rks:
                e  = essentia_cache.get(rk, {})
                sk = _sk(e)
                if sk in _excl_songs:            # version-safe: any recently-played copy excludes the song
                    continue
                # canonical copy = least version penalty, then EARLIEST year (the original studio cut, not a
                # later re-recording/rework/reissue), then rk; carry the song's listeners for the ranking.
                cand = (_canonical_penalty(e), e.get("year") or 9999, rk, e.get("lastfm_listeners") or 0)
                if sk not in best or cand[:3] < best[sk][:3]:
                    best[sk] = cand
            if not best:                         # every distinct song recently played -> artist sits out today
                continue
            # top 10 by global Last.fm popularity (lastfm_listeners) — NOT personal play count.
            top = sorted(best.items(), key=lambda kv: -kv[1][3])[:_GEO_SONGS_PER_ARTIST]
            artist_pool[a] = [(sk, cand[2]) for sk, cand in top]

        cur_ord = date.today().toordinal()
        _scenes = _load_geo_rotation()
        _state  = _scenes.get(profile_key, {})
        _jit    = random.Random(f"geo-{cur_ord}-{profile_key}")        # deterministic tiebreak within a day

        # All artists weighted equally -> rank by rotation overdue-ness only (jitter just breaks ties), so
        # the longest-unseen cycle in first and every eligible artist gets even coverage over time.
        def _rank(a):
            return -(_geo_overdue_bonus((_state.get(a) or {}).get("last"), cur_ord)
                     + _jit.random() * 1e-3)
        chosen = sorted(artist_pool, key=_rank)[:mix_size]

        pool = []
        for a in chosen:
            shown = set((_state.get(a) or {}).get("songs") or [])
            fresh = [(sk, rk) for sk, rk in artist_pool[a] if sk not in shown]
            if not fresh:                        # whole top-10 already cycled -> reshuffle this artist
                fresh, shown = artist_pool[a], set()
            sk, rk = random.Random(f"geo-song-{cur_ord}-{profile_key}-{a}").choice(fresh)
            pool.append(rk)
            shown.add(sk)
            _state[a] = {"last": cur_ord, "songs": sorted(shown & {k for k, _ in artist_pool[a]})}
        _scenes[profile_key] = _state
        _save_geo_rotation(_scenes)

        # Seed the day's input order (deterministic per day); _dj_order then re-sequences it for smooth mixing.
        random.Random(f"geo-order-{cur_ord}-{profile_key}").shuffle(pool)
        history_rks, library_rks = [], pool

    elif _is_era:
        # Drop compilation albums (Various-Artists comps + single-artist Greatest Hits/Best Of) BEFORE the
        # collapse, so a song that also has a studio copy keeps the studio copy and a comp-only song
        # vanishes. (Comps are ~2.5% of a decade; the empty-guard keeps a decade from ever emptying.)
        if _EXCLUDE_COMPS_FROM_ERA:
            _kh = [rk for rk in history_rks if not _era_is_comp(essentia_cache.get(rk, {}))]
            _kl = [rk for rk in library_rks if not _era_is_comp(essentia_cache.get(rk, {}))]
            if _kh or _kl:
                history_rks, library_rks = _kh, _kl
        # Collapse the same song appearing on multiple albums/compilations/reissues into ONE entry,
        # keeping its most CANONICAL copy (studio/original — least live/remix/demo/instrumental
        # markers, then earliest release) so e.g. "In the End" -> the Hybrid Theory studio track, not
        # the festival/instrumental/live copies. (The 00s top-150 was only ~86 unique before this.)
        best = {}   # song_key -> (sort_score, rk)
        for rk in dict.fromkeys(history_rks + library_rks):
            e  = essentia_cache.get(rk, {})
            sk = _entry_song_key(e)
            sc = (_canonical_penalty(e), e.get("year") or 9999, rk)
            if sk not in best or sc < best[sk][0]:
                best[sk] = (sc, rk)
        uniq = sorted((v[1] for v in best.values()),
                      key=lambda rk: -(essentia_cache.get(rk, {}).get("lastfm_listeners") or 0))
        floored = [rk for rk in uniq
                   if (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _ERA_MIN_LISTENERS]
        # the recognisable canon, capped at ~3x mix_size to keep the anthems FREQUENT: a bigger pool
        # dilutes them (measured: 00s anthems fall from ~33% of days at 150 to ~13% at the full ~1900).
        # Use the floor only if it still yields enough DISTINCT ARTISTS to fill one-per-artist —
        # otherwise (a small scene like Scotland, where the floor leaves a few prolific artists) fall
        # back to the full popularity-ranked canon so the mix stays artist-diverse, not Snow Patrol x15.
        def _n_artists(rks):
            return len({(essentia_cache.get(rk, {}).get("artist") or "").lower() for rk in rks if rk})
        use_floor = len(floored) >= mix_size and _n_artists(floored[:mix_size * 3]) >= mix_size
        pool = (floored if use_floor else uniq)[:mix_size * 3]
        # Anthem-rescue: ADD each artist's own Last.fm top-N (>= _ERA_RESCUE_MIN) that fell below the cut, so
        # genuine decade classics the tight top-150 drops aren't lost. Only adds from `uniq` (already in-decade
        # + comp/VA-filtered), so every rescued track passes the SAME decade gate as the base — never out-of-decade.
        _seen = set(pool)
        for rk in uniq:
            if rk in _seen:
                continue
            e = essentia_cache.get(rk, {})
            if (e.get("lastfm_listeners") or 0) < _ERA_RESCUE_MIN:
                continue
            r = _artist_top_rank(e)
            if r is not None and r <= _ERA_RESCUE_TOP_N:
                pool.append(rk); _seen.add(rk)
        random.Random(f"decade-{date.today().toordinal()}-{profile_key}").shuffle(pool)
        history_rks, library_rks = [], pool

    n_history = min(int(mix_size * 0.40), len(history_rks))
    n_library = mix_size - n_history
    # Wide-but-bounded candidate pool (8x target) so the artist ladder can prefer distinct artists
    # while still filling to mix_size; the slice is from the already-hard-filtered, score-sorted rks.
    candidate_rks = list(dict.fromkeys(
        history_rks[:n_history * 8] + library_rks[:n_library * 8]
    ))
    track_map  = resolve_tracks_by_keys(plex, candidate_rks)
    seen_songs   = set()       # shared across history + library (in-mix song dedup)
    artist_count = Counter()   # shared across history + library (in-mix artist cap)
    BASE_ARTIST_LIMIT = 1      # <=1 track per artist per mix, relaxed only to reach length

    def _take(rks, want, artist_limit, out):
        for rk in rks:
            if len(out) >= want:
                break
            t = track_map.get(rk)
            if not t or is_low_rated(t):
                continue
            if str(getattr(t, "parentRatingKey", "")) in excluded_album_keys:
                continue
            sk = _song_key(t)
            if sk in seen_songs:
                continue
            ak = _artist_key(t)
            if artist_count[ak] >= artist_limit:
                continue
            seen_songs.add(sk)
            artist_count[ak] += 1
            out.append(t)

    def _fill(out, rks, want):
        # Prefer 1 per artist; relax the cap (2 -> 3 -> unlimited) only as far as needed to reach
        # `want`. Hard excludes / album / rating / song-dedup are NEVER relaxed.
        for lim in (BASE_ARTIST_LIMIT, BASE_ARTIST_LIMIT + 1, BASE_ARTIST_LIMIT + 2, mix_size):
            if len(out) >= want:
                break
            _take(rks, want, lim, out)

    # Loved-track lean: within the already-resolved candidate pool, let tracks you really like (>=3.5★)
    # sort as if acoustically closer so they get PICKED (not just re-ordered later) when they fit the mood.
    # Rare (<2% of library) + still gated by fit / artist-cap / recently-played, so discovery holds.
    if not _is_showcase:
        def _eff_score(rk):
            t = track_map.get(rk)
            return _combined_score(rk) - _rating_dist_bonus(getattr(t, "userRating", None) if t else None)
        # City mixes: tier stays the PRIMARY key (city -> region -> nation); acoustic + loved-track lean order
        # WITHIN each tier. Selection order only — _dj_order re-sequences the final picks sonically.
        _key = (lambda rk: (_tier_of[rk], _eff_score(rk))) if _tier_of else _eff_score
        history_rks = sorted(history_rks[:n_history * 8], key=_key)
        library_rks = sorted(library_rks[:n_library * 8], key=_key)

    if profile_key in _BALANCED_PROFILES and not _is_showcase:
        # 50/50 balance: ~half ANCHORS — tracks you'll KNOW (played, or popular on Last.fm) or already
        # LIKE (rating-boosted via _eff_score) — + ~half fame-blind best-FIT discovery (unplayed, below the
        # recognisability floor). Anchor share is capped by availability, so a thin-history niche genre
        # auto-skews to more discovery. (A balanced profile carries no popularity lean, so the discovery
        # half ranks purely on acoustic fit.)
        _hist_set = set(history_rks)
        def _is_known(rk):
            return rk in _hist_set or (essentia_cache.get(rk, {}).get("lastfm_listeners") or 0) >= _ANCHOR_LISTENERS
        anchor_rks    = sorted({rk for rk in history_rks + library_rks if _is_known(rk)}, key=_eff_score)
        discovery_rks = [rk for rk in library_rks if not _is_known(rk)]
        history_tracks = []
        _fill(history_tracks, anchor_rks, round(mix_size * _ANCHOR_RATIO))
        library_tracks = []
        _fill(library_tracks, discovery_rks, mix_size - len(history_tracks))
    else:
        history_tracks = []
        _fill(history_tracks, history_rks, n_history)
        library_tracks = []
        _fill(library_tracks, library_rks, mix_size - len(history_tracks))   # library covers any history shortfall

    combined = history_tracks + library_tracks
    if not _is_showcase:                   # showcases (decade + geo) are rotation/popularity-chosen, not score-ranked
        combined.sort(key=lambda t: (
            _combined_score(str(t.ratingKey))
            - _rating_dist_bonus(getattr(t, "userRating", None))
        ))
    combined = combined[:mix_size]
    if not _is_era:                        # DJ flow: re-sequence for smooth mixing — incl. the geo scenes
        # (varied music benefits most); only the decade mixes keep their day-seeded shuffle (no DJ re-order).
        combined = _dj_order(combined, essentia_cache, arc=(profile_key in _DJ_ARC_PROFILES))
    return combined


# ===========================================================================
# Orchestrator
# ===========================================================================

def _run_playlist(playlist_id, plex, music, ec, history, centroid, excluded_album_keys,
                  existing_playlists=None, reselect_moods=False, time_context=False,
                  weather_context=False):
    ep = existing_playlists  # shorthand

    if playlist_id == "on_repeat":
        tracks = build_on_repeat(plex, history, excluded_album_keys)
        _upsert_extras_playlist(plex, "On Repeat • Meloday+", tracks,
            _pick_description("on_repeat"),
            cover_key="on_repeat", cover_title="On Repeat",
            existing_playlists=ep)

    elif playlist_id == "repeat_rewind":
        tracks = build_repeat_rewind(plex, history, excluded_album_keys)
        _upsert_extras_playlist(plex, "Repeat Rewind • Meloday+", tracks,
            _pick_description("repeat_rewind"),
            cover_key="repeat_rewind", cover_title="Repeat Rewind",
            existing_playlists=ep)

    elif playlist_id == "release_radar":
        tracks = build_release_radar(plex, music, ec, centroid, excluded_album_keys,
                                     history_entries=history)
        _upsert_extras_playlist(plex, "Release Radar • Meloday+", tracks,
            _pick_description("release_radar"),
            cover_key="release_radar", cover_title="Release Radar",
            cover_tracks=tracks,
            existing_playlists=ep)

    elif playlist_id == "discover_weekly":
        tracks = build_discover_weekly(plex, history, ec, centroid, excluded_album_keys,
                                       target=DISCOVER_WEEKLY_SIZE)
        _upsert_extras_playlist(plex, "Discover Weekly • Meloday+", tracks,
            _pick_description("discover_weekly"),
            cover_key="discover_weekly", cover_title="Discover Weekly",
            existing_playlists=ep)

    elif playlist_id == "daily_mixes":
        mixes = build_daily_mixes(plex, history, ec, excluded_album_keys,
                                  n_mixes=DAILY_MIX_COUNT)
        for name, _desc, tracks, styles_subtitle in mixes:
            # "Daily Mix 3 • Meloday+" → extract "3"
            mix_num = name.split("•")[0].strip().split()[-1]
            styles_list = [s.strip() for s in styles_subtitle.split("·") if s.strip()] \
                          if styles_subtitle else []
            cover_styles = styles_list[:2]
            cover_sub = " · ".join(cover_styles) if cover_styles else None
            _upsert_extras_playlist(plex, name, tracks,
                _daily_mix_description(styles_list),
                cover_key=f"daily_mix_{mix_num}",
                cover_title=f"Daily Mix {mix_num}",
                cover_subtitle=cover_sub,
                cover_tracks=tracks,
                existing_playlists=ep)

    elif playlist_id == "rediscovery":
        tracks = build_rediscovery(plex, history, excluded_album_keys)
        _upsert_extras_playlist(plex, "Rediscovery • Meloday+", tracks,
            _pick_description("rediscovery"),
            cover_key="rediscovery", cover_title="Rediscovery",
            existing_playlists=ep)

    elif playlist_id == "time_capsule":
        tracks, era_label = build_time_capsule(plex, music, history, ec, excluded_album_keys)
        _upsert_extras_playlist(plex, "Time Capsule • Meloday+", tracks,
            _pick_description("time_capsule", era=era_label or "your past"),
            cover_key="time_capsule", cover_title="Time Capsule",
            cover_subtitle=era_label, existing_playlists=ep)

    elif playlist_id == "time_machine":
        tracks, label = build_time_machine(plex, music, history, ec, excluded_album_keys)
        if tracks and label:
            name = f"{label} • Meloday+"
            for ptitle, pl in (ep or {}).items():     # rotating title — clear the prior month-year playlist
                if ptitle != name and _TIME_MACHINE_TITLE_RE.match(ptitle):
                    try:
                        pl.delete(); xlog(f"[OK] Removed stale Time Machine: '{ptitle}'")
                    except Exception as e:
                        xlog(f"[WARN] Could not remove '{ptitle}': {e}")
            _upsert_extras_playlist(plex, name, tracks,
                _pick_description("time_machine", era=label),
                cover_key="time_machine", cover_title=label,
                existing_playlists=ep)

    elif playlist_id == "deep_cuts":
        tracks = build_deep_cuts(plex, history, ec, excluded_album_keys)
        _upsert_extras_playlist(plex, "Deep Cuts • Meloday+", tracks,
            _pick_description("deep_cuts"),
            cover_key="deep_cuts", cover_title="Deep Cuts",
            existing_playlists=ep)

    elif playlist_id == "top_songs":
        results = build_top_songs(plex, history, excluded_album_keys,
                                  existing_playlists=ep)
        for year, tracks in results:
            year_str = str(year)
            _upsert_extras_playlist(plex, f"Top Songs {year_str} • Meloday+", tracks,
                _pick_description("top_songs", era=year_str),
                cover_key=f"top_songs_{year_str}", cover_title=f"Top Songs {year_str}",
                cover_subtitle=None,
                existing_playlists=ep)

    elif playlist_id == "all_time_favourites":
        tracks = build_all_time_favourites(music, excluded_album_keys)
        _upsert_extras_playlist(plex, "All-Time Favourites • Meloday+", tracks,
            _pick_description("all_time_favourites"),
            cover_key="all_time_favourites", cover_title="All-Time Favourites",
            existing_playlists=ep)

    elif playlist_id == "mood_mixes":
        n_active = int(_extras.get("mood_mix_count", 5))
        mixes, to_remove = build_mood_mixes(
            plex, history, ec, excluded_album_keys, n_active=n_active,
            reselect=reselect_moods, time_context=time_context,
            weather_context=weather_context, existing_playlists=ep)
        for profile_key in to_remove:
            name = _MOOD_MIX_NAMES[profile_key]
            pl   = (ep or {}).get(name)
            if pl is not None:          # identity check — `if pl:` calls len()->items() (network; can 404)
                try:
                    pl.delete()
                    xlog(f"[OK] Removed: '{name}'")
                except Exception as e:
                    xlog(f"[WARN] Could not remove '{name}': {e}")
        for name, profile_key, tracks in mixes:
            _upsert_extras_playlist(plex, name, tracks,
                _pick_description(profile_key),
                cover_key=profile_key,
                cover_title=name.replace(" • Meloday+", ""),
                existing_playlists=ep)


def main():
    args = _parse_args()
    to_run = PLAYLIST_IDS if args.playlist == "all" else [args.playlist]

    xlog("=== Meloday Extras ===")

    plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=120)
    music = plex.library.section(MUSIC_LIBRARY)
    xlog(f"[OK] Plex: {plex.friendlyName}")

    load_essentia_cache()
    ec = meloday._essentia_cache
    xlog(f"[OK] Essentia cache: {len(ec)} entries")

    excluded_album_keys = build_excluded_album_keys(music)
    xlog(f"[OK] Excluded albums: {len(excluded_album_keys)}")

    # Pre-fetch existing playlists BEFORE the lookback calculation so that
    # top_songs can skip years that are already generated.
    existing_playlists = {
        pl.title: pl
        for pl in plex.playlists(title="Meloday")
        if getattr(pl, "title", "").endswith("• Meloday+")
    }
    xlog(f"[OK] Found {len(existing_playlists)} existing Meloday+ playlist(s)")

    def _top_songs_lookback():
        """
        First run: fetch from Jan 1 of top_songs_start_year to build all years.
        Subsequent runs: fetch from Jan 1 of the current year — data for past years
        is already locked in their playlists.
        Grace period (first 60 days of new year): extend back to Jan 1 of the
        previous year so that plays from the final days of December (after the last
        run of the year) are included in the previous year's final regeneration.
        """
        current_year = datetime.now(tz=timezone.utc).year
        start_year   = int(_extras.get("top_songs_start_year", current_year - 5))

        has_past_year = any(
            f"Top Songs {yr} • Meloday+" in existing_playlists
            for yr in range(start_year, current_year)
        )
        if has_past_year:
            grace_cutoff = datetime(current_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=60)
            if datetime.now(tz=timezone.utc) < grace_cutoff:
                # Extend into the previous year to capture its final days of December
                jan1 = datetime(current_year - 1, 1, 1, tzinfo=timezone.utc)
            else:
                jan1 = datetime(current_year, 1, 1, tzinfo=timezone.utc)
            return (datetime.now(tz=timezone.utc) - jan1).days + 1

        # First run — fetch the full range so all years with data get playlists.
        jan1 = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - jan1).days + 1

    time_context_mode    = getattr(args, "time_context", False)
    weather_context_mode = getattr(args, "weather_context", False)

    def _lookback_for(pid):
        if pid == "top_songs":
            return _top_songs_lookback()
        if pid == "all_time_favourites":
            return 0   # uses track.viewCount — no history needed
        if pid == "mood_mixes" and (time_context_mode or weather_context_mode):
            return 30  # only needs recent plays for mix content weighting
        return _HISTORY_LOOKBACK_DAYS.get(pid, 180)

    lookback_days = max((_lookback_for(pid) for pid in to_run), default=0)
    if lookback_days > 0:
        xlog(f"[...] Fetching history ({lookback_days} days)...")
        history = fetch_full_history(music, lookback_days=lookback_days)
        xlog(f"[OK] History: {len(history)} entries")
    else:
        history = []
        xlog("[OK] History: not needed for selected playlists")

    needs_centroid = bool(set(to_run) & _CENTROID_PLAYLISTS)
    if needs_centroid:
        centroid = compute_listening_centroid(history, ec)
        if centroid.get("bpm"):
            xlog(f"[OK] Centroid: bpm={centroid['bpm']:.0f}  energy={centroid['energy']:.1f}  "
                 f"dance={centroid['danceability']:.2f}  bright={centroid['brightness']:.2f}")
        else:
            xlog("[OK] Centroid computed (limited acoustic data)")
    else:
        centroid = {}

    reselect_moods = getattr(args, "reselect_moods", False)

    for playlist_id in to_run:
        xlog(f"\n[...] Building: {playlist_id}")
        try:
            _run_playlist(playlist_id, plex, music, ec, history, centroid, excluded_album_keys,
                          existing_playlists=existing_playlists,
                          reselect_moods=reselect_moods,
                          time_context=time_context_mode,
                          weather_context=weather_context_mode)
        except Exception:
            xlog(f"[ERROR] {playlist_id} failed:\n{traceback.format_exc()}")

    xlog("\n=== Done ===")


if __name__ == "__main__":
    main()
