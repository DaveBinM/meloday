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
import sys
import re
import math
import random
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
    clean_title,
    track_artist_name,
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

_file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
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
RELEASE_RADAR_MIN_TRACKS  = int(_extras.get("release_radar_min_tracks", 30))
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
    "daily_mixes", "rediscovery", "time_capsule", "deep_cuts",
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
    "time_capsule":       548,    # 18 months for reliable peak-year inference
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
    "deep_cuts":          (( 50,  65, 110), ( 20,  30,  65)),  # slate → near-black blue
    # Top Songs / All-Time
    "top_songs":          ((200, 160,  30), (140,  90,  10)),  # warm gold
    "all_time_favourites":((220, 180,  20), (160, 120,  10)),  # bright gold
    # Mood / Activity Mixes (12 profiles)
    "workout":            ((220,  50,  50), (160,  20,  20)),  # energetic red
    "running":            ((220, 100,  30), (160,  60,  10)),  # orange
    "party":              ((220,  80, 180), (150,  20, 120)),  # vivid magenta
    "happy":              ((220, 190,  20), (170, 130,  10)),  # bright yellow
    "morning":            ((240, 150,  50), (190,  90,  20)),  # warm peach
    "focus":              (( 50,  80, 160), ( 20,  40, 100)),  # cool blue
    "dinner":             ((120,  50, 140), ( 70,  20,  90)),  # warm purple
    "chill":              (( 40, 150, 130), ( 20,  80,  80)),  # teal
    "rainy_day":          (( 80, 110, 160), ( 40,  60, 110)),  # grey-blue
    "melancholy":         (( 60,  50, 130), ( 30,  20,  90)),  # muted indigo
    "late_night":         (( 40,  20,  80), ( 20,  10,  50)),  # deep purple
    "sleep":              (( 30,  30, 100), ( 10,  10,  60)),  # deep navy
    "sunny":              ((255, 200,  30), (220, 140,  10)),  # bright sunshine yellow
    "cosy":               ((140, 190, 220), ( 80, 130, 175)),  # icy blue
    # --- Mood / Emotional ---
    "nostalgia_mix":      ((190, 145,  85), (120,  75,  35)),  # warm sepia-rose
    "dreamy_mix":         ((160, 120, 200), ( 80,  60, 140)),  # soft lavender
    "moody_mix":          (( 55,  55, 110), ( 25,  25,  70)),  # dark slate blue
    "emotional":          ((180,  70, 110), (110,  30,  65)),  # deep rose-mauve
    "bittersweet":        ((200, 130,  60), (130,  70,  25)),  # warm amber-rust
    "cathartic":          ((180,  40,  80), (110,  15,  45)),  # crimson-burgundy
    "confidence_boost":   ((220, 180,  40), (160, 120,  15)),  # electric gold-brass
    "empowering":         ((170,  50, 220), (100,  20, 160)),  # bright violet
    "euphoric":           ((240,  80, 140), (170,  20,  90)),  # hot pink-coral
    "angst_mix":          ((150,  30,  30), ( 80,  10,  10)),  # dark red-charcoal
    "romantic_mix":       ((210, 120, 150), (140,  65,  90)),  # soft rose-dusty pink
    "daydreaming":        ((110, 160, 210), ( 60, 100, 160)),  # pale blue-powder
    "fresh_start":        ((100, 200, 150), ( 40, 130,  90)),  # mint-sage green
    # --- Aesthetic / Time-of-Day ---
    "main_character":     (( 40,  50,  90), ( 15,  20,  55)),  # dramatic navy
    "golden_hour":        ((230, 170,  60), (170, 100,  20)),  # warm gold-amber
    "sunset_mix":         ((220, 110,  80), (160,  55,  40)),  # coral-orange-pink
    "after_dark":         (( 30,  20,  70), ( 10,   5,  40)),  # deep blue-black
    # --- Time / Occasion ---
    "after_work":         ((180, 155, 100), (110,  90,  50)),  # warm khaki-neutral
    "friday_night":       (( 40,  80, 200), ( 20,  40, 130)),  # electric blue-navy
    "weekend_mix":        ((110, 150, 220), ( 60,  90, 170)),  # sky blue-periwinkle
    "sunday_morning":     ((230, 200, 130), (175, 140,  75)),  # warm cream-yellow
    "lazy_sunday":        ((190, 155, 165), (120,  90, 100)),  # dusty rose-soft
    "brunch_mix":         ((140, 200, 100), ( 80, 140,  50)),  # fresh green-lime
    "date_night":         ((120,  30,  70), ( 70,  10,  40)),  # deep wine-burgundy
    # --- Activity ---
    "driving_mix":        (( 70,  90, 130), ( 35,  50,  85)),  # road grey-asphalt blue
    "night_drive":        (( 20,  60,  90), ( 10,  30,  60)),  # deep teal-midnight
    "driving_singalong":  (( 60, 140, 220), ( 25,  75, 160)),  # bright sky blue
    "road_trip":          ((210, 140,  60), (150,  85,  25)),  # desert orange-sandy
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
    "late_night_romance": (( 25,  20,  65), ( 10,   8,  40)),  # deep midnight blue
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
    "acoustic_romance":   ((175, 130,  70), (110,  75,  30)),  # warm wood-honey
    "indie_romance":      ((100, 140, 110), ( 55,  85,  65)),  # muted sage-dusty green
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


# Background style per cover key — (style_name, variant_int) or plain string (v=0).
# 13 style families: geometric, circles, radial, waves, floating_circles, rays,
#   arc_sweep, aurora, triangles, diamond, starburst, chevrons, spiral
_COVER_BG_STYLES = {
    # Non-mood extras — distinct styles, default variant
    "on_repeat":           ("floating_circles", 0),  # circular/looping
    "repeat_rewind":       ("triangles",        0),  # angular rewind feel
    "rediscovery":         ("aurora",           0),  # dreamy re-emergence
    "deep_cuts":           ("waves",            0),  # deep layered bands
    "all_time_favourites": ("rays",             0),  # radiating gold star
    "time_capsule":        ("circles",          0),  # concentric radar rings
    "discover_weekly":     ("radial",           0),  # warm centre glow
    # ----- ENERGY -----
    "workout":          ("starburst",       0),  # 14-ray tight centre burst
    "running":          ("geometric",       1),  # flat 12° speed strips
    "empowering":       ("rays",            1),  # 11 rays rising from bottom
    "confidence_boost": ("geometric",       2),  # 3 bold strips at 38°
    "cathartic":        ("triangles",       1),  # downward-pointing release
    "angst_mix":        ("rays",            2),  # tight corner rays, 60° spread
    "celebration":      ("floating_circles",1),  # confetti scattered circles
    "euphoric":         ("starburst",       1),  # 18 rays, peak-bliss burst
    # ----- CALM -----
    "chill":            ("waves",           1),  # 3 wide gentle rolling waves
    "sleep":            ("waves",           2),  # 2 barely-visible flat bands
    "daydreaming":      ("aurora",          1),  # 5 slow dreamy ribbons
    "lazy_sunday":      ("floating_circles",2),  # 4 large widely-spaced circles
    "sunday_morning":   ("radial",          1),  # warm glow from lower centre
    "deep_work":        ("circles",         4),  # 12 tight focus rings
    "evening_unwind":   ("arc_sweep",       1),  # arcs from left (unwinding)
    "folk_acoustic":    ("geometric",       3),  # simple 3 strips at 20°
    "melancholy":       ("waves",           3),  # 6 narrow tight sad waves
    "emotional":        ("aurora",          2),  # 8 intense fast ribbons
    # ----- UPBEAT -----
    "happy":            ("floating_circles",0),  # 7 classic joyful circles
    "sunny":            ("starburst",       2),  # 16-ray sun
    "beach_vibes":      ("waves",           8),  # 4 rolling ocean waves
    "fresh_start":      ("chevrons",        0),  # 4 forward V-shapes
    "spring_mix":       ("floating_circles",3),  # many small buds, upper half
    "brunch_mix":       ("geometric",       4),  # 4 airy light strips at 25°
    "weekend_mix":      ("arc_sweep",       2),  # arcs sweeping from top
    "cooking_mix":      ("spiral",          0),  # 9-arc clockwise spiral
    "summer_evening":   ("aurora",          3),  # 4 wide slow warm ribbons
    # ----- ROMANTIC -----
    "romantic_mix":     ("arc_sweep",       0),  # classic right-sweep arcs
    "modern_romance":   ("chevrons",        1),  # 6 contemporary V-shapes
    "slow_dance":       ("spiral",          1),  # 8-arc counter-clockwise
    "love_songs":       ("waves",           4),  # 5 lyrical flowing waves
    "first_date":       ("floating_circles",4),  # nervous scattered circles
    "acoustic_romance": ("geometric",       5),  # 2 minimal gentle strips
    "indie_romance":    ("triangles",       2),  # 3 asymmetric alt triangles
    "late_night_romance":("circles",        2),  # off-centre left rings
    "piano_romance":    ("radial",          3),  # centred spotlight glow
    "strings_romance":  ("aurora",          4),  # 6 classical flow ribbons
    "string_quartet":   ("diamond",         1),  # 6 diamonds rotated 18°
    # ----- ATMOSPHERIC -----
    "golden_hour":      ("aurora",          5),  # 4 wide slow golden bands
    "sunset_mix":       ("radial",          2),  # glow from horizon bottom
    "autumn_mix":       ("triangles",       3),  # falling leaf triangles
    "winter_mix":       ("circles",         3),  # 7 rings from upper centre
    "cosy":             ("waves",           7),  # 3 enveloping warm waves
    "main_character":   ("rays",            3),  # 8 rays, top-centre spotlight
    "morning":          ("starburst",       3),  # 10-ray gentle sunrise
    # ----- DRIVING / ACTIVITY -----
    "driving_mix":      ("triangles",       5),  # forward-pointing arrows
    "night_drive":      ("rays",            5),  # 7 rays from left (headlights)
    "driving_singalong":("floating_circles",5),  # carefree weighted-right
    "road_trip":        ("chevrons",        2),  # 8 highway arrow chevrons
    "commute_mix":      ("geometric",       6),  # 6 narrow regular strips
    "walking_mix":      ("waves",           5),  # 5 steady even-spaced waves
    # ----- DINNER / EVENING -----
    "dinner":           ("diamond",         0),  # 6 classic diamonds
    "jazz_dinner":      ("spiral",          2),  # 12-arc tight jazz spiral
    "romantic_jazz":    ("arc_sweep",       4),  # smooth bottom-curve sweep
    "candlelight":      ("radial",          4),  # candle glow just below centre
    "date_night":       ("diamond",         2),  # 8 tighter diamonds
    "romantic_dinner":  ("aurora",          6),  # 5 warm atmospheric ribbons
    # ----- FOCUS / MOOD -----
    "focus":            ("circles",         1),  # 14 tight centred focus rings
    "dreamy_mix":       ("aurora",          7),  # 7 rapid dreamy ribbons
    "moody_mix":        ("geometric",       7),  # 4 heavy dark strips at 30°
    "bittersweet":      ("waves",           6),  # 4 irregular asymmetric waves
    "heartbreak":       ("triangles",       4),  # shattered small triangles
    "rainy_day":        ("radial",          6),  # diffuse glow from top
    # ----- MEMORY / LATE NIGHT -----
    "nostalgia_mix":    ("circles",         5),  # off-centre memory rings
    "synthpop_romance": ("geometric",       8),  # steep 52° 80s strips
    "late_night":       ("radial",          5),  # dark ambient off-centre glow
    # ----- PARTY / SOCIAL -----
    "party":            ("starburst",       4),  # 22-ray max party burst
    "friday_night":     ("rays",            4),  # 10 rays, corner night energy
    "pre_party":        ("floating_circles",6),  # building anticipation circles
    "party_throwback":  ("circles",         6),  # off-centre right throwback
    "after_dark":       ("diamond",         3),  # 3 large bold club diamonds
    "after_work":       ("triangles",       6),  # sharp vertical slash
    "cool_down":        ("waves",           9),  # 4 measured cooling waves
}

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
_TOP_SONGS_BG_CYCLE = [
    "geometric", "waves", "floating_circles", "rays", "aurora",
    "circles",   "radial", "triangles",       "arc_sweep", "diamond",
    "geometric", "waves", "floating_circles", "rays", "aurora",
    "circles",   "radial", "triangles",       "arc_sweep", "diamond",
    "geometric", "waves", "floating_circles", "rays", "aurora",
    "circles",   "radial", "triangles",       "arc_sweep", "diamond",
]

# _MOOD_PROFILE_KEYS is defined after _MOOD_MIX_NAMES below.


# ---------------------------------------------------------------------------
# Playlist descriptions — picked randomly each run for variety
# ---------------------------------------------------------------------------
_DESCRIPTIONS = {
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
}


def _pick_description(playlist_id, era=None, styles=None):
    """Pick a random description for the given playlist, formatting any placeholders."""
    pool = _DESCRIPTIONS.get(playlist_id, [])
    if not pool:
        return ""
    desc = random.choice(pool)
    if era and "{era}" in desc:
        desc = desc.replace("{era}", era)
    return desc


# ---------------------------------------------------------------------------
# Mood / Activity Mix profiles — acoustic target fingerprints
# ---------------------------------------------------------------------------
_MOOD_PROFILES = {
    # General activity / mood profiles (rotation by acoustic fit)
    # beat_confidence: groove/pulse strength (0=loose, 1=driving)
    # onset_rate: note density in onsets/sec
    # dynamic_complexity: loudness variance (low=compressed, high=dynamic; 0.3=EDM, 0.6=folk/jazz)
    # arousal/valence/vocal_presence: TF-derived — None-safe, only scored when data exists
    "workout":    {"bpm": 150, "energy": -7,  "danceability": 0.60, "brightness": 0.30,
                   "beat_confidence": 0.85, "onset_rate": 7.0, "dynamic_complexity": 0.40,
                   "arousal": 0.80, "valence": 0.65, "vocal_presence": 0.60},
    "running":    {"bpm": 160, "energy": -6,  "danceability": 0.45, "brightness": 0.28,
                   "beat_confidence": 0.80, "onset_rate": 6.0, "dynamic_complexity": 0.42,
                   "arousal": 0.85, "valence": 0.60, "vocal_presence": 0.55},
    "party":      {"bpm": 125, "energy": -9,  "danceability": 0.78, "brightness": 0.38,
                   "beat_confidence": 0.75, "onset_rate": 6.5, "dynamic_complexity": 0.35,
                   "arousal": 0.75, "valence": 0.80, "vocal_presence": 0.70},
    "happy":      {"bpm": 118, "energy": -11, "danceability": 0.65, "brightness": 0.48,
                   "beat_confidence": 0.65, "onset_rate": 5.5, "dynamic_complexity": 0.45,
                   "arousal": 0.65, "valence": 0.85, "vocal_presence": 0.75},
    "focus":      {"bpm":  90, "energy": -18, "danceability": 0.22, "brightness": 0.10,
                   "beat_confidence": 0.30, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                   "arousal": 0.25, "valence": 0.55, "vocal_presence": 0.15},
    "chill":      {"bpm":  82, "energy": -15, "danceability": 0.32, "brightness": 0.16,
                   "beat_confidence": 0.45, "onset_rate": 3.0, "dynamic_complexity": 0.55,
                   "arousal": 0.30, "valence": 0.65, "vocal_presence": 0.45},
    "melancholy": {"bpm":  68, "energy": -15, "danceability": 0.15, "brightness": 0.07,
                   "beat_confidence": 0.35, "onset_rate": 2.5, "dynamic_complexity": 0.60,
                   "arousal": 0.25, "valence": 0.20, "vocal_presence": 0.65},
    # Time-of-day profiles (boosted when current time matches their window)
    "morning":    {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.45,
                   "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.55,
                   "arousal": 0.50, "valence": 0.70, "vocal_presence": 0.55},
    "dinner":     {"bpm":  88, "energy": -19, "danceability": 0.25, "brightness": 0.22,
                   "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.65,
                   "arousal": 0.25, "valence": 0.60, "vocal_presence": 0.40},
    "late_night": {"bpm":  78, "energy": -14, "danceability": 0.42, "brightness": 0.05,
                   "beat_confidence": 0.50, "onset_rate": 3.5, "dynamic_complexity": 0.52,
                   "arousal": 0.40, "valence": 0.35, "vocal_presence": 0.50},
    "sleep":      {"bpm":  65, "energy": -23, "danceability": 0.10, "brightness": 0.03,
                   "beat_confidence": 0.15, "onset_rate": 1.0, "dynamic_complexity": 0.70,
                   "arousal": 0.10, "valence": 0.50, "vocal_presence": 0.10},
    # Weather-triggered profiles (boosted when conditions match)
    "rainy_day":  {"bpm":  72, "energy": -16, "danceability": 0.18, "brightness": 0.09,
                   "beat_confidence": 0.30, "onset_rate": 2.0, "dynamic_complexity": 0.62,
                   "arousal": 0.25, "valence": 0.30, "vocal_presence": 0.60},
    "sunny":      {"bpm": 108, "energy": -11, "danceability": 0.58, "brightness": 0.52,
                   "beat_confidence": 0.65, "onset_rate": 5.0, "dynamic_complexity": 0.48,
                   "arousal": 0.70, "valence": 0.85, "vocal_presence": 0.65},
    "cosy":       {"bpm":  75, "energy": -16, "danceability": 0.20, "brightness": 0.18,
                   "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.62,
                   "arousal": 0.25, "valence": 0.65, "vocal_presence": 0.45},
    # ----------------------------------------------------------------
    # Mood / Emotional
    # ----------------------------------------------------------------
    "nostalgia_mix":    {"bpm":  88, "energy": -14, "danceability": 0.28, "brightness": 0.15,
                         "beat_confidence": 0.35, "onset_rate": 2.5, "dynamic_complexity": 0.58,
                         "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.70},
    "dreamy_mix":       {"bpm":  85, "energy": -18, "danceability": 0.25, "brightness": 0.12,
                         "beat_confidence": 0.25, "onset_rate": 2.0, "dynamic_complexity": 0.60,
                         "arousal": 0.20, "valence": 0.55, "vocal_presence": 0.50},
    "moody_mix":        {"bpm":  78, "energy": -16, "danceability": 0.22, "brightness": 0.08,
                         "beat_confidence": 0.40, "onset_rate": 2.8, "dynamic_complexity": 0.58,
                         "arousal": 0.35, "valence": 0.30, "vocal_presence": 0.60},
    "emotional":        {"bpm":  95, "energy": -12, "danceability": 0.35, "brightness": 0.25,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.65,
                         "arousal": 0.50, "valence": 0.45, "vocal_presence": 0.85},
    "bittersweet":      {"bpm":  82, "energy": -15, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.35, "valence": 0.45, "vocal_presence": 0.70},
    "cathartic":        {"bpm": 108, "energy": -10, "danceability": 0.38, "brightness": 0.22,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.70,
                         "arousal": 0.70, "valence": 0.40, "vocal_presence": 0.80},
    "confidence_boost": {"bpm": 118, "energy":  -9, "danceability": 0.58, "brightness": 0.35,
                         "beat_confidence": 0.70, "onset_rate": 5.5, "dynamic_complexity": 0.40,
                         "arousal": 0.68, "valence": 0.72, "vocal_presence": 0.70},
    "empowering":       {"bpm": 128, "energy":  -8, "danceability": 0.50, "brightness": 0.30,
                         "beat_confidence": 0.72, "onset_rate": 5.5, "dynamic_complexity": 0.55,
                         "arousal": 0.75, "valence": 0.70, "vocal_presence": 0.75},
    "euphoric":         {"bpm": 132, "energy":  -8, "danceability": 0.72, "brightness": 0.42,
                         "beat_confidence": 0.80, "onset_rate": 6.5, "dynamic_complexity": 0.35,
                         "arousal": 0.85, "valence": 0.90, "vocal_presence": 0.65},
    "angst_mix":        {"bpm": 138, "energy":  -9, "danceability": 0.40, "brightness": 0.22,
                         "beat_confidence": 0.65, "onset_rate": 6.0, "dynamic_complexity": 0.55,
                         "arousal": 0.78, "valence": 0.25, "vocal_presence": 0.82},
    "romantic_mix":     {"bpm":  88, "energy": -16, "danceability": 0.28, "brightness": 0.20,
                         "beat_confidence": 0.32, "onset_rate": 2.5, "dynamic_complexity": 0.60,
                         "arousal": 0.28, "valence": 0.70, "vocal_presence": 0.72},
    "daydreaming":      {"bpm":  80, "energy": -19, "danceability": 0.20, "brightness": 0.10,
                         "beat_confidence": 0.22, "onset_rate": 1.8, "dynamic_complexity": 0.62,
                         "arousal": 0.18, "valence": 0.60, "vocal_presence": 0.55},
    "fresh_start":      {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.40,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.78, "vocal_presence": 0.65},
    # ----------------------------------------------------------------
    # Aesthetic / Time-of-Day (general pool; soft time boost in rotation)
    # ----------------------------------------------------------------
    "main_character":   {"bpm": 115, "energy": -10, "danceability": 0.48, "brightness": 0.30,
                         "beat_confidence": 0.62, "onset_rate": 5.0, "dynamic_complexity": 0.55,
                         "arousal": 0.65, "valence": 0.58, "vocal_presence": 0.72},
    "golden_hour":      {"bpm":  95, "energy": -13, "danceability": 0.35, "brightness": 0.28,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.55,
                         "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.62},
    "sunset_mix":       {"bpm":  85, "energy": -16, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.35, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.32, "valence": 0.62, "vocal_presence": 0.55},
    "after_dark":       {"bpm": 105, "energy": -12, "danceability": 0.52, "brightness": 0.08,
                         "beat_confidence": 0.60, "onset_rate": 4.5, "dynamic_complexity": 0.42,
                         "arousal": 0.48, "valence": 0.38, "vocal_presence": 0.55},
    # ----------------------------------------------------------------
    # Time / Occasion (general pool; soft time boost in rotation)
    # ----------------------------------------------------------------
    "after_work":       {"bpm": 100, "energy": -12, "danceability": 0.45, "brightness": 0.30,
                         "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.48, "valence": 0.65, "vocal_presence": 0.60},
    "friday_night":     {"bpm": 118, "energy": -10, "danceability": 0.60, "brightness": 0.35,
                         "beat_confidence": 0.68, "onset_rate": 5.5, "dynamic_complexity": 0.42,
                         "arousal": 0.68, "valence": 0.78, "vocal_presence": 0.68},
    "weekend_mix":      {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.32,
                         "beat_confidence": 0.50, "onset_rate": 4.0, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.72, "vocal_presence": 0.62},
    "sunday_morning":   {"bpm":  82, "energy": -16, "danceability": 0.28, "brightness": 0.30,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.32, "valence": 0.72, "vocal_presence": 0.60},
    "lazy_sunday":      {"bpm":  70, "energy": -19, "danceability": 0.20, "brightness": 0.20,
                         "beat_confidence": 0.28, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.68, "vocal_presence": 0.55},
    "brunch_mix":       {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.35,
                         "beat_confidence": 0.50, "onset_rate": 3.8, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.78, "vocal_presence": 0.65},
    "date_night":       {"bpm":  88, "energy": -17, "danceability": 0.28, "brightness": 0.22,
                         "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.68, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Activity / Driving
    # ----------------------------------------------------------------
    "driving_mix":      {"bpm": 110, "energy": -10, "danceability": 0.50, "brightness": 0.32,
                         "beat_confidence": 0.62, "onset_rate": 4.8, "dynamic_complexity": 0.50,
                         "arousal": 0.58, "valence": 0.68, "vocal_presence": 0.70},
    "night_drive":      {"bpm": 100, "energy": -12, "danceability": 0.48, "brightness": 0.08,
                         "beat_confidence": 0.55, "onset_rate": 3.8, "dynamic_complexity": 0.45,
                         "arousal": 0.42, "valence": 0.35, "vocal_presence": 0.52},
    "driving_singalong":{"bpm": 112, "energy": -10, "danceability": 0.55, "brightness": 0.38,
                         "beat_confidence": 0.65, "onset_rate": 5.0, "dynamic_complexity": 0.48,
                         "arousal": 0.62, "valence": 0.78, "vocal_presence": 0.85},
    "road_trip":        {"bpm": 108, "energy": -11, "danceability": 0.48, "brightness": 0.35,
                         "beat_confidence": 0.58, "onset_rate": 4.5, "dynamic_complexity": 0.52,
                         "arousal": 0.58, "valence": 0.72, "vocal_presence": 0.72},
    "commute_mix":      {"bpm": 100, "energy": -12, "danceability": 0.42, "brightness": 0.28,
                         "beat_confidence": 0.52, "onset_rate": 4.0, "dynamic_complexity": 0.52,
                         "arousal": 0.48, "valence": 0.62, "vocal_presence": 0.65},
    "walking_mix":      {"bpm": 105, "energy": -12, "danceability": 0.48, "brightness": 0.32,
                         "beat_confidence": 0.58, "onset_rate": 4.5, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.68, "vocal_presence": 0.68},
    # ----------------------------------------------------------------
    # Social / Nostalgia
    # ----------------------------------------------------------------
    "party_throwback":  {"bpm": 128, "energy":  -8, "danceability": 0.72, "brightness": 0.42,
                         "beat_confidence": 0.75, "onset_rate": 6.0, "dynamic_complexity": 0.38,
                         "arousal": 0.78, "valence": 0.78, "vocal_presence": 0.70},
    # ----------------------------------------------------------------
    # Weather / Season (managed by _WEATHER_PROFILES / _SEASONAL_PROFILES)
    # ----------------------------------------------------------------
    "beach_vibes":      {"bpm": 100, "energy": -13, "danceability": 0.55, "brightness": 0.48,
                         "beat_confidence": 0.55, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.82, "vocal_presence": 0.65},
    "summer_evening":   {"bpm": 100, "energy": -13, "danceability": 0.50, "brightness": 0.30,
                         "beat_confidence": 0.52, "onset_rate": 4.0, "dynamic_complexity": 0.50,
                         "arousal": 0.48, "valence": 0.75, "vocal_presence": 0.62},
    "autumn_mix":       {"bpm":  80, "energy": -15, "danceability": 0.25, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.60,
                         "arousal": 0.35, "valence": 0.55, "vocal_presence": 0.65},
    "winter_mix":       {"bpm":  75, "energy": -17, "danceability": 0.22, "brightness": 0.12,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.45, "vocal_presence": 0.60},
    "spring_mix":       {"bpm": 105, "energy": -12, "danceability": 0.45, "brightness": 0.42,
                         "beat_confidence": 0.55, "onset_rate": 4.5, "dynamic_complexity": 0.52,
                         "arousal": 0.55, "valence": 0.78, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Romance
    # ----------------------------------------------------------------
    "modern_romance":   {"bpm": 100, "energy": -13, "danceability": 0.48, "brightness": 0.25,
                         "beat_confidence": 0.52, "onset_rate": 3.8, "dynamic_complexity": 0.48,
                         "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.68},
    "late_night_romance":{"bpm": 72, "energy": -18, "danceability": 0.22, "brightness": 0.08,
                         "beat_confidence": 0.28, "onset_rate": 2.0, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.62, "vocal_presence": 0.72},
    "romantic_dinner":  {"bpm":  80, "energy": -20, "danceability": 0.20, "brightness": 0.18,
                         "beat_confidence": 0.25, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.62, "vocal_presence": 0.55},
    "love_songs":       {"bpm":  90, "energy": -14, "danceability": 0.30, "brightness": 0.25,
                         "beat_confidence": 0.42, "onset_rate": 3.0, "dynamic_complexity": 0.62,
                         "arousal": 0.42, "valence": 0.75, "vocal_presence": 0.82},
    "slow_dance":       {"bpm":  68, "energy": -18, "danceability": 0.18, "brightness": 0.15,
                         "beat_confidence": 0.28, "onset_rate": 1.8, "dynamic_complexity": 0.65,
                         "arousal": 0.22, "valence": 0.68, "vocal_presence": 0.78},
    "candlelight":      {"bpm":  72, "energy": -20, "danceability": 0.15, "brightness": 0.15,
                         "beat_confidence": 0.22, "onset_rate": 1.5, "dynamic_complexity": 0.70,
                         "arousal": 0.18, "valence": 0.62, "vocal_presence": 0.65},
    "first_date":       {"bpm":  95, "energy": -14, "danceability": 0.38, "brightness": 0.30,
                         "beat_confidence": 0.45, "onset_rate": 3.5, "dynamic_complexity": 0.55,
                         "arousal": 0.42, "valence": 0.72, "vocal_presence": 0.70},
    "romantic_jazz":    {"bpm":  78, "energy": -20, "danceability": 0.22, "brightness": 0.18,
                         "beat_confidence": 0.30, "onset_rate": 2.5, "dynamic_complexity": 0.70,
                         "arousal": 0.22, "valence": 0.65, "vocal_presence": 0.72},
    "jazz_dinner":      {"bpm":  85, "energy": -21, "danceability": 0.25, "brightness": 0.20,
                         "beat_confidence": 0.28, "onset_rate": 2.8, "dynamic_complexity": 0.72,
                         "arousal": 0.20, "valence": 0.60, "vocal_presence": 0.50},
    "string_quartet":   {"bpm":  80, "energy": -19, "danceability": 0.18, "brightness": 0.20,
                         "beat_confidence": 0.20, "onset_rate": 2.0, "dynamic_complexity": 0.75,
                         "arousal": 0.22, "valence": 0.58, "vocal_presence": 0.30},
    "strings_romance":  {"bpm":  75, "energy": -20, "danceability": 0.15, "brightness": 0.18,
                         "beat_confidence": 0.18, "onset_rate": 1.8, "dynamic_complexity": 0.75,
                         "arousal": 0.20, "valence": 0.65, "vocal_presence": 0.32},
    "piano_romance":    {"bpm":  72, "energy": -22, "danceability": 0.15, "brightness": 0.16,
                         "beat_confidence": 0.18, "onset_rate": 1.5, "dynamic_complexity": 0.72,
                         "arousal": 0.18, "valence": 0.62, "vocal_presence": 0.35},
    "acoustic_romance": {"bpm":  78, "energy": -18, "danceability": 0.22, "brightness": 0.20,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.65,
                         "arousal": 0.28, "valence": 0.68, "vocal_presence": 0.78},
    "indie_romance":    {"bpm":  85, "energy": -17, "danceability": 0.28, "brightness": 0.18,
                         "beat_confidence": 0.38, "onset_rate": 2.8, "dynamic_complexity": 0.62,
                         "arousal": 0.32, "valence": 0.65, "vocal_presence": 0.72},
    "synthpop_romance": {"bpm": 100, "energy": -13, "danceability": 0.48, "brightness": 0.15,
                         "beat_confidence": 0.55, "onset_rate": 3.8, "dynamic_complexity": 0.45,
                         "arousal": 0.40, "valence": 0.68, "vocal_presence": 0.62},
    # ----------------------------------------------------------------
    # Gap fills
    # ----------------------------------------------------------------
    "evening_unwind":   {"bpm":  78, "energy": -17, "danceability": 0.22, "brightness": 0.15,
                         "beat_confidence": 0.32, "onset_rate": 2.2, "dynamic_complexity": 0.62,
                         "arousal": 0.28, "valence": 0.58, "vocal_presence": 0.50},
    "heartbreak":       {"bpm":  75, "energy": -14, "danceability": 0.18, "brightness": 0.07,
                         "beat_confidence": 0.32, "onset_rate": 2.0, "dynamic_complexity": 0.65,
                         "arousal": 0.45, "valence": 0.12, "vocal_presence": 0.88},
    "pre_party":        {"bpm": 122, "energy":  -9, "danceability": 0.65, "brightness": 0.38,
                         "beat_confidence": 0.72, "onset_rate": 5.8, "dynamic_complexity": 0.40,
                         "arousal": 0.72, "valence": 0.75, "vocal_presence": 0.68},
    "cool_down":        {"bpm":  78, "energy": -14, "danceability": 0.28, "brightness": 0.20,
                         "beat_confidence": 0.42, "onset_rate": 3.0, "dynamic_complexity": 0.55,
                         "arousal": 0.30, "valence": 0.60, "vocal_presence": 0.50},
    "cooking_mix":      {"bpm": 102, "energy": -12, "danceability": 0.48, "brightness": 0.38,
                         "beat_confidence": 0.55, "onset_rate": 4.2, "dynamic_complexity": 0.50,
                         "arousal": 0.52, "valence": 0.75, "vocal_presence": 0.65},
    "deep_work":        {"bpm":  88, "energy": -17, "danceability": 0.18, "brightness": 0.08,
                         "beat_confidence": 0.25, "onset_rate": 1.8, "dynamic_complexity": 0.68,
                         "arousal": 0.22, "valence": 0.52, "vocal_presence": 0.22},
    "folk_acoustic":    {"bpm":  82, "energy": -16, "danceability": 0.25, "brightness": 0.22,
                         "beat_confidence": 0.38, "onset_rate": 2.5, "dynamic_complexity": 0.68,
                         "arousal": 0.30, "valence": 0.62, "vocal_presence": 0.75},
    "celebration":      {"bpm": 125, "energy":  -8, "danceability": 0.62, "brightness": 0.42,
                         "beat_confidence": 0.72, "onset_rate": 5.8, "dynamic_complexity": 0.45,
                         "arousal": 0.78, "valence": 0.90, "vocal_presence": 0.78},
}

_MOOD_MIX_NAMES = {
    # Original 14
    "workout":            "Workout Mix • Meloday+",
    "running":            "Running Mix • Meloday+",
    "party":              "Party Mix • Meloday+",
    "happy":              "Happy Hits • Meloday+",
    "focus":              "Focus Mix • Meloday+",
    "chill":              "Chill Mix • Meloday+",
    "melancholy":         "Sad Songs • Meloday+",
    "morning":            "Good Morning Mix • Meloday+",
    "dinner":             "Dinner Mix • Meloday+",
    "late_night":         "Late Night Mix • Meloday+",
    "sleep":              "Sleep Mix • Meloday+",
    "rainy_day":          "Rainy Day Mix • Meloday+",
    "sunny":              "Sunny Mix • Meloday+",
    "cosy":               "Cosy Mix • Meloday+",
    # Mood / Emotional
    "nostalgia_mix":      "Nostalgia Mix • Meloday+",
    "dreamy_mix":         "Dreamy Mix • Meloday+",
    "moody_mix":          "Moody Mix • Meloday+",
    "emotional":          "Emotional Mix • Meloday+",
    "bittersweet":        "Bittersweet Mix • Meloday+",
    "cathartic":          "Cathartic Mix • Meloday+",
    "confidence_boost":   "Confidence Boost Mix • Meloday+",
    "empowering":         "Empowering Mix • Meloday+",
    "euphoric":           "Euphoric Mix • Meloday+",
    "angst_mix":          "Angst Mix • Meloday+",
    "romantic_mix":       "Romantic Mix • Meloday+",
    "daydreaming":        "Daydreaming Mix • Meloday+",
    "fresh_start":        "Fresh Start Mix • Meloday+",
    # Aesthetic / Time-of-Day
    "main_character":     "Main Character Mix • Meloday+",
    "golden_hour":        "Golden Hour Mix • Meloday+",
    "sunset_mix":         "Sunset Mix • Meloday+",
    "after_dark":         "After Dark Mix • Meloday+",
    # Time / Occasion
    "after_work":         "After Work Mix • Meloday+",
    "friday_night":       "Friday Night Mix • Meloday+",
    "weekend_mix":        "Weekend Mix • Meloday+",
    "sunday_morning":     "Sunday Morning Mix • Meloday+",
    "lazy_sunday":        "Lazy Sunday Mix • Meloday+",
    "brunch_mix":         "Brunch Mix • Meloday+",
    "date_night":         "Date Night Mix • Meloday+",
    # Activity
    "driving_mix":        "Driving Mix • Meloday+",
    "night_drive":        "Night Drive Mix • Meloday+",
    "driving_singalong":  "Driving Singalong Mix • Meloday+",
    "road_trip":          "Road Trip Mix • Meloday+",
    "commute_mix":        "Commute Mix • Meloday+",
    "walking_mix":        "Walking Mix • Meloday+",
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
    "evening_unwind":     "Evening Unwind Mix • Meloday+",
    "heartbreak":         "Heartbreak Mix • Meloday+",
    "pre_party":          "Pre-Party Mix • Meloday+",
    "cool_down":          "Cool Down Mix • Meloday+",
    "cooking_mix":        "Cooking Mix • Meloday+",
    "deep_work":          "Deep Work Mix • Meloday+",
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
# Hours past midnight use 24+: 1am=25, 2am=26, etc.
_TIME_BIASED_PROFILES = {
    "morning":    ( 5, 12),   # 5am–noon
    "dinner":     (17, 21),   # 5pm–9pm
    "late_night": (22, 26),   # 10pm–2am
    "sleep":      (22, 28),   # 10pm–4am
}

# Soft time boosts — general-pool profiles that score better during a time window
# but are NOT managed by the boundary cron. (start_hour, end_hour, reduction_amount)
_TIME_SOFT_BOOSTS = {
    "brunch_mix":         ( 9, 15, 0.12),   # 9am–3pm
    "golden_hour":        (15, 20, 0.15),   # 3pm–8pm
    "after_work":         (16, 20, 0.12),   # 4pm–8pm
    "sunset_mix":         (17, 22, 0.15),   # 5pm–10pm
    "date_night":         (17, 26, 0.10),   # 5pm–2am
    "friday_night":       (17, 26, 0.10),   # 5pm–2am (any evening, not just Friday)
    "after_dark":         (22, 28, 0.15),   # 10pm–4am
    "night_drive":        (20, 28, 0.12),   # 8pm–4am
    "late_night_romance": (21, 27, 0.12),   # 9pm–3am
    "commute_mix":        ( 7, 10, 0.10),   # morning commute
    "sunday_morning":     ( 7, 14, 0.08),   # any morning (day-of-week not tracked)
    "lazy_sunday":        (11, 18, 0.07),   # any quiet afternoon
    "evening_unwind":     (20, 23, 0.15),   # 8pm–11pm (dinner-to-late-night gap)
    "pre_party":          (17, 23, 0.12),   # 5pm–11pm evenings
    "cooking_mix":        (17, 20, 0.10),   # 5pm–8pm dinner-prep window
}

# Day-of-week boosts — applied on top of soft time boosts for profiles with a natural weekday.
# 0=Monday … 6=Sunday. Amount is an additional score reduction (lower = more likely selected).
_WEEKDAY_BOOSTS = {
    "friday_night":    ({4, 5},  0.15),   # Fri/Sat evenings
    "pre_party":       ({4, 5},  0.10),   # Fri/Sat
    "weekend_mix":     ({5, 6},  0.10),   # Sat/Sun all day
    "brunch_mix":      ({5, 6},  0.08),   # Weekend brunch
    "sunday_morning":  ({6},     0.12),   # Sunday only
    "lazy_sunday":     ({6},     0.10),   # Sunday only
    "celebration":     ({4, 5},  0.08),   # Fri/Sat
    "party_throwback": ({4, 5},  0.06),   # Fri/Sat (slight)
}

# Hard day-of-week gates — profiles excluded from the pool entirely on the wrong day.
# Different from _WEEKDAY_BOOSTS (score reduction only) — these are hard exclusions.
# 0=Monday … 6=Sunday.
_WEEKDAY_RESTRICTED = {
    "friday_night":   {4, 5},   # Fri/Sat only
    "pre_party":      {4, 5},   # Fri/Sat only
    "weekend_mix":    {5, 6},   # Sat/Sun only
    "brunch_mix":     {5, 6},   # Sat/Sun only
    "sunday_morning": {6},      # Sunday only
    "lazy_sunday":    {6},      # Sunday only
}

# Weather-conditional profiles — require weather data; add/remove when conditions match.
_WEATHER_PROFILES = {"rainy_day", "sunny", "cosy", "beach_vibes"}

# Season-conditional profiles — triggered by current calendar season; no weather API needed.
_SEASONAL_PROFILES = {"autumn_mix", "winter_mix", "spring_mix", "summer_evening"}

_TIME_PROFILES    = set(_TIME_BIASED_PROFILES.keys())
_GENERAL_PROFILES = (set(_MOOD_PROFILES)
                     - _TIME_PROFILES
                     - _WEATHER_PROFILES
                     - _SEASONAL_PROFILES)

# Category labels for diversity-aware rotation (max 2 per category in active slots).
_PROFILE_CATEGORY = {
    # High energy / physical
    "workout":          "energy",   "running":         "energy",
    "party":            "energy",   "euphoric":        "energy",
    "confidence_boost": "energy",   "empowering":      "energy",
    "cathartic":        "energy",   "angst_mix":       "energy",
    "friday_night":     "energy",   "party_throwback": "energy",
    # Relaxed / low-key
    "focus":            "calm",     "chill":           "calm",
    "sleep":            "calm",     "daydreaming":     "calm",
    "lazy_sunday":      "calm",     "candlelight":     "calm",
    "piano_romance":    "calm",     "string_quartet":  "calm",
    "strings_romance":  "calm",     "jazz_dinner":     "calm",
    "romantic_jazz":    "calm",
    # Emotional / introspective
    "melancholy":       "emotional","moody_mix":       "emotional",
    "bittersweet":      "emotional","emotional":       "emotional",
    "nostalgia_mix":    "emotional","rainy_day":       "emotional",
    # Romantic / love
    "romantic_mix":     "romantic", "modern_romance":  "romantic",
    "love_songs":       "romantic", "slow_dance":      "romantic",
    "first_date":       "romantic", "acoustic_romance":"romantic",
    "indie_romance":    "romantic", "synthpop_romance":"romantic",
    "date_night":       "romantic", "late_night_romance":"romantic",
    "romantic_dinner":  "romantic",
    # Upbeat / positive / social
    "happy":            "upbeat",   "sunny":           "upbeat",
    "beach_vibes":      "upbeat",   "morning":         "upbeat",
    "brunch_mix":       "upbeat",   "sunday_morning":  "upbeat",
    "weekend_mix":      "upbeat",   "fresh_start":     "upbeat",
    "summer_evening":   "upbeat",   "spring_mix":      "upbeat",
    # Activity / driving / occasion
    "driving_mix":      "activity", "night_drive":     "activity",
    "driving_singalong":"activity", "road_trip":       "activity",
    "commute_mix":      "activity", "walking_mix":     "activity",
    "after_work":       "activity",
    # Atmospheric / ambient / aesthetic
    "dreamy_mix":       "atmospheric","main_character": "atmospheric",
    "golden_hour":      "atmospheric","sunset_mix":     "atmospheric",
    "after_dark":       "atmospheric","late_night":     "atmospheric",
    "dinner":           "atmospheric","winter_mix":     "atmospheric",
    "autumn_mix":       "atmospheric","cosy":           "atmospheric",
    "evening_unwind":   "atmospheric","folk_acoustic":  "atmospheric",
    # Gap fills — other categories
    "heartbreak":       "emotional",
    "pre_party":        "energy",    "celebration":    "energy",
    "cool_down":        "activity",
    "cooking_mix":      "activity",
    "deep_work":        "calm",
}

# Mood tag signals per profile: (positive_substrings, negative_substrings).
# Substring matching against lowercased Plex/MusicBrainz mood tags from the Essentia cache.
# Tracks with matching positive tags get a distance boost; conflicting tags get a penalty.
# Profiles where emotional character is the key distinguishing feature benefit most.
_PROFILE_MOOD_SIGNALS = {
    "melancholy":  (["sad", "melanchol", "bittersweet", "somber", "mournful", "despair",
                     "lonely", "depressing", "heartbreak"],
                   ["happy", "euphoric", "upbeat", "energetic", "fun"]),
    "happy":       (["happy", "upbeat", "euphoric", "joyful", "cheerful", "feel good",
                     "carefree", "fun", "positive"],
                   ["sad", "melanchol", "dark", "depressing"]),
    "workout":     (["energetic", "aggressive", "powerful", "intense", "triumphant",
                     "motivating", "adrenaline"],
                   ["calm", "peaceful", "sad", "melanchol"]),
    "running":     (["energetic", "powerful", "upbeat", "motivating"],
                   ["calm", "peaceful"]),
    "party":       (["party", "energetic", "euphoric", "fun", "carefree", "celebrat"],
                   ["sad", "calm", "melanchol"]),
    "focus":       (["calm", "peaceful", "meditat", "relaxing", "ambient", "tranquil"],
                   ["aggressive", "intense", "party"]),
    "chill":       (["calm", "laid-back", "mellow", "relaxing", "peaceful", "easy"],
                   ["aggressive", "intense", "angry"]),
    "sleep":       (["calm", "peaceful", "dreamy", "ambient", "tranquil", "meditat"],
                   ["energetic", "aggressive", "upbeat"]),
    "late_night":  (["dark", "introspective", "moody", "atmospheric", "hypnotic"],
                   ["happy", "upbeat", "fun"]),
    "morning":     (["uplifting", "cheerful", "calm", "peaceful", "gentle", "fresh"],
                   ["dark", "aggressive", "sad"]),
    "dinner":      (["romantic", "sophisticated", "elegant", "smooth", "mellow"],
                   ["aggressive", "intense", "angry"]),
    "rainy_day":        (["melanchol", "bittersweet", "introspective", "nostalgic", "wistful"],
                        ["upbeat", "energetic", "euphoric"]),
    "sunny":            (["happy", "upbeat", "carefree", "fun", "joyful", "cheerful"],
                        ["sad", "dark", "melanchol"]),
    "cosy":             (["calm", "peaceful", "warm", "comfort", "nostalgic", "gentle"],
                        ["aggressive", "intense", "energetic"]),
    # New profiles
    "nostalgia_mix":    (["nostalgic", "wistful", "bittersweet", "sentimental", "reminiscent"],
                        ["aggressive", "energetic", "euphoric"]),
    "dreamy_mix":       (["dreamy", "ethereal", "atmospheric", "ambient", "surreal"],
                        ["aggressive", "intense", "angry"]),
    "moody_mix":        (["dark", "introspective", "moody", "brooding", "melanchol"],
                        ["happy", "upbeat", "fun", "euphoric"]),
    "emotional":        (["emotional", "heartbreak", "bittersweet", "intense", "powerful"],
                        []),
    "bittersweet":      (["bittersweet", "nostalgic", "wistful", "sad", "melanchol"],
                        ["upbeat", "energetic", "euphoric"]),
    "cathartic":        (["intense", "powerful", "dramatic", "triumphant", "emotional"],
                        ["calm", "peaceful", "ambient"]),
    "confidence_boost": (["confident", "powerful", "upbeat", "bold", "assertive"],
                        ["sad", "melanchol", "dark"]),
    "empowering":       (["empowering", "triumphant", "powerful", "motivating", "inspirational"],
                        ["sad", "melanchol", "dark"]),
    "angst_mix":        (["angry", "aggressive", "intense", "restless", "rebellious"],
                        ["calm", "peaceful", "happy", "upbeat"]),
    "romantic_mix":     (["romantic", "love", "tender", "intimate", "affectionate"],
                        ["aggressive", "angry", "intense"]),
    "daydreaming":      (["dreamy", "ambient", "calm", "peaceful", "ethereal"],
                        ["energetic", "aggressive", "intense"]),
    "fresh_start":      (["uplifting", "hopeful", "optimistic", "cheerful", "bright"],
                        ["dark", "aggressive", "sad"]),
    "euphoric":         (["euphoric", "ecstatic", "upbeat", "carefree", "exhilarating"],
                        ["sad", "dark", "melanchol"]),
    "beach_vibes":      (["happy", "carefree", "upbeat", "relaxing", "fun"],
                        ["sad", "dark", "aggressive"]),
    "autumn_mix":       (["nostalgic", "wistful", "melanchol", "introspective", "bittersweet"],
                        ["upbeat", "euphoric", "energetic"]),
    "winter_mix":       (["calm", "peaceful", "melanchol", "atmospheric", "nostalgic"],
                        ["energetic", "aggressive", "upbeat"]),
    "spring_mix":       (["uplifting", "hopeful", "cheerful", "fresh", "optimistic"],
                        ["dark", "sad", "aggressive"]),
    "love_songs":       (["romantic", "love", "tender", "heartfelt", "sincere"],
                        ["aggressive", "angry"]),
    "slow_dance":       (["romantic", "tender", "intimate", "love", "affectionate"],
                        ["aggressive", "energetic", "intense"]),
    "candlelight":      (["romantic", "elegant", "peaceful", "gentle", "intimate"],
                        ["aggressive", "intense", "energetic"]),
    "acoustic_romance": (["romantic", "love", "acoustic", "tender", "sincere"],
                        ["aggressive", "angry"]),
    "indie_romance":    (["romantic", "dreamy", "melanchol", "bittersweet", "indie"],
                        ["aggressive", "intense"]),
    "party_throwback":  (["party", "energetic", "fun", "carefree", "celebrat"],
                        ["sad", "calm", "melanchol"]),
    "heartbreak":       (["heartbreak", "sad", "longing", "lonely", "loss", "despair", "grief"],
                        ["happy", "upbeat", "euphoric", "carefree"]),
    "pre_party":        (["energetic", "fun", "carefree", "upbeat", "party", "celebrat"],
                        ["sad", "melanchol", "calm", "peaceful"]),
    "celebration":      (["celebrat", "triumphant", "happy", "euphoric", "upbeat", "joyful"],
                        ["sad", "melanchol", "dark"]),
    "folk_acoustic":    (["acoustic", "folk", "earthy", "organic", "natural", "singer-songwriter"],
                        ["aggressive", "intense", "electronic"]),
    "deep_work":        (["calm", "ambient", "meditat", "peaceful", "tranquil", "focus"],
                        ["aggressive", "intense", "party", "energetic"]),
    "evening_unwind":   (["calm", "peaceful", "mellow", "relaxing", "smooth", "easy"],
                        ["aggressive", "intense", "energetic"]),
    "cool_down":        (["calm", "relaxing", "peaceful", "mellow", "gentle"],
                        ["aggressive", "intense", "angry"]),
    "cooking_mix":      (["upbeat", "happy", "carefree", "fun", "cheerful"],
                        ["sad", "intense", "aggressive", "melanchol"]),
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
    """True if `hour` falls within (start, end) where end > 24 means past midnight."""
    start, end = window
    # Normalise hour to compare with potentially-past-midnight window
    h = hour if hour >= start else hour + 24
    return start <= h < end


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
_CLEAR_CODES = {113}  # sunny/clear

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

    return 0.0


def _season_active(profile_key, lat=0.0):
    """
    True if the current calendar season matches the profile's target season.
    Does not require a weather API call — uses local date only.
    Latitude (from weather data when available) adjusts for hemisphere.
    """
    season = _current_season(lat)
    return {
        "autumn_mix":     season == "autumn",
        "winter_mix":     season == "winter",
        "spring_mix":     season == "spring",
        "summer_evening": season == "summer",
    }.get(profile_key, False)


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
        if datetime.now().weekday() in days:
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


def resolve_tracks_by_keys(plex, rating_keys, workers=16):
    """Parallel fetchItem for a list of ratingKeys. Returns {rk_str: Track}."""
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
            if t:
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


def compute_listening_centroid(history_entries, essentia_cache, top_n=100):
    """
    Compute acoustic/taste centroid from the top-N most-played tracks.
    Returns dict with bpm/energy/danceability/brightness/year means
    plus styles_counter and genres_counter weighted by play count.
    """
    play_counts = Counter(str(e.ratingKey) for e in history_entries)
    top_keys = [rk for rk, _ in play_counts.most_common(top_n)]

    # Weighted sums — avoids extending large lists for heavily-played tracks
    wsum  = defaultdict(float)
    wcount = defaultdict(float)
    styles_counter = Counter()
    genres_counter = Counter()

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
    }


def _acoustic_distance_to_centroid(entry, centroid):
    """
    Normalised Euclidean distance between a track entry and a centroid.
    Returns 0.5 (neutral) when insufficient data is available.
    New fields (beat_confidence, onset_rate, dynamic_complexity, arousal, valence,
    vocal_presence) are None-safe — only scored when both entry and centroid have data,
    so the function degrades gracefully on libraries without TF features or new columns.
    """
    pairs = []
    if entry.get("bpm") and centroid.get("bpm"):
        pairs.append(((entry["bpm"] - centroid["bpm"]) / 200.0) ** 2)
    if entry.get("energy") is not None and centroid.get("energy") is not None:
        pairs.append(((entry["energy"] - centroid["energy"]) / 23.0) ** 2)
    if entry.get("danceability") is not None and centroid.get("danceability") is not None:
        pairs.append((entry["danceability"] - centroid["danceability"]) ** 2)
    if entry.get("brightness") is not None and centroid.get("brightness") is not None:
        pairs.append((entry["brightness"] - centroid["brightness"]) ** 2)
    if entry.get("year") and centroid.get("year"):
        pairs.append(min(abs(entry["year"] - centroid["year"]) / 100.0, 1.0) ** 2)
    if entry.get("beat_confidence") is not None and centroid.get("beat_confidence") is not None:
        pairs.append((entry["beat_confidence"] - centroid["beat_confidence"]) ** 2)
    if entry.get("onset_rate") is not None and centroid.get("onset_rate") is not None:
        pairs.append(min(abs(entry["onset_rate"] - centroid["onset_rate"]) / 10.0, 1.0) ** 2)
    if entry.get("dynamic_complexity") is not None and centroid.get("dynamic_complexity") is not None:
        pairs.append((entry["dynamic_complexity"] - centroid["dynamic_complexity"]) ** 2)
    if entry.get("arousal") is not None and centroid.get("arousal") is not None:
        pairs.append((entry["arousal"] - centroid["arousal"]) ** 2)
    if entry.get("valence") is not None and centroid.get("valence") is not None:
        pairs.append((entry["valence"] - centroid["valence"]) ** 2)
    if entry.get("vocal_presence") is not None and centroid.get("vocal_presence") is not None:
        pairs.append((entry["vocal_presence"] - centroid["vocal_presence"]) ** 2)
    if not pairs:
        return 0.5
    return min(math.sqrt(sum(pairs) / len(pairs)), 1.0)


def _tag_overlap_score(entry, centroid):
    """Jaccard-like overlap between track styles/genres and centroid's top tags."""
    top_styles = {s for s, _ in centroid.get("styles_counter", Counter()).most_common(10)}
    top_genres = {g for g, _ in centroid.get("genres_counter", Counter()).most_common(5)}
    track_styles = set(entry.get("styles") or [])
    track_genres = set(entry.get("genres") or [])
    if top_styles and track_styles:
        union = top_styles | track_styles
        if union:
            return len(top_styles & track_styles) / len(union)
    if top_genres and track_genres:
        union = top_genres | track_genres
        if union:
            return len(top_genres & track_genres) / len(union)
    return 0.0


def acoustic_affinity(rk, centroid, essentia_cache):
    """0–1 similarity: 60% acoustic distance (inverted) + 40% tag overlap."""
    entry = essentia_cache.get(str(rk), {})
    return (0.6 * (1.0 - _acoustic_distance_to_centroid(entry, centroid))
            + 0.4 * _tag_overlap_score(entry, centroid))


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


def _rating_dist_bonus(ur):
    """
    Distance reduction for library-scanning sorts. High-rated tracks sort as if
    acoustically closer to the target profile — Spotify's liked-track boosting.
    """
    if ur is None or ur <= 5: return 0.0
    if ur >= 9:               return 0.05  # 4.5–5★
    if ur >= 7:               return 0.02  # 3.5–4★
    return 0.01                             # 3★


def _artist_key(track):
    """Normalised primary artist key for deduplication and capping."""
    name = (getattr(track, "grandparentTitle", "") or
            getattr(track, "originalTitle", "") or "")
    return norm_text(primary_artist(name))


def _song_key(track):
    """
    Deduplication key: normalised (track_artist, clean_title) pair.
    Uses track_artist_name() which resolves the actual performing artist even
    on compilation albums where grandparentTitle is 'Various Artists' — in that
    case it falls back to track.originalTitle (where Plex stores the real artist).
    clean_title strips Live/Remastered/Deluxe suffixes.
    """
    artist = norm_text(primary_artist(track_artist_name(track)))
    title  = norm_text(clean_title(getattr(track, "title", "") or ""))
    return (artist, title)


def _dedup_filter(tracks):
    """
    Remove duplicate songs from a track list, keeping the first occurrence
    (which is always the highest-scored version since lists are pre-sorted).
    Two tracks are considered the same song if they share a normalised
    (artist, clean_title) key — so "Stars" from a studio album and
    "Stars" from a compilation are treated as one entry.
    """
    seen = set()
    result = []
    for t in tracks:
        key = _song_key(t)
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


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


def _make_geometric_background(w, h, color_top, color_bottom, v=0):
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
    for cx_f, cy_f, wf, hf, angle, alpha, color in strips:
        pts = _rotated_rect_points(cx_f * w, cy_f * h, wf * w, hf * h, angle)
        r, g, b = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(r, g, b, alpha))
    return img


def _make_concentric_circles_background(w, h, color_top, color_bottom, v=0):
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
    for i in range(n_rings, 0, -1):
        r = int(r_max * i / n_rings)
        t = 1.0 - i / n_rings
        col = tuple(int(color_bottom[k] + t * (color_top[k] - color_bottom[k])) for k in range(3))
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                     outline=(*col, 200), width=max(5, r // 7))
    return img


def _make_radial_glow_background(w, h, color_top, color_bottom, v=0):
    """Soft radial glow. v shifts the glow centre (sunrise, spotlight, candlelight, etc.)."""
    # (cx_frac, cy_frac, gamma)
    configs = [
        (0.50, 0.40, 0.65),  # v=0 discover_weekly — upper-centre
        (0.50, 0.65, 0.70),  # v=1 sunday_morning  — warm lower glow (sunrise)
        (0.50, 0.88, 0.55),  # v=2 sunset_mix       — horizon glow from bottom
        (0.50, 0.50, 0.80),  # v=3 piano_romance    — centred spotlight
        (0.50, 0.55, 0.75),  # v=4 candlelight      — just below centre, warm
        (0.32, 0.35, 0.60),  # v=5 late_night       — off-centre dark ambient
        (0.50, 0.18, 0.55),  # v=6 rainy_day        — diffuse from top (overcast)
    ]
    cx_f, cy_f, gamma = configs[min(v, len(configs) - 1)]
    if not _NUMPY_AVAILABLE:
        return _make_gradient_image(w, h, color_top, color_bottom, diagonal=True)
    cx, cy = w * cx_f, h * cy_f
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
    return Image.fromarray(arr, "RGBA")


def _make_waves_background(w, h, color_top, color_bottom, v=0):
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
        cy  = y_frac * h; amp = amp_frac * h; bh = bh_frac * h
        top = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase)) for x in range(steps)]
        bot = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase) + bh) for x in range(steps - 1, -1, -1)]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(top + bot, fill=(rc, gc, bc, alpha))
    return img


def _make_floating_circles_background(w, h, color_top, color_bottom, v=0):
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
        cx  = int(cx_f * w); cy = int(cy_f * h); rad = int(rf * min(w, h))
        rc, gc, bc = (_clamp(c) for c in color)
        draw.ellipse([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=(rc, gc, bc, alpha))
    return img


def _make_rays_background(w, h, color_top, color_bottom, v=0):
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


def _make_arc_sweep_background(w, h, color_top, color_bottom, v=0):
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
        cx  = cx_f * w; cy = cy_f * h; rad = rf * max(w, h)
        pts = [(cx + rad * math.cos(math.radians(a0 + s * (a1 - a0) / n_steps)),
                cy + rad * math.sin(math.radians(a0 + s * (a1 - a0) / n_steps)))
               for s in range(n_steps + 1)]
        pts.append((cx, cy))
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_aurora_background(w, h, color_top, color_bottom, v=0):
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
        cy = y_frac * h; amp = amp_frac * h; bh = bh_frac * h
        top = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase)) for x in range(steps)]
        bot = [(x, cy + amp * math.sin(2 * math.pi * freq * x / w + phase) + bh) for x in range(steps - 1, -1, -1)]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(top + bot, fill=(rc, gc, bc, alpha))
    return img


def _make_triangles_background(w, h, color_top, color_bottom, v=0):
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
        pts = [(xf * w, yf * h) for xf, yf in pts_frac]
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_diamond_background(w, h, color_top, color_bottom, v=0):
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
        pts = _diamond_pts(cx_f * w, cy_f * h, rw_f * w, rh_f * h, rot)
        rc, gc, bc = (_clamp(c) for c in color)
        draw.polygon(pts, fill=(rc, gc, bc, alpha))
    return img


def _make_starburst_background(w, h, color_top, color_bottom, v=0):
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
    for i in range(n_rays):
        a_mid  = math.radians(i * 360 / n_rays - 90)
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


def _make_chevrons_background(w, h, color_top, color_bottom, v=0):
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
    peak_x  = peak_f * w
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


def _make_spiral_background(w, h, color_top, color_bottom, v=0):
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
    for j in range(n_arcs):
        r_outer = max_r * (j + 1) / n_arcs
        r_inner = max_r * j / n_arcs * 0.90
        start_deg = direction * j * (360 / n_arcs) * 0.65 - 90
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
        badge_font = ImageFont.truetype(FONT_MELODAY_PATH, size=26)
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

        bar_size = 58
        try:
            font_bar = ImageFont.truetype(FONT_MAIN_PATH, size=bar_size)
        except (IOError, OSError):
            font_bar = ImageFont.load_default()
        text_x = STRIPE_W + 24
        while bar_size > 32:
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

    font_title = _load_font(92)
    font_year  = _load_font(220)

    # --- Auto-size subtitle ---
    font_sub = None
    if subtitle:
        sub_size = 46
        try:
            font_sub = ImageFont.truetype(FONT_LIGHT_PATH, size=sub_size)
            while sub_size > 28:
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

    SUBTITLE_RESERVE = 68
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
            year       = int(key[-4:])
            start_year = int(_extras.get("top_songs_start_year", datetime.now().year - 5))
            offset     = (year - start_year) % 30
            color_top, color_bottom = _TOP_SONGS_YEAR_PALETTES[offset]
            bg_style, bg_v = _TOP_SONGS_BG_CYCLE[offset], 0
        except (ValueError, IndexError):
            color_top, color_bottom = _EXTRAS_COVER_COLORS.get("top_songs", ((200, 160, 30), (140, 90, 10)))
            bg_style, bg_v = "geometric", 0
    else:
        color_top, color_bottom = _EXTRAS_COVER_COLORS.get(key, ((50, 65, 110), (20, 30, 65)))
        style_entry = _COVER_BG_STYLES.get(key, ("geometric", 0))
        if isinstance(style_entry, tuple):
            bg_style, bg_v = style_entry
        else:
            bg_style, bg_v = style_entry, 0

    text_style = "bar" if key in _MOOD_PROFILE_KEYS else "default"
    if bg_style == "circles":
        img = _make_concentric_circles_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "radial":
        img = _make_radial_glow_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "waves":
        img = _make_waves_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "floating_circles":
        img = _make_floating_circles_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "rays":
        img = _make_rays_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "arc_sweep":
        img = _make_arc_sweep_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "aurora":
        img = _make_aurora_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "triangles":
        img = _make_triangles_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "diamond":
        img = _make_diamond_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "starburst":
        img = _make_starburst_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "chevrons":
        img = _make_chevrons_background(1000, 1000, color_top, color_bottom, bg_v)
    elif bg_style == "spiral":
        img = _make_spiral_background(1000, 1000, color_top, color_bottom, bg_v)
    else:
        img = _make_geometric_background(1000, 1000, color_top, color_bottom, bg_v)
    img    = _add_bottom_vignette(img)
    result = _apply_cover_text(img, title, subtitle, accent_color=color_top, text_style=text_style)
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

def build_repeat_rewind(plex, history_entries, excluded_album_keys, target=30):
    """
    Tracks that were in heavy rotation 4–10 weeks ago but haven't been played
    in the last 21 days — the previous month's On Repeat.
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
    Priority: Last.fm popularity → acoustic centrality → positional fallback.
    """
    if not tracks:
        return []

    artist_name = getattr(album, "parentTitle", "") or ""
    is_va = artist_name.strip().casefold() in {"various artists", "various"}

    if LASTFM_API_KEY and artist_name and not is_va:
        top = _lastfm_top_tracks_for_artist(artist_name)
        if top:
            scored = []
            for t in tracks:
                count = _lastfm_popularity_score(t.title or "", artist_name)
                scored.append((count, t))
            scored.sort(key=lambda x: x[0], reverse=True)
            # Only use Last.fm ranking if the top track has meaningful data
            if scored[0][0] >= 100:
                return [t for _, t in scored[:2]]

    return _pick_representative_tracks(tracks, essentia_cache, n=2)


def build_release_radar(plex, music, essentia_cache, centroid, excluded_album_keys):
    """
    Tracks from recently released albums, ranked by recency then affinity.
    Expands the window in 7-day steps until RELEASE_RADAR_MIN_TRACKS are found.
    """
    today = date.today()

    try:
        all_albums = music.search(libtype="album", container_size=5000)
    except Exception as e:
        xlog(f"[ERROR] release_radar: album fetch failed: {e}")
        return []

    window_days = RELEASE_RADAR_START_DAYS
    result = []

    while len(result) < RELEASE_RADAR_MIN_TRACKS and window_days <= RELEASE_RADAR_MAX_DAYS:
        cutoff_date = today - timedelta(days=window_days)
        qualifying = []
        for album in all_albums:
            if str(album.ratingKey) in excluded_album_keys:
                continue
            rel = _album_release_date(album)
            if rel and rel >= cutoff_date:
                qualifying.append((rel, album))

        if qualifying:
            # Score albums
            album_data = []
            for rel, album in qualifying:
                tracks = _cached_album_tracks(album)
                if not tracks:
                    continue
                rks = [str(t.ratingKey) for t in tracks]
                alb_centroid = _album_acoustic_centroid(rks, essentia_cache)
                affinity = 1.0 - _acoustic_distance_to_centroid(alb_centroid, centroid)
                reps = _select_album_reps(album, tracks, essentia_cache)
                artist_key = _artist_key(reps[0]) if reps else ""
                album_data.append((rel, affinity, album.ratingKey, artist_key, reps))

            # Sort: newest first, affinity as tiebreaker within the same week
            def _sort_key(item):
                rel, affinity, _, _, _ = item
                week = rel.isocalendar()[1]
                return (rel.year, week, affinity)

            album_data.sort(key=_sort_key, reverse=True)

            # One album per artist (the most recent) — matches Spotify's Release Radar behaviour.
            # album_data is sorted newest-first, so the first album we see for each artist is
            # their latest release; subsequent albums from the same artist are skipped.
            seen_artists = set()
            result = []
            for rel, affinity, album_rk, art_k, reps in album_data:
                if art_k and art_k in seen_artists:
                    continue
                if art_k:
                    seen_artists.add(art_k)
                for t in reps:
                    if not is_low_rated(t):
                        result.append((rel, t))

        if len(result) < RELEASE_RADAR_MIN_TRACKS:
            window_days += RELEASE_RADAR_STEP_DAYS

    result.sort(key=lambda x: x[0], reverse=True)
    return _dedup_filter([t for _, t in result[:50]])


# --- 4. Discover Weekly ---

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
        score = acoustic_affinity(rk, centroid, essentia_cache)
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

def build_daily_mixes(plex, history_entries, essentia_cache, excluded_album_keys,
                      n_mixes=6, mix_size=50):
    """
    N Daily Mixes built via k-means acoustic clustering. Requires numpy.
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
    centroids, labels = _kmeans_fit(X, k=n_mixes)

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
        combined = history_tracks + library_tracks
        combined.sort(key=lambda t: (
            float(np.linalg.norm(X[rk_to_idx[str(t.ratingKey)]] - centroid_vec))
            if str(t.ratingKey) in rk_to_idx else 1.0
        ) - _rating_dist_bonus(getattr(t, "userRating", None)))
        tracks = combined[:mix_size]
        styles_subtitle = " · ".join(top_styles) if top_styles else ""
        mixes.append((f"Daily Mix {mix_num} • Meloday+", description, tracks, styles_subtitle))

    return mixes


# --- 6. Rediscovery Mix ---

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

    eligible.sort(key=lambda x: x[0])  # longest-neglected first

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

def build_time_capsule(plex, history_entries, essentia_cache, excluded_album_keys, target=30):
    """
    Era-anchored nostalgia mix. Uses birth_year from config (formative years = +13 to +25)
    if set; otherwise infers peak listening eras from history. Interleaved across eras.
    Not-played-in-90-days filter applied.
    """
    now = datetime.now(tz=timezone.utc)
    silence_cutoff = now - timedelta(days=90)

    last_played = {}
    all_time_plays = Counter()
    for e in history_entries:
        if not e.viewedAt:
            continue
        rk = str(e.ratingKey)
        if rk not in last_played or e.viewedAt > last_played[rk]:
            last_played[rk] = e.viewedAt
        all_time_plays[rk] += 1

    # Determine era windows and build cover subtitle
    era_label = None
    if BIRTH_YEAR:
        start_yr = int(BIRTH_YEAR) + 13
        end_yr   = int(BIRTH_YEAR) + 25
        era_windows = [(start_yr, end_yr)]
        era_label = f"{start_yr}–{end_yr}"
        xlog(f"[INFO] time_capsule: formative era {era_label} (birth_year={BIRTH_YEAR})")
    else:
        play_counts = Counter(str(e.ratingKey) for e in history_entries)
        top_keys = [rk for rk, _ in play_counts.most_common(100)]
        year_counts = Counter()
        for rk in top_keys:
            yr = (essentia_cache.get(rk) or {}).get("year")
            if yr and 1900 < yr < 2100:
                year_counts[yr] += play_counts[rk]

        if not year_counts:
            xlog("[WARN] time_capsule: no year data in essentia cache — skipping.")
            return [], None

        # Top 3 peak years, spaced ≥ 5 years apart to avoid overlapping windows
        peak_years = []
        for yr, _ in year_counts.most_common(50):
            if all(abs(yr - py) >= 5 for py in peak_years):
                peak_years.append(yr)
            if len(peak_years) >= 3:
                break

        era_windows = sorted([(yr - 2, yr + 2) for yr in peak_years])
        era_label = " · ".join(str(yr) for yr in sorted(peak_years))
        xlog(f"[INFO] time_capsule: inferred eras {era_windows}")

    # Build per-era candidate pools from essentia cache
    era_pools = []
    for start_yr, end_yr in era_windows:
        pool = []
        for rk, entry in essentia_cache.items():
            yr = entry.get("year")
            if not yr or yr < start_yr or yr > end_yr:
                continue
            lp = last_played.get(rk)
            if lp and lp >= silence_cutoff:
                continue
            pool.append((all_time_plays.get(rk, 0), rk))
        pool.sort(reverse=True)
        era_pools.append([rk for _, rk in pool[:target]])

    if not any(era_pools):
        xlog("[WARN] time_capsule: no qualifying tracks in era windows.")
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


# --- 8. Artist Deep Cuts ---

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

        # Score deep cuts by distance to artist's well-played centroid.
        # Resolve a larger pool so rating bonus can surface high-rated tracks
        # that sit slightly further from the centroid.
        has_acoustic = any(artist_centroid.get(f) for f in ("bpm", "energy"))
        scored = []
        for rk in deep_cut_rks:
            entry = essentia_cache.get(rk, {})
            score = (1.0 - _acoustic_distance_to_centroid(entry, artist_centroid)
                     if has_acoustic else 0.5)
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
            artist_deep_cuts[artist_key] = selected

    return _round_robin_interleave(
        [artist_deep_cuts[a] for a in top_artists if a in artist_deep_cuts],
        cap=target,
    )


# --- 9. Top Songs by Year ---

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
        # Past years are immutable — skip if a playlist already exists.
        # Current year is always regenerated as plays accumulate.
        if year != current_year and f"Top Songs {year} • Meloday+" in existing:
            xlog(f"[INFO] top_songs: skipping {year} (playlist already exists)")
            continue
        counts = year_plays[year]
        if len(counts) < min_distinct:
            continue
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        candidate_keys = [rk for rk, _ in ranked[:max(400, target * 4)]]
        track_map = resolve_tracks_by_keys(plex, candidate_keys)

        artist_count = Counter()
        seen_songs   = set()
        tracks = []
        for rk, _ in ranked:
            t = track_map.get(rk)
            if not t or is_low_rated(t):
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


def _select_diverse_profiles(scored_profiles, n_active, max_per_category=2):
    """
    Greedy category-diversity selection.
    `scored_profiles` is sorted (score, profile_key) ascending (lower = better fit).
    Picks n_active profiles ensuring at most max_per_category from the same category,
    then fills any remaining slots from the unselected remainder if needed.
    """
    cat_count = Counter()
    selected  = []
    remainder = []
    for _, key in scored_profiles:
        cat = _PROFILE_CATEGORY.get(key, "other")
        if cat_count[cat] < max_per_category:
            selected.append(key)
            cat_count[cat] += 1
        else:
            remainder.append(key)
        if len(selected) >= n_active:
            break
    # Fill gaps if strict diversity left us short
    for key in remainder:
        if len(selected) >= n_active:
            break
        selected.append(key)
    return selected[:n_active]


def build_mood_mixes(plex, history_entries, essentia_cache, excluded_album_keys,
                     n_active=5, mix_size=50, reselect=False, time_context=False,
                     existing_playlists=None):
    """
    Three operating modes:

    time_context=True  (boundary cron — 5am, noon, 5pm, 9pm, 10pm, 2am, 4am):
        Only adds/removes hard time-of-day playlists (Morning, Dinner, Late Night, Sleep).
        Does NOT touch general, weather, or seasonal mixes.

    reselect=False (default, daily 6am):
        Refreshes content for existing general mixes. Checks weather and season;
        adds/removes weather and seasonal mixes.

    reselect=True (weekly Monday):
        Full acoustic reselection of general mixes with category-diversity constraint.
        Also updates weather and seasonal mixes.

    Returns (mixes, profiles_to_delete) where mixes is a list of
    (playlist_name, profile_key, tracks) tuples.
    """
    name_to_key = {v: k for k, v in _MOOD_MIX_NAMES.items()}
    existing     = existing_playlists or {}

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

        mixes = []
        for profile_key in to_add:
            tracks = _build_mix_tracks(
                profile_key, essentia_cache, history_entries,
                excluded_album_keys, mix_size, plex)
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
    today_wd = datetime.now().weekday()
    if not reselect:
        active_general = [
            name_to_key[name]
            for name in existing
            if name in name_to_key and name_to_key[name] in _GENERAL_PROFILES
        ]
        # Evict any profile that is restricted to a different day of the week
        day_expired = [k for k in active_general
                       if today_wd not in _WEEKDAY_RESTRICTED.get(k, {today_wd})]
        if day_expired:
            for k in day_expired:
                active_general.remove(k)
            xlog(f"[INFO] mood_mixes: day-restricted profiles expired: {day_expired} — reselecting")
            reselect = True
        elif not active_general:
            xlog("[INFO] mood_mixes: no general mixes found, running initial selection")
            reselect = True

    if reselect:
        now = datetime.now(tz=timezone.utc)
        current_hour    = _get_active_hour()
        recent_entries  = [e for e in history_entries
                           if e.viewedAt and e.viewedAt >= now - timedelta(days=30)]
        recent_centroid = compute_listening_centroid(recent_entries, essentia_cache, top_n=100)
        if recent_centroid.get("bpm"):
            scored = sorted(
                (_mood_rotation_score(
                     k,
                     _acoustic_distance_to_centroid(target, recent_centroid),
                     current_hour, weather
                 ), k)
                for k, target in _MOOD_PROFILES.items()
                if k in _GENERAL_PROFILES
                and today_wd in _WEEKDAY_RESTRICTED.get(k, {today_wd})
            )
            n_cats      = len(set(_PROFILE_CATEGORY.values()))
            max_per_cat = max(1, math.ceil(n_active / n_cats))
            active_general = _select_diverse_profiles(scored, n_active, max_per_category=max_per_cat)
        else:
            active_general = list(_GENERAL_PROFILES)[:n_active]
        xlog(f"[INFO] mood_mixes: reselected general profiles = {active_general}")
    else:
        xlog(f"[INFO] mood_mixes: content refresh for general = {active_general}")
        # Gentle daily check: if the worst-fit active profile is significantly
        # outscored by the best inactive one, do a single swap so rotation stays
        # responsive between weekly reselections.
        now             = datetime.now(tz=timezone.utc)
        current_hour    = _get_active_hour()
        recent_entries  = [e for e in history_entries
                           if e.viewedAt and e.viewedAt >= now - timedelta(days=30)]
        recent_centroid = compute_listening_centroid(recent_entries, essentia_cache, top_n=100)
        if recent_centroid.get("bpm") and len(active_general) == n_active:
            eligible = {k for k in _GENERAL_PROFILES
                        if today_wd in _WEEKDAY_RESTRICTED.get(k, {today_wd})}
            all_scored = {
                k: _mood_rotation_score(
                    k,
                    _acoustic_distance_to_centroid(_MOOD_PROFILES[k], recent_centroid),
                    current_hour, weather
                )
                for k in eligible
            }
            active_set        = set(active_general)
            worst_active      = max(active_set,   key=lambda k: all_scored.get(k, 1.0))
            candidates        = sorted((all_scored[k], k) for k in eligible
                                       if k not in active_set)
            if candidates:
                best_score, best_inactive = candidates[0]
                if best_score < all_scored[worst_active] - 0.20:
                    incoming_cat = _PROFILE_CATEGORY.get(best_inactive, "other")
                    n_cats       = len(set(_PROFILE_CATEGORY.values()))
                    max_per_cat  = max(1, math.ceil(n_active / n_cats))
                    remaining_cats = Counter(
                        _PROFILE_CATEGORY.get(k, "other")
                        for k in active_set if k != worst_active
                    )
                    if remaining_cats[incoming_cat] < max_per_cat:
                        active_general.remove(worst_active)
                        active_general.append(best_inactive)
                        xlog(f"[INFO] mood_mixes: daily swap {worst_active} → {best_inactive} "
                             f"(scores {all_scored[worst_active]:.3f} vs {best_score:.3f})")

    # Weather-conditional profiles
    active_weather   = [k for k in _WEATHER_PROFILES if _weather_boost(k, weather) < 0]
    inactive_weather = [k for k in _WEATHER_PROFILES if k not in active_weather]

    # Season-conditional profiles (work without weather API)
    active_seasonal   = [k for k in _SEASONAL_PROFILES if _season_active(k, lat)]
    inactive_seasonal = [k for k in _SEASONAL_PROFILES if k not in active_seasonal]

    to_remove = (
        [k for k in inactive_weather  if _MOOD_MIX_NAMES[k] in existing] +
        [k for k in inactive_seasonal if _MOOD_MIX_NAMES[k] in existing]
    )

    active_profiles = active_general + active_weather + active_seasonal
    xlog(f"[INFO] mood_mixes: active = general{active_general} "
         f"+ weather{active_weather} + seasonal{active_seasonal}")

    mixes = []
    for profile_key in active_profiles:
        tracks = _build_mix_tracks(
            profile_key, essentia_cache, history_entries,
            excluded_album_keys, mix_size, plex)
        mixes.append((_MOOD_MIX_NAMES[profile_key], profile_key, tracks))

    # On reselect, remove general mixes that rotated out
    if reselect:
        for name, key in name_to_key.items():
            if key in _GENERAL_PROFILES and key not in active_general and name in existing:
                to_remove.append(key)

    return mixes, to_remove


def _build_mix_tracks(profile_key, essentia_cache, history_entries,
                      excluded_album_keys, mix_size, plex):
    """Build the 50-track list for a single mood mix profile."""
    target       = _MOOD_PROFILES[profile_key]
    play_counts  = Counter(str(e.ratingKey) for e in history_entries)

    def _combined_score(rk):
        """Acoustic distance adjusted by mood tag compatibility and play count."""
        entry = essentia_cache.get(rk, {})
        return (
            _acoustic_distance_to_centroid(entry, target)
            + _mood_tag_boost(entry, profile_key)
        )

    history_rks = sorted(
        [rk for rk in essentia_cache if rk in play_counts],
        key=lambda rk: (_combined_score(rk), -play_counts.get(rk, 0))
    )
    library_rks = sorted(
        [rk for rk in essentia_cache if not play_counts.get(rk)],
        key=_combined_score
    )

    n_history = int(mix_size * 0.40)
    n_library  = mix_size - n_history
    candidate_rks = list(dict.fromkeys(
        history_rks[:n_history * 3] + library_rks[:n_library * 3]
    ))
    track_map    = resolve_tracks_by_keys(plex, candidate_rks)
    artist_limit = max(2, int(mix_size * ARTIST_RATIO))
    artist_count = Counter()
    seen_songs   = set()  # shared across history and library pools

    history_tracks = []
    for rk in history_rks:
        if len(history_tracks) >= n_history:
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
        sk = _song_key(t)
        if sk in seen_songs:
            continue
        ak = _artist_key(t)
        if artist_count[ak] >= artist_limit:
            continue
        seen_songs.add(sk)
        artist_count[ak] += 1
        library_tracks.append(t)

    combined = history_tracks + library_tracks
    combined.sort(key=lambda t: (
        _combined_score(str(t.ratingKey))
        - _rating_dist_bonus(getattr(t, "userRating", None))
    ))
    return combined[:mix_size]


# ===========================================================================
# Orchestrator
# ===========================================================================

def _run_playlist(playlist_id, plex, music, ec, history, centroid, excluded_album_keys,
                  existing_playlists=None, reselect_moods=False, time_context=False):
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
        tracks = build_release_radar(plex, music, ec, centroid, excluded_album_keys)
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
        tracks, era_label = build_time_capsule(plex, history, ec, excluded_album_keys)
        _upsert_extras_playlist(plex, "Time Capsule • Meloday+", tracks,
            _pick_description("time_capsule", era=era_label or "your past"),
            cover_key="time_capsule", cover_title="Time Capsule",
            cover_subtitle=era_label, existing_playlists=ep)

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
            existing_playlists=ep)
        for profile_key in to_remove:
            name = _MOOD_MIX_NAMES[profile_key]
            pl   = (ep or {}).get(name)
            if pl:
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
        Subsequent runs: once any past-year playlist exists, the initial scan is
        done. Years still missing a playlist at that point simply have no data —
        don't keep fetching back to them. Only regenerate the current year.
        """
        current_year = datetime.now(tz=timezone.utc).year
        start_year   = int(_extras.get("top_songs_start_year", current_year - 5))

        # If any past-year playlist exists, the initial full scan has been done.
        # Only fetch the current year going forward.
        has_past_year = any(
            f"Top Songs {yr} • Meloday+" in existing_playlists
            for yr in range(start_year, current_year)
        )
        if has_past_year:
            jan1 = datetime(current_year, 1, 1, tzinfo=timezone.utc)
            return (datetime.now(tz=timezone.utc) - jan1).days + 1

        # First run — fetch the full range so all years with data get playlists.
        jan1 = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - jan1).days + 1

    time_context_mode = getattr(args, "time_context", False)

    def _lookback_for(pid):
        if pid == "time_capsule" and BIRTH_YEAR:
            return 180
        if pid == "top_songs":
            return _top_songs_lookback()
        if pid == "all_time_favourites":
            return 0   # uses track.viewCount — no history needed
        if pid == "mood_mixes" and time_context_mode:
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
                          time_context=time_context_mode)
        except Exception:
            xlog(f"[ERROR] {playlist_id} failed:\n{traceback.format_exc()}")

    xlog("\n=== Done ===")


if __name__ == "__main__":
    main()
