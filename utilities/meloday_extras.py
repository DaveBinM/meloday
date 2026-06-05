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
    "cathartic":          ((180,  40,  80), (110,  15,  45)),  # crimson-burgundy
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
    "rainy_day":        ("ripples",         0),  # concentric ripples where drops land
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

# Per-profile icon overlay — drawn on top of the background before text.
# Keys not listed here get no icon (plain background only).
_PROFILE_ICON = {
    # Romance / love — favourite hearts; arrangement (solitary/pair/trio/cluster) via _HEART_MODE
    "romantic_mix":      "favorite",
    "modern_romance":    "favorite",
    "late_night_romance":"favorite",
    "romantic_dinner":   "wine_bar",
    "love_songs":        "favorite",
    "slow_dance":        "favorite",
    "date_night":        "wine_bar",
    "first_date":        "favorite",
    "acoustic_romance":  "favorite",
    "indie_romance":     "favorite",
    "synthpop_romance":  "favorite",
    "heartbreak":        "heart_broken",
    # Musical / instrumental — note glyph clusters mix music_note + music_note_2
    "piano_romance":     "piano",
    "romantic_jazz":     "music_note",
    "string_quartet":    "music_note",
    "strings_romance":   "music_note",
    "folk_acoustic":     "music_note",
    # Drinks / food
    "dinner":            "restaurant",
    "jazz_dinner":       "restaurant",
    "cosy":              "local_cafe",
    "brunch_mix":        "local_cafe",
    # Night / sleep
    "sleep":             "moon_stars",
    "late_night":        "partly_cloudy_night",
    # Weather / atmosphere
    "rainy_day":         "rainy",
    "melancholy":        "water_drop",
    "dreamy_mix":        "cloud",
    # Sun / day
    "morning":           "wb_sunny",
    "sunny":             "wb_sunny",
    "golden_hour":       "wb_sunny",
    "sunday_morning":    "wb_sunny",
    "summer_evening":    "wb_sunny",
    "sunset_mix":        "wb_sunny",
    "fresh_start":       "clear_day",
    "beach_vibes":       "beach_access",
    # Season
    "winter_mix":        "snowflake",
    "autumn_mix":        "forest",
    "spring_mix":        "local_florist",
    # Energy / power
    "workout":           "fitness_center",
    "running":           "directions_run",
    "confidence_boost":  "trending_up",
    "cathartic":         "whatshot",
    # Calm / dreamy / focus
    "daydreaming":       "cloud",
    "lazy_sunday":       "cloud",
    "focus":             "center_focus_strong",
    "deep_work":         "center_focus_strong",
    "evening_unwind":    "self_improvement",
    "cool_down":         "spa",
    # Party / celebration
    "party":             "celebration",
    "pre_party":         "nightlife",
    "friday_night":      "nightlife",
    "celebration":       "flare",          # cluster of flares
    "party_throwback":   "festival",
    "euphoric":          "festival",
    "happy":             "mood",
    "weekend_mix":       "weekend",
    # Driving / activity
    "driving_mix":       "directions_car",
    "commute_mix":       "route",
    "driving_singalong": "directions_car",
    "road_trip":         "route",
    "walking_mix":       "directions_walk",
    # Spotlight
    "main_character":    "star_shine",
    # Non-mood extras (concept glyphs)
    "on_repeat":         "repeat",
    "repeat_rewind":     "replay",
    "release_radar":     "radar",
    "rediscovery":       "travel_explore",
    "discover_weekly":   "explore",
    "time_capsule":      "history",
}

# Profiles whose icon rotates weekly — the per-ISO-week rng picks one each Monday.
_PROFILE_ICON_ROTATE = {
    "date_night": ["wine_bar", "local_bar", "brunch_dining"],
}

# Per-glyph placement metadata for _draw_icon_overlay.
#   extent     ≈ glyph radius in px at scale 1.0 (size = 2*extent*scale → hero proportion)
#   base_scale size multiplier; tilt = max |rotation|° (0 = upright); anchor = (x,y) fractions
#   kind       "single" or "cluster" (music notes). Romance clusters are set via _HEART_MODE.
_ICON_DEFAULT_META = {"extent": 205, "base_scale": 1.0, "tilt": 10, "anchor": (0.50, 0.40), "kind": "single"}
_ICON_META = {
    # Music profiles render a composed note cluster (mixes music_note + music_note_2).
    "music_note":          {"kind": "cluster", "extent": 215},
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
    "main_character": {"anchor": (0.50, 0.37), "base_scale": 1.12},
    "night_drive":    {"anchor": (0.62, 0.40)},
    "late_night":     {"anchor": (0.62, 0.41)},
    "sleep":          {"anchor": (0.50, 0.38)},
    "morning":        {"anchor": (0.40, 0.37)},
    "sunny":          {"anchor": (0.58, 0.37)},
    "golden_hour":    {"anchor": (0.40, 0.37)},
    "sunset_mix":     {"anchor": (0.50, 0.36)},
    "beach_vibes":    {"anchor": (0.60, 0.37)},
    "summer_evening": {"anchor": (0.40, 0.37)},
    # Release Radar — small, lower-right quadrant, semi-transparent, no shadow.
    "release_radar":  {"anchor": (0.74, 0.60), "base_scale": 0.45, "alpha": 120, "shadow": False},
    # Rainy Day — cloud up in the top-right, mirrored (rain falls left), ripples drawn below it.
    "rainy_day":      {"anchor": (0.72, 0.31), "flip": True},
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
    # Love / warmth — rose-red
    "favorite":              (240,  90, 115),
    "heart_broken":          (214, 102, 122),
    "wine_bar":              (214, 116, 138),   # wine rose
    "local_bar":             (224, 140, 110),   # cocktail amber
    "brunch_dining":         (236, 196, 140),   # brunch warm
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
    # Food / drink — warm creams
    "local_cafe":            (226, 196, 150),   # latte
    "restaurant":            (238, 200, 150),
    "skillet":               (236, 192, 132),
    # Candle — warm glow
    "candle":                (255, 198, 120),
    # Music — warm cream accent
    "music_note":            (244, 230, 206),
    "music_note_2":          (244, 230, 206),
    "piano":                 (238, 232, 222),
    # Party / celebration — vivid festive
    "celebration":           (255, 140, 175),
    "festival":              (255, 150, 165),
    "nightlife":             (236, 150, 210),
    "flare":                 (255, 214, 130),   # festive gold spark
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
    "modern_romance":   {"bpm":  88, "energy": -15, "danceability": 0.32, "brightness": 0.22,
                         "beat_confidence": 0.40, "onset_rate": 3.0, "dynamic_complexity": 0.52,
                         "arousal": 0.35, "valence": 0.70, "vocal_presence": 0.75},
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
                         "arousal": 0.22, "valence": 0.52, "vocal_presence": 0.12},
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
}

# Hard day-of-week gates — profiles excluded from the pool entirely on the wrong day.
# Different from _WEEKDAY_BOOSTS (score reduction only) — these are hard exclusions.
# 0=Monday … 6=Sunday.
_WEEKDAY_RESTRICTED = {
    "friday_night":   {4},               # Friday only
    "pre_party":      {4, 5},            # Fri/Sat only
    "weekend_mix":    {5, 6},            # Sat/Sun only
    "brunch_mix":     {5, 6},            # Sat/Sun only
    "sunday_morning": {6},               # Sunday only
    "lazy_sunday":    {6},               # Sunday only
    "after_work":     {0, 1, 2, 3, 4},  # Mon–Fri only
    "commute_mix":    {0, 1, 2, 3, 4},  # Mon–Fri only
}

# Hard time-of-day gates — profiles excluded from the pool entirely outside their
# window. (start_hour, end_hour) in 0–23; if start > end the window wraps past midnight.
# Profiles not listed here are always eligible.
_HOUR_RESTRICTED = {
    "after_work":       (15, 20),   # 3pm-8pm only
    "brunch_mix":       ( 8, 13),   # 8am-1pm only
    "commute_mix":      ( 6,  9),   # morning commute
    "focus":            ( 5, 17),   # 5am–5pm only
    "deep_work":        ( 5, 17),   # 5am–5pm only
    "fresh_start":      ( 4, 10),   # 4am–10am — morning motivation only
    "workout":          ( 5, 22),   # 5am–10pm — no gym music in the small hours
    "running":          ( 5, 22),   # 5am–10pm
    "evening_unwind":   (17,  2),   # 5pm–2am — evening only
    "candlelight":      (17,  3),   # 5pm–3am — dinner and late romance
    "night_drive":      (19,  4),   # 7pm–4am — night driving only
    "after_dark":       (20,  6),   # 8pm–6am — after dark by definition
    "late_night_romance":(19,  4),  # 7pm–4am — evening/night only
}

# Weather-conditional profiles — require weather data; add/remove when conditions match.
_WEATHER_PROFILES = {"rainy_day", "sunny", "cosy", "beach_vibes"}

# Season-conditional profiles — triggered by current calendar season; no weather API needed.
_SEASONAL_PROFILES = {"autumn_mix", "winter_mix", "spring_mix", "summer_evening"}

# Work-hours focus guarantee — at least one of these is always active Mon-Fri 7am-3pm.
_WORK_FOCUS_PROFILES = {"focus", "deep_work"}

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
    # ----- style-defined genre mixes (positives required) -----
    "synthpop_romance": (["synth pop", "synthwave", "new romantic", "new wave",
                          "indie electronic", "dance-pop", "sophisti-pop", "dream pop",
                          "left-field pop", "neo-electro"],
                         ["metal", "rap", "country", "folk", "jazz", "punk", "hardcore", "blues"]),
    "folk_acoustic":    (["indie folk", "contemporary folk", "folk-rock", "folk-pop",
                          "singer/songwriter", "americana", "new acoustic", "alt-country",
                          "anti-folk", "folk revival", "neo-traditional folk", "folk"],
                         ["metal", "rap", "edm", "techno", "house", "club/dance", "synth"]),
    "acoustic_romance": (["singer/songwriter", "indie folk", "contemporary folk", "folk-pop",
                          "new acoustic", "americana", "folk-rock"],
                         ["metal", "rap", "edm", "club/dance", "techno", "house", "punk", "hardcore"]),
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
    "synthpop_romance", "folk_acoustic", "acoustic_romance", "indie_romance",
    "romantic_jazz", "jazz_dinner", "string_quartet", "strings_romance", "piano_romance",
}


def _has_required_style(entry, profile_key):
    """True if the track carries one of the profile's required (positive) styles. Untagged
    tracks return False so genre-defined mixes never include unconfirmable tracks."""
    sig = _PROFILE_STYLE_SIGNALS.get(profile_key)
    if not sig:
        return True
    positive_subs = sig[0]
    tags = [t.lower() for t in (entry.get("styles") or []) + (entry.get("genres") or [])]
    if not tags:
        return False
    return any(sub in tag for tag in tags for sub in positive_subs)


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
    tags = [t.lower() for t in (entry.get("styles") or []) + (entry.get("genres") or [])]
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
    result = (sum(vs) / len(vs), sum(as_) / len(as_)) if vs else (None, None)
    try:
        entry["_affect_proxy"] = result
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
        _add((entry["beat_confidence"] - centroid["beat_confidence"]) ** 2, 1.0)
    if entry.get("onset_rate") is not None and centroid.get("onset_rate") is not None:
        _add(min(abs(entry["onset_rate"] - centroid["onset_rate"]) / 10.0, 1.0) ** 2, 1.0)
    if entry.get("dynamic_complexity") is not None and centroid.get("dynamic_complexity") is not None:
        _add((entry["dynamic_complexity"] - centroid["dynamic_complexity"]) ** 2, 1.0)
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
    if entry.get("vocal_presence") is not None and centroid.get("vocal_presence") is not None:
        _add((entry["vocal_presence"] - centroid["vocal_presence"]) ** 2, 1.0)
    if den == 0:
        return 0.5
    return min(math.sqrt(num / den), 1.0)


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
    for _ in range(5 + (v % 3)):
        rx    = rng.randint(int(w * 0.10), int(w * 0.90))
        ry    = rng.randint(int(h * 0.08), int(h * 0.70))
        rings = rng.randint(3, 5)
        gap   = rng.randint(24, 40)
        for k in range(1, rings + 1):
            rr    = k * gap
            alpha = max(16, 110 - k * 20)
            tone  = light if k % 2 else dark
            draw.ellipse([rx - rr, ry - rr * 0.62, rx + rr, ry + rr * 0.62],
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
    "favorite": 0xE87E, "heart_broken": 0xEAC2, "candle": 0xF588, "music_note": 0xE405,
    "music_note_2": 0xFFFD8, "piano": 0xE521, "graphic_eq": 0xE1B8, "bedtime": 0xF159,
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
    "brunch_dining": 0xEA73, "flare": 0xE3E4,
}

_MS_FONT_CACHE = {}


def _load_ms_font(size):
    """Material Symbols variable font at `size` px em, set to Filled / heavy / opsz 48.
    Cached by size. layout_engine is left default (codepoint render needs no shaping)."""
    font = _MS_FONT_CACHE.get(size)
    if font is None:
        font = ImageFont.truetype(MS_ICON_FONT, size)
        try:
            font.set_variation_by_axes([1, 0, 48, 700])  # FILL, GRAD, opsz, wght
        except Exception:
            pass
        _MS_FONT_CACHE[size] = font
    return font


def _draw_glyph(layer, icon_name, cx, cy, size, fill, tilt=0, flip=False,
                stroke_width=0, stroke_fill=None):
    """Render a Material Symbol glyph centred at (cx, cy) at `size` px em, optionally tilted,
    mirrored (flip = horizontal) and outlined (stroke), compositing onto RGBA `layer` via its
    own tile (so it can rotate/mirror freely)."""
    cp = _MS_CODEPOINTS.get(icon_name)
    if cp is None or not _PIL_AVAILABLE:
        return
    ch   = chr(cp)
    font = _load_ms_font(max(8, int(round(size))))
    l, t, r, b = font.getbbox(ch, stroke_width=stroke_width)     # ink bounds, to centre
    gw, gh = max(1, r - l), max(1, b - t)
    pad = max(8, int(size * 0.10) + stroke_width)
    tile = Image.new("RGBA", (gw + 2 * pad, gh + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((pad - l, pad - t), ch, font=font, fill=fill,
                              stroke_width=stroke_width, stroke_fill=stroke_fill)
    if flip:
        tile = tile.transpose(Image.FLIP_LEFT_RIGHT)
    if tilt:
        tile = tile.rotate(tilt, resample=Image.BICUBIC, expand=True)
    layer.alpha_composite(tile, (int(round(cx - tile.width / 2)),
                                 int(round(cy - tile.height / 2))))


def _draw_glyph_cluster(layer, names, cx, cy, anchor_size, rng, fill, n=3, stroke=None):
    """Composed constellation: one anchor glyph plus (n-1) satellites at 55–70% size on a
    balanced ring around it, gentle varied tilts. Overlapping shapes get an outline
    (`stroke` = (width_fraction, fill)) so they stay legible. `names` is cycled across glyphs
    (e.g. ["music_note","music_note_2"]) or repeated (["favorite"], ["flare"])."""
    sw, sf = stroke or (0, None)

    def g(name, gx, gy, gsize, gtilt):
        _draw_glyph(layer, name, gx, gy, gsize, fill, tilt=gtilt,
                    stroke_width=int(round(gsize * sw)) if sw else 0, stroke_fill=sf)

    g(names[0], cx, cy, anchor_size, rng.uniform(-8, 8))
    if n <= 1:
        return
    ring   = anchor_size * 0.50
    base_a = rng.uniform(0, 2 * math.pi)
    for i in range(1, n):
        ang   = base_a + 2 * math.pi * (i - 1) / (n - 1) + rng.uniform(-0.25, 0.25)
        ssize = anchor_size * rng.uniform(0.55, 0.70)
        g(names[i % len(names)], cx + ring * math.cos(ang), cy + ring * math.sin(ang),
          ssize, rng.uniform(-15, 15))


def _draw_falling_cluster(layer, cloud_name, flake_name, cx, cy, size, rng, fill, stroke=None):
    """A cloud near the top with several flakes falling below it (downward spread, varied
    size and tilt) — used for Winter. `size` is the cloud's em size; cluster centred on (cx,cy)."""
    sw, sf = stroke or (0, None)

    def g(name, gx, gy, gsize, gtilt):
        _draw_glyph(layer, name, gx, gy, gsize, fill, tilt=gtilt,
                    stroke_width=int(round(gsize * sw)) if sw else 0, stroke_fill=sf)

    cloud_y = cy - size * 0.42
    g(cloud_name, cx, cloud_y, size, rng.uniform(-6, 6))
    n = rng.randint(4, 6)
    for i in range(n):
        fx    = cx + rng.uniform(-0.62, 0.62) * size
        fy    = cloud_y + size * (0.32 + 0.60 * (i + rng.uniform(0, 1)) / n)
        fsize = size * rng.uniform(0.26, 0.42)
        g(flake_name, fx, fy, fsize, rng.uniform(-20, 20))


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
    """A few faint concentric ripple ellipses just below a rain-cloud icon (Rainy Day)."""
    draw = ImageDraw.Draw(layer)
    ry   = cy + size * 0.42
    a0   = max(40, int(alpha * 0.55))
    lw   = max(2, int(size * 0.012))
    for k, rr in enumerate((size * 0.30, size * 0.46, size * 0.62)):
        draw.ellipse([cx - rr, ry - rr * 0.34, cx + rr, ry + rr * 0.34],
                     outline=(*rgb, max(28, a0 - k * 16)), width=lw)


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
        cluster_mode, cluster_names, cluster_n = "ring", ["flare"], rng.randint(4, 6)
    elif key == "winter_mix":
        cluster_mode = "falling"
    kind = "cluster" if cluster_mode else "single"

    scale = meta["base_scale"] * rng.uniform(0.92, 1.12)

    # Off-centre placement, clamped clear of the bottom title bar (22%) and the badge.
    R       = meta["extent"] * scale
    R_clamp = R * (1.20 if kind == "cluster" else 1.0)   # clusters spread wider than R
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
    size  = max(40, int(round(2 * R)))               # font em size → prominent hero proportion

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if cluster_mode:
        # Outline so overlapping shapes stay legible — contrast vs. the fill.
        fl = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        stroke_rgb = tuple(int(c * 0.28) for c in rgb) if fl >= 110 \
            else tuple(min(255, int(c + (255 - c) * 0.75)) for c in rgb)
        stroke = (0.045, (*stroke_rgb, alpha))
        if cluster_mode == "falling":
            _draw_falling_cluster(overlay, "cloud", "snowflake", cx, cy, size * 0.46, rng, fill, stroke=stroke)
        else:
            _draw_glyph_cluster(overlay, cluster_names, cx, cy, anchor_size=size * 0.60,
                                rng=rng, fill=fill, n=cluster_n, stroke=stroke)
    else:
        angle = rng.uniform(-meta["tilt"], meta["tilt"])
        _draw_glyph(overlay, icon_name, cx, cy, size, fill, tilt=angle, flip=meta.get("flip", False))
        if key == "rainy_day":                       # little ripples landing below the cloud
            _draw_ripples_below(overlay, cx, cy, size, rgb, alpha)

    # The glyph is the topmost art (below the title text); _place_icon adds the drop-shadow.
    return _place_icon(base, overlay, cx, cy, scale=1.0, angle=0.0,
                       shadow=meta.get("shadow", True), bg_lum=bg_lum)


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
            # Seed ALL choices (palette, style, variant, colour jitter) from the year
            # number so the look is permanently stable — historical playlists should
            # not shift weekly the way mood mixes do.
            year_rng                 = random.Random(year)
            palette_idx              = year_rng.randint(0, len(_TOP_SONGS_YEAR_PALETTES) - 1)
            color_top, color_bottom  = _TOP_SONGS_YEAR_PALETTES[palette_idx]
            _ts_style, _ts_max_v     = year_rng.choice(_TOP_SONGS_STYLE_POOL)
            bg_style                 = _ts_style
            bg_v                     = year_rng.randint(0, _ts_max_v)
            _rng                     = year_rng  # geometry jitter uses the same seed stream
            _jitter_range            = 30        # more dramatic shift between years
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

    text_style = "bar" if key in _MOOD_PROFILE_KEYS else "default"
    if bg_style == "circles":
        img = _make_concentric_circles_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "radial":
        img = _make_radial_glow_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "ripples":
        img = _make_ripples_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "waves":
        img = _make_waves_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "floating_circles":
        img = _make_floating_circles_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "rays":
        img = _make_rays_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "arc_sweep":
        img = _make_arc_sweep_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "aurora":
        img = _make_aurora_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "triangles":
        img = _make_triangles_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "diamond":
        img = _make_diamond_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "starburst":
        img = _make_starburst_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "chevrons":
        img = _make_chevrons_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    elif bg_style == "spiral":
        img = _make_spiral_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    else:
        img = _make_geometric_background(1000, 1000, color_top, color_bottom, bg_v, rng=_rng)
    img    = _add_bottom_vignette(img)
    img    = _draw_icon_overlay(img, key, color_top, color_bottom, _rng)
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
            affinity     = 1.0 - _acoustic_distance_to_centroid(alb_centroid, centroid)
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
    return _dedup_filter([t for _, t in result[:TARGET]])


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


# Two mood mixes whose (valence-weighted) centroids are closer than this select overly
# similar tracks — don't surface them at the same time. Tuned against the live cache so
# near-duplicate pairs (happy/sunny, workout/running, chill/lazy_sunday, focus/sleep) are
# caught while genuinely-distinct pairs (rainy_day/cosy) are not.
_SIM_GUARD_DISTANCE = 0.12


def _select_diverse_profiles(scored_profiles, n_active, max_per_category=2):
    """
    Greedy diversity selection.
    `scored_profiles` is sorted (score, profile_key) ascending (lower = better fit).
    Picks n_active profiles ensuring (a) at most max_per_category from the same category
    and (b) no two picks whose centroids are within _SIM_GUARD_DISTANCE (so very similar
    mixes aren't shown together). Deferred candidates backfill any remaining slots —
    preferring still-distinct ones — so the slate is never left short.
    """
    cat_count = Counter()
    selected  = []
    deferred  = []

    def _too_similar(key):
        return any(
            _acoustic_distance_to_centroid(_MOOD_PROFILES[key], _MOOD_PROFILES[s]) < _SIM_GUARD_DISTANCE
            for s in selected if s in _MOOD_PROFILES
        )

    for _, key in scored_profiles:
        if len(selected) >= n_active:
            break
        cat = _PROFILE_CATEGORY.get(key, "other")
        if cat_count[cat] >= max_per_category or (key in _MOOD_PROFILES and _too_similar(key)):
            deferred.append(key)
            continue
        selected.append(key)
        cat_count[cat] += 1

    # Backfill: first with deferred profiles that are still distinct, then with any.
    for require_distinct in (True, False):
        for key in deferred:
            if len(selected) >= n_active:
                break
            if key in selected:
                continue
            if require_distinct and key in _MOOD_PROFILES and _too_similar(key):
                continue
            selected.append(key)
        if len(selected) >= n_active:
            break
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
    today_wd     = datetime.now().weekday()
    current_hour = _get_active_hour()

    def _hour_allowed(k):
        window = _HOUR_RESTRICTED.get(k)
        return _in_time_window(current_hour, window) if window else True

    if not reselect:
        active_general = [
            name_to_key[name]
            for name in existing
            if name in name_to_key and name_to_key[name] in _GENERAL_PROFILES
        ]
        # Evict any profile that is restricted to a different day of the week
        # or is outside its allowed time-of-day window
        day_expired = [k for k in active_general
                       if today_wd not in _WEEKDAY_RESTRICTED.get(k, {today_wd})]
        hour_expired = [k for k in active_general if not _hour_allowed(k)]
        expired = list(dict.fromkeys(day_expired + hour_expired))
        if expired:
            for k in expired:
                active_general.remove(k)
            xlog(f"[INFO] mood_mixes: time-restricted profiles expired: {expired} — reselecting")
            reselect = True
        elif not active_general:
            xlog("[INFO] mood_mixes: no general mixes found, running initial selection")
            reselect = True

    if reselect:
        now = datetime.now(tz=timezone.utc)
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
                and _hour_allowed(k)
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
        recent_entries  = [e for e in history_entries
                           if e.viewedAt and e.viewedAt >= now - timedelta(days=30)]
        recent_centroid = compute_listening_centroid(recent_entries, essentia_cache, top_n=100)
        if recent_centroid.get("bpm") and len(active_general) == n_active:
            eligible = {k for k in _GENERAL_PROFILES
                        if today_wd in _WEEKDAY_RESTRICTED.get(k, {today_wd})
                        and _hour_allowed(k)}
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

    # Work-hours guarantee — Mon-Fri 7am-3pm: ensure at least one focus/work profile is active.
    if 0 <= today_wd <= 4 and 7 <= current_hour < 15:
        if not any(k in _WORK_FOCUS_PROFILES for k in active_general):
            focus_pool = sorted(
                _WORK_FOCUS_PROFILES & set(_GENERAL_PROFILES),
                key=lambda k: (
                    _acoustic_distance_to_centroid(_MOOD_PROFILES[k], recent_centroid)
                    if recent_centroid.get("bpm") else 0
                ),
            )
            if focus_pool:
                pick = focus_pool[0]
                if len(active_general) >= n_active:
                    for evict in reversed(list(active_general)):
                        if evict not in _WORK_FOCUS_PROFILES:
                            active_general.remove(evict)
                            xlog(f"[INFO] mood_mixes: work-hours guarantee: {evict} → {pick}")
                            break
                if pick not in active_general:
                    active_general.append(pick)

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
        """Acoustic distance adjusted by mood/style tag compatibility and play count."""
        entry = essentia_cache.get(rk, {})
        return (
            _acoustic_distance_to_centroid(entry, target)
            + _mood_tag_boost(entry, profile_key)
            + _style_tag_boost(entry, profile_key)
        )

    history_rks = sorted(
        [rk for rk in essentia_cache if rk in play_counts],
        key=lambda rk: (_combined_score(rk), -play_counts.get(rk, 0))
    )
    library_rks = sorted(
        [rk for rk in essentia_cache if not play_counts.get(rk)],
        key=_combined_score
    )

    # Style-defined mixes (Synth-Pop Romance, Folk & Acoustic, the jazz/classical/indie
    # mixes): keep only tracks whose styles match the genre, so the mix stays genre-pure.
    if profile_key in _STYLE_DEFINED_PROFILES:
        history_rks = [rk for rk in history_rks
                       if _has_required_style(essentia_cache.get(rk, {}), profile_key)]
        library_rks = [rk for rk in library_rks
                       if _has_required_style(essentia_cache.get(rk, {}), profile_key)]

    n_history = min(int(mix_size * 0.40), len(history_rks))
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
