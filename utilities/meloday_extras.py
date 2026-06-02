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
    "release_radar":  (( 30, 110, 200), ( 10,  50, 120)),  # sky blue → navy
    "discover_weekly":(( 30, 160, 100), ( 10,  70,  60)),  # emerald → dark teal
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
    "cosy":           ((140, 190, 220), ( 80, 130, 175)),  # icy blue
}


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
    "workout":    {"bpm": 150, "energy": -7,  "danceability": 0.60, "brightness": 0.30},
    "running":    {"bpm": 160, "energy": -6,  "danceability": 0.45, "brightness": 0.28},
    "party":      {"bpm": 125, "energy": -9,  "danceability": 0.78, "brightness": 0.38},
    "happy":      {"bpm": 118, "energy": -11, "danceability": 0.65, "brightness": 0.48},
    "focus":      {"bpm":  90, "energy": -18, "danceability": 0.22, "brightness": 0.10},
    "chill":      {"bpm":  82, "energy": -15, "danceability": 0.32, "brightness": 0.16},
    "melancholy": {"bpm":  68, "energy": -15, "danceability": 0.15, "brightness": 0.07},
    # Time-of-day profiles (boosted when current time matches their window)
    "morning":    {"bpm": 100, "energy": -13, "danceability": 0.42, "brightness": 0.45},
    "dinner":     {"bpm":  88, "energy": -19, "danceability": 0.25, "brightness": 0.22},
    "late_night": {"bpm":  78, "energy": -14, "danceability": 0.42, "brightness": 0.05},
    "sleep":      {"bpm":  65, "energy": -23, "danceability": 0.10, "brightness": 0.03},
    # Weather-triggered profiles (boosted when conditions match)
    "rainy_day":  {"bpm":  72, "energy": -16, "danceability": 0.18, "brightness": 0.09},
    "sunny":      {"bpm": 108, "energy": -11, "danceability": 0.58, "brightness": 0.52},
    "cosy":       {"bpm":  75, "energy": -16, "danceability": 0.20, "brightness": 0.18},
}

_MOOD_MIX_NAMES = {
    "workout":    "Meloday+ Workout",
    "running":    "Meloday+ Running",
    "party":      "Meloday+ Party",
    "happy":      "Meloday+ Happy Hits",
    "focus":      "Meloday+ Focus",
    "chill":      "Meloday+ Chill",
    "melancholy": "Meloday+ Sad Songs",
    "morning":    "Meloday+ Good Morning",
    "dinner":     "Meloday+ Dinner",
    "late_night": "Meloday+ Late Night",
    "sleep":      "Meloday+ Sleep",
    "rainy_day":  "Meloday+ Rainy Day",
    "sunny":      "Meloday+ Sunny",
    "cosy":       "Meloday+ Cosy",
}

# Time windows (start_hour, end_hour) in which a profile gets a strong rotation boost.
# Hours past midnight use 24+: 1am=25, 2am=26, 3am=27, 4am=28.
_TIME_BIASED_PROFILES = {
    "morning":    ( 5, 12),   # 5am–noon
    "dinner":     (17, 21),   # 5pm–9pm
    "late_night": (22, 26),   # 10pm–2am
    "sleep":      (22, 28),   # 10pm–4am
}

# Weather profile triggers: profile_key → weather condition that boosts it.
_WEATHER_PROFILES = {"rainy_day", "sunny", "cosy"}
_TIME_PROFILES    = set(_TIME_BIASED_PROFILES.keys())             # morning, dinner, late_night, sleep
_GENERAL_PROFILES = set(_MOOD_PROFILES) - _TIME_PROFILES - _WEATHER_PROFILES


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
        data = resp.json()
        current = data["current_condition"][0]
        lat_str = (data.get("nearest_area", [{}])[0]
                   .get("latitude", [{"value": "0"}])[0]
                   .get("value", "0"))
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

    return 0.0


def _mood_rotation_score(profile_key, acoustic_dist, current_hour, weather):
    """
    Hybrid rotation score for mood mix selection. Lower = selected.
    Base is acoustic distance; context (time/weather) applies reductions.
    """
    score = acoustic_dist

    # Time-of-day boost
    window = _TIME_BIASED_PROFILES.get(profile_key)
    if window and _in_time_window(current_hour, window):
        score -= 0.35  # strong enough to override pure acoustic fit

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
    """Returns set of album ratingKeys that carry any configured exclusion label."""
    excluded = set()
    for label in EXCLUDE_LABEL_NAMES:
        try:
            albums = music.search(libtype="album", label=label)
            excluded.update(str(a.ratingKey) for a in albums)
        except Exception:
            pass
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
        for field in ("bpm", "energy", "danceability", "brightness", "year"):
            val = entry.get(field)
            if val is not None:
                wsum[field]   += val * count
                wcount[field] += count
        for s in (entry.get("styles") or []):
            styles_counter[s] += count
        for g in (entry.get("genres") or []):
            genres_counter[g] += count

    return {
        "bpm":            wsum["bpm"]          / wcount["bpm"]          if wcount["bpm"]          else None,
        "energy":         wsum["energy"]        / wcount["energy"]        if wcount["energy"]        else None,
        "danceability":   wsum["danceability"]  / wcount["danceability"]  if wcount["danceability"]  else None,
        "brightness":     wsum["brightness"]    / wcount["brightness"]    if wcount["brightness"]    else None,
        "year":           wsum["year"]          / wcount["year"]          if wcount["year"]          else None,
        "styles_counter": styles_counter,
        "genres_counter": genres_counter,
    }


def _acoustic_distance_to_centroid(entry, centroid):
    """
    Normalised Euclidean distance between a track entry and a centroid.
    Returns 0.5 (neutral) when insufficient data is available.
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

def _make_gradient_image(w, h, color_top, color_bottom):
    """RGBA gradient image. Colors are 3-tuples (RGB) or 4-tuples (RGBA)."""
    img = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(img)
    at = color_top[3]   if len(color_top)    == 4 else 255
    ab = color_bottom[3] if len(color_bottom) == 4 else 255
    for y in range(h):
        t = y / h
        draw.line([(0, y), (w, y)], fill=(
            int(color_top[0] + t * (color_bottom[0] - color_top[0])),
            int(color_top[1] + t * (color_bottom[1] - color_top[1])),
            int(color_top[2] + t * (color_bottom[2] - color_top[2])),
            int(at           + t * (ab              - at)),
        ))
    return img


def _apply_cover_text(img, title, subtitle=None):
    """
    Composite right-aligned title + optional subtitle + Meloday branding onto img (RGBA).
    Returns the composited RGBA Image.
    """
    W, H = img.size
    shadow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text_layer   = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow_draw  = ImageDraw.Draw(shadow_layer)
    text_draw    = ImageDraw.Draw(text_layer)

    try:
        font_main    = ImageFont.truetype(FONT_MAIN_PATH,    size=67)
        font_sub     = ImageFont.truetype(FONT_MAIN_PATH,    size=52)
        font_meloday = ImageFont.truetype(FONT_MELODAY_PATH, size=87)
    except (IOError, OSError):
        font_main = font_sub = font_meloday = ImageFont.load_default()

    text_box_width = 630
    text_box_left  = W - 110 - text_box_width
    y_pos = 100

    for line in wrap_text(title, font_main, text_draw, text_box_width):
        bbox = text_draw.textbbox((0, 0), line, font=font_main)
        x = text_box_left + (text_box_width - (bbox[2] - bbox[0]))
        shadow_draw.text((x, y_pos), line, font=font_main, fill=(0, 0, 0, 120))
        text_draw.text((x, y_pos),   line, font=font_main, fill=(255, 255, 255, 255))
        y_pos += bbox[3] - bbox[1] + 10

    if subtitle:
        y_pos += 16
        for line in wrap_text(subtitle, font_sub, text_draw, text_box_width):
            bbox = text_draw.textbbox((0, 0), line, font=font_sub)
            x = text_box_left + (text_box_width - (bbox[2] - bbox[0]))
            shadow_draw.text((x, y_pos), line, font=font_sub, fill=(0, 0, 0, 120))
            text_draw.text((x, y_pos),   line, font=font_sub, fill=(255, 255, 255, 200))
            y_pos += bbox[3] - bbox[1] + 8

    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=40))
    shadow_draw.text((110, H - 200), "Meloday+", font=font_meloday, fill=(0, 0, 0, 120))
    text_draw.text((110, H - 200),   "Meloday+", font=font_meloday, fill=(255, 255, 255, 255))

    result = Image.alpha_composite(img, shadow_layer)
    return Image.alpha_composite(result, text_layer)


def _generate_extras_cover(key, title, subtitle=None):
    """
    Plain gradient cover for non-Daily-Mix playlists.
    Returns saved .webp path or None on failure.
    """
    if not _PIL_AVAILABLE:
        xlog("[WARN] PIL not available — cover generation skipped.")
        return None
    color_top, color_bottom = _EXTRAS_COVER_COLORS.get(key, ((50, 65, 110), (20, 30, 65)))
    img = _make_gradient_image(1000, 1000, color_top, color_bottom)
    result = _apply_cover_text(img, title, subtitle)
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

    # Gradient colour tint (~60% opacity) — gives strong colour identity while album art shows through
    tint = _make_gradient_image(W, H, color_top + (155,), color_bottom + (155,))
    collage = Image.alpha_composite(collage, tint)

    result = _apply_cover_text(collage, title, subtitle)
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
        mixes.append((f"Meloday+ Daily Mix {mix_num}", description, tracks, styles_subtitle))

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
        resolved_scored.sort(reverse=True)

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
        if year != current_year and f"Meloday+ Top Songs {year}" in existing:
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

def build_mood_mixes(plex, history_entries, essentia_cache, excluded_album_keys,
                     n_active=5, mix_size=50, reselect=False, time_context=False,
                     existing_playlists=None):
    """
    Three operating modes:

    time_context=True  (boundary cron — 5am, noon, 5pm, 9pm, 10pm, 2am, 4am):
        Only adds/removes time-of-day playlists (Morning, Dinner, Late Night, Sleep)
        based on the current hour. Does NOT touch general or weather mixes.
        Returns (mixes_to_upsert, profiles_to_delete).

    reselect=False (default, daily 6am):
        Refreshes content for existing general mixes. Checks weather and
        adds/removes weather mixes. Does NOT touch time-of-day mixes.

    reselect=True (weekly Monday):
        Full acoustic reselection of general mixes. Also updates weather mixes.
        Does NOT touch time-of-day mixes.

    Returns (mixes, profiles_to_delete) where mixes is a list of
    (playlist_name, profile_key, tracks) tuples.
    """
    name_to_key = {v: k for k, v in _MOOD_MIX_NAMES.items()}
    existing     = existing_playlists or {}

    # ------------------------------------------------------------------
    # MODE 1: Time-context — only manage time-of-day mixes
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
    # MODE 2 / 3: General + weather mixes (daily refresh or weekly reselect)
    # ------------------------------------------------------------------
    weather_location = _extras.get("weather_location")
    weather = _get_weather(weather_location) if weather_location else None
    if weather:
        xlog(f"[INFO] mood_mixes: weather = {weather['condition']}, {weather['temp_c']}°C")

    # Determine which general profiles are active
    if not reselect:
        active_general = [
            name_to_key[name]
            for name in existing
            if name in name_to_key and name_to_key[name] in _GENERAL_PROFILES
        ]
        if not active_general:
            xlog("[INFO] mood_mixes: no general mixes found, running initial selection")
            reselect = True

    if reselect:
        now = datetime.now(tz=timezone.utc)
        recent_entries = [e for e in history_entries
                          if e.viewedAt and e.viewedAt >= now - timedelta(days=30)]
        recent_centroid = compute_listening_centroid(recent_entries, essentia_cache, top_n=100)
        if recent_centroid.get("bpm"):
            # Score only general profiles acoustically (no time/weather for stable weekly selection)
            scored = sorted(
                (_acoustic_distance_to_centroid(target, recent_centroid), k)
                for k, target in _MOOD_PROFILES.items()
                if k in _GENERAL_PROFILES
            )
            active_general = [k for _, k in scored[:n_active]]
        else:
            active_general = list(_GENERAL_PROFILES)[:n_active]
        xlog(f"[INFO] mood_mixes: reselected general profiles = {active_general}")
    else:
        xlog(f"[INFO] mood_mixes: content refresh for general = {active_general}")

    # Determine weather profiles: add if conditions match, remove if not
    active_weather = [
        k for k in _WEATHER_PROFILES
        if _weather_boost(k, weather) < 0  # negative boost = conditions match
    ]
    inactive_weather = [
        k for k in _WEATHER_PROFILES
        if k not in active_weather
    ]
    # Remove inactive weather mixes that are currently in Plex
    to_remove = [k for k in inactive_weather if _MOOD_MIX_NAMES[k] in existing]

    active_profiles = active_general + active_weather
    xlog(f"[INFO] mood_mixes: active = general{active_general} + weather{active_weather}")

    mixes = []
    for profile_key in active_profiles:
        tracks = _build_mix_tracks(
            profile_key, essentia_cache, history_entries,
            excluded_album_keys, mix_size, plex)
        mixes.append((_MOOD_MIX_NAMES[profile_key], profile_key, tracks))

    # On reselect, also remove general mixes that rotated out
    if reselect:
        for name, key in name_to_key.items():
            if key in _GENERAL_PROFILES and key not in active_general and name in existing:
                to_remove.append(key)

    return mixes, to_remove

    # This block is now dead — all paths return early above.
    # Kept as a safety fallback.
    return [], []


def _build_mix_tracks(profile_key, essentia_cache, history_entries,
                      excluded_album_keys, mix_size, plex):
    """Build the 50-track list for a single mood mix profile."""
    target       = _MOOD_PROFILES[profile_key]
    play_counts  = Counter(str(e.ratingKey) for e in history_entries)

    history_rks = sorted(
        [rk for rk in essentia_cache if rk in play_counts],
        key=lambda rk: (
            _acoustic_distance_to_centroid(essentia_cache.get(rk, {}), target),
            -play_counts.get(rk, 0),
        )
    )
    library_rks = sorted(
        [rk for rk in essentia_cache if not play_counts.get(rk)],
        key=lambda rk: _acoustic_distance_to_centroid(essentia_cache.get(rk, {}), target)
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
        _acoustic_distance_to_centroid(essentia_cache.get(str(t.ratingKey), {}), target)
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
        _upsert_extras_playlist(plex, "Meloday+ On Repeat", tracks,
            _pick_description("on_repeat"),
            cover_key="on_repeat", cover_title="On Repeat",
            existing_playlists=ep)

    elif playlist_id == "repeat_rewind":
        tracks = build_repeat_rewind(plex, history, excluded_album_keys)
        _upsert_extras_playlist(plex, "Meloday+ Repeat Rewind", tracks,
            _pick_description("repeat_rewind"),
            cover_key="repeat_rewind", cover_title="Repeat Rewind",
            existing_playlists=ep)

    elif playlist_id == "release_radar":
        tracks = build_release_radar(plex, music, ec, centroid, excluded_album_keys)
        _upsert_extras_playlist(plex, "Meloday+ Release Radar", tracks,
            _pick_description("release_radar"),
            cover_key="release_radar", cover_title="Release Radar",
            existing_playlists=ep)

    elif playlist_id == "discover_weekly":
        tracks = build_discover_weekly(plex, history, ec, centroid, excluded_album_keys,
                                       target=DISCOVER_WEEKLY_SIZE)
        _upsert_extras_playlist(plex, "Meloday+ Discover Weekly", tracks,
            _pick_description("discover_weekly"),
            cover_key="discover_weekly", cover_title="Discover Weekly",
            existing_playlists=ep)

    elif playlist_id == "daily_mixes":
        mixes = build_daily_mixes(plex, history, ec, excluded_album_keys,
                                  n_mixes=DAILY_MIX_COUNT)
        for name, _desc, tracks, styles_subtitle in mixes:
            mix_num = name.split()[-1]  # "Meloday+ Daily Mix 3" → "3"
            styles_list = [s.strip() for s in styles_subtitle.split("·") if s.strip()] \
                          if styles_subtitle else []
            # Cover subtitle: cap at 2 styles so text stays readable at mobile size.
            # The full style list is used in the playlist description.
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
        _upsert_extras_playlist(plex, "Meloday+ Rediscovery", tracks,
            _pick_description("rediscovery"),
            cover_key="rediscovery", cover_title="Rediscovery",
            existing_playlists=ep)

    elif playlist_id == "time_capsule":
        tracks, era_label = build_time_capsule(plex, history, ec, excluded_album_keys)
        _upsert_extras_playlist(plex, "Meloday+ Time Capsule", tracks,
            _pick_description("time_capsule", era=era_label or "your past"),
            cover_key="time_capsule", cover_title="Time Capsule",
            cover_subtitle=era_label, existing_playlists=ep)

    elif playlist_id == "deep_cuts":
        tracks = build_deep_cuts(plex, history, ec, excluded_album_keys)
        _upsert_extras_playlist(plex, "Meloday+ Deep Cuts", tracks,
            _pick_description("deep_cuts"),
            cover_key="deep_cuts", cover_title="Deep Cuts",
            existing_playlists=ep)

    elif playlist_id == "top_songs":
        results = build_top_songs(plex, history, excluded_album_keys,
                                  existing_playlists=ep)
        for year, tracks in results:
            year_str = str(year)
            _upsert_extras_playlist(plex, f"Meloday+ Top Songs {year_str}", tracks,
                _pick_description("top_songs", era=year_str),
                cover_key="top_songs", cover_title=f"Top Songs",
                cover_subtitle=year_str,
                existing_playlists=ep)

    elif playlist_id == "all_time_favourites":
        tracks = build_all_time_favourites(music, excluded_album_keys)
        _upsert_extras_playlist(plex, "Meloday+ All-Time Favourites", tracks,
            _pick_description("all_time_favourites"),
            cover_key="all_time_favourites", cover_title="All-Time Favourites",
            existing_playlists=ep)

    elif playlist_id == "mood_mixes":
        n_active = int(_extras.get("mood_mix_count", 5))
        mixes, to_remove = build_mood_mixes(
            plex, history, ec, excluded_album_keys, n_active=n_active,
            reselect=reselect_moods, time_context=time_context,
            existing_playlists=ep)
        # Delete profiles that rotated out or whose time/weather window closed
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
                cover_key=profile_key, cover_title=name.replace("Meloday+ ", ""),
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
        if getattr(pl, "title", "").startswith("Meloday+")
    }
    xlog(f"[OK] Found {len(existing_playlists)} existing Meloday+ playlist(s)")

    def _top_songs_lookback():
        """
        Fetch only from Jan 1 of the oldest year that doesn't yet have a playlist.
        Current year is always regenerated. Uses top_songs_start_year config key.
        """
        current_year = datetime.now(tz=timezone.utc).year
        start_year   = int(_extras.get("top_songs_start_year", current_year - 5))

        oldest_needed = current_year  # assume current year is always needed
        for yr in range(start_year, current_year):
            if f"Meloday+ Top Songs {yr}" not in existing_playlists:
                oldest_needed = yr
                break  # first (oldest) missing year — fetch from here

        jan1 = datetime(oldest_needed, 1, 1, tzinfo=timezone.utc)
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
