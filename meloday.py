import yaml
import os
import re
import random
import json
import unicodedata
from datetime import datetime, timedelta
from collections import Counter
from plexapi.server import PlexServer
from plexapi.audio import Track
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_config(filepath="config.yml"):
    with open(os.path.join(BASE_DIR, filepath), "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

config = load_config()

PLEX_URL = config["plex"]["url"]
PLEX_TOKEN = config["plex"]["token"]
MUSIC_LIBRARY = config["plex"]["music_library"]
CHRISTMAS_COLLECTION_NAME = config["plex"]["christmas_collection"]
EXCLUDE_LABEL_NAME = config["plex"]["exclude_label"]

EXCLUDE_PLAYED_DAYS = config["playlist"]["exclude_played_days"]
HISTORY_LOOKBACK_DAYS = config["playlist"]["history_lookback_days"]
MAX_TRACKS = config["playlist"]["max_tracks"]
SONIC_SIMILAR_LIMIT = config["playlist"]["sonic_similar_limit"]

xmas_cfg = config.get("seasonal", {}).get("christmas", {})
XMAS_START_MONTH = xmas_cfg.get("start_month", 12)
XMAS_START_DAY   = xmas_cfg.get("start_day", 1)
XMAS_END_MONTH   = xmas_cfg.get("end_month", 12)
XMAS_END_DAY     = xmas_cfg.get("end_day", 25)

PERIOD_PHRASES = config["period_phrases"]
def get_period_phrase(period):
    return PERIOD_PHRASES.get(period, f"in the {period}")

# Convert paths to be relative to BASE_DIR
COVER_IMAGE_DIR = resolve_path(config["directories"]["cover_images"], BASE_DIR)
FONTS_DIR       = resolve_path(config["directories"]["fonts"], BASE_DIR)
MOOD_MAP_PATH   = resolve_path(config["files"]["mood_map"], BASE_DIR)

FONT_MAIN_PATH   = resolve_path(config["fonts"]["main"], FONTS_DIR)
FONT_MELODAY_PATH = resolve_path(config["fonts"]["meloday"], FONTS_DIR)


time_periods = config["time_periods"]

plex = PlexServer(PLEX_URL, PLEX_TOKEN, timeout=60)

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
        val = getattr(t, "tag", None) or getattr(t, "title", None)
        if isinstance(val, str) and val.strip().casefold() == n:
            return True
    return False

def has_label(obj, label_name: str) -> bool:
    """True if Plex item has a Labels tag equal to label_name (case-insensitive)."""
    try:
        return _tag_list_contains(getattr(obj, "labels", None), label_name)
    except Exception:
        return False

def _album_in_collection(album, collection_name: str) -> bool:
    """True if an Album is in the given Plex Collection name."""
    try:
        # collections are sometimes not populated until reload()
        try:
            album.reload()
        except Exception:
            pass
        return _tag_list_contains(getattr(album, "collections", None), collection_name)
    except Exception:
        return False

def filter_excluded_tracks(tracks, now=None):
    """Apply 'noshare' + seasonal Christmas collection exclusions to a list of Plex Track objects."""
    if not tracks:
        return []
    now = now or datetime.now()
    in_xmas = _in_christmas_window(now)

    album_cache = {}
    cleaned = []
    for t in tracks:
        # Track-level label exclusion
        if has_label(t, EXCLUDE_LABEL_NAME):
            continue

        # Album-level checks (cached)
        album = None
        parent_key = getattr(t, "parentRatingKey", None)
        if parent_key:
            if parent_key in album_cache:
                album = album_cache[parent_key]
            else:
                try:
                    album = plex.fetchItem(parent_key)
                except Exception:
                    album = None
                album_cache[parent_key] = album

        if album and has_label(album, EXCLUDE_LABEL_NAME):
            continue

        if (not in_xmas) and album and _album_in_collection(album, CHRISTMAS_COLLECTION_NAME):
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


def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # Normalize common punctuation variants
    s = (s.replace("’", "'").replace("‘", "'")
            .replace("–", "-").replace("—", "-").replace("‐", "-"))
    s = s.casefold()
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
        a = track.artist() if callable(getattr(track, "artist", None)) else None
        at = getattr(a, "title", None) if a else None
        if isinstance(at, str) and at.strip() and not ((at or '').strip().casefold() in {'various artists','various'}):
            return at.strip()
    except Exception:
        pass

    # If we still can't tell, return whatever grandparentTitle we had (even if VA), or 'unknown'.
    if isinstance(gp, str) and gp.strip():
        return gp.strip()
    return "unknown"



# --- Dedup helpers: prefer studio albums over compilations/soundtracks ---
_FEAT_SPLIT_RE = re.compile(r"\s*(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)

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
    }
    try:
        album = track.album() if callable(getattr(track, "album", None)) else None
        if album is not None:
            meta["album_title"] = (getattr(album, "title", meta["album_title"]) or meta["album_title"]).strip()
            meta["album_artist"] = (getattr(album, "parentTitle", "") or "").strip()
            # Plex may expose subtype/albumType differently depending on server/version.
            meta["album_subtype"] = (getattr(album, "subtype", "") or getattr(album, "albumType", "") or "").strip()
            # Fallback: query raw metadata XML and extract <Subformat tag="...">
            if not meta["album_subtype"]:
                try:
                    # album.key is usually like "/library/metadata/<ratingKey>"
                    data = plex.query(getattr(album, "key", f"/library/metadata/{album.ratingKey}"))
                    sub = data.find(".//Subformat")
                    if sub is not None:
                        tag = sub.get("tag", "") or ""
                        meta["album_subtype"] = tag.strip()
                except Exception:
                    pass
    except Exception:
        pass

    _album_meta_cache[cache_key] = meta
    return meta

_COMPILATION_TITLE_RE = re.compile(
    r"\b("
    r"soundtrack|ost|o\.s\.t\.|"
    r"original\s+(?:motion\s+picture\s+)?soundtrack|"
    r"motion\s+picture\s+soundtrack|"
    r"music\s+from\s+the\s+(?:motion\s+picture|film)|"
    r"various\s+artists|"
    r"greatest\s+hits|best\s+of|"
    r"anthology|compilation"
    r"triple\s*j"
    r")\b",
    re.IGNORECASE,
)
_LIVE_TITLE_RE = re.compile(r"\blive\b|unplugged|concert", re.IGNORECASE)

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
    # 1) Prefer studio albums
    a_studio = is_studio_album(a)
    b_studio = is_studio_album(b)
    if a_studio != b_studio:
        return a if a_studio else b

    # 2) Prefer the "plain/original" title within the same dedupe key
    a_rank = title_variant_rank(a)
    b_rank = title_variant_rank(b)
    if a_rank != b_rank:
        return a if a_rank < b_rank else b

    # 3) Prefer non-remix album titles (e.g., 'Changa' over 'Go Bang (Remixes) - EP')
    a_pen = remix_album_penalty(a)
    b_pen = remix_album_penalty(b)
    if a_pen != b_pen:
        return a if a_pen < b_pen else b

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
    print(f"[{bar}] {percent:3d}%  {message}")

# ---------------------------------------------------------------------
def get_current_time_period():
    """Determine which daypart the current hour belongs to."""
    current_hour = datetime.now().hour

    for period, details in time_periods.items():
        if current_hour in details["hours"]:
            return period

    # Fallback if not found
    return "Late Night"

def load_descriptor_map(filepath="moodmap.json"):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading descriptor dictionary: {e}")
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
    """Fetch tracks from Plex history that match the current daypart. while excluding recently played tracks."""
    music_section = plex.library.section(MUSIC_LIBRARY)
    now = datetime.now()
    period_hours = set(time_periods[period]["hours"])

    history_start = now - timedelta(days=HISTORY_LOOKBACK_DAYS)
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    history_entries = [
        entry for entry in music_section.history(mindate=history_start)
        if entry.viewedAt and entry.viewedAt.hour in period_hours
    ]
    excluded_entries = [
        entry for entry in music_section.history(mindate=exclude_start)
        if entry.viewedAt
    ]

    excluded_keys = {entry.ratingKey for entry in excluded_entries}
    filtered_tracks = [
        entry for entry in history_entries
        if entry.ratingKey not in excluded_keys
    ]

    # If no historical tracks found, fallback
    if not filtered_tracks:
        fallback_entries = [
            entry for entry in music_section.history(mindate=history_start)
            if entry.viewedAt and entry.viewedAt.hour in period_hours
               and entry.ratingKey not in excluded_keys
        ]
        if fallback_entries:
            filtered_tracks = fallback_entries

    # Apply label + seasonal collection exclusions
    filtered_tracks = filter_excluded_tracks(filtered_tracks, now=now)

    # Genre balancing
    track_play_counts = Counter()
    genre_count = Counter()
    for track in filtered_tracks:
        track_play_counts[track] += 1
        for genre in (getattr(track, "genres", None) or []):
            genre_count[str(genre)] += 1

    sorted_tracks = sorted(filtered_tracks, key=lambda t: track_play_counts[t], reverse=True)
    split_index = max(1, len(sorted_tracks) // 4)
    popular_tracks = sorted_tracks[:split_index]
    rare_tracks = sorted_tracks[split_index:]

    balanced_selection = (
        random.sample(rare_tracks, min(len(rare_tracks), int(MAX_TRACKS * 0.75)))
        + random.sample(popular_tracks, min(len(popular_tracks), int(MAX_TRACKS * 0.25)))
    )

    if genre_count:
        most_common_genre, most_common_count = genre_count.most_common(1)[0]
        max_genre_limit = int(MAX_TRACKS * 0.25)
        if most_common_count > max_genre_limit:
            def _has_genre(track, genre_str):
                return any(str(g) == genre_str for g in (getattr(track, "genres", None) or []))
            balanced_selection = (
                [t for t in balanced_selection if not _has_genre(t, most_common_genre)][:max_genre_limit]
                + [t for t in balanced_selection if _has_genre(t, most_common_genre)][:max_genre_limit]
            )

    return balanced_selection, excluded_keys

def filter_low_rated_tracks(tracks):
    """Filter out tracks/albums/artists with a 2-star rating (rating <= 4), skipping ephemeral tracks that lack ratingKey or parentRatingKey."""
    filtered = []
    for track in tracks:
        try:
            if not getattr(track, "ratingKey", None) or not getattr(track, "parentRatingKey", None):
                continue
            artist = track.artist() if callable(getattr(track, "artist", None)) else None
            artist_rating = getattr(artist, "userRating", None) if artist else None
            album = plex.fetchItem(track.parentRatingKey)
            album_rating = getattr(album, "userRating", None) if album else None
            track_rating = getattr(track, "userRating", None)

            if artist_rating is not None and artist_rating <= 4:
                continue
            if album_rating is not None and album_rating <= 4:
                continue
            if track_rating is not None and track_rating <= 4:
                continue

            filtered.append(track)
        except Exception:
            pass
    return filtered

def clean_title(title):
    version_keywords = [
        "extended", "deluxe", "remaster", "remastered", "live", "acoustic", "edit",
        "version", "anniversary", "special edition", "radio edit", "album version",
        "original mix", "remix", "mix", "dub", "instrumental", "karaoke", "cover",
        "rework", "re-edit", "bootleg", "vip", "session", "alternate", "take",
        "mix cut", "cut", "dj mix"
    ]

    featuring_patterns = [
        r"\(feat\.?.*?\)", r"\[feat\.?.*?\]", r"\(ft\.?.*?\)", r"\[ft\.?.*?\]",
        r"\bfeat\.?\s+\w+", r"\bfeaturing\s+\w+", r"\bft\.?\s+\w+",
        r" - .*mix$", r" - .*dub$", r" - .*remix$", r" - .*edit$", r" - .*version$"
    ]

    title_clean = title.lower().strip()

    # 1) Remove feat/ft patterns and dash-suffix patterns first
    for pattern in featuring_patterns:
        title_clean = re.sub(pattern, "", title_clean, flags=re.IGNORECASE).strip()

    # Build a regex that matches any version keyword
    kw_alt = "|".join(
        re.escape(k).replace(r"\ ", r"\s+")
        for k in sorted(version_keywords, key=len, reverse=True)
    )

    # 2) Remove parenthetical/bracketed chunks that contain version keywords
    title_clean = re.sub(rf"\(\s*[^)]*(?:{kw_alt})[^)]*\)\s*", " ", title_clean, flags=re.IGNORECASE)
    title_clean = re.sub(rf"\[\s*[^\]]*(?:{kw_alt})[^\]]*\]\s*", " ", title_clean, flags=re.IGNORECASE)

    # 3) Remove remaining standalone version keywords (not in brackets)
    for keyword in sorted(version_keywords, key=len, reverse=True):
        title_clean = re.sub(rf"\b{re.escape(keyword)}\b", " ", title_clean, flags=re.IGNORECASE).strip()

    # 4) Cleanup
    title_clean = re.sub(r"\(\s*\)", "", title_clean)   # remove empty ()
    title_clean = re.sub(r"\[\s*\]", "", title_clean)   # remove empty []
    title_clean = re.sub(r"\s+", " ", title_clean).strip()
    title_clean = re.sub(r"[\s-]+$", "", title_clean)   # trim trailing spaces or hyphens

    return title_clean


def process_tracks(tracks):
    """
    Process tracks to remove duplicates and balance artist/genre representation.

    Dedup strategy:
        - Key on (cleaned title, primary track artist) so the same recording on
        different albums (studio vs compilation/soundtrack) collapses.
        - When duplicates exist, keep the "better" copy (prefer studio album, then
        artist album over Various Artists, then higher userRating).
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
        except Exception:
            continue

    deduped_tracks = [best_by_key[k] for k in key_order]

    # Phase 2: enforce artist + genre balance
    unique_tracks = []
    artist_count = Counter()
    genre_count = Counter()
    artist_limit = round(MAX_TRACKS * 0.05)

    for track in deduped_tracks:
        try:
            artist_name = norm_text(primary_artist(track_artist_name(track)))
            if artist_count[artist_name] >= artist_limit:
                continue

            track_genre = track.genres[0] if getattr(track, "genres", None) else "Unknown"
            if genre_count[track_genre] >= int(MAX_TRACKS * 0.15):
                continue

            artist_count[artist_name] += 1
            genre_count[track_genre] += 1
            unique_tracks.append(track)
        except Exception:
            continue

    return unique_tracks

def fetch_sonically_similar_tracks(reference_tracks, excluded_keys=None):
    """Fetch sonically similar tracks while ensuring recently played tracks are removed."""
    similar_tracks = []
    now = datetime.now()
    exclude_start = now - timedelta(days=EXCLUDE_PLAYED_DAYS)

    for track in reference_tracks:
        try:
            similars = track.sonicallySimilar(limit=SONIC_SIMILAR_LIMIT)

            # Ensure we're filtering by last play date
            filtered_similars = []
            for s in similars:
                last_played = getattr(s, "lastViewedAt", None)

                # Exclude if it was played recently
                if last_played and last_played >= exclude_start:
                    print(f"EXCLUDED (sonicallySimilar): {s.title} - Last played {last_played}")
                    continue

                # Exclude if it's already in the excluded keys
                if excluded_keys and s.ratingKey in excluded_keys:
                    print(f"EXCLUDED (recent play): {s.title} - In excluded keys")
                    continue

                filtered_similars.append(s)

            # Run deduplication before adding similar tracks
            filtered_similars = filter_excluded_tracks(filtered_similars, now=now)
            final_similars = process_tracks(filter_low_rated_tracks(filtered_similars))
            similar_tracks.extend(final_similars)

        except Exception as e:
            print(f"Error fetching sonically similar tracks: {e}")
            pass

    return similar_tracks


# --- OPTIMIZED SONIC SORTING LOGIC ---
def get_sonic_distance(track_a_key, track_b_key, similarity_cache, limit=20):
    """Returns a bidirectional distance score. Lower is more similar."""
    penalty = limit * 20
    # Distance from A to B
    rank_ab = similarity_cache.get(track_a_key, {}).get(track_b_key, penalty)
    # Distance from B to A
    rank_ba = similarity_cache.get(track_b_key, {}).get(track_a_key, penalty)
    return rank_ab + rank_ba

def sort_by_sonic_similarity_refined(tracks, first_track, last_track, limit=20):
    """Combines Double-Ended Greedy + 2-opt refinement for the smoothest possible flow."""
    if not tracks:
        return []

    # 1. Pre-fetch similarity data and cache artist names for separation penalty
    all_involved = tracks + [first_track, last_track]
    similarity_cache = {}
    artist_map = {}
    for track in all_involved:
        # Cache normalized primary artist for separation penalty
        artist_map[track.ratingKey] = norm_text(primary_artist(track_artist_name(track)))
        try:
            sims = track.sonicallySimilar(limit=limit)
            similarity_cache[track.ratingKey] = {t.ratingKey: i for i, t in enumerate(sims)}
        except Exception:
            similarity_cache[track.ratingKey] = {}

    def get_adj_dist(ka, kb):
        base_dist = get_sonic_distance(ka, kb, similarity_cache, limit)
        # Apply heavy penalty if artists are the same to minimize back-to-back clustering
        if artist_map.get(ka) == artist_map.get(kb):
            return base_dist + (limit * 100)
        return base_dist

    # 2. Artist-Aware Greedy Initialization (Starting from 'first_track')
    remaining = list(tracks)
    path = []
    current_key = first_track.ratingKey
    
    while remaining:
        next_track = min(
            remaining,
            key=lambda t: get_adj_dist(current_key, t.ratingKey)
        )
        path.append(next_track)
        remaining.remove(next_track)
        current_key = next_track.ratingKey

    # 3. 2-opt Refinement using artist-aware distance
    def calculate_total_distance(p):
        # Distance from first to start of middle
        d = get_adj_dist(first_track.ratingKey, p[0].ratingKey)
        # Internal middle transitions
        for i in range(len(p) - 1):
            d += get_adj_dist(p[i].ratingKey, p[i+1].ratingKey)
        # Distance from end of middle to last
        d += get_adj_dist(p[-1].ratingKey, last_track.ratingKey)
        return d

    improved = True
    while improved:
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 1, len(path)):
                # Flip the segment and see if the artist-aware total distance improves
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                if calculate_total_distance(new_path) < calculate_total_distance(path):
                    path = new_path
                    improved = True
    return path
# ------------------------------------


def generate_playlist_title_and_description(period, tracks):
    descriptor_map = load_descriptor_map("moodmap.json")
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
    
    # Ensure highlight styles are filled
    while len(highlight_styles) < max_styles:
        additional = sorted_genres + sorted_moods
        for s in additional:
            if s not in highlight_styles:
                highlight_styles.append(s)
            if len(highlight_styles) == max_styles:
                break

    if second_common_mood:
        description = (
            f"You listened to {most_common_mood} and {most_common_genre} tracks on {day_name} {period_phrase}. "
            f"Here's some {', '.join(highlight_styles[:-1])}, and {highlight_styles[-1]} tracks as well."
        )
    else:
        description = (
            f"You listened to {most_common_genre} and {most_common_mood} tracks on {day_name} {period_phrase}. "
            f"Here's some {', '.join(highlight_styles[:-1])}, and {highlight_styles[-1]} tracks as well."
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
        print(f"[WARN] apply_text_to_cover failed: {e}")
        return image_path

import traceback

def create_or_update_playlist(name, tracks, description, cover_file):
    existing_playlist = next((pl for pl in plex.playlists() if str(getattr(pl, "title", "")).startswith("Meloday for ")), None)
    valid_tracks = [t for t in tracks if getattr(t, "ratingKey", None)]
    
    if not valid_tracks:
        raise RuntimeError("No valid tracks to add (missing ratingKey).")

    if existing_playlist:
        existing_playlist.removeItems(existing_playlist.items())
        existing_playlist.addItems(valid_tracks)
        existing_playlist.editTitle(name)
        existing_playlist.editSummary(description)
        playlist_obj = existing_playlist
    else:
        playlist_obj = plex.createPlaylist(name, items=valid_tracks)
        playlist_obj.editSummary(description)

    print(f"[OK] Playlist updated: {name} | items: {playlist_obj.leafCount}")

    cover_path = os.path.join(COVER_IMAGE_DIR, cover_file)
    if os.path.exists(cover_path):
        try:
            new_cover = apply_text_to_cover(cover_path, name)
            playlist_obj.uploadPoster(filepath=new_cover)
            print(f"[OK] Uploaded poster: {new_cover}")
        except Exception:
            print("[WARN] Poster upload failed (playlist still created):")
            traceback.print_exc()
    else:
        print(f"[WARN] Cover file not found: {cover_path}")

def find_first_and_last_tracks(tracks, period):
    if not tracks: return None, None
    valid_hours = set(time_periods[period]["hours"])
    sorted_tracks = sorted(tracks, key=lambda t: t.lastViewedAt or datetime.max)
    first = next((t for t in sorted_tracks if t.lastViewedAt and t.lastViewedAt.hour in valid_hours), sorted_tracks[0])
    last = next((t for t in reversed(sorted_tracks) if t.lastViewedAt and t.lastViewedAt.hour in valid_hours), sorted_tracks[-1])
    return first, last

def main():
    # Step 0% - Start
    print_status(0, "Starting track selection...")
    period = get_current_time_period()
    print_status(10, f"Period: {period}")

    # Step 1: Fetch historical (Guarantee ~30% historical)
    print_status(20, "Fetching historical tracks...")
    historical, excluded_keys = fetch_historical_tracks(period)
    guaranteed = random.sample(historical, min(int(MAX_TRACKS * 0.3), len(historical)))

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
        additional = process_tracks(random.sample(more_h, min(MAX_TRACKS - len(final_tracks), len(more_h))) + more_s)
        final_tracks = process_tracks(final_tracks + additional)[:MAX_TRACKS]
        if not additional: break

    print_status(70, "Finding first & last historical tracks...")
    first, last = find_first_and_last_tracks(final_tracks[:MAX_TRACKS], period)
    middle = [t for t in final_tracks[:MAX_TRACKS] if t not in {first, last}]

    # Step 4: Sonic sort (GREEDY)
    if middle and first and last:
        print_status(80, "Double-ended 2-opt sonic refinement...")
        middle = sort_by_sonic_similarity_refined(middle, first, last, limit=SONIC_SIMILAR_LIMIT)

    final_ordered_tracks = [first] + middle + [last] if first and last else final_tracks[:MAX_TRACKS]

    # Step 5: Playlist Update
    print_status(90, "Creating/Updating playlist...")
    title, desc = generate_playlist_title_and_description(period, final_ordered_tracks)
    create_or_update_playlist(title, final_ordered_tracks, desc, time_periods[period]['cover'])
    print_status(100, "Playlist creation/update complete!")

if __name__ == "__main__":
    main()