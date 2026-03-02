# Meloday: A Daylist for Plex

## Overview

Meloday is a script that **automatically updates one playlist throughout the day**, evolving with your listening habits. Inspired by Spotify's **Daylist**, it pulls tracks from your **Plex listening history**, finds **patterns in what you like at different times**, and builds a mix that feels both **familiar and fresh** — without getting repetitive.

Each update brings a **new cover, a new name, and a refreshed mix of tracks** that fit the current moment. Playlist names are generated from a **custom mood map** full of creative, evocative language, so no two updates sound the same.

> **Important:** Meloday is designed to maintain **a single evolving playlist**. It updates that playlist over and over throughout the day, rather than creating a new one each time.

---

## What It Does

- **Builds playlists from your listening history** — It looks at what you've played at this time of day before
- **Anchors the mood to the time of day** — Splits the day into seven periods, each with its own pool of history and cover art
- **Avoids repeats** — Recently played tracks are skipped to keep things fresh
- **Finds sonically similar tracks** — Expands the playlist with music that fits the current vibe
- **Bridges sonic gaps** — Inserts connecting tracks where the playlist would otherwise jump between sonically distant songs
- **Balances variety** — Caps how much any single artist, style, genre, or mood can dominate the playlist
- **Optional acoustic analysis** — Integrates Essentia to enhance sonic matching with BPM, musical key, energy level, and production era
- **Automatically tunes itself** — An included optimiser analyses your library and listening history to recommend the best config settings
- **Generates creative titles** — Pulls from a custom mood map for names like *Meloday • Nostalgic Lo-Fi & Indie Folk for Tuesday Morning*
- **Applies custom covers and descriptions** — The playlist gets a new look every time it updates
- **Runs unattended** — No manual curation needed; schedule it and forget about it

## What It *Doesn't* Do

- **Doesn't add songs from outside your Plex library** — Everything comes from what you already own
- **Doesn't use external AI recommendations** — No third-party algorithm, just your own listening history and Plex's sonic data
- **Doesn't force specific genres or moods** — Your past listening shapes each playlist organically
- **Doesn't replace your other playlists** — Runs quietly alongside whatever else you have in Plex

---

## How It Works

### 1. Identifies the Current Time Period

Meloday divides the day into **seven named periods**, each mapped to a set of hours:

| Period | Hours |
| --- | --- |
| Dawn | 3 – 5 |
| Early Morning | 6 – 8 |
| Morning | 9 – 11 |
| Afternoon | 12 – 15 |
| Evening | 16 – 18 |
| Night | 19 – 21 |
| Late Night | 22 – 2 |

The periods and their hours are fully configurable in `config.yml`.

### 2. Pulls Tracks from Your Listening History

- Searches your Plex play history for tracks played during **this time of day** within the configured lookback window
- Skips tracks played **too recently** to keep things fresh
- If no history matches the current period, falls back gracefully — first to **adjacent time periods**, then to your full history — so the playlist always has something to work with

### 3. Finds Sonically Similar Tracks

- Uses Plex's **built-in sonic similarity engine** to find related songs for each seed track from history
- This is what makes the playlist feel cohesive rather than just a random shuffle of past plays
- The candidate pool size is tunable (`sonic_similarity_limit`) — the optimiser will set this based on your library's actual density

### 4. Bridges Sonic Gaps

After sorting, Meloday scans for points in the playlist where two consecutive tracks are sonically distant. When it finds a gap, it inserts a **bridge track** — a song that sits acoustically between the two, smoothing out what would otherwise be a jarring jump.

Bridging can be disabled in `config.yml`, or set to **smart truncation** mode, which keeps the final playlist within `max_tracks` even after bridges are added.

### 5. Applies Exclusions

**Label exclusion:** Any track or album carrying one of the configured Plex exclusion labels is excluded from the playlist entirely. Accepts a single label or a list of labels. Useful for keeping private, niche, or "not for this playlist" content out of rotation.

**Seasonal rules:** If you keep holiday music in a named Plex collection (e.g. `Christmas Music`), Meloday will only include it during a configured date window — so it doesn't surface in July.

**Rating filter:** Tracks rated 1–2 stars are skipped.

### 6. Balances Variety

Meloday enforces several diversity caps to prevent the playlist being dominated by a single artist, style, or mood:

- **Style ratio** — The primary diversity axis. Limits how many tracks can share the same style tag (e.g. "Indie Rock", "Emo"). Tracks are checked against up to `style_tag_depth` of their style tags.
- **Genre ratio** — A fallback cap that applies only to tracks with no style tags.
- **Artist ratio** — Limits how many tracks can come from the same artist.
- **Mood ratio** — Limits how many tracks can share the same primary mood, preventing any single vibe from taking over the whole playlist.
- **Historical ratio** — Caps how much of the playlist comes from direct listening history, leaving room for sonic discovery.

#### Smart Deduplication

If the same song exists multiple times in your library (studio album + compilation + live recording + remix), Meloday keeps only the best copy. It prioritises:

- **Studio album versions** over compilations, live albums, remixes, etc.
- **Clean, original titles** over "remastered", "live", "radio edit", etc.
- **Higher-rated versions** when ratings exist

### 7. Sorts the Playlist for Natural Flow

Meloday doesn't shuffle. It arranges tracks so the playlist feels like a smooth journey:

- The **first track** is the earliest you've played in that time period
- The **last track** is the most recent you've played in that time period
- Everything in between is ordered to **minimise sonic distance between adjacent tracks**

After ordering, it runs a **2-opt improvement pass** — testing pairwise swaps across the playlist and keeping any swap that creates smoother transitions on both sides.

### 8. Creates a Title and Description

Every playlist gets a unique name built from three elements: an informal descriptor drawn from the dominant mood via a custom mood map, the two most common style tags in the playlist, and the current day and time period. The format is `Meloday • {descriptor} {style1} & {style2} for {day} {period}` — for example, *Meloday • Nostalgic Lo-Fi & Indie Folk for Tuesday Morning*.

The mood map translates mood labels into colloquial variations — "Cheerful" might become *Sunny* or *Jovial*, "Introspective" might become *Wistful* or *Pensive* — so the title reads naturally rather than sounding like a metadata dump. The description uses the formal mood name alongside the same style tags to give a more complete picture of the playlist's character.

### 9. Applies a Cover and Updates Plex

- A time-of-day cover image is selected
- A text overlay with the playlist name is applied
- The playlist in Plex is updated with the new tracks, title, description, and cover

---

## The Metadata Cache

`pre_analyze.py` scans your Plex library and builds a local cache of track metadata — styles, moods, genres, and year — that Meloday and the optimiser use for diversity balancing and config recommendations. Running it is worthwhile for any library, regardless of whether you use Essentia.

```bash
python utilities/pre_analyze.py
```

The cache is incremental — tracks already analysed are skipped on subsequent runs. The first run may take a while on large libraries; subsequent runs only process new or changed tracks.

### Optional Enhancement: Essentia Acoustic Analysis

Meloday also integrates with **[Essentia](https://essentia.upf.edu/)**, an open-source acoustic analysis library. When enabled, it enriches the cache with real audio features that further improve sonic matching:

- **BPM / Tempo** — matches tracks by their actual tempo
- **Musical key** — favours harmonically compatible transitions
- **Energy level** — maintains consistent loudness and intensity across the playlist
- **Production era** — keeps the sonic palette coherent across decades

To enable, set `essentia: enabled: true` in `config.yml` and re-run `pre_analyze.py`. It will upgrade existing metadata-only cache entries with acoustic data without re-fetching the Plex metadata.

See the [Essentia installation guide](https://essentia.upf.edu/installing.html) if the pip install fails on your platform.

### Maintaining the Cache

Run `meloday_admin.py` periodically to keep the cache clean:

```bash
python utilities/meloday_admin.py
```

This removes entries for tracks that no longer exist in your Plex library (orphan cleanup), removes any corrupt entries, and reclaims disk space.

---

## Optimising Your Config

`meloday_optimizer.py` analyses your library, metadata cache, and listening history to recommend the best configuration values for your specific setup:

```bash
python utilities/meloday_optimizer.py
```

By default it samples 2,000 tracks (or 5% of your library, whichever is larger) for the sonic neighbourhood analysis, but uses **all tracks in the metadata cache** for diversity calculations. You can pass a sample size or `ALL` to audit the full library:

```bash
python utilities/meloday_optimizer.py 5000
python utilities/meloday_optimizer.py ALL
```

The optimiser recommends values for:

- `sonic_similarity_limit` — calibrated to your library's actual sonic density
- `historical_ratio`, `style_ratio`, `genre_ratio`, `artist_ratio`, `mood_ratio` — calibrated to your library's genre/style/mood diversity
- `style_tag_depth` — set from the median number of style tags per track in your cache
- `exclude_played_days`, `history_lookback_days` — balanced to your listening intensity and library size
- `bpm_weight`, `key_weight`, `energy_weight`, `era_weight` — calibrated to the actual variation in your library (requires Essentia cache)

Set `auto_apply_optimization: true` in `config.yml` to have the optimiser write its recommendations directly to your config automatically.

### Sonic Density Audit

`limit_validator.py` provides a deeper audit of your `sonic_similarity_limit` setting, scanning the full library and producing a per-track CSV report showing how many sonic neighbours each track has at each distance bucket:

```bash
python utilities/limit_validator.py
```

Use this to check whether raising your limit would actually yield more candidates, or whether most tracks are already returning everything Plex has.

`sonic_islands.py` identifies your most sonically isolated **artists** — those whose tracks have the fewest sonic neighbours on average. It exports the top 5% most isolated to a CSV, which is useful for understanding which artists tend to resist bridging:

```bash
python utilities/sonic_islands.py
```

`sonic_similar.py` is a low-level inspector — it fetches and prints the raw sonic similarity results for a single track (including the numeric distance values Plex returns), which is useful for debugging why a specific track isn't getting good bridge candidates:

```bash
python utilities/sonic_similar.py --ratingkey=12345
python utilities/sonic_similar.py --ratingkey=12345 --limit=100 --maxdistance=0.20
```

---

## Best Mileage

Meloday works best with **larger, well-tagged music libraries**.

- **Library size** — The more tracks you have, the more options Meloday has for sonic variety and bridging. It's been tested on libraries ranging from ~13,500 to significantly larger collections.
- **Listening history** — The more consistently you play music through Plex, the stronger the time-of-day personalisation becomes.
- **Track ratings** — Not required, but rating tracks (1–5 stars) lets Meloday filter out music you don't enjoy.
- **Metadata cache** — Running `pre_analyze.py` on your library improves diversity balancing and optimiser recommendations even without Essentia. With Essentia enabled it also improves sonic matching quality and bridging accuracy.
- **Run the optimiser** — Default config values are conservative starting points. Running `meloday_optimizer.py` after building a reasonable cache will give you values tuned to your actual library.

---

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `plexapi`, `Pillow`, `pyyaml`, `ruamel.yaml`, `tqdm`, `psutil`, `requests`

**Essentia** (optional, for acoustic analysis) is listed in `requirements.txt` but may require a separate install step depending on your platform. See the [Essentia installation guide](https://essentia.upf.edu/installing.html) if the pip install fails.

### 2. Configure your environment

Edit `config.yml` with your Plex server details:

```yaml
plex:
  url: "http://localhost:32400"           # Your Plex server URL
  token: "YOUR_PLEX_TOKEN"               # Your Plex auth token — see link below
  music_library: "Music"                 # Name of your Plex music library
  exclude_label: "noshare"               # Plex label used to exclude tracks/albums
  christmas_collection: "Christmas Music" # Name of your holiday music collection
```

[How to find your Plex token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)

### 3. Run the script

```bash
python meloday.py
```

Your playlist will be created or updated in Plex.

---

## Configuration Reference

### `plex:`

| Key | Description |
| --- | --- |
| `url` | URL of your Plex Media Server |
| `token` | Your Plex API authentication token |
| `music_library` | Name of your Plex music library section |
| `exclude_label` | Plex label (or list of labels) — tracks/albums with any of these labels are excluded. Accepts a single string (`"noshare"`) or a list (`["noshare", "private"]`) |
| `christmas_collection` | Plex collection name for seasonal holiday music |

### `essentia:`

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Enable Essentia acoustic matching. Requires `pre_analyze.py` to have been run. |
| `cache_path` | `assets/essentia_cache.db` | Path to the SQLite acoustic cache |
| `bpm_weight` | `0.15` | Influence of tempo on track distance scoring |
| `key_weight` | `0.15` | Influence of harmonic key on track distance scoring |
| `energy_weight` | `0.10` | Influence of energy/loudness on track distance scoring |
| `era_weight` | `0.05` | Influence of production era (year) on track distance scoring |
| `path_mapping` | `{}` | Optional path remapping if your script and Plex server use different file paths |

### `bridging:`

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `true` | Insert bridge tracks to smooth out sonic gaps in the playlist |
| `smart_truncation` | `false` | If `true`, trim the playlist to stay within `max_tracks` after bridging. If `false`, bridging may exceed `max_tracks` to preserve sonic flow. |

### `playlist:`

| Key | Default | Description |
| --- | --- | --- |
| `auto_apply_optimization` | `false` | Automatically write optimiser recommendations to this config file |
| `exclude_played_days` | `3` | Skip tracks played within this many days |
| `history_lookback_days` | `60` | How far back to look in your play history |
| `max_tracks` | `50` | Target playlist length |
| `historical_ratio` | `0.40` | Maximum proportion of tracks sourced from play history |
| `style_ratio` | `0.20` | Maximum proportion of tracks sharing a single style tag |
| `style_tag_depth` | `1` | How many of each track's style tags to check. `1` = primary style only; higher = multi-dimensional balancing. Set by the optimiser. |
| `genre_ratio` | `0.12` | Fallback diversity cap — only applies to tracks with no style tags |
| `artist_ratio` | `0.05` | Maximum proportion of tracks from a single artist |
| `mood_ratio` | `0.35` | Maximum proportion of tracks sharing the same primary mood |
| `sonic_similarity_limit` | `100` | Candidate pool size per seed track for sonic matching. Set by the optimiser based on your library's density. |
| `sonic_similarity_distance` | `0.20` | Maximum similarity distance for sonic matching (lower = stricter). Plex's maximum is `0.25`; the default of `0.20` excludes the loosest matches. |

### `seasonal:`

Configure the date window for your holiday music collection:

```yaml
seasonal:
  christmas:
    start_month: 12
    start_day: 1
    end_month: 12
    end_day: 25
```

### `time_periods:`

Seven named periods covering the full day. Each has a list of `hours` (0–23) and a `cover` image filename. Fully customisable — you can rename periods, change their hours, or add your own.

---

## Automating

Meloday is designed to run at the **start of each time period**, so the playlist refreshes once when the mood naturally shifts rather than continuously throughout the day. The utility scripts run on their own separate schedules.

Here's an example cron setup (macOS/Linux):

```bash
# --- DAILY MAINTENANCE ---
# Analyse new tracks added to Plex (11:00 PM)
0 23 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/pre_analyze.py

# Clean and compact the Essentia cache (12:30 AM)
30 0 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_admin.py

# --- MONTHLY CALIBRATION ---
# Recalibrate config values against the full library (1st of the month, 12:45 AM)
45 0 1 * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_optimizer.py ALL

# --- PLAYLIST GENERATION ---
# Refresh at the start of each time period (Dawn, Early Morning, Morning,
# Afternoon, Evening, Night, Late Night)
0 3,6,9,12,16,19,22 * * * /path/to/.venv/bin/python /path/to/meloday/meloday.py
```

On **Windows**, use Task Scheduler to create equivalent triggers for each of the above.

---

## Who Made This?

Forked from [trackstacker/meloday](https://github.com/trackstacker/meloday) and significantly extended with Essentia acoustic analysis, multi-dimensional diversity balancing, sonic bridging, an automatic optimiser, and various performance improvements — with assistance from [Claude Code](https://claude.ai/claude-code).
