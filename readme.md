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
- **Generates supplementary taste-intelligence playlists** — An optional companion script adds Spotify-analogue playlists (On Repeat, Discover Weekly, Daily Mixes, and more) to your Plex library on their own schedule

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
| Dawn | 5 – 6 |
| Early Morning | 7 – 8 |
| Morning | 9 – 11 |
| Afternoon | 12 – 16 |
| Evening | 17 – 20 |
| Night | 21 – 23 |
| Late Night | 0 – 4 |

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

Every playlist gets a unique name built from three elements: an informal descriptor drawn from the dominant mood via a custom mood map, the two most common style tags in the playlist, and the current day and time period. The format is `Meloday • {descriptor} {style1} & {style2} for {day} {period}` — for example, *Meloday • Nostalgic Lo-Fi & Indie Folk for Tuesday morning*.

Period names use natural phrasing in the playlist title — e.g. *for Tuesday morning*, *for late Tuesday night*, *for Tuesday at dawn* — and are capitalised on the cover art. Both are configurable via `period_phrases` (used in the description) and `title_period_names` (used in the title and cover) in `config.yml`.

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

> **Note:** Tracks longer than 30 minutes (DJ mixes, continuous albums) are automatically skipped for acoustic analysis — Essentia's rhythm extractor can't process files of that length. They are still cached with metadata and can appear in playlists; they just won't contribute to BPM, key, or energy matching.

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

## Extras Playlists

`meloday_extras.py` is a companion script that generates Spotify-analogue "taste intelligence" playlists using the same Plex API and Essentia cache as the main engine. They run on their own schedule, entirely independently of the time-of-day playlist.

Each playlist gets a **programmatically generated cover** in the same style as the main Meloday covers — gradient background, right-aligned title, and "Meloday+" branding. Daily Mixes use a **2×2 album-art collage** tinted with the mix's colour, regenerated each run to reflect current track selection.

Star ratings feed into playlist selection throughout: tracks rated ≥ 3 stars are gently prioritised over unrated tracks in scoring and sorting. Tracks rated ≤ 2 stars are excluded everywhere. The neutral point is 2.5 stars (Plex internal value 5).

### Running the script

```bash
# Generate a specific playlist
python utilities/meloday_extras.py --playlist on_repeat

# Flags for mood mixes (see Mood Mixes section below)
python utilities/meloday_extras.py --playlist mood_mixes --reselect-moods
python utilities/meloday_extras.py --playlist mood_mixes --time-context
```

Available playlist IDs: `on_repeat`, `repeat_rewind`, `release_radar`, `discover_weekly`, `daily_mixes`, `rediscovery`, `time_capsule`, `deep_cuts`, `top_songs`, `all_time_favourites`, `mood_mixes`

### First-time setup

Run these commands in order to generate all playlists for the first time. `top_songs` is last because it fetches your full play history back to `top_songs_start_year` and can take several minutes.

```bash
python utilities/meloday_extras.py --playlist daily_mixes
python utilities/meloday_extras.py --playlist on_repeat
python utilities/meloday_extras.py --playlist repeat_rewind
python utilities/meloday_extras.py --playlist release_radar
python utilities/meloday_extras.py --playlist discover_weekly
python utilities/meloday_extras.py --playlist rediscovery
python utilities/meloday_extras.py --playlist time_capsule
python utilities/meloday_extras.py --playlist deep_cuts
python utilities/meloday_extras.py --playlist all_time_favourites
python utilities/meloday_extras.py --playlist mood_mixes
python utilities/meloday_extras.py --playlist mood_mixes --time-context
python utilities/meloday_extras.py --playlist top_songs
```

### The playlists

| Playlist | Name in Plex | Behaviour | Suggested refresh |
| --- | --- | --- | --- |
| On Repeat | Meloday+ On Repeat | Tracks played at least twice in the past 30 days, weighted by recency × frequency. Artist cap of 3. | Weekly (Monday) |
| Repeat Rewind | Meloday+ Repeat Rewind | Tracks that were in heavy rotation 4–10 weeks ago (min 2 plays) but haven't been played in the last 30 days — last month's On Repeat. | Weekly (Monday) |
| Release Radar | Meloday+ Release Radar | Tracks from recently released albums ranked by acoustic taste fit. Expands from a 14-day window in 7-day steps until 30 tracks are found. One album per artist. | Weekly (Friday) |
| Discover Weekly | Meloday+ Discover Weekly | ~70% tracks from artists you've never played, ~30% from familiar artists — all unheard. Scored by acoustic affinity. Includes an exploration slice of less-obvious picks. | Weekly (Monday) |
| Daily Mix 1–6 | Meloday+ Daily Mix 1–6 | Six 50-track mixes from k-means acoustic clustering of your listening history. Mix 1 is always your most-played cluster; numbering is stable across runs. | Daily |
| Rediscovery | Meloday+ Rediscovery | Tracks rated ≥ 3.5 stars or played ≥ 3 times that haven't been played in 6–24 months. Longest-neglected first. | Weekly (Sunday) |
| Time Capsule | Meloday+ Time Capsule | Music from your formative years. Set `birth_year` in config for era-anchored selection (ages 13–25); otherwise infers peak listening eras from history. | Weekly (Sunday) |
| Deep Cuts | Meloday+ Deep Cuts | Rarely-played tracks by your 15 most-listened artists in the past 6 months, ranked by similarity to each artist's well-played catalogue. Interleaved across artists. | Weekly (Sunday) |
| Top Songs YYYY | Meloday+ Top Songs 2024 | Annual most-played playlists, one per calendar year back to `top_songs_start_year`. Past years are generated once and never regenerated; only the current year refreshes. | Monthly |
| All-Time Favourites | Meloday+ All-Time Favourites | Your most-played tracks across your entire Plex history (uses `viewCount` — covers your full library lifetime, not just a history window). | Weekly (Sunday) |
| Mood Mixes | Meloday+ Workout, Focus, etc. | Up to 5 activity/mood playlists rotating from a pool of 14 profiles. See Mood Mixes below. | See below |

### Mood Mixes

14 acoustic profiles are defined. Up to 5 appear at any time, selected based on your listening habits, time of day, and weather. Each mix has 50 tracks.

**General profiles** (rotate weekly based on acoustic fit to your recent listening):
Workout, Running, Party, Happy Hits, Focus, Chill, Sad Songs

**Time-of-day profiles** (appear and disappear at precise boundary hours):

| Profile | Name in Plex | Active window |
| --- | --- | --- |
| morning | Meloday+ Good Morning | 5am – noon |
| dinner | Meloday+ Dinner | 5pm – 9pm |
| late_night | Meloday+ Late Night | 10pm – 2am |
| sleep | Meloday+ Sleep | 10pm – 4am |

**Weather profiles** (added/removed daily based on current conditions — requires `weather_location` in config):

| Profile | Name in Plex | Triggered when |
| --- | --- | --- |
| rainy_day | Meloday+ Rainy Day | Raining |
| sunny | Meloday+ Sunny | Clear and warm for the season |
| cosy | Meloday+ Cosy | Cold for the season |

Weather thresholds are season-adjusted using your location's latitude so a mild winter day doesn't trigger Cosy every day.

**Three operating modes** controlled by flags:

- No flag (daily 6am): refreshes content of existing general mixes; checks weather
- `--reselect-moods` (weekly Monday): re-scores all 14 profiles acoustically and rotates the general mix selection
- `--time-context` (at each time boundary): only adds/removes time-of-day mixes based on current hour; very fast (30-day history fetch only)

### Optional: Last.fm popularity ranking

Setting `lastfm_api_key` in the `extras:` config block enables Last.fm global play count data for **Release Radar** track selection — it will prefer the most-played tracks (typically singles) from each new album rather than the most acoustically central ones. Only the API key is needed; the secret is not required for read-only lookups.

Then add your key to the `extras:` config block. The integration degrades gracefully if the key is absent or a lookup fails.

### Scheduling

```bash
# Daily — content refresh + weather check
0 6 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist daily_mixes
0 6 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes

# Weekly (Mondays)
0 6 * * 1 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist on_repeat
0 6 * * 1 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist repeat_rewind
0 6 * * 1 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist discover_weekly
0 6 * * 1 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --reselect-moods

# Weekly (Fridays)
0 6 * * 5 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist release_radar

# Weekly (Sundays)
0 6 * * 0 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist rediscovery
0 6 * * 0 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist time_capsule
0 6 * * 0 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist deep_cuts
0 6 * * 0 /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist all_time_favourites

# Monthly (1st) — only regenerates the current year after first run
0 7 1 * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist top_songs

# Time-of-day mood mix boundaries
0 5  * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 12 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 17 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 21 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 22 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 2  * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
0 4  * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_extras.py --playlist mood_mixes --time-context
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

### `travel:`

An optional list of travel windows. When the current date falls within a window, Meloday uses the specified timezone for period detection and day naming instead of local server time. Handles DST automatically via IANA timezone names.

```yaml
travel:
  - timezone: "Europe/London"   # IANA timezone name
    start: "2025-12-10"         # First day away (YYYY-MM-DD)
    end: "2026-01-11"           # Last day away (YYYY-MM-DD)
  - timezone: "America/New_York"
    start: "2026-05-10"
    end: "2026-05-20"
```

Remove or leave empty when not travelling. Multiple trips can be listed and are matched in order.

### `cron:`

Controls whether Meloday manages its own playlist schedule in your crontab.

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `false` | Set to `true` to allow Meloday to manage the playlist generation cron entry |
| `python_path` | `""` | Path to the Python interpreter. Leave empty to use whichever interpreter is running the script (i.e. the active venv). |
| `script_path` | `""` | Path to `meloday.py`. Leave empty to auto-detect from the script's own location. |

When enabled, run `python utilities/meloday_cron.py` to install or update the schedule. See [Automating](#automating) below.

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

### `extras:`

Configuration for `meloday_extras.py`. All keys are optional with sensible defaults.

| Key | Default | Description |
| --- | --- | --- |
| `daily_mix_count` | `6` | Number of Daily Mix playlists to generate (1–6) |
| `discover_weekly_size` | `30` | Number of tracks in Discover Weekly |
| `release_radar_start_days` | `14` | Initial Release Radar window (days) |
| `release_radar_step_days` | `7` | Days to expand if below `release_radar_min_tracks` |
| `release_radar_min_tracks` | `30` | Keep expanding until this many tracks are found |
| `release_radar_max_days` | `90` | Hard cap on Release Radar window expansion |
| `birth_year` | *(unset)* | Your year of birth. Enables era-anchored Time Capsule (ages 13–25). If unset, peak listening eras are inferred from history. |
| `top_songs_start_year` | current year − 5 | Earliest year to generate a Top Songs playlist for. Set this to the year you started using Plex. |
| `mood_mix_count` | `5` | Number of mood/activity mixes to show at once (from 14 profiles). |
| `weather_location` | *(unset)* | City name, coordinates, or airport code (e.g. `"Melbourne"`). Enables weather-triggered mood mixes (Rainy Day, Sunny, Cosy). Uses wttr.in — no API key needed. |
| `lastfm_api_key` | *(unset)* | Last.fm API key. Enables popularity-ranked track selection in Release Radar. Only the key is needed (not the secret). |
| `lastfm_api_secret` | *(unset)* | Last.fm API secret. Not required for current functionality; reserved for future use. |

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

### `period_phrases:`

Controls how each period is referred to in the playlist **description** — e.g. `"from your late Thursday night listening"`. Supports a `{day}` placeholder for periods where the day name sits inside the phrase (e.g. `"late {day} night"`).

### `title_period_names:`

Controls how each period is referred to in the **playlist title** and **cover art**. Uses the same `{day}` placeholder convention as `period_phrases`. The title uses these values as-is (lowercase); the cover art capitalises them automatically.

---

## Automating

Meloday is designed to run at the **start of each time period**, so the playlist refreshes once when the mood naturally shifts rather than continuously throughout the day. The utility scripts run on their own separate schedules.

### Managed scheduling (recommended)

Your system's crontab is a list of scheduled commands — Meloday needs an entry there so it runs at the start of each time period. `meloday_cron.py` generates and maintains that entry for you, calculating the correct hours from your `time_periods` config. When a travel window is active, it converts each period's start hour from the destination timezone to your system clock so the playlist stays in sync with where you are.

**1. Enable it in `config.yml`:**

```yaml
cron:
  enabled: true
  python_path: ""
  script_path: ""
```

`python_path` is the full path to the Python interpreter to use in the cron entry — typically your venv's Python (run `which python` while the venv is active if you're unsure). Leave blank and it will use whichever interpreter is running the script, which is correct in most setups.

`script_path` is the full path to `meloday.py`. Leave blank to auto-detect it from `meloday_cron.py`'s own location — only set this explicitly if the files are in unexpected locations.

**2. Check the computed schedule:**

```bash
python utilities/meloday_cron.py --status
```

Shows the calculated run hours and the exact cron line that would be written, without touching your crontab. Use `--dry-run` instead to preview the full crontab output.

**3. Install or update:**

```bash
python utilities/meloday_cron.py
```

Writes a clearly marked block into your crontab. Safe to re-run at any time — it updates the existing block in place rather than adding duplicates.

**Optional — auto-switch on travel start/end:**

The above installs a static schedule. To have it shift automatically when a trip begins or ends, add one more entry to your crontab (`crontab -e`):

```bash
# Re-evaluate the playlist schedule each midnight
0 0 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_cron.py
```

Each midnight it rechecks whether a travel window is active and rewrites the schedule if needed — no manual intervention required when you leave or return.

### Maintenance and calibration entries

These are not managed by `meloday_cron.py` and should be added manually:

```bash
# --- DAILY MAINTENANCE ---
# Analyse new tracks added to Plex (11:00 PM)
0 23 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/pre_analyze.py

# Clean and compact the Essentia cache (12:30 AM)
30 0 * * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_admin.py

# --- MONTHLY CALIBRATION ---
# Recalibrate config values against the full library (1st of the month, 12:45 AM)
45 0 1 * * /path/to/.venv/bin/python /path/to/meloday/utilities/meloday_optimizer.py ALL
```

### Manual scheduling

If you'd prefer to manage the playlist entry yourself, the default schedule fires at the start of each time period:

```bash
# Refresh at the start of each time period
0 0,5,7,9,12,17,21 * * * /path/to/.venv/bin/python /path/to/meloday/meloday.py
```

On **Windows**, use Task Scheduler to create equivalent triggers.

---

## Who Made This?

Forked from [trackstacker/meloday](https://github.com/trackstacker/meloday) and significantly extended with Essentia acoustic analysis, multi-dimensional diversity balancing, sonic bridging, an automatic optimiser, and various performance improvements — with assistance from [Claude Code](https://claude.ai/claude-code).
