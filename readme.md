# Meloday: A daylist for Plex  

## Overview
Meloday is a script that **automatically updates one playlist throughout the day**, evolving with your listening habits. Inspired by Spotify’s **daylist**, it pulls tracks from your **Plex listening history**, finds **patterns in what you like at different times**, and builds a mix that feels both **familiar and fresh**—without getting repetitive.

Each update brings a **new cover, a new name, and a refreshed mix of tracks** that fit the current moment. It also reaches into a **custom-built mood map** filled with different ways to describe the playlist’s vibe, so the names always stay interesting.

> **Important:** Meloday is designed to maintain **a single evolving playlist**. It updates that playlist over and over, rather than creating a new one each time.

---

## What It Does  
- **Builds playlists based on your past listening habits** – It looks at what you’ve played before at the same time of day.  
- **Avoids repeats** – Tracks you’ve played too recently won’t be included.  
- **Finds sonically similar tracks** – It expands your playlist with music that fits the vibe.  
- **Uses Plex metadata, not AI** – Everything is based on your existing library and Plex’s own data.  
- **Automatically updates itself** – No manual curation needed.  
- **Applies custom covers and descriptions** – The playlist gets a new look each time it updates.  
- **Gets creative with playlist names** – It pulls words from a mood map for extra variety.  

## What It *Doesn’t* Do  
- **It doesn’t add songs from outside your Plex library** – Everything comes from what you already have.  
- **It doesn’t use AI recommendations** – There’s no external algorithm picking tracks, just your own listening history.  
- **It doesn’t force specific genres or moods** – Your past listening shapes each playlist organically.  
- **It doesn’t replace your other playlists** – This just runs alongside whatever else you have in Plex.  

---

## How It Works  

### 1) Identifies the Current Time Period  
- Meloday divides the day into **morning, afternoon, evening, night, etc.**  
- The script figures out the current time and selects the right time period.  

### 2) Pulls Tracks from Your Listening History  
- It looks at **what you’ve played at this time of day in the past**.  
- If a track was **played too recently**, it’s skipped to keep things fresh.  

### 3) Finds Sonically Similar Tracks  
- It uses Plex’s **built-in sonic similarity** to find related songs.  
- This helps the playlist feel cohesive instead of just being a random shuffle.  

### 4) Exclusions (Privacy + Seasonal Rules)  
Meloday supports a couple of simple “keep this out of the mix” rules:

- **Label exclusion:** If a track or album has a Plex label, such as **`noshare`** (which I use), or your label of choice, it won’t be used.  
  This is an easy way to keep private, niche, or “not for this playlist” music out of rotation.

- **Christmas collection rule:** If you keep holiday music in a Plex **album collection** named **`Christmas Music`**, for example, Meloday will:
  - allow it during **December 1st → December 25th**, and  
  - keep it out of the playlist for the rest of the year  
  (so it doesn’t keep resurfacing in July).

These are configurable in the config file

### 5) Filters & Organises Tracks (Less repetition, better variety)  
Meloday tries hard to keep the playlist from feeling like “the same 15 songs again”:

- **Skips low-rated music:** Anything rated **1–2 stars** is ignored.
- **Uses a mix of popular and rare tracks:** So you still get favourites, but also surprises.
- **Keeps things balanced:** It tries to avoid the playlist being dominated by a single artist or one general style.

#### How it handles duplicates and “different copies” of the same song  
If the same song exists multiple times in your library (for example:  
a studio album version + a compilation version + a live version + a remix), Meloday will usually only keep **one**.

When choosing which copy to keep, it generally prioritises:
- **Studio album versions** over compilations, live albums, remixes, etc.  
- The **clean/original title** (instead of “remastered”, “radio edit”, “live”, and so on)  
- A more “normal” release over things like “Various Artists” compilations  
- **Higher-rated** versions (if you’ve rated them)

The goal is simple: keep the playlist from being filled with repeated “almost the same song” entries.

### 6) Sorts the Playlist for a Natural Flow  
Meloday doesn’t just throw the final list into a shuffle. It tries to order tracks so the playlist feels like a smooth journey:

- The **first track** is the earliest one you’ve played in that time period.  
- The **last track** is the most recent one you’ve played in that time period.  
- Then it arranges everything in between to make **smooth sonic transitions**.

#### Performs a “two-track swap” improvement pass  
After the playlist is in a decent order, Meloday does an extra improvement pass where it looks for small changes that make transitions smoother.

A simple way to think of it:
- If two tracks end up next to each other as **A → B**, Meloday will sometimes try flipping them to **B → A**  
- If the flip creates better transitions on both sides, it keeps the change

This little “swap test” is repeated across the playlist to gently polish the flow without completely rearranging everything.

### 7) Creates a Playlist Title & Description  
Every playlist gets a **unique, descriptive name** based on what you’ve been listening to. Meloday doesn’t just pull from a basic list of moods—it taps into a **custom-built mood map** that expands common moods into more creative variations.

For example, if the playlist has a **cheerful vibe**, it won’t just call it "Cheerful." Instead, it might use words like: Joyous, Sunny, Happy, Upbeat, or Jovial. Or, if it leans a bit more **quirky**, it might get a title with words like: Eccentric, Unconventional, Odd, or Whimsical.

This means **every playlist name feels different**, even if the mood stays similar, so maybe a *Brash Vibrant Lo-Fi Study Wednesday Evening* is in your future.

### 8) Applies a Cover & Updates the Playlist in Plex  
- The cover image changes depending on the time of day.  
- The script applies a **text overlay** to customise the cover.  
- The playlist is updated with the new tracks, title, and description.  

---

## Best Mileage  
Meloday works best with **larger music libraries**. Since it pulls from **your own past listening**, the more variety you have, the better the playlists will be.

- If your **library is small**, Meloday might start repeating songs more often, creating a **feedback loop** where the same tracks show up frequently.  
- If you **haven't rated your tracks**, it will still work, but if you take the time to rate songs (1–5 stars), Meloday will be able to **avoid low-rated content** and refine selections over time.  
- Playlist generation *should* work for just about any size you make it, but a larger size will no doubt take longer to generate.  
- Meloday was **tested on a library of only about 13,500 tracks**, so your mileage may vary on significantly smaller or larger collections.  

---

## Who Made This?  
This was forked from https://github.com/trackstacker/meloday, and then improved by me, with a little bit of help now and again from Google Gemini. I've enjoyed improving this, and seeing how the daylists look in Plex. I hope you enjoy it too.

---

## Installation

### 1) Install dependencies:
pip install -r requirements.txt  

This will install:
- `plexapi` (for communicating with Plex)
- `Pillow` (for handling playlist cover images)
- `pyyaml` (for reading the config file)

### 2) Configure environment:
Edit the `config.yml` file to set your Plex server details:

plex:  
  url: "http://localhost:32400"             # Replace with your Plex URL  
  token: "YOUR_PLEX_TOKEN"                  # Replace with your Plex authentication token - [How to find it](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)  
  music_library: "Music"                    # Replace with the name of your Plex music library  
  exclude_label: "noshare"                  # The name of the label used to exclude things  
  christmas_collection: "Christmas Music"   # The name of the Christmas album collection  

Modify playlist settings if needed (*recommended to try it initially with these settings*):

playlist:  
  exclude_played_days: 3  # Ignore tracks played in the last X days  
  history_lookback_days: 60  # How many days of listening history to analyse  
  max_tracks: 50  # Maximum number of tracks in the playlist  
  sonic_similar_limit: 20  # How wide it looks when expanding with “similar” tracks (higher = more variety)  

seasonal:  
  christmas:  
    start_month: 12                                 # The month Christmas music can start being included  
    start_day: 1                                    # The day of the start month  
    end_month: 12                                   # The month Christmas music should be excluded again  
    end_day: 25                                     # The day Christmas music should be excluded again  

### 3) Run the script:
python meloday.py  

Your playlist should now be updated in Plex.

---

## Automating
This can be automated by using something like Windows Task Scheduler, or cron, depending on your platform.
