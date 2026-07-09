# Embedding / genre selection-quality audit — 2026-07-09

**Question:** does every mood/genre mix actually pick vibe-correct tracks ("rave_cave picks good rave songs")?

## Method
- **Scope:** all 251 non-showcase profiles (129 Discogs-gated + mood/weather/seasonal/time). Decade/geo showcases excluded (popularity-chosen by design).
- **Build:** each profile built once through the real `_build_mix_tracks` (dev cache, 50 tracks, Plex stubbed, no recently-played excludes, `excluded_album_keys` EMPTY — see festive/old_friends caveats).
- **Metrics per profile:** acoustic fit to the profile's own centroid (mean/p90 of `_acoustic_distance_to_centroid`); facet deltas (selected-mean vs target: bpm/danceability/arousal/vocal_presence); embedding self-cohesion (musicnn, + effnet for gated) mean/worst; instrumental-mix vocal violations; gate pool size; named worst tracks (both lenses — acoustic outliers AND lowest-cohesion, which catch different failures).
- **BPM octave-folding:** the centroid's BPM term is `((b-t)/200)^2` — weak and not octave-aware — and Essentia octave-doubles slow songs, so raw selected-BPM deltas mass-flag calm mixes falsely. All 69 raw BPM flags were re-measured with per-track folded delta `min(|b-t|,|b/2-t|,|2b-t|)`: **27 pure artifacts (folded <=15), 34 borderline, only 8 genuine tempo drifts**. (power_nap: raw +41 -> folded 11.7 — it IS picking slow songs; their stored BPMs are doubled.)

## Verdict: 15 SEVERE - 110 MODERATE - 126 OK

- **Healthy reference:** `rave_cave` (the motivating example) rates OK — fit 0.07 (tightest of the pilot), cohesion 0.954, facets on-target; its dance-pop outliers are the deliberate 50/50 anchors. `techno`, `indie_rock` also OK.
- **Dominant failure class: GATE LEAKS** — the Discogs audio classifier sprays low-confidence genre tags onto unrelated tracks, and gates without a per-profile confidence floor admit them (the same class as the electronic_radio pop-bleed, fixed earlier with `_PROFILE_TAG_FLOOR: 0.45`). The embedding-cohesion lens catches these precisely (an alien track scores 0.2-0.4 cosine vs the mix's own centroid).
- **Genuine tempo drifts (8):** dnb, hyperpop, swing_bigband, bebop, punk_energy, london_jungle, running, golden_afternoon — fast-genre mixes selecting materially slower material (or vice versa) even after octave-folding.
- **Instrumental violation:** `yoga_stretch` 22% vocal tracks (the other instrumental mixes are clean).
- **No structural starvation:** zero shortfalls (<45 tracks), zero thin gate pools (<150 candidates).

## SEVERE (act on these)
### swing_bigband  (jazz_lounge, gated)
- flags: tempo drift folded 34 (raw-40); vocal drift +0.26; alien track (coh 0.16)
- lowest-cohesion: 0.16 billie holiday — My Man
- lowest-cohesion: 0.35 frank sinatra — How Are Ya’ Fixed for Love?
- acoustic outlier: 0.21 stevie wonder — The Square
- acoustic outlier: 0.21 ace of base — He Decides
- **fix class:** Gate leak + genuine drift: slow vocal ballads (Billie Holiday 'My Man') and non-swing pop pass the gate. Raise per-profile Discogs confidence floor (`_PROFILE_TAG_FLOOR`) and/or nudge the centroid faster; evidence tracks below.

### chart_pop  (pop, gated)
- flags: danceability drift -0.23; alien track (coh 0.23)
- lowest-cohesion: 0.23 ramones — I Wanna Be Sedated
- lowest-cohesion: 0.37 pat benatar — Hit Me With Your Best Shot
- acoustic outlier: 0.17 nsync — Bye Bye Bye
- acoustic outlier: 0.17 ed sheeran — Bad Habits
- **fix class:** Gate leak: 'chart' tags admit punk/rock classics (Ramones 'I Wanna Be Sedated', Pat Benatar). Tighten positives or add a confidence floor; the danceability drift (-0.23) suggests the centroid also tolerates non-dance rock.

### melbourne_pubrock  (rock_classic, gated)
- flags: vocal drift +0.26; alien track (coh 0.36)
- lowest-cohesion: 0.36 tina arena — In Command
- lowest-cohesion: 0.57 tism — (He'll Never Be An) Ol' Man River
- acoustic outlier: 0.17 amyl and the sniffers — Snakes
- acoustic outlier: 0.15 magic dirt — Plastic Loveless Letter
- **fix class:** Gate leak: pop vocalists (Tina Arena) pass the pub-rock gate. Confidence floor or positives tweak.

### rockabilly_surf  (rock_classic, gated)
- flags: vocal drift +0.26; alien track (coh 0.33)
- lowest-cohesion: 0.33 the marvelettes — Please Mr. Postman
- lowest-cohesion: 0.62 chumbawamba — Always Tell the Voter What the Voter Wants t
- acoustic outlier: 0.18 the marvelettes — Please Mr. Postman
- acoustic outlier: 0.17 the who — Won't Get Fooled Again
- **fix class:** Gate leak: Motown (The Marvelettes) and novelty pop pass. Confidence floor / positives tweak.

### hyperpop  (electronic_edm_pop, gated)
- flags: tempo drift folded 28 (raw-31); danceability drift -0.22
- lowest-cohesion: 0.71 destinys child — Say My Name
- lowest-cohesion: 0.81 jennifer lopez — Jenny From the Block
- acoustic outlier: 0.20 kesha — Birthday Suit
- acoustic outlier: 0.19 sleigh bells — Young Legends
- **fix class:** Gate leak + drift: 90s/00s R&B-pop (Destiny's Child, J.Lo) passes the hyperpop gate; selections slower (folded 28) and less danceable (-0.22) than target. Floor + centroid check.

### melbourne_psych  (rock_psych, gated)
- flags: vocal drift +0.25; alien track (coh 0.23)  | notes: poor acoustic fit (p90 0.23, worst decile)
- lowest-cohesion: 0.23 olivia newtonjohn — Help Me Make It Through the Night
- lowest-cohesion: 0.43 bee gees — Indian Gin and Whisky Dry (mono)
- acoustic outlier: 0.26 spiderbait — King of the Northern
- acoustic outlier: 0.24 stepmother — Sick Thoughts
- **fix class:** Gate leak: Olivia Newton-John / Bee Gees pass the psych gate (Discogs sprays psych tags on 60s-70s pop). Confidence floor.

### melbourne_folk  (folk_acoustic, gated)
- flags: alien track (coh 0.20)  | notes: bpm raw+31 folded 18 (borderline)
- lowest-cohesion: 0.20 the smith street band — A Conversation With Billy Bragg About The Pu
- lowest-cohesion: 0.54 the seekers — When the Stars Begin to Fall
- acoustic outlier: 0.23 the avalanches — Park Music
- acoustic outlier: 0.22 the smith street band — A Conversation With Billy Bragg About The Pu
- **fix class:** Alien tracks: The Avalanches (plunderphonics) and punk-adjacent Smith Street Band in a folk mix. Floor / positives tweak.

### emo_poppunk  (rock_punk, gated)
- flags: alien track (coh 0.30)
- lowest-cohesion: 0.30 amy macdonald — This Is the Life (acoustic version)
- lowest-cohesion: 0.60 del amitri — I Won't Take the Blame (acoustic version)
- acoustic outlier: 0.20 foo fighters — Best of You
- acoustic outlier: 0.20 all time low — Break Out! Break Out!
- **fix class:** Gate leak: acoustic singer-songwriter covers (Amy Macdonald, Del Amitri acoustic) pass. Floor; consider excluding 'acoustic version' via is_alt_recording-style marker in this gate.

### festive  (festive, gated)
- flags: alien track (coh 0.30)
- lowest-cohesion: 0.30 james brown — Go Power at Christmas Time
- lowest-cohesion: 0.39 weezer — We Wish You a Merry Christmas
- acoustic outlier: 0.19 jive bunny the mastermixers — The Christmas Song
- acoustic outlier: 0.18 kylie minogue — It’s the Most Wonderful Time of the Year
- **fix class:** BY-DESIGN CAVEAT: festive gates on title/holiday signals, deliberately cross-genre (funk xmas vs pop xmas sound unalike) — low cohesion is expected here, not a defect. No fix recommended.

### london_mod  (rock_indie, gated)
- flags: alien track (coh 0.32)
- lowest-cohesion: 0.32 blue — Walk Away
- lowest-cohesion: 0.35 lulu — He’s Sure the Boy I Love
- acoustic outlier: 0.18 public service broadcasting — Signal 30
- acoustic outlier: 0.17 the bonzo dog band — I’m the Urban Spaceman
- **fix class:** Gate leak: boyband pop (Blue 'Walk Away') and 60s pop (Lulu) pass the mod gate. Confidence floor.

### swagger  (hiphop, gated)
- flags: alien track (coh 0.26)
- lowest-cohesion: 0.26 aretha franklin — Respect
- lowest-cohesion: 0.64 rita ora — Young, Single & Sexy
- acoustic outlier: 0.17 kelis — Roller Rink
- acoustic outlier: 0.17 eminem — So Bad
- **fix class:** Gate leak: soul classics (Aretha 'Respect') in a hip-hop swagger mix. Confidence floor or parent-set check.

### afrobeat  (world_latin, gated)
- flags: alien track (coh 0.26)
- lowest-cohesion: 0.26 ali farka touré — Doudou
- lowest-cohesion: 0.39 jax jones — Ring Ring (acoustic Room Session)
- acoustic outlier: 0.25 ali farka touré — Doudou
- acoustic outlier: 0.23 childish gambino — Happy Survival
- **fix class:** Partial leak: Ali Farka Touré is desert blues (adjacent, arguable); Jax Jones 'Ring Ring (acoustic)' is a clear leak. Floor.

### date_night  (romantic)
- flags: alien track (coh 0.30)  | notes: bpm raw+30 folded 16 (borderline)
- lowest-cohesion: 0.30 frank sinatra — Somethin’ Stupid
- lowest-cohesion: 0.46 the monkees — Jericho
- acoustic outlier: 0.23 michael bublé — Haven’t Met You Yet (live)
- acoustic outlier: 0.23 him — Gone With the Sin
- **fix class:** Mood mix (no gate): HIM 'Gone With the Sin' (gothic metal) via dark-romance tags; Sinatra cohesion outlier is age-diversity, arguably fine. Optional: mood-tag weighting tweak; low priority.

### melbourne_sunset  (pop, gated)
- flags: alien track (coh 0.18)
- lowest-cohesion: 0.18 nick cave the bad seeds — She Fell Away
- lowest-cohesion: 0.57 milly strange — Ghost
- acoustic outlier: 0.20 nick cave the bad seeds — She Fell Away
- acoustic outlier: 0.20 milly strange — Ghost
- **fix class:** Alien track: Nick Cave 'She Fell Away' (dark) in a sunset pop mix. Floor / centroid check.

### old_friends  (nostalgic_throwback)
- flags: alien track (coh 0.34)
- lowest-cohesion: 0.34 james brown — Soulful Christmas
- lowest-cohesion: 0.52 average white band — Let's Go Round Again (12" version)
- acoustic outlier: 0.18 bryan adams — You’ve Been a Friend to Me
- acoustic outlier: 0.17 one direction — Rock Me
- **fix class:** VERIFY-IN-PROD CAVEAT: the James Brown Christmas leak may be a sim artifact — the audit harness passed empty `excluded_album_keys`, while prod excludes the Christmas collection outside the seasonal window. Re-check on daedalus before acting.

## MODERATE (single-flag / borderline — review opportunistically)

| profile | flags / notes |
|---|---|
| acoustic_romance | bpm raw+39 folded 16 (borderline); poor acoustic fit (p90 0.25, worst decile) |
| ambient_drift | bpm raw+38 folded 12 (octave artifact) |
| angst_mix | alien track (coh 0.49); poor acoustic fit (p90 0.29, worst decile) |
| autumn_mix | bpm raw+28 folded 16 (borderline) |
| bebop | tempo drift folded 31 (raw-33) |
| bittersweet | bpm raw+32 folded 18 (borderline) |
| bluegrass | poor acoustic fit (p90 0.24, worst decile) |
| bossa_samba | bpm raw+31 folded 18 (borderline) |
| brunch_mix | bpm raw+33 folded 23 (borderline) |
| candlelight | bpm raw+34 folded 14 (octave artifact) |
| cathartic | alien track (coh 0.35) |
| chill | bpm raw+30 folded 15 (borderline); poor acoustic fit (p90 0.23, worst decile) |
| clear_night | bpm raw+28 folded 15 (borderline) |
| commute_mix | bpm raw+25 folded 21 (borderline) |
| cosy | bpm raw+31 folded 15 (borderline); poor acoustic fit (p90 0.23, worst decile) |
| dance_pop | danceability drift -0.26 |
| daydreaming | bpm raw+35 folded 14 (octave artifact); poor acoustic fit (p90 0.22, worst decile) |
| deep_reading | bpm raw+27 folded 16 (borderline) |
| defiant | alien track (coh 0.41); poor acoustic fit (p90 0.23, worst decile) |
| devotion | bpm raw+28 folded 13 (octave artifact) |
| dnb | tempo drift folded 40 (raw-40) |
| dreamy_mix | bpm raw+26 folded 16 (borderline) |
| euphoric | danceability drift -0.26 |
| evening_unwind | bpm raw+29 folded 16 (borderline) |
| festival_edm | danceability drift -0.23 |
| flirty | alien track (coh 0.37) |
| foggy | bpm raw+26 folded 17 (borderline) |
| folk_acoustic | bpm raw+28 folded 20 (borderline) |
| funk_disco | danceability drift -0.23 |
| garage_grunge | vocal drift +0.28; poor acoustic fit (p90 0.25, worst decile) |
| glasgow_house | danceability drift -0.24 |
| glasgow_postpunk | vocal drift +0.27 |
| glasgow_postrock | vocal drift +0.44; bpm raw+26 folded 20 (borderline); poor acoustic fit (p90 0.25, worst decile) |
| glasgow_soul | alien track (coh 0.47) |
| glasgow_underground | poor acoustic fit (p90 0.23, worst decile) |
| golden_afternoon | tempo drift folded 28 (raw+29) |
| gospel | alien track (coh 0.47) |
| grey_skies | bpm raw+27 folded 18 (borderline) |
| grief_release | bpm raw+39 folded 13 (octave artifact) |
| heartbreak | bpm raw+42 folded 14 (octave artifact) |
| heavy_riffs | vocal drift +0.30 |
| house_party | danceability drift -0.27 |
| indie_romance | bpm raw+28 folded 17 (borderline) |
| industrial | poor acoustic fit (p90 0.23, worst decile) |
| jazz_dinner | bpm raw+28 folded 19 (borderline) |
| late_night | bpm raw+29 folded 18 (borderline) |
| late_night_romance | bpm raw+32 folded 14 (octave artifact) |
| latin_heat | alien track (coh 0.40) |
| lazy_sunday | bpm raw+43 folded 13 (octave artifact); poor acoustic fit (p90 0.25, worst decile) |
| london_dub | bpm raw+51 folded 10 (octave artifact) |
| london_garage | danceability drift -0.20 |
| london_grime | poor acoustic fit (p90 0.22, worst decile) |
| london_jazz | poor acoustic fit (p90 0.24, worst decile) |
| london_jungle | tempo drift folded 35 (raw-40) |
| loved_up | bpm raw+27 folded 18 (borderline) |
| meditation | bpm raw+48 folded 12 (octave artifact); poor acoustic fit (p90 0.22, worst decile) |
| melancholy | bpm raw+34 folded 13 (octave artifact) |
| melbourne_club | danceability drift -0.28 |
| melbourne_garagepunk | vocal drift +0.26 |
| melbourne_postpunk | vocal drift +0.26; poor acoustic fit (p90 0.22, worst decile) |
| melbourne_soul | alien track (coh 0.36) |
| moody_mix | bpm raw+27 folded 15 (borderline) |
| morning | alien track (coh 0.37) |
| neoclassical | bpm raw+43 folded 14 (octave artifact) |
| party | danceability drift -0.29 |
| party_throwback | danceability drift -0.22 |
| piano_romance | bpm raw+33 folded 15 (borderline) |
| post_grunge | alien track (coh 0.39) |
| post_rock | bpm raw+25 folded 18 (borderline) |
| power_ballads | bpm raw+44 folded 13 (octave artifact) |
| power_nap | bpm raw+41 folded 12 (octave artifact) |
| prog_rock | alien track (coh 0.46) |
| punk_energy | tempo drift folded 25 (raw-34); poor acoustic fit (p90 0.23, worst decile) |
| rainy_day | bpm raw+33 folded 14 (octave artifact) |
| reggae_dub | bpm raw+51 folded 10 (octave artifact) |
| restless | alien track (coh 0.44) |
| road_trip | alien track (coh 0.47) |
| romantic_dinner | bpm raw+28 folded 16 (borderline) |
| romantic_jazz | bpm raw+43 folded 13 (octave artifact); poor acoustic fit (p90 0.24, worst decile) |
| romantic_mix | alien track (coh 0.43); bpm raw+31 folded 14 (octave artifact) |
| running | tempo drift folded 29 (raw-30) |
| sad_bangers | danceability drift -0.22; poor acoustic fit (p90 0.22, worst decile) |
| serene | bpm raw+30 folded 15 (octave artifact); poor acoustic fit (p90 0.22, worst decile) |
| singalong | alien track (coh 0.38) |
| ska | alien track (coh 0.48) |
| sleep | danceability drift +0.20; bpm raw+45 folded 11 (octave artifact) |
| slow_burn | bpm raw+26 folded 15 (octave artifact) |
| slow_dance | bpm raw+43 folded 12 (octave artifact); poor acoustic fit (p90 0.23, worst decile) |
| spa_bath | bpm raw+44 folded 12 (octave artifact); poor acoustic fit (p90 0.23, worst decile) |
| spring_strings | bpm raw+26 folded 18 (borderline) |
| starlit | bpm raw+36 folded 15 (octave artifact); poor acoustic fit (p90 0.22, worst decile) |
| stoner_rock | alien track (coh 0.41) |
| stormy | bpm raw+37 folded 14 (octave artifact) |
| string_quartet | bpm raw+26 folded 16 (borderline) |
| strings_romance | bpm raw+32 folded 16 (borderline) |
| summer_breeze | alien track (coh 0.47) |
| summer_heat | danceability drift -0.21 |
| sunday_morning | bpm raw+31 folded 15 (borderline); poor acoustic fit (p90 0.24, worst decile) |
| sunrise | bpm raw+27 folded 21 (borderline) |
| sunset_mix | bpm raw+25 folded 19 (borderline) |
| throwback_anthems | alien track (coh 0.43) |
| trap_mode | vocal drift +0.32 |
| uk_garage | danceability drift -0.21 |
| vulnerable | bpm raw+29 folded 16 (borderline) |
| wedding_day | alien track (coh 0.40) |
| weekend_mix | alien track (coh 0.48) |
| wind_down | bpm raw+31 folded 14 (octave artifact) |
| winter_cosy | alien track (coh 0.49); bpm raw+29 folded 13 (octave artifact) |
| winter_mix | bpm raw+33 folded 16 (borderline) |
| yoga_stretch | vocals in instrumental mix (22%); poor acoustic fit (p90 0.24, worst decile) |

## OK (126)

acid_jazz, after_dark, after_hours_rnb, after_work, autumn_embers, autumn_jazz, autumn_leaves, autumn_rain, awe_wonder, bass_drop, beach_vibes, blue_hour, blues_bar, boom_bap, britpop_rock, campfire, celebration, celtic_folk, chiptune, cinematic_epic, classic_rock, confidence_boost, conscious_flow, cooking_mix, cookout, cool_down, country_roads, creative_flow, crush, deep_house, deep_work, dinner, dinner_party, downtempo, driving_mix, driving_singalong, emotional, empowering, first_date, focus, fresh_start, friday_feeling, friday_night, frosty, g_funk, game_night, gaming, gardening, glasgow_anthems, glasgow_bass, glasgow_dream, glasgow_folk, glasgow_indie, glasgow_late, glasgow_synth, golden_hour, happy, heatwave, hopeful, housework_hustle, indie_pop, indie_rock, lofi_beats, london_britpop, london_calling, london_dubstep, london_indie, london_soul, london_triphop, long_distance, love_songs, main_character, melbourne_dream, melbourne_hiphop, melbourne_indie, melbourne_techno, memory_lane, midnight, midweek_reset, modern_romance, monday_motivation, motown_soul, moving_on, neo_soul, night_drive, nostalgia_mix, outlaw_country, overcast, pre_party, psych_haze, rap_rock, rave_cave, school_days, situationship, smooth_jazz, snow_day, soundtracks, spring_acoustic, spring_bloom, spring_jangle, spring_mix, study_session, summer_evening, summer_roadtrip, summer_tropical, sunday_scaries, sunny, synth_pop, synthpop_romance, synthwave, techno, tender, three_am, trance, treat_yourself, triumphant, vaporwave, walking_mix, windy, winter_frost, winter_jazz, winter_nights, witching_hour, workout, yacht_rock, yearning

## Appendix — full metrics

| profile | tier | cat | n | pool | fit p90 | coh mean/worst | bpmΔ(raw/folded) | dncΔ | vocΔ |
|---|---|---|---|---|---|---|---|---|---|
| acid_jazz | OK | jazz_lounge | 50 | 628 | 0.1533 | 0.81/0.521 | +5 | -0.11 | -0.03 |
| acoustic_romance | MODERATE | romantic | 50 | 159652 | 0.2543 | 0.872/0.716 | +39/16 | +0.12 | +0.03 |
| afrobeat | SEVERE | world_latin | 50 | 201 | 0.1718 | 0.804/0.255 | +5 | -0.14 | -0.02 |
| after_dark | OK | time_of_day | 50 | — | 0.1806 | 0.841/0.573 | +11 | -0.17 | +0.15 |
| after_hours_rnb | OK | soul_funk_rnb | 50 | 1660 | 0.1901 | 0.903/0.664 | +8 | -0.06 | +0.11 |
| after_work | OK | calm_unwind | 50 | — | 0.1304 | 0.915/0.738 | +18 | -0.02 | +0.10 |
| ambient_drift | MODERATE | instrumental_cinematic | 50 | 14134 | 0.1506 | 0.973/0.929 | +38/12 | +0.15 | +0.04 |
| angst_mix | MODERATE | defiant_intense | 50 | — | 0.2862 | 0.756/0.493 | -8 | +0.00 | +0.03 |
| autumn_embers | OK | season_autumn | 50 | 4704 | 0.1486 | 0.82/0.52 | +14 | -0.06 | +0.22 |
| autumn_jazz | OK | season_autumn | 50 | 4986 | 0.1258 | 0.831/0.524 | +10 | +0.01 | -0.03 |
| autumn_leaves | OK | season_autumn | 50 | 6592 | 0.1265 | 0.908/0.702 | +19 | +0.02 | +0.05 |
| autumn_mix | MODERATE | season_autumn | 50 | — | 0.1528 | 0.942/0.874 | +28/16 | +0.07 | +0.01 |
| autumn_rain | OK | season_autumn | 50 | — | 0.11 | 0.923/0.75 | +18 | +0.05 | +0.08 |
| awe_wonder | OK | dreamy_ethereal | 50 | — | 0.1777 | 0.926/0.704 | +18 | +0.09 | +0.02 |
| bass_drop | OK | electronic_bass | 50 | 2971 | 0.1865 | 0.926/0.76 | -11 | -0.15 | +0.13 |
| beach_vibes | OK | weather | 50 | — | 0.2134 | 0.9/0.543 | +22 | -0.08 | +0.11 |
| bebop | MODERATE | jazz_lounge | 50 | 1243 | 0.2169 | 0.907/0.513 | -33/31 | -0.01 | +0.02 |
| bittersweet | MODERATE | melancholy_blue | 50 | — | 0.1174 | 0.914/0.843 | +32/18 | +0.06 | +0.04 |
| blue_hour | OK | time_of_day | 50 | — | 0.1566 | 0.906/0.706 | +22 | +0.05 | -0.07 |
| bluegrass | MODERATE | country | 50 | 565 | 0.2356 | 0.835/0.557 | +5 | -0.08 | +0.14 |
| blues_bar | OK | rock_classic | 50 | 2226 | 0.1333 | 0.811/0.607 | +19 | -0.06 | +0.13 |
| boom_bap | OK | hiphop | 50 | 2918 | 0.1592 | 0.931/0.753 | +12 | -0.09 | +0.10 |
| bossa_samba | MODERATE | world_latin | 50 | 183 | 0.1977 | 0.843/0.678 | +31/18 | -0.07 | +0.07 |
| britpop_rock | OK | rock_indie | 50 | 6352 | 0.1439 | 0.874/0.659 | -0 | -0.14 | +0.21 |
| brunch_mix | MODERATE | happy_bright | 50 | — | 0.2127 | 0.875/0.557 | +33/23 | +0.03 | +0.16 |
| campfire | OK | folk_acoustic | 50 | 6839 | 0.1596 | 0.885/0.779 | +21 | +0.02 | +0.14 |
| candlelight | MODERATE | romantic | 50 | — | 0.2023 | 0.879/0.704 | +34/14 | +0.16 | +0.05 |
| cathartic | MODERATE | euphoric_triumphant | 50 | — | 0.1748 | 0.704/0.353 | +10 | -0.02 | +0.08 |
| celebration | OK | party_fun | 50 | — | 0.18 | 0.891/0.604 | -1 | -0.16 | +0.09 |
| celtic_folk | OK | folk_acoustic | 50 | 947 | 0.1809 | 0.839/0.682 | +10 | +0.01 | +0.10 |
| chart_pop | SEVERE | pop | 50 | 4879 | 0.1619 | 0.863/0.229 | +5 | -0.23 | +0.12 |
| chill | MODERATE | calm_unwind | 50 | — | 0.2305 | 0.92/0.628 | +30/15 | +0.01 | -0.13 |
| chiptune | OK | electronic_edm_pop | 50 | 532 | 0.1908 | 0.914/0.784 | -0 | -0.10 | +0.13 |
| cinematic_epic | OK | instrumental_cinematic | 50 | 10806 | 0.1534 | 0.959/0.872 | +17 | +0.04 | -0.04 |
| classic_rock | OK | rock_classic | 50 | 7407 | 0.1276 | 0.816/0.584 | +3 | -0.09 | +0.22 |
| clear_night | MODERATE | weather | 50 | — | 0.1667 | 0.944/0.797 | +28/15 | +0.06 | -0.13 |
| commute_mix | MODERATE | driving | 50 | — | 0.1435 | 0.902/0.741 | +25/21 | -0.02 | +0.10 |
| confidence_boost | OK | euphoric_triumphant | 50 | — | 0.133 | 0.911/0.691 | +4 | -0.13 | +0.14 |
| conscious_flow | OK | hiphop | 50 | 2484 | 0.1472 | 0.891/0.652 | +15 | -0.06 | +0.05 |
| cooking_mix | OK | happy_bright | 50 | — | 0.1895 | 0.857/0.57 | +21 | -0.03 | +0.12 |
| cookout | OK | party_fun | 50 | — | 0.1673 | 0.878/0.501 | +11 | -0.03 | +0.16 |
| cool_down | OK | calm_unwind | 50 | — | 0.1944 | 0.93/0.847 | +23 | +0.07 | -0.12 |
| cosy | MODERATE | weather | 50 | — | 0.2265 | 0.932/0.822 | +31/15 | +0.13 | -0.12 |
| country_roads | OK | country | 50 | 1606 | 0.1794 | 0.832/0.555 | +17 | -0.08 | +0.14 |
| creative_flow | OK | focus_study | 50 | — | 0.1412 | 0.94/0.622 | +19 | -0.00 | -0.06 |
| crush | OK | romantic | 50 | — | 0.1418 | 0.857/0.544 | +17 | -0.03 | +0.18 |
| dance_pop | MODERATE | electronic_edm_pop | 50 | 5958 | 0.1517 | 0.915/0.717 | +1 | -0.26 | +0.09 |
| date_night | SEVERE | romantic | 50 | — | 0.2157 | 0.758/0.297 | +30/16 | +0.07 | +0.15 |
| daydreaming | MODERATE | dreamy_ethereal | 50 | — | 0.2231 | 0.906/0.706 | +35/14 | +0.11 | -0.19 |
| deep_house | OK | electronic_house_techno | 50 | 7554 | 0.0999 | 0.948/0.858 | -0 | -0.13 | -0.01 |
| deep_reading | MODERATE | focus_study | 50 | — | 0.154 | 0.941/0.797 | +27/16 | +0.10 | -0.06 |
| deep_work | OK | focus_study | 50 | — | 0.1791 | 0.964/0.903 | +24 | +0.12 | -0.03 |
| defiant | MODERATE | defiant_intense | 50 | — | 0.2264 | 0.767/0.409 | -2 | -0.02 | +0.20 |
| devotion | MODERATE | romantic | 50 | — | 0.167 | 0.871/0.581 | +28/13 | +0.03 | +0.13 |
| dinner | OK | romantic | 50 | — | 0.1917 | 0.879/0.632 | +22 | +0.07 | +0.06 |
| dinner_party | OK | jazz_lounge | 50 | 3613 | 0.1291 | 0.805/0.506 | +4 | -0.05 | +0.10 |
| dnb | MODERATE | electronic_bass | 50 | 2176 | 0.1635 | 0.951/0.841 | -40/40 | -0.09 | +0.01 |
| downtempo | OK | electronic_chill | 50 | 6653 | 0.1292 | 0.932/0.778 | +6 | -0.05 | -0.10 |
| dreamy_mix | MODERATE | dreamy_ethereal | 50 | — | 0.1756 | 0.906/0.74 | +26/16 | +0.06 | -0.07 |
| driving_mix | OK | driving | 50 | — | 0.1514 | 0.903/0.532 | +13 | -0.08 | +0.16 |
| driving_singalong | OK | driving | 50 | — | 0.152 | 0.87/0.505 | +9 | -0.09 | +0.01 |
| emo_poppunk | SEVERE | rock_punk | 50 | 4816 | 0.1919 | 0.909/0.3 | -5 | -0.09 | +0.16 |
| emotional | OK | melancholy_blue | 50 | — | 0.1051 | 0.92/0.798 | +10 | -0.03 | +0.07 |
| empowering | OK | euphoric_triumphant | 50 | — | 0.1189 | 0.899/0.763 | -3 | -0.07 | +0.10 |
| euphoric | MODERATE | euphoric_triumphant | 50 | — | 0.1887 | 0.906/0.697 | -3 | -0.26 | +0.19 |
| evening_unwind | MODERATE | calm_unwind | 50 | — | 0.1785 | 0.932/0.845 | +29/16 | +0.10 | -0.10 |
| festival_edm | MODERATE | electronic_edm_pop | 50 | 4705 | 0.1711 | 0.968/0.912 | -1 | -0.23 | +0.13 |
| festive | SEVERE | festive | 50 | 1729 | 0.1709 | 0.671/0.297 | +17 | -0.06 | +0.20 |
| first_date | OK | romantic | 50 | — | 0.1956 | 0.838/0.625 | +19 | -0.01 | +0.17 |
| flirty | MODERATE | romantic | 50 | — | 0.1526 | 0.865/0.371 | +16 | -0.05 | +0.13 |
| focus | OK | focus_study | 50 | — | 0.1777 | 0.968/0.881 | +22 | +0.08 | -0.06 |
| foggy | MODERATE | weather | 50 | — | 0.1646 | 0.931/0.794 | +26/17 | +0.05 | -0.14 |
| folk_acoustic | MODERATE | folk_acoustic | 50 | 6592 | 0.1574 | 0.922/0.79 | +28/20 | +0.07 | +0.07 |
| fresh_start | OK | happy_bright | 50 | — | 0.1898 | 0.851/0.705 | +20 | -0.04 | +0.19 |
| friday_feeling | OK | party_fun | 50 | — | 0.1534 | 0.885/0.704 | +6 | -0.09 | +0.22 |
| friday_night | OK | party_fun | 50 | — | 0.1403 | 0.896/0.667 | +8 | -0.12 | +0.17 |
| frosty | OK | weather | 50 | — | 0.1418 | 0.909/0.506 | +16 | +0.05 | -0.06 |
| funk_disco | MODERATE | soul_funk_rnb | 50 | 4714 | 0.1631 | 0.908/0.646 | +3 | -0.23 | +0.10 |
| g_funk | OK | hiphop | 50 | 1958 | 0.1457 | 0.935/0.828 | +20 | -0.13 | +0.18 |
| game_night | OK | party_fun | 50 | — | 0.1521 | 0.882/0.589 | +9 | -0.04 | +0.17 |
| gaming | OK | workout_energy | 50 | — | 0.1642 | 0.907/0.634 | +3 | -0.04 | +0.09 |
| garage_grunge | MODERATE | rock_punk | 50 | 517 | 0.2456 | 0.898/0.784 | -0 | -0.07 | +0.28 |
| gardening | OK | happy_bright | 50 | — | 0.1645 | 0.899/0.621 | +16 | +0.00 | +0.08 |
| glasgow_anthems | OK | rock_indie | 50 | 20882 | 0.1536 | 0.833/0.613 | +6 | -0.16 | +0.22 |
| glasgow_bass | OK | electronic_bass | 50 | 4281 | 0.1936 | 0.909/0.725 | -10 | -0.12 | -0.03 |
| glasgow_dream | OK | rock_psych | 50 | 4104 | 0.2008 | 0.857/0.561 | +23 | +0.08 | +0.21 |
| glasgow_folk | OK | folk_acoustic | 50 | 7042 | 0.164 | 0.852/0.568 | +24 | +0.06 | +0.07 |
| glasgow_house | MODERATE | electronic_house_techno | 50 | 18697 | 0.1943 | 0.861/0.606 | -4 | -0.24 | +0.18 |
| glasgow_indie | OK | rock_indie | 50 | 1846 | 0.1549 | 0.878/0.652 | +7 | -0.03 | +0.09 |
| glasgow_late | OK | electronic_chill | 50 | 6037 | 0.1952 | 0.88/0.74 | +18 | -0.04 | +0.01 |
| glasgow_postpunk | MODERATE | rock_punk | 50 | 8861 | 0.2126 | 0.808/0.506 | -6 | -0.09 | +0.27 |
| glasgow_postrock | MODERATE | instrumental_cinematic | 50 | 677 | 0.2533 | 0.86/0.7 | +26/20 | +0.08 | +0.44 |
| glasgow_soul | MODERATE | soul_funk_rnb | 50 | 1344 | 0.1534 | 0.791/0.466 | +10 | -0.10 | -0.03 |
| glasgow_synth | OK | pop | 50 | 14613 | 0.1545 | 0.896/0.562 | +9 | -0.07 | +0.15 |
| glasgow_underground | MODERATE | electronic_house_techno | 50 | 6215 | 0.2268 | 0.929/0.676 | -3 | -0.12 | +0.06 |
| golden_afternoon | MODERATE | time_of_day | 50 | — | 0.1876 | 0.942/0.851 | +29/28 | +0.12 | -0.04 |
| golden_hour | OK | time_of_day | 50 | — | 0.1415 | 0.926/0.827 | +22 | +0.00 | +0.10 |
| gospel | MODERATE | soul_funk_rnb | 50 | 319 | 0.2001 | 0.8/0.473 | +6 | -0.07 | +0.03 |
| grey_skies | MODERATE | weather | 50 | — | 0.1167 | 0.931/0.802 | +27/18 | +0.05 | +0.10 |
| grief_release | MODERATE | heartbreak_longing | 50 | — | 0.1828 | 0.907/0.781 | +39/13 | +0.11 | +0.09 |
| happy | OK | happy_bright | 50 | — | 0.1836 | 0.897/0.698 | +7 | -0.16 | +0.10 |
| heartbreak | MODERATE | heartbreak_longing | 50 | — | 0.2071 | 0.899/0.782 | +42/14 | +0.13 | -0.09 |
| heatwave | OK | weather | 50 | — | 0.1986 | 0.918/0.793 | +18 | -0.04 | -0.16 |
| heavy_riffs | MODERATE | rock_heavy | 50 | 8389 | 0.2151 | 0.879/0.579 | -9 | -0.05 | +0.30 |
| hopeful | OK | euphoric_triumphant | 50 | — | 0.1754 | 0.87/0.696 | +24 | +0.03 | +0.13 |
| house_party | MODERATE | electronic_house_techno | 50 | 17759 | 0.1533 | 0.954/0.866 | +3 | -0.27 | +0.09 |
| housework_hustle | OK | happy_bright | 50 | — | 0.1558 | 0.896/0.757 | +9 | -0.11 | +0.23 |
| hyperpop | SEVERE | electronic_edm_pop | 50 | 845 | 0.1824 | 0.911/0.711 | -31/28 | -0.22 | +0.20 |
| indie_pop | OK | pop | 50 | 2580 | 0.1467 | 0.916/0.678 | +5 | -0.10 | +0.14 |
| indie_rock | OK | rock_indie | 50 | 20882 | 0.1401 | 0.888/0.599 | +5 | -0.11 | +0.21 |
| indie_romance | MODERATE | romantic | 50 | 21292 | 0.1907 | 0.891/0.541 | +28/17 | +0.04 | +0.04 |
| industrial | MODERATE | electronic_house_techno | 50 | 1786 | 0.2291 | 0.944/0.86 | -7 | -0.08 | -0.06 |
| jazz_dinner | MODERATE | jazz_lounge | 50 | 2075 | 0.1674 | 0.941/0.753 | +28/19 | +0.10 | -0.10 |
| late_night | MODERATE | time_of_day | 50 | — | 0.1227 | 0.93/0.827 | +29/18 | -0.09 | +0.11 |
| late_night_romance | MODERATE | romantic | 50 | — | 0.1776 | 0.854/0.612 | +32/14 | +0.11 | +0.02 |
| latin_heat | MODERATE | world_latin | 50 | 4422 | 0.1688 | 0.909/0.397 | +16 | -0.18 | +0.11 |
| lazy_sunday | MODERATE | calm_unwind | 50 | — | 0.2485 | 0.909/0.679 | +43/13 | +0.12 | -0.14 |
| lofi_beats | OK | electronic_chill | 50 | 575 | 0.164 | 0.921/0.642 | +17 | -0.06 | +0.05 |
| london_britpop | OK | rock_indie | 50 | 6352 | 0.139 | 0.861/0.634 | +3 | -0.14 | +0.23 |
| london_calling | OK | rock_punk | 50 | 9297 | 0.1856 | 0.864/0.552 | -8 | -0.04 | +0.22 |
| london_dub | MODERATE | reggae_ska | 50 | 355 | 0.206 | 0.793/0.559 | +51/10 | -0.02 | +0.19 |
| london_dubstep | OK | electronic_bass | 50 | 2066 | 0.2029 | 0.928/0.671 | -9 | -0.10 | +0.07 |
| london_garage | MODERATE | electronic_bass | 50 | 3068 | 0.1701 | 0.926/0.811 | -7 | -0.20 | +0.19 |
| london_grime | MODERATE | electronic_bass | 50 | 1813 | 0.2197 | 0.9/0.699 | -14 | -0.17 | +0.16 |
| london_indie | OK | rock_indie | 50 | 20882 | 0.1595 | 0.856/0.619 | +1 | -0.12 | +0.22 |
| london_jazz | MODERATE | jazz_lounge | 50 | 1000 | 0.2362 | 0.875/0.581 | +12 | -0.11 | -0.19 |
| london_jungle | MODERATE | electronic_bass | 50 | 2549 | 0.2151 | 0.935/0.768 | -40/35 | -0.10 | +0.09 |
| london_mod | SEVERE | rock_indie | 50 | 1372 | 0.1554 | 0.733/0.32 | +8 | -0.13 | +0.22 |
| london_soul | OK | soul_funk_rnb | 50 | 1499 | 0.153 | 0.874/0.699 | +9 | -0.11 | -0.01 |
| london_triphop | OK | electronic_chill | 50 | 6373 | 0.1338 | 0.895/0.642 | +15 | -0.06 | +0.01 |
| long_distance | OK | romantic | 50 | — | 0.1184 | 0.93/0.843 | +24 | +0.04 | +0.10 |
| love_songs | OK | romantic | 50 | — | 0.1888 | 0.856/0.66 | +21 | +0.05 | +0.06 |
| loved_up | MODERATE | romantic | 50 | — | 0.1765 | 0.843/0.55 | +27/18 | -0.05 | +0.15 |
| main_character | OK | euphoric_triumphant | 50 | — | 0.1634 | 0.924/0.695 | +10 | -0.04 | +0.10 |
| meditation | MODERATE | wellness_sleep | 50 | — | 0.2179 | 0.954/0.805 | +48/12 | +0.18 | -0.06 |
| melancholy | MODERATE | melancholy_blue | 50 | — | 0.1386 | 0.933/0.85 | +34/13 | +0.15 | +0.04 |
| melbourne_club | MODERATE | electronic_house_techno | 50 | 20182 | 0.1803 | 0.92/0.797 | -6 | -0.28 | +0.17 |
| melbourne_dream | OK | rock_psych | 50 | 3947 | 0.1699 | 0.864/0.668 | +19 | +0.02 | +0.16 |
| melbourne_folk | SEVERE | folk_acoustic | 50 | 7042 | 0.1864 | 0.848/0.196 | +31/18 | +0.04 | +0.10 |
| melbourne_garagepunk | MODERATE | rock_punk | 50 | 10394 | 0.2169 | 0.899/0.688 | -15 | -0.07 | +0.26 |
| melbourne_hiphop | OK | hiphop | 50 | 6352 | 0.1363 | 0.905/0.698 | +14 | -0.10 | +0.06 |
| melbourne_indie | OK | rock_indie | 50 | 20882 | 0.1239 | 0.87/0.616 | +3 | -0.10 | +0.11 |
| melbourne_postpunk | MODERATE | rock_punk | 50 | 1482 | 0.2222 | 0.841/0.515 | +3 | -0.07 | +0.26 |
| melbourne_psych | SEVERE | rock_psych | 50 | 2769 | 0.2327 | 0.845/0.226 | -1 | -0.08 | +0.25 |
| melbourne_pubrock | SEVERE | rock_classic | 50 | 6721 | 0.1398 | 0.823/0.364 | +2 | -0.10 | +0.26 |
| melbourne_soul | MODERATE | soul_funk_rnb | 50 | 3467 | 0.1522 | 0.799/0.365 | +12 | -0.13 | +0.04 |
| melbourne_sunset | SEVERE | pop | 50 | 6971 | 0.1788 | 0.837/0.176 | +14 | -0.09 | +0.12 |
| melbourne_techno | OK | electronic_house_techno | 50 | 6931 | 0.198 | 0.938/0.808 | -3 | -0.13 | +0.08 |
| memory_lane | OK | nostalgic_throwback | 50 | — | 0.1393 | 0.89/0.748 | +18 | +0.02 | -0.01 |
| midnight | OK | time_of_day | 50 | — | 0.1086 | 0.913/0.79 | +16 | -0.01 | +0.01 |
| midweek_reset | OK | euphoric_triumphant | 50 | — | 0.1263 | 0.918/0.841 | +17 | -0.03 | +0.05 |
| modern_romance | OK | romantic | 50 | — | 0.192 | 0.846/0.682 | +25 | +0.03 | +0.10 |
| monday_motivation | OK | euphoric_triumphant | 50 | — | 0.126 | 0.909/0.639 | +17 | -0.03 | +0.10 |
| moody_mix | MODERATE | melancholy_blue | 50 | — | 0.1094 | 0.919/0.833 | +27/15 | +0.09 | +0.08 |
| morning | MODERATE | time_of_day | 50 | — | 0.2112 | 0.808/0.368 | +22 | +0.01 | +0.24 |
| motown_soul | OK | soul_funk_rnb | 50 | 3532 | 0.1816 | 0.818/0.572 | -4 | -0.16 | -0.03 |
| moving_on | OK | heartbreak_longing | 50 | — | 0.1267 | 0.861/0.511 | +8 | -0.07 | +0.06 |
| neo_soul | OK | soul_funk_rnb | 50 | 1489 | 0.192 | 0.899/0.744 | +22 | +0.02 | +0.03 |
| neoclassical | MODERATE | instrumental_cinematic | 50 | 3764 | 0.1764 | 0.927/0.731 | +43/14 | +0.14 | -0.07 |
| night_drive | OK | driving | 50 | — | 0.1342 | 0.911/0.769 | +15 | -0.13 | +0.15 |
| nostalgia_mix | OK | nostalgic_throwback | 50 | — | 0.135 | 0.926/0.809 | +23 | +0.04 | -0.02 |
| old_friends | SEVERE | nostalgic_throwback | 50 | — | 0.1639 | 0.805/0.34 | +20 | -0.03 | +0.16 |
| outlaw_country | OK | country | 50 | 1894 | 0.1366 | 0.777/0.545 | +19 | -0.04 | +0.12 |
| overcast | OK | weather | 50 | — | 0.0918 | 0.94/0.858 | +19 | +0.06 | +0.07 |
| party | MODERATE | party_fun | 50 | — | 0.1759 | 0.881/0.653 | -0 | -0.29 | +0.15 |
| party_throwback | MODERATE | party_fun | 50 | — | 0.1808 | 0.887/0.513 | -6 | -0.22 | +0.10 |
| piano_romance | MODERATE | romantic | 50 | 1717 | 0.1976 | 0.972/0.911 | +33/15 | +0.13 | -0.18 |
| post_grunge | MODERATE | rock_punk | 50 | 2551 | 0.2048 | 0.886/0.389 | -4 | -0.09 | +0.22 |
| post_rock | MODERATE | instrumental_cinematic | 50 | 677 | 0.1975 | 0.91/0.74 | +25/18 | +0.03 | +0.21 |
| power_ballads | MODERATE | euphoric_triumphant | 50 | — | 0.1795 | 0.827/0.58 | +44/13 | +0.10 | +0.02 |
| power_nap | MODERATE | wellness_sleep | 50 | — | 0.1813 | 0.953/0.875 | +41/12 | +0.15 | -0.02 |
| pre_party | OK | party_fun | 50 | — | 0.1399 | 0.921/0.683 | +3 | -0.17 | +0.19 |
| prog_rock | MODERATE | rock_psych | 50 | 3283 | 0.1308 | 0.82/0.46 | +8 | -0.06 | +0.15 |
| psych_haze | OK | rock_psych | 50 | 4140 | 0.1229 | 0.921/0.811 | +16 | -0.01 | +0.09 |
| punk_energy | MODERATE | rock_punk | 50 | 3212 | 0.2307 | 0.928/0.784 | -34/25 | -0.03 | +0.24 |
| rainy_day | MODERATE | weather | 50 | — | 0.1082 | 0.919/0.831 | +33/14 | +0.12 | +0.05 |
| rap_rock | OK | rock_heavy | 50 | 1799 | 0.2066 | 0.912/0.736 | -13 | -0.13 | +0.23 |
| rave_cave | OK | electronic_edm_pop | 50 | 1035 | 0.0946 | 0.954/0.761 | +11 | +0.01 | +0.02 |
| reggae_dub | MODERATE | reggae_ska | 50 | 379 | 0.1806 | 0.819/0.645 | +51/10 | -0.07 | +0.15 |
| restless | MODERATE | defiant_intense | 50 | — | 0.1829 | 0.809/0.44 | +3 | -0.02 | +0.20 |
| road_trip | MODERATE | driving | 50 | — | 0.1667 | 0.867/0.473 | +17 | -0.04 | +0.15 |
| rockabilly_surf | SEVERE | rock_classic | 50 | 1526 | 0.163 | 0.779/0.333 | -7 | -0.14 | +0.26 |
| romantic_dinner | MODERATE | romantic | 50 | — | 0.2035 | 0.865/0.675 | +28/16 | +0.11 | +0.15 |
| romantic_jazz | MODERATE | romantic | 50 | 1589 | 0.2449 | 0.862/0.704 | +43/13 | +0.13 | -0.05 |
| romantic_mix | MODERATE | romantic | 50 | — | 0.2156 | 0.825/0.43 | +31/14 | +0.06 | +0.09 |
| running | MODERATE | workout_energy | 50 | — | 0.1704 | 0.907/0.546 | -30/29 | -0.00 | +0.25 |
| sad_bangers | MODERATE | melancholy_blue | 50 | — | 0.2232 | 0.904/0.693 | -20 | -0.22 | +0.18 |
| school_days | OK | nostalgic_throwback | 50 | — | 0.1322 | 0.868/0.538 | +4 | -0.05 | +0.09 |
| serene | MODERATE | calm_unwind | 50 | — | 0.2171 | 0.917/0.729 | +30/15 | +0.10 | -0.05 |
| singalong | MODERATE | party_fun | 50 | — | 0.1265 | 0.841/0.383 | +8 | -0.06 | +0.12 |
| situationship | OK | romantic | 50 | — | 0.1114 | 0.93/0.837 | +17 | +0.02 | +0.10 |
| ska | MODERATE | reggae_ska | 50 | 531 | 0.1576 | 0.817/0.481 | -12 | -0.16 | +0.22 |
| sleep | MODERATE | wellness_sleep | 50 | — | 0.209 | 0.954/0.708 | +45/11 | +0.20 | -0.00 |
| slow_burn | MODERATE | romantic | 50 | — | 0.1582 | 0.896/0.778 | +26/15 | +0.03 | -0.07 |
| slow_dance | MODERATE | romantic | 50 | — | 0.2301 | 0.827/0.642 | +43/12 | +0.16 | +0.06 |
| smooth_jazz | OK | jazz_lounge | 50 | 787 | 0.1594 | 0.928/0.668 | +20 | -0.02 | -0.07 |
| snow_day | OK | weather | 50 | — | 0.208 | 0.88/0.653 | +16 | +0.05 | -0.09 |
| soundtracks | OK | instrumental_cinematic | 50 | 10212 | 0.1151 | 0.967/0.908 | +16 | +0.05 | -0.02 |
| spa_bath | MODERATE | wellness_sleep | 50 | — | 0.2302 | 0.945/0.852 | +44/12 | +0.13 | -0.10 |
| spring_acoustic | OK | season_spring | 50 | 6592 | 0.1445 | 0.867/0.521 | +17 | +0.03 | +0.10 |
| spring_bloom | OK | season_spring | 50 | — | 0.1614 | 0.86/0.544 | +20 | -0.05 | +0.18 |
| spring_jangle | OK | season_spring | 50 | 3947 | 0.1231 | 0.936/0.835 | +6 | -0.06 | +0.16 |
| spring_mix | OK | season_spring | 50 | — | 0.1998 | 0.881/0.674 | +19 | -0.01 | +0.14 |
| spring_strings | MODERATE | season_spring | 50 | 3764 | 0.1541 | 0.939/0.847 | +26/18 | +0.03 | -0.14 |
| starlit | MODERATE | time_of_day | 50 | — | 0.2205 | 0.935/0.739 | +36/15 | +0.09 | -0.19 |
| stoner_rock | MODERATE | rock_heavy | 50 | 621 | 0.2011 | 0.873/0.414 | +12 | -0.05 | +0.23 |
| stormy | MODERATE | weather | 50 | — | 0.1607 | 0.878/0.608 | +37/14 | +0.03 | +0.18 |
| string_quartet | MODERATE | instrumental_cinematic | 50 | 1717 | 0.1694 | 0.974/0.925 | +26/16 | +0.10 | -0.12 |
| strings_romance | MODERATE | romantic | 50 | 1717 | 0.2048 | 0.974/0.926 | +32/16 | +0.13 | -0.14 |
| study_session | OK | focus_study | 50 | — | 0.1467 | 0.926/0.791 | +20 | +0.04 | -0.07 |
| summer_breeze | MODERATE | season_summer | 50 | 1296 | 0.2006 | 0.777/0.47 | +21 | -0.06 | +0.17 |
| summer_evening | OK | season_summer | 50 | — | 0.1825 | 0.897/0.632 | +19 | -0.06 | +0.11 |
| summer_heat | MODERATE | season_summer | 50 | 4705 | 0.1668 | 0.916/0.652 | +1 | -0.21 | +0.19 |
| summer_roadtrip | OK | season_summer | 50 | — | 0.1404 | 0.939/0.761 | +7 | -0.09 | +0.08 |
| summer_tropical | OK | season_summer | 50 | 4845 | 0.1585 | 0.933/0.795 | +16 | -0.17 | +0.16 |
| sunday_morning | MODERATE | calm_unwind | 50 | — | 0.2403 | 0.893/0.689 | +31/15 | +0.05 | -0.06 |
| sunday_scaries | OK | melancholy_blue | 50 | — | 0.0963 | 0.938/0.815 | +15 | +0.02 | +0.08 |
| sunny | OK | weather | 50 | — | 0.186 | 0.935/0.753 | +16 | -0.10 | +0.15 |
| sunrise | MODERATE | time_of_day | 50 | — | 0.1689 | 0.911/0.717 | +27/21 | +0.03 | +0.02 |
| sunset_mix | MODERATE | time_of_day | 50 | — | 0.1779 | 0.94/0.786 | +25/19 | +0.07 | -0.01 |
| swagger | SEVERE | hiphop | 50 | 3181 | 0.1643 | 0.891/0.255 | +4 | -0.12 | +0.16 |
| swing_bigband | SEVERE | jazz_lounge | 50 | 657 | 0.2071 | 0.719/0.162 | -40/34 | -0.14 | +0.26 |
| synth_pop | OK | pop | 50 | 7806 | 0.1307 | 0.925/0.827 | +7 | -0.11 | +0.10 |
| synthpop_romance | OK | romantic | 50 | 13624 | 0.1498 | 0.871/0.641 | +16 | -0.10 | +0.13 |
| synthwave | OK | electronic_chill | 50 | 769 | 0.1167 | 0.945/0.869 | +1 | -0.05 | +0.05 |
| techno | OK | electronic_house_techno | 50 | 6215 | 0.1442 | 0.958/0.836 | -7 | -0.06 | -0.08 |
| tender | OK | calm_unwind | 50 | — | 0.1913 | 0.846/0.677 | +25 | +0.06 | +0.12 |
| three_am | OK | time_of_day | 50 | — | 0.1318 | 0.9/0.705 | +23 | +0.01 | +0.01 |
| throwback_anthems | MODERATE | nostalgic_throwback | 50 | — | 0.1393 | 0.836/0.427 | +6 | -0.09 | +0.12 |
| trance | OK | electronic_house_techno | 50 | 3416 | 0.1283 | 0.969/0.927 | -7 | -0.16 | +0.00 |
| trap_mode | MODERATE | hiphop | 50 | 5986 | 0.1831 | 0.954/0.846 | -19 | -0.18 | +0.32 |
| treat_yourself | OK | party_fun | 50 | — | 0.1451 | 0.871/0.578 | +13 | -0.07 | +0.16 |
| triumphant | OK | euphoric_triumphant | 50 | — | 0.1444 | 0.875/0.724 | +8 | -0.04 | +0.22 |
| uk_garage | MODERATE | electronic_bass | 50 | 3072 | 0.1312 | 0.944/0.843 | -8 | -0.21 | +0.09 |
| vaporwave | OK | electronic_chill | 50 | 1183 | 0.1605 | 0.922/0.834 | +22 | -0.05 | -0.06 |
| vulnerable | MODERATE | heartbreak_longing | 50 | — | 0.1226 | 0.904/0.75 | +29/16 | +0.08 | +0.06 |
| walking_mix | OK | happy_bright | 50 | — | 0.143 | 0.913/0.727 | +18 | -0.05 | +0.09 |
| wedding_day | MODERATE | romantic | 50 | — | 0.174 | 0.847/0.404 | +18 | -0.03 | +0.19 |
| weekend_mix | MODERATE | happy_bright | 50 | — | 0.2058 | 0.857/0.483 | +24 | +0.02 | +0.21 |
| wind_down | MODERATE | calm_unwind | 50 | — | 0.2126 | 0.938/0.836 | +31/14 | +0.08 | -0.16 |
| windy | OK | weather | 50 | — | 0.1897 | 0.921/0.773 | +21 | +0.03 | +0.10 |
| winter_cosy | MODERATE | season_winter | 50 | 3200 | 0.1745 | 0.792/0.486 | +29/13 | -0.03 | +0.09 |
| winter_frost | OK | season_winter | 50 | 21387 | 0.1298 | 0.97/0.931 | +18 | +0.05 | -0.11 |
| winter_jazz | OK | season_winter | 50 | 1719 | 0.1409 | 0.945/0.831 | +18 | -0.00 | -0.14 |
| winter_mix | MODERATE | season_winter | 50 | — | 0.1348 | 0.9/0.705 | +33/16 | +0.09 | +0.06 |
| winter_nights | OK | season_winter | 50 | 6037 | 0.1271 | 0.905/0.787 | +9 | -0.05 | +0.07 |
| witching_hour | OK | time_of_day | 50 | — | 0.1186 | 0.894/0.71 | +20 | +0.01 | +0.03 |
| workout | OK | workout_energy | 50 | — | 0.2053 | 0.89/0.732 | -19 | -0.15 | +0.21 |
| yacht_rock | OK | pop | 50 | 1415 | 0.1827 | 0.793/0.555 | +19 | -0.08 | +0.18 |
| yearning | OK | heartbreak_longing | 50 | — | 0.1168 | 0.932/0.804 | +18 | +0.07 | +0.02 |
| yoga_stretch | MODERATE | wellness_sleep | 50 | — | 0.2385 | 0.931/0.662 | +24 | +0.04 | -0.12 |

*Generated by the session quality-audit harness (scratchpad `quality_audit.py` -> `aggregate_quality.py`); rebuildable any day — builds are date-deterministic. The same metrics double as the before/after regression harness for any fix.*

---

## Fixes applied — 2026-07-09 (the SEVERE tier)

Config-level batch (same session): per-profile Discogs confidence floors via `_PROFILE_TAG_FLOOR` (the
electronic_radio mechanism), one positives fix, one soft-negatives entry. Verified by re-running the audit
metrics: **every named alien gone, all 13 profiles build 50/50, controls (rave_cave/techno/motown_soul)
byte-identical.**

| profile | fix | worst-cohesion before → after |
|---|---|---|
| swing_bigband | floor 0.25 | 0.162 → 0.406 |
| chart_pop | floor 0.25 | 0.229 → 0.825 |
| melbourne_pubrock | floor 0.30 | 0.364 → 0.128 (residual: see below) |
| rockabilly_surf | floor 0.30 | 0.333 → 0.423 |
| hyperpop | drop 'bubblegum' positive + floor 0.20 | 0.711 → 0.877 (pool 149) |
| melbourne_psych | floor 0.22 (NOT 0.25 — see lesson) | 0.226 → 0.388 |
| melbourne_folk | floor 0.20 | 0.196 → 0.495 |
| emo_poppunk | floor 0.30 | 0.300 → 0.720 |
| london_mod | floor 0.25 | 0.320 → 0.509 |
| swagger | floor 0.25 | 0.255 → 0.539 |
| afrobeat | floor 0.15 (tiny pool: 134) | 0.255 → 0.477 |
| melbourne_sunset | floor 0.20 | 0.176 → 0.574 |
| date_night | soft negatives (metal/gothic/industrial/hardcore) | 0.297 → 0.304; HIM gone |

**Lessons / residuals:**
- **City mixes intersect the geo tier gate** — a style-pool measurement alone overstates their effective pool
  (melbourne_psych starved to n=32 at floor 0.25; 0.22 restores 50). Check `n` per-build when flooring a city mix.
- **melbourne_pubrock residual:** Tina Arena "Be a Man" carries a ≥0.30 Discogs conf (a *confident* mis-tag) — no
  reasonable floor ejects it. The fix class for confident mis-tags is embedding-outlier rejection at selection
  time (candidate follow-up); at 0.30 the rest of the worst-list is clean (Farnham/Men at Work ≥0.62).
- **No change:** festive (cross-genre by design) · old_friends (verify the Christmas-collection exclusion on
  prod before acting — the audit harness bypassed it).
