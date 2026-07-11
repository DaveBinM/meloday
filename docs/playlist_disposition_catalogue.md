# Playlist Disposition Catalogue — every playlist reviewed in depth (2026-07-11)

**What this is:** the complete 'does X need fixing, and why' reference for every Meloday playlist —
the follow-up the user requested to the quality audit, re-evaluating ALL playlists including the
previously-OK tier. Companion to `docs/embedding_genre_quality_audit.md` (round-1 metrics + the fixed
SEVERE tier).

**Method (4 independent lenses):**
1. **Metrics v2** — every playlist rebuilt through the real `_build_mix_tracks` (dev cache); all-10 facet
   deltas (BPM octave-folded), dual-model embedding cohesion, gate-pool size + fragility, spread,
   instrumental violations. 281/281 built, no errors.
2. **Full-tracklist editorial review** — every playlist's 50 selections judged against its stated intent
   by music-editor review agents (15 chunks); 1,619 misfit flags with reasons + confidence.
3. **Adversarial verification** — every flag auto-checked against cache evidence (festive-title regex,
   the profile's own geo specs, earliest-known original year, measured vocal presence, matched Discogs
   confidence vs the gate floor). Agent opinions did NOT enter this document unverified; contradicted
   flags were dismissed (e.g. most 'vocal' flags measured <0.25 = fine; 49 'geo leaks' are the designed
   nation-tier fill; 9 decade 'reissues' have no older copy in the library and cannot be dated).
4. **Build-mechanics code review** — the ten never-audited stats/discovery builders + the mood-mix
   machinery specifics (instrumental enforcement, balanced 50/50, seed centroids, weather/season gating).

**Review coherence across all 281:** 5/5: 35 · 4/5: 141 · 3/5: 63 · 2/5: 36 · 1/5: 6. The 4-5 range needs nothing or only the systemic fixes; the 1-2 range is dominated by gate-vocabulary failures (S3).

## The systemic issues (fix once, help many)

Most of the 1,619 review flags collapse into a small number of root causes. Each class below lists the
mechanism, the evidence, and the recommended fix. Per-playlist dispositions in the tables reference these
as S1..S15.

### S1 — Christmas/holiday titles in non-festive mixes (100 verified leaks)
Holiday songs living on NON-holiday albums pass every gate (the prod Christmas-collection exclusion only
covers albums IN the collection). Worst: loved_up (7), morning (6), romantic_dinner (3× "Silent Night"),
wedding_day, weekend_mix. **Fix: a global festive-title keyword exclusion (the audit's regex) applied at the
candidate stage for every profile except `festive`.** Cheap, mechanical, kills the whole class.

### S2 — Discogs low-confidence tag spray past floor-less gates (306+ verified `low-conf-leak`)
The same class the SEVERE batch fixed for 12 profiles — this deeper pass shows it is far wider. Worst gates
(review coherence 1-2): latin_heat (EDM swamped the latin core), ska (~15/50 genuine), gospel (matched
classic soul/disco instead), boom_bap & conscious_flow (trap/drill + non-rap singers), indie_pop (⅓
major-label chart pop), uk_garage (commercial house), stoner_rock, chiptune, summer_tropical. **Fix: extend
`_PROFILE_TAG_FLOOR` per profile using the measured sweep tables (w4: gospel 0.15, ska 0.25, glasgow_soul
0.25*, melbourne_soul 0.20*, summer_breeze 0.20, winter_cosy 0.25; *city mixes MUST be build-tested — see
S5/melbourne_psych).** Floors alone won't fix the worst ones — see S3.

### S3 — Positive-token collisions & over-broad positives (the 1/5-coherence gates)
Where the positive VOCABULARY is wrong, no floor helps:
- swing_bigband: 'swing' matches Discogs **RnB/Swing** (new-jack-swing) — ~45% 90s/00s R&B-pop SURVIVES the
  0.25 floor at high conf. Fix: replace 'swing' with explicit 'big band'/'swing jazz' spellings.
- g_funk: positives include contemporary-R&B-family tokens — only ~1 genuine g-funk track in 50. Rework to
  g-funk/west-coast/gangsta tokens only.
- uk_garage: bare 'garage', 'bass music' match all house-adjacent dance. Narrow to 'uk garage', '2-step',
  'bassline', 'speed garage'.
- london_mod: 'beat' ⊂ 'big beat' (Blue's boyband remix at .14). rockabilly_surf: 'rock & roll' pulls Motown
  + generic 60s rock. afrobeat: 'african' admits anything world-adjacent (only ~15/50 genuine). hyperpop:
  'glitch' pulls IDM (Four Tet/BoC) — after the bubblegum fix it now reads as an IDM mix; likely also
  library-bound (very little true hyperpop owned). britpop_rock: leaks Girls Aloud/Sugababes/Olly Murs
  ("British pop" ≠ Britpop). trance: mostly tech-house. dnb: house/trance REMIXES of pop pass ('breakbeat'/
  'bass music' + remix-genre tagging) — this, not tempo, is its real problem. vaporwave: matched essentially
  nothing real and filled with score cues + lofi flips. yacht_rock: AC/pop-soul breadth (boy bands, modern
  country) with no West-Coast canon. bluegrass/country_roads: folk/Americana adjacency read as country.
**Fix: per-profile positives rework (the biggest single quality lever left; each needs its own small
vocabulary pass + re-verify through the audit harness).**

### S4 — Film/game score cues leaking into non-score mixes
The score-composer pool (Discogs-tagged ambient/jazz/electronic by the audio model) pollutes: industrial
(dominated by cues), downtempo, bebop, the jazz mixes (jazz_dinner/smooth/winter/autumn/london_jazz),
post_rock (2/5), piano_romance, spring_strings (battle cues in a gentle mix), winter_frost, ambient_drift,
neoclassical, acid_jazz, bossa_samba, awe_wonder (fanfares vs hush). **Fix: reuse `_is_screen_content()`
(built for Soundtracks Radio) as an EXCLUSION for a per-profile list of non-score-intent gated mixes; keep
cues where they suit the intent (focus, meditation, three_am, witching_hour measured fine).**

### S5 — City-mix identity: nation-tier fill + one self-inflicted regression
The city gate fill ladder is city → region → NATION (UK-wide for glasgow_*/london_*; Australia-wide for
melbourne_*). With thin city pools the nation tier dominates: glasgow_bass drew ~40% English tech-house;
london_jazz "leaked the whole UK jazz map"; melbourne_folk/hiphop are ⅓-½ interstate. This is BY DESIGN but
user-surprising for city-named playlists. **Decision needed: (a) cap fill at the REGION tier (Scotland /
England / Victoria) accepting shorter or repetitive lists, (b) rename the worst to national framing, or (c)
accept + document.**
**REGRESSION TO FIX NOW: melbourne_psych.** The SEVERE-batch floor 0.22 ∩ Melbourne geo left a pool so thin
the artist ladder relaxed to **22 Bee Gees tracks (incl. 3 versions of "Idea") + 10 King Gizzard = 32/50
from two artists**. Recommend floor back to 0.12-0.15 (accept the odd Olivia Newton-John over a Bee Gees
residency) — and add "artist-cap relaxation depth" to the audit harness assertions so this can't slip again.

### S6 — Geo resolution errors in scenes/radios (PROD-INVESTIGATE, high value)
Confident wrong-country artists surfaced: **Tyla (South African) in uk_now + uk_scene; Ye in uk_now; Justin
Bieber in london_hits/london_now; Julie London (US, name collision) + Loud Luxury (London, ONTARIO) in
london_scene; Sofia Isella in australia_now.** The city TIER gates got nation-first protection precisely for
the Ontario-collision class — the pinned SCENE/hits/radio gates did not. **Fix: (1) check each artist's
`artist_mbid`/origin in the cache (wrong-MBID resolution → `--resync-geo`), (2) nation-bound the scene/radio
gate specs the way the tiers are.**

### S7 — Worship/CCM bleed into romantic & emotional mixes
"Devotion" reads religiously: devotion (~20% worship), hopeful (CCM cluster), emotional, evening_unwind,
vulnerable, wedding_day, moving_on, situationship. **Fix: date_night-pattern soft negatives (worship/ccm/
christian crowd+style tags) for the romantic/heartbreak/calm families.**

### S8 — Novelty, fragments, karaoke artifacts
Richard Cheese in serious jazz mixes; VeggieTales in weekend/cooking; comedy acts (Tripod, Tom Cardy) in AU
mixes; sub-minute interludes/skits (Janet Jackson "20, Pt N", OutKast intro, De La skit); an Amy Grant
"(Performance Track … Without Background Vocals)" KARAOKE BACKING TRACK in three mixes. **Fix: title-marker
exclusions ('(skit)', '(interlude)', 'performance track', 'karaoke') at the candidate stage; a duration
floor would be better but duration is not cached (roadmap note).**

### S9 — Near-duplicate selectors
string_quartet ≈ strings_romance (45/50 identical); friday_feeling ≈ friday_night ≈ flirty ≈ game_night
(one party-funk pool); road_trip ≈ running; grief_release ≈ grey_skies overlap. Prod's cross-mix dedup
hides same-run collisions but the POOLS are near-identical, so differentiation is arbitrary. **Decision:
merge/retire twins, or differentiate centroids/gates deliberately.**

### S10 — Cross-artist duplicate songs in one mix
Same recording/composition under different credits passes the artist+title dedup: "End Credits" (Chase &
Status / Plan B) back-to-back; "Let's Get Loud" (J.Lo + Estefan); "God Only Knows" (Beach Boys + Pentatonix)
adjacent. **Fix: exact-title in-mix soft dedup (skip the second same-title track unless a common-word
allowlist) — cheap and safe at mix scale; recording-level IDs aren't cached (roadmap).**

### S11 — Aspirational facet targets (measured, mostly cosmetic)
The punk/garage family's vocal targets (0.55-0.65) vs pool reality (~0.93), and the dance family's
danceability targets (0.70-0.78) vs library max (~0.46): selections track the pool, the numbers are
fantasy. **Fix (optional): recalibrate targets to pool medians so future audits don't false-flag; zero
audible change.** Exception: glasgow_postrock (target 0.20 = instrumental intent vs 0.80 pool) — a genuine
intent decision, not a typo.

### S12 — Decade re-record/reissue guard
decade_20s carries 2020s re-records of Chicago/TLC/PSB classics; Tiffany 2005 in the 00s; none are
catchable by original-year data (no older copies in the library — verified `in-era`). **Fix: apply the
radios' `_REISSUE_RE` title guard (+ 'Taylor's Version'-class markers) to era gating; unmarked remasters
remain library-bound (accepted).**

### S13 — Instrumental-intent leaks
yoga_stretch traced to ONE number: vocal_presence target 0.35 (clean siblings sit 0.10-0.30) — lower to
~0.25-0.30. Remix-tagged vocal tracks slip the soft pull in spa_bath/study_session/lofi_beats; the
guaranteed fix is a hard vocal<0.4 candidate filter for `_INSTRUMENTAL_VOCAL` (mirrors Soundtracks Radio) at
the cost of borderline hushed-vocal tracks (verifier found most agent vocal flags were <0.25 = fine).

### S14 — Stats/discovery builder findings (first audit)
(1) **PROD-VERIFY: history rows likely carry no userRating** → the rating gates/multipliers in On Repeat /
Repeat Rewind / Top Songs are inert (ranking degrades to plays/recency; filtering unaffected). Test on
daedalus: print userRating on a few history entries. (2) deep_cuts/rediscovery "all-time plays" actually
spans the RUN'S shared lookback (180d alone; years when co-run) — fix with explicit per-builder fetch or
rename. (3) Discover Weekly under-fills (~25/30) when the never-played pool < ~300 — scale explore_start +
back-fill. (4) Daily Mixes double-count once-played tracks (can under-fill + inflate artist counts). (5)
Release Radar ISO-week key uses calendar year (Dec/Jan priority inversion, display unaffected). All other
builders: logic sound.

### S15 — Profile-intent mismatches (per-profile tuning, taste calls)
after_work = peak-time bangers vs a decompression brief; sad_bangers matched "banger", ignored "sad";
windy has no identity (hard eurodance); cathartic is title-keyword-driven; gaming has no rock despite the
positive; heatwave reads languid; summer_roadtrip is club-not-singalong; school_days era-diffuse;
smooth_jazz contains no smooth jazz (standards instead); dinner_party has no jazz/bossa despite the gate.
Each needs an individual centroid/theme decision — listed per playlist below.


## Per-playlist dispositions

Grouped by family. *coh* = editorial coherence (5 = ship as-is). *verified issues* = adversarially
confirmed flag classes only. Fix refs point at the systemic sections above; playlists marked NO FIX
NEEDED were re-evaluated and confirmed healthy.

### calm_unwind

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| after_work | 2 | — | NEEDS FIX (major) | S15 | Family says calm_unwind (energy -12, danceability 0.45) but roughly half the list is peak-time club/trance/EDM — it plays like a Friday-nigh |
| chill | 5 | — | NO FIX NEEDED | — | Cohesive downtempo/ambient/chill-out selection throughout. |
| cool_down | 4 | — | NO FIX NEEDED | — | Consistent wind-down palette; the only risk is suite-like or glitchy tracks that spike mid-song. |
| evening_unwind | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S7 |  |
| lazy_sunday | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| serene | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Cohesive ambient/folk/score calm otherwise; only the one holiday leak. |
| sunday_morning | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| tender | 4 | — | NO FIX NEEDED | — |  |
| wind_down | 5 | — | NO FIX NEEDED | — | Very cohesive ambient/downtempo wind-down; only album-interlude fragments worth pruning. |

### country

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| bluegrass | 3 | low-conf-leak×7, mid-conf×1 | NEEDS FIX (moderate) | S3 | Reads as a general country/folk-acoustic mix; true bluegrass (high-lonesome, string-band) is a small minority of the 50. |
| country_roads | 2 | low-conf-leak×14, mid-conf×4 | NEEDS FIX (major) | S3 | Roughly 40% of the list is 60s-70s British folk and soft-rock singer-songwriters — the gate is reading folk/Americana adjacency as country. |
| outlaw_country | 3 | low-conf-leak×10, mid-conf×2 | NEEDS FIX (moderate) | S3 | Reads as 70s soft-rock/singer-songwriter Americana — almost no actual outlaw country despite the display name |

### defiant_intense

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| angst_mix | 4 | — | NO FIX NEEDED | — | Nu-metal/emo and hard rap sit together convincingly for a 'Big Feelings' concept; vibe holds. |
| defiant | 4 | — | NO FIX NEEDED | S8 | Whiplashes between death metal and mainstream hip-hop, but both lanes genuinely read as defiant; only the novelty cuts hurt. |
| restless | 4 | — | NO FIX NEEDED | — | (salvaged from truncated run) |

### dreamy_ethereal

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| awe_wonder | 3 | — | NEEDS FIX (moderate) | S4 | Splits into two halves: on-brief dream-pop/ambient ethereality (Slowdive, Beach House, Barwick) and film-score cues where hushed wonder is m |
| daydreaming | 5 | — | NO FIX NEEDED | — | Very consistent dreamy/ethereal palette; only the darkness level varies track to track. |
| dreamy_mix | 5 | — | NO FIX NEEDED | — | Genuinely on-brief throughout — dream pop, 4AD, ambient and hazy electronics all fit. |

### driving

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| commute_mix | 4 | — | NO FIX NEEDED | S10 | Duplicate-song detection fails when the same track is credited to different artists (Chase & Status vs Plan B 'End Credits' at #45/#46). |
| driving_mix | 5 | — | NO FIX NEEDED | — |  |
| driving_singalong | 4 | — | NO FIX NEEDED | — | Funk/disco-forward take on car singalongs works, but a few instrumental-leaning cuts waste the high vocal target. |
| night_drive | 4 | — | NO FIX NEEDED | S9 | Coherent nocturnal mood, but it shares many tracks with the Midnight mix (Massive Attack, Sneaker Pimps, Hollow Coves, The Beths...) and sit |
| road_trip | 4 | — | NO FIX NEEDED | S9 | Reads as an upbeat funk/dance-party pool rather than open-road rock, and shares many exact tracks with the running and singalong mixes — hea |

### electronic_bass

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| bass_drop | 4 | low-conf-leak×5, gate-no-match×2 | MINOR — systemic fixes cover | S2 | Core dubstep/grime/UKG selection is genuinely strong (Benga, Skrillex, Magnetic Man, Dizzee, Sammy Virji); the 'trap (edm)' positive blurs h |
| dnb | 2 | confident-tag×6, mid-conf×5, low-conf-leak×3 | NEEDS FIX (major) | S3 | Under half the list is actually dnb/jungle/breakbeat; the gate is being satisfied by generic house/techno/trance remixes of pop songs. |
| glasgow_bass | 2 | mid-conf×1, low-conf-leak×1 | NEEDS FIX (major) | S5 | Double failure: the geo gate leaks a dozen English tech-house names, and generic tech-house outweighs the stated idm/bassline/dubstep/breakb |
| london_dubstep | 4 | mid-conf×5 | MINOR — systemic fixes cover | S2 | Strong genuine bass/garage spine (Benga, Magnetic Man, MJ Cole, Katy B, Jamie xx); leans on remixes to pull pop names in |
| london_garage | 3 | mid-conf×8, low-conf-leak×6, confident-tag×1 | NEEDS FIX (moderate) | S5 | Real UKG anchors exist (Katy B, Zinc/Bugz in the Attic remixes, The Streets Jameson mix) but generic house-pop remixes dilute the brief |
| london_grime | 3 | low-conf-leak×6, mid-conf×3, confident-tag×1 | NEEDS FIX (moderate) | S5 | The genuine grime/drill core (Skepta, Wiley, Dizzee, Stormzy, Headie One, Dave, Central Cee) is padded to length with adjacent bass/electron |
| london_jungle | 2 | confident-tag×8, mid-conf×4, low-conf-leak×3 | NEEDS FIX (major) | S5 | Reads as a UK dance-remix grab-bag; genuine 170bpm jungle/DnB is a minority against the stated 172bpm target. |
| uk_garage | 1 | mid-conf×11, confident-tag×10, low-conf-leak×10 | NEEDS FIX (major) | S3 | Fails its stated genre: essentially a commercial house/EDM party mix — only a handful of true UKG/bassline cuts (Sammy Virji, Disclosure, Sk |

### electronic_chill

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| downtempo | 3 | mid-conf×3, low-conf-leak×3 | NEEDS FIX (moderate) | S4 | Film-score cues keep leaking into what should be an electronic downtempo/trip-hop lane — same soundtrack-composer pool as the focus mixes. |
| glasgow_late | 4 | mid-conf×1 | MINOR — systemic fixes cover | S2 | Several picks are Scottish-but-not-Glaswegian (Boards of Canada, The Shamen, King Creosote, Pictish Trail, Nina Nesbitt, KT Tunstall, Lorne  |
| lofi_beats | 3 | low-conf-leak×4, mid-conf×1 | NEEDS FIX (moderate) | S13 | Downtempo mood mostly holds, but several full vocal singer-songwriter cuts contradict the low vocal-presence, beats-focused intent |
| london_triphop | 3 | low-conf-leak×7, mid-conf×3 | NEEDS FIX (moderate) | S2 | Downtempo feel is matched by soul/indie ballads, but genuine trip-hop/IDM is thin beyond Morcheeba, Archive and Loraine James. |
| synthwave | 3 | low-conf-leak×3, mid-conf×1, confident-tag×1 | NEEDS FIX (moderate) | S2 | Reads as broad 'synthy electronica'; genuine synthwave/retrowave acts (Anoraak, Kavinsky, Luke Million, Cannons, Miami Horror) are a minorit |
| vaporwave | 2 | low-conf-leak×10, mid-conf×4, confident-tag×3 | NEEDS FIX (major) | S3, S4 | Contains no actual vaporwave or plunderphonics at all — the pool is lofi-hop flips, film-score cues and generic downtempo; the genre gate lo |

### electronic_edm_pop

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| chiptune | 2 | low-conf-leak×9, mid-conf×5, gate-no-match×1 | NEEDS FIX (major) | S3 | Only ~6-8 tracks (8bit Misfits, Mario Kart Band, Nightwave, '8 Bit Cheese', etc.) are actual chiptune/game music — the gate admits any elect |
| dance_pop | 5 | mid-conf×1 | NO FIX NEEDED | — | Tightest gate adherence in the chunk — spans 1981-2026 but everything is genuinely dance-pop/hi-NRG/nu-disco. |
| festival_edm | 3 | low-conf-leak×1 | NEEDS FIX (moderate) | S2 | Half the list is promo club-mixes of pop/R&B songs plus pre-2000 rave/house classics (Adamski, Black Box, Frankie Knuckles) — dilutes the mo |
| hyperpop | 2 | low-conf-leak×7, confident-tag×2 | NEEDS FIX (major) | S3 | Reads as an IDM/techno/electronica mix; A.G. Cook is nearly the only actual hyperpop artist present |
| rave_cave | 3 | low-conf-leak×4, confident-tag×1 | NEEDS FIX (moderate) | S2 | Genuine hard-dance spine (Clouds, KETTAMA, Hannah Laing, X CLUB, Darude, TAAHLIAH) is padded with generic mainstage big-room that stretches  |

### electronic_house_techno

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| deep_house | 4 | low-conf-leak×2, mid-conf×1 | MINOR — systemic fixes cover | S15 | Skews indie-dance/nu-disco crossover; the genuinely deep cuts mostly arrive via remixes (Luomo, Fort Romeau, Silicone Soul, Salt City Orches |
| glasgow_house | 3 | mid-conf×2, low-conf-leak×2, xmas-leak×1 | NEEDS FIX (moderate) | S2 | The Glasgow geo pool keeps forcing non-dance indie into a house-gated mix; the actual house spine (Sulta, Slam, HudMo, Big Miz, Nightcrawler |
| glasgow_underground | 4 | mid-conf×2 | MINOR — systemic fixes cover | S2 | Same Joesef 't e s t p r e s s' remix appears twice (#19 extended, #21 radio edit) — version dedup miss. |
| house_party | 4 | mid-conf×1 | MINOR — systemic fixes cover | S2 | Built almost entirely from remixes/edits of pop songs rather than club originals, but the party-house brief mostly holds |
| industrial | 2 | mid-conf×9, low-conf-leak×7, confident-tag×2 | NEEDS FIX (major) | S4 | Film/game score cues are the single biggest bloc; almost nothing here is actual industrial/EBM beyond NIN and a few industrial-techno cuts |
| melbourne_club | 4 | confident-tag×2, mid-conf×2, low-conf-leak×1 | MINOR — systemic fixes cover | S5 | Strong Melbourne dance lineage (Madison Avenue, TV Rock, Cut Copy, Avalanches) — the city gate mostly worked here. |
| melbourne_techno | 3 | low-conf-leak×3, mid-conf×3 | NEEDS FIX (moderate) | S2 | Gate says techno/minimal/acid but the pool is mostly house and indie-dance remixes of (largely Sydney) pop acts; the many vocal remixes clas |
| techno | 3 | mid-conf×2, low-conf-leak×2 | NEEDS FIX (moderate) | S15 | Skews deep/melodic house (Jimpster, Black Loops, Galcher Lustwerk, Kraak & Smaak, Tortured Soul) over actual techno; little of the stated 13 |
| trance | 2 | low-conf-leak×5, mid-conf×3, confident-tag×2 | NEEDS FIX (major) | S3 | Mostly a house/tech-house festival slate; genuine trance (Sandstorm, Armin, Mat Zo, Stunt, Madonna's Sasha Twilo mix) is maybe 15% of the li |

### era

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| decade_00s | 4 | — | NO FIX NEEDED | S12 | Credible 00s canon with an indie/emo lean; the one leak is a re-record carrying a 2005 release date. |
| decade_10s | 5 | — | NO FIX NEEDED | — | Clean era adherence throughout, with an indie/alt lean that matches the library. |
| decade_20s | 3 | — | NEEDS FIX (moderate) | S12 | Re-records and remasters of 60s-2000s classics enter via their 2020s release dates — this decade mix needs original-release-date gating most |
| decade_60s | 5 | — | NO FIX NEEDED | — | All era-true; 60s jazz instrumentals sit comfortably beside the pop/rock canon for a library showcase. |
| decade_70s | 4 | — | NO FIX NEEDED | S12 | Era-true throughout; the only leak is a 50s hit arriving via a 70s live recording. |
| decade_80s | 4 | — | NO FIX NEEDED | S12 | Watch archival releases dated by issue year rather than recording era. |
| decade_90s | 4 | — | NO FIX NEEDED | S12 | A couple of 80s works surface via 90s remix/soundtrack release dates; otherwise strong decade adherence. |

### euphoric_triumphant

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| cathartic | 3 | — | NEEDS FIX (moderate) | S15 | Selection looks title-keyword driven ('let go', 'heal', 'letting go') across wildly different genres — emotional register swings from drill  |
| confidence_boost | 4 | — | NO FIX NEEDED | — | A few classic-rock/rave-era outliers (ZZ Top, The Shamen) read as swagger-theme picks rather than euphoric pop-dance, but the set holds toge |
| empowering | 4 | — | NO FIX NEEDED | — | Leans heavily on 90s-00s diva dance-pop and remixes — on-message, just a narrow palette. |
| euphoric | 4 | — | NO FIX NEEDED | S10 | "Let's Get Loud" appears twice (Jennifer Lopez and Gloria Estefan versions) — same song, two artists, dedup miss. |
| hopeful | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S7 | Distinct CCM/worship cluster (Chris Tomlin, TobyMac, Skillet, The Afters, Jon Foreman) shifts the register of an otherwise secular uplift mi |
| main_character | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Otherwise a consistent dance-pop/electro strut set. |
| midweek_reset | 4 | — | NO FIX NEEDED | — | Spans trance bangers to quiet folk — the euphoric target is only loosely enforced, though nothing is jarring. |
| monday_motivation | 4 | — | NO FIX NEEDED | — | Essentially a dance/EDM club set — works for Monday energy even if 'motivation' framing is loose. |
| power_ballads | 4 | — | NO FIX NEEDED | — | Skews piano/AC ballads over true arena power ballads, but the emotional sing-along intent mostly holds |
| triumphant | 4 | — | NO FIX NEEDED | — | 'Triumphant' gets read broadly as generic upbeat pop/EDM in spots (Hilary Duff, Avicii 'Trouble', Rufus du Sol 'Lately'), softening the vict |

### festive

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| festive | 5 | — | NO FIX NEEDED | — | Fully on-theme; broad artist spread and even the Chanukah entry sits fine under the 'holidays' gate. |

### focus_study

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| creative_flow | 4 | — | NO FIX NEEDED | — | The melodic-house/remix spine is very consistent; a handful of rock vocal tracks are the only interruptions. |
| deep_reading | 4 | xmas-leak×2, vocal-subtle×2, vocal-confirmed×1 | MINOR — systemic fixes cover | S13 | Heavily film-score dominated (fits the brief); the only real leak is stray Christmas material. |
| deep_work | 4 | vocal-subtle×1 | NO FIX NEEDED | S13 | Strong ambient/score/dub-techno spine; a handful of dream-pop songs push past the stated vocal ceiling. |
| focus | 5 | — | NO FIX NEEDED | — | Gate discipline is excellent — solidly instrumental score/ambient/post-rock; ships as-is. |
| study_session | 4 | vocal-subtle×5, xmas-leak×1 | MINOR — systemic fixes cover | S13 | Remix/edit versions let sung tracks slip past the instrumental gate. |

### folk_acoustic

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| campfire | 5 | — | NO FIX NEEDED | — | Consistently acoustic folk/singer-songwriter; one of the cleanest gates in the batch. |
| celtic_folk | 3 | low-conf-leak×3, mid-conf×2, gate-no-match×1 | NEEDS FIX (moderate) | S2 | Excellent trad core (Fowlis, Capercaillie, Skerryvore, Breabach, Gaughan) but Scottish/Irish *nationality* seems to substitute for Celtic *g |
| folk_acoustic | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Edges drift into mainstream pop/piano balladry (Billy Joel, Conan Gray, Role Model) that stretches the folk gate without breaking it. |
| glasgow_folk | 4 | low-conf-leak×1, mid-conf×1 | MINOR — systemic fixes cover | S2 | Solid geo + acoustic feel, though several slots are acoustic versions of non-folk acts (Chvrches, Yashin, Simple Minds) rather than folk pro |
| melbourne_folk | 2 | low-conf-leak×11, mid-conf×5, confident-tag×3 | NEEDS FIX (major) | S5 | The city gate failed hardest here — roughly a third of the list is interstate (Sydney/Brisbane/Perth) folk, plus two Christmas strays. |

### geo_scene

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| australia_now | 4 | — | NO FIX NEEDED | S6 | Heavy 2024-2026 skew is exactly right for a 'Now' station and the artist roster is convincingly Australian; only borderline-origin picks sta |
| australia_scene | 3 | xmas-leak×2 | NEEDS FIX (moderate) | S1 | Equal-weight artist rotation lets Christmas cuts and very deep obscurities stand beside canon — the showcase needs a holiday filter. |
| australian_hits | 4 | — | NO FIX NEEDED | — | Canon is credible, but several artists are represented by second-tier singles rather than signatures (Gotye 'Easy Way Out', Veronicas 'Every |
| electronic_radio | 4 | confident-tag×3 | MINOR — systemic fixes cover | S2 | Contemporary-dance skew and marquee names are right for a radio showcase; the few misses are pop/R&B artists slipping the electronic gate. |
| global_12mo | 4 | — | NO FIX NEEDED | — | Reissue/remaster dates can smuggle catalogue classics into a new-releases window — needs the original-release-date guard applied here. |
| global_3mo | 5 | — | NO FIX NEEDED | — |  |
| global_radio | 4 | — | NO FIX NEEDED | — | Throwback quota (P.I.M.P., Teenage Dirtbag, West Coast, Bags, Scotland) is about right at ~10%, but reissue-dated compilation versions infla |
| global_year | 4 | — | NO FIX NEEDED | — | Re-records and reissues count as 2026 'class members' — same original-date gap as the other global showcases. |
| indie_radio | 3 | low-conf-leak×4, mid-conf×2, confident-tag×1 | NEEDS FIX (moderate) | S2 | Convincing new-release indie rotation overall, but a handful of chart-pop names slip through the gate |
| london_hits | 4 | — | NO FIX NEEDED | S6 | Otherwise a credible cross-era London canon showcase. |
| london_now | 4 | — | NO FIX NEEDED | S6 | Throwback share (Police, Clash, Kinks, Maiden, Basement Jaxx, Coldplay) runs slightly above the ~10% brief for a 'Now' station. |
| london_scene | 3 | xmas-leak×1 | NEEDS FIX (moderate) | S6 | Two name/place collisions (Julie London, London Ontario) point at the geo resolver rather than taste. |
| pop_radio | 3 | low-conf-leak×5, mid-conf×2, gate-no-match×1 | NEEDS FIX (moderate) | S2 | Strong contemporary-pop core, but a post-punk/indie contingent leaks into what should be a mainstream-hits scope |
| rock_radio | 3 | low-conf-leak×5, mid-conf×2 | NEEDS FIX (moderate) | S2 | Contemporary-single skew is by design, but the genre gate admits pop and country acts, and the emo/pop-punk block stretches the classic-rock |
| scotland_australia_now | 4 | — | NO FIX NEEDED | — | Strong dual-scene coverage; a few legacy/re-record items dilute the 'now' framing. |
| scotland_now | 4 | — | NO FIX NEEDED | — | Excellent contemporary Scottish roster; a few 2000s deep cuts undercut the 'Now' premise. |
| scotland_scene | 5 | — | NO FIX NEEDED | — | Convincing all-era Scottish spread across indie, folk/trad, dance and rock — ships as-is. |
| scottish_hits | 5 | — | NO FIX NEEDED | — | All artists verifiably Scottish and nearly every pick is the act's actual hit or canon track — strongest list in the chunk. |
| soundtracks_radio | 5 | mid-conf×1 | NO FIX NEEDED | — | Contemporary-skew and big-name composer rotation both look right for the radio format. |
| uk_australia_now | 4 | — | NO FIX NEEDED | — | Geo scope is clean — every identifiable artist is UK or Australian, with the intended ~10% throwback share (Rafferty, Kinks, Van She). |
| uk_hits | 4 | — | NO FIX NEEDED | — | Scope reads 'well-known UK songs' more than 'hits' — several remix/alternate versions (Mad Professor 'Teardrop', Cure Shiver mix) and album  |
| uk_now | 4 | — | NO FIX NEEDED | S6 | Two non-UK stars (Tyla, Ye) suggest wrong-artist MBID/geo resolution; otherwise a convincing contemporary UK slate with a sensible ~10% thro |
| uk_scene | 4 | — | NO FIX NEEDED | S6 | Equal-weight artist rotation surfaces lots of ultra-obscure names, which suits a 'Sounds of' showcase; Tyla is the one confident geo error. |

### happy_bright

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| brunch_mix | 4 | — | NO FIX NEEDED | — | Otherwise a well-judged funk/disco/indie-dance daytime mix. |
| cooking_mix | 4 | — | NO FIX NEEDED | S1, S8 | Strong disco-funk core; the misfits are one-off novelty/holiday leaks rather than genre drift. |
| fresh_start | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Festival-EDM picks (Afrojack, Galantis, Nervo, Jonas Blue, Kaskade) push harder than the gentle-optimism 'fresh start' brief. |
| gardening | 4 | — | NO FIX NEEDED | — | Leans dance-remix/EDM (Chainsmokers, Tiësto, Robin Schulz, Kungs) more than the mellow green-thumb framing implies, though the energy stays  |
| happy | 4 | xmas-leak×2 | MINOR — systemic fixes cover | S2 | Christmas cuts from artists' holiday covers albums keep slipping into non-festive mixes — a chunk-wide pattern. |
| housework_hustle | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| walking_mix | 4 | — | NO FIX NEEDED | — | Leans surprisingly clubby/EDM for a walking mix, but tempo and bright feel-good mood are mostly on-target. |
| weekend_mix | 3 | xmas-leak×4 | NEEDS FIX (moderate) | S1, S8 | Strong funk/disco party core undermined by four Christmas party songs plus a VeggieTales kids' track — holiday and novelty filters both fail |

### heartbreak_longing

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| grief_release | 5 | — | NO FIX NEEDED | S9 | Shares several tracks with Grey Skies (Keane, Ryan Adams, Tears for Fears, Ásgeir, Rob Thomas) — the sibling sad-mellow mixes could use cros |
| heartbreak | 5 | xmas-leak×1 | NO FIX NEEDED | — |  |
| moving_on | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S7 | Theme drifts from breakup-recovery into general gentle singer-songwriter fare, but the mood holds. |
| vulnerable | 5 | — | NO FIX NEEDED | S7 | Excellent tender ballad set; only leak is Christian-worship material entering via soft-piano sonics. |
| yearning | 4 | — | NO FIX NEEDED | — | Coherent, well-judged wistful set; only minor edge cases (holiday-album cuts, one likely-anthemic pick). |

### hiphop

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| boom_bap | 2 | low-conf-leak×9, mid-conf×7 | NEEDS FIX (major) | S2, S3 | Gate is passing generic hip-hop: roughly half the list is trap/drill/melodic or pop-rap; the genuine boom-bap spine (Apollo Brown, Benny, De |
| conscious_flow | 3 | low-conf-leak×7, mid-conf×4, gate-no-match×1 | NEEDS FIX (moderate) | S2, S3 | The gate admits non-rap singers (soul, folk, country, pop) and also drifts from conscious/boom-bap toward any introspective rap (trap/drill) |
| g_funk | 1 | mid-conf×17, low-conf-leak×15, confident-tag×5 | NEEDS FIX (major) | S3 | Gate failure: only Snoop's 'Gz and Hustlas' (and arguably ScHoolboy Q) is genuine g-funk/West Coast — the rest is a coast-agnostic hip-hop/R |
| melbourne_hiphop | 2 | mid-conf×11, low-conf-leak×8, confident-tag×1 | NEEDS FIX (major) | S5 | Weakest gate in the chunk — roughly half the list is interstate Aussie hip-hop (fine nationally, not Melbourne) or non-rap pop/EDM. |
| swagger | 4 | confident-tag×2 | MINOR — systemic fixes cover | S8 | Leans heavily 70s-80s funk/soul/disco relative to the rap-forward positives, though the funk gate legitimately covers most of it. |
| trap_mode | 4 | confident-tag×1 | MINOR — systemic fixes cover | S2 | Plenty of boom-bap/abstract rap (De La Soul, Apollo Brown, Armand Hammer, DMX) that is rap but not trap — softer gate adherence than the pos |

### instrumental_cinematic

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| ambient_drift | 3 | mid-conf×2 | NEEDS FIX (moderate) | S4 | Despite the ambient/dark-ambient/new-age gate this is really a quiet film-score cue mix — essentially zero ambient-genre artists (only Konx- |
| cinematic_epic | 3 | xmas-leak×1, confident-tag×1 | NEEDS FIX (moderate) | S15 | Genuinely all score/soundtrack, but many cues are playful/comedic underscore ('Last Sandwich', 'Bachelorette Party', 'Worcestershiree?', 'Ga |
| glasgow_postrock | 2 | low-conf-leak×8, confident-tag×1 | NEEDS FIX (major) | S5, S11 | Geo gate appears broken for this profile — roughly 40% of artists are English/Welsh (Led Zeppelin, Muse, Foals, Arctic Monkeys, The Cure), u |
| neoclassical | 3 | low-conf-leak×2, mid-conf×2 | NEEDS FIX (moderate) | S4 | Reads as a film/TV-score cue shuffle rather than neoclassical/modern composition (no Arnalds/Richter-style core), with several playful or te |
| post_rock | 2 | low-conf-leak×12, mid-conf×3 | NEEDS FIX (major) | S4 | The instrumental_cinematic family pulls film-score cues and mellow indie past the post-rock/math-rock gate — true post-rock (Mogwai, 65dos,  |
| soundtracks | 5 | low-conf-leak×1 | NO FIX NEEDED | — |  |
| string_quartet | 4 | low-conf-leak×3 | MINOR — systemic fixes cover | S9 | Despite the name there is no actual quartet/chamber repertoire — it's gentle film-cue strings, and it overlaps ~90% with Strings & Romance. |

### jazz_lounge

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| acid_jazz | 3 | low-conf-leak×8, mid-conf×3, confident-tag×1 | NEEDS FIX (moderate) | S4 | Pool leans classic 70s funk/disco/soul rather than the acid jazz/jazz-house club canon — US3, Jamiroquai and Grant Green are outnumbered by  |
| bebop | 3 | low-conf-leak×4, mid-conf×3, confident-tag×1 | NEEDS FIX (moderate) | S4 | Blue Note-era core (Morgan, Shorter, Silver, Blakey, Gordon) plus credible modern post-bop, but ballad standards and score covers dilute an  |
| dinner_party | 4 | low-conf-leak×1, mid-conf×1 | MINOR — systemic fixes cover | S15 | Reads as a classic soul/funk party — excellent, but there is almost no actual jazz, lounge, or bossa despite the gate. |
| jazz_dinner | 3 | low-conf-leak×4, mid-conf×3 | NEEDS FIX (moderate) | S4 | Core jazz programming is strong; film-score cues keep leaking in, and Richard Cheese is a genuine clanger |
| london_jazz | 2 | low-conf-leak×16, mid-conf×4 | NEEDS FIX (major) | S4, S5 | The London gate leaked the whole UK jazz map — Scottish, Newcastle and Manchester players plus film-score cues. |
| smooth_jazz | 3 | low-conf-leak×3, mid-conf×3 | NEEDS FIX (moderate) | S4, S15 | Reads as classic straight-ahead/vocal jazz standards, not smooth jazz or quiet storm — no actual smooth-jazz artists appear, and screen-scor |
| swing_bigband | 1 | confident-tag×12, low-conf-leak×12, xmas-leak×1 | NEEDS FIX (major) | S3 | The 'swing' positive matched new-jack-swing/swingbeat tags — roughly 45% of the list is 90s/00s R&B-pop; the genuine swing/trad-pop half (Si |

### melancholy_blue

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| bittersweet | 4 | — | NO FIX NEEDED | — | Solid mood cohesion; a few EDM/rap-origin picks (Kaskade ICE mix, Stormzy) stretch the texture but hold the happy-sad register. |
| emotional | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S7 | Solid acoustic/melancholy core; Christian-adjacent ballads keep blurring into the secular sad-songs intent. |
| melancholy | 5 | — | NO FIX NEEDED | — | Excellent fit; notable rotation overlap with long_distance (Home, Living Proof, Reason to Believe, Be My Mistake). |
| moody_mix | 5 | — | NO FIX NEEDED | — |  |
| sad_bangers | 2 | — | NEEDS FIX (major) | S15 | Selection matched 'banger' but ignored 'sad' — roughly half the list is straight party rap/funk/house with no melancholic content. |
| sunday_scaries | 5 | — | NO FIX NEEDED | — | Consistently soft, melancholic singer-songwriter fare; on-brief throughout. |

### nostalgic_throwback

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| memory_lane | 5 | — | NO FIX NEEDED | — | Consistently mellow and wistful; leans melancholy more than warm-nostalgic but would ship. |
| nostalgia_mix | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S1 | Well over half the tracks are 2024-26 releases for a 'throwback' family — nostalgia is mood-only — and several tracks repeat from Memory Lan |
| old_friends | 3 | xmas-leak×4 | NEEDS FIX (moderate) | S1 | Seven Christmas songs leaked in — the warm/nostalgic profile is clearly matching holiday material; otherwise coherent |
| school_days | 3 | — | NEEDS FIX (moderate) | S15 | Era is diffuse for a life-era nostalgia mix — 1960s soul through 2013 pop, mostly club remixes; it reads as a generic 'school disco party' r |
| throwback_anthems | 4 | — | NO FIX NEEDED | — | Several deep cuts/secondary singles (Rick James 'Ghetto Life', Darude 'Feel the Beat', SMD 'Hotdog', Iggy Azalea 'Bounce') dilute the 'anthe |

### party_fun

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| celebration | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S1 | Christmas party tracks are the only leak; the rest is on-brief party fare. |
| cookout | 4 | — | NO FIX NEEDED | — | Zero reggae despite it being a stated positive — modern house/disco-pop remixes fill that space instead. |
| friday_feeling | 4 | — | NO FIX NEEDED | S9 | Near-duplicate pool with friday_night/flirty (Darwin Derby, Tenderoni, D.A.N.C.E., Let's Groove all repeat) — the party mixes need different |
| friday_night | 4 | — | NO FIX NEEDED | S9 | Heavy track overlap with friday_feeling — the two Friday party mixes are close to interchangeable. |
| game_night | 4 | xmas-leak×2 | MINOR — systemic fixes cover | S9 | Coherent funk/disco party pool, but it shares most of its DNA (and two Christmas leaks) with the flirty/Friday mixes. |
| party | 5 | — | NO FIX NEEDED | — | Tolerates après-ski/wedding-band novelty (DJ Ötzi, Hermes House Band, Vengaboys) — on-brand for a party mix but tonally cheesier than the fu |
| party_throwback | 3 | — | NEEDS FIX (moderate) | S15 | Several 2023-2026 releases dilute the 'throwback' premise — era gating looks absent |
| pre_party | 4 | — | NO FIX NEEDED | — |  |
| singalong | 4 | — | NO FIX NEEDED | — | Largely ignores its stated album-rock/arena-rock/power-pop positives in favour of funk/disco/dance-pop — still chantable party fare, but not |
| treat_yourself | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |

### pop

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| chart_pop | 4 | low-conf-leak×2 | MINOR — systemic fixes cover | S2 | Broad but on-genre dance-pop/teen-pop spread; skews album cuts over actual hits for a 'Pop Hits' title. |
| glasgow_synth | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Same Joesef Arielle Free remix appears twice (#1 extended, #3 radio edit) — version dedup miss; otherwise the cleanest and most genuinely Gl |
| indie_pop | 2 | low-conf-leak×10, mid-conf×6 | NEEDS FIX (major) | S2 | Genre gate leaks mainstream chart pop badly — roughly a third of the list is major-label pop stars, not indie darlings |
| melbourne_sunset | 2 | confident-tag×6, mid-conf×5, low-conf-leak×1 | NEEDS FIX (major) | S5 | Positives promise surf/indie/sunshine pop but the content is a national club-remix set — extended and club mixes dominate, and Melbourne art |
| synth_pop | 4 | low-conf-leak×2, mid-conf×1 | MINOR — systemic fixes cover | S2 |  |
| yacht_rock | 2 | low-conf-leak×18, mid-conf×3 | NEEDS FIX (major) | S3 | The broad AC/pop-soul gate pulls in boy bands, modern country and Taylor Swift while almost none of the actual West-Coast yacht canon (no St |

### reggae_ska

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| london_dub | 2 | mid-conf×6, low-conf-leak×5, confident-tag×4 | NEEDS FIX (major) | S5 | There is a real reggae-pop core (Police, Culture Club, UB40, Palmer, Quantic, Finley Quaye) but it's diluted by random legacy UK pop with no |
| reggae_dub | 3 | mid-conf×8, low-conf-leak×3 | NEEDS FIX (moderate) | S2 | Roots/dub core (Wailers, Marleys, Chronixx, Fat Freddy's, ska acts, Willie's Countryman) is real but thin — the list pads out with merely re |
| ska | 2 | low-conf-leak×11, mid-conf×11, xmas-leak×1 | NEEDS FIX (major) | S2 | Gate failure — only roughly 15 of 50 tracks have any ska/two-tone content (Rancid, Madness, Bosstones, Less Than Jake, Five Iron Frenzy, No  |

### rock_classic

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| blues_bar | 3 | low-conf-leak×4, mid-conf×3 | NEEDS FIX (moderate) | S2 | Drifts from blues into general classic-rock/soul bar fare; the blues-rock core is good but roughly a third is adjacent-genre filler. |
| classic_rock | 3 | low-conf-leak×2, mid-conf×2 | NEEDS FIX (moderate) | S2 | Strong 70s-80s core undermined by a stray 2000s pop-punk/indie cluster (Cobra Starship, Relient K, Bluejuice) the gate should never pass. |
| melbourne_pubrock | 3 | low-conf-leak×10, confident-tag×2 | NEEDS FIX (moderate) | S5 | Plays as a national Aussie-rock mix — the 'aussie rock' positive lets every non-Melbourne icon through the city gate. [max/artist=7!] |
| rockabilly_surf | 3 | low-conf-leak×5, confident-tag×3, xmas-leak×1 | NEEDS FIX (moderate) | S3 | The broad 'rock & roll' positive drags in generic 60s/70s classic rock and Motown, diluting the actual rockabilly/surf core. |

### rock_heavy

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| heavy_riffs | 3 | mid-conf×4, low-conf-leak×2, confident-tag×1 | NEEDS FIX (moderate) | S2, S11 | Solid hard-rock/metal core undercut by a soft-rock and jangle tail (Smiths, LRB, R.E.M., Train, Toto). |
| rap_rock | 2 | low-conf-leak×16, mid-conf×10 | NEEDS FIX (major) | S2, S3 | The gate collapsed into a generic 90s/00s alt-rock mix — grunge, pop-punk and emo staples outnumber actual rap-rock/nu-metal (RHCP, RATM, Li |
| stoner_rock | 2 | low-conf-leak×13, mid-conf×3, gate-no-match×1 | NEEDS FIX (major) | S2, S4 | Gate collapses into generic 'heavy-ish rock + noise' — genuine stoner/doom/desert fits (QOTSA, Tumbleweed, Mars Red Sky, Opeth, Soundgarden) |

### rock_indie

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| britpop_rock | 3 | low-conf-leak×6, confident-tag×1, mid-conf×1 | NEEDS FIX (moderate) | S3 | The 'brit pop' tag appears to be matching British *pop* acts (Girls Aloud, Sugababes, Olly Murs) rather than the Britpop genre — classic sub |
| glasgow_anthems | 5 | — | NO FIX NEEDED | — | Geo discipline is flawless — every act checks out as Glasgow/scene; if anything it favors deep cuts and 2020s newcomers over actual 'anthems |
| glasgow_indie | 4 | mid-conf×1 | MINOR — systemic fixes cover | S5 | Genre gate reads well (jangle/twee/sophisti-pop throughout) but several Edinburgh/Fife acts leak through the Glasgow geo gate. |
| indie_rock | 3 | low-conf-leak×5, mid-conf×2 | NEEDS FIX (moderate) | S2 | Plays like a broad alt/modern-rock radio mix (emo, pop-punk, nu-metal, classic rock) rather than indie rock specifically |
| london_britpop | 2 | low-conf-leak×6, mid-conf×4 | NEEDS FIX (major) | S5 | Reads as 'legacy London pop/rock of any era' — actual 90s britpop (Blur, Suede, Auteurs) is a minority of the list |
| london_indie | 3 | low-conf-leak×9, mid-conf×3, confident-tag×2 | NEEDS FIX (moderate) | S5 | Legacy classic-rock and chart-pop acts leak through the indie/post-punk gate. |
| london_mod | 2 | low-conf-leak×13, confident-tag×7 | NEEDS FIX (major) | S3, S5 | The gate matched any UK 60s-adjacent guitar act — Scottish/Manchester post-punk and five Lulu tracks dominate the non-London share. |
| melbourne_indie | 4 | mid-conf×3, low-conf-leak×1 | MINOR — systemic fixes cover | S2 | City gating holds (essentially all-Melbourne) but the genre gate admits non-rock Melbourne acts (hip-hop, pop, soul). |

### rock_psych

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| glasgow_dream | 4 | low-conf-leak×1, mid-conf×1 | MINOR — systemic fixes cover | S2 | All-Glasgow as intended, but much of it is general jangle/indie pop wearing a 'dream' label — true dreamgaze (C Duncan, Cloth, Glasvegas, Ae |
| melbourne_dream | 3 | low-conf-leak×7, mid-conf×4 | NEEDS FIX (moderate) | S5 | Nation-tier fallback pulls Sydney/Brisbane/Perth acts into a Melbourne-branded mix; the Melbourne core itself is strong. |
| melbourne_psych | 1 | low-conf-leak×3, confident-tag×1 | NEEDS FIX (major) | S5 | Two artists fill 32 of 50 slots (22 Bee Gees + 10 King Gizzard) with duplicate mixes of the same songs — no editor would ship this as 'Melbo [max/artist=22!] |
| prog_rock | 4 | — | NO FIX NEEDED | S2 | Core prog/art-rock canon is strong; the drift is toward adjacent AOR/classic rock rather than anything jarring |
| psych_haze | 4 | low-conf-leak×3, mid-conf×3 | MINOR — systemic fixes cover | S2 | Gate admits mellow piano/folk singer-songwriters that match the hazy vibe but not the psych/shoegaze genres; overall mood still coheres |

### rock_punk

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| emo_poppunk | 4 | low-conf-leak×3, gate-no-match×1, confident-tag×1 | MINOR — systemic fixes cover | S2 | 2000s Warped-Tour core is dead-on (with a healthy Christian pop-punk streak); the misses are UK/Aussie alt-rock and classic punk, not emo. |
| garage_grunge | 3 | low-conf-leak×6, mid-conf×2, xmas-leak×1 | NEEDS FIX (moderate) | S2, S11 | Strong grunge/garage core (Nirvana, Hole, Pearl Jam, Pixies, JAMC, The Vines) diluted by post-grunge/pop-punk/hard-rock bleed. |
| glasgow_postpunk | 3 | low-conf-leak×9, mid-conf×3, confident-tag×1 | NEEDS FIX (moderate) | S2 | Core is strong (Franz Ferdinand, Life Without Buildings, Orange Juice, Altered Images, Skids), but roughly a quarter is general Scottish roc |
| london_calling | 2 | low-conf-leak×8, mid-conf×4, gate-no-match×1 | NEEDS FIX (major) | S5 | Any-era London classic rock crowds out the punk/post-punk brief; the genuine punk core (Clash, Wire, Gen X-era Idol, Shame) is thin |
| melbourne_garagepunk | 3 | mid-conf×5, low-conf-leak×5 | NEEDS FIX (moderate) | S5 | City gate right, genre gate loose — several soft Melbourne indie/soul acts dilute a 150bpm punk brief. |
| melbourne_postpunk | 2 | low-conf-leak×13, mid-conf×9 | NEEDS FIX (major) | S5 | Gate collapse: roughly half the list is Sydney/Brisbane/Perth/Adelaide acts and the genre gate admits soft rock, worship and folk — actual M |
| post_grunge | 4 | — | NO FIX NEEDED | S2 | Very solid canon (Bush, 3 Doors Down, Nickelback, Lifehouse, Puddle of Mudd); only nu-metal edge cases blur the gate |
| punk_energy | 4 | low-conf-leak×6, mid-conf×3 | MINOR — systemic fixes cover | S2 | Pop-punk/skate-punk core is strong; the leaks are post-grunge/alt-metal acts riding the energy profile |

### romantic

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| acoustic_romance | 4 | gate-no-match×4 | MINOR — systemic fixes cover | S2 | Solid gentle-love-song thread; the only recurring leak is orchestral swing crooners slipping past the 'acoustic' framing. |
| candlelight | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S1 | Two Christmas songs leak into an otherwise well-pitched intimate/romantic set. |
| crush | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S1 | Reads as general feel-good upbeat pop more than specifically crush/romance-themed; two Christmas covers leaked in. |
| date_night | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Mostly the right intimate register, but a few high-energy or novelty picks puncture the mood. |
| devotion | 3 | xmas-leak×1 | NEEDS FIX (moderate) | S7 | The 'devotion' concept is leaking religious devotion — roughly a fifth of the list is Christian worship inside a romantic-family mix. |
| dinner | 4 | — | NO FIX NEEDED | — | Pleasant soft-rock/jazz/classical/ambient spread; occasional cinematic bombast is the only intrusion. |
| first_date | 4 | xmas-leak×2 | MINOR — systemic fixes cover | S1 | Warm and hopeful overall; the romantic family's recurring Christmas leak is the main blemish. |
| flirty | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S9 | Reads as generic party-funk/dance pool (heavily shared with the Friday mixes) rather than anything specifically flirtatious. |
| indie_romance | 4 | low-conf-leak×2, mid-conf×2 | MINOR — systemic fixes cover | S2 | Mood is right but several picks are tender/melancholy indie rather than actually romantic in subject |
| late_night_romance | 4 | — | NO FIX NEEDED | — | Vibe holds, but it reaches for interludes and deep cuts where actual love songs would serve better |
| long_distance | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S1 | Vibe and tempo are consistent; theme drifts to general longing rather than distance specifically. |
| love_songs | 4 | — | NO FIX NEEDED | — | A few picks lean breakup/melancholy for a high-valence love brief. |
| loved_up | 2 | xmas-leak×6 | NEEDS FIX (major) | S1 | Christmas leakage is systemic here — seven holiday tracks in a non-festive romance mix. |
| modern_romance | 3 | xmas-leak×4 | NEEDS FIX (moderate) | S1 | Four Christmas titles — the festive exclusion isn't being applied to the romantic family. |
| piano_romance | 3 | low-conf-leak×5 | NEEDS FIX (moderate) | S4 | Functions as a general film/TV-score mix — genuinely piano-led romantic pieces (Richter, Greenwood, Desplat) are a minority |
| romantic_dinner | 3 | xmas-leak×3 | NEEDS FIX (moderate) | S1 | Clear holiday leakage — three Silent Nights plus other Christmas-album tracks in a romance mix. |
| romantic_jazz | 4 | low-conf-leak×4, xmas-leak×1, mid-conf×1 | MINOR — systemic fixes cover | S1 | Otherwise a strong standards/vocal-jazz selection, but Christmas titles leak past the gate. |
| romantic_mix | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| situationship | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S7 |  |
| slow_burn | 4 | — | NO FIX NEEDED | — | Several sub-2-minute intros/interludes (Flight Facilities 'Intro', Childish Gambino 'III. Urn') pad an otherwise cohesive smolder. |
| slow_dance | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S10 | 'God Only Knows' appears twice (Beach Boys original and the Pentatonix cover back-to-back). |
| strings_romance | 4 | low-conf-leak×4 | MINOR — systemic fixes cover | S9 | Near-duplicate of String Quartet Mix — roughly 45 of 50 tracks are shared and merely reordered; the two mixes need differentiation. |
| synthpop_romance | 3 | low-conf-leak×3, mid-conf×2, xmas-leak×1 | NEEDS FIX (moderate) | S2 | A pocket of festival EDM (Walker/Kygo/Aoki/Chainsmokers) undercuts the soft, dreamy romantic brief. |
| wedding_day | 3 | xmas-leak×3 | NEEDS FIX (moderate) | S7, S1 | High-valence selection keeps pulling Christmas/worship songs into the celebration pool — festive titles need gating out of romantic/party fa |

### season_autumn

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| autumn_embers | 3 | low-conf-leak×6, mid-conf×6 | NEEDS FIX (moderate) | S2 | The 70s-80s rock spine is right, but the gate leaks synth-era pop (Ultravox/ABC/Carlisle) and 2000s emo/post-grunge, blurring the era feel. |
| autumn_jazz | 4 | low-conf-leak×5, confident-tag×1 | MINOR — systemic fixes cover | S4 | Strong jazz-to-neo-soul core (Coltrane through Ezra Collective); the recurring failure mode is film-score cues passing the jazz gate. |
| autumn_leaves | 5 | mid-conf×1 | NO FIX NEEDED | — |  |
| autumn_mix | 5 | — | NO FIX NEEDED | — | Tight melancholy-autumn indie/folk thread; even the genre outliers (Stormzy 'Please') match the subdued vibe. |
| autumn_rain | 4 | — | NO FIX NEEDED | — | A soft-rock/MOR streak (Manilow, Bread, Loggins) sits oddly beside the indie-folk but stays within the quiet register. |

### season_spring

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| spring_acoustic | 4 | low-conf-leak×1 | MINOR — systemic fixes cover | S2 |  |
| spring_bloom | 4 | xmas-leak×2 | MINOR — systemic fixes cover | S1 | Christmas covers keep slipping into the seasonal feel-good pool. |
| spring_jangle | 3 | mid-conf×5, low-conf-leak×3 | NEEDS FIX (moderate) | S2 | A mainstream chart-pop cluster (Styles/Malone/Swift/McRae/Rodrigo) dilutes an otherwise solid indie/jangle gate. |
| spring_mix | 3 | xmas-leak×2 | NEEDS FIX (moderate) | S1 | Festive gate is not applied to seasonal families — Christmas covers recur across the spring mixes. |
| spring_strings | 2 | low-conf-leak×6, mid-conf×4 | NEEDS FIX (major) | S4 | Almost no actual classical/chamber repertoire — it's a film/game score pool full of action and battle cues wearing a 'spring strings' label. |

### season_summer

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| summer_breeze | 4 | low-conf-leak×2 | MINOR — systemic fixes cover | S2 | Strong 70s AC/yacht-rock core; only stray leaks. |
| summer_evening | 4 | — | NO FIX NEEDED | — |  |
| summer_heat | 5 | — | NO FIX NEEDED | — |  |
| summer_roadtrip | 4 | — | NO FIX NEEDED | S15 | Reads as a club/house set more than a roadtrip singalong — very few windows-down rock/pop anthems for the name. |
| summer_tropical | 2 | mid-conf×5, low-conf-leak×4, confident-tag×1 | NEEDS FIX (major) | S3 | Gate has drifted into generic dance/EDM (likely via 'tropical house' adjacency) — only a handful of tracks (Gloria Estefan, Wizkid, Shaggy)  |

### season_winter

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| winter_cosy | 3 | low-conf-leak×5, xmas-leak×2, mid-conf×1 | NEEDS FIX (moderate) | S2 | The soul/quiet-storm gate is admitting any smooth ballad — boy-band and AC pop dilute an otherwise genuine classic-soul list. |
| winter_frost | 2 | mid-conf×3 | NEEDS FIX (major) | S4 | Almost entirely film/TV suspense-score cues rather than ambient/modern-classical — action-cue titles ('Home Invasion', 'Scare Tactics', 'Cov |
| winter_jazz | 4 | mid-conf×4, low-conf-leak×3 | MINOR — systemic fixes cover | S4 | Genuine and classy jazz core (Coltrane/Hartman, Evans, Getz, Salvant, Scottish jazz), padded with film-score cues — the recurring score-leak |
| winter_mix | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Near-perfect hushed wintry folk/ambient set; only the carol breaks the non-festive winter frame. |
| winter_nights | 4 | low-conf-leak×6 | MINOR — systemic fixes cover | S2 | The electronic gate drifts into soul/indie balladry, but the nocturnal mellow vibe holds well overall. |

### soul_funk_rnb

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| after_hours_rnb | 4 | low-conf-leak×5, mid-conf×2 | MINOR — systemic fixes cover | S2 | Strong sultry core (quiet storm through alt-R&B); the leaks are pop A-listers whose tracks carry R&B-ish tags but read as pop. |
| funk_disco | 4 | confident-tag×1 | MINOR — systemic fixes cover | S2 | Modern pop and EDM-house remixes (Espresso, Derulo 7th Heaven mix, Kaskade, Nightcrawlers) stretch the funk/disco gate toward generic dance- |
| glasgow_soul | 1 | low-conf-leak×21, mid-conf×12, confident-tag×4 | NEEDS FIX (major) | S2, S5 | Geo gate failed outright — around 40 of 50 artists aren't Scottish at all; this reads as a generic 70s/80s UK soul/disco/pop mix wearing a G |
| gospel | 1 | low-conf-leak×15, mid-conf×2, xmas-leak×1 | NEEDS FIX (major) | S2 | The gospel gate matched general classic soul/Motown/disco instead — only 2-3 tracks are genuinely gospel (Pentatonix's How Great Thou Art, S |
| london_soul | 4 | low-conf-leak×5 | MINOR — systemic fixes cover | S2 | Strong fit overall — the modern London soul/neo-soul core is well captured. |
| melbourne_soul | 2 | mid-conf×5, low-conf-leak×3, confident-tag×2 | NEEDS FIX (major) | S2, S5 | The genuine Melbourne soul scene (Teskey Brothers, Hiatus Kaiyote, Saskwatch, Mildlife, Chet Faker) is present but outnumbered by Sydney/Bri |
| motown_soul | 3 | mid-conf×7, low-conf-leak×2, xmas-leak×1 | NEEDS FIX (moderate) | S2 | Solid Motown/disco core, but a rap/new-wave/yacht-pop fringe of roughly 20% blurs the genre gate. |
| neo_soul | 4 | low-conf-leak×2, confident-tag×2, mid-conf×1 | MINOR — systemic fixes cover | S2 | Strong genre core (Cleo Sol, Ravyn Lenae, Glasper, Isleys, Hiatus Kaiyote); misfits are isolated. |

### time_of_day

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| after_dark | 4 | — | NO FIX NEEDED | — | Cohesive nocturnal mood held across indie, R&B and downtempo; no clear misfits. |
| blue_hour | 5 | — | NO FIX NEEDED | — | Very tight dream-pop/ambient dusk mood; ships as-is. |
| golden_afternoon | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| golden_hour | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| late_night | 5 | — | NO FIX NEEDED | — | Consistently hushed, melancholy late-night material — ships as-is |
| midnight | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| morning | 2 | xmas-leak×6 | NEEDS FIX (major) | S1 | Six Christmas songs — the holiday filter is clearly not applied to time-of-day mixes; the underlying funk/soul/pop selection is otherwise st |
| starlit | 4 | — | NO FIX NEEDED | — | Occasional score cues spike energy in an otherwise well-judged ambient/dream-pop night mix. |
| sunrise | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| sunset_mix | 4 | — | NO FIX NEEDED | — |  |
| three_am | 5 | — | NO FIX NEEDED | — | Film-score cues make up roughly 15% of the list; they suit the hour but dilute the song-led '3am thoughts' framing slightly. |
| witching_hour | 4 | — | NO FIX NEEDED | — | Score cues carry much of the eerie small-hours mood surprisingly well; a few rock cuts are the main intrusions. |

### weather

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| beach_vibes | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Reads more as a sunny nu-disco/house party than lazy beach, but the feel-good high-valence brief is met throughout. |
| clear_night | 5 | — | NO FIX NEEDED | — | Beautifully consistent nocturnal ambient/downtempo palette. |
| cosy | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 | Ship-ready warm ambient/downtempo set apart from the single Christmas carol leak. |
| foggy | 4 | — | NO FIX NEEDED | — | Film-score cues carry much of the list; it works, but the mix leans 'quiet soundtrack' more than 'fog'. |
| frosty | 4 | — | NO FIX NEEDED | — | A couple of 80s adult-contemporary ballads sneak into an otherwise well-judged ambient/score/downtempo winter list. |
| grey_skies | 5 | — | NO FIX NEEDED | S9 |  |
| heatwave | 4 | — | NO FIX NEEDED | S15 | Interprets heat as languid dream-pop/ambient shimmer — coherent, but very low-energy for a 'heatwave' brief. |
| overcast | 5 | — | NO FIX NEEDED | — |  |
| rainy_day | 5 | — | NO FIX NEEDED | — |  |
| snow_day | 3 | xmas-leak×3 | NEEDS FIX (moderate) | S1 | Christmas titles leak into a general winter-weather mix; the 70s soul/funk slow-jam cluster sits oddly beside the ambient-electronic core. |
| stormy | 4 | — | NO FIX NEEDED | — |  |
| sunny | 4 | xmas-leak×1 | MINOR — systemic fixes cover | S2 |  |
| windy | 3 | — | NEEDS FIX (moderate) | S15 | No discernible 'windy' identity — plays as a hard-charging electro/eurodance party mix, hotter than the stated bpm=110/mid-energy targets. |

### wellness_sleep

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| meditation | 4 | vocal-subtle×1 | NO FIX NEEDED | — | Leans heavily on film-score cues — on-mood, but more 'soundtrack' than ambient/new-age. |
| power_nap | 3 | vocal-subtle×3, vocal-confirmed×1 | NEEDS FIX (moderate) | S13 | Vocal dream-pop keeps slipping past the instrumental gate; also leans heavily on film-score cues whose action-cue titles (A-Team, Super-Pets |
| sleep | 4 | vocal-subtle×1 | NO FIX NEEDED | S13 | Core is genuine instrumentals/scores/ambient; a handful of hushed vocal dream-pop tracks slip past the instrumental gate. |
| spa_bath | 3 | vocal-subtle×2, vocal-confirmed×1 | NEEDS FIX (moderate) | S13 | Instrumental gate leaks — sung tracks ride in via remix/interlude tagging. |
| yoga_stretch | 4 | xmas-leak×1, vocal-confirmed×1 | MINOR — systemic fixes cover | S13 | Core selection is genuinely strong (Max Richter's Sleep, Lattimore, Barwick, Four Tet, film scores), but the library's Scottish/indie skew l |

### workout_energy

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| gaming | 4 | — | NO FIX NEEDED | S15 | No hard rock at all despite the positive — the list settles into EDM plus cinematic action-score cues. |
| running | 4 | — | NO FIX NEEDED | S9, S11 | Consistent high-energy dance/indie-dance pool, but nearly all 120-128bpm house/dance-punk despite the 160bpm target, and it near-duplicates  |
| workout | 4 | — | NO FIX NEEDED | S8 | Solid high-energy slate spanning EDM/punk/rap; a couple of bubblegum/kids-pop outliers undercut the aggressive framing. |

### world_latin

| playlist | coh | verified issues | disposition | fixes | note |
|---|---|---|---|---|---|
| afrobeat | 2 | low-conf-leak×7, mid-conf×5, confident-tag×3 | NEEDS FIX (major) | S3 | The 'african' positive is admitting anything groove- or world-adjacent — only roughly 15 of 50 tracks are actually afrobeat/afrobeats/highli |
| bossa_samba | 3 | low-conf-leak×5, mid-conf×1 | NEEDS FIX (moderate) | S4 | Several film-score cues (Conti, Debney, Giacchino, John Williams) and generic jazz standards slip in via jazz adjacency — the genuinely Braz |
| latin_heat | 2 | low-conf-leak×9, mid-conf×9, confident-tag×7 | NEEDS FIX (major) | S2 | Systemically broken gate: generic EDM/dance-pop swamps the genuinely Latin core (Bad Bunny, Shakira, J Balvin, Young Miko, Quantic et al. ar |

## Showcases, radios, and the stats builders

- **Decades / geo scenes / hits / radios** were rotation-audited earlier this week (artist_songs caps,
  4-tier radio clock) and their tracklists re-reviewed here: scottish_hits, scotland_scene, decade_60s,
  festive, soundtracks & soundtracks_radio all rate 5/5 — the recent fixes visibly hold. Open items:
  the S6 geo errors above; the S12 decade re-record guard; scotland_now-class stations occasionally
  surface a 2000s deep cut in the throwback slot (accepted: rotation breadth beats strict recency).
- **Stats/discovery builders** (On Repeat, Repeat Rewind, Release Radar, Discover Weekly, Daily Mixes,
  Rediscovery, Time Capsule, Time Machine, Deep Cuts, Top Songs): first-ever logic audit — findings and
  dispositions in S14. Six of ten are sound as-is; none is broken for track FILTERING; the concerns are
  ranking degradation (rating-inert, PROD-VERIFY) and window/underfill semantics.

## The ranked shopping list (next fix rounds)

1. **S1 festive-title exclusion** — one mechanism, ~60 verified leaks across ~35 mixes, zero risk.
2. **S5 melbourne_psych floor revert** (0.22 → ~0.15) — repairs a regression the SEVERE batch introduced
   (22-track Bee Gees residency); add artist-cap-depth assertions to the harness.
3. **S6 geo investigations** — Tyla/Ye/Bieber/Julie-London/Loud-Luxury origin checks + nation-bound the
   scene/radio gates (the Ontario class).
4. **S3 positives rework** for the 1-2/5 gates: swing_bigband, g_funk, uk_garage, ska, latin_heat,
   boom_bap, afrobeat, chiptune, vaporwave, yacht_rock, dnb, trance, indie_pop, bluegrass/country pair,
   summer_tropical, britpop_rock, rockabilly_surf, london_mod (each: small vocabulary pass + re-verify).
5. **S2 floor round-2** for the moderate leak gates (with the measured sweep values; city mixes
   build-tested).
6. **S4 screen-content exclusion** for the jazz/electronic/gentle mixes the score pool pollutes.
7. **S13 yoga_stretch vocal target** (one number) + decide the hard-vs-soft instrumental gate.
8. **S7 worship negatives** for the romantic/emotional families.
9. **S8 novelty/fragment title markers** (skits, interludes, karaoke backing tracks).
10. ~~**S12 decade re-record title guard**~~ (APPLIED — see the S12 section below); **S10 same-title
    in-mix dedup**; **S14 builder fixes** (DW backfill, daily-mix double-count, deep_cuts lookback,
    radar ISO-year) + the rating PROD-VERIFY.
11. **S9 near-duplicate selectors** and **S15 intent mismatches** — design/taste decisions per playlist.
12. **S11 facet-target recalibration** — optional hygiene so future audits don't false-flag.

*Artifacts: session scratchpad (v2 metrics + tracklists, review chunks + results, verified_flags.jsonl,
w4_results.json). Builds are date-deterministic; the harness re-runs any of this on demand.*

---

## Fix round 1 — applied 2026-07-11 (commits 3ae08a0 · f62d6d3 · 6830d6e · 28a2ce5)

User-selected scope: S2 S3 S4 S6 S7 S8 S11 S13 S14 S15 + the melbourne_psych floor; S5 nation-fill kept
by choice; S1/S9/S10 deferred. All values measured from the cache before writing (vocabulary counts in the
code's WHY comments) — six genre-knowledge tokens were caught dead by that check and removed.

- **S6 (geo)**: `_origin_match` scene fallback now word-boundary and bounded — it fires only with no MB
  origin or an in-nation origin (MusicBrainz stays the authority; a crowd tag can no longer override
  Johannesburg). New `within` nation bound kills London-Ontario collisions. CORRECTIONS from review: Tyla
  is the SA singer per the files' own MBID (verified); **Sofia Isella's origin (LA-born, Queensland area)
  is VALID dual-origin MB data, not corruption — the earlier resync recommendation was wrong** and her
  australia membership is by design. Verified: Tyla/Ye/Julie London/Bieber/Loud Luxury excluded everywhere;
  21 Savage/Riva Starr/Sofia pass by design.
- **S5 regression**: melbourne_psych floor 0.22 → 0.15 — Bee Gees 22 → 2 tracks, 29 artists.
- **S2/S3**: positives reworked + floors extended across ~45 gates; blocked-subs mechanism
  (rnb/swing + new jack swing, big/new beat, garage rock, tropical house); 7 floors sweep-tuned after
  build-testing caught starvation. Verified: all 57 changed gates build healthily, every verified leak
  ejected. Residuals (confident mis-tags, floor-immune): Tiësto 'Get Down' (Bassline .303, uk_garage),
  Girls Aloud 'Some Kind of Miracle' (Brit Pop .349, britpop_rock). Library-bound: chiptune (~20 artists),
  hyperpop (zero real hyperpop subs), london_jazz (21 city artists).
- **S4**: score cues excluded from the jazz family + industrial/downtempo/vaporwave/post_rock/stoner_rock.
- **S7**: worship/praise-only soft negatives on the romantic/emotional families (CCM broadly untouched).
- **S8**: skit/interlude/karaoke/performance-track exclusion (all 4 suffixed titles caught) + comedy/parody
  negatives on the serious-jazz family.
- **S11/S13**: facet targets recalibrated to post-gate pool medians (vocal 0.90-0.95 punk family;
  danceability 0.40-0.46 dance family; tempo in stored-BPM space); yoga_stretch vocal 0.35→0.28
  (violations 22%→8%).
- **S15**: after_work/windy club-trance negatives (banger cast 8→2 artists); sad_bangers fixed decisively
  via the new `_PROFILE_FACET_BOUNDS` hard cap (valence ≤0.45): 0.57→0.40, zero party offenders, the list
  now reads as actual sad bangers. cathartic improved (gecs gone). gaming/heatwave/summer_roadtrip/
  school_days/dinner_party accepted as documented.
- **S14**: DW back-fills to 30 (pool-scaled explore band); daily mixes no longer double-count once-played
  tracks; deep_cuts/rediscovery play windows deterministic (invariant under old-history noise); Release
  Radar ISO-year sort key. PROD-VERIFY still open: history-row userRating test on daedalus.

## S12 — decade re-record/remaster guard (IMPLEMENTED)

`_era_wrong_vintage()` + `_ERA_RERECORD_RE`/`_ERA_YEARVER_RE`/`_ERA_REMASTER_RE`, applied in the era
branch before the canonical collapse (so a song with both a re-record and an era-true copy keeps the
era-true one). NOT `_REISSUE_RE` — the radio regex is deliberately broad and false-flags legit decade
material (Brandi Carlile "Anniversary" is a song; Fatboy Slim "(Calvin Harris edit 2013)" IS the 2013 hit).

**Final rule (three parts + one exemption), each part validated row-by-row against the complete
per-decade candidate lists — the full-review pass changed the spec in three places:**

1. `'NN version` titles: 60s-00s TRUST THE TITLE'S YEAR — exclude only if it falls outside the decade
   (Blondie "(1975 version)" = the original '75 demo, KEPT in the 70s; Dolly '73, Jeff Wayne "The 1978
   Version", EWF '87, Kenny Rogers '81 all kept; Nazareth "(1991 version)" of a 1973 song excluded).
   In the 10s/20s the marker itself signals an older original — exclude even in-decade (Nicole's 2022
   re-records of her 80s hits, Soft Cell "(2023 version)" of the 2002 original). ← *review refinement 1:
   the spec's flat "exclude everywhere" would have evicted five era-true 60s-00s recordings.*
2. Wordless re-record markers (re-recorded / new version / new recording / Taylor's Version): 10s/20s
   always exclude; 60s-00s only when the copy's OWN year is outside the decade — Tears for Fears
   "Change (new version)" (1983), Kevin Rowland, Go-Betweens, the Libertines "(new recording)" (2003)
   are era-true contemporary alternates and STAY; PSB's 2024 "new version"s leave the 80s/90s.
3. Remaster titles: excluded ONLY in the 10s/20s (title + modern resolved year = contradiction:
   Haddaway/Skynyrd-live/Tiësto-reissue class). The ~410 60s-00s remaster-titled tracks are correctly
   dated by original-date tags (Bee Gees/Australian Crawl/Godsmack) — verified all kept.
4. **"(from The Vault)" exemption** ← *review refinement 2: the spec lists had missed every Taylor's
   Version row (curly-apostrophe titles vs the probe's straight `'`). Vault tracks are FIRST-EVER
   releases (incl. the 10-minute All Too Well) → genuine 20s canon, kept; the 6 non-vault TV re-records
   of released 2010-12 songs (Ronan, Today Was a Fairytale…) stay excluded. The 00s/10s TV exclusions
   only fire when the library holds the originals — verified each excluded song's original copy remains
   in its true decade pool (Love Story → 00s, All Too Well → 10s).*

Final measured impact (excluded rks): 60s 0 · 70s 3 · 80s 2 · 90s 1 · 00s 19 · 10s 38 · 20s 62.
Verified: 47-row keep/exclude oracle ALL PASS; 410 protected remasters kept; all 7 decade builds n=50,
max/artist ≤2, zero wrong-vintage tracks selected. Documented residuals (reviewed, accepted): TLC
"Creep (TLC Version)" (artist-name versions unmatchable without eating "album/single/Aretha version");
Bowie "(2020 mix)" pair (matching year-mix would evict the Fatboy class; popularity-invisible);
Chicago "25 or 6 to 4" / Kinks 2020s reissues have CLEAN titles — no title guard can catch them, their
fix is an MB original-date resync (out of S12 scope).
