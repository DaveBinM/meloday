# Meloday Extras — Playlist Audit, Documentation & Optimisation Proposal

**Scope:** every playlist built by `utilities/meloday_extras.py` — **261 mood-mix profiles** (`_MOOD_MIX_NAMES`) + 11 standalone builders.
**Generated:** 2026-06-20. All gate findings **validated against the live 135k-track cache** (read-only offline sim). **Source file lines** referenced inline.

---

# 📋 Executive summary

I audited all 261 mood mixes + 11 builders. **The library is in good shape** — the ~123 genre/city gates are overwhelmingly correct, the builders are solid, and most mixes need nothing. The actionable changes fall into four small groups. Every gate finding below is backed by a real before/after track count from the offline simulation.

### A · Style-gate bugs — mixes playing the *wrong music* (recommend fixing)

| change | what's wrong now → the fix | evidence |
|---|---|---|
| `london_mod-1` | **"London Mod Mix" has zero mod-rock** — it plays modern-classical & film scores. The tag `"mod"` substring-matches `"modern"`, and the `{classical}` parent locks it there. → parent `{rock}` + drop bare `"mod"`. | 5,977 tracks, all wrong (sampled: film cues, a violin concerto) |
| `substr-fix-targeted` | Same root cause, 2 more: `"am pop"` matches **`"dream pop"`** (dream-pop leaks into Yacht Rock & Summer Breeze); `"dub"` matches **`"dubstep"`** (dubstep leaks into Reggae/London Dub). → tighten those tags. | `am pop`→dream pop 2,845 · `dub`→dubstep 1,228 |
| `summer_heat-1` | Summer Heat **can't admit the disco/funk it names** — parent `{electronic}` blocks the Funk/Soul tracks. → add `funk / soul`. | +1,479 disco/funk tracks |
| `winter_frost-1` | Winter Frost **can't admit ambient** — parent `{classical}` blocks the Electronic-parent ambient. → add `electronic`. | +12,860 (big — validate it stays delicate) |
| `glasgow_synth-1` | Glasgow Synth **rejects Scottish new-wave** (Simple Minds etc.) — parent `{electronic}` blocks the Rock-parent new-wave. → add `rock, pop`. | +1,094 (matters after geo-gating) |
| `dinner_party-1` | Dinner Party **rejects the vocal-jazz/bossa it names** — parent `{funk/soul}` blocks Jazz. → add `jazz`. | +345 |

### B · Surfacing — mixes appearing at the *wrong time* (recommend fixing)

- `synth_pop-1` — "Synth-Pop **Nights**" has no evening bias; the name promises one. → add an evening/night lean.
- `sunset_mix-1` / `starlit-1` — dusk/night mixes that can surface at midday; every one of their peers is hard hour-gated. → gate them too.
- `situationship-1` — the only one of the six new-romance mixes *without* the shared evening lean. → add it for parity.

### C · Misc (recommend fixing)

- `power_ballads-1` — "Lighters Up" is a known-anthems concept but has no fame floor; obscure ballads can fill it. → add a 100k-listener floor.
- `repeat_rewind-1` — docstring says "21 days", the code uses 30. → align the docstring.

### 🔭 Enhancements — data we already store but don't use (why each would help)

- **Embeddings (`emb_effnet`/`emb_musicnn`) — completely unused; the single biggest lever.** True "sounds-like" similarity that the 10-dimension centroid can't do — would sharpen the *fuzzy* mixes (`situationship`, `indie_rock`, `acoustic_romance`) and `discover_weekly`/`deep_cuts`.
- **Word-boundary gate matcher** — would kill the entire substring-leak class (mod/am-pop/dub) permanently, not just the three we patch now.
- **`integrated_loudness` (unused)** — consistent-loudness pacing for `workout`/`running`/`focus`/`sleep`.
- **`danceability_hl` (unused)** — a second-opinion sanity co-gate for the dance/EDM mixes.
- **Plex `skipCount` (unused)** — deprioritise tracks you repeatedly skip in `on_repeat`/`top_songs`/`daily_mixes`.
- **`lyric_lang` (unused)** — an English-only gate for lyric-led mixes, or language-scene mixes.

**→ Decide by tier at the very bottom of this doc ("Decision tiers"). The middle (Parts A/B) is the per-mix backing detail.**

---

## How to use this document

For every playlist you get:
1. **Comment block** — the 6-facet block (Theme / Sound / Era·Geo / Music / Criteria / Enhance) that will be inserted into the code above each profile / as a builder docstring. This is the *documentation* deliverable; it ships regardless of the change decisions below.
2. **Changes** — each proposed optimisation as an addressable row `ID · change · before→after · why · risk · rec`. `✓ No change` means the profile already matches its name/description and is well-tuned.

**Approve/deny:** reply by ID or group, e.g. *"approve all `*-parent` and `synth_pop-1`, deny `summer_heat-1`"*, or *"approve all low-risk"*, or *"approve section 3"*. Only approved IDs are applied. Every applied change ships with a `# WHY:` note per repo convention. **Rec key:** ✅ recommend · ⬜ optional/your call · ⚠️ recommend but higher-risk.

## Pipeline recap (how a mix is actually built)

`_build_mix_tracks` (L8841): **hard style gate** (`_has_required_style`: Discogs *subgenre* positive at confidence ≥ `_DISCOGS_TAG_FLOOR` 0.12 **AND** dominant Discogs *parent* ∈ `_PROFILE_GENRE_PARENT`; Last.fm community tags for `_LASTFM_GATED`; holiday signal for `festive`) → **year window** (`_PROFILE_YEAR_WINDOW`) → **anthem/geo gate** → **listener floor** (`_PROFILE_MIN_LISTENERS`) → **rank** by `_combined_score` (acoustic centroid distance + `_moodclass`/`_moodtheme`/`_lastfm`/`_origin`/`_popularity`/`_lyric`/`_listening_hour` boosts; mood/style tag boosts only for non-style-defined) → loved-track lean → artist-cap fill → **`_dj_order`** (smooth, or energy-arc for `_DJ_ARC_PROFILES`).
Surfacing (`build_mood_mixes` L8250): once-a-day **anytime core** + **windowed tier** (`_HOUR_RESTRICTED`/`_WEEKDAY_RESTRICTED` hard gates; `_TIME_SOFT_BOOSTS`/`_WEEKDAY_BOOSTS` soft) + **weather** (`_WEATHER_PROFILES`) + **seasonal** (`_PROFILE_SEASON`) + **pinned geo** showcases, with `_recency_penalty` rotation and max-2-per-`_PROFILE_CATEGORY`.

## Global data-enhancement catalog (referenced by the per-playlist `Enhance:` lines)

Available in the cache but **unused or under-used** by the extras selection code today:

- **`emb_effnet` (1280-d) / `emb_musicnn` (200-d) embeddings — UNUSED.** Best lever for the fuzzy, hard-to-gate mixes (`situationship`, `indie_rock`, `acoustic_romance`, the `*_dream`/`*_indie` scenes): true "sounds-like" similarity + k-means discovery clusters + seed-artist nearest-neighbour, far sharper than the 10-d centroid. *(New infrastructure — proposed once in §Roadmap, not per-mix.)*
- **`danceability_hl` (MusiCNN) — UNUSED.** A second-opinion danceability; a sanity co-gate for dance/EDM mixes against Essentia danceability misfires. *(`_PROFILE_MOODCLASS` already reads it for a handful — could extend.)*
- **`integrated_loudness` — UNUSED.** Loudness-consistency signal for pacing mixes (`workout`, `running`, `focus`, `deep_work`, `sleep`); flags over-compressed remasters.
- **`lastfm_artist_tags` / `lastfm_track_tags` — UNDER-USED.** Already power `_lastfm_tag_boost` + the `_LASTFM_GATED` mixes; could gate/boost more scene & format genres the audio model can't name.
- **`lyric_lang` — UNUSED.** English-only gate for lyric-led mixes; or language-scene mixes.
- **`mood_*` heads / `moodtheme`** — used via `_moodclass`/`_moodtheme` but only where a profile has a hand-mapped spec; many profiles fall back to the derived default.
- **Live Plex `skipCount` / `userRating` / `viewCount`** — ratings lean is live; skip-count is unused and would deprioritise repeatedly-skipped picks.
- **`release_types` (MB)** — used to drop compilations from era mixes; could also drive singles-vs-album-cuts leans.

---

# PART A — Mood-mix profiles

## §1. Meloday+ gap-fill — pop family (L1889–1899)

**Family:** the deliberate "gaps" in the library's coverage — recognisable pop/vibe lanes. All lean to hits (`_PROFILE_POPULARITY` +1) with Last.fm listener floors so they stay famous, most are hard style-gated to keep them genre-pure, none are time-gated (anytime core). Category mostly `upbeat`/`energy`.

#### `situationship` → "Situationship Mix"
```
# ── situationship → "Situationship Mix" ───────────────────────────────────
# Theme:   Romantic limbo — the undefined "what are we", yearning + anxious + bittersweet.
# Sound:   Slow-mid (90bpm), low-energy, intimate; vocal-forward bedroom-pop / alt-R&B.
# Era/Geo: Any era · any origin.
# Music:   Hushed, vulnerable vocal-led songs about wanting, uncertainty, mixed signals.
# Criteria: centroid(aro .46/val .42/voc .74) · mood +yearning/anxious −euphoric/carefree · moodtheme love/melancholic · lyric_themes(relationship_limbo,mixed_signals,…)+valence−1 · no genre gate · category romantic · no time gate.
# Enhance: lyric_lang=en (theme match assumes English); emb_musicnn seed-NN to pin the "situationship" sound; evening lean.
```
**Changes**
- `situationship-1` · Add evening soft-time lean `_TIME_SOFT_BOOSTS["situationship"]=(18,24,0.06)` · — → present · its romantic siblings (`crush`,`slow_burn`,`loved_up`,`long_distance`,`flirty`,`devotion`) all carry the 18–24 evening lean; this one is the odd omission · **low** · ✅
- `situationship-2` · Optional light listener floor / popularity lean · none → leave neutral · it's a *vibe* not a *hits* mix, so neutral is arguably correct — flagged only for completeness · **low** · ⬜ (recommend deny)

#### `sad_bangers` → "Sad Bangers Mix"
```
# ── sad_bangers → "Sad Bangers Mix" ───────────────────────────────────────
# Theme:   Crying on the dancefloor — euphoric-sad, cathartic bangers.
# Sound:   Fast (124bpm), highly danceable, bright-ish but LOW valence (.32) + high arousal (.80).
# Era/Geo: Any era · any origin.
# Music:   Upbeat-tempo songs with melancholy hearts — dance-cry anthems across pop/electronic/indie.
# Criteria: centroid(dance .70/aro .80/val .32) · moodclass danceability_hl+party · moodtheme energetic/party/melancholic/sad · lyric(euphoric_sadness,dancefloor_catharsis,…)+valence−1 · NO genre gate (spans genres) · popularity +1 · category energy.
# Enhance: emb to anchor the paradoxical "sad-but-danceable" timbre; could add Fri/Sat + late-evening lean.
```
**Changes**
- `sad_bangers-1` · Add Fri/Sat + evening lean (`_WEEKDAY_BOOSTS`+`_TIME_SOFT_BOOSTS`) · — → e.g. `({4,5},0.06)` + `(20,2,0.06)` · cathartic dance-cry peaks on a night out; currently surfaces context-blind · **low** · ⬜

#### `power_ballads` → "Lighters Up"
```
# ── power_ballads → "Lighters Up" ─────────────────────────────────────────
# Theme:   Arena-sized emotional ballads — lighters/phones in the air.
# Sound:   Slow (78bpm), huge dynamics (dyn .72), towering lead vocal (voc .86), mid valence.
# Era/Geo: Any era · any origin.
# Music:   Soft/arena/album rock + AC ballads that build to a roof-lifting climax.
# Criteria: centroid(bpm 78/voc .86/dyn .72) · style SOFT-nudge(arena/album/soft/hard rock, AC) — NOT hard-gated · mood dramatic/anthemic/yearning · moodtheme ballad/epic/dramatic · lyric(triumphant,overcoming_obstacles,…) · popularity +1 · category emotional.
# Enhance: integrated_loudness/dynamic range as a true "ballad-build" signal; a listener floor would keep them the famous ones.
```
**Changes**
- `power_ballads-1` · Add listener floor `_PROFILE_MIN_LISTENERS["power_ballads"]=100_000` · — → 100k · "Lighters Up" is a *known-anthems* concept; +1 lean nudges but doesn't floor, so obscure ballads can fill · **low** · ✅
- `power_ballads-2` · Keep soft (not hard) style gate · n/a · power-ballads genuinely span rock+pop+AC+country; a hard Discogs gate would over-thin it. Documented as intentional · — · ✓ keep

#### `restless` → "Can't Switch Off"
```
# ── restless → "Can't Switch Off" ─────────────────────────────────────────
# Theme:   Racing-mind anxiety — wired, can't relax, intrusive thoughts.
# Sound:   Mid-fast (116bpm), agitated (aro .70), low-mid valence; driving but tense.
# Era/Geo: Any era · any origin.
# Music:   Nervy, urgent, brooding songs — propulsive but unsettled.
# Criteria: centroid(bpm 116/aro .70/val .40) · mood +tense/anxious/urgent −calm/serene · moodtheme energetic/dark · lyric(intrusive_thoughts,feeling_trapped,late_night,…)+valence−1 · no genre gate · category emotional · no time gate.
# Enhance: the lyric themes lean nocturnal (late_night, late_night_introspection) — a night lean would match.
```
**Changes**
- `restless-1` · Add late-evening/night soft lean `(21,3,0.06)` · — → present · its own lyric vocab (`late_night`, `intrusive_thoughts`) is an insomnia frame; a night surfacing bias fits the name "Can't Switch Off" · **low** · ⬜

#### `neoclassical` → "Neoclassical Calm"
```
# ── neoclassical → "Neoclassical Calm" ────────────────────────────────────
# Theme:   Modern classical stillness — Einaudi/Richter-style contemplative composition.
# Sound:   Slow (78bpm), very low energy, near-instrumental (voc .18), high dynamics (.76).
# Era/Geo: Any era · any origin.
# Music:   Neo-classical / chamber / modern-composition piano & strings — NOT film score.
# Criteria: HARD style gate(neo-classical,modern composition,chamber,classical crossover) parent {classical,stage & screen}, NEG film/original score · mood peaceful/elegant/serene · moodclass mood_acoustic · popularity −1 (deep cuts) · category calm · no time gate.
# Enhance: a soft evening/study lean; integrated_loudness for true dynamic-range selection.
```
**Changes**
- `neoclassical-1` · Add evening/study soft lean `(19,24,0.06)` · — → present · calm contemplative listening skews evening; currently context-blind · **low** · ⬜

#### `yacht_rock` → "Yacht Rock Mix"
```
# ── yacht_rock → "Yacht Rock Mix" ─────────────────────────────────────────
# Theme:   Breezy late-70s/early-80s soft-rock luxe — smooth, sun-on-the-water.
# Sound:   Mid (102bpm), smooth, warm, high valence (.70); polished session-musician sheen.
# Era/Geo: Classic yacht-rock era in spirit (no hard window) · any origin.
# Music:   Soft rock, AM pop, sophisti-pop, blue-eyed soul, quiet storm.
# Criteria: HARD style gate(soft rock,am pop,AC,sophisti-pop,blue-eyed soul,pop-soul,quiet storm) parent {rock} · mood smooth/warm/mellow · moodtheme relaxing/soft/summer · lyric(summer_romance,living_in_the_moment) · popularity +0.5 · category upbeat.
# Enhance: integrated_loudness (yacht-rock is famously smooth/compressed); summer seasonal lean.
```
**Changes**
- `yacht_rock-1` · Widen parent gate `{rock}` → `{rock, pop, "funk / soul"}` · **offline: 3721→3950 (+229).** Modest — its `blue-eyed soul`/`pop-soul`/`quiet storm` (Funk/Soul) positives are rejected by `{rock}`. **More important:** `yacht_rock` is a victim of the `am pop`→`dream pop` substring leak (Part C) — it currently pulls dream-pop, off-concept for yacht rock · **low** (parent) / **med** (substring) · ✅ (fix the substring leak; parent widen optional)

#### `swagger` → "Swagger Mix"
```
# ── swagger → "Swagger Mix" ───────────────────────────────────────────────
# Theme:   Cocky strut — bravado, flexing, street-smart confidence.
# Sound:   Mid (96bpm), groove-heavy (dance .62), vocal-forward (.80); confident bounce.
# Era/Geo: Any era · any origin.
# Music:   West-coast / hardcore / contemporary rap + funk & contemporary R&B with attitude.
# Criteria: HARD style gate(contemporary/hardcore/west-coast rap, contemporary r&b, g-funk, funk) parent {funk/soul,hip hop} · mood swaggering/brash/confident · lyric(bragging_rights,luxury_flex,…)+valence+1 · popularity +1 · category groove.
# Enhance: emb for the "swagger" production timbre; could add evening/Fri lean.
```
**Changes**
- `swagger` · ✓ No change — gate, parents, leans and lyric vocab all coherent with the name.

#### `chart_pop` → "Pop Hits"
```
# ── chart_pop → "Pop Hits" ────────────────────────────────────────────────
# Theme:   Current/recent mainstream pop — the radio & chart hits.
# Sound:   Mid-fast (116bpm), bright (.50), very high valence (.78), polished.
# Era/Geo: Any era (skews contemporary by listener floor) · any origin.
# Music:   Dance-pop, teen/vocal pop, europop, power pop — the famous ones.
# Criteria: HARD style gate(big pop list, both Plex+Discogs spellings) parent {electronic,pop,rock}, NEG punk/metal/country · moodtheme happy/positive/upbeat · popularity +1 · min_listeners 500k · category upbeat.
# Enhance: danceability_hl as a co-signal; no lyric themes (pop isn't lyric-anchored) — correct.
```
**Changes**
- `chart_pop` · ✓ No change — the 500k floor + broad pop gate match "Pop Hits" exactly.

#### `dance_pop` → "Dancefloor Pop"
```
# ── dance_pop → "Dancefloor Pop" ──────────────────────────────────────────
# Theme:   Pop built for the floor — four-to-the-floor pop bangers.
# Sound:   Fast (122bpm), top danceability (.74), high valence; DJ energy-arc ordered.
# Era/Geo: Any era · any origin.
# Music:   Dance-pop, nu-disco, italo/euro-disco, hi-NRG, electroclash.
# Criteria: HARD style gate(dance-pop,nu-disco,euro-dance,…) parent {electronic,pop,rock} · moodclass danceability_hl+party · popularity +1 · min_listeners 400k · DJ-ARC · category energy.
# Enhance: danceability_hl already in moodclass — good; emb optional.
```
**Changes**
- `dance_pop` · ✓ No change — coherent; correctly in `_DJ_ARC_PROFILES`.

#### `indie_pop` → "Indie Darlings"
```
# ── indie_pop → "Indie Darlings" ──────────────────────────────────────────
# Theme:   Sweet, jangly, slightly-left-of-centre indie pop.
# Sound:   Mid (112bpm), moderate dance (.50), bright valence (.68); melodic, charming.
# Era/Geo: Any era · any origin.
# Music:   Indie/jangle/twee/chamber/baroque/bedroom pop.
# Criteria: HARD style gate(indie pop,jangle,twee,chamber,baroque,bedroom,c-86,…) parent {electronic,pop,rock} · mood bright/playful/sweet · moodtheme happy/positive/melodic · popularity +1 · min_listeners 150k · category upbeat.
# Enhance: emb (indie pop is broad; timbral NN sharpens it); no time gate needed.
```
**Changes**
- `indie_pop` · ✓ No change.

#### `synth_pop` → "Synth-Pop Nights"
```
# ── synth_pop → "Synth-Pop Nights" ────────────────────────────────────────
# Theme:   Neon synth-pop — 80s-rooted & modern; a night-time glow.
# Sound:   Mid-fast (116bpm), bright (.42), danceable; gleaming synths.
# Era/Geo: Any era · any origin.
# Music:   Synth-pop, new wave, new romantic, electroclash, sophisti-pop.
# Criteria: HARD style gate(synth pop,synthwave,new wave,new romantic,…) parent {electronic,pop,rock} · moodtheme energetic/cool/retro/upbeat · popularity +1 · min_listeners 150k · category upbeat.
# Enhance: the title says "Nights" but there is NO evening surfacing bias — add one.
```
**Changes**
- `synth_pop-1` · Add evening/night soft lean `_TIME_SOFT_BOOSTS["synth_pop"]=(19,2,0.08)` · — → present · the name explicitly promises "Nights" yet it surfaces all day; a dusk/night lean honours the title · **low** · ✅

## §2. Meloday+ gap-fill — rock / electronic / scores (L1901–1906)

**Family:** play-history-driven coverage gaps in heavier rock, dance and screen music. All hits-leaning (+1) with 150k–400k floors except the two specials (`soundtracks` neutral, `rave_cave` balanced).

#### `indie_rock` → "Indie Anthems"
```
# ── indie_rock → "Indie Anthems" ──────────────────────────────────────────
# Theme:   Big, fist-in-the-air indie/alt-rock anthems.
# Sound:   Fast (124bpm), driving (aro .70), guitar-forward.
# Era/Geo: Any era · any origin.
# Music:   Alt/indie rock, college rock, garage-rock revival, modern rock.
# Criteria: HARD style gate(alt/indie rock,college rock,garage revival,modern rock) parent {rock} · mood lively/rousing · popularity +1 · min_listeners 150k · category upbeat.
# Enhance: KNOWN broad/hard-to-narrow (per project memory) — emb_musicnn NN or a tighter centroid is the real fix; gate alone won't sharpen it.
```
**Changes**
- `indie_rock-1` · Note (no code change yet): tighten via embeddings in the §Roadmap pass · — · the gate is as good as Discogs allows; further precision needs `emb` · **n/a** · ⬜

#### `post_grunge` → "Post-Grunge"
```
# ── post_grunge → "Post-Grunge" ───────────────────────────────────────────
# Theme:   Late-90s/00s radio post-grunge — Creed/Nickelback/Foo-adjacent.
# Sound:   Fast (126bpm), heavy (aro .78), low brightness; anthemic grit.
# Era/Geo: Any era · any origin.
# Music:   Post-grunge / radio hard-rock.
# Criteria: LAST.FM-GATED (Discogs can't name post-grunge) on community tags(post-grunge) · moodclass mood_aggressive · popularity +1 · min_listeners 150k · category energy.
# Enhance: relies on Last.fm tag coverage — track-tag sync completeness is the limiter; emb fallback for un-tagged.
```
**Changes**
- `post_grunge` · ✓ No change — correctly Last.fm-gated (project-verified the centroid can't isolate it).

#### `rap_rock` → "Rap-Rock & Nu-Metal"
```
# ── rap_rock → "Rap-Rock & Nu-Metal" ──────────────────────────────────────
# Theme:   Turn-of-millennium rap-rock / nu-metal aggression.
# Sound:   Fast (128bpm), top arousal (.86), rhythmic & heavy.
# Era/Geo: Any era · any origin.
# Music:   Rap-rock, rap-metal, nu-metal, funk-metal, rapcore.
# Criteria: HARD style gate(rap-rock,nu metal,funk metal,rapcore) parent {hip hop,rock} · moodclass mood_aggressive · popularity +1 · min_listeners 150k · category energy.
# Enhance: emb optional; coherent as-is.
```
**Changes**
- `rap_rock` · ✓ No change.

#### `festival_edm` → "EDM Anthems"
```
# ── festival_edm → "EDM Anthems" ──────────────────────────────────────────
# Theme:   Main-stage festival EDM — hands-up drops & big-room euphoria.
# Sound:   Fast (128bpm), high dance (.72) + arousal (.86), bright; energy-arc ordered.
# Era/Geo: Any era · any origin.
# Music:   Big room, electro house, future bass, complextro, mainstage EDM.
# Criteria: HARD style gate(edm,big room,electro house,future bass,complextro) parent {electronic} · moodclass dance+party+electronic · popularity +1 · min_listeners 400k · DJ-ARC · category energy.
# Enhance: danceability_hl already used; good.
```
**Changes**
- `festival_edm` · ✓ No change.

#### `soundtracks` → "Soundtracks & Scores"
```
# ── soundtracks → "Soundtracks & Scores" ──────────────────────────────────
# Theme:   Cinematic & game scores — orchestral and composed screen music.
# Sound:   Mid-slow (92bpm), low energy, near-instrumental (voc .18), high dynamics (.70).
# Era/Geo: Any era · any origin.
# Music:   Film/TV scores, orchestral, modern composition, video-game music (Zelda etc. live here, not under a game-music genre).
# Criteria: HARD style gate(soundtrack,score,film/tv,orchestral,video game music) parent {classical,stage & screen}, NEG rap/punk/drill · moodtheme film/epic/dramatic · popularity NEUTRAL · category cinematic.
# Enhance: emb for "epic vs intimate" score sub-clustering; no time gate (correct — listen anytime).
```
**Changes**
- `soundtracks` · ✓ No change — neutral popularity is right (famous scores and deep cues both belong).

#### `rave_cave` → "Rave Cave"
```
# ── rave_cave → "Rave Cave" ───────────────────────────────────────────────
# Theme:   Hard-dance basement rave — hardstyle/hard-house/gabber energy.
# Sound:   Very fast (132bpm), dark (bright .17), pumping; energy-arc ordered.
# Era/Geo: Any era · any origin.
# Music:   Donk, hard house/trance/techno, hardstyle, gabber, happy hardcore, makina.
# Criteria: HARD style gate(hard-dance subgenres) parent {electronic} · 50/50 BALANCED anchor/discovery (_BALANCED_PROFILES) · moodclass dance+party+electronic+aggressive · DJ-ARC · category energy.
# Enhance: pilot for the balanced anchor/discovery split (per memory); emb could refine the hard-dance pool.
```
**Changes**
- `rave_cave` · ✓ No change — recently & deliberately calibrated (balanced split, parent gate, centroid).

## §3. Decade mixes (L1908–1914, family)

**Family (`decade_60s`…`decade_20s`):** "hits of the decade." These run the **showcase selection path** — there is *no* style gate; selection = `_PROFILE_YEAR_WINDOW` (original release year) → drop compilations → rank by Last.fm listeners with the `_ERA_MIN_LISTENERS` 300k floor (fallback if too thin) → daily shuffle. The acoustic centroid only feeds the rotation sim-guard, so the small per-decade centroid differences are cosmetic. Shared: `_PROFILE_MOOD_SIGNALS` generic (rousing/energetic/stylish/playful), `_PROFILE_POPULARITY` +1, category `era`, no time gate, no lyric themes (era is not a mood). On insertion each gets its own block; the template:
```
# ── decade_NNs → "NNs Mix" ────────────────────────────────────────────────
# Theme:   The famous, remembered hits of the NNNN–NNNN decade.
# Sound:   Era-defined (centroid only feeds the sim-guard; selection is popularity-ranked).
# Era/Geo: NNNN–NNNN (original release year, reissue-safe) · any origin.
# Music:   The big, well-known songs of the decade across all genres.
# Criteria: year_window(NNNN,NNNN) · popularity-RANKED + 300k listener floor · drop compilations · +1 lean · category era · no genre gate · no time gate.
# Enhance: release_types → favour original singles over reissues; decade-appropriate nostalgia lean (see decades-2).
```
Per-member: `decade_60s`(1960-69) · `decade_70s`(1970-79) · `decade_80s`(1980-89) · `decade_90s`(1990-99) · `decade_00s`(2000-09) · `decade_10s`(2010-19) · `decade_20s`(2020-29).

**Changes**
- `decades-1` · ✓ No change to selection — year-window + popularity-rank + 300k floor is the right recipe for "decade hits."
- `decades-2` · Optional: surface decade mixes with a gentle recency-of-relevance rotation (e.g. weight the user's own most-played decade up) · — → optional · purely additive nicety; current equal treatment is fine · **low** · ⬜ (recommend deny — `time_capsule` already covers "your top era")
- `decade_20s-1` · Confirm intent: a 2020–2029 "20s Mix" is *current-decade hits*, not nostalgia · n/a · documented as intentional (it's a decade mix, not a throwback) · — · ✓ keep

## §4. Geo showcase mixes (L1917–1919, family — PINNED)

**Family (`scotland_scene`,`australia_scene`,`london_scene`):** "Sounds of X" — the famous artists from a place, on rotation. **Always pinned** into every run (never rotated out); content refreshes once a day via an artist-coverage rotation (`_load_geo_rotation`). Selection = HARD origin gate (`_PROFILE_GEO_GATE`: artist's consolidated MB place ∈ {place}, with a Last.fm scene-tag fallback) → rank by Last.fm listeners + overdue-artist bonus → one song per artist → daily shuffle. Centroid feeds only the sim-guard. Category `geo`. `london_scene` shares the `london_` prefix but is deliberately excluded from the city-tier system (keeps its single hard gate). Template:
```
# ── X_scene → "Sounds of X" ───────────────────────────────────────────────
# Theme:   A rotating showcase of notable artists from X.
# Sound:   Place-defined, genre-spanning (selection is by origin + popularity, not sound).
# Era/Geo: Any era · HARD origin gate = X (MB place + Last.fm scene fallback).
# Music:   The recognisable artists from X, one signature-ish song each, rotating coverage.
# Criteria: PINNED (always present, daily content refresh) · geo HARD gate · popularity-RANKED · artist-rotation · category geo.
# Enhance: lastfm_artist_tags scene fallback already in use; emb could diversify the per-artist song pick beyond "most-listened".
```
Per-member: `scotland_scene`→"Sounds of Scotland" {scotland} · `australia_scene`→"Sounds of Australia" {australia} · `london_scene`→"Sounds of London" {london}.

**Changes**
- `geo_showcase-1` · ✓ No change — gates, pinning and rotation are coherent and recently built (per project memory, the 3 pinned scenes were deliberately kept on hard gates).
- `geo_showcase-2` · Optional: per-artist song pick currently favours the artist's most-listened track; could rotate among an artist's top-N by *your* play history for personalisation · — → optional · additive · **low** · ⬜

## §5. Weather mixes (L1921–1928 + rainy/sunny/cosy/beach, family — WEATHER-GATED)

**Family (`_WEATHER_PROFILES`):** mood/centroid-led mixes that surface **only when live weather matches** (`_weather_boost`, hourly `--weather-context` cron). None are style-gated (weather isn't a genre — correct). Each is shaped by its `_PROFILE_MOOD_SIGNALS`; several add `_PROFILE_MOODTHEME`/`_PROFILE_LASTFM`/`_PROFILE_LYRIC_THEMES`. Template + per-member theme/sound:
```
# ── <weather> → "<Title>" ─────────────────────────────────────────────────
# Theme:   <the feeling of that weather>.
# Sound:   <centroid gloss>.
# Era/Geo: Any era · any origin.
# Music:   Mood-matched (no genre gate); surfaces when conditions hit.
# Criteria: centroid + mood signals (+moodtheme/lyric where mapped) · WEATHER-gated · category <cat>.
# Enhance: <per-member>.
```
| key | title | theme · sound |
|---|---|---|
| `rainy_day` | Rainy Day Mix | wistful melancholy · 72bpm, dim, low-val; +lastfm `rain` + rich lyric vocab |
| `sunny` | Sunny Mix | bright joy · 108bpm, bright .52, val .85; valence +1 |
| `cosy` | Cosy Mix | blanket-warm calm · 75bpm, low; moodtheme calm/soft/relaxing |
| `beach_vibes` | Beach Vibes Mix | sun-and-sand carefree · 100bpm, bright .48, val .82 |
| `stormy` | Stormy Mix | ominous drama · 80bpm, dark; lyric tense/volatile/dread |
| `foggy` | Foggy Mix | eerie dream-haze · 78bpm, bright .14; mysterious/dreamy |
| `snow_day` | Snow Day Mix | gentle playful hush · 88bpm, val .65 |
| `heatwave` | Heatwave Mix | languid sultry swelter · 92bpm, bright .40 |
| `frosty` | Frosty Mix | austere delicate chill · 82bpm; refined/gentle |
| `grey_skies` | Grey Skies Mix | overcast introspection · 82bpm, val .40; reflective/somber |
| `windy` | Windy Mix | restless drive · 110bpm, aro .62; nervous/volatile |
| `clear_night` | Clear Night Mix | ethereal nocturne · 76bpm; also HOUR-gated (20–5) |

**Changes**
- `weather-1` · ✓ No change to the set — mood-led + weather-gated is coherent throughout; centroids match the described feelings.
- `weather-2` · Optional: give `stormy`/`grey_skies`/`foggy` a light brightness/energy ceiling so a stray bright track can't slip into a dark-weather mix · — → optional · centroid already pulls correctly; marginal · **low** · ⬜
- `clear_night-1` · ✓ Confirm dual gate is intentional — weather-managed **and** `_HOUR_RESTRICTED (20,5)`; sensible belt-and-braces (a clear night is, by definition, night).

## §6. Seasonal mixes (L1929–1945 + autumn/winter/spring/summer_evening/festive, family — SEASON-GATED)

**Family:** two tiers. The 5 **broad** seasonal mixes (`autumn_mix`,`winter_mix`,`spring_mix`,`summer_evening`,`festive`) are in `_SEASONAL_PROFILES`; the 16 **seasonal-style** mixes (`spring_*`,`summer_*`,`autumn_*`,`winter_*`) are in `_PROFILE_SEASON` and eligible **only in their season**. Most seasonal-style mixes are hard style-gated. `festive` is a special: gated on a **holiday signal** (title keyword or Plex Holiday flag), not genre/centroid. **This section has the audit's highest concentration of genre-parent mismatches** — several mixes' `_PROFILE_STYLE_SIGNALS` positives belong to a Discogs parent that their `_PROFILE_GENRE_PARENT` set excludes, so the dominant-parent rule silently rejects the very tracks the mix wants (and the yield guard may then drop the mix entirely).

Per-member highlights: `festive`(holiday-gated, +0.5) · `spring_bloom`(bright hopeful, not style-gated) · `spring_acoustic`(folk, parent ✓) · `spring_strings`(classical, parent ✓) · `spring_jangle`(jangle-pop, parent {rock} ✓) · `summer_heat`(**disco/funk ↔ parent ⚠**) · `summer_breeze`(soft-rock/AM-pop, parent {rock} mild ⚠) · `summer_roadtrip`(anthemic, not style-gated) · `summer_tropical`(**latin/reggae/bossa ↔ parent {electronic} ✗**) · `autumn_leaves`(folk ✓) · `autumn_jazz`(jazz+soul ✓) · `autumn_rain`(melancholy, not style-gated) · `autumn_embers`(classic/blues rock ✓) · `winter_frost`(**ambient ↔ parent {classical} ⚠**) · `winter_cosy`(soul ✓) · `winter_nights`(downtempo, parent {electronic} ✓) · `winter_jazz`(jazz ✓).

**Changes**
- `summer_tropical-1` · Widen genre parent `{electronic}` → `{electronic, latin, reggae, "folk, world, & country", "funk / soul"}` · **offline: 3975→4543 (+568).** Not empty as I first thought — under `{electronic}` it catches tropical-/afro-**house** (`tropical`⊂`tropical house`) rather than the latin/reggae/bossa it names; widening admits the +568 authentic tracks · **low-med** · ⬜ (genuine but modest; the `reggae`⊂`reggaeton` overlap is on-theme)
- `summer_heat-1` · Widen genre parent `{electronic}` → `{electronic, "funk / soul"}` · **offline: 2696→4175 (+1479).** Confirmed — under `{electronic}` only the `club/dance` slice passes; `+funk/soul` restores the +1479 disco/funk the code comment explicitly intends · **med** · ✅
- `winter_frost-1` · Widen genre parent `{classical, "stage & screen"}` → add `electronic` · **offline: 4514→17374 (+12860).** Big jump — unblocks the `ambient`/`experimental ambient` half (Electronic parent). The size means **validate it stays frost-delicate, not generic electronic**, before applying (the centroid should still rank it down, but confirm) · **med** · ⚠
- `summer_breeze-1` · Widen genre parent `{rock}` → `{rock, pop}` · **offline: 3721→3839 (+118).** Marginal on its own — but `summer_breeze` is also a victim of the `am pop`→`dream pop` substring leak (Part C), which is the bigger issue here · **low** · ⬜ (do as part of the substring-leak fix)
- `festive-1` · ✓ No change — holiday-signal gate is the only sensible mechanism (Christmas is an event, not a sound), correctly the one place a Plex tag is still consulted.
- `seasonal-2` · ✓ The season-eligibility gate (`_PROFILE_SEASON`) and broad/style split are coherent; no change.
- `summer_roadtrip-1` · Optional: it's a *singalong-on-the-highway* concept but has no style gate or anthem lean; consider popularity +0.5 + a road/driving lyric overlap (already has `summer/travel` moodtheme) · — → optional · **low** · ⬜

## §7. Emotional / mood mixes (L1947–1955, 2136–2174, family — general pool)

**Family:** the core feelings shelf — mood/centroid/lyric-led, **no genre gate**, mostly anytime (general pool). Each is shaped by `_PROFILE_MOOD_SIGNALS` (+`_PROFILE_MOODTHEME`, and rich `_PROFILE_LYRIC_THEMES` on the vocal ones). Centroids are well-matched to the named feeling throughout (verified: `grief_release` 72bpm/val.18, `euphoric` 132bpm/val.90, `serene` aro.16, `defiant` aro.72). Template + per-member:
```
# ── <key> → "<Title>" ─────────────────────────────────────────────────────
# Theme:   <feeling>.   Sound: <centroid gloss>.   Era/Geo: any · any.
# Music:   <what fits>.
# Criteria: centroid + mood signals (+moodtheme/lyric/popularity where mapped) · category <cat>.
```
| key | title | theme / notes |
|---|---|---|
| `hopeful` | Brighter Days | optimistic uplift; lyric second_chance/healing |
| `yearning` | Longing | aching wistful longing; lyric unrequited/waiting |
| `triumphant` | Victory Lap | epic overcoming; moodtheme epic/powerful/uplifting |
| `serene` | Calm Waters | peaceful stillness; aro.16 |
| `tender` | Soft Spot | gentle warmth; lyric emotional_safety |
| `defiant` | No Apologies | rebellious resolve; moodclass aggressive |
| `vulnerable` | Heart on Sleeve | confessional fragility |
| `awe_wonder` | Awe & Wonder Mix | majestic/spiritual; moodtheme epic/dream/space |
| `grief_release` | Letting Go | mourning catharsis; val.18 |
| `nostalgia_mix` | Take Me Back | wistful memory |
| `dreamy_mix` | Dreamy Mix | ethereal drift |
| `moody_mix` | In a Mood | brooding introspection |
| `emotional` | All the Feels | high-vocal (.85) catharsis |
| `bittersweet` | Happy-Sad | joy-and-sadness at once |
| `cathartic` | Let It Out | intense release |
| `confidence_boost` | Feelin' Myself | swagger; popularity +1 |
| `empowering` | Unstoppable | anthemic power |
| `euphoric` | Cloud Nine | ecstatic peak; val.90, popularity +1 |
| `angst_mix` | Big Feelings | angry/nervous; moodclass aggressive |
| `romantic_mix` | Romantic Mix | tender romance |
| `daydreaming` | Head in the Clouds | low-arousal reverie |
| `fresh_start` | Fresh Start | new-chapter optimism; HOUR (4,10) |
| `melancholy` | In My Feels | sad/plaintive |
| `heartbreak` | Broken Hearts Club | devastation; lastfm breakup, valence −1 |

**Changes**
- `emotional-set-1` · ✓ No change — centroids, mood signals and lyric vocab all match their names; this shelf is well-tuned.
- `triumphant-1` / `empowering-1` · Optional: a weekday-morning soft lean (motivation context) · — → `(6,11,0.06)` · "Victory Lap"/"Unstoppable" are workout/morning-motivation adjacent; today they're context-blind (`monday_motivation` covers Mondays only) · **low** · ⬜
- `awe_wonder-1` · Optional: add a light evening/night lean · — → `(20,1,0.06)` · awe/cosmic listening skews night (its moodtheme is space/dream) · **low** · ⬜

## §8. Atmospheric / time-of-day mixes (L1956–1963, 2178–2189, family — time-gated)

**Family:** daypart-named aesthetic mixes. Most carry a hard `_HOUR_RESTRICTED` gate **and** a `_TIME_SOFT_BOOSTS` lean, so they only surface in-window. Mood/centroid-led, no genre gate (except `late_night`/`dinner`, also no gate). Audit focus = surfacing consistency.

| key | title | window (hard / soft) | notes |
|---|---|---|---|
| `sunrise` | Sunrise Mix | HOUR(4,10)+soft(5,9) | ✓ |
| `blue_hour` | Blue Hour Mix | HOUR(16,21)+soft(17,20) | ✓ |
| `golden_hour` | Golden Hour Mix | HOUR(16,21)+soft(15,20) | ✓ |
| `golden_afternoon` | Golden Afternoon Mix | HOUR(12,18)+soft(12,17) | ✓ |
| `sunset_mix` | Sunset Mix | soft(17,21) only | ⚠ no hard gate (peers have one) |
| `midnight` | Midnight Mix | HOUR(22,4)+soft(22,3) | ✓ |
| `three_am` | 3am Thoughts | HOUR(23,5)+soft(0,4); pop −1 | ✓ |
| `witching_hour` | Witching Hour Mix | HOUR(21,5)+soft(22,4) | ✓ |
| `starlit` | Starlit Mix | soft(21,4) only; pop −1 | ⚠ no hard gate (peers have one) |
| `overcast` | Overcast Mix | none | ⚠ duplicates weather-gated `grey_skies` |
| `main_character` | Main Character Mix | none; pop +1 | ✓ anytime confidence mix |
| `after_dark` | After Dark Mix | HOUR(20,6)+soft(22,4) | ✓ |
| `late_night` | Late Night Feels | TIME_BIASED(22,2) hard | ✓ cron-managed |
| `dinner` | Dinner Mix | TIME_BIASED(17,21) hard | ✓ cron-managed |

**Changes**
- `sunset_mix-1` · Add hard hour gate `_HOUR_RESTRICTED["sunset_mix"]=(16,21)` · — → present · every other dusk/night atmospheric mix is hard-gated; a "Sunset Mix" surfacing at 3am is off-concept · **low** · ✅
- `starlit-1` · Add hard hour gate `_HOUR_RESTRICTED["starlit"]=(20,5)` · — → present · same consistency point — a star-field night mix shouldn't appear midday · **low** · ✅
- `overcast-1` · Resolve overlap with `grey_skies`: either add `overcast` to `_WEATHER_PROFILES` (true cloudy-weather gate) **or** document it as the deliberate *anytime* introspective twin · — → decision · today it's an ungated near-duplicate of the weather-gated `grey_skies` (identical mood signals) · **low** · ⬜ (recommend: leave anytime, document)

## §9. Occasion / day-bound mixes (L1964–1972, 2193–2213, family — day/time-gated)

**Family:** context mixes pinned to a day-part or weekday via hard `_WEEKDAY_RESTRICTED`/`_HOUR_RESTRICTED` gates plus soft leans. Audit confirms the gates are coherent and thorough. Mood/centroid-led; `dinner_party` is the only style-gated member.

| key | title | gating | notes |
|---|---|---|---|
| `monday_motivation` | Monday Motivation Mix | Mon only, HOUR(5,12); pop +1 | ✓ |
| `midweek_reset` | Midweek Reset Mix | Tue–Thu only | ✓ |
| `friday_feeling` | Finally Friday | Fri only, HOUR(11,19) | ✓ |
| `sunday_scaries` | Sunday Scaries Mix | Sun only, HOUR(15,23) | ✓ |
| `treat_yourself` | Treat Yourself Mix | Fri/Sat boost, soft(17,24) | ✓ |
| `dinner_party` | Dinner Party Mix | HOUR(17,23); STYLE-GATED | ⚠ parent mismatch (below) |
| `housework_hustle` | Tidy Up | weekend boost, soft(9,15) | ✓ |
| `study_session` | Brain Food | HOUR(8,17); soft instrumental nudge; pop −1 | ✓ |
| `wind_down` | Wind-Down Mix | HOUR(18,2) | ✓ |
| `after_work` | Clock Out | Mon–Fri, HOUR(15,20) | ✓ |
| `friday_night` | Friday Night Mix | Fri, HOUR(18,3); pop +1 | ✓ |
| `weekend_mix` | Weekend Mix | Sat/Sun | ✓ |
| `sunday_morning` | Slow Sunday | Sun, HOUR(5,12) | ✓ |
| `lazy_sunday` | Lazy Sunday Mix | Sun, soft(11,18) | ✓ |
| `brunch_mix` | Bottomless Brunch | Sat/Sun, HOUR(8,13) | ✓ |
| `date_night` | Date Night Mix | HOUR(18,1); romance lyric | ✓ |

**Changes**
- `dinner_party-1` · Widen genre parent `{"funk / soul"}` → `{"funk / soul", jazz}` · its positives are `vocal jazz`/`lounge`/`bossa nova`/`smooth` (Discogs **Jazz** parent) + `soul` (Funk/Soul); under `{funk / soul}` the jazz/bossa/lounge majority is rejected by the dominant-parent rule · **med** · ✅
- `occasion-set-1` · ✓ No change to the rest — day/time gating is comprehensive and correct.

## §10. Activity mixes (L1973–1980, 2089–2106, 2217–2234, 2322–2339, family)

**Family:** "what you're doing" mixes. Pacing/energy is centroid-driven; instrumental-leaning ones (`meditation`/`spa_bath`/`yoga_stretch`/`power_nap`/`focus`/`deep_work`/`study_session`/`deep_reading`) carry a heavier vocal-suppression weight + soft instrumental style nudge + `_PROFILE_POPULARITY −1` (deep cuts), while physical ones (`workout`/`running`) lean +1 hits. Gates: `workout`/`running` HOUR(5,22) (no small-hours gym music); `focus`/`deep_work` HOUR(5,17)+Mon–Fri; `commute_mix` HOUR(6,9)+Mon–Fri; `night_drive` HOUR(19,4). All coherent.

| key | title | notes |
|---|---|---|
| `workout` | Beast Mode | 150bpm, pop+1, lastfm gym; ENH integrated_loudness |
| `running` | Runner's High | 160bpm, pop+1, cadence-friendly |
| `focus` | In the Zone | instrumental soft-nudge, pop−1, Mon–Fri |
| `deep_work` | Locked In | as focus, deeper |
| `chill` | Chill Mix | laid-back, anytime |
| `creative_flow` | In the Flow | kinetic-but-hypnotic, pop−1 |
| `gaming` | Game On | 130bpm, electronic/synthwave/hard-rock nudge |
| `gardening` | Green Thumb | sunny/pastoral |
| `yoga_stretch` | Yoga & Stretch | relaxed, soft(6,10) |
| `meditation` | Meditation Mix | 64bpm, relaxed, soft(6,10) |
| `spa_bath` | Spa Day | relaxed, soft(19,23) |
| `power_nap` | Forty Winks | 66bpm, soft(13,16) |
| `deep_reading` | Lost in a Book | instrumental, pop−1 |
| `cooking_mix` | Kitchen Disco | soul/funk nudge, soft(17,20) |
| `driving_mix` | Windows Down | pop+1, lastfm driving |
| `night_drive` | Night Drive Mix | HOUR(19,4), dark |
| `driving_singalong` | Sing in the Car | pop+1, high vocal (.85) |
| `road_trip` | Open Road | pop+1, adventurous lyric |
| `commute_mix` | Beat the Traffic | Mon–Fri HOUR(6,9) |
| `walking_mix` | Walk It Out | bright/breezy |

**Changes**
- `activity-loudness-1` · Add an `integrated_loudness`-consistency preference to the pacing mixes (`workout`,`running`,`focus`,`deep_work`,`sleep`) · — → new signal · steady loudness keeps cadence/concentration unbroken; `integrated_loudness` is computed but unused (needs a small scorer addition) · **med** · ⬜ (depends on §Roadmap infra)
- `activity-set-1` · ✓ No change otherwise — gates, leans and centroids match the activities.

## §11. Social / nostalgia mixes (L1981–1996, 2238, 2316–2333, family)

**Family:** "music with people / from your past." The nostalgia members use `_PROFILE_YEAR_WINDOW` to *enforce* age (`throwback_anthems`/`memory_lane` ≤now−10, `old_friends` ≤now−8, `school_days` 1990–now−12 = the user's life-era, per project memory) and lean +1 hits. `throwback_anthems` additionally runs the sophisticated **anthem gate** (`_PROFILE_ANTHEM_GATE`: an artist's own Last.fm top-10 **and** ≥100k global listeners) — its standout mechanism, keeping it to genuine signature hits. `campfire` is style-gated (folk/acoustic); the party members lean +1 with rich party lyric vocab.

| key | title | notes |
|---|---|---|
| `throwback_anthems` | Throwback Anthems Mix | anthem gate (top-10 + 100k), ≤now−10, pop+1, Fri/Sat |
| `old_friends` | Old Friends Mix | ≤now−8, pop+1, weekend |
| `campfire` | Campfire Mix | STYLE-gated folk/acoustic, parent ✓ |
| `cookout` | Cookout Mix | soul/funk nudge, pop+1, weekend, summer |
| `game_night` | Game Night Mix | HOUR(18,1), Fri/Sat, playful |
| `singalong` | Singalong Mix | album/arena/power-pop nudge, pop+1, Fri/Sat |
| `school_days` | School Days Mix | 1990–now−12 life-era, pop+1, coming-of-age lyric |
| `memory_lane` | Memory Lane Mix | ≤now−10, pop+1, wistful |
| `party` | Party Mix | pop+1, party lyric/moodtheme |
| `party_throwback` | Party Throwback Mix | pop+1, Fri/Sat, retro |
| `pre_party` | Pre-Party Mix | soft(17,23), Fri/Sat, pop+1 |
| `celebration` | Celebration Mix | pop+1, Fri/Sat, celebratory lyric |
| `cool_down` | Catch Your Breath | post-exertion calm; no gate |

**Changes**
- `social-set-1` · ✓ No change — the year windows, anthem gate and party leans all match the names; this section is a model for the rest.
- `singalong-1` · Optional: add a modest listener floor (e.g. 100k) so "Singalong" stays to songs people *know the words to* · — → optional · +1 lean already nudges; a floor would harden it · **low** · ⬜

## §12. Romance mixes (L1989–1996, 2262–2306, family)

**Family:** the love shelf, from butterflies to devotion. Mood/centroid/lyric-led; the new-romance batch (`crush`,`slow_burn`,`loved_up`,`long_distance`,`flirty`,`devotion`) all share an evening lean `(18,24,0.08)`; the jazz/classical romance mixes are hard style-gated. Rich `_PROFILE_LYRIC_THEMES` throughout (the romance mixes are the heaviest lyric-theme users — appropriate). `acoustic_romance` is **seed-defined** (Jack-Johnson `_SEED_ARTISTS` centroid) + romantic lyric themes, deliberately ungated (per project memory).

| key | title | notes |
|---|---|---|
| `crush` | Crushing | new-romance evening lean; infatuation lyric |
| `slow_burn` | Slow Burn Mix | evening; pining lyric |
| `moving_on` | Over It | breakup-recovery; letting_go lyric; **no time lean** |
| `loved_up` | Loved Up Mix | evening; euphoric affection |
| `long_distance` | Miles Apart | evening; missing_someone lyric |
| `flirty` | Make a Move | evening; flirtation lyric; lastfm sexy |
| `devotion` | All Yours | evening; devotion lyric |
| `wedding_day` | Wedding Day Mix | pop+1; promise_of_forever lyric; anytime (event) |
| `modern_romance` | Modern Romance Mix | mixed-signals lyric |
| `late_night_romance` | Late Night Romance Mix | HOUR(19,4); makeout lastfm |
| `romantic_dinner` | Romantic Dinner Mix | HOUR(17,23) |
| `love_songs` | Love Songs | all-consuming-love lyric |
| `slow_dance` | Slow Dance Mix | tender/intimate |
| `candlelight` | Candlelight Mix | HOUR(17,3); intimate |
| `first_date` | First Date Mix | flirtatious/hopeful |
| `romantic_jazz` | Romantic Jazz Mix | STYLE-gated jazz, parent ✓ |
| `jazz_dinner` | Jazz Dinner Mix | STYLE-gated jazz, HOUR(17,23), parent ✓ |
| `string_quartet` | String Quartet Mix | STYLE-gated classical, parent ✓ |
| `strings_romance` | Strings & Romance Mix | STYLE-gated classical, parent ✓ |
| `piano_romance` | Piano Romance Mix | STYLE-gated; **parent mismatch (piano jazz)** |
| `acoustic_romance` | Acoustic Romance Mix | SEED-defined (Jack Johnson) + romantic lyric |
| `indie_romance` | Indie Romance Mix | STYLE-gated rock; **parent edge (folk/electronic)** |
| `synthpop_romance` | Synth-Pop Romance Mix | STYLE-gated, parent {elec,pop,rock} ✓ |

**Changes**
- `piano_romance-1` · Widen genre parent `{classical, "stage & screen"}` → add `jazz` · **offline: 1453→1458 (+5).** Negligible — piano-jazz is rare in the library · **low** · ⬜ (recommend **deny** — not worth it)
- `indie_romance-1` · Widen genre parent `{rock}` → add folk/electronic/pop · **offline: 17965→19028 (+1063).** The pool is *already* very broad (17,965); the centroid/lyric stage does the real narrowing, so widening the gate adds little and risks more breadth · **low** · ⬜ (recommend **deny** — pool already ample)
- `moving_on-1` · Optional: add evening lean `(18,24,0.06)` to match its romance siblings · — → optional · breakup-recovery listening also skews evening · **low** · ⬜
- `romance-set-1` · ✓ No change otherwise — leans, gates and the seed/lyric machinery match the names.

## §13. Genre mixes (L1998–2047, family — STYLE-GATED)

**Family:** ~45 genre-pure mixes, all hard style-gated (Discogs subgenre + parent). I checked **every** member's `_PROFILE_STYLE_SIGNALS` positives against its `_PROFILE_GENRE_PARENT` set: this section is **largely clean** — strong evidence the prior gating refactor (per project memory) worked. Most carry the +0.5 recognisability lean; ambient/vaporwave/downtempo/post_rock dig deep (−1); dark/late ones (`deep_house`,`downtempo`,`ambient_drift`,`smooth_jazz`,`techno`-adjacent) carry evening soft leans. `prog_rock` is correctly Last.fm-gated. `lofi_beats` is **retired** (`_RETIRED_PROFILES` — no Discogs "Lo-Fi"; its sound is Downtempo). On insertion each gets a full block; template:
```
# ── <genre> → "<Title>" ───────────────────────────────────────────────────
# Theme/Sound/Music: <the genre>.  Era/Geo: any · any.
# Criteria: HARD style gate(<subgenres>) parent {<parents>} · <pop lean> · <time lean> · category <cat>.
# Enhance: <per-member>.
```
Parent-check results (all **✓ OK** unless noted): funk_disco {elec,funk/soul} · neo_soul · motown_soul · after_hours_rnb · acid_jazz {elec,funk/soul,jazz} · boom_bap · conscious_flow · g_funk · trap_mode · house_party · deep_house · techno · trance · dnb · bass_drop {elec,hiphop} · uk_garage · synthwave · industrial {elec,rock} · vaporwave · downtempo · hyperpop {elec,pop} · classic_rock · heavy_riffs · punk_energy · garage_grunge · emo_poppunk · britpop_rock · blues_bar {blues,rock} · psych_haze · prog_rock(LastFM) · stoner_rock · reggae_dub {elec,reggae} · afrobeat · latin_heat {elec,latin} · bossa_samba {jazz,latin} · celtic_folk · ska {reggae,rock} · bebop · **swing_bigband ⚠** · **smooth_jazz ⚠** · country_roads · outlaw_country · bluegrass · rockabilly_surf · cinematic_epic · ambient_drift · post_rock · chiptune · gospel.

**Changes**
- `genre-set-1` · ✓ No change for the ~43 clean members — gate, parent, leans and time leans all match the genre names.
- `swing_bigband-1` · Optional: parent `{jazz}` → `{jazz, pop}` · positive `traditional pop` (Sinatra-era) is often Discogs **Pop** parent and gets rejected; the swing/big-band core is unaffected · **low** · ⬜
- `smooth_jazz-1` · Optional: parent `{jazz}` → `{jazz, "funk / soul"}` · positive `quiet storm` is Discogs **Funk / Soul** parent (rejected today); core smooth/cool jazz unaffected · **low** · ⬜
- `lofi_beats-1` · ✓ Confirm `lofi_beats` stays retired (documented; no membership signal survives the pure-Discogs gate).

## §14. City scene mixes (L2048–2083, family — STYLE-GATED + GEO-TIERED)

**Family:** 36 mixes (Glasgow/London/Melbourne × 12) — each a city's take on a genre. Two gates: the genre **style gate** (as §13) **and** the tiered, country-bounded **geo gate** (`_PROFILE_GEO_TIERS`: city → region → nation, hard-bounded to the country, no full-pool fallback — per project memory). DJ-flow ordering uses the energy arc for the dance ones (`glasgow_house/underground/bass`, `london_garage/grime/dubstep/jungle`, `melbourne_club/techno`). Selection is geo; `_dj_order` keeps it sonic. I parent-checked all 36: **two issues found.**
```
# ── <city>_<genre> → "<City> <Genre> Mix" ─────────────────────────────────
# Theme/Sound/Music: <city>'s <genre> scene.   Era/Geo: any era · GEO-TIERED to <city>→<region>→<nation>.
# Criteria: HARD style gate(<subgenres>) parent {<parents>} · geo tiers · <pop/dj> · category <cat>.
```
**Glasgow (Scotland-bounded):** folk · dream · indie · soul · postrock · anthems · **synth ⚠** · postpunk · house · underground(−1) · bass · late — all parents ✓ except `glasgow_synth`.
**London (England/UK-bounded):** dub · soul · jazz · triphop · **mod ✗** · britpop · indie · calling · garage · grime · dubstep · jungle — all parents ✓ except `london_mod`.
**Melbourne (Victoria/Australia-bounded):** folk · dream · soul · sunset · indie · pubrock · hiphop · postpunk · psych · garagepunk · club · techno(−1) — all parents ✓.

**Changes**
- `london_mod-1` · **CONFIRMED BUG (offline-verified) — "London Mod Mix" contains 0 mod-rock; it serves modern-classical & film scores.** The positive `"mod"` substring-matches Discogs `modern`(5136)/`post-modern`(1297)/`modal`(741), and the `{classical,"stage & screen"}` parent locks it there — sampled passers were David Russo & John Paesano film cues and He Zhanhao's *Butterfly Lovers* violin concerto. **Two-part fix:** (a) parent `{classical,"stage & screen"}`→`{rock}`; (b) tighten the positive — drop bare `"mod"`, use `"mod revival"` and rely on `british invasion`/`freakbeat`/`merseybeat` (under `{rock}`, bare `"mod"` would re-leak `"modern rock"`). Pool 5977 (all wrong) → 659 raw → fewer once `"mod"` is tightened · **med** · ✅ **(highest-value fix in the audit; see "Systemic finding: substring leaks" in Part C)**
- `glasgow_synth-1` · Widen genre parent `{electronic}` → `{electronic, rock, pop}` · positives include `new wave` (Discogs **Rock**) and `dance-rock` (**Rock**); only the `electro`/`synth pop` half passes today · **low** · ✅
- `city-set-1` · ✓ No change for the other 34 — genre gate + geo tiers are coherent (geo-tier system recently built per project memory).
- `glasgow_house-1` / `london_soul-1` / `london_jazz-1` · Optional micro-edges: a single cross-parent positive each (`disco`→Funk/Soul; `acid jazz`→Jazz/Elec) sits outside the parent set; impact is marginal (core positives dominate) · — → optional · **low** · ⬜ (recommend deny — noise)

---

# PART B — Standalone builders (L7091–8094)

**Overall:** these 11 are **personalisation-driven** (live Plex `viewCount`/`userRating`/play-history), not cache-criteria mixes — so the centroid/listeners/Discogs/lyrics audit axes mostly don't apply (correctly). The four that *do* use the cache (`release_radar`, `discover_weekly`, `daily_mixes`, `deep_cuts`) use acoustic centroid + Last.fm tag affinity. All have strong existing docstrings; the 6-facet summary below would be added/standardised on each. This part is **in excellent shape** — only one doc fix + a few enhancement options.

| builder | title | what it is | gates / logic |
|---|---|---|---|
| `build_on_repeat` (7091) | On Repeat | can't-stop-playing right now | 30-day window, ≥2 plays, artist-cap 3, drop userRating≤4, recency×rating score |
| `build_repeat_rewind` (7153) | Repeat Rewind | last month's On Repeat, gone quiet | plays in 30–70d ago, NOT in last 30d, ≥2 (rating-weighted), artist-cap 2 |
| `build_release_radar` (7287) | Release Radar | new releases like your taste | albums in expanding 14–90d window, album-reps (Last.fm playcount + acoustic), known-artist-first, 0.65 ac + 0.35 tag affinity |
| `build_discover_weekly` (7409) | Discover Weekly | weekly fresh-music mixtape | 70% new-artist / 30% familiar, 20% stretch (rank 300+), discovery sweet-spot (pop≈0.6), never-played, 1/artist+1/album, interleaved |
| `build_daily_mixes` (7542) | Daily Mix 1–6 | k-means acoustic clusters | numpy k-means; 40% history + 60% library per cluster, centroid-ranked |
| `build_rediscovery` (7687) | Rediscovery | loved but silent 6–24mo | rating≥7 OR plays≥3, last-play 6–24mo ago, longest-neglected first |
| `build_time_capsule` (7751) | Time Capsule | your standout earlier PLAY-years | top-3 play-years ≥2 apart (Plex API), per-year top plays, gone-quiet 90d, interleaved/deduped |
| `build_time_machine` (7846) | "Month YYYY" | this time, years ago | ±21d around today in a past year, rotates year weekly, gone-quiet 90d |
| `build_deep_cuts` (7900) | Deep Cuts | unheard tracks by artists you love | top-15 artists (6mo), their ≤2-play tracks ranked vs artist's well-played centroid + rating bonus |
| `build_top_songs` (7999) | Top Songs YYYY | most-played per calendar year | per-year top plays, ≥20 distinct, artist-cap 5, drop rating≤4, past years immutable (+prev-year 60-day grace) |
| `build_all_time_favourites` (8080) | All-Time Favourites | lifetime most-played | `viewCount:desc`, stop at 0-play, inline (not canonical) dedup |

Representative facet block (e.g. `build_on_repeat`):
```
# Theme:   What you're hammering THIS month — genuine current obsession.
# Sound:   N/A (personalisation, not acoustic).   Era/Geo: any · any.
# Music:   Songs you've played ≥2× in 30 days, your top-rated copies, ≤3 per artist.
# Criteria: 30-day window (never expands), drop userRating≤4, recency×rating score.
# Enhance: skipCount could deprioritise tracks you skip; otherwise complete.
```

**Changes**
- `repeat_rewind-1` · Doc fix: the docstring says "haven't been played in the last **21 days**" but the code uses `silence_cutoff = now − 30 days` · 21 → 30 (or change code if 21 intended) · stale doc vs behaviour · **low** · ✅
- `builders-set-1` · ✓ No logic change — every builder matches its name/description; gates (windows, caps, rating drops, immutability) are all coherent and well-commented.
- `enh-skipcount-1` · Enhancement (optional): feed Plex `skipCount` into `on_repeat`/`daily_mixes`/`top_songs` to deprioritise repeatedly-skipped tracks · — → new signal · `skipCount` is unused; would sharpen "songs you actually love" vs "songs that autoplay" · **med** · ⬜
- `enh-emb-builders-1` · Enhancement (optional): use `emb_effnet`/`emb_musicnn` nearest-neighbour in `discover_weekly`/`release_radar`/`deep_cuts` instead of (or alongside) the 10-d centroid for sharper "sounds-like" discovery · — → new infra (§Roadmap) · **med** · ⬜
- `enh-releasetypes-1` · Enhancement (optional): `release_radar` could use MB `release_types` to prefer album/EP tracks over single-only drops, or vice-versa · — → optional · **low** · ⬜

---

# PART C — Validated findings, recommended set, coverage & roadmap

## ⭐ Systemic finding: substring leaks in the style gate (offline-verified)

`_has_required_style` matches a positive against Discogs subgenres with a **plain substring test** (`sub in tag`). Three short positives leak into unintended subgenres (cache = 135,359 tracks):

| profile(s) | positive | leaks into | effect |
|---|---|---|---|
| `london_mod` | `"mod"` | `modern`(5136), `post-modern`(1297), `modal`(741) | **mix is 100% wrong** — modern-classical/film scores, 0 mod-rock (compounded by the `{classical}` parent) |
| `yacht_rock`, `summer_breeze` | `"am pop"` | `dream pop`(2845) | pulls dream-pop (dre**am pop**) into a soft-rock mix |
| `reggae_dub`, `london_dub` | `"dub"` | `dubstep`(1228) | pulls dubstep into a roots-reggae/dub mix (parent `{electronic}` doesn't stop it) |

**Recommended fix (two options):**
- `substr-fix-targeted` (safe, immediate, ✅): tighten the three positives — `london_mod`: drop `"mod"` → `"mod revival"`; `yacht_rock`/`summer_breeze`: drop `"am pop"` (the soft-rock/AC/sophisti-pop positives carry the mix) or use `"sunshine pop"`; `reggae_dub`/`london_dub`: `"dub"` → `"dub reggae"`/rely on `roots reggae`+`dancehall`. Plus `london_mod` parent → `{rock}`. **med · ✅**
- `substr-fix-systemic` (better, follow-up, ⚠): switch positive-matching from substring to **word-boundary/token** matching in `_has_required_style` — cleanly kills all three leaks and prevents future ones, but touches every style-defined mix, so **validate the full-pool yields across all 130 gated mixes before shipping** (hyphen/`'`-token edge cases like `synth-pop`, `drum'n'bass`, `post-bop` must still match). **med-high · ⚠**

## Validated offline numbers (style-gate yield, before → after proposed parent)

Run via `/tmp/parent_gate_sim.py` against the live cache (read-only). These supersede my pre-simulation guesses:

| change | before → after | verdict |
|---|---|---|
| `london_mod-1` | 5977 (all wrong) → 659 raw | ✅ confirmed bug — fix parent **and** `"mod"` |
| `winter_frost-1` (+electronic) | 4514 → 17374 | ✅ but ⚠ validate it stays frost-delicate |
| `summer_heat-1` (+funk/soul) | 2696 → 4175 | ✅ restores intended +1479 disco/funk |
| `glasgow_synth-1` (+rock,pop) | 7677 → 8771 | ✅ admits Scottish new-wave/synth (Simple Minds etc.) — matters post-geo-gate |
| `summer_tropical-1` (+latin/reggae/…) | 3975 → 4543 | ⬜ modest (+568); currently catches tropical-house not latin |
| `dinner_party-1` (+jazz) | 2842 → 3187 | ✅ admits the vocal-jazz/bossa it names (+345) |
| `yacht_rock-1` (+pop,funk/soul) | 3721 → 3950 | ⬜ modest (+229); substring leak matters more |
| `summer_breeze-1` (+pop) | 3721 → 3839 | ⬜ marginal (+118); do with substring fix |
| `indie_romance-1` (+folk,elec,pop) | 17965 → 19028 | ⬜ deny — pool already ample |
| `piano_romance-1` (+jazz) | 1453 → 1458 | ⬜ deny — negligible (+5) |

> **The four decision tiers (✅ recommend / ⬜ optional / ❌ deny / 🔭 roadmap) — with what each changes and why — are consolidated in "Decision tiers" at the very bottom of this document.**

## Coverage summary (every playlist accounted for)

| § | section | #playlists | changes proposed |
|---|---|---|---|
| 1 | gap-fill pop | 11 | situationship-1/2, sad_bangers-1, power_ballads-1, synth_pop-1 (+5 ✓ no-change) |
| 2 | rock/elec/scores | 6 | indie_rock-1 (note) (+5 ✓) |
| 3 | decades | 7 | decades-1/2, decade_20s-1 |
| 4 | geo showcases | 3 | geo_showcase-1/2 |
| 5 | weather | 12 | weather-1/2, clear_night-1 |
| 6 | seasonal | 21 | summer_tropical/heat-1, winter_frost-1, summer_breeze-1, festive-1, summer_roadtrip-1 |
| 7 | emotional | 24 | triumphant/empowering-1, awe_wonder-1 (+rest ✓) |
| 8 | atmospheric/ToD | 14 | sunset_mix-1, starlit-1, overcast-1 |
| 9 | occasion | 16 | dinner_party-1 (+rest ✓) |
| 10 | activity | 20 | activity-loudness-1 (+rest ✓) |
| 11 | social | 13 | singalong-1 (+rest ✓) |
| 12 | romance | 23 | piano_romance-1, indie_romance-1, moving_on-1 (+rest ✓) |
| 13 | genre | 45 | swing_bigband-1, smooth_jazz-1, dub-leak (+rest ✓) |
| 14 | city scenes | 36 | london_mod-1, glasgow_synth-1, dub-leak (+rest ✓) |
| B | standalone builders | 11 | repeat_rewind-1 + 3 enhancements |

Total: **~231 mood mixes + 11 builders documented**; the rest carry `✓ No change`. Nothing silently skipped.

## Data-enhancement roadmap (bigger, separate efforts — not in the ✅ set)

1. **Embeddings (`emb_effnet`/`emb_musicnn`) — highest-value unused asset.** Build a similarity layer: k-means discovery clusters; seed-artist nearest-neighbour to *define* fuzzy mixes (`situationship`, `indie_rock`, `acoustic_romance`) and sharpen `discover_weekly`/`deep_cuts`/`release_radar` beyond the 10-d centroid.
2. **Word-boundary style matcher** (`substr-fix-systemic`) — prevents the whole substring-leak class.
3. **`integrated_loudness` scorer** — loudness-consistency for pacing mixes (`activity-loudness-1`).
4. **`danceability_hl` co-gate** — sanity-check dance/EDM mixes against Essentia danceability.
5. **`skipCount`** — deprioritise repeatedly-skipped tracks in the play-history builders.
6. **`lyric_lang`** — English-only gate for lyric-led mixes; potential language scenes.

## Verification & apply plan

1. **Approve/deny** by ID/tier (reply e.g. *"approve tier ✅, plus summer_tropical-1; deny the rest"*).
2. For each approved **style-gate** change: re-run `/tmp/parent_gate_sim.py` (extended with the substring fixes) to confirm before/after pool size is sane (≥ `_MIN_PROFILE_YIELD` 25, not ballooned) — already done for the parent fixes above; the substring fixes need their own run.
3. Apply approved changes to the relevant dicts, each with a `# WHY:` note; insert the full 6-facet comment block above every `_MOOD_PROFILES` entry + builder docstrings.
4. `python -c "import ast; ast.parse(open('utilities/meloday_extras.py').read())"` — syntax gate.
5. Spot-check a changed mix's pool with the sim; on prod (`daedalus`), `python utilities/meloday_extras.py --playlist mood_mixes --debug` for a live confirmation (dev has no live Plex).

---

# 🎚️ Decision tiers

Reply to approve/deny by **tier** (e.g. *"approve ✅, plus summer_tropical-1"*) or by **ID**. The comment blocks for all 261 mixes ship regardless — these tiers only decide *behavioural* changes.

## ✅ Tier 1 — Recommended (12 items) · *fix a demonstrated mismatch between a mix's name/description and what it actually plays or when*

| ID | what it changes | why |
|---|---|---|
| `london_mod-1` | parent `{classical,stage&screen}`→`{rock}`; drop `"mod"`→`"mod revival"` | "London Mod" plays 0 mod-rock today — it's modern-classical/film-scores (`"mod"`⊂`"modern"`, 5,977 wrong tracks) |
| `substr-fix-targeted` | drop `"am pop"` (yacht_rock, summer_breeze); `"dub"`→`"dub reggae"`+roots/dancehall (reggae_dub, london_dub) | `"am pop"`⊂`"dream pop"` and `"dub"`⊂`"dubstep"` — dream-pop & dubstep leak into the wrong mixes |
| `summer_heat-1` | parent `+ "funk / soul"` | admits the disco/funk Summer Heat names but currently can't (+1,479) |
| `winter_frost-1` | parent `+ electronic` | admits ambient Winter Frost names but can't (+12,860 — **validate stays delicate**) |
| `glasgow_synth-1` | parent `+ rock, pop` | admits Scottish new-wave/synth (Simple Minds etc.) currently rejected (+1,094) |
| `dinner_party-1` | parent `+ jazz` | admits the vocal-jazz/bossa it names but rejects today (+345) |
| `synth_pop-1` | add evening soft-time lean | the name is "Synth-Pop **Nights**" yet it surfaces all day |
| `sunset_mix-1` | add hard hour-gate `(16,21)` | every other dusk/night mix is gated; today it can surface at 3am |
| `starlit-1` | add hard hour-gate `(20,5)` | same — a star-field mix shouldn't appear midday |
| `situationship-1` | add evening soft-time lean | the only one of 6 new-romance mixes missing the shared evening lean |
| `power_ballads-1` | add 100k listener floor | "Lighters Up" is a known-anthems concept with no fame floor |
| `repeat_rewind-1` | docstring "21 days"→"30" | docstring contradicts the code (`silence_cutoff = 30d`) |

## ⬜ Tier 2 — Optional (≈14 items) · *low-risk niceties; the mixes work fine without them — your taste call*

| ID(s) | what it changes | why |
|---|---|---|
| `sad_bangers-1`, `restless-1`, `neoclassical-1`, `triumphant-1`/`empowering-1`, `awe_wonder-1`, `moving_on-1` | add a contextual daypart/evening soft lean | each has a natural time-of-day but no surfacing bias |
| `summer_tropical-1` | parent `+ latin/reggae/world/funk-soul` | more authentic latin/reggae (+568); today it leans tropical-house |
| `yacht_rock-1`(parent), `summer_breeze-1`(parent) | parent `+ pop`/`+ funk-soul` | marginal (+229/+118); do alongside the substring fix |
| `swing_bigband-1`, `smooth_jazz-1` | parent micro-widen | one cross-parent positive each (`traditional pop`, `quiet storm`) |
| `singalong-1` | add listener floor | keep "Singalong" to songs people know the words to |
| `summer_roadtrip-1` | pop +0.5 / road lyric overlap | reinforce the highway-singalong feel |
| `overcast-1` | weather-gate **or** document as anytime | it's an ungated near-duplicate of weather-gated `grey_skies` |
| `geo_showcase-2`, `activity-loudness-1` | personalised song pick / loudness pacing | additive; `activity-loudness-1` needs the loudness scorer (roadmap) |

## ❌ Tier 3 — Recommend deny (≈6 items) · *the simulation showed these do nothing useful*

| ID(s) | why deny |
|---|---|
| `piano_romance-1` | +5 tracks — piano-jazz is near-absent in the library |
| `indie_romance-1` | pool is already 17,965; the centroid/lyric stage narrows it — widening the gate adds breadth, not value |
| `situationship-2` | it's a *vibe* mix, not a *hits* mix — a listener floor would fight its purpose |
| `decades-2` | `time_capsule` already personalises "your top era" |
| `glasgow_house-1`/`london_soul-1`/`london_jazz-1` | single cross-parent positives; marginal noise the core positives already dominate |

## 🔭 Tier 4 — Enhancement roadmap (6 items) · *new capability from unused data — separate builds, not config tweaks*

| item | what it adds | why it's separate |
|---|---|---|
| **Embeddings** (`emb_effnet`/`emb_musicnn`) | "sounds-like" similarity + discovery clustering; defines the fuzzy mixes far better than the 10-d centroid | needs a new similarity layer + per-mix wiring; **highest-value unused asset** |
| **Word-boundary gate matcher** (`substr-fix-systemic`) | kills the whole substring-leak class permanently | touches all 124 gated mixes — needs full-pool validation |
| **`integrated_loudness` scorer** | consistent-loudness pacing for workout/focus/sleep | new scorer + per-profile opt-in |
| **`danceability_hl` co-gate** | second-opinion sanity check on dance/EDM mixes | new co-gate logic |
| **`skipCount`** | deprioritise repeatedly-skipped tracks in the play-history builders | new live-Plex signal plumbing |
| **`lyric_lang`** | English-only gate / language-scene mixes | new gate + (for scenes) new profiles |
