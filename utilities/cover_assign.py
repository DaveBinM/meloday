#!/usr/bin/env python3
"""Authoring tool: regenerate a bespoke, genre-driven, collision-free _COVER_BG_STYLES and
print it to /tmp/new_bg_styles.txt for splicing into meloday_extras.py.
WARNING: this REGENERATES the whole map from the cluster/PIN logic below — re-splicing it
OVERWRITES any hand edits to _COVER_BG_STYLES. Run: PYTHONPATH=. python3 utilities/cover_assign.py
Then splice /tmp/new_bg_styles.txt over the _COVER_BG_STYLES block and run cover_validator.py."""
import sys, re
sys.argv = sys.argv[:1]
import utilities.meloday_extras as me
from collections import defaultdict

SRC = open("utilities/meloday_extras.py").read().splitlines()

# ---- parse facet blocks: key -> {music, theme, cat} ----
facets = {}
buf = []
for line in SRC:
    s = line.strip()
    if s.startswith("#"):
        buf.append(s)
        continue
    m = re.match(r'^"([a-z0-9_]+)":\s*\{', s)
    if m:
        key = m.group(1)
        blob = " ".join(buf).lower()
        cat = None
        cm = re.search(r'cat:([a-z_]+)', blob)
        if cm: cat = cm.group(1)
        music = ""
        for b in buf:
            if b.lower().startswith("# music:"): music = b.split(":",1)[1].strip().lower()
        theme = ""
        for b in buf:
            if b.lower().startswith("# theme:"): theme = b.split(":",1)[1].strip().lower()
        facets[key] = {"music": music, "theme": theme, "cat": cat}
    buf = []

# ---- cluster -> ordered family preference (now incl. round-2 families) ----
CF = {
 "electronic": ["equalizer","grid_perspective","laser_fan","concentric_pulse","waveform","circuit","spiral","starburst"],
 "hiphop":     ["cassette","vinyl_grooves","halftone","cityscape","brushstrokes","waveform"],
 "soul_funk":  ["disco_ball","vinyl_grooves","halftone","confetti","brushstrokes","circles"],
 "rock":       ["guitar","amp_stack","low_poly","triangles","shards","geometric"],
 "metal_punk": ["guitar","amp_stack","shards","triangles"],
 "jazz":       ["jazz_club","brushstrokes","smoke","candle_glow"],
 "classical":  ["strings","columns","arc_sweep","aurora"],
 "folk":       ["meadow","pine_forest","mountains","woodgrain","sun_horizon","gingham"],
 "latin_global":["beach","kente","palm_sunburst","sun_horizon","confetti"],
 "ambient":    ["cosmos","starfield","smoke","clouds","concentric_pulse","moonlight","aurora","spiral"],
 "pop":        ["halftone","confetti","disco_ball","floating_circles","starburst","diamond"],
 "cinematic":  ["film_strip","starfield","columns","mountains"],
 "chiptune":   ["pixel_grid","circuit","grid_perspective"],
 "atmos":      ["clouds","smoke","starfield","sun_horizon","aurora","waves","rainfall"],
 "emotional":  ["smoke","brushstrokes","low_poly","rainfall","waves","triangles","arc_sweep","starfield"],
 "romance":    ["candle_glow","bokeh","arc_sweep","smoke","brushstrokes","diamond","floating_circles"],
 "party":      ["disco_ball","confetti","starburst","laser_fan","floating_circles","circles"],
 "energetic":  ["motion","lightning","equalizer","shards","confetti","traffic"],
 "calm":       ["zen","clouds","grid_paper","starfield","smoke","aurora","woodgrain"],
 "place_scot": ["tartan"], "place_lon": ["cityscape"], "place_aus": ["sun_horizon"],
 "misc":       ["geometric","diamond","arc_sweep","floating_circles","circles","triangles","waves","chevrons","spiral"],
}

# ---- explicit per-key family PIN (weather literal + theme + disliked re-pins); highest priority ----
PIN = {
 # weather / time-of-day → literal families (ripples & rainfall are rain-only)
 "rainy_day":"ripples","autumn_rain":"rainfall","stormy":"clouds","foggy":"smoke",
 "snow_day":"snowfall","frosty":"snowfall","winter_frost":"snowfall","winter_nights":"moonlight",
 "sunny":"starburst","sunrise":"sun_horizon","sunset_mix":"sun_horizon","golden_hour":"sun_horizon",
 "golden_afternoon":"sun_horizon","heatwave":"sun_horizon","clear_night":"moonlight","blue_hour":"clouds",
 "starlit":"starfield","overcast":"clouds","grey_skies":"clouds","windy":"clouds",
 "sleep":"moonlight","midnight":"moonlight","three_am":"moonlight",
 # genre / theme → new literal families
 "boom_bap":"cassette","lofi_beats":"cassette","throwback_anthems":"cassette","school_days":"cassette","memory_lane":"cassette",
 "chiptune":"pixel_grid","gaming":"pixel_grid","hyperpop":"pixel_grid",
 "neoclassical":"columns","string_quartet":"columns","spring_strings":"columns",
 "synth_pop":"waveform",
 "love_songs":"bokeh","late_night_romance":"bokeh","modern_romance":"bokeh",
 "dreamy_mix":"clouds","daydreaming":"clouds",
 "study_session":"grid_paper","deep_work":"grid_paper","focus":"grid_paper","deep_reading":"grid_paper",
 "empowering":"lightning","confidence_boost":"lightning","monday_motivation":"lightning","triumphant":"lightning",
 "spring_bloom":"blossom","spring_mix":"blossom","spring_jangle":"blossom","gardening":"blossom",
 "celtic_folk":"pine_forest","campfire":"pine_forest","folk_acoustic":"pine_forest","cosy":"pine_forest",
 "summer_tropical":"tropical_leaves","reggae_dub":"tropical_leaves","afrobeat":"tropical_leaves","beach_vibes":"tropical_leaves",
 "cookout":"gingham","brunch_mix":"gingham","cooking_mix":"gingham","dinner_party":"gingham",
 "ambient_drift":"cosmos","awe_wonder":"cosmos",
 "funk_disco":"disco_ball","motown_soul":"disco_ball","dance_pop":"disco_ball","friday_night":"disco_ball",
 "festive":"holiday_lights",
 "britpop_rock":"mod_target","london_britpop":"mod_target","london_mod":"mod_target",
 # disliked / mismatched re-pins
 "gospel":"stained_glass","soundtracks":"film_strip","cinematic_epic":"film_strip","main_character":"film_strip",
 "candlelight":"candle_glow","romantic_dinner":"candle_glow","jazz_dinner":"candle_glow",
 "road_trip":"open_road","summer_roadtrip":"open_road","driving_mix":"open_road",
 "spa_bath":"zen","wind_down":"zen","meditation":"zen","yoga_stretch":"zen",
 "yacht_rock":"sun_horizon","sunday_morning":"clouds","cathartic":"smoke","power_ballads":"smoke",
 "celebration":"confetti","commute_mix":"chevrons","romantic_jazz":"brushstrokes",
 # amp_stack users stay on the (redrawn) amp_stack
 "glasgow_anthems":"amp_stack","glasgow_postpunk":"amp_stack","london_calling":"amp_stack",
 "melbourne_psych":"amp_stack","rockabilly_surf":"amp_stack",
}

# ---- round 3: flagged genre/activity covers + every city mix → best-fit family (overrides above) ----
PIN.update({
 # jazz → jazz_club; strings → strings; rock → guitar
 "london_jazz":"jazz_club","smooth_jazz":"jazz_club","romantic_jazz":"jazz_club","jazz_dinner":"jazz_club",
 "acid_jazz":"jazz_club","autumn_jazz":"jazz_club","winter_jazz":"jazz_club","bossa_samba":"jazz_club",
 "dinner":"jazz_club","bebop":"jazz_club",
 "string_quartet":"strings","spring_strings":"strings","strings_romance":"strings","neoclassical":"strings",
 # rock: jangle/indie/punk → guitar; loud/heavy/classic → amp_stack (keeps guitar varied + under cap)
 "indie_rock":"guitar","emo_poppunk":"guitar","punk_energy":"guitar",
 "garage_grunge":"amp_stack","heavy_riffs":"amp_stack","defiant":"amp_stack","classic_rock":"amp_stack",
 "ska":"checkerboard","stoner_rock":"desert","outlaw_country":"desert",
 "running":"motion","walking_mix":"motion","workout":"motion",
 "commute_mix":"traffic","beach_vibes":"beach","summer_tropical":"beach",
 "wedding_day":"wedding_rings","after_hours_rnb":"lounge","late_night":"lounge","after_dark":"laser_fan",
 "prog_rock":"prism","post_rock":"crescendo",
 "spring_acoustic":"meadow","spring_bloom":"meadow","spring_mix":"meadow",
 "latin_heat":"palm_sunburst","summer_heat":"sun_horizon","afrobeat":"kente",
 "decade_10s":"equalizer","decade_70s":"vinyl_grooves","yearning":"moonlight",
 "dinner":"candle_glow","time_capsule":"radial",   # dinner: calmer bg; time_capsule: plainer for the glyph
 # round 4: retired-family reassignments (neon/flames/marquee_lights/papel_picado) + new families
 "blues_bar":"smoke","slow_burn":"smoke","party":"laser_fan","trap_mode":"waveform",
 "swagger":"cassette","synthwave":"grid_perspective","vaporwave":"grid_perspective",
 "folk_acoustic":"acoustic_guitar",
 # city mixes — best family per sub-genre (Melbourne Sunset kept on palm_sunburst)
 "glasgow_folk":"meadow","glasgow_dream":"clouds","glasgow_indie":"guitar","glasgow_soul":"disco_ball",
 "glasgow_postrock":"crescendo","glasgow_anthems":"amp_stack","glasgow_synth":"grid_perspective",
 "glasgow_postpunk":"guitar","glasgow_house":"laser_fan","glasgow_underground":"grid_perspective",
 "glasgow_bass":"waveform","glasgow_late":"smoke",
 "london_dub":"beach","london_soul":"disco_ball","london_triphop":"smoke","london_mod":"mod_target",
 "london_britpop":"mod_target","london_indie":"guitar","london_calling":"guitar","london_garage":"laser_fan",
 "london_grime":"cityscape","london_dubstep":"waveform","london_jungle":"waveform",
 "melbourne_folk":"meadow","melbourne_dream":"clouds","melbourne_soul":"disco_ball",  # melbourne_sunset → PIN_EXACT
 "melbourne_indie":"guitar","melbourne_pubrock":"amp_stack","melbourne_hiphop":"cassette","melbourne_postpunk":"guitar",
 "melbourne_psych":"prism","melbourne_garagepunk":"guitar","melbourne_club":"grid_perspective","melbourne_techno":"grid_perspective",
})

# ---- round 5: genre-fit re-pins ----
PIN.update({
 "rave_cave": "grid_perspective",   # same family as Glasgow Underground, rave palette
 "pre_party": "confetti",           # a party design, not abstract chevrons
 "rap_rock":  "shards",             # aggressive nu-metal, not generic triangles
 "uk_garage": "laser_fan",          # club/2-step lasers (like london_garage)
})

# ---- round 6: techno warehouse → industrial circuit (neon palette, not plain bars) ----
PIN.update({"techno": "circuit"})

# ---- keys pinned to an EXACT (family, variant) — must NEVER change (user-loved covers) ----
# WHY: melbourne_sunset is loved as-is; pinning only the family would let a sibling (latin_heat, also
# palm_sunburst) claim variant 0 alphabetically and bump it. Reserve its exact slot instead.
PIN_EXACT = {"melbourne_sunset": ("palm_sunburst", 0)}

# ---- explicit cluster for music-defined keys ----
EXPLICIT = {
 # Meloday+ gap-fill
 "situationship":"emotional","sad_bangers":"pop","power_ballads":"rock","restless":"emotional",
 "neoclassical":"classical","yacht_rock":"soul_funk","swagger":"hiphop","chart_pop":"pop",
 "dance_pop":"electronic","indie_pop":"pop","synth_pop":"electronic",
 "indie_rock":"rock","post_grunge":"rock","rap_rock":"metal_punk","festival_edm":"electronic",
 "soundtracks":"cinematic","rave_cave":"electronic",
 # decades
 "decade_60s":"soul_funk","decade_70s":"latin_global","decade_80s":"electronic","decade_90s":"pop",
 "decade_00s":"electronic","decade_10s":"pop","decade_20s":"electronic",
 # geo scenes
 "scotland_scene":"place_scot","london_scene":"place_lon","australia_scene":"place_aus",
 # genre-sonic
 "funk_disco":"soul_funk","neo_soul":"soul_funk","motown_soul":"soul_funk","after_hours_rnb":"hiphop",
 "acid_jazz":"jazz","boom_bap":"hiphop","conscious_flow":"hiphop","g_funk":"hiphop","trap_mode":"hiphop",
 "lofi_beats":"ambient","house_party":"electronic","deep_house":"electronic","techno":"electronic",
 "trance":"electronic","dnb":"electronic","bass_drop":"electronic","uk_garage":"electronic",
 "synthwave":"electronic","industrial":"metal_punk","vaporwave":"electronic","downtempo":"ambient",
 "hyperpop":"pop","classic_rock":"rock","heavy_riffs":"metal_punk","punk_energy":"metal_punk",
 "garage_grunge":"rock","emo_poppunk":"rock","britpop_rock":"rock","blues_bar":"jazz","psych_haze":"rock",
 "prog_rock":"rock","stoner_rock":"rock","reggae_dub":"latin_global","afrobeat":"latin_global",
 "latin_heat":"latin_global","bossa_samba":"jazz","celtic_folk":"folk","ska":"pop","bebop":"jazz",
 "swing_bigband":"jazz","smooth_jazz":"jazz","country_roads":"folk","outlaw_country":"folk",
 "bluegrass":"folk","rockabilly_surf":"rock","cinematic_epic":"cinematic","ambient_drift":"ambient",
 "post_rock":"ambient","chiptune":"chiptune","gospel":"soul_funk",
 # city: glasgow
 "glasgow_folk":"folk","glasgow_dream":"ambient","glasgow_indie":"rock","glasgow_soul":"soul_funk",
 "glasgow_postrock":"ambient","glasgow_anthems":"rock","glasgow_synth":"electronic","glasgow_postpunk":"metal_punk",
 "glasgow_house":"electronic","glasgow_underground":"electronic","glasgow_bass":"electronic","glasgow_late":"ambient",
 # city: london
 "london_dub":"latin_global","london_soul":"soul_funk","london_jazz":"jazz","london_triphop":"ambient",
 "london_mod":"pop","london_britpop":"rock","london_indie":"rock","london_calling":"metal_punk",
 "london_garage":"electronic","london_grime":"hiphop","london_dubstep":"electronic","london_jungle":"electronic",
 # city: melbourne
 "melbourne_folk":"folk","melbourne_dream":"ambient","melbourne_soul":"soul_funk","melbourne_sunset":"latin_global",
 "melbourne_indie":"rock","melbourne_pubrock":"rock","melbourne_hiphop":"hiphop","melbourne_postpunk":"metal_punk",
 "melbourne_psych":"rock","melbourne_garagepunk":"metal_punk","melbourne_club":"electronic","melbourne_techno":"electronic",
 # spotify-style non-mood (kept, glyphed) — give them distinct concept families
 "on_repeat":"misc","repeat_rewind":"misc","rediscovery":"ambient","deep_cuts":"ambient",
 "time_capsule":"misc","time_machine":"electronic","all_time_favourites":"party",
}

EXPLICIT.update({
 # time of day / day of week / aesthetic
 "sunrise":"atmos","blue_hour":"atmos","midnight":"ambient","three_am":"emotional",
 "golden_afternoon":"calm","overcast":"atmos","starlit":"ambient","witching_hour":"emotional",
 "monday_motivation":"energetic","midweek_reset":"calm","friday_feeling":"party","sunday_scaries":"emotional",
 "main_character":"cinematic","golden_hour":"atmos","sunset_mix":"atmos","after_dark":"electronic",
 "after_work":"calm","friday_night":"party","weekend_mix":"party","sunday_morning":"calm",
 "lazy_sunday":"calm","brunch_mix":"pop","date_night":"romance","morning":"atmos","late_night":"ambient",
 # activity
 "treat_yourself":"party","dinner_party":"party","housework_hustle":"energetic","study_session":"calm",
 "wind_down":"calm","yoga_stretch":"calm","meditation":"calm","deep_reading":"calm","creative_flow":"calm",
 "gaming":"chiptune","gardening":"folk","spa_bath":"calm","power_nap":"calm","deep_work":"calm",
 "cooking_mix":"folk","cool_down":"calm","driving_mix":"energetic","night_drive":"ambient",
 "driving_singalong":"energetic","road_trip":"folk","commute_mix":"energetic","walking_mix":"energetic",
 "workout":"energetic","running":"energetic","focus":"calm","chill":"calm","happy":"pop","dinner":"jazz",
 # social / nostalgia
 "throwback_anthems":"pop","old_friends":"folk","campfire":"folk","cookout":"party","game_night":"party",
 "singalong":"party","school_days":"pop","memory_lane":"emotional","party_throwback":"party",
 # romance
 "crush":"romance","slow_burn":"romance","moving_on":"emotional","loved_up":"romance","long_distance":"romance",
 "flirty":"romance","devotion":"romance","wedding_day":"romance","romantic_mix":"romance","modern_romance":"romance",
 "late_night_romance":"romance","romantic_dinner":"romance","love_songs":"romance","slow_dance":"romance",
 "first_date":"romance","romantic_jazz":"jazz","jazz_dinner":"jazz","string_quartet":"classical",
 "strings_romance":"romance","piano_romance":"romance","acoustic_romance":"folk","indie_romance":"romance",
 "synthpop_romance":"romance","heartbreak":"emotional","candlelight":"romance",
 # emotional / mood
 "hopeful":"emotional","yearning":"emotional","triumphant":"energetic","serene":"calm","tender":"emotional",
 "defiant":"metal_punk","vulnerable":"emotional","awe_wonder":"ambient","grief_release":"emotional",
 "nostalgia_mix":"emotional","dreamy_mix":"ambient","moody_mix":"emotional","emotional":"emotional",
 "bittersweet":"emotional","cathartic":"emotional","confidence_boost":"energetic","empowering":"energetic",
 "euphoric":"party","angst_mix":"emotional","daydreaming":"ambient","fresh_start":"pop","melancholy":"emotional",
 # weather / season
 "stormy":"atmos","foggy":"atmos","snow_day":"atmos","heatwave":"atmos","frosty":"atmos","grey_skies":"atmos",
 "windy":"atmos","clear_night":"atmos","festive":"party","rainy_day":"atmos","sunny":"atmos","cosy":"calm",
 "spring_bloom":"folk","spring_acoustic":"folk","spring_strings":"classical","spring_jangle":"pop",
 "summer_heat":"party","summer_breeze":"folk","summer_roadtrip":"folk","summer_tropical":"latin_global",
 "autumn_leaves":"folk","autumn_jazz":"jazz","autumn_rain":"atmos","autumn_embers":"rock",
 "winter_frost":"atmos","winter_cosy":"calm","winter_nights":"ambient","winter_jazz":"jazz",
 "spring_mix":"folk","summer_evening":"atmos","autumn_mix":"folk","winter_mix":"atmos",
 "beach_vibes":"latin_global",
})

CAT2CLUSTER = {
 # genre / sound families
 "rock_classic":"rock","rock_indie":"rock","rock_psych":"rock","rock_punk":"metal_punk","rock_heavy":"metal_punk",
 "pop":"pop","electronic_house_techno":"electronic","electronic_bass":"electronic","electronic_edm_pop":"electronic",
 "electronic_chill":"ambient","hiphop":"hiphop","soul_funk_rnb":"soul_funk","jazz_lounge":"jazz",
 "folk_acoustic":"folk","country":"folk","world_latin":"latin_global","reggae_ska":"latin_global",
 "instrumental_cinematic":"cinematic",
 # mood / theme families
 "happy_bright":"pop","party_fun":"party","euphoric_triumphant":"energetic","defiant_intense":"energetic",
 "melancholy_blue":"emotional","heartbreak_longing":"emotional","nostalgic_throwback":"emotional",
 "romantic":"romance","calm_unwind":"calm","dreamy_ethereal":"ambient",
 # activity
 "workout_energy":"energetic","focus_study":"calm","wellness_sleep":"calm","driving":"energetic",
 # functional / context
 "weather":"atmos","time_of_day":"atmos","season_autumn":"atmos","season_winter":"atmos",
 "season_spring":"atmos","season_summer":"atmos","festive":"party","era":"pop","geo_scene":"misc",
}

EXEMPT = {f"daily_mix_{i}" for i in range(1,7)} | {"release_radar","discover_weekly","top_songs"}
routed = [k for k in me._EXTRAS_COVER_COLORS if k not in EXEMPT]

def cluster_for(key):
    if key in EXPLICIT: return EXPLICIT[key]
    f = facets.get(key, {})
    if f.get("cat") in CAT2CLUSTER: return CAT2CLUSTER[f["cat"]]
    return "misc"

# group keys by cluster
by_cluster = defaultdict(list)
for k in routed:
    by_cluster[cluster_for(k)].append(k)

print("# MISC members:", sorted(by_cluster.get("misc", [])))
print("# AMBIENT members:", sorted(by_cluster.get("ambient", [])))
print("# ENERGETIC members:", sorted(by_cluster.get("energetic", [])))
print()
ORDER = ["electronic","hiphop","soul_funk","rock","metal_punk","jazz","classical","folk",
         "latin_global","chiptune","cinematic","pop","ambient","atmos","romance","party",
         "energetic","emotional","calm","place_scot","place_lon","place_aus","misc"]

used = set()
used.add(("radial", 0))   # reserve discover_weekly's existing cover
for _f, _v in PIN_EXACT.values():
    used.add((_f, _v))    # reserve exact-pinned slots so the loop can't take them
fam_max = defaultdict(lambda: -1)
assign = {}

def _lowest_free(fam):
    cap = me._BG_VARIANT_COUNTS.get(fam, 1); v = 0
    while (fam, v) in used: v += 1
    return v if v < cap else None

# Process PINNED keys first (so they claim their literal family), then the rest by cluster.
def _order(k):
    return (0 if k in PIN else 1, ORDER.index(cluster_for(k)) if cluster_for(k) in ORDER else 99, k)

for key in sorted([k for k in routed if k not in PIN_EXACT], key=_order):
    placed = None
    if key in PIN:                                   # pinned family wins unless it's full
        v = _lowest_free(PIN[key])
        if v is not None:
            placed = (PIN[key], v)
    if placed is None:                               # balanced pick across the cluster's families
        fams = CF.get(cluster_for(key), CF["misc"])
        best = None
        for idx, fam in enumerate(fams):
            v = _lowest_free(fam)
            if v is None:
                continue
            if best is None or (v, idx) < best[0]:
                best = ((v, idx), fam, v)
        if best is None:                             # everything full — spill anywhere
            for fam in me._BG_GENERATORS:
                v = _lowest_free(fam)
                if v is not None:
                    best = ((v, 99), fam, v); break
        placed = (best[1], best[2])
    fam, v = placed
    used.add((fam, v)); fam_max[fam] = max(fam_max[fam], v); assign[key] = (fam, v)

assign["discover_weekly"] = ("radial", 0)
for k, fv in PIN_EXACT.items():
    assign[k] = fv          # restore exact-pinned covers (e.g. melbourne_sunset)

# ---- report ----
print("# family max-variant needed (count = max+1):")
for fam in sorted(fam_max, key=lambda f:-fam_max[f]):
    print(f"#   {fam:18} needs {fam_max[fam]+1:2d} variants   (have {me._BG_VARIANT_COUNTS.get(fam,'?')})")
print(f"# total routed assigned: {len(assign)}  | clusters: {dict((c,len(v)) for c,v in sorted(by_cluster.items()))}")
print()

# ---- emit dict grouped by cluster ----
out = ["_COVER_BG_STYLES = {"]
keyset_by_cluster = defaultdict(list)
for k,(f,v) in assign.items():
    keyset_by_cluster[cluster_for(k) if k!="discover_weekly" else "misc"].append(k)
for cl in ORDER + [c for c in by_cluster if c not in ORDER]:
    ks = sorted([k for k in assign if (cluster_for(k) if k!="discover_weekly" else "misc")==cl])
    if not ks: continue
    out.append(f"    # --- {cl} ---")
    for k in ks:
        f,v = assign[k]
        out.append(f'    "{k}": ("{f}", {v}),')
out.append("}")
open("/tmp/new_bg_styles.txt","w").write("\n".join(out))
print("wrote /tmp/new_bg_styles.txt", "(", len(assign), "entries )")
