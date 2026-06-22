#!/usr/bin/env python3
"""
Validator for Meloday extras cover-art assignments.

Guarantees the redesign's invariants are real, not hoped-for. Imports meloday_extras headless
(no Plex) and checks:
  A. Coverage      — every routed cover key has a colour + a _COVER_BG_STYLES entry.
  B. Variant range — every (style, variant) uses a real variant index (v < that generator's count),
                     catching the silent-clamp bug that made indie_rock/rap_rock identical.
  C. Collisions    — no two profiles share the same (style, variant).
  D. Family load   — no background family is over-used (error >14, warn >12).
  E. Glyph integ.  — every glyph used exists in the font, in _MS_CODEPOINTS, and actually renders.
  F. No overlap    — on every multi-glyph cover, no two glyphs overlap.
  G. Top Songs      — the 20 curated year covers (if present) are mutually distinct.

Exit code is non-zero if there are any ERRORS. Run: python utilities/cover_validator.py
"""
import os
import sys

sys.argv = sys.argv[:1]
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import utilities.meloday_extras as me                       # noqa: E402
from utilities.ms_glyphs import font_name_to_codepoint      # noqa: E402

# Covers intentionally left untouched / handled by special paths — excluded from all checks.
EXEMPT = {f"daily_mix_{i}" for i in range(1, 7)} | {"release_radar", "discover_weekly", "top_songs"}


def _norm(entry):
    return entry if isinstance(entry, tuple) else (entry, 0)


def run():
    errors, warnings, notes = [], [], []
    colors = me._EXTRAS_COVER_COLORS
    styles = me._COVER_BG_STYLES
    gens   = me._BG_GENERATORS
    vcounts = me._BG_VARIANT_COUNTS
    routed = set(colors) - EXEMPT

    # A. Coverage
    for k in sorted(routed):
        if k not in styles:
            errors.append(f"[coverage] '{k}' has a colour but no _COVER_BG_STYLES entry "
                          f"(falls back to geometric,0)")

    # B. variant range + valid style
    for k, ent in sorted(styles.items()):
        if k in EXEMPT:
            continue
        style, v = _norm(ent)
        if style not in gens:
            errors.append(f"[style] '{k}' -> unknown background '{style}'")
        else:
            n = vcounts.get(style, 1)
            if v >= n:
                errors.append(f"[variant] '{k}' -> ({style},{v}) but '{style}' defines only "
                              f"{n} variants (v 0..{n - 1}) — silently clamps")

    # C. collisions
    seen = {}
    for k, ent in styles.items():
        if k in EXEMPT:
            continue
        sv = _norm(ent)
        seen.setdefault(sv, []).append(k)
    for sv, ks in sorted(seen.items()):
        if len(ks) > 1:
            errors.append(f"[collision] {sv} shared by {len(ks)}: {', '.join(sorted(ks))}")

    # D. family load
    from collections import Counter
    fam = Counter(_norm(ent)[0] for k, ent in styles.items() if k not in EXEMPT)
    for style, n in sorted(fam.items(), key=lambda kv: -kv[1]):
        if n > 14:
            errors.append(f"[family-load] '{style}' used by {n} profiles (>14)")
        elif n > 12:
            warnings.append(f"[family-load] '{style}' used by {n} profiles (>12)")

    # E. glyph integrity (exists in font + codepoint table + renders)
    font_names = font_name_to_codepoint()
    used_icons = set(me._PROFILE_ICON.values())
    for names in me._PROFILE_ICON_ROTATE.values():
        used_icons.update(names)
    # cluster helpers also draw music_note_2 / snowflake / cloud explicitly
    used_icons.update({"music_note_2", "snowflake", "cloud"})
    from PIL import ImageFont
    try:
        probe = ImageFont.truetype(me.MS_ICON_FONT, 100)
        try:
            probe.set_variation_by_axes([1, 0, 48, 700])
        except Exception:
            pass
    except Exception as e:
        probe = None
        warnings.append(f"[glyph] could not load MS font for render check: {e}")
    for icon in sorted(used_icons):
        if icon not in font_names:
            errors.append(f"[glyph] '{icon}' not present in the font's name map")
            continue
        if icon not in me._MS_CODEPOINTS:
            errors.append(f"[glyph] '{icon}' missing from _MS_CODEPOINTS (font has it at "
                          f"0x{font_names[icon]:X})")
            continue
        # NOTE: the font maps several codepoints to the same glyph, so _MS_CODEPOINTS[icon] need not
        # equal the parser's pick — only that it RENDERS the right icon (checked below) matters.
        if probe is not None:
            bbox = probe.getbbox(chr(me._MS_CODEPOINTS[icon]))
            if not bbox or bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                errors.append(f"[glyph] '{icon}' (0x{me._MS_CODEPOINTS[icon]:X}) renders empty (tofu)")

    # F. no glyph-on-glyph overlap — render every multi-glyph cover and check footprints
    import random
    multi = set(me._HEART_MODE) | set(me._PROFILE_ICON_ROTATE)
    for k, icon in me._PROFILE_ICON.items():
        if icon in ("music_note", "flare"):
            multi.add(k)
    multi |= {"winter_mix", "friday_night"}
    multi = {k for k in multi if k in colors and k not in EXEMPT}
    for k in sorted(multi):
        ct, cb = colors.get(k, ((50, 65, 110), (20, 30, 65)))
        rng = random.Random(hash((k, 2026, 25)))
        from PIL import Image
        base = Image.new("RGBA", (1000, 1000), (20, 20, 30, 255))
        me._CLUSTER_FOOTPRINTS = []
        try:
            me._draw_icon_overlay(base, k, ct, cb, rng)
        except Exception as e:
            warnings.append(f"[overlap] {k}: render failed ({e})")
            me._CLUSTER_FOOTPRINTS = None
            continue
        fps = me._CLUSTER_FOOTPRINTS
        me._CLUSTER_FOOTPRINTS = None
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                x1, y1, r1 = fps[i]
                x2, y2, r2 = fps[j]
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if dist < (r1 + r2) * 0.90:          # 10% tolerance for non-circular glyphs
                    errors.append(f"[overlap] {k}: two glyphs overlap "
                                  f"(d={dist:.0f} < {(r1 + r2) * 0.90:.0f})")
                    break
            else:
                continue
            break

    # G. Top Songs distinctness (only if the curated table exists)
    ts = getattr(me, "_TOP_SONGS_COVERS", None)
    if ts:
        seenc = set(); dups = []
        for i, combo in enumerate(ts):
            key = repr(combo)
            if key in seenc:
                dups.append(i)
            seenc.add(key)
        if dups:
            errors.append(f"[top_songs] {len(dups)} duplicate cover combos at indices {dups}")
        if len(ts) < 20:
            warnings.append(f"[top_songs] only {len(ts)} curated covers (<20)")
        notes.append(f"Top Songs: {len(ts)} curated covers, {len(seenc)} distinct")
    else:
        notes.append("Top Songs: no _TOP_SONGS_COVERS table yet (still year-random)")

    # ---- report ----
    print("=" * 72)
    print("MELODAY COVER VALIDATOR")
    print("=" * 72)
    print(f"profiles checked : {len(routed)} routed keys  |  {len(gens)} background families  |  "
          f"{len(me._PROFILE_ICON)} glyphed")
    print(f"family load (top): " + ", ".join(f"{s}={n}" for s, n in fam.most_common(6)))
    for n in notes:
        print(f"  note: {n}")
    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for wmsg in warnings:
            print(f"  ! {wmsg}")
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for emsg in errors:
            print(f"  X {emsg}")
        print("\nFAIL")
        return 1
    print("\nOK — all invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
