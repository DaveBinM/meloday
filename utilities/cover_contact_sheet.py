#!/usr/bin/env python3
"""
Offline contact sheet for Meloday extras covers.

Renders every extras cover into one labelled grid so the whole set can be eyeballed for
uniqueness / suitability / composition without a live Plex server. `_generate_extras_cover`
is Plex-independent, so this runs fully offline (only Daily Mix collage covers need Plex and
are skipped).

Modes:
  --mode deterministic  (default) replicate the cover pipeline in-memory with a pinned ISO-week
                        seed — NO writes to the real covers dir, stable for diffing.
  --mode prod           call _generate_extras_cover (writes real .webp to the covers dir) then
                        re-open it — the truest preview of what ships.

Examples:
  python utilities/cover_contact_sheet.py
  python utilities/cover_contact_sheet.py --filter glasgow --out /tmp/glasgow.png
  python utilities/cover_contact_sheet.py --names love_songs,romantic_mix,celebration,winter_mix
  python utilities/cover_contact_sheet.py --family equalizer
"""
import math
import os
import random
import sys

_ARGV = list(sys.argv)                        # keep our CLI args before neutralizing argv
sys.argv = sys.argv[:1]                       # neutralize meloday_extras' argparse on import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import utilities.meloday_extras as me        # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

EXEMPT = {f"daily_mix_{i}" for i in range(1, 7)} | {"release_radar", "discover_weekly"}
PIN_YEAR, PIN_WEEK = 2026, 25                 # fixed seed → stable sheet


def _title_for(key):
    return me._MOOD_MIX_NAMES.get(key, key.replace("_", " ").title())


def render_inmemory(key, year=PIN_YEAR, week=PIN_WEEK):
    """Replicate _generate_extras_cover's pipeline in-memory (no disk writes), pinned seed.
    Mirrors the prod rng call order (jitter top, jitter bottom, then generator, then icon)."""
    ct, cb = me._EXTRAS_COVER_COLORS.get(key, ((50, 65, 110), (20, 30, 65)))
    ent = me._COVER_BG_STYLES.get(key, ("geometric", 0))
    style, v = ent if isinstance(ent, tuple) else (ent, 0)
    rng = random.Random(hash((key, year, week)))
    jit = lambda c: tuple(max(0, min(255, ch + rng.randint(-18, 18))) for ch in c)
    ct, cb = jit(ct), jit(cb)
    img = me._render_bg(style, ct, cb, v, rng)
    img = me._add_bottom_vignette(img)
    img = me._draw_icon_overlay(img, key, ct, cb, rng)
    text_style = "bar" if key in me._MOOD_PROFILE_KEYS else "default"
    img = me._apply_cover_text(img, _title_for(key), None, accent_color=ct, text_style=text_style)
    return img.convert("RGB")


def render_prod(key):
    p = me._generate_extras_cover(key, _title_for(key))
    return Image.open(p).convert("RGB") if p else None


def build_sheet(keys, out_path, mode="deterministic", thumb=300, cols=8):
    pad, label_h = 10, 22
    rows = (len(keys) + cols - 1) // cols
    cell_w, cell_h = thumb + pad, thumb + label_h + pad
    sheet = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    try:
        lfont = ImageFont.truetype(os.path.join(_REPO_ROOT, "assets", "fonts", "Circular", "Circular-Book.ttf"), 16)
    except Exception:
        lfont = ImageFont.load_default()
    render = render_prod if mode == "prod" else render_inmemory
    for i, key in enumerate(keys):
        try:
            cov = render(key)
        except Exception as e:
            print(f"  [warn] {key}: {e}")
            cov = None
        r, c = divmod(i, cols)
        x, y = pad + c * cell_w, pad + r * cell_h
        if cov is not None:
            sheet.paste(cov.resize((thumb, thumb), Image.LANCZOS), (x, y))
        draw.text((x + thumb // 2, y + thumb + label_h // 2), key, font=lfont, fill=(180, 184, 196), anchor="mm")
    sheet.save(out_path)
    print(f"Wrote {out_path}  ({len(keys)} covers, {cols}x{rows}, mode={mode})")
    return out_path


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Meloday extras cover contact sheet")
    ap.add_argument("--mode", choices=["deterministic", "prod"], default="deterministic")
    ap.add_argument("--filter", help="substring filter on cover keys")
    ap.add_argument("--names", help="comma-separated explicit keys")
    ap.add_argument("--family", help="only keys whose _COVER_BG_STYLES family == this")
    ap.add_argument("--out", default="/tmp/cover_sheet.png")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=300)
    args = ap.parse_args(argv)

    all_keys = sorted(set(me._EXTRAS_COVER_COLORS) - EXEMPT)
    if args.names:
        keys = [k.strip() for k in args.names.split(",") if k.strip()]
    elif args.family:
        keys = [k for k in all_keys
                if (me._COVER_BG_STYLES.get(k, ("", 0)) or ("", 0))[0] == args.family]
    elif args.filter:
        keys = [k for k in all_keys if args.filter.lower() in k.lower()]
    else:
        keys = all_keys
    if not keys:
        print("No matching keys.")
        return 1
    build_sheet(keys, args.out, mode=args.mode, thumb=args.thumb, cols=args.cols)
    return 0


if __name__ == "__main__":
    sys.exit(_main(_ARGV[1:]))
