#!/usr/bin/env python3
"""
Material Symbols glyph helper for Meloday cover art.

The bundled variable font ships no `.codepoints` sidecar and fontTools is not a
project dependency, so this module recovers the full Material Symbol *name -> codepoint*
map by parsing the font's `cmap` (format 12) and `post` (v2.0) tables directly. That
map is the source of truth for:
  * choosing a glyph for a cover (browse names, render a labelled reference sheet to SEE them), and
  * validating that every glyph used in meloday_extras really exists in this font (cover_validator.py).

WHY a hand parser: the font has ~3,969 named icons but no friendly name list anywhere on disk;
extracting names from the font itself means any glyph we pick is guaranteed present (no tofu).

CLI:
  python utilities/ms_glyphs.py --grep music            # list names containing "music" + hex
  python utilities/ms_glyphs.py --grep weather --sheet   # render a labelled sheet of matches
  python utilities/ms_glyphs.py --names sailing,forest,casino --sheet --out /tmp/glyphs.png
  python utilities/ms_glyphs.py --count                  # just print how many glyphs were found
"""
import os
import struct
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
MS_ICON_FONT = os.path.join(
    _REPO_ROOT, "assets", "Material_Symbols_Rounded",
    "MaterialSymbolsRounded-VariableFont_FILL,GRAD,opsz,wght.ttf")

_NAME2CP_CACHE = None


def font_name_to_codepoint(font_path=MS_ICON_FONT):
    """Parse the TTF and return {icon_name: codepoint}. Pure stdlib (struct), no fontTools.

    Material Symbols glyph names ARE the icon names (e.g. "equalizer", "wb_sunny"); each is
    reachable both by a ligature and by a Private-Use-Area codepoint. We map codepoint->glyphID
    from `cmap` and glyphID->name from `post`, then invert to name->codepoint.
    """
    global _NAME2CP_CACHE
    if _NAME2CP_CACHE is not None and font_path == MS_ICON_FONT:
        return _NAME2CP_CACHE

    d = open(font_path, "rb").read()
    u16 = lambda o: struct.unpack(">H", d[o:o + 2])[0]
    s16 = lambda o: struct.unpack(">h", d[o:o + 2])[0]
    u32 = lambda o: struct.unpack(">I", d[o:o + 4])[0]

    num_tables = u16(4)
    tables = {}
    off = 12
    for _ in range(num_tables):
        tag = d[off:off + 4].decode("latin1")
        tables[tag] = (u32(off + 8), u32(off + 12))
        off += 16

    # ---- cmap: codepoint -> glyphID (prefer format 12 for full Unicode incl. >0xFFFF) ----
    cmoff = tables["cmap"][0]
    nsub = u16(cmoff + 2)
    best = None
    for i in range(nsub):
        p = cmoff + 4 + i * 8
        pid, eid, so = u16(p), u16(p + 2), u32(p + 4)
        fmt = u16(cmoff + so)
        score = (fmt == 12) * 10 + (pid == 3 and eid == 10) * 4 + (pid == 3 and eid == 1) * 2 + 1
        if best is None or score > best[0]:
            best = (score, cmoff + so, fmt)
    _, so, fmt = best

    cp2gid = {}
    if fmt == 12:
        ngroups = u32(so + 12)
        for g in range(ngroups):
            gp = so + 16 + g * 12
            sc, ec, sg = u32(gp), u32(gp + 4), u32(gp + 8)
            for c in range(sc, ec + 1):
                cp2gid[c] = sg + (c - sc)
    elif fmt == 4:
        seg_x2 = u16(so + 6)
        segc = seg_x2 // 2
        end_o = so + 14
        start_o = end_o + seg_x2 + 2
        delta_o = start_o + seg_x2
        range_o = delta_o + seg_x2
        for s in range(segc):
            end = u16(end_o + s * 2)
            start = u16(start_o + s * 2)
            delta = s16(delta_o + s * 2)
            ro = u16(range_o + s * 2)
            for c in range(start, end + 1):
                if c == 0xFFFF:
                    continue
                if ro == 0:
                    g = (c + delta) & 0xFFFF
                else:
                    g = u16(range_o + s * 2 + ro + (c - start) * 2)
                    if g:
                        g = (g + delta) & 0xFFFF
                if g:
                    cp2gid[c] = g
    else:
        raise ValueError(f"unsupported cmap format {fmt}")

    # ---- post v2.0: glyphID -> name ----
    name2cp = {}
    po, pl = tables.get("post", (0, 0))
    ver = u32(po) if po else 0
    if po and ver == 0x00020000:
        ng = u16(po + 32)
        idx = [u16(po + 34 + i * 2) for i in range(ng)]
        p = po + 34 + ng * 2
        names = []
        while p < po + pl and len(names) < 70000:
            ln = d[p]
            names.append(d[p + 1:p + 1 + ln].decode("latin1"))
            p += 1 + ln
        MAC = 258  # first 258 indices are the standard Macintosh glyph set
        gid2name = {}
        for gid, ix in enumerate(idx):
            if ix >= MAC and (ix - MAC) < len(names):
                gid2name[gid] = names[ix - MAC]
        for cp, gid in cp2gid.items():
            n = gid2name.get(gid)
            # keep the lowest codepoint for a name (the canonical PUA slot)
            if n and (n not in name2cp or cp < name2cp[n]):
                name2cp[n] = cp
    else:
        raise ValueError(f"post table not v2.0 (got {hex(ver)}) - cannot recover names")

    if font_path == MS_ICON_FONT:
        _NAME2CP_CACHE = name2cp
    return name2cp


def render_reference_sheet(names, out_path, cols=8, cell=160, glyph_px=96):
    """Render a labelled grid of the given glyph names so they can be eyeballed. Skips unknown names."""
    from PIL import Image, ImageDraw, ImageFont
    name2cp = font_name_to_codepoint()
    names = [n for n in names if n in name2cp]
    if not names:
        print("No known glyph names to render.")
        return None
    rows = (len(names) + cols - 1) // cols
    label_h = 26
    W, H = cols * cell, rows * cell
    img = Image.new("RGB", (W, H), (24, 26, 32))
    draw = ImageDraw.Draw(img)
    gfont = ImageFont.truetype(MS_ICON_FONT, glyph_px)
    try:
        gfont.set_variation_by_axes([1, 0, 48, 700])  # FILL, GRAD, opsz, wght
    except Exception:
        pass
    try:
        lfont = ImageFont.truetype(os.path.join(_REPO_ROOT, "assets", "fonts", "Circular", "Circular-Book.ttf"), 15)
    except Exception:
        lfont = ImageFont.load_default()
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        cx = c * cell + cell // 2
        cy = r * cell + (cell - label_h) // 2
        draw.text((cx, cy), chr(name2cp[name]), font=gfont, fill=(232, 234, 240), anchor="mm")
        draw.text((cx, r * cell + cell - label_h // 2 - 2), name, font=lfont, fill=(150, 156, 168), anchor="mm")
    img.save(out_path)
    print(f"Wrote {out_path}  ({len(names)} glyphs, {cols}x{rows})")
    return out_path


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Material Symbols glyph extractor / reference sheet")
    ap.add_argument("--grep", help="substring filter on glyph names")
    ap.add_argument("--names", help="comma-separated explicit glyph names")
    ap.add_argument("--sheet", action="store_true", help="render a labelled reference sheet of the selection")
    ap.add_argument("--out", default="/tmp/ms_glyphs.png", help="sheet output path")
    ap.add_argument("--count", action="store_true", help="print total glyph count and exit")
    ap.add_argument("--limit", type=int, default=400, help="max glyphs to list/render")
    args = ap.parse_args(argv)

    name2cp = font_name_to_codepoint()
    if args.count:
        print(f"{len(name2cp)} named glyphs in {os.path.basename(MS_ICON_FONT)}")
        return 0

    if args.names:
        sel = [n.strip() for n in args.names.split(",") if n.strip()]
    elif args.grep:
        sel = sorted(n for n in name2cp if args.grep.lower() in n.lower())
    else:
        sel = sorted(name2cp)
    sel = sel[:args.limit]

    if not args.sheet:
        for n in sel:
            print(f"  {n:<34} 0x{name2cp[n]:X}")
        print(f"\n{len(sel)} shown"
              + (f" (of {sum(1 for n in name2cp if args.grep.lower() in n.lower())} matches)" if args.grep else ""))
    else:
        render_reference_sheet(sel, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
