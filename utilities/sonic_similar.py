#!/usr/bin/env python3
"""
sonic_similar.py

Print the most sonically similar tracks for a given track ratingKey,
including the numeric 'distance' returned by Plex's /nearest endpoint.

Uses config.yml keys:
  plex:
    url: http://...
    token: ...
"""

import argparse
import os
import xml.etree.ElementTree as ET
import requests
import yaml
from typing import Optional

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    plex = cfg["plex"]
    baseurl = (plex.get("url") or plex.get("baseurl") or "").rstrip("/")
    token = plex["token"]
    if not baseurl:
        raise SystemExit("Config error: expected plex.url (or plex.baseurl)")
    return baseurl, token

def fetch_nearest_xml(baseurl: str, token: str, ratingkey: str, limit: int, maxdistance: Optional[float]):
    params = {"X-Plex-Token": token, "limit": str(limit)}
    if maxdistance is not None:
        params["maxDistance"] = str(maxdistance)

    url = f"{baseurl}/library/metadata/{ratingkey}/nearest"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def parse_tracks(xml_text: str):
    root = ET.fromstring(xml_text)
    items = []
    # Returned items are typically <Track ... distance="0.123" .../>
    for el in root.iter():
        if el.tag.lower().endswith("track"):
            rating_key = el.attrib.get("ratingKey")
            title = el.attrib.get("title")
            grandparent = el.attrib.get("grandparentTitle")  # artist (usually)
            parent = el.attrib.get("parentTitle")            # album (usually)
            dist = el.attrib.get("distance")
            items.append({
                "ratingKey": rating_key,
                "title": title,
                "artist": grandparent,
                "album": parent,
                "distance": float(dist) if dist is not None else None,
            })
    # If Plex already sorts by similarity, this will already be ordered,
    # but sorting by distance is safe.
    items.sort(key=lambda x: (x["distance"] is None, x["distance"]))
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml"))
    ap.add_argument("--ratingkey", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--maxdistance", type=float, default=None)
    args = ap.parse_args()

    baseurl, token = load_cfg(args.config)

    xml_text = fetch_nearest_xml(
        baseurl=baseurl,
        token=token,
        ratingkey=args.ratingkey,
        limit=args.limit,
        maxdistance=args.maxdistance,
    )

    tracks = parse_tracks(xml_text)

    print(f"Reference ratingKey: {args.ratingkey}")
    print(f"Returned: {len(tracks)} (limit={args.limit}, maxDistance={args.maxdistance})\n")

    if not tracks:
        print("No similar tracks returned.")
        return

    for i, t in enumerate(tracks, start=1):
        d = t["distance"]
        d_str = f"{d:.6f}" if isinstance(d, float) else "None"
        print(f"{i:>3}. distance={d_str}  {t['artist']} — {t['title']}  ({t['album']})  [rk={t['ratingKey']}]")

if __name__ == "__main__":
    main()
