#!/usr/bin/env python3
"""
Build a face-clustering test dataset from Wikimedia Commons.

Bootstrap data for the pipeline (§10 of docs/architecture.html) before a real
wedding folder exists. Cricket players are the default subject: many public,
freely-licensed photos per person, taken across different events, angles and
lighting -- closer to "same face, many different shots" than a single studio
headshot, though still not a substitute for real wedding photos when tuning
clustering thresholds later.

Usage:
    python3 scripts/build_test_dataset.py
    python3 scripts/build_test_dataset.py --players "Virat Kohli,MS Dhoni" --per-player 5
    python3 scripts/build_test_dataset.py --players-file players.txt --out test-data/cricket-players

Output:
    <out>/<player-slug>/000.jpg, 001.jpg, ...
    <out>/manifest.csv   -- player, file, source title, source url, license

Stdlib only, no dependencies. Talks to the public Wikimedia Commons API.
"""
import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Download rendered thumbnails at this width rather than multi-megabyte
# originals. Commons throttles original downloads aggressively; thumbnails come
# off the cache. 1600px still leaves faces far larger than the 112px ArcFace
# needs, so nothing this dataset can test is lost -- and per docs/ARCHITECTURE.md
# these cropped press photos could never test the resolution work anyway.
THUMB_WIDTH = 1600
USER_AGENT = "wedding-face-finder-test-dataset/1.0 (internal test data script, non-commercial testing)"

DEFAULT_PLAYERS = [
    "Virat Kohli", "Sachin Tendulkar", "MS Dhoni", "Rohit Sharma", "Ravindra Jadeja",
    "Jasprit Bumrah", "Rishabh Pant", "Hardik Pandya", "KL Rahul", "Shubman Gill",
    "Ben Stokes", "Joe Root", "Steve Smith", "Pat Cummins", "David Warner",
    "Kane Williamson", "Babar Azam", "Shaheen Afridi", "Smriti Mandhana", "Ellyse Perry",
]

SKIP_PATTERNS = re.compile(r"(logo|flag|signature|coa|crest|map|icon|stamp|graph|chart)", re.I)
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


def api_get(params, retries=5):
    """Query the Commons API, backing off when throttled.

    Commons rate-limits the API itself, not just file downloads. Without this
    retry a single 429 raised straight out of here and killed an entire
    multi-player run -- losing every player after the one that tripped it.
    """
    params = {**params, "format": "json"}
    url = COMMONS_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                # Honour Retry-After when Commons sends one.
                wait = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(wait)
                except (TypeError, ValueError):
                    wait = delay
                print(f"  API throttled ({e.code}), waiting {wait:.0f}s...")
                time.sleep(min(wait, 120))
                delay = min(delay * 2, 120)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise


def find_category(player, sleep):
    direct = f"Category:{player}"
    data = api_get({"action": "query", "titles": direct})
    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" not in page:
            return direct
    time.sleep(sleep)

    data = api_get({
        "action": "query", "list": "search", "srnamespace": 14,
        "srsearch": player, "srlimit": 1,
    })
    hits = data.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def category_image_titles(category_title, limit, sleep):
    titles = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": category_title, "cmtype": "file",
            "cmlimit": min(50, limit - len(titles)),
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            title = m["title"]
            if SKIP_PATTERNS.search(title):
                continue
            if Path(title).suffix.lower() not in ALLOWED_EXT:
                continue
            titles.append(title)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(sleep)
    return titles[:limit]


def fetch_image_info(titles, sleep):
    """Batch-fetch url/size/license for up to 50 file titles per call."""
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = api_get({
            "action": "query", "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            # Ask for a rendered thumbnail alongside the original. Commons
            # rate-limits (HTTP 429) full-size downloads from upload.wikimedia.org
            # hard enough that a 100-image run mostly fails; its own 429 body
            # tells you to use thumbnails instead. Thumbnails are cached and
            # served without throttling.
            "iiurlwidth": THUMB_WIDTH,
        })
        for page in data.get("query", {}).get("pages", {}).values():
            infos = page.get("imageinfo")
            if not infos:
                continue
            info = infos[0]
            license_short = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "unknown")
            out[page["title"]] = {
                "url": info["url"],
                "thumb_url": info.get("thumburl"),
                "thumb_width": info.get("thumbwidth", 0),
                "thumb_height": info.get("thumbheight", 0),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "license": license_short,
            }
        if i + 50 < len(titles):
            time.sleep(sleep)
    return out


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def download(url, dest, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            return
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def build_for_player(player, out_dir, per_player, min_side, sleep):
    print(f"[{player}] resolving Commons category...")
    category = find_category(player, sleep)
    if not category:
        print(f"[{player}] no Commons category found, skipping")
        return []
    time.sleep(sleep)

    titles = category_image_titles(category, per_player * 3, sleep)
    if not titles:
        print(f"[{player}] no usable files in {category}")
        return []

    infos = fetch_image_info(titles, sleep)

    player_dir = out_dir / slugify(player)
    player_dir.mkdir(parents=True, exist_ok=True)

    rows, kept = [], 0
    for title in titles:
        if kept >= per_player:
            break
        info = infos.get(title)
        if not info or info["width"] < min_side or info["height"] < min_side:
            continue
        # Prefer the cached thumbnail; fall back to the original only if
        # Commons did not render one (e.g. an unusual format).
        fetch_url = info.get("thumb_url") or info["url"]
        # Thumbnails are always rendered as .jpg regardless of the source format.
        ext = ".jpg" if info.get("thumb_url") else Path(title).suffix.lower()
        dest = player_dir / f"{kept:03d}{ext}"
        try:
            download(fetch_url, dest)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[{player}] failed to download {title}: {e}")
            continue
        rows.append({
            "player": player, "file": str(dest.relative_to(out_dir)),
            "source_title": title, "source_url": info["url"], "license": info["license"],
        })
        kept += 1
        time.sleep(sleep)
    print(f"[{player}] kept {kept} images")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", help="Comma-separated player names, overrides the default list")
    ap.add_argument("--players-file", help="Text file, one player name per line")
    ap.add_argument("--out", default="test-data/cricket-players", help="Output directory")
    ap.add_argument("--per-player", type=int, default=15, help="Max images to keep per player")
    ap.add_argument("--min-side", type=int, default=200, help="Skip images smaller than this on either side (px)")
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between API/download calls (politeness)")
    args = ap.parse_args()

    if args.players:
        players = [p.strip() for p in args.players.split(",") if p.strip()]
    elif args.players_file:
        players = [l.strip() for l in Path(args.players_file).read_text().splitlines() if l.strip()]
    else:
        players = DEFAULT_PLAYERS

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for player in players:
        # One player failing must not cost every player after them. A long run
        # is expensive to restart, and Commons throttling is unpredictable.
        try:
            all_rows.extend(
                build_for_player(player, out_dir, args.per_player, args.min_side, args.sleep)
            )
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[{player}] giving up on this player: {e}")
            continue

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["player", "file", "source_title", "source_url", "license"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} images across {len(players)} players.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
