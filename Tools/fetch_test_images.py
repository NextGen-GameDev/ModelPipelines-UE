#!/usr/bin/env python
"""Reference test images (WP4.2, contract C31): Wikimedia Commons files under CC0 or a Public Domain mark only.

Default mode: reads Tools/data/test_images.json (pinned URL + sha256 + licence + author per image), downloads each URL
into <dev>/cache/test_images/<id>.<ext> (a cached file whose sha256 matches is not downloaded again), refuses any entry
whose licence is not CC0 / Public Domain, verifies the sha256 (mismatch -> error), and converts to an 8-bit RGB PNG at
<dev>/goldens/vision/reference_images/<id>.png (no EXIF transpose; thumbnail((1024, 1024), LANCZOS) only when a side
exceeds 1024). The PNG is the canonical input for Python and Unreal. Only the metadata JSON is committed, never pixels.

--discover: (re)builds the pools in the JSON. It queries the Commons API (search in the File namespace restricted to
CC-Zero / public-domain categories), reads the licence from `extmetadata`, keeps only whitelisted licences, downloads the
1024-px thumbnail from upload.wikimedia.org, records its sha256, and splits the files into the four disjoint pools.
Network is used only here and in the default mode's download step; tests never touch it.
"""
import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths  # noqa: E402

SCHEMA = "pipelines-test-images/1"
API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ModelPipelines-test-fetch/1.0 (https://github.com/NextGen-GameDev/ModelPipelines-UE)"
POOLS = ["cls_visible", "cls_hidden", "det_visible", "det_hidden"]
ID_PREFIX = {"cls_visible": "c", "cls_hidden": "ch", "det_visible": "d", "det_hidden": "dh"}
DATA_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_images.json")
MAX_SIDE = 1024
MAX_POOL = {"cls_visible": 45, "cls_hidden": 45, "det_visible": 45, "det_hidden": 45}
DELAY = 0.6  # seconds between requests

CLS_WORDS = ["cat", "dog", "horse", "cow", "sheep", "elephant", "zebra", "giraffe", "lion", "tiger", "bear", "butterfly",
             "beetle", "spider", "bird", "owl", "eagle", "parrot", "duck", "goldfish", "frog", "turtle", "snake", "lizard",
             "crab", "snail", "jellyfish", "starfish", "car", "truck", "bus", "train", "airplane", "ship", "sailboat",
             "bicycle", "motorcycle", "tractor", "fire engine", "pizza", "banana", "orange", "apple", "strawberry",
             "broccoli", "cheeseburger", "coffee", "bread", "cake", "ice cream", "guitar", "violin", "piano", "drum",
             "trumpet", "flute", "chair", "sofa", "lamp", "clock", "teapot", "mug", "bottle", "umbrella", "backpack",
             "hammer", "scissors", "screwdriver", "wrench", "keyboard", "telephone", "camera", "lighthouse", "bridge",
             "windmill", "castle", "mushroom", "sunflower", "volcano", "tent", "football", "soccer ball", "helmet"]
DET_WORDS = ["street people", "traffic cars", "city street bus", "bicycles street", "dog and person", "cat on sofa",
             "harbour boats", "living room chairs", "kitchen bottles", "market people", "park bench people",
             "parked cars", "railway station", "bus stop", "people walking", "cyclists", "marina boats", "cafe terrace",
             "dining table", "crowd", "horses and riders", "airport planes", "motorbikes", "pedestrians crosswalk",
             "street traffic lights", "beach people", "road truck", "bridge people", "sheep field", "playground", "bus", "tram", "ferry", "traffic jam", "festival", "skateboard", "train station platform", "farm", "tourists", "boats river", "street cafe", "road cars", "schoolchildren", "ships harbor", "town square", "sports game", "bench", "restaurant"]


def is_open_license(name):
    """True for CC0 and Public Domain marks only (LicenseShortName / License strings of the Commons extmetadata)."""
    s = (name or "").strip().lower()
    if not s:
        return False
    if s.startswith("cc0") or s.startswith("cc-zero") or s.startswith("cc zero"):
        return True
    if s.startswith("public domain") or s.startswith("pdm"):
        return True
    return s == "pd" or s.startswith("pd-") or s.startswith("pd ")


def strip_html(s):
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def http_get(url, tries=5):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            time.sleep(DELAY)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < tries:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt + 1 < tries:
                time.sleep(3 * (attempt + 1))
                continue
            raise


def api_get(params):
    q = dict(params, format="json", formatversion="2")
    return json.loads(http_get(API + "?" + urllib.parse.urlencode(q)).decode("utf-8"))


def validate_doc(doc):
    """Structural checks on the committed JSON; returns a list of problems (empty = fine)."""
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append("schema is not " + SCHEMA)
    seen = set()
    for pool in POOLS:
        for i, e in enumerate(doc.get("pools", {}).get(pool, [])):
            for k in ("id", "title", "url", "license", "author", "sha256"):
                if not e.get(k):
                    bad.append("%s[%d] has no %s" % (pool, i, k))
            if not is_open_license(e.get("license")):
                bad.append("%s %s: licence %r is not CC0 / Public Domain" % (pool, e.get("id"), e.get("license")))
            if e.get("id") in seen or e.get("url") in seen or e.get("title") in seen:
                bad.append("%s %s is in two pools or listed twice" % (pool, e.get("id")))
            seen.update([e.get("id"), e.get("url"), e.get("title")])
    return bad


def ext_of(url):
    path = urllib.parse.urlparse(url).path
    e = os.path.splitext(path)[1].lower().lstrip(".")
    return e if e in ("jpg", "jpeg", "png") else "bin"


def to_reference_png(src, dst):
    from PIL import Image
    with Image.open(src) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        im.save(dst, format="PNG")
        return im.size


def fetch_all(doc, dev, only_pools):
    bad = validate_doc(doc)
    if bad:
        for b in bad:
            print("REFUSED " + b)
        return 1
    cache = os.path.join(dev, "cache", "test_images")
    out = os.path.join(dev, "goldens", "vision", "reference_images")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    downloaded = reused = 0
    total = 0
    for pool in only_pools:
        for e in doc["pools"][pool]:
            total += 1
            path = os.path.join(cache, "%s.%s" % (e["id"], ext_of(e["url"])))
            if os.path.isfile(path) and sha256_file(path) == e["sha256"]:
                reused += 1
            else:
                data = http_get(e["url"])
                if sha256_bytes(data) != e["sha256"]:
                    print("ERROR sha256 mismatch for %s (%s): got %s, pinned %s" % (e["id"], e["url"], sha256_bytes(data),
                                                                               e["sha256"]))
                    return 1
                with open(path, "wb") as f:
                    f.write(data)
                downloaded += 1
            to_reference_png(path, os.path.join(out, e["id"] + ".png"))
    print("%d images verified (%d downloaded, %d from cache) -> %s" % (total, downloaded, reused, out))
    return 0


# ------------------------------------------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------------------------------------------

def search_candidates(word, category, limit=30):
    """Commons file-namespace search restricted to a category; returns raw page dicts with imageinfo."""
    r = api_get({"action": "query", "generator": "search", "gsrnamespace": "6", "gsrlimit": str(limit),
                 "gsrsearch": 'filetype:bitmap incategory:"%s" %s' % (category, word), "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size|mime", "iiurlwidth": str(MAX_SIDE),
                 "iiextmetadatafilter": "LicenseShortName|License|Artist|Credit"})
    pages = r.get("query", {}).get("pages", [])
    return sorted(pages, key=lambda p: p.get("index", 0))


def candidate_entry(page, min_width, min_height):
    ii = (page.get("imageinfo") or [None])[0]
    if not ii or ii.get("mime") not in ("image/jpeg", "image/png"):
        return None
    if ii.get("width", 0) < min_width or ii.get("height", 0) < min_height:
        return None
    meta = ii.get("extmetadata", {})
    lic = strip_html((meta.get("LicenseShortName") or {}).get("value", ""))
    if not is_open_license(lic):
        return None
    author = strip_html((meta.get("Artist") or {}).get("value", "")) or strip_html((meta.get("Credit") or {}).get("value", ""))
    url = ii.get("thumburl") or ii.get("url")
    url = url.split("?")[0] if url else url  # drop the utm tracking query
    if not author or not url or not url.startswith("https://upload.wikimedia.org/"):
        return None
    return {"title": page["title"], "url": url, "license": lic, "author": author[:200]}


def discover(args):
    kinds = args.kinds.split(",")
    categories = ["CC-Zero", "Public domain"]
    pools_raw = {"cls": [], "det": []}
    seen = set()
    if os.path.isfile(args.data):  # kinds that are kept must not be picked again by the other kind
        with open(args.data, "r", encoding="utf-8") as f:
            old = json.load(f).get("pools", {})
        for pool, entries in old.items():
            if pool.split("_")[0] not in kinds:
                seen.update(e["title"] for e in entries)
    for kind, words, per_word in (("cls", CLS_WORDS, 2), ("det", DET_WORDS, 5)):
        if kind not in kinds:
            continue
        for word in words:
            got = 0
            for cat in categories:
                if got >= per_word:
                    break
                try:
                    pages = search_candidates(word, cat)
                except Exception as ex:  # keep going on a transient API error
                    print("search %r %r failed: %s" % (word, cat, ex))
                    continue
                for p in pages:
                    if got >= per_word:
                        break
                    ent = candidate_entry(p, 640, 480 if kind == "det" else 400)
                    if ent is None or ent["title"] in seen:
                        continue
                    seen.add(ent["title"])
                    pools_raw[kind].append(dict(ent, word=word))
                    got += 1
            print("%s %-22s %d candidates (total %d)" % (kind, word, got, len(pools_raw[kind])))
    # shuffle (results come grouped by subject word), cap, split alternately into the visible / hidden pools, download
    # and hash. Kinds not rediscovered keep their pools from the existing JSON.
    doc = {"schema": SCHEMA, "pools": {p: [] for p in POOLS}}
    if os.path.isfile(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            doc["pools"].update(json.load(f).get("pools", {}))
    for kind, vis, hid in (("cls", "cls_visible", "cls_hidden"), ("det", "det_visible", "det_hidden")):
        if kind not in kinds:
            continue
        raw = pools_raw[kind]
        random.Random(4242).shuffle(raw)
        del raw[MAX_POOL[vis] + MAX_POOL[hid]:]
        new = {vis: [], hid: []}
        for i, ent in enumerate(raw):
            pool = vis if i % 2 == 0 else hid
            data = http_get(ent["url"])
            ent = {k: v for k, v in ent.items() if k != "word"}
            ent["sha256"] = sha256_bytes(data)
            ent["id"] = "%s%02d" % (ID_PREFIX[pool], len(new[pool]))
            new[pool].append(dict((k, ent[k]) for k in ("id", "title", "url", "license", "author", "sha256")))
        doc["pools"].update(new)
    with open(args.data, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print({p: len(doc["pools"][p]) for p in POOLS}, "->", args.data)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Fetch the pinned Wikimedia Commons CC0 / Public Domain test images "
                                             "(WP4.2, C31) and convert them to reference PNGs.")
    ap.add_argument("--data", default=DATA_JSON, help="pinned metadata JSON (default Tools/data/test_images.json)")
    ap.add_argument("--dev", default=_paths.dev_dir(), help="dev data folder (cache/ and goldens/ below it)")
    ap.add_argument("--pools", default=",".join(POOLS), help="comma-separated pools to fetch")
    ap.add_argument("--kinds", default="cls,det", help="with --discover: which pool kinds to rebuild (cls, det)")
    ap.add_argument("--discover", action="store_true",
                    help="rebuild the pools in --data from the Commons API (overwrites the pinned JSON)")
    args = ap.parse_args()
    if args.discover:
        return discover(args)
    with open(args.data, "r", encoding="utf-8") as f:
        doc = json.load(f)
    pools = [p for p in args.pools.split(",") if p]
    for p in pools:
        if p not in POOLS:
            raise SystemExit("unknown pool " + p)
    return fetch_all(doc, args.dev, pools)


if __name__ == "__main__":
    sys.exit(main())
