"""Draw the SigLIP 2 tutorial's fresh test from new iNaturalist photographs (docs/siglip2-preregistration.md §4).

Run once by the maintainer; its log (`docs/siglip2-fresh-test-draw.json`) and the printed `FRESH_TEST_RECORDS`
are committed with the pins. The notebook never runs this: it fetches the pinned files by id, size and SHA-256.

    python tools/draw_fresh_test.py --corpus-cache weights/inat-birds --cache work/fresh-draw \
        --log docs/siglip2-fresh-test-draw.json --records work/fresh_test_records.py

Metadata comes from the iNaturalist API v1 (generic User-Agent, >= 1 s between requests, no personal identifier),
photographs from the open-data bucket. The threshold of the near-duplicate screen is set on synthetic copies of the
pinned 360 before any candidate is downloaded; candidates are ordered by a seeded shuffle and never chosen by
looking at images or scores.
"""
# ruff: noqa: E501  -- a maintainer tool written at the 110-column fleet width, like samples.py

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from siglip2_pipeline.samples import (  # noqa: E402
    CORPUS_BASE_URL,
    SAMPLE_RECORDS,
    SPECIES,
    fetch_corpus,
)

API = "https://api.inaturalist.org/v1/"
USER_AGENT = {"User-Agent": "dimer-tutorial-fresh-test-draw/1.0"}
CREATED_D1, CREATED_D2 = "2025-03-01", "2026-10-17"
PER_SPECIES = 16
SEED = 20261018
PLACE_DEGREES = 0.01
CONFIRM_CORRELATION = 0.90
_last_request = [0.0]


def api_get(path: str, **query: Any) -> dict[str, Any]:
    wait = 1.1 - (time.monotonic() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    url = API + path + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=60) as response:  # noqa: S310
        _last_request[0] = time.monotonic()
        return json.load(response)


def fetch_bytes(url: str) -> bytes:
    time.sleep(0.2)
    with urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120) as response:  # noqa: S310
        return response.read()


def phash(image: Image.Image) -> int:
    """64-bit DCT perceptual hash of the 32x32 grayscale image (low 8x8 frequencies against their median)."""
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    n = 32
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    low = (basis @ pixels @ basis.T)[:8, :8].ravel()
    bits = low > np.median(low)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def correlation(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64).ravel()
    y = np.asarray(b.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64).ravel()
    return float(np.corrcoef(x, y)[0, 1])


def synthetic_copies(image: Image.Image) -> list[Image.Image]:
    w, h = image.size
    jpeg = io.BytesIO()
    image.save(jpeg, format="JPEG", quality=70)
    jpeg.seek(0)
    crop_w, crop_h = int(w * 0.95), int(h * 0.95)
    left, top = (w - crop_w) // 2, (h - crop_h) // 2
    return [
        image.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BILINEAR),
        Image.open(jpeg).convert("RGB"),
        ImageEnhance.Brightness(image).enhance(1.1),
        image.crop((left, top, left + crop_w, top + crop_h)),
    ]


def location(obs: dict[str, Any]) -> tuple[float, float] | None:
    raw = obs.get("location")
    if not raw:
        return None
    lat, lng = (float(v) for v in str(raw).split(","))
    return lat, lng


def same_place_day(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a["observed_on"] == b["observed_on"]
        and abs(a["lat"] - b["lat"]) <= PLACE_DEGREES
        and abs(a["lng"] - b["lng"]) <= PLACE_DEGREES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-cache", type=Path, default=Path("weights") / "inat-birds")
    parser.add_argument("--cache", type=Path, default=Path("work") / "fresh-draw")
    parser.add_argument("--log", type=Path, default=Path("docs") / "siglip2-fresh-test-draw.json")
    parser.add_argument("--records", type=Path, default=Path("work") / "fresh_test_records.py")
    args = parser.parse_args(argv)
    args.cache.mkdir(parents=True, exist_ok=True)
    log: dict[str, Any] = {
        "preregistration": "docs/siglip2-preregistration.md section 4 (commit 1b5f790)",
        "query": {"endpoint": API + "observations", "quality_grade": "research", "photo_license": "cc0", "photos": "true", "created_d1": CREATED_D1, "created_d2": CREATED_D2, "order_by": "id", "order": "asc", "per_page": 200, "user_agent": USER_AGENT["User-Agent"]},
        "seed": SEED,
        "per_species": PER_SPECIES,
    }

    # 1. The pinned 360: bytes (re-hashed), observers, and place/day from the API.
    corpus = fetch_corpus(cache_dir=args.corpus_cache)
    pinned_images = {rid: Image.open(io.BytesIO(data)).convert("RGB") for rid, data in corpus.items()}
    pinned_sha = {hashlib.sha256(data).hexdigest() for data in corpus.values()}
    pinned_observers = {r[4] for r in SAMPLE_RECORDS}
    obs_ids = [r[3] for r in SAMPLE_RECORDS]
    pinned_place: list[dict[str, Any]] = []
    for start in range(0, len(obs_ids), 100):
        chunk = obs_ids[start : start + 100]
        for obs in api_get("observations", id=",".join(map(str, chunk)), per_page=200)["results"]:
            loc = location(obs)
            if obs.get("observed_on") and loc:
                pinned_place.append({"observed_on": obs["observed_on"], "lat": loc[0], "lng": loc[1], "source": f"pinned obs {obs['id']}"})
    log["pinned"] = {"photographs": len(corpus), "observers": len(pinned_observers), "with_place_and_day": len(pinned_place)}

    # 2. Threshold on synthetic copies of the 360, before any candidate is downloaded.
    pinned_hash = {rid: phash(img) for rid, img in pinned_images.items()}
    copy_distances = [hamming(pinned_hash[rid], phash(c)) for rid, img in pinned_images.items() for c in synthetic_copies(img)]
    threshold = int(np.ceil(np.percentile(copy_distances, 99)))
    hashes = np.array(list(pinned_hash.values()), dtype=np.uint64)
    distinct = [hamming(int(hashes[i]), int(hashes[j])) for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    log["threshold"] = {
        "rule": "99th percentile of original-to-copy pHash distances over 4 synthetic copies of each of the 360, rounded up",
        "copies": ["50% downscale", "JPEG quality 70", "brightness x1.1", "95% centre crop"],
        "copy_distance_max": max(copy_distances),
        "copy_distance_p99": float(np.percentile(copy_distances, 99)),
        "T": threshold,
        "distinct_pairs": len(distinct),
        "distinct_pairs_within_T": int(sum(d <= threshold for d in distinct)),
        "false_positive_rate": round(sum(d <= threshold for d in distinct) / len(distinct), 6),
    }
    print({"threshold": log["threshold"]}, flush=True)

    # 3. Pools, seeded order, screens.
    selected: list[dict[str, Any]] = []
    used_observers: set[str] = set()
    species_log: dict[str, Any] = {}
    for index, (key, (scientific, _common)) in enumerate(SPECIES.items()):
        taxon = next(t for t in api_get("taxa", q=scientific, rank="species", per_page=10)["results"] if t["name"] == scientific)
        pool: list[dict[str, Any]] = []
        id_above = 0
        while True:
            page = api_get("observations", taxon_id=taxon["id"], quality_grade="research", photo_license="cc0", photos="true", created_d1=CREATED_D1, created_d2=CREATED_D2, order_by="id", order="asc", per_page=200, id_above=id_above)["results"]
            if not page:
                break
            pool.extend(page)
            id_above = page[-1]["id"]
        eligible = []
        reasons: dict[str, int] = {}
        for obs in pool:
            login = obs["user"]["login"]
            photo = next((p for p in obs.get("photos", []) if (p.get("license_code") or "").lower() == "cc0"), None)
            loc = location(obs)
            why = None
            if login in pinned_observers:
                why = "observer in the pinned 360"
            elif not obs.get("observed_on") or loc is None:
                why = "no observed_on or coordinates"
            elif photo is None:
                why = "no CC0 photograph"
            if why:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            eligible.append((obs, photo, loc))
        random.Random(SEED + index).shuffle(eligible)
        picked, exclusions = [], []
        for obs, photo, loc in eligible:
            if len(picked) == PER_SPECIES:
                break
            login = obs["user"]["login"]
            base = {"observation_id": obs["id"], "photo_id": photo["id"], "observer": login}
            if login in used_observers:
                exclusions.append({**base, "reason": "observer already selected"})
                continue
            here = {"observed_on": obs["observed_on"], "lat": loc[0], "lng": loc[1]}
            clash = next((p["source"] for p in pinned_place + [{**s["place"], "source": f"fresh {s['id']}"} for s in selected] if same_place_day(here, p)), None)
            if clash:
                exclusions.append({**base, "reason": f"same place and day as {clash}"})
                continue
            ext = Path(urllib.parse.urlparse(photo["url"]).path).suffix.lstrip(".") or "jpg"
            url = f"{CORPUS_BASE_URL}{photo['id']}/medium.{ext}"
            local = args.cache / f"{photo['id']}.{ext}"
            data = local.read_bytes() if local.is_file() else fetch_bytes(url)
            local.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest in pinned_sha:
                exclusions.append({**base, "reason": "exact SHA-256 copy of a pinned photograph"})
                continue
            image = Image.open(io.BytesIO(data))
            image.load()
            image = image.convert("RGB")
            h = phash(image)
            refs = [(f"pinned {rid}", pinned_hash[rid], pinned_images[rid]) for rid in pinned_hash] + [(f"fresh {s['id']}", s["phash"], s["image"]) for s in selected]
            hits = [(name, hamming(h, rh), correlation(image, rimg)) for name, rh, rimg in refs if hamming(h, rh) <= threshold]
            confirmed = [hit for hit in hits if hit[2] >= CONFIRM_CORRELATION]
            if confirmed:
                exclusions.append({**base, "reason": "confirmed near-duplicate", "hits": [{"of": n, "distance": d, "correlation": round(c, 3)} for n, d, c in confirmed]})
                continue
            rid = f"fresh-{key}-{len(picked):02d}"
            entry = {
                "id": rid, "label": key, "photo_id": photo["id"], "observation_id": obs["id"], "observer": login,
                "created_at": str(obs.get("created_at", ""))[:10], "observed_on": obs["observed_on"], "bytes": len(data),
                "sha256": digest, "ext": ext, "place": here, "phash": h, "image": image,
                "unconfirmed_hash_hits": [{"of": n, "distance": d, "correlation": round(c, 3)} for n, d, c in hits],
            }
            picked.append(entry)
            selected.append(entry)
            used_observers.add(login)
        species_log[key] = {
            "taxon_id": taxon["id"], "pool": len(pool), "eligible": len(eligible), "ineligible": reasons,
            "walked": len(picked) + len(exclusions), "excluded": exclusions, "selected": len(picked),
        }
        print({key: {k: v for k, v in species_log[key].items() if k != "excluded"}, "excluded": len(exclusions)}, flush=True)
    log["species"] = species_log
    counts = {key: sum(1 for s in selected if s["label"] == key) for key in SPECIES}
    balanced = min(counts.values())
    log["balanced_size_per_species"] = balanced
    public = [{k: v for k, v in s.items() if k not in ("image", "phash", "place")} for s in selected]
    log["selected"] = public
    log["manifest_sha256"] = hashlib.sha256(json.dumps([[s["id"], s["label"], s["photo_id"], s["observation_id"], s["observer"], s["created_at"], s["observed_on"], s["bytes"], s["sha256"], s["ext"]] for s in selected], separators=(",", ":")).encode("utf-8")).hexdigest()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    if balanced < PER_SPECIES:
        print(f"STOP: only {balanced} per species pass the screens ({counts}); no pins written (pre-registration §4).")
        return 2
    lines = ["FRESH_TEST_RECORDS: tuple[tuple[str, str, int, int, str, str, str, int, str, str], ...] = ("]
    for s in selected:
        lines.append(f"    ({s['id']!r}, {s['label']!r}, {s['photo_id']}, {s['observation_id']}, {s['observer']!r}, {s['created_at']!r}, {s['observed_on']!r}, {s['bytes']}, {s['sha256']!r}, {s['ext']!r}),")
    lines.append(")")
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.records.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print({"selected": len(selected), "manifest_sha256": log["manifest_sha256"], "records": str(args.records)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
