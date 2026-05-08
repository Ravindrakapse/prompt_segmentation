"""Scrape a Roboflow project via undocumented /search + /images/<id> endpoints.

Use when project has no published version (so SDK download() fails).

Output layout (compatible with src.data.dataset.PromptedSegDataset):
    out_root/
        images/{train,val,test}/<image_id>__<orig_name>
        masks/{train,val,test}/<image_id>__<orig_name_stem>.png   # binary 0/255
        manifest.json

Usage:
    python -m src.data.scrape_roboflow \
        --workspace fyp-ny1jt --project cracks-3ii36 \
        --out data/processed/crack \
        --api-key $ROBOFLOW_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests

API_BASE = "https://api.roboflow.com"
PAGE_LIMIT = 250  # server cap
SPLIT_REMAP = {"valid": "val", "validation": "val"}


def _post_with_retries(sess, url, params, json_body, timeout=120, retries=6):
    delay = 2.0
    last = None
    for attempt in range(retries):
        try:
            r = sess.post(url, params=params, json=json_body, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            print(f"[search] attempt {attempt+1}/{retries} failed: {e}; sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"POST {url} failed after {retries} retries: {last}")


def list_image_ids(workspace: str, project: str, api_key: str,
                   cache_path: Path | None = None) -> list[dict]:
    """Page through search endpoint. Returns list of {id, split} dicts.
    Caches result to `cache_path` so reruns after a partial scrape don't have
    to re-hit the (often flaky) /search endpoint."""
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[search] loaded {len(cached)} cached ids from {cache_path}")
        return cached
    sess = requests.Session()
    out: list[dict] = []
    offset = 0
    total = None
    while True:
        r = _post_with_retries(
            sess,
            f"{API_BASE}/{workspace}/{project}/search",
            params={"api_key": api_key},
            json_body={"limit": PAGE_LIMIT, "offset": offset},
        )
        d = r.json()
        if total is None:
            total = d.get("total", 0)
            print(f"[search] total={total}")
        results = d.get("results", [])
        if not results:
            break
        for it in results:
            split = SPLIT_REMAP.get(it.get("split", "train"), it.get("split", "train"))
            out.append({"id": it["id"], "split": split})
        offset += len(results)
        if offset >= total:
            break
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(out, f)
        print(f"[search] cached {len(out)} ids to {cache_path}")
    return out


def fetch_image_record(workspace: str, project: str, image_id: str, api_key: str) -> dict:
    r = requests.get(
        f"{API_BASE}/{workspace}/{project}/images/{image_id}",
        params={"api_key": api_key},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("image", {})


def download_bytes(url: str, retries: int = 3) -> bytes:
    last = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"download failed: {last}")


def polygon_mask(boxes: list, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes or []:
        pts = b.get("points")
        if pts and len(pts) >= 3:
            poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
        else:
            x, y, bw, bh = b.get("x", 0), b.get("y", 0), b.get("width", 0), b.get("height", 0)
            x0 = int(round(x - bw / 2))
            y0 = int(round(y - bh / 2))
            x1 = int(round(x + bw / 2))
            y1 = int(round(y + bh / 2))
            mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 255
    return mask


def process_one(workspace: str, project: str, item: dict, api_key: str, out_root: Path,
                skip_existing: bool = True) -> dict:
    image_id = item["id"]
    split = item["split"]
    if skip_existing:
        img_dir = out_root / "images" / split
        msk_dir = out_root / "masks" / split
        if img_dir.exists() and msk_dir.exists():
            existing_img = list(img_dir.glob(f"{image_id}__*"))
            existing_msk = list(msk_dir.glob(f"{image_id}__*"))
            if existing_img and existing_msk:
                return {"id": image_id, "ok": True, "split": split, "skipped": True,
                        "img": str(existing_img[0]), "mask": str(existing_msk[0])}
    rec = fetch_image_record(workspace, project, image_id, api_key)
    name = rec.get("name", f"{image_id}.jpg")
    ann = rec.get("annotation", {}) or {}
    h = int(ann.get("height") or 0)
    w = int(ann.get("width") or 0)
    url = (rec.get("urls") or {}).get("original")
    if not url:
        return {"id": image_id, "ok": False, "err": "no original url"}

    img_bytes = download_bytes(url)
    img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_arr is None:
        return {"id": image_id, "ok": False, "err": "decode fail"}
    if not h or not w:
        h, w = img_arr.shape[:2]

    stem = Path(name).stem
    img_dir = out_root / "images" / split
    msk_dir = out_root / "masks" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"{image_id}__{name}"
    msk_path = msk_dir / f"{image_id}__{stem}.png"
    cv2.imwrite(str(img_path), img_arr)
    mask = polygon_mask(ann.get("boxes", []), h, w)
    cv2.imwrite(str(msk_path), mask)
    return {"id": image_id, "ok": True, "split": split, "name": name,
            "img": str(img_path), "mask": str(msk_path),
            "n_polys": len(ann.get("boxes", []) or [])}


def _rebalance_val_test(out_root: Path, seed: int = 42) -> None:
    """Roboflow projects often dump nearly everything into train+val with a
    near-empty test split. Deterministically rebalance val:test to 50:50."""
    import random as _rnd
    val_img = out_root / "images" / "val"
    test_img = out_root / "images" / "test"
    val_msk = out_root / "masks" / "val"
    test_msk = out_root / "masks" / "test"
    if not val_img.exists():
        return
    val_files = sorted([p.name for p in val_img.iterdir() if p.is_file()])
    test_files = sorted([p.name for p in test_img.iterdir() if p.is_file()]) if test_img.exists() else []
    total = len(val_files) + len(test_files)
    if total == 0 or len(test_files) >= total // 2:
        return
    need = (total // 2) - len(test_files)
    rng = _rnd.Random(seed)
    rng.shuffle(val_files)
    test_img.mkdir(parents=True, exist_ok=True)
    test_msk.mkdir(parents=True, exist_ok=True)
    for fname in val_files[:need]:
        stem = Path(fname).stem
        (val_img / fname).rename(test_img / fname)
        msk = val_msk / f"{stem}.png"
        if msk.exists():
            msk.rename(test_msk / f"{stem}.png")
    print(f"[rebalance] moved {need} samples val -> test (seed {seed})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--rebalance", action="store_true",
                    help="after scrape, deterministically rebalance val/test 50:50 (seed 42)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("set ROBOFLOW_API_KEY or pass --api-key")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[scrape] {args.workspace}/{args.project} -> {out_root}")
    cache = out_root / "_search_ids.json"
    items = list_image_ids(args.workspace, args.project, args.api_key, cache_path=cache)
    print(f"[scrape] listed {len(items)} ids")
    if args.limit:
        items = items[: args.limit]

    ok = 0
    errs = 0
    manifest = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, args.workspace, args.project, it, args.api_key, out_root): it for it in items}
        for i, f in enumerate(as_completed(futs)):
            it = futs[f]
            try:
                r = f.result()
            except Exception as e:
                r = {"id": it["id"], "split": it["split"], "ok": False, "err": str(e)[:200]}
            manifest.append(r)
            if r["ok"]:
                ok += 1
            else:
                errs += 1
            if (i + 1) % 100 == 0 or (i + 1) == len(items):
                rate = (i + 1) / max(1e-3, time.time() - t0)
                print(f"[scrape] {i+1}/{len(items)} ok={ok} err={errs} rate={rate:.1f}/s")

    if args.rebalance:
        _rebalance_val_test(out_root, seed=42)

    counts = {}
    for split in ("train", "val", "test"):
        d = out_root / "images" / split
        counts[split] = sum(1 for _ in d.iterdir()) if d.exists() else 0
    summary = {
        "workspace": args.workspace,
        "project": args.project,
        "total": len(items),
        "ok": ok,
        "errs": errs,
        "counts": counts,
    }
    with open(out_root / "manifest.json", "w") as f:
        json.dump({"summary": summary, "items": manifest}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
