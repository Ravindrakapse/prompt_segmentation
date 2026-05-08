"""Download Roboflow datasets and convert to processed binary-mask layout.

- `crack`  -> ravindra-kapse/cracks-3ii36-fdqni v1 (republished copy of
              fyp-ny1jt/cracks-3ii36, which has 0 published versions).
- `taping` -> objectdetect-pu6rn/drywall-join-detect v2.

Usage:
    python -m src.data.download --task crack  --api-key $ROBOFLOW_API_KEY
    python -m src.data.download --task taping --api-key $ROBOFLOW_API_KEY

Produces data/processed/<task>/{images,masks}/{train,val,test}/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

DATASETS = {
    "crack": {
        # User-republished copy of fyp-ny1jt/cracks-3ii36 with a published v1
        # so the SDK can reach it. Original project has 0 published versions.
        "workspace": "ravindra-kapse",
        "project": "cracks-3ii36-fdqni",
        "version": 1,
        "format": "coco",
    },
    "taping": {
        "workspace": "objectdetect-pu6rn",
        "project": "drywall-join-detect",
        "version": 2,
        "format": "coco-segmentation",
    },
}


def _ensure_dirs(out_dir: Path):
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "masks" / split).mkdir(parents=True, exist_ok=True)


def _coco_to_masks(coco_dir: Path, out_dir: Path, split_name: str):
    """Convert one Roboflow COCO split (images/ + _annotations.coco.json) to masks."""
    from pycocotools import mask as cocomask
    from pycocotools.coco import COCO

    ann_file = coco_dir / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"[warn] missing {ann_file}; skipping")
        return 0
    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    n = 0
    for iid in img_ids:
        info = coco.loadImgs([iid])[0]
        fname = info["file_name"]
        src = coco_dir / fname
        if not src.exists():
            continue
        H, W = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid]))
        full_mask = np.zeros((H, W), dtype=np.uint8)
        for a in anns:
            seg = a.get("segmentation")
            if not seg:
                # fall back to bbox if no polygon
                x, y, w, h = a["bbox"]
                x0, y0 = int(round(x)), int(round(y))
                x1, y1 = int(round(x + w)), int(round(y + h))
                full_mask[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = 255
                continue
            if isinstance(seg, list):
                rle = cocomask.frPyObjects(seg, H, W)
                m = cocomask.decode(rle)
                if m.ndim == 3:
                    m = m.any(axis=2)
            else:
                m = cocomask.decode(seg)
            full_mask[m.astype(bool)] = 255
        # write image (copy) + mask
        img_id = Path(fname).stem
        ext = Path(fname).suffix.lower()
        dst_img = out_dir / "images" / split_name / f"{img_id}{ext}"
        dst_msk = out_dir / "masks" / split_name / f"{img_id}.png"
        shutil.copy2(src, dst_img)
        cv2.imwrite(str(dst_msk), full_mask)
        n += 1
    return n


def download_roboflow(task: str, api_key: str, out_dir: Path, version: int | None = None):
    from roboflow import Roboflow
    meta = DATASETS[task]
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(meta["workspace"]).project(meta["project"])
    if version is None:
        try:
            versions = proj.versions()
            if not versions:
                raise RuntimeError("no versions returned")
            ids = []
            for v in versions:
                vid = getattr(v, "version", None) or getattr(v, "id", None)
                if isinstance(vid, str) and "/" in vid:
                    vid = vid.rsplit("/", 1)[-1]
                try:
                    ids.append(int(vid))
                except (TypeError, ValueError):
                    pass
            v = max(ids) if ids else meta["version"]
            print(f"[roboflow] using latest version v{v}")
        except Exception as e:
            print(f"[roboflow] versions() failed ({e}); falling back to v{meta['version']}")
            v = meta["version"]
    else:
        v = version
    fmts = [meta["format"], "coco", "yolov8"]
    last_err = None
    for fmt in fmts:
        try:
            ds = proj.version(v).download(fmt)
            print(f"[roboflow] downloaded format={fmt} -> {ds.location}")
            return Path(ds.location)
        except Exception as e:
            last_err = e
            print(f"[roboflow] format={fmt} failed: {e}")
    raise RuntimeError(f"all formats failed: {last_err}")


def _resplit_80_10_10(out_dir: Path, seed: int = 42,
                      ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> int:
    """Pool all (image, mask) pairs across train/val/test and deterministically
    re-split into 80/10/10 (or custom ratios). Roboflow ships wildly skewed
    splits (e.g. crack test=4, val=200; taping val=test=101) — this normalizes
    so each task has the same held-out fractions."""
    import random as _rnd
    splits = ("train", "val", "test")
    img_dirs = {sp: out_dir / "images" / sp for sp in splits}
    msk_dirs = {sp: out_dir / "masks" / sp for sp in splits}
    for d in list(img_dirs.values()) + list(msk_dirs.values()):
        d.mkdir(parents=True, exist_ok=True)

    pool = []  # (origin_split, fname)
    for sp in splits:
        if img_dirs[sp].exists():
            for p in img_dirs[sp].iterdir():
                if p.is_file():
                    pool.append((sp, p.name))
    total = len(pool)
    if total == 0:
        return 0

    rng = _rnd.Random(seed)
    pool.sort(key=lambda x: x[1])     # deterministic ordering by filename
    rng.shuffle(pool)
    n_train = int(round(total * ratios[0]))
    n_val = int(round(total * ratios[1]))
    n_test = total - n_train - n_val
    assignment = {}
    for i, (_origin, fname) in enumerate(pool):
        if i < n_train:
            assignment[fname] = "train"
        elif i < n_train + n_val:
            assignment[fname] = "val"
        else:
            assignment[fname] = "test"

    moved = 0
    for origin, fname in pool:
        target = assignment[fname]
        if origin == target:
            continue
        stem = Path(fname).stem
        msk_name = f"{stem}.png"
        shutil.move(str(img_dirs[origin] / fname), str(img_dirs[target] / fname))
        msk = msk_dirs[origin] / msk_name
        if msk.exists():
            shutil.move(str(msk), str(msk_dirs[target] / msk_name))
        moved += 1
    print(f"[resplit] {ratios} -> train={n_train} val={n_val} test={n_test} "
          f"(moved {moved}, total {total}, seed {seed})")
    return moved


# Backwards-compatible alias (older code may still call this name).
_rebalance_val_test = _resplit_80_10_10


def _count_splits(out_dir: Path) -> dict:
    return {sp: sum(1 for _ in (out_dir / "images" / sp).iterdir())
            for sp in ("train", "val", "test")
            if (out_dir / "images" / sp).exists()}


def process(task: str, api_key: str, out_root: Path, version: int | None = None,
            rebalance_val_test: bool = True, force: bool = False):
    out_dir = out_root / task
    _ensure_dirs(out_dir)
    train_dir = out_dir / "images" / "train"

    already_populated = train_dir.exists() and any(train_dir.iterdir())
    if force or not already_populated:
        raw_dir = download_roboflow(task, api_key, out_root, version=version)
        for split, alt in (("train", "train"), ("valid", "val"), ("test", "test")):
            coco_split = raw_dir / split
            if not coco_split.exists():
                continue
            _coco_to_masks(coco_split, out_dir, alt)
        raw_dir_str = str(raw_dir)
    else:
        print(f"[skip-download] {out_dir} already populated; pass --force to redownload")
        raw_dir_str = "(existing)"

    if rebalance_val_test and sum(_count_splits(out_dir).values()) > 0:
        _resplit_80_10_10(out_dir, seed=42)

    counts = _count_splits(out_dir)
    summary = {"task": task, "raw_dir": raw_dir_str, "counts": counts}
    with open(out_dir / "splits.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(DATASETS), required=True)
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--out-root", default="data/processed")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="redownload even if target dir already populated")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("ROBOFLOW_API_KEY not set; pass --api-key or export env var")
    process(args.task, args.api_key, Path(args.out_root), version=args.version, force=args.force)


if __name__ == "__main__":
    main()
