"""Fine-tune YOLOE-seg on the unified task pool (YOLO segmentation format).

Uses Ultralytics' YOLOEPESegTrainer (linear-probing-style head fine-tune
with backbone/neck frozen except cv3). Best ckpt copied to
`outputs/checkpoints/yoloe/best.pt` so `src.predict_yoloe` auto-detects it.

**Prompt augmentation:** mirroring CLIPSeg, each train batch samples one
paraphrase per class from `cfg.tasks[*].prompts` and re-encodes the text PE
on the fly via an `on_train_batch_start` callback. This makes YOLOE
prompt-conditioned at train time rather than locked to a single class name.

Usage:
    python -m src.train_yoloe --config configs/yoloe.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

from .utils import load_config, set_seed


def train(cfg_path: str):
    cfg = load_config(cfg_path)
    if "tasks" not in cfg:
        raise ValueError("config must define `tasks`")
    set_seed(cfg.get("seed", 42))

    yolo_cfg = cfg.get("yoloe", {})
    train_cfg = cfg.get("yoloe_train", {})
    weights = yolo_cfg.get("weights", "yoloe-11l-seg.pt")
    data_yaml = train_cfg.get("data_yaml", "data/processed/yolo_unified/dataset.yaml")
    epochs = train_cfg.get("epochs", 30)
    batch_size = train_cfg.get("batch_size", 16)
    imgsz = train_cfg.get("imgsz", 640)
    ckpt_dir = Path(train_cfg.get("ckpt_dir", "outputs/checkpoints/yoloe"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    project = str(ckpt_dir.parent)
    name = ckpt_dir.name + "_run"

    if not Path(data_yaml).exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — run `python -m src.data.build_yolo_dataset` first")

    print(f"[yoloe-train] weights={weights} data={data_yaml} epochs={epochs} bs={batch_size} imgsz={imgsz}")
    model = YOLOE(weights)
    # Encode the dataset class names with text encoder so the head matches.
    with open(data_yaml) as f:
        ds_meta = yaml.safe_load(f)
    names = list(ds_meta["names"].values()) if isinstance(ds_meta["names"], dict) else list(ds_meta["names"])
    text_pe = model.get_text_pe(names)
    model.set_classes(names, text_pe)
    print(f"[yoloe-train] base classes={names}")

    # Per-batch prompt augmentation: cfg.tasks[i].prompts -> paraphrase pool
    # for class i. At each train batch, pick one paraphrase per class, re-
    # encode text PE, swap into model so the head sees a different prompt
    # phrasing every step. Mirrors CLIPSeg's prompt augmentation.
    prompt_pool = []
    for cls_name in names:
        match = next((t for t in cfg["tasks"] if t["name"] == cls_name), None)
        prompt_pool.append(match["prompts"] if match else [cls_name])
    use_prompt_aug = train_cfg.get("prompt_augmentation", True)
    rng = random.Random(cfg.get("seed", 42))

    if use_prompt_aug and any(len(p) > 1 for p in prompt_pool):
        print(f"[yoloe-train] prompt augmentation ON; pool sizes={[len(p) for p in prompt_pool]}")

        def _resample_prompts(trainer):
            new_names = [rng.choice(p) for p in prompt_pool]
            try:
                pe = trainer.model.get_text_pe(new_names)
                trainer.model.set_classes(new_names, pe)
            except Exception as e:
                print(f"[prompt-aug] skipped: {e}")

        model.add_callback("on_train_batch_start", _resample_prompts)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=0,
        project=project,
        name=name,
        trainer=YOLOEPESegTrainer,
        verbose=True,
        exist_ok=True,
    )

    # Restore the canonical class names so saved ckpt's PE matches dataset.yaml
    if use_prompt_aug:
        try:
            pe = model.get_text_pe(names)
            model.set_classes(names, pe)
        except Exception:
            pass

    # Copy best.pt out of ultralytics' run dir into our canonical path.
    # Ultralytics resolves project against its `runs_dir` setting, so the
    # actual save location may be prefixed with `~/runs/segment/...`. Trust
    # the trainer's `save_dir` when available; fall back to a pattern match.
    best_src = None
    try:
        best_src = Path(model.trainer.save_dir) / "weights" / "best.pt"
    except Exception:
        pass
    if not (best_src and best_src.exists()):
        candidates = [
            Path(project) / name / "weights" / "best.pt",
        ]
        from pathlib import Path as _P
        for runs_root in (_P.home() / "runs" / "segment",
                          _P("/scratch/users/rajarshi/runs/segment")):
            candidates.append(runs_root / project / name / "weights" / "best.pt")
            candidates.append(runs_root / name / "weights" / "best.pt")
        for c in candidates:
            if c.exists():
                best_src = c
                break
    if best_src and best_src.exists():
        shutil.copy2(best_src, ckpt_dir / "best.pt")
        print(f"[yoloe-train] copied {best_src} -> {ckpt_dir/'best.pt'}")
    else:
        print(f"[yoloe-train] WARN: best.pt not found (searched trainer.save_dir + fallback paths)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
