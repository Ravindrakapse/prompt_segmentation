"""Pull best ckpts from HF Hub into outputs/checkpoints/{clipseg,gd_finetuned,sam_finetuned,yoloe}/.

After this, scripts/predict_*.sh work without any local training.
"""
from __future__ import annotations

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, snapshot_download


HF_USER = "ravindrakapse"


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def fetch_clipseg():
    dst = Path("outputs/checkpoints/clipseg")
    _ensure_dir(dst)
    f = hf_hub_download(repo_id=f"{HF_USER}/drywall-clipseg", filename="best.pt")
    shutil.copy2(f, dst / "best.pt")
    print(f"[hf-download] clipseg -> {dst/'best.pt'}")


def fetch_grounded_sam():
    local = snapshot_download(repo_id=f"{HF_USER}/drywall-grounded-sam")
    for sub, dst_root in [("gd_finetuned", "outputs/checkpoints/gd_finetuned"),
                          ("sam_finetuned", "outputs/checkpoints/sam_finetuned")]:
        src = Path(local) / sub
        dst = Path(dst_root) / "best"
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"[hf-download] grounded-sam {sub} -> {dst}")


def fetch_yoloe():
    dst = Path("outputs/checkpoints/yoloe")
    _ensure_dir(dst)
    f = hf_hub_download(repo_id=f"{HF_USER}/drywall-yoloe", filename="best.pt")
    shutil.copy2(f, dst / "best.pt")
    print(f"[hf-download] yoloe -> {dst/'best.pt'}")


def main():
    fetch_clipseg()
    fetch_grounded_sam()
    fetch_yoloe()
    print("[hf-download] all done.")


if __name__ == "__main__":
    main()
