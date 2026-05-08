"""Utility helpers: seed, config loading, logging, mask IO."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def slugify_prompt(prompt: str) -> str:
    s = prompt.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
