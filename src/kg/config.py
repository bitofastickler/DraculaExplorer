# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "configs" / "kg.yml"
    cfg = load_yaml(cfg_path)
    cfg["_path"] = str(cfg_path)
    return cfg
