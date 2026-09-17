"""Portable paths shared by the reproduction commands."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TV_DATA_DIR", ROOT / "data")).expanduser().resolve()
OUTPUT = Path(os.environ.get("TV_OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
RESULTS = OUTPUT / "results"
CONFIGS = ROOT / "configs"
