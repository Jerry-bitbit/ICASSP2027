"""Prepare the exact Kodak images and PolyU filename groups used in the paper."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
from PIL import Image
from skimage.transform import resize

from experiments.paths import CONFIGS, DATA


def obtain(path: Path, url: str, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(f"Missing {path}; supply the file or add --download")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {path.name}", flush=True)
    with urlopen(url, timeout=60) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target)
    partial.replace(path)


def prepare_kodak(download: bool) -> None:
    for index in range(1, 25):
        name = f"kodim{index:02d}.png"
        raw = DATA / "raw" / "kodak24" / name
        obtain(raw, f"https://r0k.us/graphics/kodak/kodak/{name}", download)
        with Image.open(raw) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = array.shape[:2]
        scale = min(1.0, 256 / max(h, w))
        if scale < 1.0:
            array = resize(array, (round(h * scale), round(w * scale)), anti_aliasing=True)
        array = np.clip(np.asarray(array, np.float32), 0.0, 1.0)
        # The original pipeline saved resized 8-bit PNGs, then loaded them again.
        # Preserve this rounding before noise generation.
        out = DATA / "processed" / "kodak24" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(array * 255.0 + 0.5, 0, 255).astype(np.uint8)).save(out)
    print("Prepared 24 Kodak RGB images (maximum side 256).")


def prepare_polyu(download: bool) -> None:
    manifest = json.loads((CONFIGS / "icassp_polyu_budget_manifest.json").read_text())
    base = "https://raw.githubusercontent.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset/master/CroppedImages/"
    folder = DATA / "raw" / "PolyU-Real-World-Noisy-Images-Dataset-master" / "CroppedImages"
    for name in manifest["images_in_order"]:
        for suffix in ("real", "mean"):
            filename = f"{name}_{suffix}.JPG"
            path = folder / filename
            obtain(path, base + quote(filename), download)
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise ValueError(f"Expected a 512 x 512 crop: {path}")
    print("Prepared 29 PolyU pairs: 8 validation and 21 test filename groups.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("kodak", "polyu", "all"), default="all")
    parser.add_argument("--download", action="store_true", help="Download missing source images")
    parser.add_argument("--weights", action="store_true", help="Also obtain the color DRUNet weights")
    args = parser.parse_args()
    if args.dataset in ("kodak", "all"):
        prepare_kodak(args.download)
    if args.dataset in ("polyu", "all"):
        prepare_polyu(args.download)
    if args.weights:
        obtain(DATA / "models" / "drunet_color.pth",
               "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
               args.download)


if __name__ == "__main__":
    main()
