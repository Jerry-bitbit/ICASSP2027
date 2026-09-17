from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paths import DATA, OUTPUT, RESULTS, CONFIGS

import experiments.run_icassp_budget_audit as common  # noqa: E402
from experiments.run_drunet_subset import denoise_drunet, load_drunet, metric_row, read_image  # noqa: E402


POLYU_ROOT = DATA / "raw" / "PolyU-Real-World-Noisy-Images-Dataset-master" / "CroppedImages"
NOISE_LEVELS255 = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0)
VAL_INDICES = (0, 4, 8, 12, 16, 20, 25, 28)


def scene_key(name: str) -> str:
    return re.sub(r"_+\d+$", "", name)


def deduplicate_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for case in cases:
        key = scene_key(str(case["image"]))
        if key not in seen:
            seen.add(key)
            output.append(case)
    return output


def independent_scene_names() -> list[str]:
    """Historical name retained; the split uses filename groups, not verified scenes."""
    manifest = json.loads((CONFIGS / "icassp_polyu_budget_manifest.json").read_text())
    return manifest["images_in_order"]


def image_pairs() -> list[dict[str, object]]:
    pairs = []
    for name in independent_scene_names():
        noisy_path = POLYU_ROOT / f"{name}_real.JPG"
        clean_path = POLYU_ROOT / f"{name}_mean.JPG"
        if not noisy_path.exists() or not clean_path.exists():
            raise FileNotFoundError(f"Missing PolyU pair for {name}")
        pairs.append({"image": name, "noisy": read_image(noisy_path), "clean": read_image(clean_path)})
    return pairs


def load_or_make_cases(cache_path: Path, regenerate: bool) -> tuple[list[dict[str, object]], float, dict[str, float]]:
    if cache_path.exists() and not regenerate:
        cached = np.load(cache_path, allow_pickle=False)
        cases = []
        for name in cached["names"].tolist():
            cases.append(
                {
                    "image": str(name),
                    "clean": cached[f"{name}_clean"],
                    "noisy": cached[f"{name}_noisy"],
                    "proposal": cached[f"{name}_proposal"],
                    "proposal_runtime": float(cached[f"{name}_runtime"]),
                }
            )
        scores = json.loads(str(cached["validation_scores_json"]))
        return cases, float(cached["selected_sigma255"]), scores

    pairs = image_pairs()
    val_names = json.loads((CONFIGS / "icassp_polyu_budget_manifest.json").read_text())["validation_images"]
    validation = [case for case in pairs if case["image"] in val_names]
    model, torch, torch_f = load_drunet()
    proposal_by_sigma: dict[float, dict[str, np.ndarray]] = {}
    validation_scores: dict[str, float] = {}
    for sigma255 in NOISE_LEVELS255:
        proposals: dict[str, np.ndarray] = {}
        psnrs = []
        for case in validation:
            proposal = denoise_drunet(model, torch, torch_f, np.asarray(case["noisy"]), sigma255 / 255.0)
            proposals[str(case["image"])] = proposal
            psnrs.append(metric_row(np.asarray(case["clean"]), np.asarray(case["noisy"]), proposal)["psnr"])
        proposal_by_sigma[sigma255] = proposals
        validation_scores[str(sigma255)] = float(np.mean(psnrs))
        print(f"[validation] sigma={sigma255:.1f}: {validation_scores[str(sigma255)]:.3f} dB", flush=True)
    selected_sigma255 = float(max(NOISE_LEVELS255, key=lambda value: validation_scores[str(value)]))
    print(f"[validation] selected sigma={selected_sigma255:.1f}/255", flush=True)

    cases: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {
        "names": np.asarray([str(case["image"]) for case in pairs]),
        "selected_sigma255": np.asarray(selected_sigma255),
        "validation_scores_json": np.asarray(json.dumps(validation_scores)),
    }
    cached_validation = proposal_by_sigma[selected_sigma255]
    for index, case in enumerate(pairs, 1):
        name = str(case["image"])
        start = time.perf_counter()
        if name in cached_validation:
            proposal = cached_validation[name]
            proposal_runtime = np.nan
        else:
            proposal = denoise_drunet(model, torch, torch_f, np.asarray(case["noisy"]), selected_sigma255 / 255.0)
            proposal_runtime = time.perf_counter() - start
        print(f"[proposal] {index:02d}/29 {name}", flush=True)
        output_case = {**case, "proposal": proposal, "proposal_runtime": proposal_runtime}
        cases.append(output_case)
        payload[f"{name}_clean"] = np.asarray(case["clean"])
        payload[f"{name}_noisy"] = np.asarray(case["noisy"])
        payload[f"{name}_proposal"] = proposal
        payload[f"{name}_runtime"] = np.asarray(proposal_runtime)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return cases, selected_sigma255, validation_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    results = RESULTS
    generated = OUTPUT / "generated"
    figures = OUTPUT / "figures"
    for directory in (results, generated, figures):
        directory.mkdir(parents=True, exist_ok=True)

    cache = DATA / "cache" / "icassp_polyu_drunet.npz"
    cases, selected_sigma255, validation_scores = load_or_make_cases(cache, args.regenerate)
    cases = deduplicate_cases(cases)
    names = [str(case["image"]) for case in cases]
    manifest = json.loads((CONFIGS / "icassp_polyu_budget_manifest.json").read_text())
    common.VAL_IMAGES = tuple(manifest["validation_images"])
    common.TEST_IMAGES = tuple(manifest["test_images"])
    frame, summary, metadata = common.evaluate(cases)
    metadata.update(
        {
            "dataset": "PolyU CroppedImages, one 512x512 crop per filename group",
            "selected_drunet_sigma255": selected_sigma255,
            "drunet_sigma_candidates255": list(NOISE_LEVELS255),
            "drunet_validation_psnr": validation_scores,
            "validation_images": list(common.VAL_IMAGES),
            "test_images": list(common.TEST_IMAGES),
        }
    )
    frame.to_csv(results / "icassp_polyu_budget_audit.csv", index=False)
    summary.to_csv(results / "icassp_polyu_budget_summary.csv", index=False)
    (results / "icassp_polyu_budget_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    common.write_table(summary, generated / "tab_icassp_polyu_budget_audit.tex")
    common.make_figure(summary, figures / "fig_icassp_polyu_budget_tradeoff.pdf")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
