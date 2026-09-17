from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paths import DATA, OUTPUT, RESULTS, CONFIGS

import experiments.run_icassp_budget_audit as common  # noqa: E402
from experiments.run_drunet_subset import add_gaussian, denoise_drunet, load_drunet, read_image  # noqa: E402


def cases_for_sigma(sigma255: float, model, torch, torch_f) -> list[dict[str, object]]:
    cache_path = DATA / "cache" / f"icassp_drunet_sigma{int(sigma255)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return [
            {
                "image": str(name),
                "clean": cached[f"{name}_clean"],
                "noisy": cached[f"{name}_noisy"],
                "proposal": cached[f"{name}_proposal"],
                "proposal_runtime": float(cached[f"{name}_runtime"]),
            }
            for name in cached["names"].tolist()
        ]

    names = common.VAL_IMAGES + common.TEST_IMAGES
    root = DATA / "processed" / "kodak24"
    cases: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {"names": np.asarray(names)}
    for index, name in enumerate(names, 1):
        clean = read_image(root / f"{name}.png")
        noisy = add_gaussian(clean, sigma255, common.SEED)
        start = time.perf_counter()
        proposal = denoise_drunet(model, torch, torch_f, noisy, sigma255 / 255.0)
        runtime = time.perf_counter() - start
        print(f"[sigma {sigma255:.0f}] {index:02d}/24 {name}", flush=True)
        cases.append({"image": name, "clean": clean, "noisy": noisy, "proposal": proposal, "proposal_runtime": runtime})
        payload[f"{name}_clean"] = clean
        payload[f"{name}_noisy"] = noisy
        payload[f"{name}_proposal"] = proposal
        payload[f"{name}_runtime"] = np.asarray(runtime)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return cases


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_paths = [DATA / "cache" / f"icassp_drunet_sigma{sigma}.npz" for sigma in (15, 50)]
    if all(path.exists() for path in cache_paths):
        model = torch = torch_f = None
    else:
        model, torch, torch_f = load_drunet()
    for sigma255 in (15.0, 50.0):
        cases = cases_for_sigma(sigma255, model, torch, torch_f)
        frame, summary, metadata = common.evaluate(cases)
        tag = f"sigma{int(sigma255)}"
        frame.to_csv(RESULTS / f"icassp_kodak_{tag}_audit.csv", index=False)
        summary.to_csv(RESULTS / f"icassp_kodak_{tag}_summary.csv", index=False)
        print(f"\nKodak {tag}\n{summary.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
