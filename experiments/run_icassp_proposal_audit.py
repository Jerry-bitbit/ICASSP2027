from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.restoration import denoise_nl_means, denoise_wavelet

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paths import DATA, OUTPUT, RESULTS, CONFIGS

import experiments.run_icassp_budget_audit as common  # noqa: E402


def base_cases() -> list[dict[str, object]]:
    cached = np.load(DATA / "cache" / "icassp_drunet_sigma25.npz", allow_pickle=False)
    cases = []
    for name in cached["names"].tolist():
        cases.append(
            {
                "image": str(name),
                "clean": cached[f"{name}_clean"],
                "noisy": cached[f"{name}_noisy"],
                "drunet": cached[f"{name}_proposal"],
            }
        )
    return cases


def proposal(image: np.ndarray, method: str) -> np.ndarray:
    sigma = 25.0 / 255.0
    if method == "wavelet":
        output = denoise_wavelet(
            image,
            sigma=sigma,
            wavelet="db1",
            wavelet_levels=4,
            method="BayesShrink",
            mode="soft",
            channel_axis=-1,
            rescale_sigma=True,
        )
    elif method == "nlm":
        output = denoise_nl_means(
            image,
            h=0.85 * sigma,
            fast_mode=True,
            sigma=0.0,
            patch_size=5,
            patch_distance=6,
            channel_axis=-1,
        )
    else:
        raise ValueError(method)
    return np.clip(np.asarray(output, dtype=np.float32), 0.0, 1.0)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = base_cases()
    output_rows = []
    for method in ("drunet", "wavelet", "nlm"):
        cases = []
        for index, case in enumerate(base, 1):
            candidate = np.asarray(case["drunet"]) if method == "drunet" else proposal(np.asarray(case["noisy"]), method)
            cases.append({**case, "proposal": candidate, "proposal_runtime": np.nan})
            print(f"[{method}] {index:02d}/24", flush=True)
        frame, summary, _ = common.evaluate(cases)
        frame.to_csv(RESULTS / f"{method}_per_image.csv", index=False)
        summary.to_csv(RESULTS / f"{method}_summary.csv", index=False)
        test_endpoint = frame[(frame.split == "test") & (frame.method == "proposal") & frame.beta.isna()]
        for beta in (0.20, 0.60):
            row = summary[np.isclose(summary.beta, beta)].iloc[0]
            output_rows.append(
                {
                    "proposal": method,
                    "beta": beta,
                    "proposal_psnr": float(test_endpoint.psnr.mean()),
                    "adaptive_eta_median": row.adaptive_eta_median,
                    "adaptive_psnr": row.adaptive_psnr,
                    "delta_psnr": row.delta_psnr,
                    "ci_low": row.delta_psnr_ci_low,
                    "ci_high": row.delta_psnr_ci_high,
                    "fixed_violation_percent": row.fixed_violation_percent,
                }
            )
    output = pd.DataFrame(output_rows)
    output.to_csv(RESULTS / "icassp_proposal_summary.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
