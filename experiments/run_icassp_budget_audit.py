from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paths import DATA, OUTPUT, RESULTS, CONFIGS

from experiments.run_drunet_subset import (  # noqa: E402
    add_gaussian,
    denoise_drunet,
    load_drunet,
    metric_row,
    read_image,
)
from methods.ectv import certify_candidate, tv_energy_float64  # noqa: E402


BETAS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
EPS = 1e-8
SIGMA255 = 25.0
SEED = 0
VAL_IMAGES = tuple(f"kodim{i:02d}" for i in range(1, 9))
TEST_IMAGES = tuple(f"kodim{i:02d}" for i in range(9, 25))


def load_or_make_cases(cache_path: Path, regenerate: bool) -> list[dict[str, object]]:
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
        return cases

    model, torch, torch_f = load_drunet()
    cases: list[dict[str, object]] = []
    payload: dict[str, np.ndarray] = {"names": np.asarray(VAL_IMAGES + TEST_IMAGES)}
    image_root = DATA / "processed" / "kodak24"
    for index, name in enumerate(VAL_IMAGES + TEST_IMAGES, 1):
        clean = read_image(image_root / f"{name}.png")
        noisy = add_gaussian(clean, SIGMA255, SEED)
        start = time.perf_counter()
        proposal = denoise_drunet(model, torch, torch_f, noisy, SIGMA255 / 255.0)
        proposal_runtime = time.perf_counter() - start
        print(f"[proposal] {index:02d}/24 {name}: {proposal_runtime:.3f} s", flush=True)
        cases.append(
            {
                "image": name,
                "clean": clean,
                "noisy": noisy,
                "proposal": proposal,
                "proposal_runtime": proposal_runtime,
            }
        )
        payload[f"{name}_clean"] = clean
        payload[f"{name}_noisy"] = noisy
        payload[f"{name}_proposal"] = proposal
        payload[f"{name}_runtime"] = np.asarray(proposal_runtime)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return cases


def budget_for(noisy: np.ndarray, beta: float) -> float:
    return float(beta) * max(tv_energy_float64(noisy) - EPS, 0.0)


def removed_tv(noisy: np.ndarray, output: np.ndarray) -> float:
    return max(0.0, tv_energy_float64(noisy) - tv_energy_float64(output))


def blend(noisy: np.ndarray, proposal: np.ndarray, eta: float) -> np.ndarray:
    return (noisy + float(eta) * (proposal - noisy)).astype(np.float32)


def select_fixed_eta(validation: list[dict[str, object]], beta: float) -> tuple[float, float]:
    """Select a global blend on validation images only.

    Among a fixed 0.005 grid, choose the compliant value with highest validation
    PSNR. Compliance is required for every validation image.
    """
    best_eta = 0.0
    best_psnr = -np.inf
    for eta in np.linspace(0.0, 1.0, 201):
        psnrs = []
        feasible = True
        for case in validation:
            noisy = np.asarray(case["noisy"])
            proposal = np.asarray(case["proposal"])
            output = blend(noisy, proposal, float(eta))
            if removed_tv(noisy, output) > budget_for(noisy, beta) + 1e-12:
                feasible = False
                break
            mse = float(np.mean((np.asarray(case["clean"], dtype=np.float64) - output) ** 2))
            psnrs.append(float(-10.0 * np.log10(max(mse, 1e-15))))
        score = float(np.mean(psnrs)) if feasible else -np.inf
        if score > best_psnr:
            best_eta = float(eta)
            best_psnr = score
    return best_eta, best_psnr


def evaluate(cases: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    validation = [case for case in cases if case["image"] in VAL_IMAGES]
    test = [case for case in cases if case["image"] in TEST_IMAGES]
    fixed = {beta: select_fixed_eta(validation, beta)[0] for beta in BETAS}
    rows: list[dict[str, object]] = []

    for split, subset in (("validation", validation), ("test", test)):
        for case in subset:
            clean = np.asarray(case["clean"])
            noisy = np.asarray(case["noisy"])
            proposal = np.asarray(case["proposal"])
            for method, output, eta in (("noisy", noisy, 0.0), ("proposal", proposal, 1.0)):
                metrics = metric_row(clean, noisy, output)
                rows.append(
                    {
                        "split": split,
                        "image": case["image"],
                        "beta": np.nan,
                        "method": method,
                        "eta": eta,
                        "budget": np.nan,
                        "removed64": removed_tv(noisy, output),
                        "violation64": np.nan,
                        "wrapper_runtime": 0.0,
                        **metrics,
                    }
                )

            for beta in BETAS:
                budget = budget_for(noisy, beta)
                start = time.perf_counter()
                certified = certify_candidate(
                    noisy,
                    proposal,
                    sigma_est=SIGMA255 / 255.0,
                    budget_override=budget,
                    bisection_steps=28,
                    feasibility_tolerance=1e-12,
                )
                wrapper_runtime = time.perf_counter() - start
                adaptive_metrics = metric_row(clean, noisy, certified.image)
                adaptive_removed = removed_tv(noisy, certified.image)
                rows.append(
                    {
                        "split": split,
                        "image": case["image"],
                        "beta": beta,
                        "method": "adaptive",
                        "eta": certified.eta,
                        "budget": budget,
                        "removed64": adaptive_removed,
                        "violation64": max(0.0, adaptive_removed - budget),
                        "wrapper_runtime": wrapper_runtime,
                        **adaptive_metrics,
                    }
                )

                fixed_output = blend(noisy, proposal, fixed[beta])
                fixed_metrics = metric_row(clean, noisy, fixed_output)
                fixed_removed = removed_tv(noisy, fixed_output)
                rows.append(
                    {
                        "split": split,
                        "image": case["image"],
                        "beta": beta,
                        "method": "fixed",
                        "eta": fixed[beta],
                        "budget": budget,
                        "removed64": fixed_removed,
                        "violation64": max(0.0, fixed_removed - budget),
                        "wrapper_runtime": 0.0,
                        **fixed_metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(2027)
    for beta in BETAS:
        adaptive = frame[(frame.split == "test") & (frame.method == "adaptive") & (frame.beta == beta)].sort_values("image")
        fixed_rows = frame[(frame.split == "test") & (frame.method == "fixed") & (frame.beta == beta)].sort_values("image")
        delta = adaptive.psnr.to_numpy() - fixed_rows.psnr.to_numpy()
        draws = rng.integers(0, len(delta), size=(10000, len(delta)))
        boot = delta[draws].mean(axis=1)
        summary_rows.append(
            {
                "beta": beta,
                "adaptive_eta_median": float(adaptive.eta.median()),
                "adaptive_psnr": float(adaptive.psnr.mean()),
                "adaptive_ssim": float(adaptive.ssim.mean()),
                "fixed_eta": fixed[beta],
                "fixed_psnr": float(fixed_rows.psnr.mean()),
                "fixed_ssim": float(fixed_rows.ssim.mean()),
                "fixed_violation_percent": 100.0 * float((fixed_rows.violation64 > 1e-12).mean()),
                "delta_psnr": float(delta.mean()),
                "delta_psnr_ci_low": float(np.quantile(boot, 0.025)),
                "delta_psnr_ci_high": float(np.quantile(boot, 0.975)),
                "max_adaptive_violation64": float(adaptive.violation64.max()),
                "wrapper_runtime_ms": 1000.0 * float(adaptive.wrapper_runtime.mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    metadata = {
        "dataset": "Kodak24 resized to max side 256",
        "sigma255": SIGMA255,
        "seed": SEED,
        "validation_images": list(VAL_IMAGES),
        "test_images": list(TEST_IMAGES),
        "betas": list(BETAS),
        "fixed_eta_grid_step": 0.005,
        "bisection_steps": 28,
        "tv_dtype": "float64",
        "bootstrap": "image-level, 10000 resamples",
    }
    return frame, summary, metadata


def write_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{c cc cc c}",
        r"\toprule",
        r"$\beta$ & $\widetilde\eta$ & PSNR & fixed $\eta$ & PSNR & viol. (\%) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.beta:.2f} & {row.adaptive_eta_median:.3f} & {row.adaptive_psnr:.2f} & "
            f"{row.fixed_eta:.3f} & {row.fixed_psnr:.2f} & {row.fixed_violation_percent:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15))
    axes[0].plot(summary.beta, summary.adaptive_eta_median, "o-", label="adaptive")
    axes[0].plot(summary.beta, summary.fixed_eta, "s--", label="fixed")
    axes[0].set(xlabel=r"relative budget $\beta$", ylabel=r"retention $\eta$", ylim=(-0.03, 1.03))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(summary.beta, summary.adaptive_psnr, "o-", label="adaptive")
    axes[1].plot(summary.beta, summary.fixed_psnr, "s--", label="fixed")
    axes[1].set(xlabel=r"relative budget $\beta$", ylabel="PSNR (dB)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    results = RESULTS
    generated = OUTPUT / "generated"
    figures = OUTPUT / "figures"
    for directory in (results, generated, figures):
        directory.mkdir(parents=True, exist_ok=True)
    cache = DATA / "cache" / "icassp_drunet_sigma25.npz"
    cases = load_or_make_cases(cache, args.regenerate)
    frame, summary, metadata = evaluate(cases)
    frame.to_csv(results / "icassp_budget_audit.csv", index=False)
    summary.to_csv(results / "icassp_budget_summary.csv", index=False)
    (results / "icassp_budget_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_table(summary, generated / "tab_icassp_budget_audit.tex")
    make_figure(summary, figures / "fig_icassp_budget_tradeoff.pdf")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
