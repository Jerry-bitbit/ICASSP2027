"""Check released summaries against their per-image experimental records."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.paths import ROOT


def verify(results: Path) -> None:
    for prefix in ("icassp_budget", "icassp_polyu_budget",
                   "icassp_kodak_sigma15", "icassp_kodak_sigma50"):
        frame = pd.read_csv(results / f"{prefix}_audit.csv", float_precision="round_trip")
        summary = pd.read_csv(results / f"{prefix}_summary.csv", float_precision="round_trip")
        rng = np.random.default_rng(2027)
        for row in summary.itertuples():
            group = frame[(frame.split == "test") & (frame.beta == row.beta)]
            a = group[group.method == "adaptive"].sort_values("image")
            f = group[group.method == "fixed"].sort_values("image")
            assert a.image.tolist() == f.image.tolist()
            assert len(a) == (21 if "polyu" in prefix else 16)
            delta = a.psnr.to_numpy() - f.psnr.to_numpy()
            draws = rng.integers(0, len(delta), size=(10000, len(delta)))
            low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            computed = [a.eta.median(), a.psnr.mean(), a.ssim.mean(), f.psnr.mean(),
                        f.ssim.mean(), delta.mean(), low, high,
                        100 * (f.violation64 > 1e-12).mean()]
            reported = [row.adaptive_eta_median, row.adaptive_psnr, row.adaptive_ssim,
                        row.fixed_psnr, row.fixed_ssim, row.delta_psnr,
                        row.delta_psnr_ci_low, row.delta_psnr_ci_high,
                        row.fixed_violation_percent]
            np.testing.assert_allclose(computed, reported, rtol=0, atol=1e-9)
            assert (a.violation64 <= 1e-12).all()

    proposals = pd.read_csv(results / "icassp_proposal_summary.csv")
    for method in ("wavelet", "nlm"):
        per_image = pd.read_csv(results / f"{method}_per_image.csv")
        for row in proposals[proposals.proposal == method].itertuples():
            group = per_image[(per_image.split == "test") & (per_image.beta == row.beta)]
            a = group[group.method == "adaptive"].sort_values("image")
            f = group[group.method == "fixed"].sort_values("image")
            np.testing.assert_allclose(
                [a.eta.median(), a.psnr.mean(), (a.psnr.to_numpy()-f.psnr.to_numpy()).mean()],
                [row.adaptive_eta_median, row.adaptive_psnr, row.delta_psnr], rtol=0, atol=2e-6)
            assert (a.violation64 <= 1e-12).all()

    frame = pd.read_csv(results / "controllers" / "per_image.csv", float_precision="round_trip")
    methods = ["halving", "coarse_0p05", "bisection8", "bisection12", "bisection28"]
    adaptive = frame[frame.method.isin(methods)]
    assert len(adaptive) == 1380
    assert not adaptive.duplicated(["dataset", "image", "beta", "method"]).any()
    assert (adaptive.groupby("method").size() == 276).all()
    assert (adaptive.residual <= 1e-12).all()
    assert not adaptive.violation.any()
    precision = pd.read_csv(results / "controllers" / "precision_summary.csv")
    for row in precision.itertuples():
        group = frame[frame.dataset == row.dataset]
        pairs = group[group.method == "bisection28"].merge(
            group[group.method == row.method], on=["image", "beta"], suffixes=("_28", "_other"),
            validate="one_to_one")
        np.testing.assert_allclose(
            [np.max(np.abs(pairs.psnr_28-pairs.psnr_other)),
             pairs.time_seconds_28.mean()/pairs.time_seconds_other.mean()],
            [row.max_absolute_psnr_difference, row.runtime_ratio_28_to_other], rtol=0, atol=1e-9)
    print("Verified table summaries and bootstrap intervals against per-image records.")
    print("Verified controller records: 276 conditions, 1,380 adaptive outputs, 0 violations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    results = parser.parse_args().results
    verify(results)
    from experiments.weak_structure.paper import verify as verify_weak_structure
    verify_weak_structure(results / "weak_structure")
