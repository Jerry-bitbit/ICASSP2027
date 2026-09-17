"""Fixed raw Sato responses and per-image, tie-aware average precision."""
import numpy as np
from skimage.filters import sato
from .audit import tv_reference


def gray(u):
    return np.asarray(u, dtype=np.float64).mean(axis=2)


def detect(u, detector):
    return sato(gray(u), sigmas=tuple(detector["sigmas"]),
                black_ridges=detector["black_ridges"],
                mode=detector["mode"], cval=detector["cval"])


def average_precision(labels, scores):
    """Sum recall increments times precision at each distinct score threshold.

    Non-interpolated AP, including all equal scores together. No random tie
    breaking or pixel pooling across observations. Undefined labels fail.
    """
    y = np.asarray(labels, dtype=bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size != s.size or not y.any() or y.all() or not np.isfinite(s).all():
        raise ValueError("AP requires finite aligned scores and both classes")
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), s.size - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    gains = np.diff(np.r_[0, tp])
    return float(np.sum(gains * (tp / (ends + 1)), dtype=np.float64) / tp[-1])


def score(u, f, geometry, detector, response=None):
    x = geometry["clean"]
    ug, xg, fg = gray(u), gray(x), gray(f)
    valid, mask, bg = (geometry[k] for k in ("valid_region", "structure_mask", "background_mask"))
    if response is None:
        response = detect(u, detector)
    ap = average_precision(mask[valid], response[valid])
    rmse = float(np.sqrt(np.mean((ug[bg] - xg[bg]) ** 2)))
    baseline = float(np.sqrt(np.mean((fg[bg] - xg[bg]) ** 2)))
    line_rows, errors = [], []
    for i, (core, side, usable) in enumerate(zip(geometry["core_masks"], geometry["side_masks"], geometry["line_valid"])):
        contrast = reference = bias = float("nan")
        if usable:
            contrast = float(ug[core].mean() - ug[side].mean())
            reference = float(xg[core].mean() - xg[side].mean())
            bias = contrast - reference
            errors.append(bias)
        line_rows.append({"line": i, "line_valid": bool(usable),
                          "core_pixels": int(core.sum()), "side_pixels": int(side.sum()),
                          "contrast": contrast, "clean_contrast": reference, "contrast_bias": bias})
    mse = float(np.mean((np.asarray(u, np.float64) - x) ** 2))
    return {"ap": ap, "background_rmse": rmse, "background_rmse_ratio": rmse / baseline,
            "contrast_signed_bias": float(np.mean(errors)) if errors else float("nan"),
            "contrast_rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan"),
            "valid_lines": len(errors), "psnr": -10 * np.log10(mse) if mse else float("inf"),
            "tv": tv_reference(u)}, line_rows, response
