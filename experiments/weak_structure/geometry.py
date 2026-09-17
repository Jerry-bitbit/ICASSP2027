"""Antialiased geometric targets; no noisy or restored input is accepted."""
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

SPLIT_CODES = {"development": 0, "validation": 1, "test": 2}


def geometry_seed(config, split, index):
    return [config["seeds"]["geometry"], SPLIT_CODES[split], index]


def noise_seed(config, split, index, sigma_index, repeat):
    return [config["seeds"]["noise"], SPLIT_CODES[split], index, sigma_index, repeat]


def generate(config, split, index):
    g = config["geometry"]
    n, ss = g["size"], g["supersampling"]
    rng = np.random.default_rng(np.random.SeedSequence(geometry_seed(config, split, index)))
    background = float(rng.uniform(*g["background"]))
    count = int(rng.integers(g["line_count"][0], g["line_count"][1] + 1))
    t = np.linspace(0, 1, g["curve_samples"])
    yy, xx = np.mgrid[:n, :n]
    pixels = np.stack((xx + .5, yy + .5), axis=-1).reshape(-1, 2)
    coverages, nearest, line_info = [], [], []
    for i in range(count):
        center = rng.uniform(g["center_margin"], n - g["center_margin"], size=2)
        theta = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)])
        normal = np.array([-direction[1], direction[0]])
        length = float(rng.uniform(*g["length"]))
        curvature = float(rng.uniform(*g["curvature_fraction"]))
        width, contrast = float(rng.uniform(*g["width"])), float(rng.uniform(*g["contrast"]))
        p0, p2 = center - length * direction / 2, center + length * direction / 2
        p1 = center + curvature * length * normal
        curve = (1 - t[:, None]) ** 2 * p0 + 2 * (1 - t[:, None]) * t[:, None] * p1 + t[:, None] ** 2 * p2
        tangent = 2 * (1 - t[:, None]) * (p1 - p0) + 2 * t[:, None] * (p2 - p1)
        normals = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        polygon = np.concatenate((curve + width / 2 * normals, (curve - width / 2 * normals)[::-1]))
        canvas = Image.new("L", (n * ss, n * ss), 0)
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in (polygon * ss - .5)], fill=1)
        # Exact block area average of the supersampled binary geometry.
        coverage = np.asarray(canvas, np.float32).reshape(n, ss, n, ss).mean(axis=(1, 3))
        coverages.append(coverage)
        _, idx = cKDTree(curve).query(pixels)
        nearest.append(t[idx].reshape(n, n))
        line_info.append({"line": i, "control_points_xy": [p0.tolist(), p1.tolist(), p2.tolist()],
                          "width": width, "contrast": contrast, "length": length,
                          "curvature_fraction": curvature})
    cov = np.stack(coverages)
    # Max coverage defines the union; additive contrasts define brightness.
    coverage = cov.max(axis=0)
    image = np.clip(background + np.sum(cov * np.array([r["contrast"] for r in line_info])[:, None, None], axis=0), 0, 1).astype(np.float32)
    valid = np.zeros((n, n), bool)
    border = g["boundary"]
    valid[border:n-border, border:n-border] = True
    structure = coverage >= g["structure_threshold"]
    support = cov > 0
    all_support = support.any(axis=0)
    bg = valid & (distance_transform_edt(~all_support) > g["background_clearance"])
    cores, sides, usable = [], [], []
    for i in range(count):
        others = np.delete(support, i, axis=0).any(axis=0)
        uncontaminated = distance_transform_edt(~others) > g["other_line_clearance"] if others.any() else np.ones((n,n), bool)
        # End caps cannot contribute to either core or side estimates.
        segment = (nearest[i] >= .05) & (nearest[i] <= .95)
        eligible = valid & uncontaminated & segment
        core = (cov[i] >= g["core_threshold"]) & eligible
        distance = distance_transform_edt(~support[i])
        side = (distance >= g["side_distance"][0]) & (distance <= g["side_distance"][1]) & eligible
        reason = []
        if core.sum() < g["minimum_core_pixels"]:
            reason.append("insufficient_geometry_core")
        if side.sum() < g["minimum_side_pixels"]:
            reason.append("insufficient_geometry_side")
        line_info[i].update(valid=not reason, invalid_reasons=reason,
                            core_pixels=int(core.sum()), side_pixels=int(side.sum()))
        cores.append(core); sides.append(side); usable.append(not reason)
    arrays = {"clean": np.repeat(image[..., None], 3, axis=2), "coverage": coverage,
              "line_coverage": cov, "structure_mask": structure, "valid_region": valid,
              "core_masks": np.stack(cores), "side_masks": np.stack(sides),
              "background_mask": bg, "line_valid": np.array(usable, bool)}
    info = {"id": f"{split}_{index:03d}", "split": split, "index": index,
            "geometry_seed": geometry_seed(config, split, index), "background": background,
            "geometry_sha256": hashlib.sha256(cov.tobytes()).hexdigest(), "lines": line_info,
            "positive_pixels": int((structure & valid).sum()), "background_pixels": int(bg.sum()),
            "image_invalid_reasons": []}
    if bg.sum() < g["minimum_background_pixels"]:
        info["image_invalid_reasons"].append("insufficient_geometry_background")
    if not structure[valid].any() or structure[valid].all():
        info["image_invalid_reasons"].append("AP_requires_both_geometry_classes")
    return arrays, info
