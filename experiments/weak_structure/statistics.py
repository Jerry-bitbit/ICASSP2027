"""Paired cluster statistics; one resampling unit is one clean geometry."""
import numpy as np
import pandas as pd

METRICS = ["ap", "background_rmse", "background_rmse_ratio", "contrast_signed_bias", "contrast_rmse", "psnr"]


def selected_frame(frame, frozen):
    pieces = [frame[frame.method.isin(["noisy", "proposal", "clean"])].copy()]
    for source, value, label in [("fixed_scan", frozen["eta"], "fixed"),
                                  ("tv_scan", frozen["beta"], "tv"),
                                  ("tv_scan", .2, "beta_0.2")]:
        subset = frame[(frame.method == source) & (frame.parameter == value)].copy()
        subset["method"] = label
        pieces.append(subset)
    return pd.concat(pieces, ignore_index=True)


def geometry_means(frame, group_keys):
    # Explicit hierarchy makes the correlation unit and condition weights clear.
    keys = list(dict.fromkeys(group_keys + ["geometry", "sigma255", "a"]))
    within = frame.groupby(keys, dropna=False)[METRICS].mean()
    return within.groupby(level=group_keys + ["geometry"], dropna=False).mean().reset_index()


def bootstrap_mean(values, config):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError("Bootstrap requires finite geometry-level values; no automatic cluster removal")
    rng = np.random.default_rng(config["seeds"]["bootstrap"])
    indices = rng.integers(0, len(values), size=(config["statistics"]["bootstrap_repeats"], len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, config["statistics"]["percentiles"])
    return float(values.mean()), float(low), float(high)


def paired(frame, target, reference, metric, config):
    aggregates = geometry_means(frame[frame.method.isin([target, reference])], ["method"])
    table = aggregates.pivot(index="geometry", columns="method", values=metric).sort_index()
    if table.isna().any().any():
        raise ValueError("Missing paired geometry; no dropping incomplete pairs")
    differences = table[target] - table[reference]
    estimate, low, high = bootstrap_mean(differences.to_numpy(), config)
    return {"target": target, "reference": reference, "metric": metric,
            "difference": estimate, "ci_low": low, "ci_high": high, "n_geometries": len(differences)}, differences


def summarize(frame, config):
    per_geometry = geometry_means(frame, ["a", "method"])
    rows = []
    for (a, method), group in per_geometry.groupby(["a", "method"]):
        row = {"a": a, "method": method, "n_geometries": len(group)}
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    comparisons, differences = [], []
    strata = [("overall", frame)]
    strata += [(f"a={a:g}", group) for a, group in frame.groupby("a")]
    strata += [(f"sigma={s},a={a:g}", group) for (s,a),group in frame.groupby(["sigma255","a"])]
    for stratum, group in strata:
        for target, reference, metric in [("tv", "proposal", "ap"), ("tv", "fixed", "ap"),
                                           ("tv", "noisy", "background_rmse_ratio"),
                                           ("beta_0.2", "proposal", "ap")]:
            result, delta = paired(group, target, reference, metric, config)
            result.update(stratum=stratum, analysis="primary" if stratum == "overall" and target == "tv" and reference == "proposal" else "secondary/exploratory")
            comparisons.append(result)
            differences.extend({"stratum": stratum, "target": target, "reference": reference,
                                "metric": metric, "geometry": geometry, "paired_difference": value}
                               for geometry, value in delta.items())
    return pd.DataFrame(rows), pd.DataFrame(comparisons), pd.DataFrame(differences), per_geometry
