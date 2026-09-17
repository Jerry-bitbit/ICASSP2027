"""Complete diagnostic curves and a report generated only from executed runs."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
from .pipeline import open_run, require_frozen, read_json, load_npz, case_id, jobs, write_json
from .statistics import METRICS, selected_frame, geometry_means, summarize, bootstrap_mean, paired

COLORS = {"noisy": "#7b8794", "proposal": "#c75b39", "fixed": "#2d7bb6", "tv": "#279267", "clean": "#333333", "beta_0.2": "#a766ad"}
LABELS = {"noisy": "Noisy", "proposal": "Proposal", "fixed": "Fixed (val-selected)", "tv": "TV (val-selected)", "clean": "Clean reference", "beta_0.2": "TV beta=0.2"}


def markdown_table(frame, columns, decimals=5):
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}" if np.isfinite(value) else "inf" if np.isinf(value) else "NA"
        return str(value)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(fmt(row[c]) for c in columns) + " |" for _,row in frame.iterrows())
    return "\n".join(lines)


def curves(frame, selected, frozen, config, out, split):
    labels = LABELS if split != "development" else LABELS | {"fixed": "Fixed (preassigned)", "tv": "TV (preassigned)"}
    scans = frame[frame.method.isin(["fixed_scan", "tv_scan"])]
    geom = geometry_means(scans, ["method", "parameter"])
    curve = geom.groupby(["method", "parameter"])[METRICS].mean().reset_index()
    curve.to_csv(out / f"{split}_tradeoff_curve.csv", index=False)
    points = geometry_means(selected, ["method"]).groupby("method")[METRICS].mean()
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    for source, label, color in [("fixed_scan", "All 101 fixed eta values", COLORS["fixed"]), ("tv_scan", "All 6 TV beta values", COLORS["tv"])]:
        g = curve[curve.method == source].sort_values("parameter")
        ax.plot(g.background_rmse_ratio, g.ap, "-", color=color, label=label, linewidth=1.8)
        if source == "tv_scan":
            ax.scatter(g.background_rmse_ratio, g.ap, color=color, s=23)
            for row in g.itertuples():
                ax.annotate(f"{row.parameter:g}", (row.background_rmse_ratio, row.ap), xytext=(4,4), textcoords="offset points", fontsize=8)
    for method in ["noisy", "proposal", "fixed", "tv"]:
        point = points.loc[method]
        ax.scatter(point.background_rmse_ratio, point.ap, marker="*" if method in ["fixed","tv"] else "o", s=160 if method in ["fixed","tv"] else 55, color=COLORS[method], edgecolors="white", zorder=5, label=labels[method])
    ax.set(xlabel="Background RMSE / noisy background RMSE", ylabel="Mean per-image AP",
           title=f"{split.capitalize()} trade-off — complete diagnostic scans")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / f"{split}_tradeoff.png", dpi=170)
    fig.savefig(out / f"{split}_tradeoff.pdf")
    plt.close(fig)
    # Separate all sigma/a curves, with the same globally selected points.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout="constrained")
    for (s, a), ax in zip([(s,a) for s in config["sigma255"] for a in config["strengths"]], axes.flat):
        subset = scans[(scans.sigma255 == s) & (scans.a == a)]
        means = subset.groupby(["method", "parameter"])[METRICS].mean().reset_index()
        for source, color, chosen in [("fixed_scan", COLORS["fixed"], frozen["eta"]), ("tv_scan", COLORS["tv"], frozen["beta"])]:
            g = means[means.method == source].sort_values("parameter")
            ax.plot(g.background_rmse_ratio, g.ap, color=color)
            p = g[g.parameter == chosen].iloc[0]
            ax.scatter(p.background_rmse_ratio, p.ap, marker="*", s=80, color=color)
        for method in ["noisy", "proposal"]:
            p = selected[(selected.sigma255 == s) & (selected.a == a) & (selected.method == method)][METRICS].mean()
            ax.scatter(p.background_rmse_ratio, p.ap, color=COLORS[method], s=25)
        ax.set_title(f"sigma={s}/255, a={a:g}"); ax.grid(alpha=.2)
    fig.supxlabel("Background RMSE ratio"); fig.supylabel("Mean AP")
    fig.suptitle(f"{split.capitalize()} strata; stars are " + ("preassigned smoke points" if split == "development" else "shared validation selections"))
    fig.savefig(out / f"{split}_tradeoff_strata.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    aggregates = geometry_means(selected, ["a", "method"])
    for method in LABELS:
        rows = []
        for a, g in aggregates[aggregates.method == method].groupby("a"):
            mean, low, high = bootstrap_mean(g.ap.to_numpy(), config)
            rows.append((a, mean, low, high))
        data = np.array(rows)
        ax.plot(data[:,0], data[:,1], "o-", color=COLORS[method], label=labels[method])
        ax.fill_between(data[:,0], data[:,2], data[:,3], color=COLORS[method], alpha=.1)
    ax.set(xlabel="DRUNet noise-map multiplier a", ylabel="Mean per-image AP", xticks=config["strengths"], title=f"{split.capitalize()} robustness (geometry bootstrap 95% CI)")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / f"{split}_robustness.png", dpi=170)
    fig.savefig(out / f"{split}_robustness.pdf"); plt.close(fig)


@lru_cache(maxsize=1)
def _visual_bundle(output_dir, split, index, sigma255, repeat):
    root = Path(output_dir)
    name = case_id(split, index, sigma255, repeat)
    geom = load_npz(root / "cache" / split / "geometry" / f"{split}_{index:03d}.npz")
    cache = load_npz(root / "cache" / split / "proposals" / f"{name}.npz")
    outputs = load_npz(root / "arrays" / split / f"{name}.npz")
    response = load_npz(root / "responses" / split / f"{name}.npz")
    return geom, cache, outputs, response


def _visual_case(config, split, index, sigma255, repeat, ai, frozen):
    geom, cache, outputs, response = _visual_bundle(config["output_dir"], split, index, sigma255, repeat)
    bi = config["betas"].index(frozen["beta"])
    return [geom["clean"].mean(axis=2), cache["noisy"].mean(axis=2), cache["proposals"][ai].mean(axis=2),
            outputs["fixed_outputs"][ai].mean(axis=2), outputs["tv_outputs"][ai,bi].mean(axis=2),
            geom["structure_mask"], outputs["tv_outputs"][ai,0].mean(axis=2)], \
           [response["clean"], response["noisy"], response[f"proposal_{ai}"], response[f"fixed_{ai}"],
            response[f"tv_{ai}"], None, response[f"beta02_{ai}"]]


def visualizations(config, split, frozen, out):
    vis = config["visualization"]
    titles = ["Clean", "Noisy", "Proposal", "Fixed", "TV selected", "GT mask", "TV beta=0.2"]
    ids = vis["example_ids"] if split == "test" else [f"development_{i:03d}" for i in range(config["splits"][split])]
    for identifier in ids:
        arrays, responses = _visual_case(config, split, int(identifier.split("_")[-1]), vis["sigma255"], vis["noise_repeat"], config["strengths"].index(vis["strength"]), frozen)
        fig, axes = plt.subplots(2, 7, figsize=(17, 5.8), layout="constrained")
        for j, (u, response) in enumerate(zip(arrays,responses)):
            axes[0,j].imshow(u, cmap="gray", vmin=vis["gray_limits"][0], vmax=vis["gray_limits"][1])
            axes[0,j].set_title(titles[j])
            if response is not None:
                axes[1,j].imshow(response, cmap="magma", vmin=vis["response_limits"][0], vmax=vis["response_limits"][1])
            axes[0,j].axis("off"); axes[1,j].axis("off")
        fig.suptitle(f"Preassigned {identifier}, sigma={vis['sigma255']}/255, repeat={vis['noise_repeat']}, a={vis['strength']:g}\nImages [0,1]; raw detector responses [0,0.06] (same display scales for every method)")
        fig.savefig(out / f"example_{identifier}.png", dpi=160)
        plt.close(fig)
    # Every observation and every a appears in manifest order. Raster pages keep
    # the complete 1800-case sheet practical; raw arrays/responses remain saved.
    contacts = out / "contact_sheet"
    contacts.mkdir(exist_ok=True)
    thumb, gap, header, rows_per_page = 96, 4, 28, vis["contact_sheet_cases_per_page"]
    width = 7 * (thumb + gap)
    height = header + rows_per_page * (2*thumb + 24)
    page_number, row_number = 0, 0
    page = None
    index_rows = []
    with PdfPages(out / "all_test_examples_contact_sheet.pdf" if split == "test" else out / "development_contact_sheet.pdf") as pdf:
        def save_page(current, number):
            current.save(contacts / f"page_{number:03d}.jpg", quality=88)
            fig = plt.figure(figsize=(width/90, height/90))
            ax = fig.add_axes([0,0,1,1]); ax.imshow(current); ax.axis("off")
            pdf.savefig(fig, dpi=90); plt.close(fig)
        for _, _, i, si, repeat in jobs(config, split):
            sigma = config["sigma255"][si]
            for ai, a in enumerate(config["strengths"]):
                if row_number == 0:
                    page_number += 1
                    page = Image.new("RGB", (width,height), "white")
                    draw = ImageDraw.Draw(page)
                    for j,title in enumerate(titles): draw.text((j*(thumb+gap),7), title, fill="black")
                arrays, responses = _visual_case(config, split, i, sigma, repeat, ai, frozen)
                y = header + row_number*(2*thumb+24)
                label = f"{split}_{i:03d} s={sigma} r={repeat} a={a:g}"
                draw.text((0,y), label, fill="black")
                for j,(u,r) in enumerate(zip(arrays,responses)):
                    for k,arr in enumerate((u,r)):
                        if arr is None: continue
                        if k == 0:
                            norm = np.clip((arr-vis["gray_limits"][0])/(vis["gray_limits"][1]-vis["gray_limits"][0]),0,1)
                            pixels = np.repeat((norm[...,None]*255).astype(np.uint8),3,axis=2)
                        else:
                            norm = np.clip((arr-vis["response_limits"][0])/(vis["response_limits"][1]-vis["response_limits"][0]),0,1)
                            pixels = (plt.get_cmap("magma")(norm)[...,:3]*255).astype(np.uint8)
                        tile = Image.fromarray(pixels).resize((thumb,thumb), Image.Resampling.BOX)
                        page.paste(tile, (j*(thumb+gap),y+20+k*thumb))
                index_rows.append({"geometry":f"{split}_{i:03d}","sigma255":sigma,"repeat":repeat,"a":a,"page":page_number,"row":row_number+1})
                row_number += 1
                if row_number == rows_per_page:
                    save_page(page,page_number); row_number = 0
                    print(f"contact sheet page {page_number}", flush=True)
        if row_number: save_page(page,page_number)
    pd.DataFrame(index_rows).to_csv(out / "contact_sheet_index.csv", index=False)


def report(config, split="test"):
    root = open_run(config)
    if split not in ("development", "test"):
        raise ValueError("report accepts development or test")
    if not (root / f"{split}_complete.json").exists():
        raise RuntimeError("No completed evaluation to report; results will not be estimated")
    frozen = require_frozen(config, split)
    frame = pd.read_csv(root / f"{split}_per_sample.csv")
    selected = selected_frame(frame, frozen)
    out = root / ("report" if split == "test" else "smoke_report")
    out.mkdir(exist_ok=True)
    main, comparisons, differences, geometry = summarize(selected, config)
    # Retain per-beta paired diagnostics without selecting a test-best beta.
    beta_rows = []
    for beta in config["betas"]:
        candidates = frame[(frame.method == "tv_scan") & (frame.parameter == beta)].copy()
        candidates["method"] = "beta_candidate"
        paired_data = pd.concat([candidates, frame[frame.method == "proposal"]], ignore_index=True)
        strata = [("overall", paired_data)] + [(f"a={a:g}", part) for a,part in paired_data.groupby("a")]
        for stratum, part in strata:
            result, _ = paired(part, "beta_candidate", "proposal", "ap", config)
            beta_rows.append(result | {"beta": beta, "stratum": stratum, "analysis": "exploratory"})
    pd.DataFrame(beta_rows).to_csv(out / "all_beta_paired_comparisons.csv", index=False)
    geometric_regions = []
    for path in sorted((root / "cache" / split / "geometry").glob("*.json")):
        details = read_json(path)
        for line in details["lines"]:
            geometric_regions.append({"geometry": details["id"], "line": line["line"], "valid": line["valid"],
                                      "core_pixels": line["core_pixels"], "side_pixels": line["side_pixels"],
                                      "invalid_reasons": ";".join(line["invalid_reasons"])})
    pd.DataFrame(geometric_regions).to_csv(out / "geometric_region_validity.csv", index=False)
    invalid_lines = sum(not line["valid"] for line in geometric_regions)
    main.to_csv(out / "main_table.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    differences.to_csv(out / "geometry_paired_differences.csv", index=False)
    geometry.to_csv(out / "geometry_means.csv", index=False)
    selected.to_csv(out / "selected_per_sample.csv", index=False)
    # Explicitly join AP paired CIs into the main table's controller rows.
    joined = main.copy()
    for a in config["strengths"]:
        for reference in ("proposal", "fixed"):
            r = comparisons[(comparisons.stratum == f"a={a:g}") & (comparisons.target == "tv") & (comparisons.reference == reference) & (comparisons.metric == "ap")].iloc[0]
            for key in ("difference", "ci_low", "ci_high"):
                joined.loc[(joined.a == a) & (joined.method == "tv"), f"ap_vs_{reference}_{key}"] = r[key]
    joined.to_csv(out / "main_table_with_paired_ci.csv", index=False)
    curves(frame, selected, frozen, config, out, split)
    if split == "test":
        validation = pd.read_csv(root / "validation_per_sample.csv")
        curves(validation, selected_frame(validation,frozen), frozen, config, out, "validation")
    overall = geometry_means(selected, ["method"]).groupby("method")[METRICS].mean()
    p = comparisons[(comparisons.stratum == "overall") & (comparisons.target == "tv") & (comparisons.reference == "proposal")].iloc[0]
    fixed = comparisons[(comparisons.stratum == "overall") & (comparisons.target == "tv") & (comparisons.reference == "fixed")].iloc[0]
    bg_geometries = geometry_means(selected[selected.method == "tv"], ["method"])
    bg_mean, bg_low, bg_high = bootstrap_mean(bg_geometries.background_rmse_ratio.to_numpy(), config)
    equivalent_margin = config["statistics"]["fixed_comparison_equivalence_margin_ap"]
    categories = {
        "a_AP_improvement_and_background_below_noisy": bool(p.ci_low > 0 and bg_high < 1),
        "b_contrast_bias_closer_to_clean_without_mean_AP_improvement": bool(abs(overall.loc['tv','contrast_signed_bias']) < abs(overall.loc['proposal','contrast_signed_bias']) and p.difference <= 0),
        "c_comparable_to_fixed_within_predeclared_margin": bool(fixed.ci_low >= -equivalent_margin and fixed.ci_high <= equivalent_margin),
        "d_negative_mean_AP_effect": bool(p.difference < 0),
        "d_negative_AP_effect_CI_below_zero": bool(p.ci_high < 0),
        "d_background_error_above_noisy": bool(bg_mean > 1)}
    adverse = comparisons[(comparisons.target == "tv") & (comparisons.reference == "proposal") &
                          (comparisons.metric == "ap") & (comparisons.stratum != "overall") & (comparisons.difference < 0)]
    categories["d_any_secondary_stratum_negative_mean_AP"] = bool(len(adverse))
    categories["d_any_secondary_stratum_AP_CI_below_zero"] = bool((adverse.ci_high < 0).any())
    info = read_json(root / f"{split}_complete.json")
    env = read_json(root / "environment.json")
    clean_by_geometry = selected[selected.method == "clean"].groupby("geometry").ap.mean()
    low_clean_ids = clean_by_geometry[clean_by_geometry < config["selection"]["clean_ap_warning_threshold"]].index.tolist()
    write_json(out / "interpretation.json", categories | {"background_ratio": bg_mean, "background_ratio_ci": [bg_low,bg_high],
                                                          "low_clean_reference_geometry_ids": low_clean_ids})
    text = [f"# {config['experiment']}",
            f"\nExecuted split: **{split}**; {info['geometries']} clean geometry clusters. " + ("Development smoke only; excluded from formal statistics." if split == "development" else "Held-out test; all sigma/a conditions and all noise repeats retained."),
            f"\nShared operating points: eta={frozen['eta']:g}, beta={frozen['beta']:g}. " + ("Preassigned development values." if split == "development" else "Selected solely by validation macro AP across all 40 geometries, both sigma and all three a values, with 3 repeats each."),
            "\n## Primary comparison",
            f"\nSelected TV minus proposal overall macro AP: **{p.difference:+.6f}**, paired geometry bootstrap percentile 95% CI **[{p.ci_low:+.6f}, {p.ci_high:+.6f}]**.",
            f"\nSelected TV background RMSE / noisy background RMSE: **{bg_mean:.6f}**, geometry bootstrap 95% CI **[{bg_low:.6f}, {bg_high:.6f}]**.",
            "\n## Interpretation",
            f"\n(a) AP improvement with background error below noisy, supported by both intervals: **{categories['a_AP_improvement_and_background_below_noisy']}**.",
            f"\n(b) Signed contrast bias closer to clean, without mean AP improvement: **{categories['b_contrast_bias_closer_to_clean_without_mean_AP_improvement']}**.",
            f"\n(c) Comparable to fixed within the predeclared ±{equivalent_margin:g} AP margin (entire CI inside margin): **{categories['c_comparable_to_fixed_within_predeclared_margin']}**. TV minus fixed = {fixed.difference:+.6f}, 95% CI [{fixed.ci_low:+.6f}, {fixed.ci_high:+.6f}]. An interval merely overlapping zero is not evidence of equivalence.",
            f"\n(d) Negative mean AP effect: **{categories['d_negative_mean_AP_effect']}**; AP CI entirely below zero: **{categories['d_negative_AP_effect_CI_below_zero']}**; mean background error above noisy: **{categories['d_background_error_above_noisy']}**.",
            "\nSecondary strata with negative mean TV-minus-proposal AP: " + (", ".join(f"{r.stratum}: {r.difference:+.6f} [{r.ci_low:+.6f}, {r.ci_high:+.6f}]" for r in adverse.itertuples()) if len(adverse) else "none") + ". These exploratory intervals are not multiplicity-adjusted.",
            "\nThese are separately reported patterns and may coexist. Contrast recovery alone is not treated as detection improvement. Mixed or uncertain patterns remain visible in the estimates and intervals.",
            f"\nClean reference macro AP: {info['clean_macro_ap']:.6f}; detector/task mismatch flag (threshold {config['selection']['clean_ap_warning_threshold']}): **{info['detector_task_mismatch_flag']}**. No detector settings changed after formal configuration freeze.",
            f"\nIndividual clean-reference geometries below the same predeclared AP threshold: {len(low_clean_ids)}" + (" (" + ", ".join(low_clean_ids) + ")" if low_clean_ids else "") + ". They remain in all applicable statistics; per-image clean AP is retained in the CSV.",
            "\n## Main table by mismatch strength",
            "\n" + markdown_table(main, ["a","method","ap","background_rmse_ratio","contrast_rmse","psnr"]),
            "\n### AP paired intervals by a (secondary)",
            "\n" + markdown_table(comparisons[(comparisons.stratum.str.startswith("a=")) & (comparisons.target=="tv") & (comparisons.metric=="ap")], ["stratum","reference","difference","ci_low","ci_high","n_geometries"]),
            "\n### Strict beta=0.2 (exploratory)",
            "\n" + markdown_table(main[main.method=="beta_0.2"], ["a","ap","background_rmse_ratio","contrast_rmse","psnr"]),
            "\nBeta=0.2 is displayed independently of the selected beta. Its paired AP comparisons with proposal are in paired_comparisons.csv; a stricter budget is not assumed better.",
            "\n## Protocol and isolation",
            "\nImages are 256×256 grayscale geometries replicated to RGB. Gaussian noise is independent by RGB channel, geometry, sigma and repeat; the identical clipped float32 observation is shared across a and methods. DRUNet uses noise_map=a*sigma and the original CPU inference/preprocessing. Geometry and noise use separate seeded streams; validation/test geometries are disjoint.",
            "\nMasks derive only from antialiased coverage (8× supersampling, block area averaging); AP uses coverage≥0.5 inside a common 12-pixel boundary. Per-line core/side masks exclude end caps and pixels within 8 pixels of other line support. Geometric degeneracy reasons are saved for every line; invalid line contrast is NA, while the geometry remains in AP and noise metrics. Background is more than 10 pixels from all line support. No output-dependent exclusion occurs.",
            f"\nGeometric line regions: {len(geometric_regions)} total, {invalid_lines} invalid by the fixed pixel-count rules. See geometric_region_validity.csv. All geometries remain in the primary AP analysis.",
            "\nAP, background RMSE and contrast use float64 arithmetic RGB mean. No per-image input or response normalization. Sato uses sigmas=(1,2,3), black_ridges=False, mode=reflect, cval=0; raw responses feed tie-aware, non-interpolated pixel AP, computed separately per image. RGB PSNR uses range 1; clean-reference PSNR is infinite.",
            "\nClean/masks are used for validation operating-point selection and test scoring. The controller interface accepts only f, v, beta. Global fixed eta is unconstrained by TV. All 101 fixed eta and six beta values are evaluated; test scans are diagnostic/exploratory and do not select deployed points. Ties within 1e-12 choose the smaller parameter.",
            "\nStatistics first average repeats within each geometry/sigma/a, then equally average conditions within geometry. Paired differences are resampled by clean geometry with 10,000 draws and percentile 95% intervals. Pixels, repeats, sigma/a and beta are never independent bootstrap units. Strata, beta scans and fixed comparisons are secondary/exploratory; no multiplicity-adjusted claims are made.",
            f"\nTV uses the original RGB vector-valued zero-outward forward-difference definition, epsilon=1e-8, float64 TV, float32 endpoints/outputs and 28 bisection steps. All {info['controller_arrays_audited']} actual returned arrays were saved and reopened for independent audit at tolerance 1e-12; maximum signed budget residual={info['maximum_budget_residual']:.4g}. Non-TV scan budgets are NA because those outputs have no budget constraint.",
            "\n## Figures and complete records",
            f"\n![Complete trade-off]({split}_tradeoff.png)\n\n![Robustness]({split}_robustness.png)",
            "\nThe stars identify validation selections (preassigned values in development), not test optima. All method images share [0,1]; every detector response shares [0,0.06] for display only. Numeric scoring uses unclipped raw responses. Main example IDs were specified in the manifest before generation. contact_sheet_index.csv enumerates every geometry/noise/sigma/a case; contact_sheet/ and the PDF contain all examples.",
            "\n## Actual execution environment",
            f"\nPython: {env['python']}. Device: {env['device']}. Packages: " + ", ".join(f"{k}={v}" for k,v in env['versions'].items()) + ".",
            f"\nModel: `{env['model_path']}`. SHA256: `{env['model_sha256']}`.",
            f"\nSato runtime signature: `{env['sato_signature']}`. Seeds: `{env['seeds']}`.",
            "\nDetector specification: [scikit-image Sato](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.sato). Metric definition: [non-interpolated average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).",
            "\n[Run commands and cache layout](../../../experiments/weak_structure/README.md)."
            ]
    (out / "report.md").write_text("\n".join(text) + "\n")
    visualizations(config, split, frozen, out)
    write_json(out / "report_complete.json", {"complete": True, "split": split, "cases": len(jobs(config,split))*len(config['strengths'])})
    print(f"Report written: {out / 'report.md'}", flush=True)
