# Weak-structure detectability under denoising-strength mismatch

Independent experiment for **Section 3.3 and Figure 4 of the revised manuscript**. The existing paper experiments, controller,
DRUNet implementation, and reference result files are unchanged. Both positive and
negative results are retained. The scientific question is whether observation-relative
TV control can improve weak-line detection under some denoising-strength conditions
while retaining background noise suppression relative to the noisy observation.

## Repository inspection and reused code

| Existing implementation | Use in this experiment |
| --- | --- |
| `experiments/run_drunet_subset.py::load_drunet,denoise_drunet` | Exact color DRUNet architecture, strict weights loading, noise channel, reflection padding, clipping, float32 CPU inference |
| `methods/ectv.py::certify_candidate,tv_energy_float64` | Original 28-step bisection and RGB vector-valued smoothed TV |
| `experiments/run_controller_comparison.py::tv_audit` and prior audit `ICASSP_FINAL_AUDIT_20260916/scripts/reference_tv.py` | Independent accumulation pattern for the saved-array audit |
| `data/cache/icassp_drunet_*.npz`, original budget scripts | Shared float32 endpoint cache convention; this experiment uses a separate cache |
| `experiments/rebuild_paper.py` | Existing Matplotlib/PDF workflow; new experiment-specific figures use the same installed plotting library |

The revised manuscript supplied as `F:\ICASSP2027_final (1).pdf` is included at
`paper/manuscript.pdf`. Section 3.3 and Figure 4 correspond to this experiment.

## Regenerate the revised paper from reference records

```bash
python -m experiments.weak_structure.paper --verify-only
python -m experiments.weak_structure.paper
python -m experiments.rebuild_paper
```

The reference exporter is separate from inference, tuning and evaluation. It verifies
all 101 eta and six beta curves from geometry-level scan records, recomputes paired
intervals from the selected per-observation CSV, and checks the reported Section 3.3
values. Figure 4 includes both panels and preserves beta=0.2. Generated LaTeX has
`\label{fig:weak_structure}` and matching text references. The original manuscript's
restoration-only compliance count remains 1,380; the separate weak-structure test
run audited 10,800 TV return arrays.

After a completed full rerun, export compact paper records using:

```bash
python -m experiments.weak_structure.paper --export-run outputs/weak_structure --records outputs/results/weak_structure
python -m experiments.weak_structure.paper --records outputs/results/weak_structure --output outputs/paper_rerun
```

Released records in `results/weak_structure/` preserve the executed config and model
hash. The 18 GB-scale full cache and all per-line/grid records are delivered separately
at `F:\Weak_Structure_20260917\weak_structure_complete.zip`; fresh runs create the same
artifact types locally. The compact GitHub records contain all scan parameters after
aggregation within each clean geometry, plus the selected per-observation rows.

## Commands

Run from the repository root. The available environment for this execution is
`/tmp/ruijie-ectv-venv/bin/python`; replace `python` below with that executable,
or use an environment containing NumPy, SciPy, scikit-image, pandas, Pillow,
Matplotlib and PyTorch. No scikit-learn dependency is needed: the AP implementation
uses the documented non-interpolated threshold definition and explicitly groups ties.
Versions are read from the running environment and recorded, not assumed from pins.

```bash
python -m unittest discover -s experiments/weak_structure/tests -v
python -m experiments.weak_structure prepare --split development
python -m experiments.weak_structure evaluate --split development
python -m experiments.weak_structure report --split development
python -m experiments.weak_structure prepare --split validation
python -m experiments.weak_structure tune
python -m experiments.weak_structure prepare --split test
python -m experiments.weak_structure evaluate
python -m experiments.weak_structure report
```

Every command accepts `--config /path/to/config.json`. Set `model_path` to the
actual **color DRUNet** `.pth` file. The default config uses the repository-relative
`data/models/drunet_color.pth`. The original execution path remains recorded in
`results/weak_structure/environment.json` and its frozen run configuration. A missing weight file
raises an error; there is no alternative denoiser or automatic model download.
Set `output_dir` for a separate run. Relative paths resolve from the repository root.
The reused paper inference supports CPU; unsupported device settings raise an error.
`workers` controls geometry/inference processes; `scoring_workers` controls scoring
processes; `inference_threads` controls threads in each model process. The CLI limits
BLAS/OpenMP threads to one to avoid oversubscription.

Commands resume completed case caches. A run locks its full config at creation;
changing configuration requires a new output directory. `tune` never overwrites
an existing selected file and refuses tuning after test preparation. `evaluate`
imports no tuning module and requires a compatible frozen selection file. Development
uses its own preassigned frozen eta/beta and contributes no formal statistics.

## Frozen design

- 256×256 synthetic grayscale, background U[0.35,0.65], 3–6 random quadratic
  Bézier lines, widths U[1,3] px, contrasts U[0.03,0.12], random position/orientation,
  lengths U[70,190] px, curvature fractions U[-0.35,0.35]. Each line is rasterized
  at 8× resolution; nonoverlapping 8×8 block means give area coverage. Intensities
  add line contrasts and clip to [0,1]; the maximum per-line coverage defines
  the union coverage used for the binary structure mask at threshold 0.5.
- Independent seed namespaces for geometry, noise and bootstrap. Development: 2
  geometries; validation: 40; test: 100. Each has 3 independent repeats at both
  sigma=15/255 and 25/255. The grayscale clean image is copied to RGB; Gaussian
  noise is independent in all three channels, clipped and stored float32.
- All a∈{1,1.5,2} use the identical observation. DRUNet receives noise_map=a*sigma.
  Proposals are cached once and shared by every method.
- AP and background/contrast metrics use float64 arithmetic RGB mean with no
  image or detector-response normalization. Sato is fixed to sigmas=(1,2,3),
  black_ridges=False, mode='reflect', cval=0. The installed signature is recorded.
  AP is calculated independently on every image with the same 12-pixel boundary
  excluded. Intersections remain in AP targets; contrast-only exclusions do not
  remove AP pixels. Undefined geometric AP/background cases stop with a saved
  reason, without replacement or model-dependent filtering.
- Line cores: coverage≥0.75. Sides: 3–6 px from that line's nonzero geometric
  support. Both exclude the outer 5% of curve parameters, the scoring boundary,
  and points ≤8 px from other lines. A line needs ≥20 core and ≥40 side pixels;
  otherwise contrast values are NA and the reason/counts are recorded. The image
  remains in AP and background statistics. Background is >10 px from all supports,
  with at least 1000 pixels. Per-line errors reference the clean image evaluated
  on the exact same masks, and the image contrast RMSE averages valid lines equally.
- Fixed blends: eta=0,0.01,…,1, unconstrained by TV. TV: beta∈{0.2,0.4,0.6,0.8,0.9,0.95}.
  Select one eta and one beta by validation macro AP: equal repeats, geometries and
  sigma/a conditions. Ties within 1e-12 choose the smaller parameter. Exact eta=0/1
  returns its endpoint. No sigma/a-specific deployment selection.
- Controller interface is only `control(f,v,beta)`. Budget beta*(TV(f)-1e-8),
  28 bisection steps, float32 endpoints/actual returned arrays, float64 RGB
  vector-valued TV, zero outward differences, audit tolerance 1e-12. Arrays are
  saved and reloaded for an independent channel-wise TV accumulation; the audit
  never reconstructs them from eta. Methods without a TV constraint record budget
  fields as NA, not as failed constraints.
- A clean-reference macro AP below 0.7 flags task/detector mismatch. Development
  diagnostics precede formal validation; no detector change is allowed after the
  formal freeze. Clean/masks are used only for scoring and validation selection,
  never for proposal generation, blending or TV control.
- Primary: selected TV minus unchanged proposal overall macro AP. Also report
  selected-TV background RMSE ratio relative to noisy. First aggregate within
  geometry; then perform paired geometry bootstrap, 10,000 draws, percentile
  95% CI. All repeats/sigma/a of a geometry form one cluster. Fixed comparisons,
  strata and scans are secondary/exploratory. The predeclared ±0.01 AP margin
  is used to distinguish a CI contained in an equivalence region from an
  inconclusive interval that merely overlaps zero.
- Examples are `test_000`, `test_001`, `test_002`, sigma25/repeat0/a2, assigned before
  generation. Images share [0,1]; responses share display range [0,0.06]. Display
  clipping never affects AP. Complete contact sheets include all 1,800 test
  geometry/noise/sigma/a cases. Beta=0.2 is shown separately, including its image,
  response, table rows and paired AP intervals.

## Output layout and size estimate

```text
outputs/weak_structure/
  run_config.json, environment.json, manifest.json
  development_operating_points.json
  selected_operating_points.json          written only after validation
  cache/{development,validation,test}/
    geometry/<id>.npz, <id>.json           clean, coverage, per-line masks/reasons
    proposals/<id>_s<sigma>_r<repeat>.npz   one observation plus all three a proposals
  arrays/<split>/<case>.npz               actual six-beta RGB outputs, eta; selected fixed
  responses/<split>/<case>.npz            raw float64 responses for displayed methods
  scores/<split>/<case>.csv               complete 101 eta + six beta + endpoints
  scores/<split>/<case>.lines.csv         every line, including invalidity records
  <split>_per_sample.csv, <split>_per_line.csv
  validation_selection_scores.csv
  report/
    report.md, interpretation.json
    main_table.csv, main_table_with_paired_ci.csv
    paired_comparisons.csv, geometry_paired_differences.csv, geometry_means.csv
    selected_per_sample.csv
    {validation,test}_tradeoff_curve.csv
    {validation,test}_tradeoff.{png,pdf}, *_tradeoff_strata.png
    {validation,test}_robustness.{png,pdf}
    example_<preassigned-id>.png
    contact_sheet/, contact_sheet_index.csv, all_test_examples_contact_sheet.pdf
```

Expected full-run storage is roughly 20–25 GiB depending on compression, including
all beta return arrays. All fixed-grid metrics are retained, but intermediate fixed
arrays are not permanently saved (saving every eta array would exceed 180 GiB).
The selected fixed arrays and raw responses are saved for development and test.
Validation candidate arrays for all beta values are saved; validation selection
does not need selected fixed-image caches. All original main-paper caches stay intact.

## Metric references

[Sato documentation](https://scikit-image.org/docs/0.26.x/api/skimage.filters.html#skimage.filters.sato)
specifies the detector parameters. [Average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
specifies the non-interpolated sum of recall increments weighted by precision;
unit tests compare the implementation against direct threshold enumeration and
verify invariance under strictly increasing affine score transforms.
