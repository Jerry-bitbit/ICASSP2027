# Auditable Denoising Control via Observation-Relative Total Variation Budgets

Reproducibility code for the manuscript by **Ruijie Xing**, prepared for submission to **ICASSP 2027**. This release matches **`ICASSP2027_final (1).pdf`**, including Section 3.3 and Figure 4 on weak-structure detection. The supplied manuscript is saved as [paper/manuscript.pdf](paper/manuscript.pdf); its identity and figure mapping are recorded in [configs/paper_revision.json](configs/paper_revision.json).

Given a noisy observation `f` and a denoising proposal `v`, the controller finds the largest feasible proposal weight along `u = f + eta * (v - f)`. The observation-relative budget is

$$
R(u;f)=\max\{0,V_\epsilon(f)-V_\epsilon(u)\},\qquad
B_\beta(f)=\beta[V_\epsilon(f)-\epsilon],\qquad \epsilon=10^{-8}.
$$

A feasible proposal passes through unchanged. Otherwise, the controller bisects the observation–proposal segment, retaining the last accepted **float32 array**. TV is evaluated in float64, with absolute acceptance tolerance `1e-12`. This controls aggregate net TV removal; it does not guarantee preservation of local detail or maximize reference PSNR.

## Quick start

Use Python 3.11 or newer and NumPy 2.x. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experiments.demo
python -m unittest discover -s tests
python -m unittest discover -s experiments/weak_structure/tests -v
python -m experiments.rebuild_paper
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The demo uses synthetic arrays. The figure/table command reads the included reference records and the fixed Kodak example, checks the numerical claims, and rebuilds **Tables 1–3 and Figures 1–5** under `outputs/paper/`. It also writes the Figure 4 panels as PNG, regenerated weak-structure statistics, and LaTeX insertion fragments with figure references. These commands need no external dataset, model weights, GPU, or PyTorch. Prebuilt materials are in **`paper/generated/`**. The original boundary-localization plot remains available as `controllers.pdf`, an additional diagnostic rather than Figure 4 in this revision.

To check the included records without generating figures:

```bash
python -m experiments.verify_results
```

The recorded CPU analysis environment is Python **3.14.4**, NumPy **2.5.3**, pandas **3.0.5**, SciPy **1.18.1**, scikit-image **0.26.0**, PyWavelets **1.10.0**, Pillow **12.3.0**, and Matplotlib **3.11.1**. Exact package pins for that environment are in `requirements-paper.txt`; `requirements.txt` supports a less restrictive installation. DRUNet inference additionally needs PyTorch; the packaged inference path was checked with **PyTorch 2.14.0+cpu**.

## Repository contents

| Path | Purpose |
| --- | --- |
| `methods/ectv.py` | Original TV functional and `certify_candidate` implementation used for Tables 1–3. The ICASSP experiments always supply `budget_override`; the file also retains the earlier solver API. |
| `experiments/run_controller_comparison.py` | Fixed blending, halving, coarse grid, 8/12/28-step bisection, independent TV audit, and randomized timing. |
| `experiments/run_icassp_budget_audit.py` | Kodak at sigma = 25/255: validation selection and adaptive/fixed evaluation. |
| `experiments/run_icassp_polyu_audit.py` | PolyU proposal noise-map selection and adaptive/fixed evaluation. |
| `experiments/run_icassp_noise_audit.py` | Kodak at sigma = 15/255 and 50/255. |
| `experiments/run_icassp_proposal_audit.py` | DRUNet, BayesShrink, and non-local means proposal comparison. |
| `experiments/run_drunet_subset.py` | Original DRUNet inference, noise generation, image loading, and metrics used by the experiment scripts. |
| `experiments/prepare_data.py` | Dataset download and exact Kodak preprocessing. |
| `experiments/finalize_results.py` | Controller summaries, paired bootstrap comparisons, precision statistics, and compliance checks. |
| `experiments/rebuild_paper.py` | Rebuild revised Tables 1–3, Figures 1–5, statistics and LaTeX fragments. |
| `experiments/paper_assets/` | Technical overview and the fixed kodim23 visual example, using actual controller returns. |
| `experiments/prepare_figure_inputs.py` | Extract the fixed kodim23 inputs from a completed Kodak rerun. |
| `experiments/weak_structure/` | Independent generation, DRUNet mismatch, validation selection, evaluation, auditing and reporting for Section 3.3. |
| `results/weak_structure/` | Frozen selections, selected per-sample results, complete scans by geometry, manifest and actual runtime metadata. |
| `results/figure_inputs/` | One cached kodim23 example for Figures 1 and 3. |
| `paper/` | Supplied revised manuscript and regenerated figures, tables, statistics and LaTeX. |
| `configs/` | Exact image names, validation/test splits, and recorded protocol settings. |
| `results/` | Reference per-image records and summaries for the manuscript. |
| `results/controllers/` | Controller outputs, validation grid, selected coefficients, all timing repeats, and runtime metadata. |
| `third_party/kair/` | Minimal DRUNet network definitions and their upstream license. |
| `tests/` | Tests for budget compliance and unchanged acceptance of feasible proposals. |

The table experiments originate from `ECTV_code-main`; the controller comparison and precision study originate from `TV_CONTROLLER_BASELINES_20260912`. The plotting code is adapted from the final manuscript's figure generator. Paths now resolve within this repository, and the PolyU split is loaded directly from its manifest. Original reference results are under `results/`; reruns write to `outputs/`.

## Data and DRUNet weights

The external inputs are:

| Material | Source | Local location |
| --- | --- | --- |
| Kodak24 | [Kodak image suite](https://r0k.us/graphics/kodak/) | `data/raw/kodak24/kodim01.png` through `kodim24.png` |
| PolyU noisy/reference crops | [Official dataset repository](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) | `data/raw/PolyU-Real-World-Noisy-Images-Dataset-master/CroppedImages/` |
| Color DRUNet weights | [DPIR project](https://github.com/cszn/DPIR), [official weight file](https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth) | `data/models/drunet_color.pth` |

Install PyTorch and prepare all inputs:

```bash
python -m pip install torch
python -m experiments.prepare_data --dataset all --download --weights
```

If the images and weights are already in the locations above, omit `--download`. The script checks all 29 required PolyU pairs and produces the processed Kodak PNGs. Individual dataset preparation is available with `--dataset kodak` or `--dataset polyu`.

The GitHub package includes the single cached kodim23 example used in Figures 1 and 3. Full image datasets, DRUNet weights and large run caches are supplied/prepared separately. The preparation script obtains the restoration inputs from the sources above; their own distribution terms apply. The recorded weak-structure DRUNet SHA256 is `479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4`.

### Exact evaluation protocol

- **Kodak:** images 01–08 are validation; 09–24 are test. Convert to RGB, resize to maximum side 256 using `skimage.transform.resize(..., anti_aliasing=True)`, round to 8-bit PNG, and reload to float32 `[0,1]`. Generate Gaussian noise with `np.random.default_rng(0)` **reinitialized for each image**, then clip to `[0,1]`. Use sigma = 15/255, 25/255, or 50/255 and the matching DRUNet noise map.
- **PolyU:** use the exact 512 × 512 crops listed in `configs/icassp_polyu_budget_manifest.json`, without resizing. The filename-based deduplication leaves 8 validation and 21 test groups. These groups do not establish verified scene-level independence. Select the DRUNet noise-map level by mean validation PSNR from `{1,2,3,5,7,10,15,25}/255`; the recorded selection is **15/255**. The reference cache historically contains an extra crop from an existing filename group; it is removed before evaluation.
- **DRUNet:** color `UNetRes`, channels `[64,128,256,512]`, four residual blocks per scale, stride-convolution downsampling, transposed-convolution upsampling, and `bias=False`. Add a spatially constant noise channel, reflect-pad to multiples of 8 if needed, run in evaluation mode on CPU, crop back, and clip the output to `[0,1]`. No retraining is required.
- **TV:** forward differences; zero outward differences; vector-valued norm over both spatial directions and RGB channels; average over `H × W`; smoothing epsilon `1e-8`.
- **Main budgets:** `{0.2,0.4,0.6,0.8}`. Table-generation experiments also retain the historical `0.05` and `0.10` rows; `0.05` supplies the separate small-budget retention result in the paper.
- **Fixed blending:** choose one coefficient per dataset/noise/budget by validation PSNR over `{0,0.005,...,1}`, requiring every validation image to satisfy its budget. Apply it to the test images with no fallback.
- **Per-image baselines:** halving tests `1,1/2,...`; coarse search tests `1,0.95,...,0`; bisection first tests the proposal and then performs 8, 12, or 28 midpoint evaluations when needed.
- **Proposal sensitivity:** BayesShrink uses `db1`, four levels, soft thresholding, and separate RGB channels. NLM uses `patch_size=5`, `patch_distance=6`, `h=0.85*(25/255)`, `fast_mode=True`, and `sigma=0`.
- **Metrics:** mean per-image RGB PSNR and SSIM, data range 1, no additional boundary cropping. SSIM uses scikit-image with `channel_axis=-1`, window size 7, and its default sample-covariance convention.
- **Intervals:** paired image bootstrap, 10,000 draws, percentile 95% intervals. Tables 1–3 use RNG seed 2027, advanced in the six-budget order; controller comparisons restart seed 20260912 for each comparison. PolyU resamples the selected crops. Comparisons are exploratory on fixed, previously inspected test sets.
- **Audit:** accept only when the returned array has `removed_TV - budget <= 1e-12`. A second implementation in `tv_audit` independently accumulates gradient energy. Compliance is checked before file quantization; audit an exported/reloaded image separately if quantization changes it.

## Weak-structure detection: Section 3.3 and Figure 4

The experiment measures whether observation-relative TV control improves weak-line detection under DRUNet strength mismatch while retaining noise suppression relative to the noisy observation. It uses 40 validation and 100 test clean geometries, three independent RGB noise repeats at each of sigma = 15/255 and 25/255, and all noise-map multipliers a = 1, 1.5 and 2. A separate two-geometry development split supplies the smoke test.

A fixed bright-line Sato detector uses scales (1,2,3), `black_ridges=False`, `mode=reflect` and `cval=0`. Masks come from antialiased geometry. AP is computed per image from raw detector responses; AP, background error and contrast use the arithmetic RGB mean. Geometry is the bootstrap unit, retaining all noise repeats and conditions together in each of 10,000 resamples.

Validation macro AP across all conditions selects **eta = 0.76** for the unconstrained fixed blend and **beta = 0.8** for TV control. Test macro AP is **0.6070** for the proposal, **0.6501** for TV and **0.6502** for fixed blending. TV minus proposal is **0.0431**, paired 95% CI **[0.0397, 0.0467]**; selected TV/fixed background RMSE ratios are **0.1975/0.2410**. The full scans retain the normal-condition decrease and beta = 0.2 results.

Rebuild or verify this part alone, using included records:

```bash
python -m experiments.weak_structure.paper
python -m experiments.weak_structure.paper --verify-only
```

These commands read the frozen selection; they do not tune. They recalculate geometry-cluster intervals from selected per-sample records and check all scan curves against the included per-geometry records. `paper_values.json` contains the rounded values printed in the supplied PDF and is checked against the calculated values.

For a complete new run, install PyTorch and place the actual color DRUNet weights at `data/models/drunet_color.pth`, or set `model_path` in `experiments/weak_structure/config.json`. The default config uses repository-relative paths and CPU inference. Missing weights produce an explicit error. Then run:

```bash
python -m experiments.weak_structure prepare --split development
python -m experiments.weak_structure evaluate --split development
python -m experiments.weak_structure report --split development
python -m experiments.weak_structure prepare --split validation
python -m experiments.weak_structure tune
python -m experiments.weak_structure prepare --split test
python -m experiments.weak_structure evaluate
python -m experiments.weak_structure report
python -m experiments.weak_structure.paper --export-run outputs/weak_structure --records outputs/results/weak_structure
```

Each generation/evaluation/report command accepts `--config PATH`. The full run writes its own locked config and actual environment to `outputs/weak_structure/`; the original frozen records in `results/weak_structure/` retain the original execution paths. The detailed mask rules, metrics, audit, selection isolation, seeds and cache layout are in [experiments/weak_structure/README.md](experiments/weak_structure/README.md).

The GitHub package contains selected **per-observation** rows and **all 101 eta / six beta scan points aggregated by geometry**, sufficient to regenerate the paper results. The full execution archive remains at `F:\Weak_Structure_20260917\weak_structure_complete.zip` in the supplied local delivery. It includes all per-sample/per-line scans, shared noisy/proposal caches, actual TV arrays, raw detector responses, and the 150-page contact sheet covering all 1,800 test cases. Its executed experiment metadata is preserved; the code in this GitHub release adds portable paths and revised-paper exports. A new full run regenerates those caches locally.

## Rerun the restoration experiments

Run these commands **from the repository root**, after preparing the data and installing PyTorch:

```bash
python -m experiments.run_icassp_budget_audit
python -m experiments.run_icassp_polyu_audit
python -m experiments.run_icassp_noise_audit
python -m experiments.run_icassp_proposal_audit
```

The first three commands create or reuse these caches:

```text
data/cache/icassp_drunet_sigma25.npz
data/cache/icassp_polyu_drunet.npz
data/cache/icassp_drunet_sigma15.npz
data/cache/icassp_drunet_sigma50.npz
```

Each cache stores `names` and `<image>_clean`, `<image>_noisy`, `<image>_proposal`, and `<image>_runtime` arrays. The PolyU cache also stores its selected noise level and validation scores. When all caches exist, the analysis commands run without importing PyTorch. `--regenerate` forces proposal regeneration for the first two commands; move the sigma15/sigma50 caches aside to regenerate the noise-sensitivity proposals.

Then run the controller comparison, without other compute-heavy jobs running alongside it:

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python -m experiments.run_controller_comparison
python -m experiments.finalize_results
python -m experiments.prepare_figure_inputs
python -m experiments.rebuild_paper --results outputs/results --weak-results results/weak_structure --output outputs/paper_rerun
```

The command above combines the restoration rerun with the frozen weak-structure reference. After also running and exporting the new weak-structure experiment, use `--weak-results outputs/results/weak_structure` to build entirely from the rerun records. The verification reports any differences from the rounded numerical values in the supplied paper.

PowerShell thread settings:

```powershell
$env:OPENBLAS_NUM_THREADS = "4"
$env:OMP_NUM_THREADS = "4"
```

Timing includes budget construction, control, and a separate float64 audit. It excludes proposal generation and quality scoring. Each dataset/controller receives one warm-up, followed by three randomly ordered calls per image/budget/controller condition; a condition's time is its median, and reported times average those medians across images. No TV values are cached across calls. Absolute runtimes depend on the machine; the included timing CSVs preserve the measurements reported in the manuscript.

To reuse data elsewhere, set `TV_DATA_DIR` to a directory containing `raw/`, `processed/`, `models/`, and/or `cache/`. Set `TV_OUTPUT_DIR` to change the destination for rerun results. Relative paths are resolved from the current working directory. For example:

```bash
TV_DATA_DIR=/path/to/data TV_OUTPUT_DIR=/path/to/run python -m experiments.run_icassp_budget_audit
```

## Paper-to-code map and expected results

| Paper item | Experimental records | Reproduction |
| --- | --- | --- |
| Algorithm 1 | `methods/ectv.py::certify_candidate` | `python -m experiments.demo` |
| Table 1: Kodak/PolyU | `icassp_budget_{audit,summary}.csv`, `icassp_polyu_budget_{audit,summary}.csv` | Budget and PolyU scripts |
| Table 2: noise sensitivity | `icassp_kodak_sigma{15,50}_{audit,summary}.csv` plus sigma25 records | Noise script |
| Table 3: proposal sensitivity | `icassp_proposal_summary.csv`, `wavelet_per_image.csv`, `nlm_per_image.csv` | Proposal script |
| Figure 1: controller overview | `figure_inputs/overview_sample.npz`, `overview_row.csv` | `paper_assets/overview.py` → `ICASSP_Overview_Technical.pdf` |
| Figure 2: segment geometry | Exact two-pixel RGB example | `rebuild_paper` → `geometry.pdf` |
| Figure 3: budget-controlled kodim23 | `figure_inputs/overview_sample.npz`, `overview_row.csv` | `paper_assets/visual.py` → `visual_single_column.pdf` |
| Figure 4 / Section 3.3: weak-structure detection | `weak_structure/selected_per_sample.csv`, `*_scan_geometry_means.csv`, frozen selection | `weak_structure.paper` → `test_robustness.pdf`, `test_tradeoff.pdf` |
| Figure 5: precision/cost | `controllers/precision_summary.csv` and `timing_repeats.csv` | Controller script + `finalize_results` → `precision.pdf` |
| Section 3.4: boundary-localization gains | `controllers/paired_comparisons.csv` | Additional `controllers.pdf` diagnostic |
| 1,380-output compliance result | `controllers/per_image.csv`, `evaluator_checks.csv` | Controller script + `finalize_results` |

Paths in the table are relative to `results/` for the reference run and `outputs/results/` for a rerun. `python -m experiments.rebuild_paper` generates the revised tables and figures together. Figures 1 and 3 recompute, save and independently audit the actual float32 controller returns. `outputs/paper/latex/figures.tex` supplies all five figure environments, including the side-by-side Figure 4 layout in the supplied PDF; `weak_structure_section.tex` includes the corresponding figure references. Figures 1 and 3 use the fixed kodim23 example, with no image selection during rebuilding.

Representative reference values:

| Setting | Adaptive PSNR | Fixed PSNR | Adaptive − fixed |
| --- | ---: | ---: | ---: |
| Kodak, sigma=25/255, beta=0.6 | 28.35 dB | 27.56 dB | +0.79 dB |
| PolyU, beta=0.2 | 37.13 dB | 36.60 dB | +0.53 dB |
| PolyU, beta=0.8 | 37.96 dB | 38.04 dB | −0.07 dB |

Differences are calculated before rounding. The controller study has **276 conditions**: `16 × 3 × 4` Kodak conditions and `21 × 4` PolyU conditions. Five per-image methods yield **1,380 outputs**, all with independent budget residuals at most `1e-12`. The 12-step maximum absolute PSNR difference from 28 steps is **0.004411916 dB**; the 8-step maximum is **0.068085304 dB**.

Fixed blending has zero test violations in Table 1. At Kodak sigma=50/255, it violates one of 16 test budgets at each of beta=0.4 and 0.8. Fixed wavelet blending also violates one of 16 budgets at each reported setting. The released records retain these outcomes. Maximal feasible retention can reduce PSNR relative to fixed blending, as the PolyU beta=0.8 row illustrates.

## Use the controller on your own arrays

```python
import numpy as np
from methods.ectv import certify_candidate, tv_energy_float64
from experiments.run_controller_comparison import tv_audit

# f and v must have matching spatial/channel dimensions.
f = np.clip(noisy_rgb, 0, 1).astype(np.float32)
v = np.clip(denoised_rgb, 0, 1).astype(np.float32)
beta = 0.6
budget = beta * max(tv_energy_float64(f) - 1e-8, 0.0)
result = certify_candidate(
    f, v, budget_override=budget,
    bisection_steps=28, feasibility_tolerance=1e-12,
)
u = result.image  # retain this array rather than rebuilding it from eta
residual = max(0.0, tv_audit(f) - tv_audit(u)) - budget
assert residual <= 1e-12
print(result.eta)
```

## Attribution and citation

The vendored network definitions are from [Kai Zhang's KAIR project](https://github.com/cszn/KAIR); their MIT license is included in `third_party/kair/LICENSE`. The pretrained denoiser is described in [Plug-and-Play Image Restoration with Deep Denoiser Prior](https://github.com/cszn/DPIR). Please also cite the original datasets when using them.

```bibtex
@misc{xing2026auditable,
  author = {Ruijie Xing},
  title = {Auditable Denoising Control via Observation-Relative Total Variation Budgets},
  year = {2026},
  note = {Manuscript prepared for submission to ICASSP 2027}
}
```

An open-source license for the original project code has not yet been selected. The upstream license above applies to the vendored KAIR files.
