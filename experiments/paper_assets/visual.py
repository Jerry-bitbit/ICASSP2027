"""Render the existing kodim23 comparison as a native 86 mm single-column figure.

Called by experiments.rebuild_paper.
Requires NumPy and Matplotlib. Inputs are included in data/.
The sample, observation, proposal, weights, and display range are unchanged.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods.ectv import certify_candidate, tv_energy_float64
from experiments.weak_structure.audit import audit_output
WIDTH_MM = 86.0
FONT_PT = 9.2
MARGIN_MM = 0.5
COLUMN_GAP_MM = 2.0
ROW_GAP_MM = 1.0
LABEL_HEIGHT_MM = 4.0


def build(records, output):
    records = Path(records)
    output = Path(output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "arrays").mkdir(parents=True, exist_ok=True)
    with np.load(records / "figure_inputs/overview_sample.npz") as cache:
        reference, observation, proposal = [
            cache[f"kodim23_{key}"] for key in ("clean", "noisy", "proposal")
        ]
    with (records / "figure_inputs/overview_row.csv").open(newline="") as f:
        weights = {float(row["beta"]): float(row["eta"])
                   for row in csv.DictReader(f)}
    panels = [("Reference", reference), ("Observation", observation)]
    for beta in (0.2, 0.6, 0.8):
        eta = weights[beta]
        result = certify_candidate(observation, proposal,
            budget_override=beta*(tv_energy_float64(observation)-1e-8),
            bisection_steps=28, feasibility_tolerance=1e-12)
        np.testing.assert_allclose(result.eta, eta, rtol=0, atol=1e-12)
        path = output / f"arrays/visual_beta_{beta:g}.npz"
        np.savez_compressed(path, output=result.image)
        with np.load(path, allow_pickle=False) as saved:
            returned = saved['output']
        assert audit_output(observation, returned, beta)['audit_pass']
        panels.append((rf"$\beta={beta:.1f},\ \eta={eta:.2f}$", returned))
    panels.append(("DRUNet proposal", proposal))

    panel_width = (WIDTH_MM - 2*MARGIN_MM - COLUMN_GAP_MM) / 2
    image_height = panel_width * observation.shape[0] / observation.shape[1]
    row_height = image_height + LABEL_HEIGHT_MM
    height_mm = 2*MARGIN_MM + 3*row_height + 2*ROW_GAP_MM
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig = plt.figure(figsize=(WIDTH_MM/25.4, height_mm/25.4), facecolor="white")
    labels = []
    for i, (label, im) in enumerate(panels):
        row, col = divmod(i, 2)
        x = MARGIN_MM + col * (panel_width + COLUMN_GAP_MM)
        row_top = height_mm - MARGIN_MM - row * (row_height + ROW_GAP_MM)
        image_bottom = row_top - LABEL_HEIGHT_MM - image_height
        ax = fig.add_axes([x/WIDTH_MM, image_bottom/height_mm,
                           panel_width/WIDTH_MM, image_height/height_mm])
        ax.imshow(im, interpolation="none", vmin=0, vmax=1)
        ax.set_axis_off()
        labels.append(fig.text((x+panel_width/2)/WIDTH_MM,
                               (row_top-LABEL_HEIGHT_MM/2)/height_mm,
                               label, ha="center", va="center", fontsize=FONT_PT))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        box = label.get_window_extent(renderer)
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)
        assert label.get_fontsize() >= 9
    # Keep the declared physical width: do not use bbox_inches="tight".
    fig.savefig(output / "figures/visual_single_column.pdf", facecolor="white")
    fig.savefig(output / "figures/visual_single_column.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Two columns, three rows; {WIDTH_MM:g} x {height_mm:.2f} mm; {FONT_PT:g} pt labels.")
    print("Use \\includegraphics[width=\\columnwidth]{figures/visual_single_column.pdf}")


