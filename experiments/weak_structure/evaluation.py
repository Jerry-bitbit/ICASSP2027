"""Evaluation consumes a frozen file and has no tuning dependency."""
from .pipeline import open_run, require_frozen, write_json
from .scan import run_scan


def evaluate(config, split="test"):
    if split not in ("development", "test"):
        raise ValueError("Evaluation split must be development or test")
    root = open_run(config)
    frozen = require_frozen(config, split)
    selected = {key: frozen[key] for key in ("eta", "beta")}
    frame = run_scan(config, split, selected)
    clean_ap = float(frame[frame.method == "clean"].groupby("geometry").ap.mean().mean())
    tv = frame[frame.method == "tv_scan"]
    write_json(root / f"{split}_complete.json", {
        "complete": True, "geometries": int(frame.geometry.nunique()),
        "per_sample_rows": len(frame), "controller_arrays_audited": len(tv),
        "maximum_budget_residual": float(tv.budget_residual.max()),
        "clean_macro_ap": clean_ap,
        "detector_task_mismatch_flag": clean_ap < config["selection"]["clean_ap_warning_threshold"],
        "selected": selected, "statistical_use": "smoke only" if split == "development" else "held-out test"})
    print(f"{split} complete: clean AP={clean_ap:.6f}; {len(tv)} returned arrays audited", flush=True)
