"""Validation-only selection. This module is never imported by evaluate."""
import numpy as np
from datetime import datetime, timezone
from .pipeline import open_run, read_json, write_json
from .scan import run_scan, macro_scores


def select_parameter(scores, method, tolerance):
    candidates = scores[scores.method == method].sort_values("parameter")
    best = candidates.macro_ap.max()
    chosen = candidates[candidates.macro_ap >= best - tolerance].iloc[0]
    return float(chosen.parameter), float(chosen.macro_ap)


def tune(config):
    root = open_run(config)
    frozen_path = root / "selected_operating_points.json"
    if frozen_path.exists():
        print("Frozen operating points already exist; keeping them unchanged.", flush=True)
        return read_json(frozen_path)
    if not (root / "development_complete.json").exists():
        raise RuntimeError("Development smoke must complete before tuning")
    if (root / "test_prepared.json").exists() or (root / "test_per_sample.csv").exists():
        raise RuntimeError("Cannot tune after test data preparation/evaluation")
    frame = run_scan(config, "validation")
    scores = macro_scores(frame)
    scores.to_csv(root / "validation_selection_scores.csv", index=False)
    eta, eta_ap = select_parameter(scores, "fixed_scan", config["selection"]["tie_tolerance"])
    beta, beta_ap = select_parameter(scores, "tv_scan", config["selection"]["tie_tolerance"])
    clean_ap = float(frame[frame.method == "clean"].groupby("geometry").ap.mean().mean())
    frozen = {"frozen": True, "source_split": "validation", "eta": eta, "beta": beta,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "validation_fixed_macro_ap": eta_ap, "validation_tv_macro_ap": beta_ap,
              "validation_clean_macro_ap": clean_ap,
              "detector_task_mismatch_flag": clean_ap < config["selection"]["clean_ap_warning_threshold"],
              "config": config, "selection": config["selection"],
              "environment": read_json(root / "environment.json"),
              "validation_geometry_ids": sorted(frame.geometry.unique().tolist()),
              "validation_cases": int(frame.case.nunique()),
              "ground_truth_use": "Clean/masks used for validation AP selection and subsequent test scoring only; controller sees f,v,beta."}
    write_json(frozen_path, frozen)
    print(f"FROZEN shared eta={eta}, beta={beta}; validation clean AP={clean_ap:.6f}", flush=True)
    return frozen
