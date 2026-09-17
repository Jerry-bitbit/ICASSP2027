"""Score every predeclared operating point, without selecting one."""
from pathlib import Path
import numpy as np
import pandas as pd
from .audit import audit_output
from .controller import control, blend
from .metrics import score, detect
from .pipeline import case_id, jobs, load_npz, parallel, read_json, save_npz, write_json


def parameter_grid(config):
    grid = config["eta_grid"]
    return [i / grid["denominator"] for i in range(grid["start"], grid["stop"] + 1)]


def _scan_case(task):
    config, split, index, si, repeat, selected = task
    sigma255 = config["sigma255"][si]
    root = Path(config["output_dir"])
    name = case_id(split, index, sigma255, repeat)
    scores = root / "scores" / split
    arrays_dir = root / "arrays" / split
    done = scores / f"{name}.complete.json"
    if done.exists():
        return name + " cached"
    geom = load_npz(root / "cache" / split / "geometry" / f"{split}_{index:03d}.npz")
    cache = load_npz(root / "cache" / split / "proposals" / f"{name}.npz")
    f = cache["noisy"]
    # First persist actual returned arrays, then independently reopen for audit.
    controlled, etas = [], []
    for v in cache["proposals"]:
        outputs, weights = [], []
        for beta in config["betas"]:
            u, info = control(f, v, beta)
            outputs.append(u); weights.append(info["eta"])
        controlled.append(outputs); etas.append(weights)
    output_path = arrays_dir / f"{name}.npz"
    payload = {"tv_outputs": np.array(controlled, np.float32), "tv_etas": np.array(etas, np.float64)}
    if selected:
        payload["fixed_outputs"] = np.stack([blend(f, v, selected["eta"]) for v in cache["proposals"]])
    save_npz(output_path, **payload)
    del controlled, outputs, payload
    returned = load_npz(output_path)
    rows, line_rows = [], []
    responses = {}
    # Endpoints are identical across a; reuse their exact detector computation.
    common = {"noisy": score(f, f, geom, config["detector"]),
              "clean": score(geom["clean"], f, geom, config["detector"])}
    responses["noisy"] = common["noisy"][2]
    responses["clean"] = common["clean"][2]

    def append(method, parameter, eta, u, ai, calculated=None, beta=None):
        a = config["strengths"][ai]
        metrics, lines, response = calculated if calculated is not None else score(u, f, geom, config["detector"])
        base = {"geometry": f"{split}_{index:03d}", "case": name, "split": split,
                "sigma255": sigma255, "repeat": repeat, "a": a,
                "method": method, "parameter": parameter, "eta": eta}
        if beta is not None:
            audit = audit_output(f, u, beta)
            if not audit["audit_pass"]:
                # Preserve the failing array and diagnostic before surfacing error.
                write_json(scores / f"{name}.audit_failure.json", base | audit)
                raise RuntimeError(f"Returned array failed independent audit: {name} a={a} beta={beta}")
            metrics = metrics | audit
        else:
            metrics = metrics | {"budget": np.nan, "budget_residual": np.nan,
                                  "budget_violation": np.nan, "audit_pass": np.nan}
        rows.append(base | metrics)
        line_rows.extend(base | line for line in lines)
        return response

    for ai, v in enumerate(cache["proposals"]):
        proposal = score(v, f, geom, config["detector"])
        responses[f"proposal_{ai}"] = proposal[2]
        for method, eta, u in [("noisy", 0., f), ("proposal", 1., v), ("clean", np.nan, geom["clean"])]:
            append(method, np.nan, eta, u, ai, proposal if method == "proposal" else common[method])
        for eta in parameter_grid(config):
            u = blend(f, v, eta)
            calc = common["noisy"] if eta == 0 else proposal if eta == 1 else None
            response = append("fixed_scan", eta, eta, u, ai, calc)
            if selected and eta == selected["eta"]:
                responses[f"fixed_{ai}"] = response
        for bi, beta in enumerate(config["betas"]):
            u = returned["tv_outputs"][ai, bi]
            response = append("tv_scan", beta, float(returned["tv_etas"][ai, bi]), u, ai, beta=beta)
            if selected and beta == selected["beta"]:
                responses[f"tv_{ai}"] = response
            if beta == .2:
                responses[f"beta02_{ai}"] = response
    scores.mkdir(parents=True, exist_ok=True)
    # Numeric responses retain their scoring float64 representation.
    save_npz(root / "responses" / split / f"{name}.npz", **responses)
    pd.DataFrame(rows).to_csv(scores / f"{name}.csv", index=False)
    pd.DataFrame(line_rows).to_csv(scores / f"{name}.lines.csv", index=False)
    write_json(done, {"complete": True, "rows": len(rows), "selected": selected,
                      "array_audit": "actual returned float32 arrays saved and reopened before audit"})
    return name


def run_scan(config, split, selected=None):
    root = Path(config["output_dir"])
    if not (root / f"{split}_prepared.json").exists():
        raise RuntimeError(f"Run prepare --split {split} first")
    tasks = [task + (selected,) for task in jobs(config, split)]
    parallel(_scan_case, tasks, config["scoring_workers"], f"scan {split}")
    paths = [root / "scores" / split / f"{case_id(split, i, config['sigma255'][si], r)}.csv"
             for _, _, i, si, r in jobs(config, split)]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(tasks) * len(config["strengths"]) * (3 + len(parameter_grid(config)) + len(config["betas"]))
    if len(frame) != expected or frame.duplicated(["case", "a", "method", "parameter"]).any():
        raise RuntimeError("Incomplete or duplicated scan")
    if not np.isfinite(frame["ap"]).all():
        raise RuntimeError("Nonfinite AP found; no automatic sample removal")
    frame.to_csv(root / f"{split}_per_sample.csv", index=False)
    # Stream the potentially large per-line table instead of pooling it in RAM.
    with (root / f"{split}_per_line.csv").open("w") as output:
        for i, path in enumerate(paths):
            with path.with_suffix(".lines.csv").open() as source:
                header = next(source)
                if i == 0:
                    output.write(header)
                for line in source:
                    output.write(line)
    return frame


def macro_scores(frame):
    """Equal repeat -> geometry -> sigma/a means, independently of row order."""
    columns = ["method", "parameter"]
    geometry = frame.groupby(columns + ["sigma255", "a", "geometry"], dropna=False)["ap"].mean()
    conditions = geometry.groupby(level=columns + ["sigma255", "a"]).mean()
    return conditions.groupby(level=columns).mean().reset_index(name="macro_ap")
