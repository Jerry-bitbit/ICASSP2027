"""Protocol locking, deterministic caches and shared I/O."""
import hashlib
import inspect
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import numpy as np
from skimage.filters import sato
from .geometry import generate, noise_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment(config):
    import torch
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": {name: metadata.version(name) for name in
                         ["numpy", "scipy", "scikit-image", "pandas", "matplotlib", "Pillow", "torch"]},
            "device": config["device"], "cuda_available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "inference_threads": config["inference_threads"], "workers": config["workers"],
            "scoring_workers": config["scoring_workers"],
            "seeds": config["seeds"], "sato_signature": str(inspect.signature(sato)),
            "sato_parameters": config["detector"], "model_path": config["model_path"],
            "model_sha256": sha256(config["model_path"]),
            "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in
                             [ROOT / "methods/ectv.py", ROOT / "experiments/run_drunet_subset.py"]},
            "deterministic_algorithms": True, "inference_precision": "float32, CPU; no autocast"}


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    if config["controller"] != {"epsilon": 1e-8, "steps": 28, "audit_tolerance": 1e-12}:
        raise ValueError("This experiment requires the original epsilon=1e-8, 28-step, 1e-12 controller")
    if config["detector"] != {"name": "skimage.filters.sato", "sigmas": [1,2,3], "black_ridges": False, "mode": "reflect", "cval": 0}:
        raise ValueError("This experiment fixes Sato sigmas=(1,2,3), bright ridges, reflect, cval=0")
    for key in ("model_path", "output_dir"):
        p = Path(config[key]).expanduser()
        config[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    return config


def open_run(config, create=False):
    root = Path(config["output_dir"])
    lock = root / "run_config.json"
    if lock.exists():
        if read_json(lock) != config:
            raise ValueError("Configuration differs from locked run. Use a new output_dir for a different protocol.")
        recorded = read_json(root / "environment.json")
        actual_versions = {name: metadata.version(name) for name in recorded["versions"]}
        if actual_versions != recorded["versions"] or str(inspect.signature(sato)) != recorded["sato_signature"]:
            raise RuntimeError("Installed software differs from this run's recorded environment; use a new output_dir")
    elif create:
        if not Path(config["model_path"]).is_file():
            raise FileNotFoundError(f"Missing color DRUNet weights: {config['model_path']}. Set model_path in config; no fallback denoiser.")
        if config["device"] != "cpu":
            raise ValueError("The reused paper inference is CPU-only. Set device=cpu; GPU requires an explicit separately verified adapter.")
        root.mkdir(parents=True, exist_ok=True)
        write_json(lock, config)
        write_json(root / "environment.json", environment(config))
        manifest = {"experiment": config["experiment"], "splits": config["splits"],
                    "sigma255": config["sigma255"], "strengths": config["strengths"],
                    "noise_repeats": config["noise_repeats"], "example_ids": config["visualization"]["example_ids"],
                    "geometries": [{"id": f"{split}_{i:03d}", "split": split, "index": i,
                                     "geometry_file": f"cache/{split}/geometry/{split}_{i:03d}.npz"}
                                   for split, size in config["splits"].items() for i in range(size)],
                    "exclusion_policy": "Only predeclared geometric line-region invalidity. No image or condition excluded by model results."}
        write_json(root / "manifest.json", manifest)
        write_json(root / "development_operating_points.json",
                   {"frozen": True, "source_split": "development", "selection": "preassigned smoke only",
                    "config": config, **config["development_operating_points"]})
    else:
        raise FileNotFoundError("Run prepare --split development first")
    return root


def case_id(split, index, sigma255, repeat):
    return f"{split}_{index:03d}_s{sigma255}_r{repeat}"


def jobs(config, split):
    return [(config, split, i, si, repeat) for i in range(config["splits"][split])
            for si in range(len(config["sigma255"])) for repeat in range(config["noise_repeats"])]


def parallel(function, tasks, workers, label):
    if workers == 1:
        for i, task in enumerate(tasks):
            result = function(task)
            print(f"{label}: {i+1}/{len(tasks)} {result}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function, task): task for task in tasks}
        for i, future in enumerate(as_completed(pending)):
            result = future.result()
            print(f"{label}: {i+1}/{len(tasks)} {result}", flush=True)


def _geometry_job(task):
    config, split, i = task
    base = Path(config["output_dir"]) / "cache" / split / "geometry"
    name = f"{split}_{i:03d}"
    if (base / f"{name}.json").exists() and (base / f"{name}.npz").exists():
        return name + " cached"
    arrays, info = generate(config, split, i)
    save_npz(base / f"{name}.npz", **arrays)
    write_json(base / f"{name}.json", info)
    if info["image_invalid_reasons"]:
        raise ValueError(f"Geometrically invalid image retained at {base / name}: {info['image_invalid_reasons']}; no automatic replacement")
    return name


_MODEL = None


def _prepare_job(task):
    global _MODEL
    config, split, i, si, repeat = task
    sigma255 = config["sigma255"][si]
    name = case_id(split, i, sigma255, repeat)
    root = Path(config["output_dir"])
    path = root / "cache" / split / "proposals" / f"{name}.npz"
    if path.exists():
        return name + " cached"
    if _MODEL is None:
        from experiments import run_drunet_subset as inference
        inference.DRUNET_WEIGHTS = Path(config["model_path"])
        model, torch, functional = inference.load_drunet()
        torch.manual_seed(config["seeds"]["torch"])
        torch.set_num_threads(config["inference_threads"])
        torch.use_deterministic_algorithms(True)
        _MODEL = model, torch, functional
    from experiments.run_drunet_subset import denoise_drunet
    clean = load_npz(root / "cache" / split / "geometry" / f"{split}_{i:03d}.npz")["clean"]
    seed = noise_seed(config, split, i, si, repeat)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    noise = rng.normal(0, sigma255 / 255, clean.shape).astype(np.float32)
    f = np.clip(clean + noise, 0, 1).astype(np.float32)
    proposals = [denoise_drunet(*_MODEL, f, a * sigma255 / 255) for a in config["strengths"]]
    save_npz(path, noisy=f, proposals=np.stack(proposals), noise_seed=np.array(seed, np.int64),
             strengths=np.array(config["strengths"]), sigma=np.array(sigma255 / 255))
    return name


def require_frozen(config, split):
    root = open_run(config)
    name = "development_operating_points.json" if split == "development" else "selected_operating_points.json"
    frozen = read_json(root / name)
    if not frozen.get("frozen") or frozen["config"] != config:
        raise ValueError("Missing or incompatible frozen operating points")
    if split != "development" and frozen["source_split"] != "validation":
        raise ValueError("Formal evaluation requires validation-selected operating points")
    return frozen


def prepare(config, split):
    root = open_run(config, create=True)
    if split == "validation" and not (root / "development_complete.json").exists():
        raise RuntimeError("Run and pass development smoke evaluation before formal validation")
    if split == "test":
        require_frozen(config, split)
    current = environment(config)
    initial = read_json(root / "environment.json")
    if current["model_sha256"] != initial["model_sha256"] or current["versions"] != initial["versions"]:
        raise RuntimeError("Model or software changed since run creation")
    parallel(_geometry_job, [(config, split, i) for i in range(config["splits"][split])], config["workers"], "geometry")
    infos = [read_json(p) for p in sorted((root / "cache").glob("*/geometry/*.json"))]
    if len({i["geometry_sha256"] for i in infos}) != len(infos):
        raise RuntimeError("Duplicate clean geometry detected; no automatic replacement")
    if any(i["image_invalid_reasons"] for i in infos):
        raise RuntimeError("Geometric invalidity recorded; inspect geometry metadata")
    manifest = read_json(root / "manifest.json")
    by_id = {x["id"]: x for x in infos}
    manifest["geometries"] = [entry | by_id.get(entry["id"], {}) for entry in manifest["geometries"]]
    manifest["observations"] = [
        {"case": case_id(part,i,config["sigma255"][si],r), "geometry": f"{part}_{i:03d}",
         "split": part, "sigma255": config["sigma255"][si], "repeat": r,
         "noise_seed": noise_seed(config,part,i,si,r),
         "proposal_multipliers": config["strengths"],
         "cache_file": f"cache/{part}/proposals/{case_id(part,i,config['sigma255'][si],r)}.npz"}
        for part in config["splits"] for _,_,i,si,r in jobs(config,part)]
    write_json(root / "manifest.json", manifest)
    parallel(_prepare_job, jobs(config, split), config["workers"], f"DRUNet {split}")
    write_json(root / f"{split}_prepared.json", {"complete": True, "cases": len(jobs(config, split)), "environment": current})
