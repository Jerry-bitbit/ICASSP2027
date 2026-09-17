import ast
import inspect
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.weak_structure.audit import audit_output, tv_reference
from experiments.weak_structure.controller import control, blend
from experiments.weak_structure.geometry import generate, geometry_seed, noise_seed
from experiments.weak_structure.metrics import average_precision, detect
from experiments.weak_structure.pipeline import load_config, save_npz, load_npz, require_frozen
from experiments.weak_structure.scan import macro_scores
from experiments.weak_structure.tuning import select_parameter
from experiments.weak_structure.statistics import bootstrap_mean, paired, METRICS
from methods.ectv import certify_candidate, tv_energy_float64


class AveragePrecisionTests(unittest.TestCase):
    def test_strictly_increasing_affine_invariance(self):
        rng = np.random.default_rng(431)
        y = rng.random(10000) > .8
        scores = rng.integers(-50, 51, 10000).astype(np.float64) / 8
        for a, b in [(2., 3.), (.125, -20.), (16., 100.)]:
            self.assertAlmostEqual(average_precision(y, scores), average_precision(y, a * scores + b), places=14)

    def test_ties_and_known_examples(self):
        self.assertAlmostEqual(average_precision([1, 0, 1], [3, 2, 1]), 5/6)
        self.assertAlmostEqual(average_precision([1, 0, 1], [1, 1, 1]), 2/3)
        self.assertAlmostEqual(average_precision([1, 0, 1], [2, 2, 1]), 7/12)
        self.assertEqual(average_precision([1, 0, 1], [3, 1, 2]), 1.)

    def test_matches_threshold_definition(self):
        rng = np.random.default_rng(93)
        for _ in range(20):
            y = rng.integers(0, 2, 30).astype(bool)
            s = rng.integers(0, 8, 30)
            recall, expected = 0., 0.
            for threshold in sorted(set(s), reverse=True):
                positive = s >= threshold
                current = np.sum(y & positive) / y.sum()
                expected += (current - recall) * np.mean(y[positive])
                recall = current
            self.assertAlmostEqual(average_precision(y, s), expected, places=14)


class ControllerTests(unittest.TestCase):
    def test_interface(self):
        self.assertEqual(list(inspect.signature(control).parameters), ["f", "v", "beta"])

    def test_reuses_paper_and_audits_saved_array(self):
        rng = np.random.default_rng(93)
        f = rng.random((17, 19, 3), dtype=np.float32)
        v = np.full_like(f, .5)
        with tempfile.TemporaryDirectory() as tmp:
            for beta in [.2, .4, .6, .8, .9, .95]:
                u, meta = control(f, v, beta)
                expected = certify_candidate(f, v, budget_override=beta*(tv_energy_float64(f)-1e-8), bisection_steps=28, feasibility_tolerance=1e-12)
                np.testing.assert_array_equal(u, expected.image)
                self.assertEqual(meta["eta"], expected.eta)
                save_npz(Path(tmp) / "actual.npz", output=u)
                stored = load_npz(Path(tmp) / "actual.npz")["output"]
                self.assertEqual(stored.dtype, np.float32)
                self.assertTrue(audit_output(f, stored, beta)["audit_pass"])
                self.assertFalse(audit_output(f, v, beta)["audit_pass"])

    def test_endpoints(self):
        f = .5 + .01 * np.random.default_rng(2).random((8, 8, 3), dtype=np.float32)
        v = np.random.default_rng(3).random(f.shape, dtype=np.float32)
        u, meta = control(f, v, .2)
        np.testing.assert_array_equal(u, v)
        self.assertEqual(meta["eta"], 1.)
        np.testing.assert_array_equal(blend(f, v, 0), f)
        np.testing.assert_array_equal(blend(f, v, 1), v)

    def test_zero_outward_boundary_scalar_reference(self):
        a = np.random.default_rng(7).random((3, 4, 3), dtype=np.float32)
        values = []
        for y in range(3):
            for x in range(4):
                squares = [1e-16]
                for c in range(3):
                    if y < 2: squares.append((float(a[y+1,x,c])-float(a[y,x,c]))**2)
                    if x < 3: squares.append((float(a[y,x+1,c])-float(a[y,x,c]))**2)
                values.append(math.sqrt(math.fsum(squares)))
        self.assertAlmostEqual(tv_reference(a), math.fsum(values)/12, places=15)
        self.assertAlmostEqual(tv_reference(a), tv_energy_float64(a), places=15)


class IsolationTests(unittest.TestCase):
    def test_geometry_masks_and_independent_seeds(self):
        cfg = load_config()
        arrays, info = generate(cfg, "development", 0)
        again, info2 = generate(cfg, "development", 0)
        np.testing.assert_array_equal(arrays["clean"], again["clean"])
        np.testing.assert_array_equal(arrays["structure_mask"], arrays["coverage"] >= .5)
        self.assertEqual(info["geometry_sha256"], info2["geometry_sha256"])
        self.assertFalse(arrays["valid_region"][:12].any())
        self.assertFalse((arrays["background_mask"] & arrays["structure_mask"]).any())
        self.assertFalse((arrays["core_masks"] & arrays["side_masks"]).any())
        self.assertNotEqual(geometry_seed(cfg, "test", 0), geometry_seed(cfg, "validation", 0))
        self.assertNotEqual(noise_seed(cfg, "test", 0, 0, 0), noise_seed(cfg, "test", 0, 0, 1))
        self.assertNotEqual(noise_seed(cfg, "test", 0, 0, 0)[0], geometry_seed(cfg, "test", 0)[0])
        response = detect(arrays["clean"], cfg["detector"])
        self.assertEqual(response.shape, (256, 256))

    def test_evaluate_has_no_tuning_import_or_call(self):
        from experiments.weak_structure import evaluation
        tree = ast.parse(inspect.getsource(evaluation))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn("tuning", node.module or "")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotEqual(node.func.id, "tune")
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output_dir"] = tmp
            with self.assertRaises(FileNotFoundError):
                evaluation.evaluate(cfg)

    def test_global_macro_selection_and_tie_rule(self):
        frame = pd.DataFrame([
            {"method": "fixed_scan", "parameter": eta, "geometry": geom,
             "sigma255": s, "a": a, "ap": .5 + eta * .1}
            for eta in [0., .5, 1.] for geom in ["v1", "v2"]
            for s in [15, 25] for a in [1., 1.5, 2.] for _ in range(3)])
        scores = macro_scores(frame)
        self.assertEqual(select_parameter(scores, "fixed_scan", 1e-12)[0], 1.)
        scores["macro_ap"] = .5
        self.assertEqual(select_parameter(scores, "fixed_scan", 1e-12)[0], 0.)


class StatisticsTests(unittest.TestCase):
    def test_repeats_do_not_become_clusters(self):
        rows = []
        for geom, improvement in [("g1", .1), ("g2", -.2), ("g3", .4)]:
            for s in [15,25]:
                for a in [1.,1.5,2.]:
                    for repeat in range(3):
                        for method, effect in [("proposal", 0.), ("tv", improvement)]:
                            rows.append({"geometry": geom,"sigma255":s,"a":a,"repeat":repeat,"method":method,
                                         **{metric:.5+effect for metric in METRICS}})
        config = load_config()
        result, differences = paired(pd.DataFrame(rows), "tv", "proposal", "ap", config)
        self.assertEqual(result["n_geometries"], 3)
        np.testing.assert_allclose(differences.to_numpy(), [.1,-.2,.4])
        self.assertAlmostEqual(result["difference"], .1)
        expected = bootstrap_mean(np.array([.1,-.2,.4]), config)
        self.assertAlmostEqual(result["ci_low"], expected[1])
        self.assertAlmostEqual(result["ci_high"], expected[2])

    def test_missing_pair_is_an_error(self):
        data = pd.DataFrame([{"geometry":g,"method":m,"sigma255":15,"a":1.,**{k:1. for k in METRICS}}
                             for g,m in [("g1","tv"),("g1","proposal"),("g2","tv")]])
        with self.assertRaises(ValueError):
            paired(data,"tv","proposal","ap",load_config())


if __name__ == "__main__":
    unittest.main()
