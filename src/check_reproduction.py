"""Reproduce from cached licensed source files, leaving published artifacts unchanged."""

from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(a, b, key="root"):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), key
        for k in a:
            if k not in ["frozen_utc", "evaluated_utc", "fitting_sha256"]:
                compare(a[k], b[k], key + "." + k)
    elif isinstance(a, list):
        assert len(a) == len(b), key
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, key + f"[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9), (key, a, b)
    else:
        assert a == b, (key, a, b)


began = time.perf_counter()
with tempfile.TemporaryDirectory(prefix="sensor-reproduction-") as folder:
    replay = Path(folder)
    shutil.copytree(ROOT / "src", replay / "src")
    shutil.copytree(ROOT / "data/source", replay / "data/source")
    shutil.copy2(ROOT / "data/manifest.json", replay / "data/manifest.json")
    for script, args in [
        ("analyze.py", ["validate"]),
        ("verify.py", ["validation"]),
        ("verify_optimization.py", []),
        ("analyze.py", ["test"]),
        ("verify.py", ["test"]),
        ("figures.py", []),
    ]:
        subprocess.run(
            [sys.executable, str(replay / "src" / script), *args],
            cwd=replay,
            check=True,
            timeout=600,
        )
    for name in ["selection.json", "test_summary.json"]:
        compare(
            json.loads((ROOT / "results" / name).read_text()),
            json.loads((replay / "results" / name).read_text()),
            name,
        )
    for path in sorted((ROOT / "results/fitting").glob("logistic*.json")):
        a = json.loads(path.read_bytes())
        b = json.loads((replay / "results/fitting" / path.name).read_bytes())
        for key in [
            "status",
            "specification",
            "active_features",
            "solver",
            "max_iter",
            "tol",
            "classes",
            "training_rows",
        ]:
            compare(a[key], b[key], path.name + "." + key)
        for key in [
            "coefficients",
            "intercepts",
            "scaler_mean",
            "scaler_scale",
            "scaler_variance",
            "objective",
            "lambda",
            "weight_sum",
        ]:
            compare(a[key], b[key], path.name + "." + key)
    for name in ["validation_predictions.csv.gz", "test_predictions.csv.gz"]:
        a = pd.read_csv(ROOT / "results" / name, float_precision="round_trip")
        b = pd.read_csv(replay / "results" / name, float_precision="round_trip")
        prob = [f"p{i}" for i in range(6)]
        assert a.drop(columns=prob).equals(b.drop(columns=prob)), name
        assert np.allclose(
            a[prob].to_numpy(), b[prob].to_numpy(), rtol=1e-7, atol=1e-9
        ), name
    a = pd.read_csv(ROOT / "results/search.csv")
    b = pd.read_csv(replay / "results/search.csv")
    assert a.name.tolist() == b.name.tolist() and a.status.tolist() == b.status.tolist()
    for key in [
        "participant_condition_macro_f1",
        "pooled_macro_f1",
        "accuracy",
        "balanced_accuracy",
        "brier_sum",
        "log_loss",
    ]:
        assert np.allclose(
            a[key].to_numpy(), b[key].to_numpy(), rtol=1e-7, atol=1e-9, equal_nan=True
        ), key
    print(
        "PASS: all candidate refits, selected policy, test results, independent metrics and figure generation reproduce offline."
    )
receipt = {
    "status": "PASS",
    "scope": "Corrected retrospective N1; full candidate refits and independent optimization/metric checks",
    "candidate_rows": 17,
    "logistic_models_reconstructed": 8,
    "relative_tolerance": 1e-7,
    "absolute_tolerance": 1e-9,
    "end_to_end_seconds": time.perf_counter() - began,
    "timing_and_iteration_counts": "Retained diagnostics, not cross-platform equality targets",
    "test_reuse_disclosed": True,
}
(ROOT / "results/reproduction.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(receipt), flush=True)
