"""Window features with explicit source row identities and modality lineage."""

from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = [
    f"{sensor}_{axis}"
    for sensor in ["body_acc", "total_acc", "body_gyro"]
    for axis in ["x", "y", "z"]
]
STATS = ["mean", "sd", "rms", "range", "mean_abs_step"]
FEATURES = [f"{channel}_{stat}" for channel in CHANNELS for stat in STATS] + [
    "acc_present",
    "gyro_present",
]
VAL_IDS = {3, 8, 15, 22, 27, 30}
CONDITIONS = ["intact", "acc_missing", "gyro_missing"]


def source(part):
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    selected = [
        r for r in manifest["files"] if r["original_file"].startswith(part + "/")
    ]
    assert len(selected) == 11
    for row in selected:
        blob = (ROOT / row["file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row["compressed_sha256"]
        assert hashlib.sha256(gzip.decompress(blob)).hexdigest() == row["sha256"]

    def load(name):
        return np.loadtxt(ROOT / f"data/source/{part}/{name}.gz")

    ids = load(f"subject_{part}.txt").astype(int)
    y = load(f"y_{part}.txt").astype(int) - 1
    assert len(ids) == len(y) and set(y) == set(range(6))
    f = pd.DataFrame({"source_row": np.arange(len(y)), "participant": ids, "label": y})
    for channel in CHANNELS:
        a = load(f"Inertial Signals/{channel}_{part}.txt")
        assert a.shape == (len(y), 128) and np.isfinite(a).all()
        values = [
            a.mean(axis=1),
            a.std(axis=1),
            np.sqrt((a * a).mean(axis=1)),
            np.ptp(a, axis=1),
            np.abs(np.diff(a, axis=1)).mean(axis=1),
        ]
        for stat, value in zip(STATS, values):
            f[f"{channel}_{stat}"] = value
    f["acc_present"] = 1.0
    f["gyro_present"] = 1.0
    f["partition"] = part
    return f


def mask(frame, condition):
    answer = frame.copy()
    answer["condition"] = condition
    if condition in ["acc_missing", "both_missing"]:
        answer[[c for c in FEATURES if c.startswith(("body_acc", "total_acc"))]] = 0.0
        answer["acc_present"] = 0.0
    if condition in ["gyro_missing", "both_missing"]:
        answer[[c for c in FEATURES if c.startswith("body_gyro")]] = 0.0
        answer["gyro_present"] = 0.0
    assert condition in CONDITIONS + ["both_missing"]
    return answer


def conditions(frame):
    return pd.concat([mask(frame, c) for c in CONDITIONS], ignore_index=True)


def fitting_validation():
    f = source("train")
    assert VAL_IDS.issubset(set(f.participant))
    fit = f[~f.participant.isin(VAL_IDS)].copy()
    valid = f[f.participant.isin(VAL_IDS)].copy()
    assert set(fit.participant).isdisjoint(valid.participant)
    return fit, valid


def weights(frame):
    counts = frame.groupby("participant").participant.transform("size")
    w = 1.0 / counts.to_numpy()
    return w * len(w) / w.sum()


def export(frame, path):
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None,
    )
