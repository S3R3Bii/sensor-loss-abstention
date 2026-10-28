"""Independent CSV arithmetic and source reconciliation; imports no fitting helpers."""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-9), (a, b)


def confusion(data):
    matrix = [[0] * 6 for _ in range(6)]
    for row in data:
        p = [float(row[f"p{k}"]) for k in range(6)]
        truth = int(row["label"])
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
        close(math.fsum(p), 1)
        guess = max(range(6), key=lambda i: p[i])
        matrix[truth][guess] += 1
    return matrix


def f1(matrix):
    values = []
    for i in range(6):
        tp = matrix[i][i]
        den = sum(matrix[i]) + sum(row[i] for row in matrix)
        values.append(2 * tp / den if den else 0.0)
    return math.fsum(values) / 6


def metrics(data):
    matrix = confusion(data)
    n = len(data)
    groups = defaultdict(list)
    for row in data:
        groups[(row["participant"], row["condition"])].append(row)
    brier = []
    loss = []
    for row in data:
        truth = int(row["label"])
        p = [float(row[f"p{k}"]) for k in range(6)]
        brier.append(math.fsum((v - (i == truth)) ** 2 for i, v in enumerate(p)))
        epsilon = float(np.finfo(float).eps)
        loss.append(-math.log(max(epsilon, min(1 - epsilon, p[truth]))))
    return dict(
        n=n,
        participants=len({r["participant"] for r in data}),
        participant_condition_macro_f1=math.fsum(
            f1(confusion(g)) for g in groups.values()
        )
        / len(groups),
        pooled_macro_f1=f1(matrix),
        accuracy=sum(matrix[i][i] for i in range(6)) / n,
        balanced_accuracy=math.fsum(
            matrix[i][i] / sum(matrix[i]) for i in range(6) if sum(matrix[i])
        )
        / sum(bool(sum(row)) for row in matrix),
        brier_sum=math.fsum(brier) / n,
        log_loss=math.fsum(loss) / n,
    )


def selective(data, threshold):
    n = len(data)
    accepted = right = 0
    for row in data:
        p = [float(row[f"p{i}"]) for i in range(6)]
        use = (
            threshold is not None
            and max(p) >= threshold
            and row["condition"] != "both_missing"
        )
        accepted += int(use)
        right += int(use and max(range(6), key=lambda i: p[i]) == int(row["label"]))
    return dict(
        n=n,
        accepted=accepted,
        accepted_correct=right,
        coverage=accepted / n,
        selective_accuracy=right / accepted if accepted else None,
        accepted_correct_fraction=right / n,
    )


def feature_checks(part):
    subjects = np.loadtxt(ROOT / f"data/source/{part}/subject_{part}.txt.gz", dtype=int)
    labels = np.loadtxt(ROOT / f"data/source/{part}/y_{part}.txt.gz", dtype=int) - 1
    path = R / (
        "training_features.csv.gz" if part == "train" else "test_features.csv.gz"
    )
    features = list(rows(path))
    assert len(features) == len(subjects)
    assert len({int(row["source_row"]) for row in features}) == len(features)
    for row in features:
        ix = int(row["source_row"])
        assert (
            int(row["participant"]) == subjects[ix] and int(row["label"]) == labels[ix]
        )
        if part == "train":
            assert row["role"] == (
                "validation" if subjects[ix] in [3, 8, 15, 22, 27, 30] else "fit"
            )
    # Fixed, deterministic boundary/interior spot checks supplement full source hashes and row-identity checks.
    indices = sorted(
        set(
            [
                0,
                len(subjects) - 1,
                *range(0, len(subjects), max(1, len(subjects) // 12)),
            ]
        )
    )
    lookup = {int(row["source_row"]): row for row in features}
    for sensor in ["body_acc", "total_acc", "body_gyro"]:
        for axis in ["x", "y", "z"]:
            channel = f"{sensor}_{axis}"
            a = np.loadtxt(
                ROOT / f"data/source/{part}/Inertial Signals/{channel}_{part}.txt.gz"
            )
            assert a.shape == (len(subjects), 128)
            for ix in indices:
                values = list(map(float, a[ix]))
                mean = math.fsum(values) / 128
                stats = dict(
                    mean=mean,
                    sd=math.sqrt(math.fsum((v - mean) ** 2 for v in values) / 128),
                    rms=math.sqrt(math.fsum(v * v for v in values) / 128),
                    range=max(values) - min(values),
                    mean_abs_step=math.fsum(
                        abs(a - b) for a, b in zip(values[:-1], values[1:])
                    )
                    / 127,
                )
                for key, value in stats.items():
                    close(lookup[ix][f"{channel}_{key}"], value)
    return {
        "part": part,
        "rows": len(subjects),
        "participants": sorted(map(int, set(subjects))),
        "feature_windows_spot_checked": len(indices),
    }


def main(phase):
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for item in manifest["files"]:
        data = (ROOT / item["file"]).read_bytes()
        raw = gzip.decompress(data)
        assert hashlib.sha256(data).hexdigest() == item["compressed_sha256"]
        assert (
            hashlib.sha256(raw).hexdigest() == item["sha256"]
            and len(raw) == item["bytes"]
        )
    assert len(manifest["files"]) == 24
    selection = json.loads((R / "selection.json").read_text())
    table = list(rows(R / "search.csv"))
    by_model = defaultdict(list)
    for row in rows(R / "validation_predictions.csv.gz"):
        by_model[row["model"]].append(row)
    assert len(table) == len(selection["candidates"]) == 17
    all_metrics = {name: metrics(data) for name, data in by_model.items()}
    for row in table:
        if row["status"] == "completed":
            for key, value in all_metrics[row["name"]].items():
                close(row[key], value)
    winner = min(
        all_metrics,
        key=lambda name: (-all_metrics[name]["participant_condition_macro_f1"], name),
    )
    clean = [
        r["name"]
        for r in table
        if r["status"] == "completed"
        and r["augmentation"] == "False"
        and r["kind"] != "prior"
    ]
    baseline = min(
        clean,
        key=lambda name: (-all_metrics[name]["participant_condition_macro_f1"], name),
    )
    assert (
        winner == selection["selected_model"]
        and baseline == selection["clean_only_comparator"]
    )
    curves = list(rows(R / "threshold_search.csv"))
    for name, threshold in selection["thresholds"].items():
        eligible = []
        for row in [r for r in curves if r["model"] == name]:
            t = float(row["threshold"])
            v = selective(by_model[name], t)
            for key, value in v.items():
                if value is not None:
                    close(row[key], value)
                else:
                    assert row[key] == ""
            if v["accepted"] >= 100 and v["selective_accuracy"] >= 0.9:
                eligible.append((t, v))
        expected = (
            min(eligible, key=lambda pair: (-pair[1]["coverage"], pair[0]))[0]
            if eligible
            else None
        )
        assert threshold == expected
    quality = [feature_checks("train")]
    fit = set(selection["fit_participants"])
    valid = set(selection["validation_participants"])
    assert fit.isdisjoint(valid) and fit | valid == set(quality[0]["participants"])
    checks = [
        "24 compressed and original source hashes reconcile",
        "Training row labels/participant identities and split roles reconcile",
        "All17 candidate outcomes retained; successful candidate F1/Brier/log-loss metrics independently reproduced",
        "Validation model/comparator and threshold selection independently reproduced",
    ]
    if phase == "test":
        quality.append(feature_checks("test"))
        test_ids = set(quality[-1]["participants"])
        assert test_ids.isdisjoint(fit | valid)
        summary = json.loads((R / "test_summary.json").read_text())
        assert summary["evaluated_utc"] > selection["frozen_utc"]
        grouped = defaultdict(list)
        for row in rows(R / "test_predictions.csv.gz"):
            grouped[(row["model"], row["condition"])].append(row)
        for reported in rows(R / "test_metrics.csv"):
            name = reported["model"]
            group = grouped[(name, reported["condition"])]
            for key, value in {
                **metrics(group),
                **selective(group, selection["thresholds"].get(name, 0.0)),
            }.items():
                if value is not None:
                    close(reported[key], value)
                else:
                    assert reported[key] == ""
        # Aggregate participant numerators separately from the fitting implementation.
        names = list(dict.fromkeys([winner, baseline]))
        blocks = {name: defaultdict(lambda: [0, 0, 0, 0]) for name in names}
        for (name, _), data in grouped.items():
            if name not in blocks:
                continue
            for row in data:
                p = [float(row[f"p{i}"]) for i in range(6)]
                correct = max(range(6), key=lambda i: p[i]) == int(row["label"])
                t = selection["thresholds"][name]
                accepted = t is not None and max(p) >= t
                b = blocks[name][int(row["participant"])]
                b[0] += 1
                b[1] += int(correct)
                b[2] += int(accepted)
                b[3] += int(correct and accepted)
        ids = sorted(test_ids)
        rng = np.random.default_rng(20261028)
        draws = {
            key: [] for key in ["accuracy", "coverage", "accepted_correct_fraction"]
        }
        for _ in range(2000):
            sample = rng.choice(ids, size=len(ids), replace=True)
            value = {}
            for name in names:
                total = [sum(blocks[name][int(i)][k] for i in sample) for k in range(4)]
                value[name] = [total[k] / total[0] for k in [1, 2, 3]]
            for k, key in enumerate(draws):
                draws[key].append(value[winner][k] - value[baseline][k])
        for key, values in draws.items():
            ordered = sorted(values)
            for p, reported in zip(
                [0.025, 0.975], summary["paired_participant_bootstrap_intervals"][key]
            ):
                x = p * 1999
                lo = math.floor(x)
                fraction = x - lo
                close(
                    reported, ordered[lo] * (1 - fraction) + ordered[lo + 1] * fraction
                )
        checks.append(
            "Protected test participant split, metrics, confusion-derived F1, probability scores and selective outcomes independently reproduced"
        )
        checks.append(
            "Paired participant-block bootstrap intervals independently reproduced"
        )
    receipt = dict(
        status="PASS",
        phase=phase,
        source_files=24,
        checks=checks,
        source_reconciliation=quality,
        limits="Feature statistics independently spot checked on fixed boundary/interior windows; all source hashes and exported row identities checked. These finite checks do not establish real device failure or population performance.",
    )
    (R / f"{phase}_verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validation", "test"])
    main(parser.parse_args().phase)
