"""Fit declared candidates, freeze validation decisions, then score protected participants."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import time
import warnings
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from fitting import fit_predict
from prepare import (
    ROOT,
    FEATURES,
    VAL_IDS,
    CONDITIONS,
    source,
    conditions,
    mask,
    weights,
    export,
    fitting_validation,
)

R = ROOT / "results"
SEED = 20261028
KEYS = ["source_row", "participant", "label", "partition", "condition"]
PROBS = [f"p{i}" for i in range(6)]
THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def save(name, value):
    (R / name).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def candidates():
    rows = [dict(name="prior", kind="prior", augmentation=False)]
    for aug in [False, True]:
        for c in [0.01, 0.1, 1.0, 10.0]:
            rows.append(
                dict(
                    name=f"logistic_C{c:g}_aug{int(aug)}",
                    kind="logistic",
                    augmentation=aug,
                    C=c,
                )
            )
        for depth in [8, 16]:
            for leaf in [5, 20]:
                rows.append(
                    dict(
                        name=f"forest_d{depth}_leaf{leaf}_aug{int(aug)}",
                        kind="forest",
                        augmentation=aug,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                    )
                )
    return rows


def fitted_predict(spec, fit, evaluation, use_frozen=False):
    return fit_predict(spec, fit, evaluation, use_frozen=use_frozen)


def score(pred):
    y = pred.label.to_numpy()
    p = pred[PROBS].to_numpy()
    yh = p.argmax(axis=1)
    grouped = [
        f1_score(
            g.label,
            g[PROBS].to_numpy().argmax(axis=1),
            labels=np.arange(6),
            average="macro",
            zero_division=0,
        )
        for _, g in pred.groupby(["participant", "condition"])
    ]
    return dict(
        n=len(pred),
        participants=int(pred.participant.nunique()),
        participant_condition_macro_f1=float(np.mean(grouped)),
        pooled_macro_f1=float(
            f1_score(y, yh, labels=np.arange(6), average="macro", zero_division=0)
        ),
        accuracy=float(accuracy_score(y, yh)),
        balanced_accuracy=float(balanced_accuracy_score(y, yh)),
        brier_sum=float(np.mean(np.sum((p - np.eye(6)[y]) ** 2, axis=1))),
        log_loss=float(log_loss(y, p, labels=np.arange(6))),
    )


def selective(pred, threshold):
    p = pred[PROBS].to_numpy()
    accepted = (
        np.zeros(len(pred), bool) if threshold is None else p.max(axis=1) >= threshold
    )
    accepted &= pred.condition.to_numpy() != "both_missing"
    good = p.argmax(axis=1) == pred.label.to_numpy()
    n = int(accepted.sum())
    right = int((accepted & good).sum())
    return dict(
        n=len(pred),
        accepted=n,
        accepted_correct=right,
        coverage=n / len(pred),
        selective_accuracy=right / n if n else None,
        accepted_correct_fraction=right / len(pred),
    )


def calibrate(pred):
    curve = [dict(threshold=t, **selective(pred, t)) for t in THRESHOLDS]
    eligible = [
        r for r in curve if r["accepted"] >= 100 and r["selective_accuracy"] >= 0.9
    ]
    selected = (
        min(eligible, key=lambda r: (-r["coverage"], r["threshold"]))["threshold"]
        if eligible
        else None
    )
    return selected, curve


def export_pred(spec, fit, evaluation, use_frozen=False):
    answer = evaluation[KEYS].copy()
    answer["model"] = spec["name"]
    answer[PROBS] = fitted_predict(spec, fit, evaluation, use_frozen)
    return answer


def validate():
    R.mkdir(exist_ok=True)
    fit, valid = fitting_validation()
    evaluation = conditions(valid)
    export(
        pd.concat([fit.assign(role="fit"), valid.assign(role="validation")]),
        R / "training_features.csv.gz",
    )
    records = []
    all_pred = []
    for spec in candidates():
        start = time.perf_counter()
        try:
            pred = export_pred(spec, fit, evaluation)
            metrics = score(pred)
            records.append(
                {
                    **spec,
                    **metrics,
                    "status": "completed",
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
            all_pred.append(pred)
        except ConvergenceWarning as err:
            records.append(
                {
                    **spec,
                    "status": "convergence_failure",
                    "error": str(err),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            )
        print(
            spec["name"],
            records[-1]["status"],
            records[-1].get("participant_condition_macro_f1"),
            flush=True,
        )
    table = pd.DataFrame(records)
    table.to_csv(R / "search.csv", index=False)
    successful = table[table.status == "completed"].sort_values(
        ["participant_condition_macro_f1", "name"], ascending=[False, True]
    )
    winner = successful.iloc[0]["name"]
    baseline = successful[
        (successful.augmentation == False) & (successful.kind != "prior")
    ].iloc[0]["name"]
    combined = pd.concat(all_pred, ignore_index=True)
    export(combined, R / "validation_predictions.csv.gz")
    thresholds = {}
    curves = []
    for name in dict.fromkeys([winner, baseline]):
        threshold, curve = calibrate(combined[combined.model == name])
        thresholds[name] = threshold
        curves.extend([dict(model=name, **row) for row in curve])
    pd.DataFrame(curves).to_csv(R / "threshold_search.csv", index=False)
    save(
        "selection.json",
        dict(
            frozen_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_status="CORRECTED_RETROSPECTIVE_N1",
            selected_model=winner,
            clean_only_comparator=baseline,
            thresholds=thresholds,
            fitting_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((R / "fitting").glob("logistic*.json"))
            },
            primary_objective="Maximize mean participant-condition macro F1 on validation; name tie break",
            fit_participants=sorted(map(int, fit.participant.unique())),
            validation_participants=sorted(VAL_IDS),
            source_test_partition="Previously inspected; N1 numerical correction, not an untouched test",
            candidates=candidates(),
            fixed_settings=dict(
                seed=SEED,
                forest_trees=160,
                forest_workers=2,
                forest_features="sqrt",
                forest_bootstrap=True,
                logistic_max_iter=100,
                logistic_tol=1e-12,
                logistic_solver="newton-cholesky",
            ),
            features=FEATURES,
            conditions=CONDITIONS,
        ),
    )


def test():
    selection = json.loads((R / "selection.json").read_text())
    fit, _ = fitting_validation()
    test = source("test")
    assert set(test.participant).isdisjoint(
        selection["fit_participants"] + selection["validation_participants"]
    )
    evaluation = conditions(test)
    export(test, R / "test_features.csv.gz")
    names = list(
        dict.fromkeys(
            [selection["selected_model"], selection["clean_only_comparator"], "prior"]
        )
    )
    for name, digest in selection["fitting_sha256"].items():
        assert hashlib.sha256((R / "fitting" / name).read_bytes()).hexdigest() == digest
    certificate = json.loads((R / "INDEPENDENT_OPTIMIZATION_CHECK.json").read_bytes())
    assert (
        certificate["status"] == "PASS"
        and certificate["fitting_sha256"] == selection["fitting_sha256"]
    )
    specs = {s["name"]: s for s in candidates()}
    predictions = [
        export_pred(specs[name], fit, evaluation, use_frozen=True) for name in names
    ]
    combined = pd.concat(predictions, ignore_index=True)
    export(combined, R / "test_predictions.csv.gz")
    records = []
    reliability = []
    for (name, condition), group in combined.groupby(["model", "condition"]):
        threshold = selection["thresholds"].get(name, 0.0)
        records.append(
            dict(
                model=name,
                condition=condition,
                **{**score(group), **selective(group, threshold)},
            )
        )
        p = group[PROBS].to_numpy()
        conf = p.max(axis=1)
        good = p.argmax(axis=1) == group.label.to_numpy()
        for lo in np.arange(0, 1, 0.1):
            ix = (conf >= lo) & (conf < lo + 0.1 if lo < 0.9 else conf <= 1.0)
            reliability.append(
                dict(
                    model=name,
                    condition=condition,
                    lower=float(lo),
                    n=int(ix.sum()),
                    confidence=float(conf[ix].mean()) if ix.any() else None,
                    accuracy=float(good[ix].mean()) if ix.any() else None,
                )
            )
    pd.DataFrame(records).to_csv(R / "test_metrics.csv", index=False)
    pd.DataFrame(reliability).to_csv(R / "reliability.csv", index=False)
    people = []
    for (name, person, condition), g in combined.groupby(
        ["model", "participant", "condition"]
    ):
        people.append(
            dict(
                model=name,
                participant=int(person),
                condition=condition,
                **selective(g, selection["thresholds"].get(name, 0.0)),
                accuracy=float((g[PROBS].to_numpy().argmax(axis=1) == g.label).mean()),
            )
        )
    pd.DataFrame(people).to_csv(R / "participant_metrics.csv", index=False)
    winner = selection["selected_model"]
    baseline = selection["clean_only_comparator"]
    ids = sorted(map(int, test.participant.unique()))
    rng = np.random.default_rng(SEED)
    accum = {
        metric: [] for metric in ["accuracy", "coverage", "accepted_correct_fraction"]
    }
    for _ in range(2000):
        sample = rng.choice(ids, len(ids), replace=True)
        # Resampling multiplicities weight entire participant blocks, including all conditions.
        counts = {int(k): int((sample == k).sum()) for k in ids}
        vals = {}
        for name in dict.fromkeys([winner, baseline]):
            p = combined[combined.model == name]
            w = p.participant.map(counts).to_numpy()
            pr = p[PROBS].to_numpy()
            good = pr.argmax(axis=1) == p.label.to_numpy()
            accepted = (
                np.zeros(len(p), bool)
                if selection["thresholds"][name] is None
                else pr.max(axis=1) >= selection["thresholds"][name]
            )
            den = w.sum()
            vals[name] = dict(
                accuracy=float(np.dot(w, good) / den),
                coverage=float(np.dot(w, accepted) / den),
                accepted_correct_fraction=float(np.dot(w, accepted & good) / den),
            )
        for key in accum:
            accum[key].append(vals[winner][key] - vals[baseline][key])
    intervals = {
        key: np.quantile(value, [0.025, 0.975]).tolist() for key, value in accum.items()
    }
    save(
        "test_summary.json",
        dict(
            evaluated_utc=datetime.now(timezone.utc).isoformat(),
            evaluation_status="CORRECTED_RETROSPECTIVE_N1",
            test_participants=ids,
            test_windows=len(test),
            conditions=CONDITIONS,
            selected_model=winner,
            comparator=baseline,
            metrics=records,
            paired_participant_bootstrap_intervals=intervals,
            bootstrap_draws=2000,
            seed=SEED,
            both_modalities_missing="Compulsory abstention; no model confidence can override absent inputs",
            limitations=[
                "Previously inspected test rescored after numerical solver repair; no untouched-test claim",
                "Augmentation changes weight_sum and effective L2 penalty at fixed C; configured-system comparison, not an isolated causal augmentation effect",
                "Controlled failures are synthetic interventions, not observed device failures",
                "Source windows overlap and upstream signals are processed",
                "Nine test participants do not establish population or device transfer",
                "Validation selective accuracy target is not guaranteed on unseen participants",
            ],
        ),
    )
    print(
        json.dumps(
            {"winner": winner, "test_windows": len(test), "intervals": intervals}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["validate", "test"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate() if args.phase == "validate" else test()
