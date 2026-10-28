"""Numerically certified fitting and frozen logistic inference for repair N1."""

import hashlib
import json
import warnings
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from prepare import ROOT, FEATURES, conditions, mask, weights

SEED = 20261028
FITTING = ROOT / "results/fitting"


def training_fingerprint(train, weight):
    raw = train[FEATURES].to_numpy(dtype="<f8").tobytes()
    raw += train.label.to_numpy(dtype="<i8").tobytes()
    raw += np.asarray(weight, dtype="<f8").tobytes()
    return hashlib.sha256(raw).hexdigest()


def save(path, evidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def frozen_probabilities(evidence, evaluation):
    active = np.asarray(evidence["active_features"], dtype=bool)
    z = evaluation[FEATURES].to_numpy()[:, active]
    z = (z - np.array(evidence["scaler_mean"])) / np.array(evidence["scaler_scale"])
    logits = np.einsum(
        "ij,kj->ik", z, np.array(evidence["coefficients"]), optimize=False
    )
    return softmax(logits + np.array(evidence["intercepts"]), axis=1)


def fit_predict(spec, fit, evaluation, use_frozen=False):
    train = conditions(fit) if spec["augmentation"] else mask(fit, "intact")
    weight = weights(train)
    if spec["kind"] == "prior":
        p = np.bincount(train.label, weights=weight, minlength=6)
        return np.tile(p / p.sum(), (len(evaluation), 1))
    path = FITTING / (spec["name"] + ".json")
    fingerprint = training_fingerprint(train, weight)
    if use_frozen and spec["kind"] == "logistic":
        evidence = json.loads(path.read_bytes())
        assert (
            evidence["status"] == "completed"
            and evidence["training_fingerprint"] == fingerprint
        )
        return frozen_probabilities(evidence, evaluation)

    x = train[FEATURES].to_numpy()
    z = evaluation[FEATURES].to_numpy()
    if spec["kind"] == "logistic":
        active = np.ptp(x, axis=0) > 0
        scaler = StandardScaler().fit(x[:, active], sample_weight=weight)
        x = scaler.transform(x[:, active])
        z = scaler.transform(z[:, active])
        model = LogisticRegression(
            C=spec["C"],
            solver="newton-cholesky",
            max_iter=100,
            tol=1e-12,
            random_state=SEED,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=160,
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            max_features="sqrt",
            bootstrap=True,
            random_state=SEED,
            n_jobs=2,
        )
    assert np.isfinite(x).all() and np.isfinite(z).all()
    evidence = {
        "status": "started",
        "specification": spec,
        "training_rows": len(train),
        "training_fingerprint": fingerprint,
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x, train.label, sample_weight=weight)
    except ConvergenceWarning as error:
        evidence.update({"status": "convergence_failure", "error": str(error)})
        save(path, evidence)
        raise
    assert list(model.classes_) == list(range(6))
    evidence.update(
        {
            "status": "completed",
            "classes": list(map(int, model.classes_)),
            "warnings": [str(item.message) for item in caught],
        }
    )
    if spec["kind"] != "logistic":
        evidence["actual_trees"] = len(model.estimators_)
        save(path, evidence)
        return model.predict_proba(z)

    normalized = weight / weight.sum()
    penalty = 1 / (spec["C"] * weight.sum())
    eta = x @ model.coef_.T + model.intercept_
    residual = softmax(eta, axis=1) - np.eye(6)[train.label.to_numpy()]
    gradient = residual.T @ (x * normalized[:, None]) + penalty * model.coef_
    intercept_gradient = (residual * normalized[:, None]).sum(axis=0)
    norm = float(max(abs(gradient).max(), abs(intercept_gradient).max()))
    objective = float(
        np.sum(
            normalized
            * (
                logsumexp(eta, axis=1)
                - eta[np.arange(len(train)), train.label.to_numpy()]
            )
        )
        + penalty * np.sum(model.coef_**2) / 2
    )
    evidence.update(
        {
            "active_features": active.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "scaler_variance": scaler.var_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercepts": model.intercept_.tolist(),
            "actual_iterations": model.n_iter_.tolist(),
            "solver": "newton-cholesky",
            "max_iter": 100,
            "tol": 1e-12,
            "gradient_inf": norm,
            "objective": objective,
            "lambda": penalty,
            "weight_sum": float(weight.sum()),
        }
    )
    probability = frozen_probabilities(evidence, evaluation)
    difference = float(abs(probability - model.predict_proba(z)).max())
    evidence["evaluation_probability_max_abs_library_difference"] = difference
    if norm > 1e-10 or difference > 1e-9:
        evidence["status"] = "numerical_gate_failure"
        save(path, evidence)
        raise ConvergenceWarning("Declared numerical fitting gate failed")
    save(path, evidence)
    return probability
