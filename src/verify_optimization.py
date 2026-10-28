"""Independent multinomial objective, Hessian and frozen-probability checks."""

from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def main():
    selection = json.loads((ROOT / "results/selection.json").read_bytes())
    features = selection["features"]
    source = pd.read_csv(
        ROOT / "results/training_features.csv.gz", float_precision="round_trip"
    )
    fit = source[source.role == "fit"].copy()
    valid = source[source.role == "validation"].copy()
    assert set(fit.participant).isdisjoint(valid.participant)

    def conditions(frame):
        frames = []
        for condition in ["intact", "acc_missing", "gyro_missing"]:
            f = frame.copy()
            f["condition"] = condition
            if condition == "acc_missing":
                for column in features:
                    if column.startswith(("body_acc", "total_acc")):
                        f[column] = 0.0
                f["acc_present"] = 0.0
            if condition == "gyro_missing":
                for column in features:
                    if column.startswith("body_gyro"):
                        f[column] = 0.0
                f["gyro_present"] = 0.0
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    evaluation = conditions(valid)
    predictions = pd.read_csv(
        ROOT / "results/validation_predictions.csv.gz", float_precision="round_trip"
    )
    evidence = []
    for filename, digest in selection["fitting_sha256"].items():
        path = ROOT / "results/fitting" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        model = json.loads(path.read_bytes())
        spec = model["specification"]
        assert (
            model["status"] == "completed"
            and model["solver"] == "newton-cholesky"
            and model["tol"] == 1e-12
        )
        train = conditions(fit) if spec["augmentation"] else fit.copy()
        x = train[features].to_numpy(float)
        y = train.label.to_numpy(int)
        counts = {
            p: sum(train.participant.to_numpy() == p)
            for p in train.participant.unique()
        }
        weight = np.array([1 / counts[p] for p in train.participant])
        # The freeze hash binds exact computation bytes. Separately check the
        # participant-balance identity rather than treating hash equality as it.
        weight = weight * len(weight) / weight.sum()
        for participant in counts:
            assert math.isclose(
                math.fsum(
                    float(v)
                    for v in weight[train.participant.to_numpy() == participant]
                ),
                len(weight) / len(counts),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        wn = weight / weight.sum()
        fingerprint = hashlib.sha256(
            x.astype("<f8").tobytes()
            + y.astype("<i8").tobytes()
            + weight.astype("<f8").tobytes()
        ).hexdigest()
        assert fingerprint == model["training_fingerprint"]
        active = np.ptp(x, axis=0) > 0
        assert active.tolist() == model["active_features"]
        mean = np.sum(x[:, active] * wn[:, None], axis=0)
        variance = np.sum((x[:, active] - mean) ** 2 * wn[:, None], axis=0)
        assert np.allclose(mean, model["scaler_mean"], atol=1e-12, rtol=1e-10)
        assert np.allclose(variance, model["scaler_variance"], atol=1e-14, rtol=1e-9)
        scale = np.array(model["scaler_scale"])
        assert np.allclose(scale, np.sqrt(variance), atol=1e-12, rtol=1e-9)
        xx = (x[:, active] - np.array(model["scaler_mean"])) / scale
        coefficients = np.array(model["coefficients"])
        intercept = np.array(model["intercepts"])
        penalty = 1 / (spec["C"] * weight.sum())

        def objective(beta, bias):
            eta = xx @ beta.T + bias
            maximum = eta.max(axis=1)
            normalizer = maximum + np.log(np.exp(eta - maximum[:, None]).sum(axis=1))
            return (
                math.fsum(
                    float(w * l)
                    for w, l in zip(wn, normalizer - eta[np.arange(len(y)), y])
                )
                + penalty * math.fsum(float(v) ** 2 for v in beta.flat) / 2
            )

        eta = xx @ coefficients.T + intercept
        exponential = np.exp(eta - eta.max(axis=1)[:, None])
        p = exponential / exponential.sum(axis=1)[:, None]
        residual = p - np.eye(6)[y]
        gradient = residual.T @ (xx * wn[:, None]) + penalty * coefficients
        bias_gradient = (residual * wn[:, None]).sum(axis=0)
        maximum_gradient = float(max(abs(gradient).max(), abs(bias_gradient).max()))
        assert maximum_gradient <= 1e-10, maximum_gradient
        assert math.isclose(
            objective(coefficients, intercept),
            model["objective"],
            rel_tol=1e-8,
            abs_tol=1e-9,
        )
        augmented = np.column_stack((xx, np.ones(len(xx))))
        dimension = augmented.shape[1]
        hessian = np.zeros((6 * dimension, 6 * dimension))
        for k in range(6):
            for l in range(k, 6):
                curvature = wn * p[:, k] * ((1 if k == l else 0) - p[:, l])
                block = augmented.T @ (augmented * curvature[:, None])
                if k == l:
                    block[np.arange(dimension - 1), np.arange(dimension - 1)] += penalty
                hessian[
                    k * dimension : (k + 1) * dimension,
                    l * dimension : (l + 1) * dimension,
                ] = block
                hessian[
                    l * dimension : (l + 1) * dimension,
                    k * dimension : (k + 1) * dimension,
                ] = block.T
        minimum_eigenvalue = float(np.linalg.eigvalsh((hessian + hessian.T) / 2)[0])
        assert minimum_eigenvalue >= -1e-10
        direction = np.arange(1, 6 * dimension + 1, dtype=float).reshape(6, dimension)
        direction /= np.linalg.norm(direction)
        step = 1e-5
        numerical = (
            objective(
                coefficients + step * direction[:, :-1],
                intercept + step * direction[:, -1],
            )
            - objective(
                coefficients - step * direction[:, :-1],
                intercept - step * direction[:, -1],
            )
        ) / (2 * step)
        analytic = float(
            np.sum(gradient * direction[:, :-1])
            + np.sum(bias_gradient * direction[:, -1])
        )
        assert math.isclose(numerical, analytic, rel_tol=1e-6, abs_tol=1e-6)
        z = (
            evaluation[features].to_numpy()[:, active] - np.array(model["scaler_mean"])
        ) / scale
        logits = z @ coefficients.T + intercept
        ep = np.exp(logits - logits.max(axis=1)[:, None])
        probability = ep / ep.sum(axis=1)[:, None]
        exported = predictions[predictions.model == spec["name"]]
        assert (
            len(exported) == len(evaluation)
            and exported.source_row.tolist() == evaluation.source_row.tolist()
        )
        assert np.allclose(
            probability,
            exported[[f"p{i}" for i in range(6)]].to_numpy(),
            rtol=1e-7,
            atol=1e-9,
        )
        evidence.append(
            {
                "model": spec["name"],
                "training_rows": len(train),
                "maximum_gradient": maximum_gradient,
                "minimum_hessian_eigenvalue": minimum_eigenvalue,
                "finite_difference_error": abs(numerical - analytic),
                "validation_probability_rows": len(exported),
                "all_checks": "PASS",
            }
        )
        print(
            "Independent optimization PASS:",
            spec["name"],
            "gradient",
            maximum_gradient,
            flush=True,
        )
    assert len(evidence) == 8
    result = {
        "status": "PASS",
        "scope": "Eight logistic fits; convex objective/stationarity/Hessian/probability checks, not generalization proof",
        "fitting_sha256": selection["fitting_sha256"],
        "evidence": evidence,
    }
    (ROOT / "results/INDEPENDENT_OPTIMIZATION_CHECK.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
