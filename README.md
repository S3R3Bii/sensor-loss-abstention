# Sensor loss and the decision to request another measurement

**Corrected retrospective evaluation (N1).** The original L-BFGS-B-only candidate failed Linux refitting. A validation-only probe found insufficient optimization precision. The same logistic objectives now use Newton-Cholesky with independent gradient, Hessian and frozen-probability checks. All model grids, participant splits, decision thresholds and replay tolerances remain fixed; selected configurations and thresholds are unchanged. Original outputs and failures remain in Git history and [the correction evidence](results/corrections). This reused test is not described as untouched.

**Finding:** training with missing sensors improves classification and coverage on unseen participants, but a confidence cutoff that meets 90% retained-window accuracy on validation fails that target under both test sensor-loss conditions. A high-confidence subset can still be unreliable after input loss.

The practical decision is when a movement system should request another measurement. This study compares clean-only and dropout-augmented training, then tests a frozen abstention policy on people excluded from fitting and selection. The next step is a new participant cohort with condition-specific calibration and measured sensor failures.

![Coverage and selective accuracy on held-out participants](figures/decision-summary.png)

## Data and protected split

The [UCI smartphone activity dataset](https://doi.org/10.24432/C54S4K) provides measured inertial windows and six instructed activity labels. All 10,299 source windows are retained: 5,322 from 15 fitting participants, 2,030 from six validation participants, and 2,947 from nine official-test participants. No participant crosses a split. Windows overlap by 50%, so participant blocks govern separation and uncertainty.

Forty-five window summaries describe body acceleration, total acceleration and angular velocity. Two flags identify available modalities. The **missing-sensor conditions are simulated interventions** on measured windows. Accelerometer loss removes both total and derived body acceleration; gyro loss removes angular velocity. The source's processed 2.56-second windows support an offline benchmark, with no real-time latency claim. Video and demographic records are excluded.

## Full optimization evidence

[The protocol](PROTOCOL.md) fixed the split, conditions, search and decision rule before retrieval/fitting. Seventeen candidates comprise a prior baseline, eight L2 logistic configurations and eight forests. Each model family compares clean-only and three-condition augmented training. Training weights equalize participant totals. Logistic scaling and constant-feature removal use fitting rows alone.

Selection maximizes validation macro F1 averaged over participant-condition groups. The winner is augmented logistic regression with **C=10**, the largest tested value; the best clean-only comparator has C=0.1. This is a declared finite grid, not proof of a global optimum. [Every candidate score and runtime](results/search.csv), [frozen settings and identities](results/selection.json), and validation probability exports are retained.

![Complete candidate comparison](figures/optimization-evidence.png)

Confidence cutoffs 0, .4, .5, .6, .7, .8, .9, .95 and .99 were tested on validation. The rule maximizes retained coverage subject to at least 90% selective accuracy and 100 accepted windows. The selected cutoffs are 0.5 (augmented) and 0.95 (clean-only). Probability reliability is measured separately with Brier score, log loss and fixed confidence bins; threshold calibration does not establish calibrated probabilities.

![Validation coverage and accuracy frontier](figures/abstention-frontier.png)

## Frozen policy on unseen participants

Each condition contains the same 2,947 test windows. Selective accuracy uses only accepted windows; coverage uses all windows in that condition.

| Training | Test condition | Raw accuracy | Retained coverage | Selective accuracy |
|---|---|---:|---:|---:|
| Augmented | Intact | 90.7% | 99.5% | 91.0% |
| Augmented | Accelerometer absent | 72.1% | 63.9% | 82.0% |
| Augmented | Gyroscope absent | 84.2% | 99.3% | 84.6% |
| Clean-only | Intact | 88.9% | 45.5% | 98.7% |
| Clean-only | Accelerometer absent | 18.4% | 10.8% | 0.0% |
| Clean-only | Gyroscope absent | 79.7% | 33.8% | 98.0% |

The corrected clean-only policy accepts **317 accelerometer-missing windows and classifies every one incorrectly**. The earlier incomplete numerical fit reported 178; the original evidence remains preserved. Confidence alone does not reliably detect that input failure. The validation-selected augmented system improves raw accuracy in every tested condition and retains much more data, but its fixed rule also misses the reliability target when sensors are absent.

The augmented model's multiclass Brier scores are 0.139 intact, 0.410 without acceleration and 0.230 without gyro. Brier is the mean **sum** of squared errors across six class probabilities, not a per-class average. Full macro F1, balanced accuracy, Brier, log loss, denominators and selective outcomes are in [test metrics](results/test_metrics.csv).

Across the controlled three-condition mixture, paired 2,000-draw participant-block bootstrap intervals are +17.1 to +23.1 percentage points for raw-accuracy improvement, +54.0 to +60.5 points for coverage improvement and +45.4 to +53.4 points for accepted-correct fraction. These are conditional on nine sampled participants, not population/device uncertainty or a natural failure mix. Both sensors missing forces abstention regardless of confidence.

![Participant variation and probability reliability](figures/reliability-and-participants.png)

## Independent checks and reproduction

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/verify.py test
python src/verify_optimization.py
python src/check_reproduction.py
```

The metric verifier imports no fitting/metric helpers. It independently recomputes all successful candidate metrics, selection, thresholds, test scores and participant-block intervals. The optimization verifier reconstructs all eight logistic objectives, gradients, Hessians, finite differences, scalers and validation probabilities. Maximum gradient is **2.17×10⁻¹³**, below the **10⁻¹⁰** independent gate. All 24 source-file hashes and all row identities reconcile; feature arithmetic is independently spot checked at 14 fixed windows per source partition. Eight contracts also cover frozen-model identity, label-independent inference and extreme logits. The fresh offline replay refits all candidates and compares coefficients, all validation/test probabilities and metrics at the original **1e-7 relative / 1e-9 absolute** tolerance.

Corrected coefficients were committed at `75d7b829b57e33249ff651a70ca3ce86bc004a2f` before rescoring. Test inference reuses those frozen logistic coefficients and checks their hashes. It cannot silently reuse an older probability export after a model change. [Fresh Windows replay evidence](results/reproduction.json) passes; exact-head Linux replay remains a release gate.

Earlier corrections are retained too: a constant-column scaling warning before testing and a duplicate-count-field score-export repair. That earlier export repair reused its retained predictions. N1 subsequently refits the same objectives to tighter numerical convergence and rescores the previously inspected test. It does not erase those earlier records or create a new independent test.

## Boundaries and reuse

The study uses one historical smartphone setup, instructed activities and a small participant sample. Controlled sensor masking does not reproduce hardware faults, timing interruptions or new tasks. No clinical validation, production deployment, new algorithm or exhaustive novelty claim is made. The original recognition benchmark and preprocessing come from Anguita and colleagues; attribution and [CC BY 4.0 reuse details](SOURCES.md) accompany the source files.

The comparison is between validation-selected configured systems. Augmented row counts change the effective L2 penalty at fixed C, so the result does not isolate a causal effect of augmentation alone. A matched-penalty ablation on a new cohort remains future work.

This is independent, AI-assisted research. Manuscript drafts remain private and outside Git history. Public code, licensed source excerpts, predictions, figures and checks provide the inspectable portfolio evidence.
