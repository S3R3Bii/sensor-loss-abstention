# When a movement classifier should abstain after sensor loss

Frozen before source retrieval or fitting, 28 October 2026.

## Decision and scope

Compare training with intact sensors against explicit dropout augmentation. Test whether a validation-selected confidence cutoff can retain reliable classification on previously unseen participants. The intended use is deciding when to request a new measurement. This benchmark cannot establish clinical performance, real device-failure prevalence or a deployed product's accuracy.

Source: UCI Human Activity Recognition Using Smartphones, DOI 10.24432/C54S4K, CC BY 4.0. Use only anonymous participant IDs, activity labels and nine preprocessed inertial channels. Exclude video and demographic information. Attribute Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013). Source windows overlap by 50%; all windows from a participant stay together.

## Protected split

Preserve the source's official test partition without examining its performance before selection freezes. In the official training partition, participants 3, 8, 15, 22, 27 and 30 form validation; all other official-training participants form fitting. If a declared ID is absent or partitions overlap, stop rather than invent a replacement. Do not refit using validation participants.

## Features and controlled failures

For every axis of body acceleration, total acceleration and angular velocity, compute mean, population standard deviation, root mean square, peak-to-peak range and mean absolute adjacent difference over the source's 128-sample window. Retain two modality-presence indicators. Accelerometer failure removes both total- and body-acceleration features; gyroscope failure removes all angular-velocity features. Mask feature values to zero and change the appropriate indicator. Evaluate intact, accelerometer-missing and gyroscope-missing conditions equally. Both missing means compulsory abstention and is a separate edge case, not an additional population-weighted condition. Failures are synthetic interventions on public measured windows, not observed failures.

## Candidate search

Seventeen configurations: a training-prior baseline; logistic regression with L2 C in 0.01, 0.1, 1, 10, each fitted on clean-only or all three sensor conditions; random forest with 160 trees, depth 8 or 16 and minimum leaf size 5 or 20, each clean-only or augmented. Logistic scaling is fitted only on fitting rows. Forest uses two CPU workers, bootstrap sampling and square-root feature subsampling. Seed 20261028. Logistic uses lbfgs, maximum 1500 iterations, tolerance 1e-4. Convergence failures are retained and ineligible for selection. No pretrained fine-tuning.

Within training, each participant has equal total weight; augmented copies divide that weight over conditions. Primary selection objective: mean macro F1 across participant-condition groups on validation. Ties resolve by candidate name. Report pooled macro F1, balanced accuracy, accuracy, multiclass Brier score (sum across six classes per row) and log loss as additional metrics. Select the strongest clean-only model separately as the engineering comparator; retain the prior baseline.

## Abstention and final evaluation

For the selected model and clean-only comparator, search maximum-probability cutoffs 0, .4, .5, .6, .7, .8, .9, .95 and .99 on validation. Choose maximum pooled coverage subject to selective accuracy at least 90%, breaking ties by lower threshold. Require at least 100 retained validation windows; if none qualify, abstain on all. Confidence is not assumed calibrated: retain fixed-bin confidence/reliability summaries and Brier/log-loss evidence.

Freeze candidate, threshold and fitting IDs in a committed artifact before the final-test command. Evaluate once on the source test participants, with no threshold or model changes based on their outcomes. Export probabilities, identities, conditions, predictions and acceptance flags. Report each condition and participant; compare fixed thresholds with compulsory abstention when both modalities are absent. Use 2000 participant-block bootstrap draws for paired changes in accuracy, coverage and accepted-correct fraction. These intervals describe sampled participant variation, not device or patient populations.

## Acceptance criteria

Source hashes and window counts reconcile; participant partitions are disjoint; all 17 candidate outcomes and parameters are retained; fitting never consumes test rows; independent calculations reproduce selection, F1, Brier, log loss and selective metrics; error cases are tested; quantitative figures and private manuscript are rendered and inspected; public files pass a bounded privacy/rights scan. A low-coverage or missed 90% target remains a valid negative finding. Novelty is a question-specific evaluation, not a claim of a new algorithm or exhaustive research gap.

## Pre-test numerical correction

Initial validation emitted a weighted-scaler square-root warning on constant presence-indicator columns in clean-only training. The original validation search and selection are retained. Exactly constant columns are now identified and removed using fitting data alone before logistic scaling. The participant split, candidate grid, objectives and threshold rules remain unchanged; validation is repeated before examining any test result.
