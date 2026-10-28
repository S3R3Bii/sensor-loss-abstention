# Measurements, intervention and rights

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X. (2013), *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

- [Official dataset, methods and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
- [Official archive](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip).
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).

Public redistribution retains original attribution. `data/manifest.json` records the archive hash and original/decompressed and compressed hashes for the selected files. The source's 24 selected files are gzip-compressed without altering their contents; video and 561-feature tables are excluded. This work adds window summaries, fixed sensor-loss interventions and model/evaluation outputs. No endorsement by the original researchers or UCI is implied.

The recordings contain 30 anonymous participants and six instructed activities. The source preprocessed the 50 Hz inertial signals into 128-sample windows with 50% overlap. This study therefore groups by participant and does not treat windows as independent people. It does not establish performance under natural sensor failure, other devices or unconstrained activity. The controlled missing-modality patterns are simulated, while the underlying measurements and activity labels come from the source.

No clinical/private participant records, video, credentials or manuscript drafts are included. Algorithm settings follow the installed scikit-learn 1.7.0 implementation; [official logistic-regression documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html) describes C as inverse regularization strength. Default/inherited choices and fixed settings are recorded in the protocol and candidate manifest.

Source notice discrepancy checked October 28, 2026: the unchanged 2013 archive README contains an older noncommercial-use sentence, while the current official UCI dataset landing page explicitly identifies CC BY 4.0 and sharing/adaptation for any purpose. This package records the current dataset-specific UCI license and preserves the old notice rather than silently removing it. The legacy README is byte-preserved; its non-UTF-8 text was separately inspected for the publication scan. No endorsement or independent legal determination is claimed.
