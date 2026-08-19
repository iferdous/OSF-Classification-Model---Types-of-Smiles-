# Phase 4 Baseline Models

Phase 4 trains conservative baseline classifiers from the Phase 3 media feature
table. These baselines are intended for measurement and comparison, not final
deployment in the video-conferencing app.

## Generated Files

```text
data/model_outputs/phase4_basic_expression_predictions.csv
data/model_outputs/phase4_smile_subtype_predictions.csv
data/model_outputs/phase4_metrics.json
```

Local model pickle files are written under:

```text
models/phase4/
```

Those pickle files are intentionally ignored by Git. The committed artifacts are
the source code, prediction CSVs, metrics, and documentation.

## Baselines

Basic expression baseline:

- Source: FEI aligned images only.
- Labels: `neutral`, `smile`.
- Rows: 400.
- Validation: grouped by FEI subject with `GroupKFold(n_splits=5)`.

Smile subtype baseline:

- Source: OSF Smile Types videos only.
- Labels: `reward`, `affiliative`, `dominance`.
- Rows: 45.
- Validation: leave-one-actor-out by OSF actor.

## App-Style Output Fields

Prediction CSVs include:

- `predicted_label`
- `prediction_confidence`
- `labelConfidence`
- `smileType`
- `smileTypeConfidence`
- `classifier_mode`
- `classifier_version`
- `uncertain`

Rows with confidence below `0.60` are marked `uncertain`.

## Current Results

Basic expression baseline:

- Accuracy: `0.81`.
- Macro F1: `0.810`.
- Uncertain rows: `58 / 400`.
- Coverage after uncertainty threshold: `0.855`.
- Confusion matrix order: `neutral`, `smile`.
- Confusion matrix:

```text
[[155, 45],
 [ 31, 169]]
```

Smile subtype baseline:

- Accuracy: `0.422`.
- Macro F1: `0.427`.
- Uncertain rows: `10 / 45`.
- Coverage after uncertainty threshold: `0.778`.
- Confusion matrix order: `affiliative`, `dominance`, `reward`.
- Confusion matrix:

```text
[[7, 3, 5],
 [1, 6, 8],
 [5, 4, 6]]
```

## Important Limits

The subtype model is trained on only 45 OSF videos, so it should be treated as a
research baseline. It is useful for testing the full modeling workflow and
finding failure patterns, but it is not strong enough to call production-ready.

The frown class is not trained as a final app label yet. OSF `Sadness` remains a
`frown_candidate`, not confirmed app frown ground truth.

## Next Step

Phase 5 should compare these baseline results against stronger feature sets,
especially MediaPipe blendshapes/landmarks when the Python environment supports
them.
