# Data Handling

Raw OSF and FEI media files are intentionally not stored in this repository.

Expected local source folder:

```text
/Volumes/SSK/DataSet of Smiles/
```

Generated Phase 1 manifests live in:

```text
data/manifests/
```

The manifests contain file paths, labels, actors/subjects, and split groups so
later modeling scripts can reproduce the dataset index without committing raw
videos or images.

Phase 2 adds:

```text
data/manifests/label_contract.csv
```

That file is the strict source of truth for how every indexed video/image may be
used during model development.

Phase 3 adds:

```text
data/features/phase3_frame_features.csv
data/features/phase3_media_features.csv
```

Those files are generated only from label-contract rows marked for training use
by default. FEI original images remain excluded as `audit_only`.

Phase 4 adds:

```text
data/model_outputs/phase4_basic_expression_predictions.csv
data/model_outputs/phase4_smile_subtype_predictions.csv
data/model_outputs/phase4_metrics.json
```

Those files store cross-validated baseline predictions and metrics. They do not
replace the Phase 2 label contract.
