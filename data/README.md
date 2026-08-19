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
