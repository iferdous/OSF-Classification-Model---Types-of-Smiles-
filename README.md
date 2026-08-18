# OSF Smile-Type Classification Model

This repository is a separate modeling workspace for classifying smile and
expression categories that may later support the Niedenthal video-conferencing
application.

## Goal

Build and validate a conservative classifier for:

- `neutral`
- `frown`
- `reward`
- `affiliative`
- `dominance`
- `unknown / uncertain`

The immediate Phase 1 goal is not model training. Phase 1 audits the available
datasets, creates clean manifests, and generates visual grids that can be shown
to the research group.

## Data Sources

Raw data is expected locally at:

```text
/Volumes/SSK/DataSet of Smiles/
```

Sources:

- OSF Smile Types / Martin et al. 2021: reward, affiliation, and dominance smile videos.
- FEI Face Database: neutral and generic smiling face images.

Raw `.mp4` and `.jpg` files are intentionally excluded from Git.

## Phase 1 Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate manifests:

```bash
PYTHONPATH=src python3 -m smile_model.dataset_index
```

Generate visual grids:

```bash
PYTHONPATH=src python3 -m smile_model.make_grids
```

Run tests:

```bash
PYTHONPATH=src pytest
```

## App Integration Target

Future model output should match the video app's expression contract:

```text
label
smileType
labelConfidence
smileTypeConfidence
classifierMode
classifierVersion
uncertain
```

The classifier should be allowed to return `unknown` or `uncertain` when the
model is not confident enough, especially for affiliative vs dominance smiles.

