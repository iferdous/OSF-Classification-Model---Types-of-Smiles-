# Smile-Type Classification Model

This repository is a separate modeling workspace for exploring automatic
classification of smile and expression types for the Niedenthal video
conferencing project.

## Goal

Build and validate a conservative classifier for:

- `neutral`
- `frown`
- `reward`
- `affiliative`
- `dominance`
- `unknown / uncertain`

The model should return confidence scores and be allowed to say `uncertain`
instead of forcing a smile subtype when the evidence is weak.

## Data Sources

- **OSF Smile Types / Martin et al. 2021**: video stimuli for reward,
  affiliative, and dominance smiles, plus anger, disgust, and sadness examples.
- **FEI Face Database**: neutral and generic smiling face images used for basic
  neutral/smile support, not smile-subtype labels.

Raw dataset files are kept outside this repo and are not committed to GitHub.

## Current Phase

The repository currently includes the Phase 1 dataset audit, the Phase 2 label
contract, the Phase 3 feature extraction layer, and Phase 4 baseline models.
Together, these define what each local video/image is allowed to mean, convert
reviewed media into numeric feature files, and measure first-pass classifiers
with person-separated validation.

Current artifacts:

- dataset manifests for OSF videos and FEI images
- a unified label contract for every indexed video and image
- frame/image feature tables and per-media feature summaries
- baseline prediction CSVs and metrics
- grid-style visual summaries for review
- a short dataset summary report
- validation tests for the dataset assumptions

The current models are baselines only. They are meant to guide the next research
iteration, not to be treated as final app-ready classifiers.

## Documentation

- `reports/dataset_summary.md`
- `reports/label_contract.md`
- `reports/feature_extraction.md`
- `reports/baseline_models.md`
- `docs/CHANGELOG.md`

## Intended App Contract

Future model output should match the app’s expression fields:

```text
label
smileType
labelConfidence
smileTypeConfidence
classifierMode
classifierVersion
uncertain
```
