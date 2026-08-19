# Phase 1 Dataset Summary

## Sources

Raw data folder:

```text
/Volumes/SSK/DataSet of Smiles/
```

Primary sources:

- Martin et al. 2021 OSF Smile Types dataset: https://osf.io/qs35g/
- Martin et al. 2021 paper, "Evidence for Distinct Facial Signals of Reward, Affiliation, and Dominance": https://link.springer.com/article/10.1007/s42761-020-00024-8
- FEI Face Database: https://fei.edu.br/~cet/facedatabase.html
- Gao et al. 2026 "Shades of Smiles" paper for related smile-type stimulus-generation context: https://link.springer.com/article/10.1007/s00426-026-02263-z

## Local Inventory

OSF Smile Types:

- 90 MP4 videos.
- 15 actors.
- 6 source labels per actor: `Reward`, `Affiliation`, `Dominance`, `Anger`, `Disgust`, `Sadness`.
- OSF README notes that the best 6 actors from stimulus selection were `Andrew`, `Dmonte`, `Jessica`, `Joeseph`, `Jourdan`, and `Lulu`.

FEI Face Database:

- 3,200 local JPG images.
- 2,800 original images across 200 subjects.
- 400 manually aligned frontal images.
- Aligned `a` variants are treated as `neutral`.
- Aligned `b` variants are treated as `generic_smile`.
- Original FEI images are indexed as `audit_only` until a later phase decides how to use individual poses/variants.

## Label Mapping

OSF mapping:

| Source label | Phase 1 target label | Modeling role |
| --- | --- | --- |
| Reward | reward | Smile subtype |
| Affiliation | affiliative | Smile subtype |
| Dominance | dominance | Smile subtype |
| Sadness | frown_candidate | Negative/frown candidate, not final frown ground truth |
| Anger | negative_other | Out-of-distribution / non-target expression |
| Disgust | negative_other | Out-of-distribution / non-target expression |

FEI mapping:

| Source pattern | Phase 1 target label | Modeling role |
| --- | --- | --- |
| aligned `a` images | neutral | Basic expression support |
| aligned `b` images | generic_smile | Basic smile support |
| original images | audit_only | Indexed for inspection only |

## What Can Be Trained

Reasonable early targets:

- Basic neutral vs generic smile support from FEI.
- Reward vs affiliative vs dominance subtype exploration from OSF.
- Out-of-distribution checks using OSF anger/disgust/sadness.

Not recommended yet:

- Training a raw deep video model from scratch.
- Treating FEI generic smiles as reward/affiliative/dominance.
- Treating OSF sadness as a validated frown label without further review.

## Recommendation

Start Phase 2 with a feature-based model rather than a raw video neural network.
The OSF video dataset is valuable but small: 90 videos and 15 actors. A stronger
first pass should extract interpretable face features over time, then train
person-separated baseline models with calibrated confidence and an explicit
`uncertain` output.

## Generated Phase 1 Artifacts

Manifests:

- `data/manifests/osf_videos.csv`
- `data/manifests/fei_images.csv`
- `data/manifests/label_contract.csv`

Visual grids:

- `reports/randy_grid_examples/osf_best_actor_smile_grid.png`
- `reports/randy_grid_examples/osf_full_expression_grid.png`
- `reports/randy_grid_examples/fei_neutral_smile_grid.png`

Label contract:

- `reports/label_contract.md`
