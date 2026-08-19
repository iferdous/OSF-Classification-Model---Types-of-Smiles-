# Phase 2 Label Contract

This document defines the labels used before model training. The goal is to
label every available local media file conservatively and avoid pretending a
dataset supports labels it does not actually contain.

## Contract Labels

| Contract label | Meaning | Source support |
| --- | --- | --- |
| `reward` | Reward smile subtype | OSF Reward videos |
| `affiliative` | Affiliative smile subtype | OSF Affiliation videos |
| `dominance` | Dominance smile subtype | OSF Dominance videos |
| `neutral` | Neutral/non-smiling face | FEI aligned `a` images |
| `generic_smile` | Generic smile, not subtype-labeled | FEI aligned `b` images |
| `frown_candidate` | Negative expression that may help frown exploration | OSF Sadness videos |
| `negative_other` | Non-target negative expression | OSF Anger and Disgust videos |
| `audit_only` | Indexed but not used as a training label yet | FEI original images |

## Contract Manifest

Main file:

```text
data/manifests/label_contract.csv
```

Columns:

```text
source
media_type
relative_path
identity_id
source_label
contract_label
training_role
split_group
is_smile_subtype
include_in_phase2_training
label_status
notes
```

## Source Mapping

OSF mappings:

| OSF source label | Contract label | Training role |
| --- | --- | --- |
| `Reward` | `reward` | `smile_subtype_positive` |
| `Affiliation` | `affiliative` | `smile_subtype_positive` |
| `Dominance` | `dominance` | `smile_subtype_positive` |
| `Sadness` | `frown_candidate` | `negative_oob` |
| `Anger` | `negative_other` | `negative_oob` |
| `Disgust` | `negative_other` | `negative_oob` |

FEI mappings:

| FEI source pattern | Contract label | Training role |
| --- | --- | --- |
| aligned `a` image | `neutral` | `basic_expression_support` |
| aligned `b` image | `generic_smile` | `basic_expression_support` |
| original image variants | `audit_only` | `audit_only` |

## Training Use

Phase 2 training may use:

- OSF `reward`, `affiliative`, and `dominance` videos for smile subtype exploration.
- FEI `neutral` and `generic_smile` aligned images for basic expression support.
- OSF `frown_candidate` and `negative_other` videos as out-of-distribution or negative-expression checks.

Phase 2 should not use:

- FEI images for reward/affiliative/dominance labels.
- FEI original images as training labels until their variants are reviewed.
- OSF sadness as confirmed app `frown` ground truth without additional review.

## Validation Summary

The generated contract currently covers:

- 3,290 total media files.
- 90 OSF videos.
- 3,200 FEI images.
- 45 OSF smile-subtype videos.
- 400 FEI aligned neutral/generic-smile images.
- 2,800 FEI original images marked `audit_only`.

Every row has a split group:

- OSF split groups are actor names.
- FEI split groups are subject IDs.

This prevents later model training from accidentally putting the same person in
both training and validation/test sets.
