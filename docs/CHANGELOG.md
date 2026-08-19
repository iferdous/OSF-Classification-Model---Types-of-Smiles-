# Project Changelog

*Note to team: Every time you make a significant update to the code, please copy the blank template below and paste a filled-out version right below the "Update History" header (so the most recent change is always on top).*

## Blank Template (Copy this)

* **Date:** [DD-MM-YYYY]
* **Author:** [Your Name]
* **Changes Made:** [Short title of the update]

* **Previous behavior:** 
[What did the app do before this change?]
* **New behavior:** 
[What does the app do now?]
* **Why this matters:** 
[Why was this change necessary?]

---

## Update History

* **Date:** 19-08-2026
* **Author:** Ismam Ferdous
* **Changes Made:** Add Phase 3 feature extraction

* **Previous behavior:**
The repo had dataset manifests and a label contract, but it did not yet create numeric feature files for model training.
* **New behavior:**
The repo now extracts OpenCV-based frame/image features and per-media summary features from the reviewed label contract. It also documents the feature extraction layer and keeps FEI original images excluded as audit-only data.
* **Why this matters:**
This creates a safer bridge between raw smile media and future model training. The team can inspect the feature outputs before training, confirm labels stay attached correctly, and add stronger MediaPipe landmark features later without changing the label contract.

---

* **Date:** 18-08-2026
* **Author:** Ismam Ferdous
* **Changes Made:** Add Phase 2 label contract

* **Previous behavior:**
The repo indexed the OSF videos and FEI images, but each file did not yet have a strict training role or label-status rule.
* **New behavior:**
Every indexed media file now has a contract label, training role, split group, label status, and notes explaining how it can or cannot be used.
* **Why this matters:**
This prevents the model work from accidentally treating FEI smiles as reward/affiliative/dominance labels or treating OSF sadness as confirmed frown ground truth.

---
