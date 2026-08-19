# Phase 3 Feature Extraction

Phase 3 converts the reviewed Phase 2 label contract into numeric features that
can be used by later training scripts. This phase still does not train a model.
It only produces verified feature tables.

## Generated Files

```text
data/features/phase3_frame_features.csv
data/features/phase3_media_features.csv
```

The frame feature file stores one row per sampled video frame or FEI image.
The media feature file stores one summarized row per video/image.

Current generated counts:

- 1,480 frame/image feature rows.
- 490 media summary rows.
- 0 `audit_only` rows.
- Extractor version: `phase3-opencv-v1`.

## Included Data

The default extractor includes only rows marked for Phase 2 training in the
label contract:

- 45 OSF smile-subtype videos.
- 45 OSF negative-expression videos.
- 400 FEI aligned neutral/generic-smile images.

FEI original images remain excluded as `audit_only`.

## Feature Families

Current extractor:

- image dimensions
- brightness and contrast
- sharpness
- edge density
- OpenCV face detection count and largest face box
- OpenCV smile-region detection inside the lower face
- lower-face brightness, edge density, dark-pixel ratio, and symmetry proxy
- per-video aggregation across sampled frames

Important limitation:

These are not MediaPipe landmarks and they are not a final smile classifier.
They are an auditable first feature table that lets us check whether the
dataset can be read consistently and whether basic face/smile quality signals
are available before model training.

Observed QA note:

OpenCV face detection found faces consistently in this dataset, but OpenCV's
smile-region detector produced false positives in some neutral and negative
examples. The `smile_found` and smile-box columns should therefore be treated
as model input features, not as final labels or predictions.

## Why This Matters

Training should not start directly from raw MP4/JPG files. Phase 3 gives us a
stable middle layer:

```text
raw media -> label contract -> frame features -> media features -> model training
```

That makes the future model easier to debug, easier to reproduce, and safer to
connect back to the video-conferencing app.

## Future Upgrade Path

The next extractor can add MediaPipe blendshapes/landmarks when the Python
environment supports it. Those features should be added as a new extractor
version instead of overwriting this Phase 3 OpenCV feature set.
