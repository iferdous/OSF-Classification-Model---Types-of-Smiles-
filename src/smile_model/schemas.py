from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/Volumes/SSK/DataSet of Smiles")
OSF_ROOT = DATA_ROOT / "OSF Smile Types"
OSF_VIDEO_DIR = OSF_ROOT / "Types of Smiles"
FEI_ROOT = DATA_ROOT / "FEI Face Database"
FEATURE_EXTRACTOR_VERSION = "phase3-opencv-v1"

OSF_EXPECTED_ACTORS = {
    "Andrew",
    "Anna",
    "Anthony",
    "Ashley",
    "Dmonte",
    "Jessica",
    "Joeseph",
    "Jourdan",
    "KJ",
    "Lulu",
    "Marcus",
    "Mary",
    "Maurice",
    "Naomi",
    "Zach",
}

OSF_BEST_ACTORS = ["Andrew", "Dmonte", "Jessica", "Joeseph", "Jourdan", "Lulu"]

OSF_EXPECTED_LABELS = {
    "Reward",
    "Affiliation",
    "Dominance",
    "Anger",
    "Disgust",
    "Sadness",
}

OSF_LABEL_MAP = {
    "Reward": "reward",
    "Affiliation": "affiliative",
    "Dominance": "dominance",
    "Sadness": "frown_candidate",
    "Anger": "negative_other",
    "Disgust": "negative_other",
}

SMILE_SUBTYPE_SOURCE_LABELS = {"Reward", "Affiliation", "Dominance"}
SMILE_SUBTYPE_TARGET_LABELS = {"reward", "affiliative", "dominance"}

CONTRACT_LABELS = {
    "neutral",
    "generic_smile",
    "reward",
    "affiliative",
    "dominance",
    "frown_candidate",
    "negative_other",
    "audit_only",
}

TRAINING_ROLES = {
    "smile_subtype_positive",
    "basic_expression_support",
    "negative_oob",
    "audit_only",
}

OSF_TRAINING_ROLE_MAP = {
    "Reward": "smile_subtype_positive",
    "Affiliation": "smile_subtype_positive",
    "Dominance": "smile_subtype_positive",
    "Sadness": "negative_oob",
    "Anger": "negative_oob",
    "Disgust": "negative_oob",
}

OSF_LABEL_NOTES = {
    "Reward": "Published OSF reward-smile stimulus; valid smile-subtype supervision.",
    "Affiliation": "Published OSF affiliation-smile stimulus; mapped to app label affiliative.",
    "Dominance": "Published OSF dominance-smile stimulus; valid smile-subtype supervision.",
    "Sadness": "Negative expression; useful as a frown candidate but not validated as app frown ground truth.",
    "Anger": "Negative expression; useful as non-target/out-of-distribution data.",
    "Disgust": "Negative expression; useful as non-target/out-of-distribution data.",
}

FEI_LABEL_NOTES = {
    "neutral": "Manually aligned FEI a-image; usable for basic neutral support.",
    "generic_smile": "Manually aligned FEI b-image; usable for basic smile support only.",
    "audit_only": "Original FEI pose/variant image; indexed for audit but not assigned a training label yet.",
}


@dataclass(frozen=True)
class OsfVideoRecord:
    source: str
    relative_path: str
    actor: str
    source_label: str
    target_label: str
    split_group: str
    file_ext: str


@dataclass(frozen=True)
class FeiImageRecord:
    source: str
    relative_path: str
    subject_id: str
    image_variant: str
    target_label: str
    file_ext: str


@dataclass(frozen=True)
class LabelContractRecord:
    source: str
    media_type: str
    relative_path: str
    identity_id: str
    source_label: str
    contract_label: str
    training_role: str
    split_group: str
    is_smile_subtype: bool
    include_in_phase2_training: bool
    label_status: str
    notes: str


@dataclass(frozen=True)
class FrameFeatureRecord:
    source: str
    media_type: str
    relative_path: str
    identity_id: str
    contract_label: str
    training_role: str
    split_group: str
    is_smile_subtype: bool
    label_status: str
    sample_kind: str
    sample_index: int
    frame_index: int
    timestamp_ms: float
    image_width: int
    image_height: int
    face_count: int
    face_found: bool
    face_x_pct: float | None
    face_y_pct: float | None
    face_w_pct: float | None
    face_h_pct: float | None
    face_area_pct: float | None
    face_center_x_pct: float | None
    face_center_y_pct: float | None
    smile_count: int
    smile_found: bool
    smile_x_pct: float | None
    smile_y_pct: float | None
    smile_w_pct: float | None
    smile_h_pct: float | None
    smile_area_pct: float | None
    smile_to_face_width_ratio: float | None
    smile_to_face_height_ratio: float | None
    brightness_mean: float
    brightness_std: float
    contrast_rms: float
    sharpness_laplacian_var: float
    edge_density: float
    lower_face_brightness_mean: float | None
    lower_face_edge_density: float | None
    lower_face_symmetry_mad: float | None
    lower_face_dark_ratio: float | None
    extraction_status: str
    extractor_version: str
    notes: str


@dataclass(frozen=True)
class MediaFeatureRecord:
    source: str
    media_type: str
    relative_path: str
    identity_id: str
    contract_label: str
    training_role: str
    split_group: str
    is_smile_subtype: bool
    frames_sampled: int
    frames_ok: int
    face_detection_rate: float
    smile_detection_rate: float
    image_width_median: float
    image_height_median: float
    brightness_mean: float
    brightness_std: float
    contrast_rms_mean: float
    sharpness_laplacian_var_mean: float
    edge_density_mean: float
    lower_face_brightness_mean: float | None
    lower_face_edge_density_mean: float | None
    lower_face_symmetry_mad_mean: float | None
    lower_face_dark_ratio_mean: float | None
    face_area_pct_mean: float | None
    face_area_pct_std: float | None
    face_center_x_pct_mean: float | None
    face_center_y_pct_mean: float | None
    smile_area_pct_mean: float | None
    smile_area_pct_std: float | None
    smile_to_face_width_ratio_mean: float | None
    smile_to_face_width_ratio_std: float | None
    duration_ms: float | None
    extraction_status: str
    extractor_version: str
    notes: str
