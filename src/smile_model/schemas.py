from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/Volumes/SSK/DataSet of Smiles")
OSF_ROOT = DATA_ROOT / "OSF Smile Types"
OSF_VIDEO_DIR = OSF_ROOT / "Types of Smiles"
FEI_ROOT = DATA_ROOT / "FEI Face Database"

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

