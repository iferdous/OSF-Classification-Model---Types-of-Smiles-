from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .dataset_index import index_fei_images, index_osf_videos
from .schemas import (
    CONTRACT_LABELS,
    DATA_ROOT,
    FEI_LABEL_NOTES,
    FEI_ROOT,
    OSF_LABEL_NOTES,
    OSF_TRAINING_ROLE_MAP,
    OSF_VIDEO_DIR,
    SMILE_SUBTYPE_TARGET_LABELS,
    TRAINING_ROLES,
    FeiImageRecord,
    LabelContractRecord,
    OsfVideoRecord,
)

CONTRACT_FIELDS = [
    "source",
    "media_type",
    "relative_path",
    "identity_id",
    "source_label",
    "contract_label",
    "training_role",
    "split_group",
    "is_smile_subtype",
    "include_in_phase2_training",
    "label_status",
    "notes",
]


def osf_to_contract(record: OsfVideoRecord) -> LabelContractRecord:
    is_subtype = record.target_label in SMILE_SUBTYPE_TARGET_LABELS
    return LabelContractRecord(
        source=record.source,
        media_type="video",
        relative_path=record.relative_path,
        identity_id=record.actor,
        source_label=record.source_label,
        contract_label=record.target_label,
        training_role=OSF_TRAINING_ROLE_MAP[record.source_label],
        split_group=record.split_group,
        is_smile_subtype=is_subtype,
        include_in_phase2_training=True,
        label_status="published_stimulus_label",
        notes=OSF_LABEL_NOTES[record.source_label],
    )


def fei_to_contract(record: FeiImageRecord) -> LabelContractRecord:
    include = record.target_label in {"neutral", "generic_smile"}
    role = "basic_expression_support" if include else "audit_only"
    return LabelContractRecord(
        source=record.source,
        media_type="image",
        relative_path=record.relative_path,
        identity_id=record.subject_id,
        source_label=record.image_variant,
        contract_label=record.target_label,
        training_role=role,
        split_group=f"fei_subject_{record.subject_id}",
        is_smile_subtype=False,
        include_in_phase2_training=include,
        label_status="filename_rule" if include else "not_training_labeled",
        notes=FEI_LABEL_NOTES[record.target_label],
    )


def build_label_contract(
    osf_video_dir: Path = OSF_VIDEO_DIR,
    fei_root: Path = FEI_ROOT,
) -> list[LabelContractRecord]:
    records = [osf_to_contract(record) for record in index_osf_videos(osf_video_dir)]
    records.extend(fei_to_contract(record) for record in index_fei_images(fei_root))
    validate_label_contract(records)
    return sorted(records, key=lambda r: (r.source, r.media_type, r.relative_path))


def validate_label_contract(records: list[LabelContractRecord]) -> None:
    paths = [record.relative_path for record in records]
    if len(paths) != len(set(paths)):
        duplicates = [path for path, count in Counter(paths).items() if count > 1]
        raise AssertionError(f"Duplicate contract paths: {duplicates[:10]}")
    for record in records:
        if record.contract_label not in CONTRACT_LABELS:
            raise AssertionError(f"Unexpected contract label: {record}")
        if record.training_role not in TRAINING_ROLES:
            raise AssertionError(f"Unexpected training role: {record}")
        if record.source == "fei_face_database" and record.is_smile_subtype:
            raise AssertionError(f"FEI row incorrectly marked as smile subtype: {record}")
        if record.source == "fei_face_database" and record.contract_label in SMILE_SUBTYPE_TARGET_LABELS:
            raise AssertionError(f"FEI row incorrectly mapped to smile subtype: {record}")
        if record.is_smile_subtype and record.contract_label not in SMILE_SUBTYPE_TARGET_LABELS:
            raise AssertionError(f"Subtype flag disagrees with contract label: {record}")


def write_contract(path: Path, records: list[LabelContractRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONTRACT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def summarize(records: list[LabelContractRecord]) -> str:
    labels = Counter(record.contract_label for record in records)
    roles = Counter(record.training_role for record in records)
    sources = Counter(record.source for record in records)
    included = Counter(str(record.include_in_phase2_training) for record in records)
    return "\n".join(
        [
            f"Contract rows: {len(records)}",
            f"Sources: {dict(sorted(sources.items()))}",
            f"Contract labels: {dict(sorted(labels.items()))}",
            f"Training roles: {dict(sorted(roles.items()))}",
            f"Include in Phase 2 training: {dict(sorted(included.items()))}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 2 label contract.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out", type=Path, default=Path("data/manifests/label_contract.csv"))
    args = parser.parse_args()
    records = build_label_contract(
        args.data_root / "OSF Smile Types" / "Types of Smiles",
        args.data_root / "FEI Face Database",
    )
    write_contract(args.out, records)
    print(summarize(records))


if __name__ == "__main__":
    main()
