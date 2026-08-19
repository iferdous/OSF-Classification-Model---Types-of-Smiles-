import csv
from pathlib import Path

from smile_model.dataset_index import index_fei_images, index_osf_videos
from smile_model.feature_extraction import build_features
from smile_model.label_contract import build_label_contract
from smile_model.schemas import (
    FEI_ROOT,
    OSF_EXPECTED_ACTORS,
    OSF_EXPECTED_LABELS,
    OSF_VIDEO_DIR,
    SMILE_SUBTYPE_SOURCE_LABELS,
    SMILE_SUBTYPE_TARGET_LABELS,
)


def test_osf_manifest_shape() -> None:
    records = index_osf_videos(OSF_VIDEO_DIR)
    assert len(records) == 90
    assert {record.actor for record in records} == OSF_EXPECTED_ACTORS
    for actor in OSF_EXPECTED_ACTORS:
        labels = {record.source_label for record in records if record.actor == actor}
        assert labels == OSF_EXPECTED_LABELS


def test_osf_smile_subtypes_only_from_rad_labels() -> None:
    records = index_osf_videos(OSF_VIDEO_DIR)
    subtype_sources = {
        record.source_label
        for record in records
        if record.target_label in SMILE_SUBTYPE_TARGET_LABELS
    }
    assert subtype_sources == {"Reward", "Affiliation", "Dominance"}


def test_fei_labels_do_not_create_smile_subtypes() -> None:
    records = index_fei_images(FEI_ROOT)
    assert len(records) == 3200
    labels = {record.target_label for record in records}
    assert "neutral" in labels
    assert "generic_smile" in labels
    assert "audit_only" in labels
    assert labels.isdisjoint(SMILE_SUBTYPE_TARGET_LABELS)


def test_grids_exist_after_generation() -> None:
    out_dir = Path("reports/randy_grid_examples")
    expected = [
        out_dir / "osf_best_actor_smile_grid.png",
        out_dir / "osf_full_expression_grid.png",
        out_dir / "fei_neutral_smile_grid.png",
    ]
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0


def test_label_contract_covers_every_indexed_item_once() -> None:
    osf = index_osf_videos(OSF_VIDEO_DIR)
    fei = index_fei_images(FEI_ROOT)
    contract = build_label_contract(OSF_VIDEO_DIR, FEI_ROOT)
    indexed_paths = {record.relative_path for record in osf} | {record.relative_path for record in fei}
    contract_paths = {record.relative_path for record in contract}
    assert len(contract) == 3290
    assert contract_paths == indexed_paths


def test_label_contract_smile_subtypes_are_osf_only() -> None:
    contract = build_label_contract(OSF_VIDEO_DIR, FEI_ROOT)
    subtype_rows = [record for record in contract if record.is_smile_subtype]
    assert len(subtype_rows) == 45
    assert {record.source for record in subtype_rows} == {"osf_smile_types"}
    assert {record.source_label for record in subtype_rows} == SMILE_SUBTYPE_SOURCE_LABELS
    assert {record.contract_label for record in subtype_rows} == SMILE_SUBTYPE_TARGET_LABELS


def test_label_contract_fei_training_labels_are_basic_only() -> None:
    contract = build_label_contract(OSF_VIDEO_DIR, FEI_ROOT)
    fei_rows = [record for record in contract if record.source == "fei_face_database"]
    fei_training_rows = [record for record in fei_rows if record.include_in_phase2_training]
    assert len(fei_rows) == 3200
    assert len(fei_training_rows) == 400
    assert {record.contract_label for record in fei_training_rows} == {"neutral", "generic_smile"}
    assert all(not record.is_smile_subtype for record in fei_rows)


def test_label_contract_negative_rows_are_not_subtypes() -> None:
    contract = build_label_contract(OSF_VIDEO_DIR, FEI_ROOT)
    negative_rows = [
        record
        for record in contract
        if record.contract_label in {"frown_candidate", "negative_other"}
    ]
    assert len(negative_rows) == 45
    assert all(record.training_role == "negative_oob" for record in negative_rows)
    assert all(not record.is_smile_subtype for record in negative_rows)


def test_phase3_feature_builder_keeps_labels_attached() -> None:
    frame_rows, media_rows = build_features(video_samples=3)
    assert len(media_rows) == 490
    assert len(frame_rows) == 670
    assert {row.contract_label for row in media_rows} == {
        "affiliative",
        "dominance",
        "frown_candidate",
        "generic_smile",
        "negative_other",
        "neutral",
        "reward",
    }
    assert "audit_only" not in {row.contract_label for row in media_rows}
    assert all(row.extractor_version == "phase3-opencv-v1" for row in frame_rows)
    assert all(row.frames_sampled > 0 for row in media_rows)


def test_phase3_feature_files_exist_after_generation() -> None:
    expected_counts = {
        Path("data/features/phase3_frame_features.csv"): 1480,
        Path("data/features/phase3_media_features.csv"): 490,
    }
    for path, expected_rows in expected_counts.items():
        assert path.exists()
        assert path.stat().st_size > 0
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == expected_rows
        assert "audit_only" not in {row["contract_label"] for row in rows}
