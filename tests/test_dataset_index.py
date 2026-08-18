from pathlib import Path

from smile_model.dataset_index import index_fei_images, index_osf_videos
from smile_model.schemas import (
    FEI_ROOT,
    OSF_EXPECTED_ACTORS,
    OSF_EXPECTED_LABELS,
    OSF_VIDEO_DIR,
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

