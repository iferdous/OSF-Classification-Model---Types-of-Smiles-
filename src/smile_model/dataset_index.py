from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from .schemas import (
    DATA_ROOT,
    FEI_ROOT,
    OSF_EXPECTED_ACTORS,
    OSF_EXPECTED_LABELS,
    OSF_LABEL_MAP,
    OSF_VIDEO_DIR,
    FeiImageRecord,
    OsfVideoRecord,
)


def _relative(path: Path, root: Path = DATA_ROOT) -> str:
    return path.relative_to(root).as_posix()


def index_osf_videos(video_dir: Path = OSF_VIDEO_DIR) -> list[OsfVideoRecord]:
    records: list[OsfVideoRecord] = []
    for path in sorted(video_dir.glob("*.mp4")):
        if "_" not in path.stem:
            raise ValueError(f"OSF filename does not match Actor_Label.mp4: {path}")
        actor, source_label = path.stem.rsplit("_", 1)
        if source_label not in OSF_LABEL_MAP:
            raise ValueError(f"Unexpected OSF source label {source_label!r} in {path.name}")
        records.append(
            OsfVideoRecord(
                source="osf_smile_types",
                relative_path=_relative(path),
                actor=actor,
                source_label=source_label,
                target_label=OSF_LABEL_MAP[source_label],
                split_group=actor,
                file_ext=path.suffix.lower().lstrip("."),
            )
        )
    return records


def _index_fei_aligned(folder: Path) -> list[FeiImageRecord]:
    records: list[FeiImageRecord] = []
    for path in sorted(folder.glob("*.jpg")):
        match = re.fullmatch(r"(\d+)([ab])", path.stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Unexpected FEI aligned filename: {path.name}")
        subject_id, variant = match.groups()
        variant = variant.lower()
        records.append(
            FeiImageRecord(
                source="fei_face_database",
                relative_path=_relative(path),
                subject_id=subject_id,
                image_variant=f"aligned_{variant}",
                target_label="neutral" if variant == "a" else "generic_smile",
                file_ext=path.suffix.lower().lstrip("."),
            )
        )
    return records


def _index_fei_original(folder: Path) -> list[FeiImageRecord]:
    records: list[FeiImageRecord] = []
    for path in sorted(folder.glob("*.jpg")):
        match = re.fullmatch(r"(\d+)-(\d+)", path.stem)
        if not match:
            raise ValueError(f"Unexpected FEI original filename: {path.name}")
        subject_id, variant = match.groups()
        records.append(
            FeiImageRecord(
                source="fei_face_database",
                relative_path=_relative(path),
                subject_id=subject_id,
                image_variant=f"original_{variant}",
                target_label="audit_only",
                file_ext=path.suffix.lower().lstrip("."),
            )
        )
    return records


def index_fei_images(fei_root: Path = FEI_ROOT) -> list[FeiImageRecord]:
    records: list[FeiImageRecord] = []
    for folder in sorted(fei_root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name.startswith("frontalimages_manuallyaligned"):
            records.extend(_index_fei_aligned(folder))
        elif folder.name.startswith("originalimages"):
            records.extend(_index_fei_original(folder))
    return records


def write_csv(path: Path, rows: list[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def validate_osf(records: list[OsfVideoRecord]) -> None:
    if len(records) != 90:
        raise AssertionError(f"Expected 90 OSF videos, found {len(records)}")
    actors = {r.actor for r in records}
    if actors != OSF_EXPECTED_ACTORS:
        raise AssertionError(f"Unexpected OSF actors: {sorted(actors)}")
    by_actor: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_actor[record.actor].add(record.source_label)
    for actor, labels in sorted(by_actor.items()):
        if labels != OSF_EXPECTED_LABELS:
            raise AssertionError(f"{actor} has labels {sorted(labels)}")


def summarize(osf_records: list[OsfVideoRecord], fei_records: list[FeiImageRecord]) -> str:
    osf_labels = Counter(r.source_label for r in osf_records)
    fei_labels = Counter(r.target_label for r in fei_records)
    fei_variants = Counter(r.image_variant.split("_")[0] for r in fei_records)
    return "\n".join(
        [
            f"OSF videos: {len(osf_records)}",
            f"OSF actors: {len({r.actor for r in osf_records})}",
            f"OSF source labels: {dict(sorted(osf_labels.items()))}",
            f"FEI images: {len(fei_records)}",
            f"FEI target labels: {dict(sorted(fei_labels.items()))}",
            f"FEI image groups: {dict(sorted(fei_variants.items()))}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 dataset manifests.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()

    osf_records = index_osf_videos(args.data_root / "OSF Smile Types" / "Types of Smiles")
    fei_records = index_fei_images(args.data_root / "FEI Face Database")
    validate_osf(osf_records)

    write_csv(
        args.out_dir / "osf_videos.csv",
        osf_records,
        ["source", "relative_path", "actor", "source_label", "target_label", "split_group", "file_ext"],
    )
    write_csv(
        args.out_dir / "fei_images.csv",
        fei_records,
        ["source", "relative_path", "subject_id", "image_variant", "target_label", "file_ext"],
    )
    print(summarize(osf_records, fei_records))


if __name__ == "__main__":
    main()

