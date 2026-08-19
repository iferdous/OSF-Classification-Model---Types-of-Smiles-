from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Iterable

import cv2
import numpy as np

from .schemas import (
    DATA_ROOT,
    FEATURE_EXTRACTOR_VERSION,
    FrameFeatureRecord,
    LabelContractRecord,
    MediaFeatureRecord,
)

FRAME_FEATURE_FIELDS = list(FrameFeatureRecord.__dataclass_fields__.keys())
MEDIA_FEATURE_FIELDS = list(MediaFeatureRecord.__dataclass_fields__.keys())
DEFAULT_FRAME_SAMPLES = 12


def _parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _optional_float(values: list[float | None], reducer: str) -> float | None:
    real_values = [value for value in values if value is not None]
    if not real_values:
        return None
    if reducer == "mean":
        return float(mean(real_values))
    if reducer == "median":
        return float(median(real_values))
    if reducer == "std":
        return float(pstdev(real_values)) if len(real_values) > 1 else 0.0
    raise ValueError(f"Unsupported reducer: {reducer}")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _read_contract(path: Path, include_audit: bool) -> list[LabelContractRecord]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [
            LabelContractRecord(
                source=row["source"],
                media_type=row["media_type"],
                relative_path=row["relative_path"],
                identity_id=row["identity_id"],
                source_label=row["source_label"],
                contract_label=row["contract_label"],
                training_role=row["training_role"],
                split_group=row["split_group"],
                is_smile_subtype=_parse_bool(row["is_smile_subtype"]),
                include_in_phase2_training=_parse_bool(row["include_in_phase2_training"]),
                label_status=row["label_status"],
                notes=row["notes"],
            )
            for row in reader
        ]
    if include_audit:
        return records
    return [record for record in records if record.include_in_phase2_training]


def _write_csv(path: Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _largest_box(boxes: np.ndarray) -> tuple[int, int, int, int] | None:
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(int(v) for v in max(boxes, key=lambda box: box[2] * box[3]))


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges) / edges.size)


def _symmetry_mad(gray: np.ndarray) -> float | None:
    if gray.shape[1] < 2:
        return None
    width = gray.shape[1]
    half = width // 2
    left = gray[:, :half].astype(np.float32)
    right = np.fliplr(gray[:, width - half :]).astype(np.float32)
    return float(np.mean(np.abs(left - right)) / 255.0)


def _dark_ratio(gray: np.ndarray) -> float:
    threshold = float(np.percentile(gray, 25))
    return float(np.count_nonzero(gray <= threshold) / gray.size)


class OpenCvFeatureExtractor:
    def __init__(self) -> None:
        cascade_root = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascade_root / "haarcascade_frontalface_default.xml"))
        self.smile_cascade = cv2.CascadeClassifier(str(cascade_root / "haarcascade_smile.xml"))
        if self.face_cascade.empty():
            raise RuntimeError("OpenCV face cascade did not load.")
        if self.smile_cascade.empty():
            raise RuntimeError("OpenCV smile cascade did not load.")

    def extract_frame(
        self,
        image: np.ndarray,
        contract: LabelContractRecord,
        sample_kind: str,
        sample_index: int,
        frame_index: int,
        timestamp_ms: float,
    ) -> FrameFeatureRecord:
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        contrast_rms = float(np.sqrt(np.mean((gray.astype(np.float32) - brightness_mean) ** 2)))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edge_density = _edge_density(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        face = _largest_box(faces)

        face_count = int(len(faces)) if faces is not None else 0
        smile_count = 0
        smile: tuple[int, int, int, int] | None = None
        lower_face_brightness: float | None = None
        lower_face_edges: float | None = None
        lower_face_symmetry: float | None = None
        lower_face_dark: float | None = None

        if face is not None:
            x, y, w, h = face
            lower_y = y + int(h * 0.45)
            lower_roi = gray[lower_y : y + h, x : x + w]
            if lower_roi.size:
                lower_face_brightness = float(np.mean(lower_roi))
                lower_face_edges = _edge_density(lower_roi)
                lower_face_symmetry = _symmetry_mad(lower_roi)
                lower_face_dark = _dark_ratio(lower_roi)

            smile_boxes = self.smile_cascade.detectMultiScale(
                lower_roi,
                scaleFactor=1.7,
                minNeighbors=18,
                minSize=(20, 10),
            )
            smile_count = int(len(smile_boxes)) if smile_boxes is not None else 0
            local_smile = _largest_box(smile_boxes)
            if local_smile is not None:
                sx, sy, sw, sh = local_smile
                smile = (x + sx, lower_y + sy, sw, sh)

        face_area = (face[2] * face[3]) if face is not None else None
        image_area = width * height
        smile_area = (smile[2] * smile[3]) if smile is not None else None

        return FrameFeatureRecord(
            source=contract.source,
            media_type=contract.media_type,
            relative_path=contract.relative_path,
            identity_id=contract.identity_id,
            contract_label=contract.contract_label,
            training_role=contract.training_role,
            split_group=contract.split_group,
            is_smile_subtype=contract.is_smile_subtype,
            label_status=contract.label_status,
            sample_kind=sample_kind,
            sample_index=sample_index,
            frame_index=frame_index,
            timestamp_ms=float(timestamp_ms),
            image_width=width,
            image_height=height,
            face_count=face_count,
            face_found=face is not None,
            face_x_pct=_safe_ratio(face[0], width) if face is not None else None,
            face_y_pct=_safe_ratio(face[1], height) if face is not None else None,
            face_w_pct=_safe_ratio(face[2], width) if face is not None else None,
            face_h_pct=_safe_ratio(face[3], height) if face is not None else None,
            face_area_pct=_safe_ratio(face_area, image_area) if face_area is not None else None,
            face_center_x_pct=_safe_ratio(face[0] + face[2] / 2, width) if face is not None else None,
            face_center_y_pct=_safe_ratio(face[1] + face[3] / 2, height) if face is not None else None,
            smile_count=smile_count,
            smile_found=smile is not None,
            smile_x_pct=_safe_ratio(smile[0], width) if smile is not None else None,
            smile_y_pct=_safe_ratio(smile[1], height) if smile is not None else None,
            smile_w_pct=_safe_ratio(smile[2], width) if smile is not None else None,
            smile_h_pct=_safe_ratio(smile[3], height) if smile is not None else None,
            smile_area_pct=_safe_ratio(smile_area, image_area) if smile_area is not None else None,
            smile_to_face_width_ratio=_safe_ratio(smile[2], face[2]) if smile is not None and face is not None else None,
            smile_to_face_height_ratio=_safe_ratio(smile[3], face[3]) if smile is not None and face is not None else None,
            brightness_mean=brightness_mean,
            brightness_std=brightness_std,
            contrast_rms=contrast_rms,
            sharpness_laplacian_var=sharpness,
            edge_density=edge_density,
            lower_face_brightness_mean=lower_face_brightness,
            lower_face_edge_density=lower_face_edges,
            lower_face_symmetry_mad=lower_face_symmetry,
            lower_face_dark_ratio=lower_face_dark,
            extraction_status="ok",
            extractor_version=FEATURE_EXTRACTOR_VERSION,
            notes="OpenCV Haar face/smile and image-quality features; not MediaPipe landmarks.",
        )


def _sample_video_frames(path: Path, sample_count: int) -> tuple[list[tuple[int, float, np.ndarray]], float | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    duration_ms = (frame_total / fps * 1000.0) if frame_total > 0 and fps > 0 else None
    if frame_total <= 0:
        raise ValueError(f"Video has no readable frames: {path}")

    if frame_total <= sample_count:
        frame_indices = list(range(frame_total))
    else:
        frame_indices = sorted({int(round(v)) for v in np.linspace(0, frame_total - 1, sample_count)})

    frames: list[tuple[int, float, np.ndarray]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        timestamp_ms = (frame_index / fps * 1000.0) if fps > 0 else float(capture.get(cv2.CAP_PROP_POS_MSEC))
        frames.append((frame_index, timestamp_ms, frame))
    capture.release()
    return frames, duration_ms


def _extract_media_frames(
    extractor: OpenCvFeatureExtractor,
    contract: LabelContractRecord,
    data_root: Path,
    video_samples: int,
) -> tuple[list[FrameFeatureRecord], float | None]:
    media_path = data_root / contract.relative_path
    if contract.media_type == "image":
        image = cv2.imread(str(media_path))
        if image is None:
            raise ValueError(f"Could not read image: {media_path}")
        return [
            extractor.extract_frame(
                image=image,
                contract=contract,
                sample_kind="image",
                sample_index=0,
                frame_index=0,
                timestamp_ms=0.0,
            )
        ], None

    frames, duration_ms = _sample_video_frames(media_path, video_samples)
    rows = [
        extractor.extract_frame(
            image=frame,
            contract=contract,
            sample_kind="video_frame",
            sample_index=sample_index,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        for sample_index, (frame_index, timestamp_ms, frame) in enumerate(frames)
    ]
    return rows, duration_ms


def summarize_media_features(
    contract: LabelContractRecord,
    rows: list[FrameFeatureRecord],
    duration_ms: float | None,
) -> MediaFeatureRecord:
    frames_sampled = len(rows)
    frames_ok = sum(row.extraction_status == "ok" for row in rows)
    face_detection_rate = mean([1.0 if row.face_found else 0.0 for row in rows]) if rows else 0.0
    smile_detection_rate = mean([1.0 if row.smile_found else 0.0 for row in rows]) if rows else 0.0

    return MediaFeatureRecord(
        source=contract.source,
        media_type=contract.media_type,
        relative_path=contract.relative_path,
        identity_id=contract.identity_id,
        contract_label=contract.contract_label,
        training_role=contract.training_role,
        split_group=contract.split_group,
        is_smile_subtype=contract.is_smile_subtype,
        frames_sampled=frames_sampled,
        frames_ok=frames_ok,
        face_detection_rate=float(face_detection_rate),
        smile_detection_rate=float(smile_detection_rate),
        image_width_median=float(median(row.image_width for row in rows)) if rows else 0.0,
        image_height_median=float(median(row.image_height for row in rows)) if rows else 0.0,
        brightness_mean=float(mean(row.brightness_mean for row in rows)) if rows else 0.0,
        brightness_std=float(mean(row.brightness_std for row in rows)) if rows else 0.0,
        contrast_rms_mean=float(mean(row.contrast_rms for row in rows)) if rows else 0.0,
        sharpness_laplacian_var_mean=float(mean(row.sharpness_laplacian_var for row in rows)) if rows else 0.0,
        edge_density_mean=float(mean(row.edge_density for row in rows)) if rows else 0.0,
        lower_face_brightness_mean=_optional_float([row.lower_face_brightness_mean for row in rows], "mean"),
        lower_face_edge_density_mean=_optional_float([row.lower_face_edge_density for row in rows], "mean"),
        lower_face_symmetry_mad_mean=_optional_float([row.lower_face_symmetry_mad for row in rows], "mean"),
        lower_face_dark_ratio_mean=_optional_float([row.lower_face_dark_ratio for row in rows], "mean"),
        face_area_pct_mean=_optional_float([row.face_area_pct for row in rows], "mean"),
        face_area_pct_std=_optional_float([row.face_area_pct for row in rows], "std"),
        face_center_x_pct_mean=_optional_float([row.face_center_x_pct for row in rows], "mean"),
        face_center_y_pct_mean=_optional_float([row.face_center_y_pct for row in rows], "mean"),
        smile_area_pct_mean=_optional_float([row.smile_area_pct for row in rows], "mean"),
        smile_area_pct_std=_optional_float([row.smile_area_pct for row in rows], "std"),
        smile_to_face_width_ratio_mean=_optional_float([row.smile_to_face_width_ratio for row in rows], "mean"),
        smile_to_face_width_ratio_std=_optional_float([row.smile_to_face_width_ratio for row in rows], "std"),
        duration_ms=duration_ms,
        extraction_status="ok" if frames_ok == frames_sampled and frames_sampled > 0 else "partial_or_failed",
        extractor_version=FEATURE_EXTRACTOR_VERSION,
        notes="Aggregated from Phase 3 frame features.",
    )


def build_features(
    contract_path: Path = Path("data/manifests/label_contract.csv"),
    data_root: Path = DATA_ROOT,
    video_samples: int = DEFAULT_FRAME_SAMPLES,
    include_audit: bool = False,
) -> tuple[list[FrameFeatureRecord], list[MediaFeatureRecord]]:
    contracts = _read_contract(contract_path, include_audit=include_audit)
    extractor = OpenCvFeatureExtractor()
    frame_rows: list[FrameFeatureRecord] = []
    media_rows: list[MediaFeatureRecord] = []

    for contract in contracts:
        rows, duration_ms = _extract_media_frames(extractor, contract, data_root, video_samples)
        if not rows:
            raise ValueError(f"No feature rows generated for {contract.relative_path}")
        frame_rows.extend(rows)
        media_rows.append(summarize_media_features(contract, rows, duration_ms))

    validate_features(contracts, frame_rows, media_rows)
    return frame_rows, media_rows


def validate_features(
    contracts: list[LabelContractRecord],
    frame_rows: list[FrameFeatureRecord],
    media_rows: list[MediaFeatureRecord],
) -> None:
    expected_paths = {record.relative_path for record in contracts}
    media_paths = {row.relative_path for row in media_rows}
    if media_paths != expected_paths:
        missing = sorted(expected_paths - media_paths)
        extra = sorted(media_paths - expected_paths)
        raise AssertionError(f"Media feature path mismatch. Missing={missing[:5]} Extra={extra[:5]}")
    frame_paths = {row.relative_path for row in frame_rows}
    if not expected_paths.issubset(frame_paths):
        missing = sorted(expected_paths - frame_paths)
        raise AssertionError(f"Frame feature rows missing paths: {missing[:5]}")
    if any(row.contract_label == "audit_only" for row in media_rows):
        raise AssertionError("Default Phase 3 features should not include audit_only rows.")
    for row in frame_rows:
        if row.image_width <= 0 or row.image_height <= 0:
            raise AssertionError(f"Bad image dimensions: {row}")
        if not 0 <= row.edge_density <= 1:
            raise AssertionError(f"Edge density out of range: {row}")
        if row.face_area_pct is not None and not 0 < row.face_area_pct <= 1:
            raise AssertionError(f"Face area out of range: {row}")
        if row.smile_area_pct is not None and not 0 < row.smile_area_pct <= 1:
            raise AssertionError(f"Smile area out of range: {row}")


def summarize(rows: list[MediaFeatureRecord]) -> str:
    by_label = Counter(row.contract_label for row in rows)
    by_source = Counter(row.source for row in rows)
    face_by_label: dict[str, list[float]] = defaultdict(list)
    smile_by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        face_by_label[row.contract_label].append(row.face_detection_rate)
        smile_by_label[row.contract_label].append(row.smile_detection_rate)

    label_quality = {
        label: {
            "face_detection_rate_mean": round(float(mean(face_rates)), 4),
            "smile_detection_rate_mean": round(float(mean(smile_by_label[label])), 4),
        }
        for label, face_rates in sorted(face_by_label.items())
    }
    return "\n".join(
        [
            f"Media feature rows: {len(rows)}",
            f"Sources: {dict(sorted(by_source.items()))}",
            f"Labels: {dict(sorted(by_label.items()))}",
            f"Quality by label: {label_quality}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 3 feature CSVs.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--contract", type=Path, default=Path("data/manifests/label_contract.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--video-samples", type=int, default=DEFAULT_FRAME_SAMPLES)
    parser.add_argument("--include-audit", action="store_true")
    args = parser.parse_args()

    frame_rows, media_rows = build_features(
        contract_path=args.contract,
        data_root=args.data_root,
        video_samples=args.video_samples,
        include_audit=args.include_audit,
    )
    _write_csv(args.out_dir / "phase3_frame_features.csv", frame_rows, FRAME_FEATURE_FIELDS)
    _write_csv(args.out_dir / "phase3_media_features.csv", media_rows, MEDIA_FEATURE_FIELDS)
    print(summarize(media_rows))


if __name__ == "__main__":
    main()
