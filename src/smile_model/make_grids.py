from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .schemas import DATA_ROOT, FEI_ROOT, OSF_BEST_ACTORS, OSF_EXPECTED_ACTORS, OSF_VIDEO_DIR


SMILE_COLUMNS = ["Reward", "Affiliation", "Dominance"]
FULL_COLUMNS = ["Reward", "Affiliation", "Dominance", "Anger", "Disgust", "Sadness"]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ["Arial.ttf", "Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def extract_representative_frame(video_path: Path, out_path: Path) -> None:
    # These clips are short acted expressions; the midpoint is usually close
    # to the peak and avoids title/settling frames. OpenCV avoids requiring a
    # separate ffmpeg install on lab machines.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, frame_count // 2) if frame_count else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok and target > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
      raise RuntimeError(f"Could not read frame from video: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame).save(out_path)


def _prepare_cell(image_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))


def make_video_grid(
    actors: list[str],
    labels: list[str],
    out_path: Path,
    title: str,
    video_dir: Path = OSF_VIDEO_DIR,
) -> None:
    cell_w, cell_h = 180, 132
    row_header_w = 122
    col_header_h = 48
    title_h = 58
    pad = 12
    width = row_header_w + len(labels) * cell_w + pad * 2
    height = title_h + col_header_h + len(actors) * cell_h + pad * 2
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    header_font = _font(16)
    label_font = _font(14)
    small_font = _font(11)

    draw.text((pad, pad), title, fill="#172033", font=title_font)
    draw.text(
        (pad, pad + 31),
        "Representative midpoint frames from OSF smile-type videos",
        fill="#586174",
        font=small_font,
    )

    x0 = pad + row_header_w
    y0 = pad + title_h
    for j, label in enumerate(labels):
        draw.rectangle((x0 + j * cell_w, y0, x0 + (j + 1) * cell_w - 2, y0 + col_header_h - 2), fill="#e7ebf3")
        draw.text((x0 + j * cell_w + 10, y0 + 15), label, fill="#172033", font=header_font)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, actor in enumerate(actors):
            y = y0 + col_header_h + i * cell_h
            draw.rectangle((pad, y, pad + row_header_w - 2, y + cell_h - 2), fill="#edf1f7")
            draw.text((pad + 10, y + cell_h // 2 - 9), actor, fill="#172033", font=label_font)
            for j, label in enumerate(labels):
                x = x0 + j * cell_w
                video = video_dir / f"{actor}_{label}.mp4"
                frame = tmp_dir / f"{actor}_{label}.jpg"
                extract_representative_frame(video, frame)
                cell = _prepare_cell(frame, (cell_w - 4, cell_h - 4))
                canvas.paste(cell, (x, y))
                draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), outline="#ffffff", width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_fei_grid(out_path: Path, fei_root: Path = FEI_ROOT) -> None:
    subjects = ["10", "25", "50", "75", "100", "125", "150", "175"]
    cell_w, cell_h = 150, 190
    row_header_w = 96
    col_header_h = 48
    title_h = 64
    pad = 12
    labels = [("a", "Neutral"), ("b", "Generic smile")]
    width = row_header_w + len(labels) * cell_w + pad * 2
    height = title_h + col_header_h + len(subjects) * cell_h + pad * 2
    canvas = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    header_font = _font(16)
    label_font = _font(14)
    small_font = _font(11)
    draw.text((pad, pad), "FEI Neutral vs Generic Smile Examples", fill="#172033", font=title_font)
    draw.text((pad, pad + 30), "Not reward/affiliative/dominance subtype-labeled", fill="#8a3340", font=small_font)
    x0 = pad + row_header_w
    y0 = pad + title_h
    for j, (_, label) in enumerate(labels):
        draw.rectangle((x0 + j * cell_w, y0, x0 + (j + 1) * cell_w - 2, y0 + col_header_h - 2), fill="#e7ebf3")
        draw.text((x0 + j * cell_w + 10, y0 + 15), label, fill="#172033", font=header_font)
    aligned_dirs = sorted(fei_root.glob("frontalimages_manuallyaligned_part*"))
    by_name = {p.name: p for d in aligned_dirs for p in d.glob("*.jpg")}
    for i, subject in enumerate(subjects):
        y = y0 + col_header_h + i * cell_h
        draw.rectangle((pad, y, pad + row_header_w - 2, y + cell_h - 2), fill="#edf1f7")
        draw.text((pad + 10, y + cell_h // 2 - 9), subject, fill="#172033", font=label_font)
        for j, (variant, _) in enumerate(labels):
            x = x0 + j * cell_w
            path = by_name.get(f"{subject}{variant}.jpg")
            if not path:
                continue
            cell = _prepare_cell(path, (cell_w - 4, cell_h - 4))
            canvas.paste(cell, (x, y))
            draw.rectangle((x, y, x + cell_w - 4, y + cell_h - 4), outline="#ffffff", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase 1 visual dataset grids.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/randy_grid_examples"))
    args = parser.parse_args()
    video_dir = args.data_root / "OSF Smile Types" / "Types of Smiles"
    fei_root = args.data_root / "FEI Face Database"

    make_video_grid(
        OSF_BEST_ACTORS,
        SMILE_COLUMNS,
        args.out_dir / "osf_best_actor_smile_grid.png",
        "OSF Best Actor Smile-Type Grid",
        video_dir,
    )
    make_video_grid(
        sorted(OSF_EXPECTED_ACTORS),
        FULL_COLUMNS,
        args.out_dir / "osf_full_expression_grid.png",
        "OSF Full Expression Grid",
        video_dir,
    )
    make_fei_grid(args.out_dir / "fei_neutral_smile_grid.png", fei_root)
    print(f"Wrote grids to {args.out_dir}")


if __name__ == "__main__":
    main()
