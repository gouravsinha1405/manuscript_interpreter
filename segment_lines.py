from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess a manuscript page image and segment it into line crops."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="page_images/page_004.png",
        help="Path to the page image to segment",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="line_images/page_004",
        help="Directory where cropped line images will be saved",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory for debug artifacts such as overlays and binary masks",
    )
    parser.add_argument(
        "--min-line-height",
        type=int,
        default=20,
        help="Minimum retained band height in pixels",
    )
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=12,
        help="Maximum vertical gap for merging nearby text bands",
    )
    parser.add_argument(
        "--max-band-height",
        type=int,
        default=600,
        help="Split oversized bands taller than this using low-ink valleys.",
    )
    parser.add_argument(
        "--min-gap-rows",
        type=int,
        default=15,
        help="Minimum contiguous low-ink row run used as a split point inside oversized bands.",
    )
    parser.add_argument(
        "--pad-y",
        type=int,
        default=8,
        help="Vertical padding added around each detected line band",
    )
    parser.add_argument(
        "--pad-x",
        type=int,
        default=12,
        help="Horizontal padding added around the detected text extent",
    )
    return parser.parse_args()


def preprocess_page(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pillow autocontrast helps stretch manuscript tones before local enhancement.
    gray_autocontrast = np.array(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))

    # Normalize uneven background illumination by dividing by a blurred background estimate.
    background = cv2.GaussianBlur(gray_autocontrast, (0, 0), sigmaX=31, sigmaY=31)
    normalized = cv2.divide(gray_autocontrast, background, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # A mild bilateral filter suppresses speckle without erasing stroke edges.
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )

    open_kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # A small horizontal close reconnects broken character fragments for row projection.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return img, gray_autocontrast, enhanced, cleaned


def find_line_bands(
    binary: np.ndarray,
    min_line_height: int = 20,
    gap_threshold: int = 12,
) -> list[tuple[int, int]]:
    row_sum = np.sum(binary > 0, axis=1)
    threshold = 0.02 * binary.shape[1]
    active_rows = row_sum > threshold

    bands: list[tuple[int, int]] = []
    start: int | None = None

    for row_index, active in enumerate(active_rows):
        if active and start is None:
            start = row_index
        elif not active and start is not None:
            bands.append((start, row_index))
            start = None

    if start is not None:
        bands.append((start, len(active_rows)))

    merged: list[tuple[int, int]] = []
    for band in bands:
        if not merged:
            merged.append(band)
            continue

        prev_start, prev_end = merged[-1]
        curr_start, curr_end = band
        if curr_start - prev_end <= gap_threshold:
            merged[-1] = (prev_start, curr_end)
        else:
            merged.append(band)

    return [(start_row, end_row) for start_row, end_row in merged if end_row - start_row >= min_line_height]


def split_tall_bands(
    binary: np.ndarray,
    bands: list[tuple[int, int]],
    min_line_height: int,
    max_band_height: int,
    min_gap_rows: int,
) -> list[tuple[int, int]]:
    fallback_overlap = 40
    split_bands: list[tuple[int, int]] = []

    def append_bounded_segments(start_row: int, end_row: int) -> None:
        height = end_row - start_row
        if height <= max_band_height:
            split_bands.append((start_row, end_row))
            return

        step = max(min_line_height, max_band_height - fallback_overlap)
        current_start = start_row
        while current_start < end_row:
            current_end = min(end_row, current_start + max_band_height)
            if current_end - current_start >= min_line_height:
                split_bands.append((current_start, current_end))
            if current_end >= end_row:
                break
            current_start += step

    for start_row, end_row in bands:
        height = end_row - start_row
        if height <= max_band_height:
            split_bands.append((start_row, end_row))
            continue

        band = binary[start_row:end_row, :]
        row_sum = np.sum(band > 0, axis=1)
        low_threshold = float(np.percentile(row_sum, 10))
        low_mask = row_sum <= low_threshold

        gap_runs: list[tuple[int, int]] = []
        gap_start: int | None = None
        for index, is_low in enumerate(low_mask):
            if is_low and gap_start is None:
                gap_start = index
            elif not is_low and gap_start is not None:
                if index - gap_start >= min_gap_rows:
                    gap_runs.append((gap_start, index))
                gap_start = None

        if gap_start is not None and len(low_mask) - gap_start >= min_gap_rows:
            gap_runs.append((gap_start, len(low_mask)))

        if not gap_runs:
            append_bounded_segments(start_row, end_row)
            continue

        last_start = start_row
        for gap_start, gap_end in gap_runs:
            split_at = start_row + (gap_start + gap_end) // 2
            if split_at - last_start >= min_line_height:
                append_bounded_segments(last_start, split_at)
            last_start = split_at

        if end_row - last_start >= min_line_height:
            append_bounded_segments(last_start, end_row)

    return split_bands


def find_text_columns(binary_band: np.ndarray, min_fraction: float = 0.003) -> tuple[int, int]:
    col_sum = np.sum(binary_band > 0, axis=0)
    threshold = max(1, int(binary_band.shape[0] * min_fraction))
    active_columns = np.flatnonzero(col_sum > threshold)

    if active_columns.size == 0:
        return 0, binary_band.shape[1]

    return int(active_columns[0]), int(active_columns[-1] + 1)


def save_debug_artifacts(
    original: np.ndarray,
    grayscale: np.ndarray,
    enhanced: np.ndarray,
    binary: np.ndarray,
    bands: list[tuple[int, int]],
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    grayscale_path = debug_dir / "grayscale_autocontrast.png"
    cv2.imwrite(str(grayscale_path), grayscale)

    enhanced_path = debug_dir / "enhanced.png"
    cv2.imwrite(str(enhanced_path), enhanced)

    binary_path = debug_dir / "binary.png"
    cv2.imwrite(str(binary_path), binary)

    overlay = original.copy()
    for index, (start_row, end_row) in enumerate(bands, start=1):
        cv2.rectangle(overlay, (0, start_row), (overlay.shape[1] - 1, end_row), (0, 0, 255), 4)
        cv2.putText(
            overlay,
            f"{index:02}",
            (20, max(30, start_row - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    overlay_path = debug_dir / "line_bands_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)


def crop_lines(
    image_path: Path,
    output_dir: Path,
    debug_dir: Path | None = None,
    min_line_height: int = 20,
    gap_threshold: int = 12,
    pad_y: int = 8,
    max_band_height: int = 600,
    min_gap_rows: int = 15,
    pad_x: int = 12,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    original, grayscale, enhanced, binary = preprocess_page(image_path)
    bands = find_line_bands(
        binary,
        min_line_height=min_line_height,
        gap_threshold=gap_threshold,
    )

    bands = split_tall_bands(
        binary,
        bands,
        min_line_height=min_line_height,
        max_band_height=max_band_height,
        min_gap_rows=min_gap_rows,
    )
    if debug_dir is not None:
        save_debug_artifacts(original, grayscale, enhanced, binary, bands, debug_dir)

    for idx, (y1, y2) in enumerate(bands, start=1):
        top = max(0, y1 - pad_y)
        bottom = min(original.shape[0], y2 + pad_y)
        line_binary = binary[top:bottom, :]
        x1, x2 = find_text_columns(line_binary)
        left = max(0, x1 - pad_x)
        right = min(original.shape[1], x2 + pad_x)
        line_img = original[top:bottom, left:right]
        out_path = output_dir / f"line_{idx:03}.png"
        cv2.imwrite(str(out_path), line_img)

    print(f"Saved {len(bands)} lines to {output_dir}")
    return len(bands)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else None

    try:
        crop_lines(
            image_path,
            output_dir,
            debug_dir=debug_dir,
            min_line_height=args.min_line_height,
            gap_threshold=args.gap_threshold,
            max_band_height=args.max_band_height,
            min_gap_rows=args.min_gap_rows,
            pad_y=args.pad_y,
            pad_x=args.pad_x,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())