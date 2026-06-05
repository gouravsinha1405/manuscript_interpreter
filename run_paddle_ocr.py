from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PaddleOCR is not installed. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on a manuscript page image and save raw outputs."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="page_images/page_004.png",
        help="Path to the page image to process",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_output/page_004",
        help="Directory for OCR text, JSON, and visualization outputs",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        type=int,
        default=None,
        help="Optional crop region in source-image pixels. OCR runs only on this rectangle.",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="PaddleOCR recognition language code. Use `hi` for Devanagari manuscripts.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for keeping recognized text",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1800,
        help="Maximum tile width or height for OCR. Large pages are split into overlapping tiles.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=200,
        help="Overlap in pixels between neighboring OCR tiles.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=3200,
        help="Downscale the page so its longest side is at most this many pixels before OCR. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=9,
        help="Abort if OCR would require more than this many tiles.",
    )
    parser.add_argument(
        "--write-overlays",
        action="store_true",
        help="Write visual overlay images. Disabled by default because large pages can consume a lot of RAM and disk.",
    )
    return parser.parse_args()


def _entries_from_ocr_result_page(page: object) -> list[dict[str, object]]:
    if not hasattr(page, "get"):
        return []

    polys = page.get("dt_polys", [])
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    return [
        {
            "box": np.asarray(box).tolist(),
            "text": str(text),
            "confidence": float(confidence),
        }
        for box, text, confidence in zip(polys, texts, scores)
    ]


def _entries_from_legacy_result(raw_result: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    pages = raw_result if isinstance(raw_result, list) else [raw_result]
    if pages and isinstance(pages[0], list) and pages and pages[0] and isinstance(pages[0][0], list):
        pages = pages[0]

    for item in pages:
        if not isinstance(item, list) or len(item) < 2:
            continue

        box, payload = item[0], item[1]
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            continue

        text = str(payload[0])
        confidence = float(payload[1])
        entries.append(
            {
                "box": box,
                "text": text,
                "confidence": confidence,
            }
        )

    return entries


def normalize_result(raw_result: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    if not raw_result:
        return entries

    if isinstance(raw_result, list) and raw_result and hasattr(raw_result[0], "keys"):
        for page in raw_result:
            entries.extend(_entries_from_ocr_result_page(page))
        return entries

    return _entries_from_legacy_result(raw_result)


def polygon_bounds(box: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_iou(first_box: list[list[float]], second_box: list[list[float]]) -> float:
    first_left, first_top, first_right, first_bottom = polygon_bounds(first_box)
    second_left, second_top, second_right, second_bottom = polygon_bounds(second_box)

    inter_left = max(first_left, second_left)
    inter_top = max(first_top, second_top)
    inter_right = min(first_right, second_right)
    inter_bottom = min(first_bottom, second_bottom)

    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    first_area = max(0.0, first_right - first_left) * max(0.0, first_bottom - first_top)
    second_area = max(0.0, second_right - second_left) * max(0.0, second_bottom - second_top)
    union_area = first_area + second_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def generate_tiles(image: np.ndarray, tile_size: int, tile_overlap: int) -> list[dict[str, object]]:
    height, width = image.shape[:2]
    if tile_size <= tile_overlap:
        raise ValueError("tile-size must be larger than tile-overlap.")

    if width <= tile_size and height <= tile_size:
        return [
            {
                "x": 0,
                "y": 0,
                "width": width,
                "height": height,
                "image": image,
                "index": 1,
            }
        ]

    step = tile_size - tile_overlap
    x_starts = list(range(0, max(1, width - tile_size + 1), step))
    y_starts = list(range(0, max(1, height - tile_size + 1), step))

    if not x_starts or x_starts[-1] != max(0, width - tile_size):
        x_starts.append(max(0, width - tile_size))
    if not y_starts or y_starts[-1] != max(0, height - tile_size):
        y_starts.append(max(0, height - tile_size))

    tiles: list[dict[str, object]] = []
    index = 1
    for y_start in y_starts:
        for x_start in x_starts:
            x_end = min(width, x_start + tile_size)
            y_end = min(height, y_start + tile_size)
            tiles.append(
                {
                    "x": x_start,
                    "y": y_start,
                    "width": x_end - x_start,
                    "height": y_end - y_start,
                    "image": image[y_start:y_end, x_start:x_end],
                    "index": index,
                }
            )
            index += 1

    return tiles


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def downscale_image(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max_side <= 0:
        return image, 1.0

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image, 1.0

    scale = max_side / float(longest_side)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def validate_tile_budget(image: np.ndarray, tile_size: int, tile_overlap: int, max_tiles: int) -> None:
    height, width = image.shape[:2]
    step = tile_size - tile_overlap
    if step <= 0:
        raise ValueError("tile-size must be larger than tile-overlap.")

    x_tiles = 1 if width <= tile_size else math.ceil((width - tile_size) / step) + 1
    y_tiles = 1 if height <= tile_size else math.ceil((height - tile_size) / step) + 1
    tile_count = x_tiles * y_tiles
    if tile_count > max_tiles:
        raise ValueError(
            f"Refusing OCR run: {tile_count} tiles would be generated, which exceeds --max-tiles={max_tiles}. "
            "Lower --max-side, increase --tile-size, or raise --max-tiles explicitly if you want to accept the load."
        )


def resolve_crop_bounds(image: np.ndarray, crop: list[int] | None) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    if crop is None:
        return 0, 0, width, height

    x1, y1, x2, y2 = crop
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"Crop {crop} is outside the image bounds {width}x{height}.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop must satisfy X2 > X1 and Y2 > Y1.")
    return x1, y1, x2, y2


def map_box_to_source(box: list[list[float]], scale: float, x_offset: int, y_offset: int) -> list[list[float]]:
    if scale <= 0:
        raise ValueError("Scale must be positive.")
    return [[point[0] / scale + x_offset, point[1] / scale + y_offset] for point in box]


def map_entries_to_source(
    entries: list[dict[str, object]],
    scale: float,
    x_offset: int,
    y_offset: int,
) -> list[dict[str, object]]:
    mapped: list[dict[str, object]] = []
    for entry in entries:
        mapped.append(
            {
                **entry,
                "box": map_box_to_source(entry["box"], scale=scale, x_offset=x_offset, y_offset=y_offset),
            }
        )
    return mapped


def map_tiles_to_source(
    tiles: list[dict[str, int]],
    scale: float,
    x_offset: int,
    y_offset: int,
) -> list[dict[str, int]]:
    if scale <= 0:
        raise ValueError("Scale must be positive.")

    mapped: list[dict[str, int]] = []
    for tile in tiles:
        mapped.append(
            {
                "tile_index": tile["tile_index"],
                "x": int(round(tile["x"] / scale + x_offset)),
                "y": int(round(tile["y"] / scale + y_offset)),
                "width": int(round(tile["width"] / scale)),
                "height": int(round(tile["height"] / scale)),
            }
        )
    return mapped


def offset_entry(entry: dict[str, object], x_offset: int, y_offset: int, tile_index: int) -> dict[str, object]:
    box = [
        [float(point[0]) + x_offset, float(point[1]) + y_offset]
        for point in entry["box"]
    ]
    return {
        "box": box,
        "text": entry["text"],
        "confidence": entry["confidence"],
        "tile_index": tile_index,
    }


def deduplicate_entries(entries: list[dict[str, object]], iou_threshold: float = 0.5) -> list[dict[str, object]]:
    deduplicated: list[dict[str, object]] = []
    for entry in sorted(entries, key=lambda item: float(item["confidence"]), reverse=True):
        is_duplicate = False
        for kept in deduplicated:
            if entry["text"] != kept["text"]:
                continue
            if polygon_iou(entry["box"], kept["box"]) >= iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            deduplicated.append(entry)

    return sorted(deduplicated, key=lambda item: (polygon_bounds(item["box"])[1], polygon_bounds(item["box"])[0]))


def run_tiled_ocr(
    ocr: PaddleOCR,
    image: np.ndarray,
    tile_size: int,
    tile_overlap: int,
) -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    tiles = generate_tiles(image, tile_size=tile_size, tile_overlap=tile_overlap)
    entries: list[dict[str, object]] = []
    tile_manifest: list[dict[str, int]] = []

    for tile in tiles:
        raw_result = ocr.predict(tile["image"])
        tile_entries = normalize_result(raw_result)
        entries.extend(
            offset_entry(entry, x_offset=int(tile["x"]), y_offset=int(tile["y"]), tile_index=int(tile["index"]))
            for entry in tile_entries
        )
        tile_manifest.append(
            {
                "tile_index": int(tile["index"]),
                "x": int(tile["x"]),
                "y": int(tile["y"]),
                "width": int(tile["width"]),
                "height": int(tile["height"]),
            }
        )

    return deduplicate_entries(entries), tile_manifest


def save_visualization(image: np.ndarray, entries: list[dict[str, object]], output_path: Path) -> None:
    overlay = image.copy()
    for index, entry in enumerate(entries, start=1):
        box = entry["box"]
        points = []
        for point in box:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                points.append([int(point[0]), int(point[1])])
        if len(points) != 4:
            continue

        contour = cv2.convexHull(cv2.UMat(np.array(points, dtype="int32"))).get()
        cv2.polylines(overlay, [contour], isClosed=True, color=(0, 0, 255), thickness=3)
        anchor_x, anchor_y = points[0]
        cv2.putText(
            overlay,
            f"{index:02}",
            (anchor_x, max(20, anchor_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def save_tile_overlay(image: np.ndarray, tiles: list[dict[str, int]], output_path: Path) -> None:
    overlay = image.copy()
    for tile in tiles:
        left = tile["x"]
        top = tile["y"]
        right = left + tile["width"]
        bottom = top + tile["height"]
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), 4)
        cv2.putText(
            overlay,
            f"T{tile['tile_index']:02}",
            (left + 12, max(36, top + 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        source_image = load_image(image_path)
        crop_x1, crop_y1, crop_x2, crop_y2 = resolve_crop_bounds(source_image, args.crop)
        cropped_image = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
        working_image, scale = downscale_image(cropped_image, args.max_side)
        validate_tile_budget(
            working_image,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_tiles=args.max_tiles,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
    )

    entries, tiles = run_tiled_ocr(
        ocr,
        image=working_image,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
    )
    entries = map_entries_to_source(entries, scale=scale, x_offset=crop_x1, y_offset=crop_y1)
    tiles = map_tiles_to_source(tiles, scale=scale, x_offset=crop_x1, y_offset=crop_y1)
    entries = [entry for entry in entries if entry["confidence"] >= args.min_confidence]

    metadata = {
        "source_image": str(image_path),
        "source_height": int(source_image.shape[0]),
        "source_width": int(source_image.shape[1]),
        "crop": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
            "width": crop_x2 - crop_x1,
            "height": crop_y2 - crop_y1,
        },
        "processed_height": int(working_image.shape[0]),
        "processed_width": int(working_image.shape[1]),
        "scale": scale,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_count": len(tiles),
    }

    json_path = output_dir / "ocr.json"
    json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    tiles_path = output_dir / "tiles.json"
    tiles_path.write_text(json.dumps(tiles, indent=2), encoding="utf-8")

    text_path = output_dir / "ocr.txt"
    text_path.write_text("\n".join(entry["text"] for entry in entries), encoding="utf-8")

    if args.write_overlays:
        overlay_path = output_dir / "ocr_overlay.png"
        if entries:
            overlay_entries = map_entries_to_source(entries, scale=1.0, x_offset=-crop_x1, y_offset=-crop_y1)
            if not math.isclose(scale, 1.0):
                overlay_entries = map_entries_to_source(overlay_entries, scale=1.0 / scale, x_offset=0, y_offset=0)
            save_visualization(working_image, overlay_entries, overlay_path)

        tile_overlay_path = output_dir / "tile_overlay.png"
        if tiles:
            overlay_tiles = map_tiles_to_source(tiles, scale=1.0, x_offset=-crop_x1, y_offset=-crop_y1)
            if not math.isclose(scale, 1.0):
                overlay_tiles = map_tiles_to_source(overlay_tiles, scale=1.0 / scale, x_offset=0, y_offset=0)
            save_tile_overlay(working_image, overlay_tiles, tile_overlay_path)

    print(f"Saved {len(entries)} OCR entries from {len(tiles)} tile(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())