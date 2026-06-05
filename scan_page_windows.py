from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_paddle_ocr import (
    PaddleOCR,
    deduplicate_entries,
    load_image,
    polygon_bounds,
    resolve_crop_bounds,
    run_tiled_ocr,
    downscale_image,
    map_entries_to_source,
    map_tiles_to_source,
    validate_tile_budget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially scan a manuscript page with safe sliding windows and aggregate OCR output."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="page_images/page_004.png",
        help="Path to the page image to scan",
    )
    parser.add_argument(
        "--output-dir",
        default="page_scan/page_004",
        help="Directory where aggregated page outputs will be written",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="PaddleOCR recognition language code. Use `hi` for Devanagari manuscripts.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=3200,
        help="Sliding window width in source-image pixels",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1800,
        help="Sliding window height in source-image pixels",
    )
    parser.add_argument(
        "--step-x",
        type=int,
        default=2400,
        help="Horizontal step between neighboring windows in source-image pixels",
    )
    parser.add_argument(
        "--step-y",
        type=int,
        default=1200,
        help="Vertical step between neighboring windows in source-image pixels",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=8,
        help="Abort after this many windows. Raise explicitly if you want broader collection.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for retained OCR entries",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1800,
        help="Downscale each window so its longest side is at most this many pixels before OCR",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1800,
        help="Maximum tile width or height inside each window",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=160,
        help="Overlap in pixels between neighboring OCR tiles inside each window",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=4,
        help="Abort a single window if it would require more than this many tiles",
    )
    return parser.parse_args()


def generate_windows(
    image_width: int,
    image_height: int,
    window_width: int,
    window_height: int,
    step_x: int,
    step_y: int,
) -> list[dict[str, int]]:
    if window_width <= 0 or window_height <= 0:
        raise ValueError("Window dimensions must be positive.")
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Step sizes must be positive.")

    x_starts = list(range(0, max(1, image_width - window_width + 1), step_x))
    y_starts = list(range(0, max(1, image_height - window_height + 1), step_y))

    if not x_starts or x_starts[-1] != max(0, image_width - window_width):
        x_starts.append(max(0, image_width - window_width))
    if not y_starts or y_starts[-1] != max(0, image_height - window_height):
        y_starts.append(max(0, image_height - window_height))

    windows: list[dict[str, int]] = []
    index = 1
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(image_width, x1 + window_width)
            y2 = min(image_height, y1 + window_height)
            windows.append(
                {
                    "window_index": index,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )
            index += 1

    return windows


def window_entries_text(entries: list[dict[str, object]]) -> str:
    ordered = sorted(entries, key=lambda entry: (polygon_bounds(entry["box"])[1], polygon_bounds(entry["box"])[0]))
    return "\n".join(str(entry["text"]) for entry in ordered)


def scan_window(
    ocr: PaddleOCR,
    source_image,
    window: dict[str, int],
    max_side: int,
    tile_size: int,
    tile_overlap: int,
    max_tiles: int,
    min_confidence: float,
) -> tuple[list[dict[str, object]], list[dict[str, int]], dict[str, object]]:
    crop = resolve_crop_bounds(source_image, [window["x1"], window["y1"], window["x2"], window["y2"]])
    crop_x1, crop_y1, crop_x2, crop_y2 = crop
    cropped_image = source_image[crop_y1:crop_y2, crop_x1:crop_x2]
    working_image, scale = downscale_image(cropped_image, max_side)
    validate_tile_budget(
        working_image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        max_tiles=max_tiles,
    )

    entries, tiles = run_tiled_ocr(
        ocr,
        image=working_image,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )
    entries = map_entries_to_source(entries, scale=scale, x_offset=crop_x1, y_offset=crop_y1)
    entries = [entry for entry in entries if float(entry["confidence"]) >= min_confidence]
    tiles = map_tiles_to_source(tiles, scale=scale, x_offset=crop_x1, y_offset=crop_y1)

    metadata = {
        "window_index": window["window_index"],
        "crop": {
            "x1": crop_x1,
            "y1": crop_y1,
            "x2": crop_x2,
            "y2": crop_y2,
            "width": crop_x2 - crop_x1,
            "height": crop_y2 - crop_y1,
        },
        "processed_width": int(working_image.shape[1]),
        "processed_height": int(working_image.shape[0]),
        "scale": scale,
        "tile_count": len(tiles),
        "entry_count": len(entries),
    }
    return entries, tiles, metadata


def write_page_outputs(
    output_dir: Path,
    image_path: Path,
    scan_metadata: dict[str, object],
    windows: list[dict[str, object]],
    entries: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_entries = deduplicate_entries(entries)
    scan_path = output_dir / "page_scan.json"
    scan_path.write_text(
        json.dumps(
            {
                "metadata": scan_metadata,
                "windows": windows,
                "entries": aggregated_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text_path = output_dir / "page_text.txt"
    text_path.write_text(window_entries_text(aggregated_entries), encoding="utf-8")

    report_lines = [
        f"# Page Scan for {image_path.name}",
        "",
        f"Windows scanned: {scan_metadata['windows_scanned']}",
        f"Aggregated OCR entries: {len(aggregated_entries)}",
        f"Language: {scan_metadata['lang']}",
        "",
    ]
    for window in windows:
        report_lines.append(
            f"## Window {window['window_index']} ({window['crop']['x1']},{window['crop']['y1']})-({window['crop']['x2']},{window['crop']['y2']})"
        )
        report_lines.append("")
        report_lines.append(window.get("text", ""))
        report_lines.append("")

    report_path = output_dir / "page_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return 1

    source_image = load_image(image_path)
    image_height, image_width = source_image.shape[:2]
    windows = generate_windows(
        image_width=image_width,
        image_height=image_height,
        window_width=args.window_width,
        window_height=args.window_height,
        step_x=args.step_x,
        step_y=args.step_y,
    )

    if len(windows) > args.max_windows:
        windows = windows[: args.max_windows]

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
    )

    all_entries: list[dict[str, object]] = []
    window_reports: list[dict[str, object]] = []
    for window in windows:
        entries, tiles, metadata = scan_window(
            ocr,
            source_image=source_image,
            window=window,
            max_side=args.max_side,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_tiles=args.max_tiles,
            min_confidence=args.min_confidence,
        )
        all_entries.extend(entries)
        window_reports.append(
            {
                **metadata,
                "tiles": tiles,
                "text": window_entries_text(entries),
            }
        )

    scan_metadata = {
        "source_image": str(image_path),
        "source_width": int(image_width),
        "source_height": int(image_height),
        "lang": args.lang,
        "window_width": args.window_width,
        "window_height": args.window_height,
        "step_x": args.step_x,
        "step_y": args.step_y,
        "max_side": args.max_side,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "max_tiles": args.max_tiles,
        "windows_scanned": len(window_reports),
    }
    write_page_outputs(output_dir, image_path, scan_metadata, window_reports, all_entries)
    print(f"Scanned {len(window_reports)} window(s) and wrote page outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())