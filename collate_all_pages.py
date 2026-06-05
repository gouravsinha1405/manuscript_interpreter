from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_phase2_input import build_chunks, write_outputs as write_phase2_outputs
from run_paddle_ocr import PaddleOCR, load_image
from scan_page_windows import generate_windows, scan_window, write_page_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially scan every rendered page image and prepare phase-2 inputs."
    )
    parser.add_argument(
        "--input-dir",
        default="page_images",
        help="Directory containing rendered page PNG files",
    )
    parser.add_argument(
        "--scan-root",
        default="page_scan",
        help="Root directory for per-page scan outputs",
    )
    parser.add_argument(
        "--phase2-root",
        default="phase2_input",
        help="Root directory for per-page phase-2 artifacts",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="PaddleOCR recognition language code. Use `hi` for Devanagari manuscripts.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=2800,
        help="Sliding window width in source-image pixels",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=1400,
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
        default=6,
        help="Maximum windows to scan per page",
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
        default=1600,
        help="Downscale each window so its longest side is at most this many pixels before OCR",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1600,
        help="Maximum tile width or height inside each window",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=120,
        help="Overlap in pixels between neighboring OCR tiles inside each window",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=2,
        help="Abort a single window if it would require more than this many tiles",
    )
    parser.add_argument(
        "--phase2-max-lines",
        type=int,
        default=10,
        help="Maximum OCR lines per phase-2 chunk",
    )
    parser.add_argument(
        "--phase2-max-chars",
        type=int,
        default=600,
        help="Maximum approximate characters per phase-2 chunk",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit for bounded runs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild outputs even if per-page scan and phase-2 outputs already exist",
    )
    return parser.parse_args()


def sorted_page_images(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("page_*.png"))


def scan_page(image_path: Path, scan_dir: Path, ocr: PaddleOCR, args: argparse.Namespace) -> dict[str, object]:
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
        window_reports.append({**metadata, "tiles": tiles, "text": "\n".join(str(entry["text"]) for entry in entries)})

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
    write_page_outputs(scan_dir, image_path, scan_metadata, window_reports, all_entries)
    return {
        "metadata": scan_metadata,
        "windows": window_reports,
        "entries": all_entries,
    }


def prepare_page_phase2(scan_data: dict[str, object], scan_dir: Path, phase2_dir: Path, args: argparse.Namespace) -> int:
    chunks = build_chunks(
        scan_data,
        max_lines=args.phase2_max_lines,
        max_chars=args.phase2_max_chars,
    )
    write_phase2_outputs(phase2_dir, scan_dir / "page_scan.json", scan_data, chunks)
    return len(chunks)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    scan_root = Path(args.scan_root)
    phase2_root = Path(args.phase2_root)

    images = sorted_page_images(input_dir)
    if not images:
        print(f"No page images found in {input_dir}")
        return 1

    if args.max_pages is not None:
        images = images[: args.max_pages]

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
    )

    summary: list[dict[str, object]] = []
    for image_path in images:
        page_name = image_path.stem
        scan_dir = scan_root / page_name
        phase2_dir = phase2_root / page_name

        scan_json = scan_dir / "page_scan.json"
        phase2_summary = phase2_dir / "summary.json"
        if not args.overwrite and scan_json.is_file() and phase2_summary.is_file():
            summary.append(
                {
                    "page": page_name,
                    "status": "skipped",
                    "scan_dir": str(scan_dir),
                    "phase2_dir": str(phase2_dir),
                }
            )
            print(f"Skipped {page_name}: outputs already exist")
            continue

        scan_data = scan_page(image_path, scan_dir, ocr, args)
        chunk_count = prepare_page_phase2(scan_data, scan_dir, phase2_dir, args)
        summary.append(
            {
                "page": page_name,
                "status": "processed",
                "entries": len(scan_data.get("entries", [])),
                "chunks": chunk_count,
                "scan_dir": str(scan_dir),
                "phase2_dir": str(phase2_dir),
            }
        )
        print(f"Processed {page_name}: {len(scan_data.get('entries', []))} entries, {chunk_count} chunk(s)")

    batch_summary = {
        "input_dir": str(input_dir),
        "page_count": len(images),
        "processed_pages": len([item for item in summary if item["status"] == "processed"]),
        "skipped_pages": len([item for item in summary if item["status"] == "skipped"]),
        "pages": summary,
    }
    summary_path = phase2_root / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(batch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote batch summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())