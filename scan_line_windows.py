from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from paddleocr import PaddleOCR
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "PaddleOCR is not installed. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially scan one oversized line or crop with bounded sliding windows and optional upscaling."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="ocr_benchmark/page_002_seed_paddleocr/lines/line_006.png",
        help="Path to the oversized line or crop image",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_benchmark/page_002_line_006_windows_paddleocr",
        help="Directory where per-window OCR outputs will be written",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="PaddleOCR recognition language code. Use `hi` for Devanagari.",
    )
    parser.add_argument(
        "--upscale",
        type=float,
        default=1.5,
        help="Upscale factor applied before windowing. Use modest values to avoid memory spikes.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=2200,
        help="Window width after upscaling",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Window height after upscaling",
    )
    parser.add_argument(
        "--step-x",
        type=int,
        default=1600,
        help="Horizontal step between neighboring windows",
    )
    parser.add_argument(
        "--step-y",
        type=int,
        default=700,
        help="Vertical step between neighboring windows",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=6,
        help="Maximum number of windows to process in one run",
    )
    parser.add_argument(
        "--window-offset",
        type=int,
        default=0,
        help="Skip this many generated windows before OCR starts",
    )
    parser.add_argument(
        "--write-window-images",
        action="store_true",
        help="Write the individual window images to disk for inspection",
    )
    parser.add_argument(
        "--trim-content",
        action="store_true",
        help="Trim surrounding whitespace before upscaling and windowing so the windows land on text sooner.",
    )
    parser.add_argument(
        "--densest-first",
        action="store_true",
        help="Rank generated windows by ink density and scan the densest windows first instead of raw grid order.",
    )
    return parser.parse_args()


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def upscale_image(image, factor: float):
    if factor <= 0:
        raise ValueError("upscale must be positive.")
    if factor == 1.0:
        return image

    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_CUBIC if factor > 1.0 else cv2.INTER_AREA,
    )


def trim_to_content(image, margin: int = 20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return image, {"x1": 0, "y1": 0, "x2": image.shape[1], "y2": image.shape[0]}

    x, y, width, height = cv2.boundingRect(points)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + width + margin)
    y2 = min(image.shape[0], y + height + margin)
    return image[y1:y2, x1:x2], {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def generate_windows(width: int, height: int, window_width: int, window_height: int, step_x: int, step_y: int):
    if window_width <= 0 or window_height <= 0:
        raise ValueError("Window dimensions must be positive.")
    if step_x <= 0 or step_y <= 0:
        raise ValueError("Step sizes must be positive.")

    x_starts = list(range(0, max(1, width - window_width + 1), step_x))
    y_starts = list(range(0, max(1, height - window_height + 1), step_y))

    if not x_starts or x_starts[-1] != max(0, width - window_width):
        x_starts.append(max(0, width - window_width))
    if not y_starts or y_starts[-1] != max(0, height - window_height):
        y_starts.append(max(0, height - window_height))

    windows = []
    index = 1
    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(width, x1 + window_width)
            y2 = min(height, y1 + window_height)
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


def annotate_window_density(image, windows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    annotated = []
    for window in windows:
        region = binary[window["y1"] : window["y2"], window["x1"] : window["x2"]]
        density = int(cv2.countNonZero(region))
        annotated.append({**window, "ink_pixels": density})
    return annotated


def recognize_window(ocr: PaddleOCR, window_image) -> list[str]:
    raw_result = ocr.predict(window_image)
    if not raw_result:
        return []
    page = raw_result[0]
    if not hasattr(page, "get"):
        return []
    texts = page.get("rec_texts", [])
    return [str(text).strip() for text in texts if str(text).strip()]


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    try:
        source_image = load_image(image_path)
        content_bounds = {"x1": 0, "y1": 0, "x2": source_image.shape[1], "y2": source_image.shape[0]}
        base_image = source_image
        if args.trim_content:
            base_image, content_bounds = trim_to_content(source_image)
        working_image = upscale_image(base_image, args.upscale)
        windows = generate_windows(
            width=working_image.shape[1],
            height=working_image.shape[0],
            window_width=args.window_width,
            window_height=args.window_height,
            step_x=args.step_x,
            step_y=args.step_y,
        )
        windows = annotate_window_density(working_image, windows)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    if args.densest_first:
        windows = sorted(windows, key=lambda window: int(window["ink_pixels"]), reverse=True)

    windows = windows[args.window_offset : args.window_offset + args.max_windows]
    if not windows:
        print("No windows selected for OCR.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=args.lang,
    )

    window_reports = []
    aggregated_texts = []
    windows_dir = output_dir / "windows"
    if args.write_window_images:
        windows_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        crop = working_image[window["y1"] : window["y2"], window["x1"] : window["x2"]]
        texts = recognize_window(ocr, crop)
        joined = " ".join(texts)
        aggregated_texts.append(joined)
        if args.write_window_images:
            cv2.imwrite(str(windows_dir / f"window_{window['window_index']:03}.png"), crop)
        window_reports.append(
            {
                **window,
                "text_segments": texts,
                "text": joined,
            }
        )

    metadata = {
        "source_image": str(image_path),
        "source_width": int(source_image.shape[1]),
        "source_height": int(source_image.shape[0]),
        "content_bounds": content_bounds,
        "upscale": args.upscale,
        "processed_width": int(working_image.shape[1]),
        "processed_height": int(working_image.shape[0]),
        "window_width": args.window_width,
        "window_height": args.window_height,
        "step_x": args.step_x,
        "step_y": args.step_y,
        "window_offset": args.window_offset,
        "windows_scanned": len(window_reports),
        "lang": args.lang,
    }
    (output_dir / "scan.json").write_text(
        json.dumps({"metadata": metadata, "windows": window_reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "scan.txt").write_text(
        "\n".join(text for text in aggregated_texts if text),
        encoding="utf-8",
    )

    report_lines = [
        f"# Window OCR for {image_path.name}",
        "",
        f"Windows scanned: {len(window_reports)}",
        f"Upscale: {args.upscale}",
        "",
    ]
    for window in window_reports:
        report_lines.append(
            f"## Window {window['window_index']} ({window['x1']},{window['y1']})-({window['x2']},{window['y2']})"
        )
        report_lines.append("")
        report_lines.append(window["text"])
        report_lines.append("")
    (output_dir / "scan.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Scanned {len(window_reports)} window(s) from {image_path} into {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())