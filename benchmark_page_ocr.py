from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DEVANAGARI_TOKEN_RE = re.compile(r"[\u0900-\u097F]+")
DEVANAGARI_ONLY_RE = re.compile(r"[^\u0900-\u097F]+")

from segment_lines import crop_lines

try:
    import easyocr
except ImportError:
    easyocr = None

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment one manuscript page into lines and benchmark OCR models on the resulting crops."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="page_images/page_001.png",
        help="Path to the page image to benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_benchmark/page_001",
        help="Directory where segmented lines and OCR comparison outputs will be written",
    )
    parser.add_argument(
        "--reuse-lines-dir",
        default=None,
        help="Optional existing line-crop directory. If set, segmentation is skipped and these line images are reused.",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="Primary OCR language code. `hi` is the Devanagari option used for PaddleOCR.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["paddleocr", "easyocr", "tesseract"],
        default=["paddleocr"],
        help="OCR engines to run for this benchmark. Defaults to the currently safe installed engine only.",
    )
    parser.add_argument(
        "--easyocr-langs",
        nargs="+",
        default=["hi", "en"],
        help="EasyOCR language codes to load",
    )
    parser.add_argument(
        "--min-line-height",
        type=int,
        default=20,
        help="Minimum retained line-band height in pixels",
    )
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=12,
        help="Maximum vertical gap for merging nearby line bands",
    )
    parser.add_argument(
        "--pad-y",
        type=int,
        default=8,
        help="Vertical padding around detected line bands",
    )
    parser.add_argument(
        "--pad-x",
        type=int,
        default=12,
        help="Horizontal padding around detected text extents",
    )
    parser.add_argument(
        "--debug-segmentation",
        action="store_true",
        help="Write segmentation debug artifacts for the page",
    )
    parser.add_argument(
        "--line-offset",
        type=int,
        default=0,
        help="Skip this many segmented lines before OCR starts.",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=8,
        help="Maximum number of line crops to send to OCR in one run.",
    )
    parser.add_argument(
        "--max-line-side",
        type=int,
        default=3200,
        help="Downscale each line crop so its longest side is at most this many pixels before OCR.",
    )
    parser.add_argument(
        "--paddle-variants",
        nargs="+",
        choices=[
            "raw",
            "gray",
            "autocontrast",
            "clahe",
            "normalized",
            "adaptive",
            "otsu",
            "upscale_adaptive",
            "sharpen_adaptive",
        ],
        default=["raw"],
        help="Image variants to try per line for PaddleOCR. The best-scoring variant is kept for each line.",
    )
    parser.add_argument(
        "--full-throttle-paddle",
        action="store_true",
        help="Run a stronger PaddleOCR preset across multiple aggressive preprocessing variants.",
    )
    parser.add_argument(
        "--lexicon",
        default="resources/modern_sanskrit_seed_lexicon.tsv",
        help="Optional Sanskrit lexicon TSV used to score OCR variant plausibility.",
    )
    return parser.parse_args()


def count_characters(text: str) -> dict[str, int]:
    return {
        "characters": len(text),
        "non_whitespace_characters": sum(1 for char in text if not char.isspace()),
    }


def line_paths(line_dir: Path) -> list[Path]:
    return sorted(line_dir.glob("line_*.png"))


def select_lines(line_images: list[Path], line_offset: int, max_lines: int) -> list[Path]:
    if line_offset < 0:
        raise ValueError("line-offset must be non-negative.")
    if max_lines <= 0:
        raise ValueError("max-lines must be positive.")
    return line_images[line_offset : line_offset + max_lines]


def load_resized_line(line_image: Path, max_line_side: int):
    image = cv2.imread(str(line_image))
    if image is None:
        raise FileNotFoundError(f"Could not read line image: {line_image}")

    if max_line_side <= 0:
        return image

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_line_side:
        return image

    scale = max_line_side / float(longest_side)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def prepare_paddle_variant(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "raw":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if variant == "gray":
        return ensure_bgr(gray)

    auto = np.array(ImageOps.autocontrast(Image.fromarray(gray), cutoff=1))
    if variant == "autocontrast":
        return ensure_bgr(auto)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(auto)
    if variant == "clahe":
        return ensure_bgr(clahe_image)

    background = cv2.GaussianBlur(clahe_image, (0, 0), sigmaX=31, sigmaY=31)
    normalized = cv2.divide(clahe_image, background, scale=255)
    if variant == "normalized":
        return ensure_bgr(normalized)

    if variant == "adaptive":
        adaptive = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        return ensure_bgr(adaptive)

    if variant == "otsu":
        _, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ensure_bgr(otsu)

    if variant == "upscale_adaptive":
        enlarged = cv2.resize(normalized, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        sharpened = cv2.GaussianBlur(enlarged, (0, 0), sigmaX=1.2)
        sharpened = cv2.addWeighted(enlarged, 1.6, sharpened, -0.6, 0)
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            9,
        )
        return ensure_bgr(adaptive)

    if variant == "sharpen_adaptive":
        sharpened = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(normalized, 1.8, sharpened, -0.8, 0)
        adaptive = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        return ensure_bgr(adaptive)

    raise ValueError(f"Unsupported PaddleOCR variant: {variant}")


def best_paddle_variant_names(args: argparse.Namespace) -> list[str]:
    if args.full_throttle_paddle:
        return [
            "raw",
            "autocontrast",
            "clahe",
            "normalized",
            "adaptive",
            "otsu",
            "upscale_adaptive",
            "sharpen_adaptive",
        ]
    return args.paddle_variants


def score_paddle_text(segments: list[str], scores: list[float]) -> tuple[float, int]:
    confidence = sum(scores) / len(scores) if scores else 0.0
    non_whitespace = sum(1 for char in " ".join(segments) if not char.isspace())
    return confidence, non_whitespace


def normalize_token(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = DEVANAGARI_ONLY_RE.sub("", normalized)
    return normalized.strip()


def load_lexicon_tokens(lexicon_path: Path) -> set[str]:
    if not lexicon_path.is_file():
        return set()
    tokens: set[str] = set()
    for raw_line in lexicon_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [part.strip() for part in stripped.split("\t")]
        if len(parts) != 4:
            continue
        token = normalize_token(parts[0])
        if token:
            tokens.add(token)
    return tokens


def lexical_token_ratio(text: str, lexicon_tokens: set[str]) -> float:
    if not lexicon_tokens:
        return 0.0
    tokens = [normalize_token(token) for token in DEVANAGARI_TOKEN_RE.findall(unicodedata.normalize("NFC", text))]
    tokens = [token for token in tokens if len(token) >= 2]
    if not tokens:
        return 0.0
    matched = sum(1 for token in tokens if token in lexicon_tokens)
    return matched / float(len(tokens))


def devanagari_ratio(text: str) -> float:
    visible_chars = [char for char in text if not char.isspace()]
    if not visible_chars:
        return 0.0
    devanagari_chars = sum(1 for char in visible_chars if DEVANAGARI_RE.match(char))
    return devanagari_chars / float(len(visible_chars))


def latin_ratio(text: str) -> float:
    visible_chars = [char for char in text if not char.isspace()]
    if not visible_chars:
        return 0.0
    latin_chars = sum(1 for char in visible_chars if "A" <= char <= "Z" or "a" <= char <= "z")
    return latin_chars / float(len(visible_chars))


def digit_ratio(text: str) -> float:
    visible_chars = [char for char in text if not char.isspace()]
    if not visible_chars:
        return 0.0
    digits = sum(1 for char in visible_chars if char.isdigit())
    return digits / float(len(visible_chars))


def heuristic_variant_score(
    text: str,
    mean_score: float,
    lexicon_tokens: set[str],
) -> dict[str, float]:
    non_whitespace = sum(1 for char in text if not char.isspace())
    script_ratio = devanagari_ratio(text)
    alpha_noise = latin_ratio(text)
    numeric_noise = digit_ratio(text)
    lexicon_ratio = lexical_token_ratio(text, lexicon_tokens)
    length_bonus = min(1.0, math.log1p(non_whitespace) / 5.0) if non_whitespace else 0.0
    short_penalty = 0.15 if 0 < non_whitespace < 6 else 0.0
    overall = (
        0.35 * mean_score
        + 0.30 * script_ratio
        + 0.25 * lexicon_ratio
        + 0.10 * length_bonus
        - 0.35 * alpha_noise
        - 0.20 * numeric_noise
        - short_penalty
    )
    return {
        "heuristic_score": round(overall, 4),
        "devanagari_ratio": round(script_ratio, 4),
        "latin_ratio": round(alpha_noise, 4),
        "digit_ratio": round(numeric_noise, 4),
        "lexicon_ratio": round(lexicon_ratio, 4),
        "length_bonus": round(length_bonus, 4),
        "short_penalty": round(short_penalty, 4),
    }


def choose_best_variant(variant_results: list[dict[str, object]], raw_text: str) -> dict[str, object]:
    if not variant_results:
        return {"variant": "raw", "text": raw_text, "segments": []}

    raw_candidate = next((item for item in variant_results if item.get("variant") == "raw"), variant_results[0])
    best_candidate = max(
        variant_results,
        key=lambda item: (
            float(item.get("heuristic_score", 0.0)),
            float(item.get("mean_score", 0.0)),
            int(item.get("non_whitespace_characters", 0)),
        ),
    )

    raw_non_whitespace = int(raw_candidate.get("non_whitespace_characters", 0))
    best_non_whitespace = int(best_candidate.get("non_whitespace_characters", 0))
    best_script_ratio = float(best_candidate.get("devanagari_ratio", 0.0))

    if raw_non_whitespace == 0 and (best_non_whitespace < 8 or best_script_ratio < 0.6):
        return {**raw_candidate, "text": "", "segments": [], "selected_by": "blank_suppression"}

    raw_heuristic = float(raw_candidate.get("heuristic_score", 0.0))
    best_heuristic = float(best_candidate.get("heuristic_score", 0.0))
    best_lexicon_ratio = float(best_candidate.get("lexicon_ratio", 0.0))
    if best_candidate.get("variant") != "raw" and raw_non_whitespace > 0 and best_lexicon_ratio <= 1e-6 and best_non_whitespace <= raw_non_whitespace + 2:
        return {**raw_candidate, "selected_by": "raw_anchor"}
    if best_candidate.get("variant") != "raw" and best_heuristic < raw_heuristic + 0.08:
        return {**raw_candidate, "selected_by": "raw_anchor"}

    return {**best_candidate, "selected_by": "heuristic_best"}


def recognize_with_paddleocr(
    line_images: list[Path],
    lang: str,
    max_line_side: int,
    variants: list[str],
    lexicon_tokens: set[str],
) -> dict[str, object]:
    if PaddleOCR is None:
        return {"available": False, "reason": "paddleocr package is not installed"}

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang=lang,
    )

    line_results: list[dict[str, object]] = []
    texts: list[str] = []
    for line_image in line_images:
        base_image = load_resized_line(line_image, max_line_side=max_line_side)
        variant_results: list[dict[str, object]] = []
        for variant in variants:
            prepared_image = prepare_paddle_variant(base_image, variant)
            raw_result = ocr.predict(prepared_image)
            line_texts: list[str] = []
            line_scores: list[float] = []
            if raw_result:
                page = raw_result[0]
                texts_in_page = page.get("rec_texts", []) if hasattr(page, "get") else []
                scores_in_page = page.get("rec_scores", []) if hasattr(page, "get") else []
                line_texts = [str(text) for text in texts_in_page if str(text).strip()]
                line_scores = [float(score) for score in scores_in_page[: len(line_texts)]]

            confidence, non_whitespace = score_paddle_text(line_texts, line_scores)
            heuristic = heuristic_variant_score(" ".join(line_texts).strip(), confidence, lexicon_tokens)
            variant_results.append(
                {
                    "variant": variant,
                    "text": " ".join(line_texts).strip(),
                    "segments": line_texts,
                    "scores": [round(score, 4) for score in line_scores],
                    "mean_score": round(confidence, 4),
                    "non_whitespace_characters": non_whitespace,
                    **heuristic,
                }
            )

        best_variant = choose_best_variant(variant_results, raw_text="")
        merged_text = str(best_variant.get("text", "")).strip()
        texts.append(merged_text)
        line_results.append(
            {
                "line_image": str(line_image),
                "selected_variant": best_variant.get("variant", "raw"),
                "selected_mean_score": best_variant.get("mean_score", 0.0),
                "selected_heuristic_score": best_variant.get("heuristic_score", 0.0),
                "selected_by": best_variant.get("selected_by", "heuristic_best"),
                "text": merged_text,
                "segments": best_variant.get("segments", []),
                "variant_results": variant_results,
            }
        )

    joined = "\n".join(text for text in texts if text)
    metrics = count_characters(joined)
    return {
        "available": True,
        "model": "paddleocr",
        "lang": lang,
        "variants": variants,
        "line_results": line_results,
        "text": joined,
        **metrics,
    }


def recognize_with_easyocr(line_images: list[Path], languages: list[str], max_line_side: int) -> dict[str, object]:
    if easyocr is None:
        return {"available": False, "reason": "easyocr package is not installed"}

    reader = easyocr.Reader(languages, gpu=False, verbose=False)
    line_results: list[dict[str, object]] = []
    texts: list[str] = []

    for line_image in line_images:
        prepared_image = load_resized_line(line_image, max_line_side=max_line_side)
        detections = reader.readtext(prepared_image, detail=0, paragraph=True)
        line_texts = [str(text).strip() for text in detections if str(text).strip()]
        merged_text = " ".join(line_texts)
        texts.append(merged_text)
        line_results.append(
            {
                "line_image": str(line_image),
                "text": merged_text,
                "segments": line_texts,
            }
        )

    joined = "\n".join(text for text in texts if text)
    metrics = count_characters(joined)
    return {
        "available": True,
        "model": "easyocr",
        "languages": languages,
        "line_results": line_results,
        "text": joined,
        **metrics,
    }


def detect_tesseract_status() -> dict[str, object]:
    binary = shutil.which("tesseract")
    if binary is None:
        return {"available": False, "reason": "tesseract binary is not installed"}
    return {"available": False, "reason": f"tesseract found at {binary}, but no integration is wired yet"}


def write_outputs(output_dir: Path, image_path: Path, line_images: list[Path], results: dict[str, dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = {
        "source_image": str(image_path),
        "line_count": len(line_images),
        "models": results,
    }
    (output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        f"# OCR Benchmark for {image_path.name}",
        "",
        f"Line crops: {len(line_images)}",
        "",
    ]
    for name, result in results.items():
        report_lines.append(f"## {name}")
        report_lines.append("")
        if not result.get("available", False):
            report_lines.append(f"Unavailable: {result.get('reason', 'unknown reason')}")
            report_lines.append("")
            continue

        report_lines.append(f"Characters: {result['characters']}")
        report_lines.append(f"Non-whitespace characters: {result['non_whitespace_characters']}")
        report_lines.append("")
        report_lines.append(result.get("text", ""))
        report_lines.append("")

    (output_dir / "benchmark_report.md").write_text("\n".join(report_lines), encoding="utf-8")


def resolved_output_dir(base_output_dir: Path, model_names: list[str]) -> Path:
    if not model_names:
        return base_output_dir

    suffix = "_".join(model_names)
    if base_output_dir.name.endswith(suffix):
        return base_output_dir
    return base_output_dir.parent / f"{base_output_dir.name}_{suffix}"


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_dir = resolved_output_dir(Path(args.output_dir), args.models)
    line_dir = output_dir / "lines"
    debug_dir = output_dir / "segmentation_debug" if args.debug_segmentation else None

    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    if args.reuse_lines_dir is not None:
        line_dir = Path(args.reuse_lines_dir)
        if not line_dir.is_dir():
            print(f"Line directory not found: {line_dir}", file=sys.stderr)
            return 1
    else:
        crop_lines(
            image_path,
            line_dir,
            debug_dir=debug_dir,
            min_line_height=args.min_line_height,
            gap_threshold=args.gap_threshold,
            pad_y=args.pad_y,
            pad_x=args.pad_x,
        )

    lines = line_paths(line_dir)
    try:
        lines = select_lines(lines, line_offset=args.line_offset, max_lines=args.max_lines)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not lines:
        print("No segmented lines selected for OCR.", file=sys.stderr)
        return 1

    results: dict[str, dict[str, object]] = {}
    lexicon_tokens = load_lexicon_tokens(Path(args.lexicon))
    for model_name in args.models:
        if model_name == "paddleocr":
            results[model_name] = recognize_with_paddleocr(
                lines,
                lang=args.lang,
                max_line_side=args.max_line_side,
                variants=best_paddle_variant_names(args),
                lexicon_tokens=lexicon_tokens,
            )
        elif model_name == "easyocr":
            results[model_name] = recognize_with_easyocr(lines, languages=args.easyocr_langs, max_line_side=args.max_line_side)
        elif model_name == "tesseract":
            results[model_name] = detect_tesseract_status()

    write_outputs(output_dir, image_path, lines, results)

    available_models = [name for name, result in results.items() if result.get("available")]
    print(f"Benchmarked {len(available_models)} OCR model(s) across {len(lines)} line image(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())