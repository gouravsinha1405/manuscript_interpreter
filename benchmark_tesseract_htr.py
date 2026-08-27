from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run line-level Tesseract HTR on manuscript crops and optionally score "
            "predictions against gold transcriptions using CER/WER."
        )
    )
    parser.add_argument(
        "lines_dir",
        nargs="?",
        default="line_images/page_002",
        help="Directory containing line_*.png crops",
    )
    parser.add_argument(
        "--output-dir",
        default="ocr_benchmark/page_002_tesseract_htr",
        help="Directory for predictions and metrics",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="san",
        help=(
            "Tesseract language/model name. Use a custom historical-Devanagari "
            "traineddata model name here when available."
        ),
    )
    parser.add_argument(
        "--tessdata-dir",
        default=None,
        help="Optional directory containing custom .traineddata files",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=7,
        help="Tesseract page segmentation mode. 7 = single text line.",
    )
    parser.add_argument(
        "--gold-tsv",
        default=None,
        help=(
            "Optional UTF-8 TSV with two columns: line image filename and gold transcription. "
            "Example: line_001.png<TAB>अथ ..."
        ),
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Maximum number of lines to process; 0 means all lines.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def levenshtein_distance(source: list[str], target: list[str]) -> int:
    if len(source) < len(target):
        source, target = target, source

    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            substitution = previous[j - 1] + (source_item != target_item)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / float(len(reference))


def cer(reference: str, hypothesis: str) -> float:
    return error_rate(list(reference), list(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    return error_rate(reference.split(), hypothesis.split())


def load_gold(path: Path) -> dict[str, str]:
    gold: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" not in raw_line:
            raise ValueError(f"Invalid gold TSV at line {line_number}: expected filename<TAB>text")
        filename, text = raw_line.split("\t", 1)
        gold[filename.strip()] = normalize_text(text)
    return gold


def run_tesseract(
    image_path: Path,
    lang: str,
    psm: int,
    tessdata_dir: Path | None,
) -> str:
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        "1",
    ]
    if tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(tessdata_dir)])

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Tesseract failed for {image_path.name}: {stderr}")
    return normalize_text(completed.stdout)


def main() -> int:
    args = parse_args()

    if shutil.which("tesseract") is None:
        raise SystemExit("Tesseract executable was not found on PATH.")

    lines_dir = Path(args.lines_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not lines_dir.is_dir():
        raise SystemExit(f"Line directory not found: {lines_dir}")

    line_paths = sorted(lines_dir.glob("line_*.png"))
    if args.max_lines > 0:
        line_paths = line_paths[: args.max_lines]
    if not line_paths:
        raise SystemExit(f"No line_*.png images found in {lines_dir}")

    tessdata_dir = Path(args.tessdata_dir) if args.tessdata_dir else None
    if tessdata_dir is not None and not tessdata_dir.is_dir():
        raise SystemExit(f"Tessdata directory not found: {tessdata_dir}")

    gold = load_gold(Path(args.gold_tsv)) if args.gold_tsv else {}

    line_results: list[dict[str, object]] = []
    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0

    for image_path in line_paths:
        hypothesis = run_tesseract(
            image_path,
            lang=args.tesseract_lang,
            psm=args.psm,
            tessdata_dir=tessdata_dir,
        )

        result: dict[str, object] = {
            "line_image": image_path.name,
            "text": hypothesis,
        }

        if image_path.name in gold:
            reference = gold[image_path.name]
            char_errors = levenshtein_distance(list(reference), list(hypothesis))
            word_errors = levenshtein_distance(reference.split(), hypothesis.split())
            result.update(
                {
                    "gold": reference,
                    "cer": round(cer(reference, hypothesis), 6),
                    "wer": round(wer(reference, hypothesis), 6),
                    "char_errors": char_errors,
                    "reference_chars": len(reference),
                    "word_errors": word_errors,
                    "reference_words": len(reference.split()),
                }
            )
            total_char_errors += char_errors
            total_chars += len(reference)
            total_word_errors += word_errors
            total_words += len(reference.split())

        line_results.append(result)

    aggregate_cer = total_char_errors / total_chars if total_chars else None
    aggregate_wer = total_word_errors / total_words if total_words else None

    payload = {
        "engine": "tesseract",
        "language": args.tesseract_lang,
        "psm": args.psm,
        "tessdata_dir": str(tessdata_dir) if tessdata_dir else None,
        "line_count": len(line_results),
        "gold_line_count": sum(1 for item in line_results if "gold" in item),
        "aggregate_cer": round(aggregate_cer, 6) if aggregate_cer is not None else None,
        "aggregate_wer": round(aggregate_wer, 6) if aggregate_wer is not None else None,
        "lines": line_results,
    }

    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    transcript = "\n".join(str(item["text"]) for item in line_results)
    (output_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

    report = [
        "# Tesseract historical HTR benchmark",
        "",
        f"Lines processed: {len(line_results)}",
        f"Language/model: `{args.tesseract_lang}`",
        f"PSM: `{args.psm}`",
    ]
    if aggregate_cer is not None:
        report.extend(
            [
                f"Gold lines scored: {payload['gold_line_count']}",
                f"Aggregate CER: `{aggregate_cer:.4f}`",
                f"Aggregate WER: `{aggregate_wer:.4f}`",
            ]
        )

    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Processed {len(line_results)} line(s) -> {output_dir}")
    if aggregate_cer is not None:
        print(f"CER={aggregate_cer:.4f} WER={aggregate_wer:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
