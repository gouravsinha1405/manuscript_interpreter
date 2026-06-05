from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OCR output for phase-2 normalization, grammar repair, and interpretation."
    )
    parser.add_argument(
        "scan_json",
        nargs="?",
        default="page_scan/page_004/page_scan.json",
        help="Path to page_scan.json or benchmark.json produced by the OCR pipeline",
    )
    parser.add_argument(
        "--output-dir",
        default="phase2_input/page_004",
        help="Directory where chunked phase-2 artifacts will be written",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=12,
        help="Maximum OCR lines per chunk",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=800,
        help="Maximum approximate characters per chunk",
    )
    return parser.parse_args()


def load_scan(scan_path: Path) -> dict[str, object]:
    if not scan_path.is_file():
        raise FileNotFoundError(f"Scan file not found: {scan_path}")
    return json.loads(scan_path.read_text(encoding="utf-8"))


def page_scan_entries(source_data: dict[str, object]) -> tuple[list[dict[str, object]] | None, dict[str, object]]:
    entries = source_data.get("entries")
    if not isinstance(entries, list):
        return None, {}
    metadata = source_data.get("metadata", {})
    return entries, metadata if isinstance(metadata, dict) else {}


def available_model_result(models: dict[str, object]) -> tuple[str, dict[str, object]]:
    preferred_model_order = ["paddleocr", "easyocr", "tesseract"]

    for model_name in preferred_model_order:
        candidate = models.get(model_name)
        if isinstance(candidate, dict) and candidate.get("available"):
            return model_name, candidate

    for model_name, candidate in models.items():
        if isinstance(candidate, dict) and candidate.get("available"):
            return str(model_name), candidate

    raise ValueError("Benchmark JSON does not contain an available OCR model result.")


def normalized_line_results(line_results: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, line_result in enumerate(line_results, start=1):
        if not isinstance(line_result, dict):
            continue

        text = str(line_result.get("text", "")).strip()
        if not text:
            continue

        normalized.append(
            {
                "text": text,
                "line_index": index,
                "line_image": line_result.get("line_image", ""),
                "segments": line_result.get("segments", []),
                "box": [],
            }
        )
    return normalized


def normalized_entries(source_data: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    entries, metadata = page_scan_entries(source_data)
    if entries is not None:
        return entries, metadata

    models = source_data.get("models")
    if not isinstance(models, dict):
        raise ValueError("OCR JSON does not contain a valid `entries` list or `models` mapping.")

    selected_model_name, selected_result = available_model_result(models)

    line_results = selected_result.get("line_results")
    if not isinstance(line_results, list):
        raise ValueError("Benchmark JSON does not contain a valid `line_results` list.")

    normalized = normalized_line_results(line_results)

    metadata = {
        "source_image": source_data.get("source_image", ""),
        "line_count": source_data.get("line_count", len(normalized)),
        "source_type": "benchmark",
        "model": selected_model_name,
    }
    return normalized, metadata


def chunk_entries(entries: list[dict[str, object]], max_lines: int, max_chars: int) -> list[list[dict[str, object]]]:
    chunks: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_chars = 0

    for entry in entries:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue

        projected_chars = current_chars + len(text) + (1 if current else 0)
        if current and (len(current) >= max_lines or projected_chars > max_chars):
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(entry)
        current_chars += len(text) + (1 if current_chars else 0)

    if current:
        chunks.append(current)

    return chunks


def chunk_text(entries: list[dict[str, object]]) -> str:
    return "\n".join(str(entry.get("text", "")).strip() for entry in entries if str(entry.get("text", "")).strip())


def chunk_bounds(entries: list[dict[str, object]]) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for entry in entries:
        for point in entry.get("box", []):
            if isinstance(point, list) and len(point) == 2:
                xs.append(float(point[0]))
                ys.append(float(point[1]))

    if not xs or not ys:
        return {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}

    return {
        "x1": min(xs),
        "y1": min(ys),
        "x2": max(xs),
        "y2": max(ys),
    }


def build_chunks(scan_data: dict[str, object], max_lines: int, max_chars: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    entries, metadata = normalized_entries(scan_data)

    chunks: list[dict[str, object]] = []
    for index, entry_group in enumerate(chunk_entries(entries, max_lines=max_lines, max_chars=max_chars), start=1):
        chunks.append(
            {
                "chunk_index": index,
                "line_count": len(entry_group),
                "char_count": sum(len(str(entry.get("text", ""))) for entry in entry_group),
                "bounds": chunk_bounds(entry_group),
                "text": chunk_text(entry_group),
                "entries": entry_group,
                "phase2_tasks": [
                    "normalize OCR noise",
                    "infer likely Devanagari/Sanskrit readings",
                    "repair grammar using local context",
                    "produce modern Hindi or modern Sanskrit interpretation",
                ],
            }
        )
    return chunks, metadata


def write_outputs(
    output_dir: Path,
    scan_path: Path,
    metadata: dict[str, object],
    chunks: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_scan": str(scan_path),
        "page_metadata": metadata,
        "chunk_count": len(chunks),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    jsonl_lines = [json.dumps(chunk, ensure_ascii=False) for chunk in chunks]
    (output_dir / "chunks.jsonl").write_text("\n".join(jsonl_lines), encoding="utf-8")

    md_lines = ["# Phase 2 Input", ""]
    for chunk in chunks:
        md_lines.append(f"## Chunk {chunk['chunk_index']}")
        md_lines.append("")
        md_lines.append(f"Bounds: {chunk['bounds']}")
        md_lines.append(f"Lines: {chunk['line_count']}")
        md_lines.append("")
        md_lines.append(chunk["text"])
        md_lines.append("")
        md_lines.append("Tasks: normalize OCR, infer likely reading, repair grammar, interpret into modern language.")
        md_lines.append("")
    (output_dir / "chunks.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scan_path = Path(args.scan_json)
    output_dir = Path(args.output_dir)

    try:
        scan_data = load_scan(scan_path)
        chunks, metadata = build_chunks(scan_data, max_lines=args.max_lines, max_chars=args.max_chars)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    write_outputs(output_dir, scan_path, metadata, chunks)
    print(f"Prepared {len(chunks)} phase-2 chunk(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())