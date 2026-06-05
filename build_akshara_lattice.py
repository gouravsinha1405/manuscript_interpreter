from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
import os
from pathlib import Path


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
DEVANAGARI_ONLY_RE = re.compile(r"[^\u0900-\u097F]+")
MATRAS_AND_MARKS_RE = re.compile(r"[\u0900-\u0903\u093A-\u094D\u0951-\u0957\u0962-\u0963]")

VIRAMA = "\u094d"
NUKTA = "\u093c"
DEPENDENT_SIGNS = {
    "\u0900",
    "\u0901",
    "\u0902",
    "\u0903",
    "\u093a",
    "\u093b",
    "\u093c",
    "\u093e",
    "\u093f",
    "\u0940",
    "\u0941",
    "\u0942",
    "\u0943",
    "\u0944",
    "\u0945",
    "\u0946",
    "\u0947",
    "\u0948",
    "\u0949",
    "\u094a",
    "\u094b",
    "\u094c",
    "\u094e",
    "\u094f",
    "\u0951",
    "\u0952",
    "\u0953",
    "\u0954",
    "\u0955",
    "\u0956",
    "\u0957",
    "\u0962",
    "\u0963",
}


@dataclass
class LexiconEntry:
    token: str
    lemma: str
    pos: str
    gloss: str
    normalized: str
    skeleton: str
    aksharas: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an akshara-level candidate lattice from noisy OCR segments."
    )
    parser.add_argument(
        "chunks_json",
        nargs="?",
        default="phase2_input/page_002_from_full_lines/chunks.json",
        help="Prepared OCR chunk JSON to analyze",
    )
    parser.add_argument(
        "--lexicon",
        default="resources/modern_sanskrit_seed_lexicon.tsv",
        help="TSV lexicon with columns: token, lemma, pos, gloss",
    )
    parser.add_argument(
        "--output-dir",
        default="phase2_input/page_002_akshara_lattice",
        help="Directory where akshara lattice outputs will be written",
    )
    parser.add_argument(
        "--max-span-aksharas",
        type=int,
        default=8,
        help="Maximum OCR akshara span length considered for matching against lexicon tokens",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.62,
        help="Minimum candidate score retained in the akshara lattice",
    )
    parser.add_argument(
        "--max-candidates-per-span",
        type=int,
        default=5,
        help="Maximum number of lexicon candidates stored for each OCR akshara span",
    )
    parser.add_argument(
        "--indic-nlp-resources",
        default="",
        help="Path to Indic NLP resources for library-backed orthographic syllabification.",
    )
    return parser.parse_args()


def normalize_token(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = DEVANAGARI_ONLY_RE.sub("", normalized)
    return normalized.strip()


def token_skeleton(text: str) -> str:
    return MATRAS_AND_MARKS_RE.sub("", text)


def local_akshara_split(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text)
    units: list[str] = []
    current = ""
    pending_join = False

    for char in normalized:
        if not DEVANAGARI_RE.match(char):
            if current:
                units.append(current)
                current = ""
            pending_join = False
            continue

        if not current:
            current = char
        elif pending_join:
            current += char
            pending_join = False
        elif char in DEPENDENT_SIGNS:
            current += char
        else:
            units.append(current)
            current = char

        if char == VIRAMA or char == NUKTA:
            pending_join = True

    if current:
        units.append(current)

    return [unit for unit in units if normalize_token(unit)]


def candidate_resource_paths(explicit_path: str) -> list[Path]:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path))
    env_path = os.environ.get("INDIC_NLP_RESOURCES", "").strip()
    if env_path:
        paths.append(Path(env_path))
    paths.extend(
        [
            Path("indic_nlp_resources"),
            Path.home() / "indic_nlp_resources",
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def build_akshara_splitter(resources_path: str):
    try:
        from indicnlp import common, loader
        from indicnlp.syllable import syllabifier
    except ImportError:
        return local_akshara_split, "local_fallback"

    for candidate in candidate_resource_paths(resources_path):
        if not candidate.exists():
            continue
        try:
            common.set_resources_path(str(candidate))
            loader.load()

            def split_with_indicnlp(text: str) -> list[str]:
                cleaned = normalize_token(text)
                if not cleaned:
                    return []
                return [unit for unit in syllabifier.orthographic_syllabify_improved(cleaned, "sa") if normalize_token(unit)]

            return split_with_indicnlp, f"indicnlp:{candidate}"
        except Exception:
            continue

    return local_akshara_split, "local_fallback"


def akshara_split(text: str) -> list[str]:
    return local_akshara_split(text)


def load_chunks(chunks_path: Path) -> list[dict[str, object]]:
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Chunk JSON must contain a list of chunks.")
    return data


def entry_segments(entry: dict[str, object]) -> list[str]:
    segments = entry.get("segments")
    if isinstance(segments, list) and segments:
        return [str(segment) for segment in segments]
    return [str(entry.get("text", ""))]


def load_lexicon(lexicon_path: Path, splitter) -> list[LexiconEntry]:
    entries: list[LexiconEntry] = []
    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        token, lemma, pos, gloss = [part.strip() for part in stripped.split("\t")]
        normalized = normalize_token(token)
        if not normalized:
            continue
        entries.append(
            LexiconEntry(
                token=token,
                lemma=lemma,
                pos=pos,
                gloss=gloss,
                normalized=normalized,
                skeleton=token_skeleton(normalized),
                aksharas=splitter(normalized),
            )
        )
    return entries


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def akshara_window_score(observed: str, entry: LexiconEntry, observed_aksharas: list[str]) -> tuple[float, dict[str, object]]:
    observed_skeleton = token_skeleton(observed)
    char_ratio = ratio(observed, entry.normalized)
    skeleton_ratio = ratio(observed_skeleton, entry.skeleton)
    akshara_text = "".join(observed_aksharas)
    akshara_ratio = ratio(akshara_text, entry.normalized)

    length_gap = abs(len(observed_aksharas) - len(entry.aksharas))
    length_bonus = max(0.0, 0.08 - (0.02 * length_gap))

    score = (0.45 * char_ratio) + (0.25 * skeleton_ratio) + (0.22 * akshara_ratio) + length_bonus
    return round(min(score, 1.0), 4), {
        "char_ratio": round(char_ratio, 4),
        "skeleton_ratio": round(skeleton_ratio, 4),
        "akshara_ratio": round(akshara_ratio, 4),
        "length_gap": length_gap,
        "length_bonus": round(length_bonus, 4),
    }


def segment_record(
    chunk_index: int,
    entry_index: int,
    segment_index: int,
    entry: dict[str, object],
    segment: str,
    splitter,
) -> dict[str, object] | None:
    cleaned = normalize_token(segment)
    aksharas = splitter(segment)
    if not cleaned or not aksharas:
        return None
    return {
        "chunk_index": chunk_index,
        "entry_index": entry_index,
        "segment_index": segment_index,
        "line_index": entry.get("line_index"),
        "line_image": entry.get("line_image"),
        "segment_text": segment,
        "normalized_segment": cleaned,
        "aksharas": aksharas,
    }


def segment_records_for_entry(
    chunk_index: int,
    entry_index: int,
    entry: dict[str, object],
    splitter,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for segment_index, segment in enumerate(entry_segments(entry), start=1):
        record = segment_record(chunk_index, entry_index, segment_index, entry, segment, splitter)
        if record is not None:
            records.append(record)
    return records


def segment_records(chunks: list[dict[str, object]], splitter) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index", 0))
        entries = chunk.get("entries", [])
        if not isinstance(entries, list):
            continue
        for entry_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            records.extend(segment_records_for_entry(chunk_index, entry_index, entry, splitter))
    return records


def lattice_candidates_for_segment(
    record: dict[str, object],
    lexicon: list[LexiconEntry],
    max_span_aksharas: int,
    candidate_threshold: float,
    max_candidates_per_span: int,
) -> list[dict[str, object]]:
    aksharas = [str(unit) for unit in record["aksharas"]]
    candidates: list[dict[str, object]] = []
    for start in range(len(aksharas)):
        for end in range(start + 1, min(len(aksharas), start + max_span_aksharas) + 1):
            window_aksharas = aksharas[start:end]
            observed = "".join(window_aksharas)
            span_candidates: list[dict[str, object]] = []
            for entry in lexicon:
                score, details = akshara_window_score(observed, entry, window_aksharas)
                if score < candidate_threshold:
                    continue
                span_candidates.append(
                    {
                        "token": entry.token,
                        "lemma": entry.lemma,
                        "pos": entry.pos,
                        "gloss": entry.gloss,
                        "score": score,
                        "details": details,
                    }
                )
            if not span_candidates:
                continue
            span_candidates.sort(key=lambda item: float(item["score"]), reverse=True)
            candidates.append(
                {
                    "start_akshara": start,
                    "end_akshara": end,
                    "observed_span": observed,
                    "observed_aksharas": window_aksharas,
                    "candidates": span_candidates[:max_candidates_per_span],
                }
            )
    candidates.sort(key=lambda item: float(item["candidates"][0]["score"]), reverse=True)
    return candidates


def best_path(records: list[dict[str, object]]) -> list[dict[str, object]]:
    path: list[dict[str, object]] = []
    for record in records:
        lattice = record.get("lattice_candidates", [])
        if not isinstance(lattice, list) or not lattice:
            continue
        best = lattice[0]
        candidate = best["candidates"][0]
        path.append(
            {
                "line_index": record.get("line_index"),
                "segment_index": record.get("segment_index"),
                "observed_span": best["observed_span"],
                "observed_aksharas": best["observed_aksharas"],
                "best_token": candidate["token"],
                "best_score": candidate["score"],
                "gloss": candidate["gloss"],
            }
        )
    return path


def write_outputs(
    output_dir: Path,
    records: list[dict[str, object]],
    best_candidates: list[dict[str, object]],
    backend: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segment_lattice.json").write_text(
        json.dumps({"backend": backend, "segments": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "best_akshara_candidates.json").write_text(
        json.dumps({"backend": backend, "candidates": best_candidates}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    chunks = load_chunks(Path(args.chunks_json))
    splitter, backend = build_akshara_splitter(args.indic_nlp_resources)
    lexicon = load_lexicon(Path(args.lexicon), splitter)
    records = segment_records(chunks, splitter)
    for record in records:
        record["lattice_candidates"] = lattice_candidates_for_segment(
            record=record,
            lexicon=lexicon,
            max_span_aksharas=args.max_span_aksharas,
            candidate_threshold=args.candidate_threshold,
            max_candidates_per_span=args.max_candidates_per_span,
        )
    best_candidates = best_path(records)
    write_outputs(Path(args.output_dir), records, best_candidates, backend)
    print(
        f"Built akshara lattice for {len(records)} OCR segments using {backend} and wrote outputs to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())