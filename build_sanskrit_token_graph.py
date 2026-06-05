from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


DEVANAGARI_TOKEN_RE = re.compile(r"[\u0900-\u097F]+")
DEVANAGARI_ONLY_RE = re.compile(r"[^\u0900-\u097F]+")
MATRAS_AND_MARKS_RE = re.compile(r"[\u0900-\u0903\u093A-\u094D\u0951-\u0957\u0962-\u0963]")


@dataclass
class LexiconEntry:
    token: str
    lemma: str
    pos: str
    gloss: str
    normalized: str
    skeleton: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a candidate graph that maps noisy OCR Sanskrit tokens to standardized lexicon entries."
    )
    parser.add_argument(
        "chunks_json",
        nargs="?",
        default="phase2_input/page_002_from_full_lines/chunks.json",
        help="Path to the chunk JSON prepared for phase 2",
    )
    parser.add_argument(
        "--lexicon",
        default="resources/modern_sanskrit_seed_lexicon.tsv",
        help="TSV lexicon with columns: token, lemma, pos, gloss",
    )
    parser.add_argument(
        "--output-dir",
        default="phase2_input/page_002_token_graph_akshara",
        help="Directory where graph artifacts will be written",
    )
    parser.add_argument(
        "--akshara-lattice",
        default="phase2_input/page_002_akshara_lattice/best_akshara_candidates.json",
        help="Optional best_akshara_candidates.json file used to inject akshara-derived token candidates.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=2,
        help="Minimum Devanagari token length to keep",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum lexicon candidates stored per observed token",
    )
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=0.58,
        help="Minimum score for keeping a lexicon candidate edge",
    )
    parser.add_argument(
        "--replace-threshold",
        type=float,
        default=0.78,
        help="Minimum score for using a candidate as the chunk hypothesis token",
    )
    return parser.parse_args()


def load_chunks(chunks_path: Path) -> list[dict[str, object]]:
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Chunk JSON must contain a list of chunks.")
    return data


def load_akshara_candidates(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Akshara lattice file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = data.get("candidates", []) if isinstance(data, dict) else data
    if not isinstance(candidates, list):
        raise ValueError("Akshara lattice file must contain a candidate list.")
    return [item for item in candidates if isinstance(item, dict)]


def normalize_token(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = DEVANAGARI_ONLY_RE.sub("", normalized)
    return normalized.strip()


def token_skeleton(text: str) -> str:
    return MATRAS_AND_MARKS_RE.sub("", text)


def tokenize_segment(text: str, min_token_length: int) -> list[str]:
    tokens: list[str] = []
    for raw_token in DEVANAGARI_TOKEN_RE.findall(unicodedata.normalize("NFC", text)):
        cleaned = normalize_token(raw_token)
        if len(cleaned) >= min_token_length:
            tokens.append(cleaned)
    return tokens


def load_lexicon(lexicon_path: Path) -> list[LexiconEntry]:
    if not lexicon_path.is_file():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    entries: list[LexiconEntry] = []
    for line in lexicon_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parts = [part.strip() for part in stripped.split("\t")]
        if len(parts) != 4:
            raise ValueError("Each lexicon row must contain exactly 4 tab-separated columns.")

        token, lemma, pos, gloss = parts
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
            )
        )

    if not entries:
        raise ValueError("Lexicon file did not yield any usable entries.")
    return entries


def ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def best_window_ratio(observed: str, candidate: str) -> tuple[float, str]:
    if len(observed) <= len(candidate) + 2:
        return ratio(observed, candidate), observed

    window_span = max(2, len(candidate) + 2)
    best_score = 0.0
    best_window = observed[:window_span]

    for start in range(0, len(observed) - window_span + 1):
        window = observed[start : start + window_span]
        score = ratio(window, candidate)
        if score > best_score:
            best_score = score
            best_window = window

    return best_score, best_window


def candidate_score(observed: str, entry: LexiconEntry) -> tuple[float, dict[str, object]]:
    observed_skeleton = token_skeleton(observed)
    full_ratio = ratio(observed, entry.normalized)
    full_skeleton_ratio = ratio(observed_skeleton, entry.skeleton)
    window_ratio, best_window = best_window_ratio(observed, entry.normalized)
    window_skeleton_ratio = ratio(token_skeleton(best_window), entry.skeleton)

    prefix_bonus = 0.0
    if observed[:2] and entry.normalized.startswith(observed[:2]):
        prefix_bonus = 0.08
    elif observed[:1] and entry.normalized.startswith(observed[:1]):
        prefix_bonus = 0.03

    best_char_ratio = max(full_ratio, window_ratio)
    best_skeleton_ratio = max(full_skeleton_ratio, window_skeleton_ratio)
    score = (0.6 * best_char_ratio) + (0.3 * best_skeleton_ratio) + prefix_bonus
    details = {
        "full_ratio": round(full_ratio, 4),
        "full_skeleton_ratio": round(full_skeleton_ratio, 4),
        "window_ratio": round(window_ratio, 4),
        "window_skeleton_ratio": round(window_skeleton_ratio, 4),
        "best_window": best_window,
        "prefix_bonus": round(prefix_bonus, 4),
    }
    return round(min(score, 1.0), 4), details


def source_segments(entry: dict[str, object]) -> list[str]:
    segments = entry.get("segments")
    if isinstance(segments, list) and segments:
        return [str(segment) for segment in segments]
    return [str(entry.get("text", ""))]


def akshara_candidates_by_segment(candidates: list[dict[str, object]]) -> dict[tuple[int, int], list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
    for item in candidates:
        line_index = item.get("line_index")
        segment_index = item.get("segment_index")
        if not isinstance(line_index, int) or not isinstance(segment_index, int):
            continue
        grouped.setdefault((line_index, segment_index), []).append(item)
    return grouped


def observed_tokens_for_entry(
    chunk_index: int,
    entry_index: int,
    entry: dict[str, object],
    min_token_length: int,
    running_index: int,
    akshara_by_segment: dict[tuple[int, int], list[dict[str, object]]],
) -> tuple[list[dict[str, object]], int]:
    records: list[dict[str, object]] = []
    for segment_index, segment in enumerate(source_segments(entry), start=1):
        tokens = tokenize_segment(segment, min_token_length=min_token_length)
        for token_index, token in enumerate(tokens, start=1):
            records.append(
                {
                    "id": f"obs_{running_index:04}",
                    "chunk_index": chunk_index,
                    "entry_index": entry_index,
                    "segment_index": segment_index,
                    "token_index": token_index,
                    "token": token,
                    "skeleton": token_skeleton(token),
                    "line_index": entry.get("line_index"),
                    "line_image": entry.get("line_image"),
                    "source": "ocr_segment",
                }
            )
            running_index += 1

        line_index = entry.get("line_index")
        if not isinstance(line_index, int):
            continue
        for akshara_index, candidate in enumerate(akshara_by_segment.get((line_index, segment_index), []), start=1):
            token = normalize_token(str(candidate.get("best_token", "")))
            if len(token) < min_token_length:
                continue
            records.append(
                {
                    "id": f"obs_{running_index:04}",
                    "chunk_index": chunk_index,
                    "entry_index": entry_index,
                    "segment_index": segment_index,
                    "token_index": 1000 + akshara_index,
                    "token": token,
                    "skeleton": token_skeleton(token),
                    "line_index": line_index,
                    "line_image": entry.get("line_image"),
                    "source": "akshara_lattice",
                    "observed_span": candidate.get("observed_span", ""),
                    "observed_aksharas": candidate.get("observed_aksharas", []),
                    "akshara_best_score": candidate.get("best_score", 0.0),
                }
            )
            running_index += 1
    return records, running_index


def dedupe_observed_tokens(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for token in tokens:
        key = (
            token.get("chunk_index"),
            token.get("entry_index"),
            token.get("segment_index"),
            token.get("token"),
            token.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(token)
    return deduped


def observed_tokens(
    chunks: list[dict[str, object]],
    min_token_length: int,
    akshara_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    observed: list[dict[str, object]] = []
    running_index = 1
    akshara_by_segment = akshara_candidates_by_segment(akshara_candidates)

    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index", 0))
        entries = chunk.get("entries", [])
        if not isinstance(entries, list):
            continue

        for entry_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue

            entry_tokens, running_index = observed_tokens_for_entry(
                chunk_index=chunk_index,
                entry_index=entry_index,
                entry=entry,
                min_token_length=min_token_length,
                running_index=running_index,
                akshara_by_segment=akshara_by_segment,
            )
            observed.extend(entry_tokens)

    return dedupe_observed_tokens(observed)


def candidate_matches(
    token_text: str,
    lexicon: list[LexiconEntry],
    max_candidates: int,
    candidate_threshold: float,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for entry in lexicon:
        score, details = candidate_score(token_text, entry)
        if score < candidate_threshold:
            continue
        candidates.append(
            {
                "token": entry.token,
                "lemma": entry.lemma,
                "pos": entry.pos,
                "gloss": entry.gloss,
                "score": score,
                "details": details,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_candidates]


def extend_graph_with_candidates(
    token_id: str,
    candidates: list[dict[str, object]],
    lexicon_nodes: dict[str, dict[str, object]],
    edges: list[dict[str, object]],
) -> None:
    for candidate in candidates:
        candidate_id = f"lex::{candidate['token']}::{candidate['lemma']}"
        if candidate_id not in lexicon_nodes:
            lexicon_nodes[candidate_id] = {
                "id": candidate_id,
                "type": "lexicon_token",
                "token": candidate["token"],
                "lemma": candidate["lemma"],
                "pos": candidate["pos"],
                "gloss": candidate["gloss"],
            }

        edges.append(
            {
                "source": token_id,
                "target": candidate_id,
                "type": "candidate_match",
                "score": candidate["score"],
                "details": candidate["details"],
            }
        )


def build_graph(
    observed: list[dict[str, object]],
    lexicon: list[LexiconEntry],
    max_candidates: int,
    candidate_threshold: float,
    replace_threshold: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    lexicon_nodes: dict[str, dict[str, object]] = {}
    token_results: list[dict[str, object]] = []

    for index, token in enumerate(observed):
        token_id = token["id"]
        nodes.append({"id": token_id, "type": "observed_token", **token})

        candidates = candidate_matches(
            token_text=str(token["token"]),
            lexicon=lexicon,
            max_candidates=max_candidates,
            candidate_threshold=candidate_threshold,
        )

        best_candidate = candidates[0] if candidates else None
        chosen_token = str(token["token"])
        if best_candidate is not None and float(best_candidate["score"]) >= replace_threshold:
            chosen_token = str(best_candidate["token"])

        token_results.append(
            {
                **token,
                "best_candidate": best_candidate,
                "chosen_token": chosen_token,
                "candidates": candidates,
            }
        )

        extend_graph_with_candidates(token_id, candidates, lexicon_nodes, edges)

        if index > 0:
            previous = observed[index - 1]
            if previous["chunk_index"] == token["chunk_index"]:
                edges.append({"source": previous["id"], "target": token_id, "type": "sequence"})

    nodes.extend(lexicon_nodes.values())
    graph = {
        "metadata": {
            "observed_token_count": len(observed),
            "akshara_token_count": sum(1 for token in observed if token.get("source") == "akshara_lattice"),
            "lexicon_node_count": len(lexicon_nodes),
            "edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph, token_results


def chunk_hypotheses(token_results: list[dict[str, object]]) -> list[dict[str, object]]:
    chunk_map: dict[int, list[dict[str, object]]] = {}
    for token in token_results:
        chunk_map.setdefault(int(token["chunk_index"]), []).append(token)

    hypotheses: list[dict[str, object]] = []
    for chunk_index in sorted(chunk_map):
        tokens = chunk_map[chunk_index]
        hypotheses.append(
            {
                "chunk_index": chunk_index,
                "observed_text": " ".join(str(token["token"]) for token in tokens),
                "hypothesis_text": " ".join(str(token["chosen_token"]) for token in tokens),
                "matched_tokens": sum(1 for token in tokens if token.get("best_candidate") is not None),
                "high_confidence_replacements": sum(
                    1 for token in tokens if token.get("best_candidate") and token["chosen_token"] != token["token"]
                ),
            }
        )
    return hypotheses


def segment_tokens(token_results: list[dict[str, object]]) -> dict[tuple[int, int, int], list[dict[str, object]]]:
    grouped: dict[tuple[int, int, int], list[dict[str, object]]] = {}
    for token in token_results:
        key = (int(token["chunk_index"]), int(token["entry_index"]), int(token["segment_index"]))
        grouped.setdefault(key, []).append(token)
    return grouped


def dedupe_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_token(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def fallback_segment_text(segment_text: str) -> str:
    normalized = normalize_token(segment_text)
    return normalized if normalized else segment_text.strip()


def segment_reconstruction(segment_text: str, tokens: list[dict[str, object]]) -> dict[str, object]:
    akshara_supported = [
        token
        for token in tokens
        if token.get("source") == "akshara_lattice" and isinstance(token.get("best_candidate"), dict)
    ]
    if akshara_supported:
        anchors = dedupe_texts([str(token.get("chosen_token") or token.get("token") or "") for token in akshara_supported])
        return {
            "text": " ".join(anchors) if anchors else fallback_segment_text(segment_text),
            "strategy": "akshara_lattice",
            "token_count": len(anchors),
        }

    matched_ocr = [
        token
        for token in tokens
        if token.get("source") == "ocr_segment" and isinstance(token.get("best_candidate"), dict)
    ]
    if matched_ocr:
        anchors = dedupe_texts([str(token.get("chosen_token") or token.get("token") or "") for token in matched_ocr])
        return {
            "text": " ".join(anchors) if anchors else fallback_segment_text(segment_text),
            "strategy": "ocr_matched",
            "token_count": len(anchors),
        }

    observed = dedupe_texts([str(token.get("token") or "") for token in tokens if token.get("source") == "ocr_segment"])
    return {
        "text": " ".join(observed) if observed else fallback_segment_text(segment_text),
        "strategy": "ocr_fallback",
        "token_count": len(observed),
    }


def reconstructed_lines(
    chunks: list[dict[str, object]],
    token_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_segment = segment_tokens(token_results)
    reconstructed: list[dict[str, object]] = []

    for chunk in chunks:
        chunk_index = int(chunk.get("chunk_index", 0))
        entries = chunk.get("entries", [])
        if not isinstance(entries, list):
            continue

        for entry_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                continue
            segments = source_segments(entry)
            repaired_segments: list[dict[str, object]] = []
            for segment_index, segment_text in enumerate(segments, start=1):
                repair = segment_reconstruction(
                    segment_text,
                    by_segment.get((chunk_index, entry_index, segment_index), []),
                )
                repaired_segments.append(
                    {
                        "segment_index": segment_index,
                        "source_text": segment_text,
                        "reconstructed_text": repair["text"],
                        "strategy": repair["strategy"],
                        "token_count": repair["token_count"],
                    }
                )

            reconstructed.append(
                {
                    "chunk_index": chunk_index,
                    "entry_index": entry_index,
                    "line_index": entry.get("line_index"),
                    "line_image": entry.get("line_image"),
                    "source_text": str(entry.get("text", "")),
                    "reconstructed_text": " ".join(
                        segment["reconstructed_text"] for segment in repaired_segments if segment["reconstructed_text"]
                    ).strip(),
                    "segments": repaired_segments,
                }
            )

    return reconstructed


def unmatched_token_summary(token_results: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for token in token_results:
        if token["candidates"]:
            continue

        surface = str(token["token"])
        group = groups.setdefault(
            surface,
            {
                "token": surface,
                "count": 0,
                "chunk_indices": set(),
                "line_indices": set(),
            },
        )
        group["count"] = int(group["count"]) + 1
        group["chunk_indices"].add(int(token["chunk_index"]))
        if token.get("line_index") is not None:
            group["line_indices"].add(int(token["line_index"]))

    summary: list[dict[str, object]] = []
    for group in groups.values():
        summary.append(
            {
                "token": group["token"],
                "count": group["count"],
                "chunk_indices": sorted(group["chunk_indices"]),
                "line_indices": sorted(group["line_indices"]),
            }
        )

    summary.sort(key=lambda item: (-int(item["count"]), str(item["token"])))
    return summary


def write_outputs(
    output_dir: Path,
    chunks_path: Path,
    lexicon_path: Path,
    graph: dict[str, object],
    token_results: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    line_reconstructions: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_chunks": str(chunks_path),
        "lexicon": str(lexicon_path),
        "observed_tokens": len(token_results),
        "tokens_with_candidates": sum(1 for token in token_results if token["candidates"]),
        "unmatched_unique_tokens": len(unmatched_token_summary(token_results)),
        "high_confidence_replacements": sum(
            1 for token in token_results if token.get("best_candidate") and token["chosen_token"] != token["token"]
        ),
        "reconstructed_lines": len(line_reconstructions),
        "akshara_reconstructed_segments": sum(
            1
            for line in line_reconstructions
            for segment in line.get("segments", [])
            if isinstance(segment, dict) and segment.get("strategy") == "akshara_lattice"
        ),
    }
    unmatched = unmatched_token_summary(token_results)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "token_candidates.json").write_text(
        json.dumps(token_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "chunk_hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "reconstructed_lines.json").write_text(
        json.dumps(line_reconstructions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "unmatched_tokens.json").write_text(
        json.dumps(unmatched, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# Sanskrit Token Graph",
        "",
        f"Source chunks: {chunks_path}",
        f"Lexicon: {lexicon_path}",
        f"Observed tokens: {summary['observed_tokens']}",
        f"Tokens with candidates: {summary['tokens_with_candidates']}",
        f"Unmatched unique tokens: {summary['unmatched_unique_tokens']}",
        f"High-confidence replacements: {summary['high_confidence_replacements']}",
        f"Reconstructed lines: {summary['reconstructed_lines']}",
        f"Akshara-reconstructed segments: {summary['akshara_reconstructed_segments']}",
        "",
        "## Reconstructed Lines",
        "",
    ]

    for line in line_reconstructions[:20]:
        md_lines.append(f"### Line {line['line_index']}")
        md_lines.append("")
        md_lines.append(f"Source: {line['source_text']}")
        md_lines.append("")
        md_lines.append(f"Reconstructed: {line['reconstructed_text']}")
        md_lines.append("")

    md_lines.extend([
        "## Chunk Hypotheses",
        "",
    ])

    for hypothesis in hypotheses:
        md_lines.append(f"### Chunk {hypothesis['chunk_index']}")
        md_lines.append("")
        md_lines.append(f"Observed: {hypothesis['observed_text']}")
        md_lines.append("")
        md_lines.append(f"Hypothesis: {hypothesis['hypothesis_text']}")
        md_lines.append("")

    md_lines.append("## Top Candidate Samples")
    md_lines.append("")
    for token in token_results[:40]:
        best_candidate = token.get("best_candidate")
        if not best_candidate:
            continue
        md_lines.append(
            f"- {token['token']} -> {best_candidate['token']} ({best_candidate['lemma']}, score={best_candidate['score']})"
        )

    md_lines.append("")
    md_lines.append("## Unmatched Tokens")
    md_lines.append("")
    for item in unmatched[:25]:
        md_lines.append(
            f"- {item['token']} | count={item['count']} | chunks={item['chunk_indices']} | lines={item['line_indices']}"
        )

    (output_dir / "graph_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    chunks_path = Path(args.chunks_json)
    lexicon_path = Path(args.lexicon)
    output_dir = Path(args.output_dir)

    try:
        chunks = load_chunks(chunks_path)
        lexicon = load_lexicon(lexicon_path)
        akshara_candidates = load_akshara_candidates(Path(args.akshara_lattice)) if args.akshara_lattice else []
        observed = observed_tokens(
            chunks,
            min_token_length=args.min_token_length,
            akshara_candidates=akshara_candidates,
        )
        graph, token_results = build_graph(
            observed,
            lexicon,
            max_candidates=args.max_candidates,
            candidate_threshold=args.candidate_threshold,
            replace_threshold=args.replace_threshold,
        )
        hypotheses = chunk_hypotheses(token_results)
        line_reconstructions = reconstructed_lines(chunks, token_results)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    write_outputs(output_dir, chunks_path, lexicon_path, graph, token_results, hypotheses, line_reconstructions)
    print(f"Built Sanskrit token graph with {len(token_results)} observed tokens in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
