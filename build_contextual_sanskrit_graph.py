from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from indic_transliteration import sanscript


PARSE_COST_RE = re.compile(r"Parse 0 : \(Cost = ([0-9.]+)\)")
SPLIT_LINE_RE = re.compile(r"Split:\s*(\[.*\])")
MORPH_TAG_RE = re.compile(r"^\([^\n]+\)$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment a Sanskrit token graph with parser split candidates and local contextual scores."
    )
    parser.add_argument(
        "--base-graph",
        default="phase2_input/page_002_token_graph_akshara/graph.json",
        help="Path to the base token graph JSON",
    )
    parser.add_argument(
        "--token-candidates",
        default="phase2_input/page_002_token_graph_akshara/token_candidates.json",
        help="Path to token candidate results from the base graph builder",
    )
    parser.add_argument(
        "--parser-probe",
        default="phase2_input/page_002_parser_probe/probe_results.json",
        help="Path to parser probe results for unmatched or noisy tokens",
    )
    parser.add_argument(
        "--output-dir",
        default="phase2_input/page_002_contextual_graph_akshara_quick",
        help="Directory where contextual graph artifacts will be written",
    )
    parser.add_argument(
        "--max-parser-candidates",
        type=int,
        default=1,
        help="Maximum parser split candidates to keep per observed token",
    )
    parser.add_argument(
        "--vakya-timeout-seconds",
        type=int,
        default=2,
        help="Timeout for each vakya scoring subprocess",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel vakya scoring subprocesses",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=0,
        help="Number of neighboring observed tokens to include on each side of the context window",
    )
    parser.add_argument(
        "--max-window-chars",
        type=int,
        default=32,
        help="Maximum total characters in a vakya context window before scoring is skipped",
    )
    parser.add_argument(
        "--local-weight",
        type=float,
        default=0.4,
        help="Weight assigned to token-local candidate quality in the combined score",
    )
    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.6,
        help="Weight assigned to vakya context compatibility in the combined score",
    )
    return parser.parse_args()


def load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return str(value)


def combined_output(result: dict[str, object]) -> str:
    return "\n".join([
        normalize_output(result.get("stdout", "")),
        normalize_output(result.get("stderr", "")),
    ])


def parse_split_candidates(result: dict[str, object], max_candidates: int) -> list[list[str]]:
    splits: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for line in combined_output(result).splitlines():
        match = SPLIT_LINE_RE.search(line)
        if not match:
            continue
        payload = match.group(1)
        if payload == "No Splits Found":
            continue
        try:
            split = ast.literal_eval(payload)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(split, list) or not all(isinstance(part, str) for part in split):
            continue
        split_key = tuple(split)
        if split_key in seen:
            continue
        seen.add(split_key)
        splits.append(split)
        if len(splits) >= max_candidates:
            break
    return splits


def split_parts_to_devanagari(parts: list[str]) -> list[str]:
    converted: list[str] = []
    for part in parts:
        if re.search(r"[\u0900-\u097F]", part):
            converted.append(part)
            continue
        converted.append(sanscript.transliterate(part, sanscript.SLP1, sanscript.DEVANAGARI))
    return converted


def parse_cost(output: str) -> float | None:
    match = PARSE_COST_RE.search(output)
    if not match:
        return None
    return float(match.group(1))


def should_score_context(tokens: list[str], max_window_chars: int) -> bool:
    return len(" ".join(tokens)) <= max_window_chars


def vakya_score(tokens: list[str], timeout_seconds: int) -> tuple[float, dict[str, object]]:
    try:
        completed = subprocess.run(
            ["sanskrit_parser", "vakya", "--pre-segmented", "--score", "--min-cost", " ".join(tokens)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return 0.0, {
            "returncode": None,
            "stdout": normalize_output(exc.stdout),
            "stderr": normalize_output(exc.stderr),
            "cost": None,
            "timed_out": True,
        }

    output = "\n".join([completed.stdout, completed.stderr])
    cost = parse_cost(output)
    score = 0.0 if cost is None else 1.0 / (1.0 + max(cost, 0.0))
    return score, {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "cost": cost,
        "timed_out": False,
    }


def token_surface(token_result: dict[str, object]) -> str:
    best_candidate = token_result.get("best_candidate")
    if isinstance(best_candidate, dict) and best_candidate.get("token"):
        return str(best_candidate["token"])
    return str(token_result.get("chosen_token") or token_result.get("token") or "")


def parser_probe_index(parser_probe_results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(result.get("token", "")): result
        for result in parser_probe_results
        if isinstance(result, dict) and str(result.get("token", "")).strip()
    }


def local_candidate_score(token_result: dict[str, object], split_rank: int) -> float:
    best_candidate = token_result.get("best_candidate")
    base_score = 0.0
    if isinstance(best_candidate, dict):
        try:
            base_score = float(best_candidate.get("score", 0.0))
        except (TypeError, ValueError):
            base_score = 0.0
    rank_bonus = 1.0 / float(split_rank + 1)
    return max(base_score, rank_bonus)


def context_window_tokens(
    token_results: list[dict[str, object]],
    index: int,
    candidate_parts: list[str],
    radius: int,
) -> list[str]:
    tokens: list[str] = []
    left_start = max(0, index - radius)
    for neighbor in token_results[left_start:index]:
        surface = token_surface(neighbor)
        if surface:
            tokens.append(surface)

    tokens.extend(candidate_parts)

    right_end = min(len(token_results), index + radius + 1)
    for neighbor in token_results[index + 1 : right_end]:
        surface = token_surface(neighbor)
        if surface:
            tokens.append(surface)
    return tokens


def parser_candidates_for_token(
    token_result: dict[str, object],
    probe_result: dict[str, object] | None,
    max_candidates: int,
    morphology_cache: dict[str, float],
    tags_timeout_seconds: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    best_candidate = token_result.get("best_candidate")
    if isinstance(best_candidate, dict):
        candidates.append(
            {
                "source": "lexicon",
                "surface": str(best_candidate.get("token", token_surface(token_result))),
                "parts": [str(best_candidate.get("token", token_surface(token_result)))],
                "local_score": float(best_candidate.get("score", 0.0)),
                "rank": 0,
            }
        )
    else:
        surface = token_surface(token_result)
        candidates.append(
            {
                "source": "observed",
                "surface": surface,
                "parts": [surface] if surface else [],
                "local_score": 0.0,
                "rank": 0,
            }
        )

    if probe_result is None:
        return candidates

    splits = parse_split_candidates(dict(probe_result.get("sandhi", {})), max_candidates=max_candidates)
    for rank, split in enumerate(splits, start=1):
        devanagari_parts = split_parts_to_devanagari(split)
        candidates.append(
            {
                "source": "parser_split",
                "surface": " ".join(devanagari_parts),
                "parts": devanagari_parts,
                "local_score": local_candidate_score(token_result, split_rank=rank),
                "rank": rank,
            }
        )

    candidates.extend(
        dp_probe_candidates(
            probe_result,
            max_candidates=max_candidates,
            morphology_cache=morphology_cache,
            tags_timeout_seconds=tags_timeout_seconds,
        )
    )
    return candidates


def dp_probe_candidates(
    probe_result: dict[str, object],
    max_candidates: int,
    morphology_cache: dict[str, float],
    tags_timeout_seconds: int,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for rank, split_candidate in enumerate(list(probe_result.get("dp_candidates", []))[:max_candidates], start=1):
        candidate = dp_probe_candidate(
            split_candidate,
            rank=rank,
            morphology_cache=morphology_cache,
            tags_timeout_seconds=tags_timeout_seconds,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def dp_probe_candidate(
    split_candidate: object,
    rank: int,
    morphology_cache: dict[str, float],
    tags_timeout_seconds: int,
) -> dict[str, object] | None:
    if not isinstance(split_candidate, dict):
        return None

    parts = lexical_parts(split_candidate.get("steps", []))
    if not parts:
        return None

    morphology_score = split_morphology_score(
        parts,
        cache=morphology_cache,
        timeout_seconds=tags_timeout_seconds,
    )
    if morphology_score < 1.0:
        return None

    step_scores = lexical_step_scores(split_candidate.get("steps", []))
    average_step_score = sum(step_scores) / len(step_scores) if step_scores else 0.0
    skipped_chars = safe_int(split_candidate.get("skipped_chars", 0))
    split_penalty = 0.06 * max(len(parts) - 1, 0)
    noise_penalty = 0.08 * skipped_chars
    normalized_local = max(0.0, min(average_step_score - split_penalty - noise_penalty, 1.0))
    return {
        "source": "dp_split",
        "surface": " ".join(parts),
        "parts": parts,
        "local_score": normalized_local,
        "morphology_score": morphology_score,
        "rank": rank,
    }


def lexical_parts(steps: object) -> list[str]:
    if not isinstance(steps, list):
        return []
    return [
        str(step.get("normalized", ""))
        for step in steps
        if isinstance(step, dict) and step.get("kind") == "lexical" and str(step.get("normalized", "")).strip()
    ]


def lexical_step_scores(steps: object) -> list[float]:
    if not isinstance(steps, list):
        return []
    scores: list[float] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("kind") != "lexical":
            continue
        scores.append(safe_float(step.get("score", 0.0)))
    return scores


def split_morphology_score(parts: list[str], cache: dict[str, float], timeout_seconds: int) -> float:
    if not parts:
        return 0.0
    part_scores = [part_morphology_score(part, cache=cache, timeout_seconds=timeout_seconds) for part in parts]
    return sum(part_scores) / len(part_scores)


def part_morphology_score(part: str, cache: dict[str, float], timeout_seconds: int) -> float:
    normalized_part = part.strip()
    if not normalized_part:
        return 0.0
    cached_score = cache.get(normalized_part)
    if cached_score is not None:
        return cached_score

    try:
        completed = subprocess.run(
            ["sanskrit_parser", "tags", normalized_part],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        cache[normalized_part] = 0.0
        return 0.0

    output = "\n".join([completed.stdout, completed.stderr])
    score = 1.0 if has_substantive_morphology(output) else 0.0
    cache[normalized_part] = score
    return score


def has_substantive_morphology(output: str) -> bool:
    if "Morphological tags:" not in output:
        return False
    return bool(MORPH_TAG_RE.search(output))


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def augment_graph(
    base_graph: dict[str, object],
    token_results: list[dict[str, object]],
    parser_probe_results: list[dict[str, object]],
    max_parser_candidates: int,
    vakya_timeout_seconds: int,
    window_radius: int,
    max_window_chars: int,
    local_weight: float,
    context_weight: float,
    max_workers: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    nodes = list(base_graph.get("nodes", []))
    edges = list(base_graph.get("edges", []))
    probe_index = parser_probe_index(parser_probe_results)
    contextual_candidates: list[dict[str, object]] = []
    scored_windows: dict[tuple[str, ...], tuple[float, dict[str, object]]] = {}
    morphology_cache: dict[str, float] = {}

    pending_windows: set[tuple[str, ...]] = set()
    for index, token_result in enumerate(token_results):
        probe_result = probe_index.get(str(token_result.get("token", "")))
        candidates = parser_candidates_for_token(
            token_result,
            probe_result,
            max_candidates=max_parser_candidates,
            morphology_cache=morphology_cache,
            tags_timeout_seconds=vakya_timeout_seconds,
        )
        for candidate in candidates:
            context_tokens = context_window_tokens(token_results, index=index, candidate_parts=list(candidate["parts"]), radius=window_radius)
            if not should_score_context(context_tokens, max_window_chars=max_window_chars):
                continue
            pending_windows.add(tuple(context_tokens))

    if pending_windows:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            future_map = {
                executor.submit(vakya_score, list(window_tokens), timeout_seconds=vakya_timeout_seconds): window_tokens
                for window_tokens in pending_windows
            }
            for future, window_tokens in future_map.items():
                scored_windows[window_tokens] = future.result()

    for index, token_result in enumerate(token_results):
        token_id = str(token_result.get("id", ""))
        probe_result = probe_index.get(str(token_result.get("token", "")))
        candidates = parser_candidates_for_token(
            token_result,
            probe_result,
            max_candidates=max_parser_candidates,
            morphology_cache=morphology_cache,
            tags_timeout_seconds=vakya_timeout_seconds,
        )
        for candidate_index, candidate in enumerate(candidates, start=1):
            context_tokens = context_window_tokens(token_results, index=index, candidate_parts=list(candidate["parts"]), radius=window_radius)
            if should_score_context(context_tokens, max_window_chars=max_window_chars):
                context_score, vakya_result = scored_windows[tuple(context_tokens)]
            else:
                context_score = 0.0
                vakya_result = {
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "cost": None,
                    "timed_out": False,
                    "skipped": True,
                }
            combined_score = (local_weight * float(candidate["local_score"])) + (context_weight * context_score)
            candidate_id = f"cand::{token_id}::{candidate['source']}::{candidate_index}"
            candidate_record = {
                "id": candidate_id,
                "type": "context_candidate",
                "observed_token_id": token_id,
                "observed_token": str(token_result.get("token", "")),
                "source": candidate["source"],
                "surface": candidate["surface"],
                "parts": candidate["parts"],
                "local_score": round(float(candidate["local_score"]), 4),
                "context_score": round(context_score, 4),
                "combined_score": round(combined_score, 4),
                "context_window": context_tokens,
                "vakya": vakya_result,
            }
            nodes.append(candidate_record)
            edges.append(
                {
                    "source": token_id,
                    "target": candidate_id,
                    "type": "contextual_candidate",
                    "local_score": candidate_record["local_score"],
                    "context_score": candidate_record["context_score"],
                    "combined_score": candidate_record["combined_score"],
                }
            )
            contextual_candidates.append(candidate_record)

    graph = {
        "metadata": {
            **dict(base_graph.get("metadata", {})),
            "contextual_candidate_count": len(contextual_candidates),
            "augmented_edge_count": len(edges),
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph, contextual_candidates


def best_candidates_by_token(contextual_candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for candidate in contextual_candidates:
        grouped.setdefault(str(candidate["observed_token_id"]), []).append(candidate)

    best: list[dict[str, object]] = []
    for token_id, candidates in grouped.items():
        top = max(candidates, key=lambda item: (float(item["combined_score"]), float(item["context_score"])))
        best.append(
            {
                "observed_token_id": token_id,
                "observed_token": top["observed_token"],
                "best_surface": top["surface"],
                "source": top["source"],
                "local_score": top["local_score"],
                "context_score": top["context_score"],
                "combined_score": top["combined_score"],
            }
        )
    best.sort(key=lambda item: item["observed_token_id"])
    return best


def write_outputs(output_dir: Path, graph: dict[str, object], contextual_candidates: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = best_candidates_by_token(contextual_candidates)
    summary = {
        "contextual_candidate_count": len(contextual_candidates),
        "tokens_with_parser_candidates": sum(1 for item in best if item["source"] in {"parser_split", "dp_split"}),
        "tokens_with_positive_context": sum(1 for item in contextual_candidates if float(item["context_score"]) > 0.0),
        "best_candidate_count": len(best),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "augmented_graph.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "contextual_candidates.json").write_text(
        json.dumps(contextual_candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "best_contextual_candidates.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md_lines = [
        "# Contextual Sanskrit Graph",
        "",
        f"Contextual candidates: {summary['contextual_candidate_count']}",
        f"Best candidate count: {summary['best_candidate_count']}",
        f"Tokens with parser-selected candidates: {summary['tokens_with_parser_candidates']}",
        f"Candidates with positive context score: {summary['tokens_with_positive_context']}",
        "",
        "## Best Candidates",
        "",
    ]
    for item in best[:80]:
        md_lines.append(
            f"- {item['observed_token']} -> {item['best_surface']} | source={item['source']} | combined={item['combined_score']} | context={item['context_score']}"
        )
    (output_dir / "context_report.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        base_graph = load_json(Path(args.base_graph))
        token_results = load_json(Path(args.token_candidates))
        parser_probe_results = load_json(Path(args.parser_probe))
        if not isinstance(base_graph, dict):
            raise ValueError("Base graph JSON must contain an object.")
        if not isinstance(token_results, list):
            raise ValueError("Token candidate JSON must contain a list.")
        if not isinstance(parser_probe_results, list):
            raise ValueError("Parser probe JSON must contain a list.")
        graph, contextual_candidates = augment_graph(
            base_graph=base_graph,
            token_results=token_results,
            parser_probe_results=parser_probe_results,
            max_parser_candidates=args.max_parser_candidates,
            vakya_timeout_seconds=args.vakya_timeout_seconds,
            window_radius=args.window_radius,
            max_window_chars=args.max_window_chars,
            local_weight=args.local_weight,
            context_weight=args.context_weight,
            max_workers=args.max_workers,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    write_outputs(Path(args.output_dir), graph, contextual_candidates)
    print(f"Built contextual graph with {len(contextual_candidates)} candidates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())