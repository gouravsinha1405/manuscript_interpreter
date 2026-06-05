from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from indic_transliteration import sanscript

from build_sanskrit_token_graph import LexiconEntry, candidate_score, load_lexicon, normalize_token


@dataclass(frozen=True)
class SpanCandidate:
    start: int
    end: int
    observed: str
    entry_token: str
    lemma: str
    pos: str
    gloss: str
    score: float


@dataclass(frozen=True)
class SplitStep:
    kind: str
    surface: str
    normalized: str
    score: float
    lemma: str = ""
    pos: str = ""
    gloss: str = ""


@dataclass(frozen=True)
class SplitPath:
    score: float
    steps: tuple[SplitStep, ...]
    lexical_steps: int
    skipped_chars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a noisy Sanskrit token using standard sandhi splitting plus a dynamic-programming lexicon fallback."
    )
    parser.add_argument("token", help="Devanagari token to analyze")
    parser.add_argument(
        "--lexicon",
        default="resources/modern_sanskrit_seed_lexicon.tsv",
        help="Lexicon TSV with columns: token, lemma, pos, gloss",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.72,
        help="Minimum approximate-match score for a lexicon segment candidate",
    )
    parser.add_argument(
        "--skip-penalty",
        type=float,
        default=0.55,
        help="Penalty applied per skipped noisy character",
    )
    parser.add_argument(
        "--split-penalty",
        type=float,
        default=0.08,
        help="Small penalty applied per accepted lexical segment to avoid over-splitting",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Maximum number of split candidates to return",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save the structured result as JSON",
    )
    return parser.parse_args()


def transliteration_views(token: str) -> dict[str, str]:
    return {
        "devanagari": token,
        "slp1": sanscript.transliterate(token, sanscript.DEVANAGARI, sanscript.SLP1),
        "iast": sanscript.transliterate(token, sanscript.DEVANAGARI, sanscript.IAST),
    }


def run_sandhi(token: str) -> dict[str, object]:
    completed = subprocess.run(
        ["sanskrit_parser", "sandhi", token],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "combined_output": combined,
        "has_split": "Split:" in combined and "No Splits Found" not in combined,
    }


def span_candidates(token: str, lexicon: list[LexiconEntry], score_threshold: float) -> dict[int, list[SpanCandidate]]:
    by_start: dict[int, list[SpanCandidate]] = {}
    max_entry_len = max(len(entry.normalized) for entry in lexicon)

    for start in range(len(token)):
        options: list[SpanCandidate] = []
        max_end = min(len(token), start + max_entry_len + 2)
        for end in range(start + 2, max_end + 1):
            observed = token[start:end]
            for entry in lexicon:
                score, _ = candidate_score(observed, entry)
                if score < score_threshold:
                    continue
                options.append(
                    SpanCandidate(
                        start=start,
                        end=end,
                        observed=observed,
                        entry_token=entry.token,
                        lemma=entry.lemma,
                        pos=entry.pos,
                        gloss=entry.gloss,
                        score=score,
                    )
                )

        options.sort(
            key=lambda item: (
                item.score * len(item.observed),
                -abs(len(item.entry_token) - len(item.observed)),
                item.score,
            ),
            reverse=True,
        )
        by_start[start] = options[:12]
    return by_start


def merge_noise(step: SplitStep, steps: tuple[SplitStep, ...]) -> tuple[SplitStep, ...]:
    if steps and steps[0].kind == "noise":
        merged = SplitStep(
            kind="noise",
            surface=step.surface + steps[0].surface,
            normalized=step.normalized + steps[0].normalized,
            score=step.score + steps[0].score,
        )
        return (merged,) + steps[1:]
    return (step,) + steps


def top_split_paths(
    token: str,
    candidates_by_start: dict[int, list[SpanCandidate]],
    top_k: int,
    skip_penalty: float,
    split_penalty: float,
) -> list[SplitPath]:
    lexical_bonus_scale = 1.0

    @lru_cache(maxsize=None)
    def solve(index: int) -> tuple[SplitPath, ...]:
        if index >= len(token):
            return (SplitPath(score=0.0, steps=(), lexical_steps=0, skipped_chars=0),)

        results: list[SplitPath] = []

        noise_char = token[index]
        noise_step = SplitStep(kind="noise", surface=noise_char, normalized=noise_char, score=-skip_penalty)
        for tail in solve(index + 1):
            results.append(
                SplitPath(
                    score=tail.score - skip_penalty,
                    steps=merge_noise(noise_step, tail.steps),
                    lexical_steps=tail.lexical_steps,
                    skipped_chars=tail.skipped_chars + 1,
                )
            )

        for option in candidates_by_start.get(index, []):
            length_gap_penalty = 0.25 * abs(len(option.entry_token) - len(option.observed))
            weighted_score = (option.score * len(option.observed) * lexical_bonus_scale) - length_gap_penalty - split_penalty
            lexical_step = SplitStep(
                kind="lexical",
                surface=option.observed,
                normalized=option.entry_token,
                score=round(option.score, 4),
                lemma=option.lemma,
                pos=option.pos,
                gloss=option.gloss,
            )
            for tail in solve(option.end):
                results.append(
                    SplitPath(
                        score=tail.score + weighted_score,
                        steps=(lexical_step,) + tail.steps,
                        lexical_steps=tail.lexical_steps + 1,
                        skipped_chars=tail.skipped_chars,
                    )
                )

        deduped: dict[tuple[tuple[str, str], ...], SplitPath] = {}
        for result in results:
            signature = tuple((step.kind, step.normalized) for step in result.steps)
            current = deduped.get(signature)
            if current is None or result.score > current.score:
                deduped[signature] = result

        ranked = sorted(
            deduped.values(),
            key=lambda item: (item.score, item.lexical_steps, -item.skipped_chars),
            reverse=True,
        )
        return tuple(ranked[: max(top_k * 6, 20)])

    return [path for path in solve(0) if path.lexical_steps > 0][:top_k]


def path_record(path: SplitPath) -> dict[str, object]:
    lexical_tokens = [step.normalized for step in path.steps if step.kind == "lexical"]
    glosses = [step.gloss for step in path.steps if step.kind == "lexical" and step.gloss]
    return {
        "score": round(path.score, 4),
        "lexical_steps": path.lexical_steps,
        "skipped_chars": path.skipped_chars,
        "split": " + ".join(lexical_tokens),
        "interpretation": " + ".join(glosses),
        "steps": [
            {
                "kind": step.kind,
                "surface": step.surface,
                "normalized": step.normalized,
                "score": round(step.score, 4),
                "lemma": step.lemma,
                "pos": step.pos,
                "gloss": step.gloss,
            }
            for step in path.steps
        ],
    }


def report_lines(result: dict[str, object]) -> list[str]:
    lines = [
        f"Token: {result['token']}",
        f"Normalized: {result['normalized']}",
        f"SLP1: {result['transliterations']['slp1']}",
        f"IAST: {result['transliterations']['iast']}",
        "",
        "Sandhi parser:",
        str(result["sandhi"]["combined_output"]).strip() or "<no output>",
        "",
        "Dynamic-programming split candidates:",
    ]
    for index, candidate in enumerate(result["dp_candidates"], start=1):
        lines.append("")
        lines.append(f"{index}. {candidate['split']}")
        lines.append(
            f"   score={candidate['score']} | lexical_steps={candidate['lexical_steps']} | skipped_chars={candidate['skipped_chars']}"
        )
        if candidate["interpretation"]:
            lines.append(f"   meaning={candidate['interpretation']}")
        for step in candidate["steps"]:
            if step["kind"] == "noise":
                lines.append(f"   noise: {step['surface']}")
            else:
                lines.append(
                    f"   {step['surface']} -> {step['normalized']} ({step['pos']}; {step['gloss']}; score={step['score']})"
                )
    return lines


def main() -> int:
    args = parse_args()
    lexicon = load_lexicon(Path(args.lexicon))
    normalized = normalize_token(args.token)
    sandhi = run_sandhi(normalized or args.token)
    candidates_by_start = span_candidates(normalized, lexicon, args.score_threshold)
    dp_paths = top_split_paths(
        normalized,
        candidates_by_start,
        top_k=max(1, args.top_k),
        skip_penalty=args.skip_penalty,
        split_penalty=args.split_penalty,
    )

    result = {
        "token": args.token,
        "normalized": normalized,
        "transliterations": transliteration_views(normalized or args.token),
        "sandhi": sandhi,
        "dp_candidates": [path_record(path) for path in dp_paths],
    }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n".join(report_lines(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())