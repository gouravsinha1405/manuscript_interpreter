from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request as urllib_request

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for


BASE_DIR = Path(__file__).resolve().parent
PAGE_IMAGE_DIR = BASE_DIR / "page_images"
ANNOTATION_DIR = BASE_DIR / "annotations"
ANNOTATION_FILE = ANNOTATION_DIR / "page_002_verdicts.json"
INTERPRETATION_FILE = ANNOTATION_DIR / "page_002_interpretations.json"
ENV_FILE = BASE_DIR / ".env"

PAGE_NUMBER = 2
RAW_CHUNKS_PATH = BASE_DIR / "phase2_input" / "page_002_from_full_lines" / "chunks.json"
TOKEN_GRAPH_DIR_CANDIDATES = [
    BASE_DIR / "phase2_input" / "page_002_token_graph_akshara",
]
CONTEXTUAL_DIR_CANDIDATES = [
    BASE_DIR / "phase2_input" / "page_002_contextual_graph_akshara",
    BASE_DIR / "phase2_input" / "page_002_contextual_graph_akshara_balanced",
    BASE_DIR / "phase2_input" / "page_002_contextual_graph_akshara_r1",
    BASE_DIR / "phase2_input" / "page_002_contextual_graph_akshara_quick",
]
DEFAULT_GROQ_LINE_MODEL = "qwen/qwen3-32b"
DEFAULT_GROQ_PAGE_MODEL = "openai/gpt-oss-120b"
PAGE_CHUNK_SIZE = 4

TOKEN_RE = re.compile(r"[\u0900-\u097F]+")

app = Flask(__name__)


@dataclass
class TokenSuggestion:
    token_id: str
    observed_token: str
    best_surface: str
    source: str
    observed_source: str
    segment_index: int | None
    local_score: float
    context_score: float
    combined_score: float
    candidate_gloss: str | None

    @property
    def changed(self) -> bool:
        return self.observed_token != self.best_surface

    @property
    def confidence_label(self) -> str:
        if self.combined_score >= 0.5:
            return "high"
        if self.combined_score >= 0.25:
            return "medium"
        return "low"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> object | None:
    if not path.is_file():
        return None
    return load_json(path)


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_value(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return str(os.environ[name])
    return load_dotenv(ENV_FILE).get(name, default)


def groq_config() -> dict[str, str | bool]:
    api_key = env_value("GROQ_API_KEY")
    line_model = env_value("GROQ_LINE_MODEL") or env_value("GROQ_MODEL", DEFAULT_GROQ_LINE_MODEL)
    page_model = env_value("GROQ_PAGE_MODEL", DEFAULT_GROQ_PAGE_MODEL)
    return {
        "configured": bool(api_key),
        "api_key": api_key,
        "model": line_model,
        "line_model": line_model,
        "page_model": page_model,
    }


def extract_json_payload(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise ValueError("Groq response did not contain a JSON object.")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Groq response JSON must be an object.")
    return parsed


def page_image_path(page_number: int) -> Path:
    return PAGE_IMAGE_DIR / f"page_{page_number:03d}.png"


def score_token_graph_dir(path: Path) -> tuple[int, int, int]:
    summary = load_json_if_exists(path / "summary.json")
    if not isinstance(summary, dict):
        return (-1, -1, -1)
    return (
        int(summary.get("tokens_with_candidates", 0)),
        int(summary.get("akshara_reconstructed_segments", 0)),
        int(summary.get("observed_tokens", 0)),
    )


def score_contextual_dir(path: Path) -> tuple[int, int, int]:
    summary = load_json_if_exists(path / "summary.json")
    if not isinstance(summary, dict):
        return (-1, -1, -1)
    return (
        int(summary.get("tokens_with_positive_context", 0)),
        int(summary.get("best_candidate_count", 0)),
        int(summary.get("contextual_candidate_count", 0)),
    )


def choose_best_dir(candidates: list[Path], scorer) -> Path:
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        raise FileNotFoundError("No review artifact directories were found.")
    return max(existing, key=scorer)


def selected_token_graph_dir() -> Path:
    return choose_best_dir(TOKEN_GRAPH_DIR_CANDIDATES, score_token_graph_dir)


def selected_contextual_dir() -> Path:
    return choose_best_dir(CONTEXTUAL_DIR_CANDIDATES, score_contextual_dir)


def best_contextual_index(best_contextual: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(item.get("observed_token_id", "")): item
        for item in best_contextual
        if isinstance(item, dict) and item.get("observed_token_id")
    }


def optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def token_candidate_gloss(token: dict[str, object]) -> str | None:
    best_candidate = token.get("best_candidate")
    if not isinstance(best_candidate, dict):
        return None
    gloss = best_candidate.get("gloss")
    return str(gloss) if gloss else None


def annotation_store() -> dict[str, object]:
    if not ANNOTATION_FILE.is_file():
        return {"page_number": PAGE_NUMBER, "annotations": {}}
    return json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))


def interpretation_store() -> dict[str, object]:
    if not INTERPRETATION_FILE.is_file():
        return {"page_number": PAGE_NUMBER, "interpretations": {}}
    return json.loads(INTERPRETATION_FILE.read_text(encoding="utf-8"))


def save_annotation_store(store: dict[str, object]) -> None:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def save_interpretation_store(store: dict[str, object]) -> None:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    INTERPRETATION_FILE.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def token_display(raw_token: str, suggestion: TokenSuggestion | None) -> str:
    if suggestion is None:
        return raw_token
    if suggestion.combined_score < 0.25:
        return raw_token
    return suggestion.best_surface


def build_line_draft(entry_text: str, suggestions: list[TokenSuggestion]) -> str:
    queue: dict[str, list[TokenSuggestion]] = defaultdict(list)
    for suggestion in suggestions:
        queue[suggestion.observed_token].append(suggestion)

    def replace(match: re.Match[str]) -> str:
        raw_token = match.group(0)
        token_suggestions = queue.get(raw_token)
        if not token_suggestions:
            return raw_token
        suggestion = token_suggestions.pop(0)
        return token_display(raw_token, suggestion)

    return TOKEN_RE.sub(replace, entry_text)


def load_review_inputs() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str],
]:
    raw_chunks = load_json(RAW_CHUNKS_PATH)
    token_graph_dir = selected_token_graph_dir()
    contextual_dir = selected_contextual_dir()
    token_candidates = load_json(token_graph_dir / "token_candidates.json")
    reconstructed_lines = load_json(token_graph_dir / "reconstructed_lines.json")
    best_contextual = load_json(contextual_dir / "best_contextual_candidates.json")
    if (
        not isinstance(raw_chunks, list)
        or not isinstance(token_candidates, list)
        or not isinstance(reconstructed_lines, list)
        or not isinstance(best_contextual, list)
    ):
        raise ValueError("Unexpected annotation input JSON format.")
    return raw_chunks, token_candidates, best_contextual, reconstructed_lines, {
        "token_graph_dir": token_graph_dir.name,
        "contextual_dir": contextual_dir.name,
    }


def suggestion_from_token(
    token: dict[str, object],
    best_index: dict[str, dict[str, object]],
) -> tuple[int | None, TokenSuggestion | None]:
    token_id = str(token.get("id", ""))
    line_index = optional_int(token.get("line_index"))
    if not token_id or line_index is None:
        return None, None

    best = best_index.get(token_id)
    candidate_gloss = token_candidate_gloss(token)
    segment_index = optional_int(token.get("segment_index"))
    suggestion = TokenSuggestion(
        token_id=token_id,
        observed_token=str(token.get("token", "")),
        best_surface=str((best or {}).get("best_surface") or token.get("token", "")),
        source=str((best or {}).get("source") or "observed"),
        observed_source=str(token.get("source") or "ocr_segment"),
        segment_index=segment_index,
        local_score=float((best or {}).get("local_score") or 0.0),
        context_score=float((best or {}).get("context_score") or 0.0),
        combined_score=float((best or {}).get("combined_score") or 0.0),
        candidate_gloss=candidate_gloss,
    )
    return line_index, suggestion


def build_suggestions_by_line(
    token_candidates: list[dict[str, object]],
    best_contextual: list[dict[str, object]],
) -> dict[int, list[TokenSuggestion]]:
    best_index = best_contextual_index(best_contextual)
    suggestions_by_line: dict[int, list[TokenSuggestion]] = defaultdict(list)
    for token in token_candidates:
        if not isinstance(token, dict):
            continue
        line_index, suggestion = suggestion_from_token(token, best_index)
        if line_index is None or suggestion is None:
            continue
        suggestions_by_line[line_index].append(suggestion)
    return suggestions_by_line


def reconstructed_line_index(reconstructed_lines: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    index: dict[int, dict[str, object]] = {}
    for item in reconstructed_lines:
        if not isinstance(item, dict):
            continue
        line_index = item.get("line_index")
        if isinstance(line_index, int):
            index[line_index] = item
    return index


def suggestion_priority(item: TokenSuggestion) -> tuple[float, float, float, int, int]:
    candidate_rank = {"lexicon": 2, "parser_split": 1, "observed": 0}
    evidence_rank = {"akshara_lattice": 2, "ocr_segment": 1}
    return (
        item.combined_score,
        item.context_score,
        item.local_score,
        candidate_rank.get(item.source, 0),
        evidence_rank.get(item.observed_source, 0),
    )


def segment_map(reconstructed_line: dict[str, object] | None) -> dict[int, dict[str, object]]:
    if not isinstance(reconstructed_line, dict):
        return {}
    segments = reconstructed_line.get("segments")
    if not isinstance(segments, list):
        return {}
    mapped: dict[int, dict[str, object]] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_index = segment.get("segment_index")
        if isinstance(segment_index, int):
            mapped[segment_index] = segment
    return mapped


def build_line_views(
    entry: dict[str, object],
    reconstructed_line: dict[str, object] | None,
    line_suggestions: list[TokenSuggestion],
) -> tuple[str, str, list[dict[str, object]]]:
    grouped: dict[int, list[TokenSuggestion]] = defaultdict(list)
    for suggestion in line_suggestions:
        if suggestion.segment_index is not None:
            grouped[suggestion.segment_index].append(suggestion)

    reconstructed_segments = segment_map(reconstructed_line)
    segments = entry.get("segments")
    raw_segments = [str(segment) for segment in segments] if isinstance(segments, list) and segments else [str(entry.get("text", ""))]
    comparison_segments: list[dict[str, object]] = []

    for segment_index, raw_segment in enumerate(raw_segments, start=1):
        reconstructed_segment = reconstructed_segments.get(segment_index, {})
        reconstructed_text = str(reconstructed_segment.get("reconstructed_text", "")).strip() or raw_segment
        candidates = [item for item in grouped.get(segment_index, []) if item.combined_score >= 0.25]
        best = max(candidates, key=suggestion_priority) if candidates else None
        contextual_text = best.best_surface.strip() if best and best.best_surface.strip() else reconstructed_text
        comparison_segments.append(
            {
                "segment_index": segment_index,
                "raw_text": raw_segment,
                "reconstructed_text": reconstructed_text,
                "contextual_text": contextual_text,
                "strategy": str(reconstructed_segment.get("strategy", "raw")),
                "best_source": best.source if best else str(reconstructed_segment.get("strategy", "raw")),
            }
        )

    reconstructed_text = " ".join(segment["reconstructed_text"] for segment in comparison_segments if segment["reconstructed_text"]).strip()
    contextual_text = " ".join(segment["contextual_text"] for segment in comparison_segments if segment["contextual_text"]).strip()
    return reconstructed_text, contextual_text, comparison_segments


def line_record(
    entry: dict[str, object],
    line_suggestions: list[TokenSuggestion],
    reconstructed_line: dict[str, object] | None,
) -> dict[str, object] | None:
    line_index = entry.get("line_index")
    if not isinstance(line_index, int):
        return None

    raw_text = str(entry.get("text", ""))
    reconstructed_text, contextual_text, comparison_segments = build_line_views(entry, reconstructed_line, line_suggestions)
    preferred_text = contextual_text or reconstructed_text or raw_text
    return {
        "line_index": line_index,
        "raw_text": raw_text,
        "draft_text": preferred_text,
        "reconstructed_text": reconstructed_text,
        "contextual_text": contextual_text,
        "preferred_text": preferred_text,
        "line_image": str(entry.get("line_image", "")),
        "segments": entry.get("segments", []),
        "comparison_segments": comparison_segments,
        "suggestions": [
            {
                "token_id": item.token_id,
                "observed_token": item.observed_token,
                "best_surface": item.best_surface,
                "source": item.source,
                "observed_source": item.observed_source,
                "segment_index": item.segment_index,
                "local_score": round(item.local_score, 4),
                "context_score": round(item.context_score, 4),
                "combined_score": round(item.combined_score, 4),
                "candidate_gloss": item.candidate_gloss,
                "changed": item.changed,
                "confidence": item.confidence_label,
            }
            for item in line_suggestions
        ],
    }


def load_review_lines() -> list[dict[str, object]]:
    raw_chunks, token_candidates, best_contextual, reconstructed_lines, artifact_info = load_review_inputs()
    suggestions_by_line = build_suggestions_by_line(token_candidates, best_contextual)
    reconstructed_by_line = reconstructed_line_index(reconstructed_lines)

    lines: list[dict[str, object]] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        for entry in chunk.get("entries", []):
            if not isinstance(entry, dict):
                continue
            line_index = entry.get("line_index")
            line_suggestions = suggestions_by_line.get(line_index, [])
            record = line_record(entry, line_suggestions, reconstructed_by_line.get(line_index))
            if record is not None:
                record["artifact_info"] = artifact_info
                lines.append(record)

    return sorted(lines, key=lambda item: int(item["line_index"]))


def line_lookup(lines: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(line["line_index"]): line for line in lines}


def compact_suggestion_rows(line: dict[str, object], limit: int = 8) -> list[str]:
    suggestions = line.get("suggestions", []) if isinstance(line.get("suggestions"), list) else []
    scored = sorted(
        [item for item in suggestions if isinstance(item, dict)],
        key=lambda item: float(item.get("combined_score", 0.0)),
        reverse=True,
    )
    rows: list[str] = []
    for item in scored:
        observed = str(item.get("observed_token", "")).strip()
        best = str(item.get("best_surface", "")).strip()
        source = str(item.get("source", "")).strip()
        score = float(item.get("combined_score", 0.0))
        gloss = str(item.get("candidate_gloss", "")).strip()
        if not observed:
            continue
        if best and best != observed:
            row = f"- {observed} -> {best} | source={source} | score={score:.3f}"
        else:
            row = f"- {observed} | source={source} | score={score:.3f}"
        if gloss:
            row += f" | gloss={gloss}"
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def trim_text(value: object, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def build_interpretation_prompt(line: dict[str, object]) -> tuple[str, str]:
    annotation = line.get("annotation", {}) if isinstance(line.get("annotation"), dict) else {}
    suggestion_rows = compact_suggestion_rows(line)
    evidence_block = "\n".join(suggestion_rows) if suggestion_rows else "- none"
    system_prompt = (
        "You are helping scholars review a noisy Sanskrit manuscript line. "
        "Use raw OCR, reconstructed machine readings, contextual token evidence, and reviewer notes as evidence. "
        "Return strict JSON only. Do not add markdown fences."
    )
    user_prompt = "\n".join(
        [
            "Generate exactly five distinct interpretation candidates for this manuscript line.",
            "Use the raw OCR as primary evidence. Prefer the contextual machine reading over raw OCR only when it is well supported by the token evidence.",
            "Use machine hints conservatively. Do not invent missing text.",
            "Return a JSON object with one key: candidates.",
            "Each candidate must contain: id, strategy, normalized_sanskrit, interpretation_english, interpretation_hindi, uncertainty_notes, evidence_notes.",
            f"Page: {PAGE_NUMBER}",
            f"Line: {line['line_index']}",
            f"Raw OCR: {trim_text(line.get('raw_text', ''), 900)}",
            f"Akshara reconstruction: {trim_text(line.get('reconstructed_text', ''), 900) or 'none'}",
            f"Contextual reading: {trim_text(line.get('contextual_text', ''), 900) or 'none'}",
            f"Preferred machine reading: {trim_text(line.get('preferred_text', ''), 900) or 'none'}",
            f"Reviewer final text: {trim_text(annotation.get('final_text', ''), 600) or 'none'}",
            f"Reviewer notes: {trim_text(annotation.get('notes', ''), 600) or 'none'}",
            "Top token suggestions:",
            evidence_block,
            "Strategies to cover across the five candidates: conservative, literal, grammar-first, semantic, alternate-compound.",
        ]
    )
    return system_prompt, user_prompt


def interpretation_input_snapshot(line: dict[str, object]) -> dict[str, object]:
    return {
        "line_index": int(line["line_index"]),
        "raw_text": str(line.get("raw_text", "")),
        "reconstructed_text": str(line.get("reconstructed_text", "")),
        "contextual_text": str(line.get("contextual_text", "")),
        "preferred_text": str(line.get("preferred_text", "")),
        "comparison_segments": [
            segment
            for segment in line.get("comparison_segments", [])
            if isinstance(segment, dict)
        ],
        "artifacts": dict(line.get("artifact_info", {})) if isinstance(line.get("artifact_info"), dict) else {},
    }


def interpretation_snapshot_or_current(line: dict[str, object]) -> dict[str, object]:
    generated = line.get("interpretations", {}) if isinstance(line.get("interpretations"), dict) else {}
    snapshot = generated.get("input_snapshot") if isinstance(generated.get("input_snapshot"), dict) else None
    if isinstance(snapshot, dict):
        return snapshot
    return interpretation_input_snapshot(line)


def request_groq_json(system_prompt: str, user_prompt: str, model: str) -> dict[str, object]:
    config = groq_config()
    api_key = str(config["api_key"])
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to .env before generating interpretations.")

    payload = {
        "model": model,
        "temperature": 0.55,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    encoded_payload = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=encoded_payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "manuscript-interpreter/1.0",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "1010" in detail:
            body = request_groq_via_curl(api_key=api_key, payload_json=encoded_payload)
        else:
            raise RuntimeError(f"Groq API request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Groq API connection failed: {exc.reason}") from exc

    choices = body.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Groq API returned no choices.")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = str(message.get("content", ""))
    result = extract_json_payload(content)
    return result


def request_groq_interpretations(line: dict[str, object]) -> dict[str, object]:
    config = groq_config()
    system_prompt, user_prompt = build_interpretation_prompt(line)
    result = request_groq_json(system_prompt, user_prompt, str(config["line_model"]))
    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("Groq response JSON must contain a 'candidates' list.")
    return {
        "model": str(config["line_model"]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_snapshot": interpretation_input_snapshot(line),
        "candidates": candidates,
        "raw_response": result,
    }


def request_groq_via_curl(api_key: str, payload_json: bytes) -> dict[str, object]:
    completed = subprocess.run(
        [
            "curl",
            "-sS",
            "https://api.groq.com/openai/v1/chat/completions",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ],
        input=payload_json,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq curl fallback failed: {stderr.strip() or completed.returncode}")
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq curl fallback returned invalid JSON.") from exc


def save_generated_interpretations(line_index: int, payload: dict[str, object]) -> None:
    store = interpretation_store()
    entries = store.setdefault("interpretations", {})
    if not isinstance(entries, dict):
        entries = {}
        store["interpretations"] = entries
    entries[str(line_index)] = payload
    save_interpretation_store(store)


def save_chunk_syntheses(chunks: list[dict[str, object]]) -> None:
    store = interpretation_store()
    store["chunk_syntheses"] = chunks
    save_interpretation_store(store)


def save_page_synthesis(payload: dict[str, object]) -> None:
    store = interpretation_store()
    store["page_synthesis"] = payload
    save_interpretation_store(store)


def error_interpretation_payload(message: str) -> dict[str, object]:
    config = groq_config()
    return {
        "status": "error",
        "model": str(config["line_model"]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message": message,
        "input_snapshot": {},
        "candidates": [],
    }


def success_payload(payload: dict[str, object]) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("candidates"), list) and bool(payload.get("candidates"))


def should_generate_line(line: dict[str, object]) -> bool:
    generated = line.get("interpretations", {}) if isinstance(line.get("interpretations"), dict) else {}
    if not success_payload(generated):
        return True
    annotation = line.get("annotation", {}) if isinstance(line.get("annotation"), dict) else {}
    updated_at = str(annotation.get("updated_at", "")).strip()
    generated_at = str(generated.get("generated_at", "")).strip()
    return bool(updated_at and (not generated_at or updated_at > generated_at))


def generate_line_payload(line: dict[str, object]) -> dict[str, object]:
    try:
        return request_groq_interpretations(line)
    except (RuntimeError, ValueError) as exc:
        return error_interpretation_payload(str(exc))


def line_summary_record(line: dict[str, object]) -> dict[str, object] | None:
    payload = line.get("interpretations", {}) if isinstance(line.get("interpretations"), dict) else {}
    if not success_payload(payload):
        return None
    candidates = payload.get("candidates", [])
    top = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else {}
    annotation = line.get("annotation", {}) if isinstance(line.get("annotation"), dict) else {}
    snapshot = interpretation_snapshot_or_current(line)
    return {
        "line_index": line["line_index"],
        "verdict": annotation.get("verdict", "unreviewed"),
        "final_text": trim_text(annotation.get("final_text", ""), 300) or "none",
        "preferred_text": trim_text(snapshot.get("preferred_text", ""), 300) or "none",
        "notes": trim_text(annotation.get("notes", ""), 240) or "none",
        "normalized_sanskrit": trim_text(top.get("normalized_sanskrit", ""), 300) or "none",
        "interpretation_english": trim_text(top.get("interpretation_english", ""), 360) or "none",
        "interpretation_hindi": trim_text(top.get("interpretation_hindi", ""), 360) or "none",
        "uncertainty_notes": trim_text(top.get("uncertainty_notes", ""), 240) or "none",
    }


def chunked_line_summaries(lines: list[dict[str, object]], chunk_size: int = PAGE_CHUNK_SIZE) -> list[list[dict[str, object]]]:
    summaries = [summary for line in lines if (summary := line_summary_record(line)) is not None]
    return [summaries[index : index + chunk_size] for index in range(0, len(summaries), chunk_size)]


def build_chunk_synthesis_prompt(chunk_index: int, chunk: list[dict[str, object]]) -> tuple[str, str]:
    line_block = "\n\n".join(
        [
            "\n".join(
                [
                    f"Line {item['line_index']}",
                    f"Verdict: {item['verdict']}",
                    f"Reviewer final text: {item['final_text']}",
                    f"Preferred machine reading: {item['preferred_text']}",
                    f"Top normalized Sanskrit: {item['normalized_sanskrit']}",
                    f"English gloss: {item['interpretation_english']}",
                    f"Hindi gloss: {item['interpretation_hindi']}",
                    f"Uncertainty: {item['uncertainty_notes']}",
                    f"Reviewer notes: {item['notes']}",
                ]
            )
            for item in chunk
        ]
    )
    system_prompt = (
        "You are synthesizing a chunk of Sanskrit manuscript line interpretations. "
        "Stay conservative, preserve uncertainty, and return strict JSON only."
    )
    user_prompt = "\n".join(
        [
            "Synthesize this chunk of line-level interpretations into a compact scholarly note.",
            "Return a JSON object with keys: chunk_id, line_range, synthesis_english, synthesis_hindi, thematic_notes, open_questions.",
            f"Page: {PAGE_NUMBER}",
            f"Chunk: {chunk_index}",
            line_block,
        ]
    )
    return system_prompt, user_prompt


def build_page_synthesis_prompt(chunks: list[dict[str, object]]) -> tuple[str, str]:
    chunk_block = "\n\n".join(
        [
            "\n".join(
                [
                    f"Chunk {index}",
                    f"Line range: {chunk.get('line_range', 'unknown')}",
                    f"English synthesis: {trim_text(chunk.get('synthesis_english', ''), 500) or 'none'}",
                    f"Hindi synthesis: {trim_text(chunk.get('synthesis_hindi', ''), 500) or 'none'}",
                    f"Themes: {trim_text(chunk.get('thematic_notes', ''), 320) or 'none'}",
                    f"Open questions: {trim_text(chunk.get('open_questions', ''), 320) or 'none'}",
                ]
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
    )
    system_prompt = (
        "You are preparing a page-level scholarly synthesis from chunked Sanskrit manuscript interpretations. "
        "Be conservative, preserve unresolved uncertainty, and return strict JSON only."
    )
    user_prompt = "\n".join(
        [
            "Generate a page-level synthesis from the chunk summaries below.",
            "Return a JSON object with keys: page_overview_english, page_overview_hindi, likely_theme, collation_notes, open_questions.",
            f"Page: {PAGE_NUMBER}",
            chunk_block,
        ]
    )
    return system_prompt, user_prompt


def synthesize_chunk(chunk_index: int, chunk: list[dict[str, object]]) -> dict[str, object]:
    config = groq_config()
    system_prompt, user_prompt = build_chunk_synthesis_prompt(chunk_index, chunk)
    result = request_groq_json(system_prompt, user_prompt, str(config["page_model"]))
    result["model"] = str(config["page_model"])
    result["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line_indexes = [int(item["line_index"]) for item in chunk]
    if line_indexes:
        result.setdefault("chunk_id", chunk_index)
        result.setdefault("line_range", f"{min(line_indexes)}-{max(line_indexes)}")
    return result


def synthesize_page_from_lines(lines: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    config = groq_config()
    chunks: list[dict[str, object]] = []
    for chunk_index, chunk in enumerate(chunked_line_summaries(lines), start=1):
        chunks.append(synthesize_chunk(chunk_index, chunk))
    if not chunks:
        raise ValueError("No successful line interpretations are available yet for page collation.")
    system_prompt, user_prompt = build_page_synthesis_prompt(chunks)
    page_result = request_groq_json(system_prompt, user_prompt, str(config["page_model"]))
    page_result["model"] = str(config["page_model"])
    page_result["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    page_result["chunk_count"] = len(chunks)
    return chunks, page_result


def build_page_context() -> dict[str, object]:
    store = annotation_store()
    interpretation_state = interpretation_store()
    interpretation_entries = interpretation_state.get("interpretations", {})
    stored_annotations = store.get("annotations", {}) if isinstance(store, dict) else {}
    lines = load_review_lines()
    for line in lines:
        line_key = str(line["line_index"])
        annotation = stored_annotations.get(line_key, {}) if isinstance(stored_annotations, dict) else {}
        line["annotation"] = {
            "reviewer": annotation.get("reviewer", ""),
            "verdict": annotation.get("verdict", "unreviewed"),
            "final_text": annotation.get("final_text", line["preferred_text"]),
            "notes": annotation.get("notes", ""),
            "updated_at": annotation.get("updated_at", ""),
        }
        generated = interpretation_entries.get(line_key, {}) if isinstance(interpretation_entries, dict) else {}
        line["interpretations"] = generated if isinstance(generated, dict) else {}
        line["interpretation_snapshot"] = interpretation_snapshot_or_current(line)
    groq = groq_config()
    artifact_info = lines[0].get("artifact_info", {}) if lines else {}
    return {
        "page_number": PAGE_NUMBER,
        "page_image": page_image_path(PAGE_NUMBER).name,
        "line_count": len(lines),
        "line_generation": {
            "successful": sum(1 for line in lines if success_payload(line.get("interpretations", {}))),
            "pending": sum(1 for line in lines if should_generate_line(line)),
        },
        "lines": lines,
        "chunk_syntheses": interpretation_state.get("chunk_syntheses", []),
        "page_synthesis": interpretation_state.get("page_synthesis", {}),
        "artifacts": artifact_info,
        "groq": {
            "configured": bool(groq["configured"]),
            "model": str(groq["line_model"]),
            "line_model": str(groq["line_model"]),
            "page_model": str(groq["page_model"]),
        },
    }


@app.get("/")
def index():
    return render_template("annotation_app.html", page=build_page_context())


@app.get("/api/page/<int:page_number>")
def page_data(page_number: int):
    if page_number != PAGE_NUMBER:
        return jsonify({"error": "Only page 002 is wired in this MVP."}), 404
    return jsonify(build_page_context())


@app.get("/page-image/<path:filename>")
def serve_page_image(filename: str):
    path = PAGE_IMAGE_DIR / filename
    if not path.is_file():
        return jsonify({"error": "Image not found"}), 404
    return send_file(path)


@app.get("/line-image/<int:line_index>")
def serve_line_image(line_index: int):
    line_path = BASE_DIR / "line_images" / "page_002_cv_split2" / f"line_{line_index:03d}.png"
    if not line_path.is_file():
        return jsonify({"error": "Line image not found"}), 404
    return send_file(line_path)


@app.post("/annotate/<int:line_index>")
def annotate_line(line_index: int):
    store = annotation_store()
    annotations = store.setdefault("annotations", {})
    if not isinstance(annotations, dict):
        annotations = {}
        store["annotations"] = annotations
    annotations[str(line_index)] = {
        "reviewer": request.form.get("reviewer", "").strip(),
        "verdict": request.form.get("verdict", "unreviewed").strip() or "unreviewed",
        "final_text": request.form.get("final_text", "").strip(),
        "notes": request.form.get("notes", "").strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_annotation_store(store)
    return redirect(url_for("index", line=line_index))


@app.post("/generate-interpretations/<int:line_index>")
def generate_interpretations(line_index: int):
    page = build_page_context()
    lines = page.get("lines", []) if isinstance(page, dict) else []
    line = line_lookup(lines).get(line_index)
    if line is None:
        abort(404, description=f"Line {line_index} was not found.")
    try:
        result = request_groq_interpretations(line)
    except (RuntimeError, ValueError) as exc:
        save_generated_interpretations(line_index, error_interpretation_payload(str(exc)))
        return redirect(url_for("index", line=line_index))
    save_generated_interpretations(line_index, result)
    return redirect(url_for("index", line=line_index))


@app.post("/generate-page-interpretation")
def generate_page_interpretation():
    page = build_page_context()
    lines = page.get("lines", []) if isinstance(page, dict) else []
    line_results: dict[int, dict[str, object]] = {}
    for line in lines:
        line_index = int(line["line_index"])
        if should_generate_line(line):
            payload = generate_line_payload(line)
            save_generated_interpretations(line_index, payload)
        else:
            payload = line.get("interpretations", {}) if isinstance(line.get("interpretations"), dict) else {}
        line_results[line_index] = payload

    refreshed_page = build_page_context()
    refreshed_lines = refreshed_page.get("lines", []) if isinstance(refreshed_page, dict) else []
    try:
        chunk_summaries, page_payload = synthesize_page_from_lines(refreshed_lines)
    except (RuntimeError, ValueError) as exc:
        store = interpretation_store()
        store["page_synthesis"] = {
            "status": "error",
            "model": str(groq_config()["page_model"]),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "message": str(exc),
            "line_results": {str(key): value.get("status", "ok") for key, value in line_results.items()},
        }
        save_interpretation_store(store)
        return redirect(url_for("index"))

    save_chunk_syntheses(chunk_summaries)
    page_payload["line_results"] = {str(key): value.get("status", "ok") for key, value in line_results.items()}
    save_page_synthesis(page_payload)
    return redirect(url_for("index"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manuscript annotation review app.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)