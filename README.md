# Manuscript Interpreter

Minimal Python pipeline for OCR-assisted interpretation of a Sanskrit manuscript page set.

This repo is prepared for replication, not as a polished product release. It keeps the code, dependency list, environment template, page-scan data, and scholar-facing `parinam` markdown outputs, while excluding secrets and heavy derived artifacts.

## What is included

- Python pipeline scripts for page rendering, line segmentation, OCR, token graph construction, parser probing, contextual reranking, and review export
- Page-scan data already present in `page_scan/`
- Scholar-facing result markdown in `annotations/*parinam*.md`
- `requirements.txt`
- `.env.example`

## What is intentionally excluded

- `.env` and any API keys
- local virtualenvs
- intermediate pipeline artifacts in `phase2_input/`
- OCR output caches and benchmark folders
- generated Excel/JSON review artifacts
- page image PNGs
- the source PDF

## Core stack

- Python 3.12
- PaddleOCR / PaddlePaddle
- OpenCV
- PyMuPDF
- pdf2image
- `sanskrit_parser`
- `indic-transliteration`
- Flask
- OpenPyXL
- Groq API with `qwen/qwen3-32b` and optional `openai/gpt-oss-120b`

## Quick setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then fill `.env` with your Groq key.

## Minimal replication flow

1. Render page images if needed:

```bash
python render_pdf_pages.py
```

2. Segment lines / windows and run OCR:

```bash
python segment_lines.py
python run_paddle_ocr.py
```

3. Prepare phase-2 page inputs:

```bash
python prepare_phase2_input.py
```

4. Build Sanskrit token graph:

```bash
python build_sanskrit_token_graph.py
```

5. Probe difficult Sanskrit tokens:

```bash
python probe_unmatched_sanskrit_tokens.py
```

6. Build contextual graph:

```bash
python build_contextual_sanskrit_graph.py \
  --base-graph phase2_input/page_002_token_graph_akshara/graph.json \
  --token-candidates phase2_input/page_002_token_graph_akshara/token_candidates.json \
  --parser-probe phase2_input/page_002_parser_probe/probe_results.json \
  --output-dir phase2_input/page_002_contextual_graph_akshara_quick
```

7. Run the review / interpretation app:

```bash
python annotation_app.py
```

8. Export the SME workbook:

```bash
python export_sme_review_workbook.py
```

## Main files

- `annotation_app.py`: Groq-backed line/page interpretation and review flow
- `build_sanskrit_token_graph.py`: token candidate graph construction
- `probe_unmatched_sanskrit_tokens.py`: parser and DP split probing for difficult OCR tokens
- `build_contextual_sanskrit_graph.py`: contextual reranking and augmentation
- `analyze_sanskrit_token_splits.py`: direct token split analysis utilities
- `export_sme_review_workbook.py`: Excel export for SME review

## Notes

- The current working slice is centered on page 002.
- LLM outputs are interpretive aids, not validated philological truth.
- If you want a clean rerun, recreate `phase2_input/` locally instead of relying on excluded derived artifacts.
