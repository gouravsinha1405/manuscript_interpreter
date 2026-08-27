# Historical Devanagari HTR research notes

## Why the current OCR stage is the main bottleneck

The downstream token-graph and interpretation stages are already useful for preserving ambiguity, but they currently receive heavily corrupted text. The page-002 review output itself warns that the OCR is highly distorted and that only fragments can be trusted. Improving recognition quality therefore has higher leverage than adding more semantic heuristics at this stage.

## Relevant recent work

### AnciDev (BHASHA 2025)

AnciDev is a public dataset built specifically for ancient Devanagari handwritten text recognition. It contains 3,000 transcribed text lines from 500 manuscript pages and reports large gains after fine-tuning HTR models on this domain.

Repository: https://github.com/vriti2003/AnciDev
Paper: https://aclanthology.org/2025.bhasha-1.8/

The repository includes line-preparation scripts, a Tesseract-5 training workflow, and CER/WER evaluation notebooks. This is unusually well aligned with this project because our pages are also historical Devanagari and our pipeline already segments pages into line crops.

## Recommended direction

Treat OCR as a ranked HTR ensemble rather than a single recognizer:

1. Segment a page into line crops.
2. Run the current PaddleOCR recognizer.
3. Run a historical-Devanagari Tesseract model (for example, one trained with AnciDev-style data).
4. Preserve both hypotheses rather than forcing an early winner.
5. Compare models on manually transcribed line-level ground truth using CER and WER.
6. Feed top-N recognition hypotheses into the Sanskrit token lattice/contextual graph.

This changes the architecture from:

`image -> OCR string -> token correction`

into:

`image -> multiple HTR hypotheses -> calibrated evidence lattice -> Sanskrit/contextual reranking`

That is important because a philological system should represent uncertainty explicitly instead of silently rewriting an uncertain glyph sequence into a plausible Sanskrit word.

## Evaluation protocol

Use a small gold set before optimizing the whole 73-page manuscript.

Recommended first benchmark:

- 5 pages chosen for different handwriting/background quality
- 8-15 manually transcribed lines per page
- Unicode NFC normalization before scoring
- Character Error Rate (CER) as the primary metric
- Word Error Rate (WER) as the secondary metric
- report per-line and aggregate scores

CER is more informative than apparent word plausibility for historical Sanskrit because sandhi, compounds, orthographic variants, and manuscript spelling can make word-level matching brittle.

## Avoid this failure mode

Do not select preprocessing/OCR variants primarily because the output contains words from a modern Sanskrit lexicon. That can reward hallucinated modern-looking forms and suppress unusual but genuine historical spellings.

Lexicon and grammar evidence should remain downstream evidence, not ground truth for visual recognition.

## Proposed phases

### Phase A — establish a real OCR benchmark

- Add Tesseract line-level HTR runner.
- Allow custom `.traineddata` models via `--tessdata-dir` and `--tesseract-lang`.
- Add CER/WER scoring against line-level gold transcriptions.
- Compare PaddleOCR vs historical Tesseract on exactly the same crops.

### Phase B — retain ambiguity

Instead of storing only one string per line, save candidates in a structure such as:

```json
{
  "line": "line_007.png",
  "hypotheses": [
    {"engine": "paddleocr", "text": "...", "score": 0.61},
    {"engine": "tesseract_historical", "text": "...", "score": 0.55}
  ]
}
```

Then let the akshara/token graph consume both.

### Phase C — domain adaptation

Once 100-300 lines from this manuscript have been manually corrected, use them for manuscript-specific adaptation. A small amount of writer/domain-specific supervision may be more valuable than adding increasingly complicated post-OCR correction rules.

## Immediate code change

`benchmark_tesseract_htr.py` has been added as a first experiment. It works directly on the line crops already produced by `segment_lines.py`, accepts a custom Tesseract model, and optionally computes CER/WER from a TSV gold file.
