from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None

try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
except ImportError:
    convert_from_path = None
    PDFInfoNotInstalledError = RuntimeError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render each page of a PDF into high-resolution PNG images."
    )
    parser.add_argument("pdf", type=Path, help="Path to the source PDF file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("page_images"),
        help="Directory where rendered PNG files will be written",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Render resolution in DPI. Manuscript OCR usually benefits from 300-600 DPI.",
    )
    parser.add_argument(
        "--prefix",
        default="page",
        help="Filename prefix for the generated images",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        default=None,
        help="Optional first page to render (1-based)",
    )
    parser.add_argument(
        "--last-page",
        type=int,
        default=None,
        help="Optional last page to render (1-based)",
    )
    parser.add_argument(
        "--poppler-path",
        type=Path,
        default=None,
        help="Optional path to a local Poppler installation if pdftoppm is not on PATH",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "pdf2image", "pymupdf"),
        default="auto",
        help="Renderer backend. `auto` prefers pdf2image and falls back to PyMuPDF.",
    )
    return parser.parse_args()


def render_with_pdf2image(pdf_path: Path, output_dir: Path, dpi: int, prefix: str, **kwargs: object) -> int:
    if convert_from_path is None:
        raise RuntimeError(
            "pdf2image is not installed. Install dependencies with `pip install -r requirements.txt`."
        )

    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        fmt="png",
        **kwargs,
    )

    for index, page in enumerate(pages, start=1):
        destination = output_dir / f"{prefix}_{index:03}.png"
        page.save(destination, "PNG")
        print(destination)

    return len(pages)


def render_with_pymupdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int,
    prefix: str,
    first_page: int | None = None,
    last_page: int | None = None,
) -> int:
    if fitz is None:
        raise RuntimeError(
            "PyMuPDF is not installed. Install dependencies with `pip install -r requirements.txt`."
        )

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    with fitz.open(pdf_path) as document:
        start = 1 if first_page is None else first_page
        stop = document.page_count if last_page is None else min(last_page, document.page_count)

        if start < 1 or stop < start:
            raise ValueError("Invalid page range requested.")

        rendered = 0
        for page_number in range(start, stop + 1):
            page = document.load_page(page_number - 1)
            destination = output_dir / f"{prefix}_{page_number:03}.png"
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(destination)
            print(destination)
            rendered += 1

        return rendered


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int, prefix: str, backend: str, **kwargs: object) -> int:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "pymupdf":
        return render_with_pymupdf(
            pdf_path,
            output_dir,
            dpi,
            prefix,
            first_page=kwargs.get("first_page"),
            last_page=kwargs.get("last_page"),
        )

    try:
        return render_with_pdf2image(pdf_path, output_dir, dpi, prefix, **kwargs)
    except PDFInfoNotInstalledError:
        if backend == "pdf2image":
            raise RuntimeError(
                "Poppler is required by pdf2image. Install `pdftoppm` or pass --poppler-path."
            )
        return render_with_pymupdf(
            pdf_path,
            output_dir,
            dpi,
            prefix,
            first_page=kwargs.get("first_page"),
            last_page=kwargs.get("last_page"),
        )


def main() -> int:
    args = parse_args()

    try:
        page_count = render_pdf(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            dpi=args.dpi,
            prefix=args.prefix,
            backend=args.backend,
            first_page=args.first_page,
            last_page=args.last_page,
            poppler_path=str(args.poppler_path) if args.poppler_path else None,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Rendered {page_count} page(s) at {args.dpi} DPI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())