"""Run a local PDF inspection and write its results as JSON."""

from argparse import ArgumentParser
from dataclasses import asdict
import json
from pathlib import Path

from pypdfium2 import PdfiumError

from .native_pdf import inspect_pdf


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Native PDF to inspect")
    args = parser.parse_args()

    try:
        document = inspect_pdf(args.pdf)
    except (OSError, PdfiumError) as error:
        parser.exit(1, f"Unable to inspect PDF: {error}\n")

    result = {
        "file_sha256": document.file_sha256,
        "suggested_type": document.suggested_type,
        "coordinate_system": "PDF user space (left, bottom, right, top)",
        "offsets": "Zero-based Unicode character offsets within each page, with exclusive end",
        "pages": [
            {
                "number": page.number,
                "label": page.label,
                "bounds": page.bounds,
                "rotation": page.rotation,
                "status": page.status,
                "text": page.text,
            }
            for page in document.pages
        ],
        "provisions": [asdict(provision) for provision in document.provisions],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
