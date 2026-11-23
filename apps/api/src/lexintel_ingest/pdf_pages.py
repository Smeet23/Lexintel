"""Read PDF text and repair scanned or mixed pages with local Tesseract OCR."""

from collections import defaultdict
from contextlib import closing
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import csv
import subprocess

import pypdfium2 as pdfium
from pypdfium2 import raw

from .config import settings
from .native_pdf import Page, Rectangle, enclose, read_page


def _intersection(a: Rectangle, b: Rectangle) -> float:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(box: Rectangle) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _native_lines(page: Page):
    start = 0
    for text in page.text.split("\n"):
        boxes = page.character_boxes[start:start + len(text)]
        if text.strip():
            yield text, boxes
        start += len(text) + 1


def _box_from_pixels(converter, x, y, width, height) -> Rectangle:
    a, b = converter.to_page(x, y), converter.to_page(x + width, y + height)
    return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])


def _ocr_lines(page, images, previous_boxes):
    # Keep the longest side bounded for large scanned drawings and posters.
    scale = min(3, 3800 / max(page.get_size()))
    with closing(page.render(scale=scale, grayscale=True)) as bitmap:
        buffer = BytesIO()
        bitmap.to_pil().save(buffer, format="PNG")
        process = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", settings.ocr_languages, "--psm", "3", "tsv"],
            input=buffer.getvalue(), capture_output=True, check=True, timeout=55,
        )
        converter = bitmap.get_posconv(page)
        rows = csv.DictReader(process.stdout.decode("utf-8").splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
        lines = defaultdict(list)
        low_confidence = 0
        words = 0
        for row in rows:
            text = (row.get("text") or "").strip()
            if not text or row["level"] != "5":
                continue
            box = _box_from_pixels(converter, *(int(row[k]) for k in ("left", "top", "width", "height")))
            area = _area(box)
            if area <= 0:
                continue
            if previous_boxes:
                if not any(_intersection(box, image) > area * .5 for image in images):
                    continue
                overlap = sum(_intersection(box, native_box) for native_box in previous_boxes)
                if overlap >= area * .4:
                    continue
            key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
            lines[key].append((text, box))
            words += 1
            low_confidence += float(row["conf"]) < 55

    output = []
    for line in lines.values():
        text = " ".join(word for word, _ in line)
        boxes = []
        for word, box in line:
            if boxes:
                boxes.append(None)
            boxes.extend([box] * len(word))
        output.append((text, tuple(boxes)))
    return output, words > 0 and low_confidence / words > .15


def _reading_position(line, page):
    boxes = [box for box in line[1] if box is not None]
    if not boxes:
        return (float("inf"), 0)
    left, bottom, right, top = enclose(boxes)
    return {
        0: (-top, left),
        90: (left, bottom),
        180: (bottom, -right),
        270: (-right, -top),
    }.get(page.rotation, (-top, left))


def _read_with_ocr(pdf_page, page: Page):
    warnings = []
    if page.status == "text_layer_with_gaps":
        warnings.append({"code": "native_geometry_incomplete", "page": page.number})

    page_area = _area(page.bounds)
    images = [
        image.get_bounds() for image in pdf_page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE])
    ]
    images = [box for box in images if _intersection(box, page.bounds) >= page_area * .025]
    native = bool(page.text.strip())
    if native and not images:
        return page, "native", warnings
    if not native and not images and not any(pdf_page.get_objects()):
        return page, "blank", warnings

    try:
        native_boxes = list({box for box in page.character_boxes if box is not None})
        if native and not native_boxes:
            warnings.append({"code": "mixed_page_text_cannot_be_aligned", "page": page.number})
            return page, "native", warnings
        ocr, low_confidence = _ocr_lines(pdf_page, images, native_boxes)
    except (OSError, subprocess.SubprocessError, ValueError, pdfium.PdfiumError):
        warnings.append({"code": "ocr_failed", "page": page.number})
        return page, "native" if native else "unreadable", warnings

    if not ocr:
        if not native:
            warnings.append({"code": "page_without_readable_text", "page": page.number})
        return page, "native" if native else "unreadable", warnings
    if low_confidence:
        warnings.append({"code": "ocr_low_confidence", "page": page.number})

    lines = ocr
    if native:
        lines = sorted([*_native_lines(page), *ocr], key=lambda line: _reading_position(line, page))
    characters, boxes = [], []
    for text, line_boxes in lines:
        if characters:
            characters.append("\n")
            boxes.append(None)
        characters.append(text)
        boxes.extend(line_boxes)

    combined = replace(page, text="".join(characters), character_boxes=tuple(boxes))
    return combined, "mixed" if native else "ocr", warnings


def load_pdf(path: Path):
    pages, modes, warnings = [], [], []
    with pdfium.PdfDocument(path) as document:
        if len(document) > settings.max_pdf_pages:
            raise ValueError(f"PDF exceeds the {settings.max_pdf_pages}-page limit")
        for index in range(len(document)):
            with closing(document[index]) as handle:
                label = document.get_page_label(index) or None
                native = read_page(handle, index + 1, label)
                page, mode, page_warnings = _read_with_ocr(handle, native)
                pages.append(page)
                modes.append(mode)
                warnings.extend(page_warnings)
    return pages, modes, warnings
