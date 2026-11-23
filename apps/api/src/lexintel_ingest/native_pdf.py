"""Inspect the native text layer of a legal PDF and preserve its source locations."""

from contextlib import closing
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
from typing import Literal

import pypdfium2 as pdfium
from pypdfium2 import raw

Rectangle = tuple[float, float, float, float]
DocumentType = Literal["agreement", "pleading", "unknown"]
PageStatus = Literal["text_layer", "text_layer_with_gaps", "no_text_layer"]

AGREEMENT_TITLE = re.compile(
    r"^[^\n]{0,90}\b(?:AGREEMENT|CONTRACT|ADDENDUM|AMENDMENT)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
PLEADING_TITLE = re.compile(
    r"^\s*(?:(?:FIRST|SECOND|AMENDED)\s+)*(?:COMPLAINT|ANSWER|PETITION)(?:\s+(?:FOR|TO)\b[^\n]*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
COURT_CAPTION = re.compile(r"^\s*(?:IN THE )?(?:UNITED STATES |SUPREME |HIGH |DISTRICT ).*\bCOURT\b", re.IGNORECASE | re.MULTILINE)
BOUNDARY = re.compile(r"^\s*(?:[IVXLCDM]+\.\s+[A-Z]|(?:ARTICLE|COUNT|EXHIBIT|SCHEDULE|ANNEX)\s+[A-Z0-9]+\b)")
DOTTED_NUMBER = re.compile(r"^\s*(?:SECTION\s+)?(?P<label>\d+(?:\.\d+)+(?:\([\w]+\))*)\.?[ \t]+(?=\S)", re.IGNORECASE)
PARAGRAPH_NUMBER = re.compile(r"^\s*(?:¶\s*)?(?P<label>\d{1,4})[.)][ \t]+(?=\S)")


@dataclass(frozen=True)
class SourceSpan:
    page_number: int
    start: int
    end: int
    text: str
    rectangles: tuple[Rectangle, ...]


@dataclass(frozen=True)
class Page:
    number: int
    label: str | None
    bounds: Rectangle
    rotation: int
    status: PageStatus
    text: str
    character_boxes: tuple[Rectangle | None, ...] = field(repr=False)

    def span(self, start: int, end: int) -> SourceSpan:
        """Resolve Python character offsets to rectangles in native PDF coordinates."""
        if not 0 <= start < end <= len(self.text):
            raise ValueError("Source span must be a nonempty range within the page")

        lines = []
        line_boxes: list[Rectangle] = []
        for offset in range(start, end):
            if self.text[offset] == "\n":
                if line_boxes:
                    lines.append(enclose(line_boxes))
                    line_boxes.clear()
            elif (box := self.character_boxes[offset]) is not None:
                line_boxes.append(box)
        if line_boxes:
            lines.append(enclose(line_boxes))

        return SourceSpan(self.number, start, end, self.text[start:end], tuple(lines))


@dataclass(frozen=True)
class Provision:
    kind: Literal["clause", "numbered_paragraph"]
    label: str
    spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class ParsedDocument:
    file_sha256: str
    suggested_type: DocumentType
    pages: tuple[Page, ...]
    provisions: tuple[Provision, ...]


def enclose(boxes: list[Rectangle]) -> Rectangle:
    left, bottom, right, top = zip(*boxes)
    return min(left), min(bottom), max(right), max(top)


def read_page(page: pdfium.PdfPage, number: int, label: str | None) -> Page:
    characters = []
    boxes: list[Rectangle | None] = []
    missing_geometry = False
    previous = ""
    with closing(page.get_textpage()) as text_page:
        for index in range(text_page.count_chars()):
            codepoint = raw.FPDFText_GetUnicode(text_page.raw, index)
            if not codepoint:
                if raw.FPDFText_HasUnicodeMapError(text_page.raw, index) == 1:
                    characters.append("\ufffd")
                    boxes.append(None)
                    missing_geometry = True
                continue

            char = chr(codepoint)
            if char == "\n" and previous == "\r":
                previous = char
                continue
            previous = char
            characters.append("\n" if char == "\r" else char)

            box = None
            if not char.isspace():
                try:
                    left, bottom, right, top = text_page.get_charbox(index, loose=True)
                    if right > left and top > bottom:
                        box = (left, bottom, right, top)
                except pdfium.PdfiumError:
                    pass
                missing_geometry |= box is None
            boxes.append(box)

    text = "".join(characters)
    status: PageStatus = "text_layer"
    if not text.strip():
        status = "no_text_layer"
    elif missing_geometry:
        status = "text_layer_with_gaps"
    return Page(number, label, page.get_bbox(), page.get_rotation(), status, text, tuple(boxes))


def _document_type(pages: tuple[Page, ...]) -> DocumentType:
    preview = "\n".join(page.text for page in pages[:2])[:4000]
    if COURT_CAPTION.search(preview) and PLEADING_TITLE.search(preview):
        return "pleading"
    if AGREEMENT_TITLE.search(preview):
        return "agreement"
    return "unknown"


def _provisions(pages: tuple[Page, ...], document_type: DocumentType) -> tuple[Provision, ...]:
    if document_type == "unknown":
        return ()

    kind: Literal["clause", "numbered_paragraph"] = "clause" if document_type == "agreement" else "numbered_paragraph"
    pattern = DOTTED_NUMBER if kind == "clause" else PARAGRAPH_NUMBER
    results = []
    active_label = None
    active_spans: list[SourceSpan] = []

    for page in pages:
        if page.status == "no_text_layer":
            if active_label and active_spans:
                results.append(Provision(kind, active_label, tuple(active_spans)))
            active_label, active_spans = None, []
            continue

        start = 0
        offset = 0
        for line in page.text.splitlines(keepends=True):
            match = pattern.match(line)
            if match or BOUNDARY.match(line):
                end = start + len(page.text[start:offset].rstrip())
                if active_label and end > start:
                    active_spans.append(page.span(start, end))
                if active_label and active_spans:
                    results.append(Provision(kind, active_label, tuple(active_spans)))
                active_label = match.group("label") if match else None
                active_spans = []
                start = offset
            offset += len(line)

        end = start + len(page.text[start:].rstrip())
        if active_label and end > start:
            active_spans.append(page.span(start, end))

    if active_label and active_spans:
        results.append(Provision(kind, active_label, tuple(active_spans)))
    return tuple(results)


def inspect_pdf(path: Path) -> ParsedDocument:
    source = path.read_bytes()
    pages = []
    with pdfium.PdfDocument(source) as document:
        for index in range(len(document)):
            with closing(document[index]) as page:
                pages.append(read_page(page, index + 1, document.get_page_label(index) or None))
    page_records = tuple(pages)
    document_type = _document_type(page_records)
    return ParsedDocument(sha256(source).hexdigest(), document_type, page_records, _provisions(page_records, document_type))
