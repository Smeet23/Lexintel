"""Convert Docling layout and table predictions into source-backed records."""

from dataclasses import asdict
from pathlib import Path
import re

from .config import settings
from .native_pdf import Page, Rectangle


def text_key(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def find_source(page: Page, text: str, bounds: Rectangle | None = None):
    terms = text.split()
    if not terms:
        return None

    matches = list(re.finditer(r"\s+".join(re.escape(term) for term in terms), page.text, re.IGNORECASE))
    if bounds:
        inside = []
        for match in matches:
            points = [
                box for box in page.character_boxes[match.start():match.end()] if box is not None
            ]
            if points and sum(_inside(box, bounds) for box in points) / len(points) >= .65:
                inside.append(match)
        matches = inside
    if len(matches) != 1:
        return None
    match = matches[0]
    return asdict(page.span(match.start(), match.end()))


def _inside(box: Rectangle, bounds: Rectangle) -> bool:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return bounds[0] - 1 <= x <= bounds[2] + 1 and bounds[1] - 1 <= y <= bounds[3] + 1


def _pdf_box(box, page: Page, display_width: float, display_height: float) -> Rectangle:
    top_left = box.to_top_left_origin(display_height)
    left, bottom, right, top = page.bounds
    width, height = right - left, top - bottom

    def to_pdf(x, y):
        u, v = x / display_width, y / display_height
        if page.rotation == 90:
            return left + v * width, bottom + u * height
        if page.rotation == 180:
            return right - u * width, bottom + v * height
        if page.rotation == 270:
            return right - v * width, top - u * height
        return left + u * width, top - v * height

    first = to_pdf(top_left.l, top_left.t)
    second = to_pdf(top_left.r, top_left.b)
    return min(first[0], second[0]), min(first[1], second[1]), max(first[0], second[0]), max(first[1], second[1])


def _location(doc, page_map, provenance, box):
    page = page_map.get(provenance.page_no)
    size = doc.pages[provenance.page_no].size if provenance.page_no in doc.pages else None
    if page is None or size is None or box is None or not size.width or not size.height:
        return page, None
    return page, _pdf_box(box, page, size.width, size.height)


def _item_sources(doc, page_map, item, text: str):
    evidence = []
    for provenance in getattr(item, "prov", []):
        page, bounds = _location(doc, page_map, provenance, provenance.bbox)
        start, end = provenance.charspan
        piece = text[start:end] or text
        if page and (span := find_source(page, piece, bounds)):
            evidence.append(span)
    return evidence


def _table(doc, pages, item, ordinal):
    provenances = item.prov
    cells = []
    missing_locations = False
    for cell in item.data.table_cells:
        # Docling's usual PDF table record is per page. Preserve table fragments
        # when a cell's page cannot be determined independently.
        source = None
        bounds = None
        page_number = None
        if len(provenances) == 1:
            page, bounds = _location(doc, pages, provenances[0], cell.bbox)
            if page:
                page_number = page.number
                source = find_source(page, cell.text, bounds)
        missing_locations |= bool(cell.text.strip()) and source is None
        cells.append({
            "row": cell.start_row_offset_idx, "column": cell.start_col_offset_idx,
            "row_span": cell.row_span, "column_span": cell.col_span,
            "column_header": cell.column_header, "row_header": cell.row_header,
            "row_section": cell.row_section, "text": cell.text,
            "page_number": page_number, "rectangle": bounds, "source": source,
            "status": "grounded" if source else "unresolved" if cell.text.strip() else "empty",
        })

    caption = item.caption_text(doc)
    caption_sources = [
        source for reference in item.captions
        if (caption_item := reference.resolve(doc)) is not None
        for source in _item_sources(doc, pages, caption_item, caption_item.text)
    ]
    return {
        "id": ordinal, "caption": caption, "caption_sources": caption_sources,
        "pages": sorted({prov.page_no for prov in provenances}),
        "rows": item.data.num_rows, "columns": item.data.num_cols,
        "cells": cells, "footnotes": [], "status": "partial" if missing_locations else "grounded",
    }


def extract_layout(path: Path, pages: list[Page], needs_ocr: bool):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import TableItem

    options = PdfPipelineOptions(
        do_ocr=needs_ocr, do_table_structure=True, enable_remote_services=False,
        ocr_options=TesseractCliOcrOptions(lang=settings.ocr_languages.split("+")),
        accelerator_options=AcceleratorOptions(num_threads=2, device=AcceleratorDevice.CPU),
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    conversion = converter.convert(path, raises_on_error=False, max_num_pages=settings.max_pdf_pages)
    if conversion.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
        raise RuntimeError("Layout conversion failed")

    doc = conversion.document
    page_map = {page.number: page for page in pages}
    blocks, tables, footnotes, warnings = [], [], [], []
    if conversion.status == ConversionStatus.PARTIAL_SUCCESS:
        warnings.append({"code": "layout_partial"})

    for item, depth in doc.iterate_items():
        if isinstance(item, TableItem):
            table = _table(doc, page_map, item, len(tables) + 1)
            if table["status"] == "partial":
                warnings.append({"code": "table_cells_not_aligned", "table": table["id"]})
            tables.append(table)
        else:
            text = getattr(item, "text", "")
            if not text:
                continue
            label = item.label.value
            sources = _item_sources(doc, page_map, item, text)
            block = {
                "kind": label, "depth": depth, "text": text,
                "level": getattr(item, "level", None), "sources": sources,
            }
            blocks.append(block)
            if label == "footnote" and sources:
                footnotes.append(block)

    # Record possible continuation only. Identical headers on adjacent pages
    # can also belong to different tables.
    for previous, current in zip(tables, tables[1:]):
        a, b = previous["pages"], current["pages"]
        prior_headers = [text_key(cell["text"]) for cell in previous["cells"] if cell["column_header"]]
        headers = [text_key(cell["text"]) for cell in current["cells"] if cell["column_header"]]
        if a and b and a[-1] + 1 == b[0] and prior_headers and prior_headers == headers:
            current["possible_continuation_of"] = previous["id"]

    for table in tables:
        candidates = list(footnotes)
        for page_number in table["pages"]:
            page = page_map[page_number]
            for match in re.finditer(r"^[ \t]*(?:\*+|†|‡|\[\d+\])\s+[^\n]+", page.text, re.M):
                candidates.append({"text": match[0].strip(), "sources": [asdict(page.span(match.start(), match.end()))]})
        seen = set()
        for footnote in candidates:
            marker = re.match(r"^\s*(\*+|†|‡|\[\d+\])", footnote["text"])
            source = footnote["sources"][0]
            location = (source["page_number"], source["start"], source["end"])
            if location not in seen and marker and source["page_number"] in table["pages"] and any(
                cell["text"].rstrip().endswith(marker[1]) for cell in table["cells"]
            ):
                table["footnotes"].append(footnote | {"marker": marker[1]})
                seen.add(location)
    return blocks, tables, warnings
