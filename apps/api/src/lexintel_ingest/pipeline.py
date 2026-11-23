"""Prepare a versioned document artifact for indexing. This process never calls an LLM."""

from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile
import json
import subprocess

from lxml import etree

from .layout import extract_layout
from .native_pdf import Page
from .pdf_pages import load_pdf
from .structure import build_structure, classify, extract_metadata, extract_references


class ParseError(Exception):
    """A file cannot be safely or faithfully parsed."""


def _text_page(path: Path):
    text = path.read_text(encoding="utf-8-sig").replace("\0", "\ufffd")
    return Page(1, None, (0, 0, 0, 0), 0, "text_layer", text, (None,) * len(text))


def _render_docx(source: Path, folder: Path):
    preview = folder / f"{source.stem}.pdf"
    config = (folder / "office-profile").as_uri()
    with ZipFile(source) as archive:
        if any(name.lower().endswith("vbaproject.bin") for name in archive.namelist()):
            raise ParseError("Macro-enabled Word files are unsupported")
        xml = etree.fromstring(
            archive.read("word/document.xml"),
            parser=etree.XMLParser(resolve_entities=False, no_network=True),
        )
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        revisions = int(xml.xpath("count(//w:ins | //w:del | //w:moveFrom | //w:moveTo)", namespaces=ns))
        numbering = "word/numbering.xml" in archive.namelist()

    try:
        subprocess.run(
            ["soffice", f"-env:UserInstallation={config}", "--headless", "--norestore",
             "--convert-to", "pdf:writer_pdf_Export", "--outdir", str(folder), str(source.resolve())],
            capture_output=True, check=True, timeout=100,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ParseError("Word preview conversion failed") from error
    if not preview.is_file():
        raise ParseError("Word preview conversion did not produce a PDF")
    return preview, {"numbering_definitions_present": numbering, "tracked_revision_elements": revisions}


def _table_chunks(tables, title):
    chunks = []
    for table in tables:
        header_cells = [cell for cell in table["cells"] if cell["column_header"]]
        for row in range(table["rows"]):
            cells = [cell for cell in table["cells"]
                     if cell["row"] == row and not cell["column_header"] and cell["source"]]
            if not cells:
                continue
            lines, evidence = [], []
            for cell in cells:
                # A merged header can apply to multiple columns.
                headers = [
                    header for header in header_cells
                    if header["source"] and header["row"] <= row
                    and header["column"] <= cell["column"] < header["column"] + header["column_span"]
                ]
                label = " / ".join(h["source"]["text"] for h in headers) or f"Column {cell['column'] + 1}"
                lines.append(f"{label}: {cell['source']['text']}")
                evidence.extend(h["source"] for h in headers)
                evidence.append(cell["source"])
            for footnote in table["footnotes"]:
                if any(cell["text"].rstrip().endswith(footnote["marker"]) for cell in cells):
                    evidence.extend(footnote["sources"])
                    lines.append(footnote["text"])
            # Keep first occurrence of shared header or footnote sources.
            unique = {(src["page_number"], src["start"], src["end"]): src for src in evidence}
            label = table["caption"] or f"Table {table['id']}"
            content = "\n".join(lines)
            chunks.append({
                "kind": "table_row", "section_name": label, "node_id": None,
                "table_id": table["id"], "row": row,
                "content": content, "search_text": "\n".join(x for x in (title, label, content) if x),
                "source_spans": list(unique.values()),
            })
    return chunks


def parse_document(source: Path, kind: str, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    preview = None
    rendition = source
    native_properties = {}
    warnings = []
    if kind == "docx":
        preview, native_properties = _render_docx(source, output)
        rendition = preview
        if native_properties["tracked_revision_elements"]:
            warnings.append({"code": "word_tracked_changes_rendered_view"})
    if kind == "txt":
        pages, modes = [_text_page(source)], ["plain_text"]
    else:
        pages, modes, page_warnings = load_pdf(rendition)
        warnings.extend(page_warnings)

    if not any(page.text.strip() for page in pages):
        raise ParseError("The document has no readable text after extraction")

    suggested_type = classify(pages, [])["type"]
    blocks, tables = [], []
    if kind != "txt":
        try:
            blocks, tables, layout_warnings = extract_layout(rendition, pages, any(m in {"ocr", "mixed"} for m in modes))
            warnings.extend(layout_warnings)
        except Exception as error:
            warnings.append({"code": "layout_unavailable", "error_type": type(error).__name__})
    profile = extract_metadata(classify(pages, blocks), pages)
    profile["suggested_type"] = suggested_type
    if suggested_type != "unknown" and profile["type"] != suggested_type:
        warnings.append({"code": "document_type_changed_after_layout"})
    profile["page_count"] = len(pages) if kind != "txt" else None
    profile["page_modes"] = sorted(set(modes))
    unreadable_pages = [page.number for page, mode in zip(pages, modes, strict=True) if mode == "unreadable"]
    profile["has_unreadable_pages"] = bool(unreadable_pages)
    if unreadable_pages:
        for field in profile["metadata"].values():
            if field["status"] == "not_found":
                field.update(status="unreadable", unreadable_pages=unreadable_pages)
    profile["has_tables"] = bool(tables) if kind != "txt" and not any(
        warning["code"] in {"layout_unavailable", "layout_partial"} for warning in warnings
    ) else None
    profile["file_type"] = kind
    profile["native_properties"] = native_properties

    sections, chunks = build_structure(pages, profile, blocks)
    profile["components"] = [
        {"node_id": node["id"], "role": node["kind"], "label": node["label"], "source": node["source"]}
        for node in sections if node["kind"] in {"schedule", "exhibit", "annex", "annexure", "appendix"}
    ]
    chunks.extend(_table_chunks(tables, profile["metadata"]["title"]["value"]))
    if not chunks:
        raise ParseError("No indexable passages were found")
    for chunk in chunks:
        for source_span in chunk["source_spans"]:
            page = pages[source_span["page_number"] - 1]
            start, end = source_span["start"], source_span["end"]
            if not 0 <= start < end <= len(page.text) or page.text[start:end] != source_span["text"]:
                raise ParseError("An extracted passage could not be matched to its source")
            if kind == "txt":
                source_span["line_start"] = page.text.count("\n", 0, start) + 1
                source_span["line_end"] = page.text.count("\n", 0, end) + 1

    return {
        "schema_version": 1,
        "profile": profile,
        "rendition": "rendered_pdf" if preview else "original",
        "coordinate_system": "PDF user space (left, bottom, right, top)" if kind != "txt" else None,
        "pages": [
            {"number": page.number, "label": page.label, "bounds": page.bounds,
             "rotation": page.rotation, "status": page.status, "text": page.text,
             "mode": mode, "character_boxes": [
                [round(coordinate, 3) for coordinate in box] if box else None for box in page.character_boxes
            ] if kind != "txt" else []}
            for page, mode in zip(pages, modes, strict=True)
        ],
        "layout_blocks": blocks, "sections": sections, "tables": tables,
        "references": extract_references(pages, sections, profile["type"]),
        "chunks": chunks, "warnings": warnings,
        "preview_file": preview.name if preview else None,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("kind", choices=("pdf", "docx", "txt"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        artifact = parse_document(args.source, args.kind, args.output)
    except (ParseError, ValueError) as error:
        # Internal parsing exceptions can contain document text. Only expose
        # messages raised by our own validation checks.
        reason = str(error) if isinstance(error, ParseError) else "Document validation failed"
        (args.output / "failure.json").write_text(json.dumps({"detail": reason}), encoding="utf-8")
        raise SystemExit(2) from None
    (args.output / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
