"""Detect legal structure and produce fields and chunks that point to original text."""

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
import re

from .native_pdf import DOTTED_NUMBER, PARAGRAPH_NUMBER, Page

TITLE_PATTERNS = [
    ("amendment", r"(?:\bAMENDMENT\b|\bADDENDUM\b|\bAMENDED AND RESTATED\b)"),
    ("trust", r"(?:\bTRUST (?:AGREEMENT|DEED)\b|\bLAST WILL AND TESTAMENT\b)"),
    ("pleading", r"^(?:CIVIL |AMENDED |FIRST |SECOND )*(?:COMPLAINT|PETITION|ANSWER)(?:$|\s+(?:FOR|TO)\b)"),
    ("motion", r"^(?:MOTION\b|MEMORANDUM OF LAW\b|BRIEF\b)"),
    ("judgment", r"^(?:J\s*U\s*D\s*G\s*[E]?\s*M\s*E\s*N\s*T|O\s*R\s*D\s*E\s*R)(?:$|\s+OF\b)"),
    ("statute", r"^THE .{2,100}\b(?:ACT|RULES|REGULATIONS)(?:[ ,]+\d{4})?$"),
    ("transcript", r"\b(?:DEPOSITION|TRANSCRIPT OF PROCEEDINGS)\b"),
    ("agreement", r"\b(?:AGREEMENT|CONTRACT)$"),
]
DATE = r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})"
DOTTED_REFERENCE = r"\d+(?:\.\d+)*(?:\([\w]+\))*"
HEADING = re.compile(r"^\s*(?P<kind>article|chapter|part|section|rule|count|schedule|exhibit|annex(?:ure)?|appendix)\s+(?P<label>[A-Z0-9][\w.()-]*)(?=\s|:|$)", re.I)
ROMAN_HEADING = re.compile(r"^\s*(?P<label>[IVXLCDM]+)\.\s+[A-Z][A-Z\s&,/-]{3,}$")
SUBCLAUSE = re.compile(r"^\s*(?P<label>\([A-Za-z0-9]+\))[ \t]+(?=\S)")
REFERENCE = re.compile(
    rf"\b(?P<kind>section|clause|article|schedule|exhibit|annex(?:ure)?|appendix)\s+"
    rf"(?P<label>{DOTTED_REFERENCE}|[IVXLCDM]+|[A-Z])(?!\w)", re.I
)
PAGE_MARKER = re.compile(r"^\s*(?:Page\s+\d+(?:\s+of\s+\d+)?|\d{1,4})\s*$", re.I)
JUDGMENT_SUBPARAGRAPH = re.compile(
    r"^\s*(?:¶\s*)?(?!\d{1,2}\.\d{1,2}\.\d{4}\b)(?P<label>\d{1,4}(?:\.\d{1,3})+)\.?[ \t]+(?=\S)"
)
JUDGMENT_HEADINGS = {
    "background", "facts", "factual background", "submissions", "arguments",
    "analysis", "discussion", "findings", "conclusion", "order", "directions",
}
SENTENCE_END = re.compile(r"""[.!?]["”’')\]]*(?=\s+[“‘"'(]*[A-Z0-9])""")
SENTENCE_ABBREVIATION = re.compile(r"\b(?:[A-Z]|No|Nos|Ltd|Inc|Mr|Mrs|Ms|Dr|Sr|Jr|Ors|Anr|v|vs|para)\.$")
CHUNK_CHARACTERS = 1750
JUDGMENT_OVERLAP_CHARACTERS = 250


def _lines(pages):
    for page in pages:
        offset = 0
        for line in page.text.splitlines(keepends=True):
            end = offset + len(line.rstrip())
            if end > offset:
                yield page, offset, end, page.text[offset:end]
            offset += len(line)


def _span(page, start, end):
    return asdict(page.span(start, end))


def _field(value=None, sources=(), method="rule", status=None):
    return {
        "value": value,
        "status": status or ("found" if value is not None else "not_found"),
        "method": method,
        "sources": list(sources),
    }


def classify(pages, blocks):
    preview = list(_lines(pages[:2]))[:75]
    title_hints = [
        (block["text"], block["sources"])
        for block in blocks
        if block["kind"] in {"title", "section_header"} and block["sources"]
        and block["sources"][0]["page_number"] in {1, 2}
    ]
    candidates = [(line.strip(), [_span(page, start, end)])
                  for page, start, end, line in preview if len(line.strip()) <= 125]
    candidates = title_hints[:18] + candidates
    for text, evidence in candidates:
        if len(text) > 125 or len(text.split()) > 16:
            continue
        for kind, pattern in TITLE_PATTERNS:
            if re.search(pattern, text, re.I) and (text.isupper() or len(text.split()) <= 8):
                if kind == "amendment" and not re.match(
                    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|AMENDMENT|ADDENDUM|AMENDED|RESTATEMENT|\d)", text, re.I
                ):
                    continue
                return {"type": kind, "classification_basis": evidence,
                        "metadata": {"title": _field(text, evidence)}}
    subject = re.compile(r"^Subject:[ \t]*(?P<value>.+)$", re.I)
    for page, start, end, line in preview:
        match = subject.match(line)
        if match:
            evidence = [_span(page, start, end)]
            return {"type": "correspondence", "classification_basis": evidence,
                    "metadata": {"title": _field(match["value"].strip(), evidence)}}
    return {"type": "unknown", "classification_basis": [], "metadata": {"title": _field()}}


def _date_value(raw):
    clean = raw.replace(",", "")
    for fmt in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(clean, fmt).date().isoformat()
        except ValueError:
            pass
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", clean)
    if match:
        first, second, year = map(int, match.groups())
        if first > 12 or second > 12 or first == second:
            month, day = (second, first) if first > 12 else (first, second)
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass
    return None


def _capture(pages, pattern, normalize=lambda value: value.strip(), multiple=False):
    found, ambiguous = {}, []
    for page in pages:
        for match in re.finditer(pattern, page.text, re.I | re.M):
            value = normalize(match["value"])
            span = _span(page, match.start(), match.end())
            if value is None:
                ambiguous.append(span)
            else:
                found.setdefault(value, []).append(span)
    if multiple and found:
        return _field(list(found), [span for sources in found.values() for span in sources])
    if len(found) == 1 and not ambiguous:
        value, sources = next(iter(found.items()))
        return _field(value, sources)
    if found or ambiguous:
        return _field(
            sources=[span for sources in found.values() for span in sources] + ambiguous,
            status="ambiguous",
        ) | {"candidates": list(found)}
    return _field()


def _parties(pages, pattern):
    people, sources = [], []
    seen = set()
    for page in pages:
        for match in re.finditer(pattern, page.text, re.I | re.M):
            name, role = match["name"].strip().rstrip(",."), match["role"].lower()
            key = (name.casefold(), role)
            if key in seen or not name:
                continue
            seen.add(key)
            people.append({"name": name, "role": role})
            sources.append(_span(page, match.start(), match.end()))
    return _field(people or None, sources)


def extract_metadata(profile, pages):
    kind = profile["type"]
    first = pages[:3]
    fields = profile["metadata"]
    if kind in {"agreement", "amendment", "trust"}:
        fields["agreement_date"] = _capture(
            first, rf"\b(?:dated|made|entered into)(?:\s+on|\s+as of)?[ \t]+(?P<value>{DATE})", _date_value
        )
        fields["effective_date"] = _capture(
            pages, rf"\b(?:effective(?: date)?(?:\s+(?:is|of|as of|on))?)[ \t:]+(?P<value>{DATE})", _date_value
        )
        if fields["effective_date"]["status"] == "not_found":
            condition = _capture(
                pages, r"\b(?:becomes?|be|is) effective (?P<value>(?:on|upon|when) (?:the |its )?(?:date|execution|signature|signing|receipt|approval)[^\n.;]{0,90})"
            )
            if condition["status"] == "found":
                fields["effective_date"] = _field(sources=condition["sources"], status="conditional")
        fields["governing_law"] = _capture(
            pages, r"\b(?:governed by|construed in accordance with) the laws of[ \t]+(?P<value>[^\n.;]{2,100})"
        )
        fields["forum"] = _capture(
            pages, r"\b(?:submit(?:s)? to|consent(?:s)? to) (?:the )?(?:exclusive )?jurisdiction of[ \t]+(?P<value>[^\n.;]{2,100})"
        )
        fields["signature_dates"] = _capture(
            pages, rf"\b(?:signed on|signature date)[ \t:]+(?P<value>{DATE})", _date_value, multiple=True
        )
        fields["expiration_date"] = _capture(
            pages, rf"\b(?:expiration date|expires on|shall expire on)[ \t:]+(?P<value>{DATE})", _date_value
        )
        fields["parties"] = _parties(
            first,
            r"(?:\bbetween|\band|\bby)[ \t]+(?P<name>[^\n()]{2,85}?)[, \t]+"
            r"\((?:the\s+)?[“\"]?(?P<role>client|customer|provider|supplier|seller|buyer|licensor|licensee|grantor|settlor|trustee|landlord|tenant)[”\"]?\)",
        )
        fields["defined_terms"] = _capture(
            pages, r"[“\"](?P<value>[^”\"\n]{2,65})[”\"][ \t]+(?:means|shall mean|has the meaning)",
            multiple=True,
        )
    if kind in {"pleading", "motion", "judgment", "transcript"}:
        fields["case_number"] = _capture(
            first, r"^\s*(?:Case|Docket|Civil Appeal|Criminal Appeal|Writ Petition)\s+(?:No\.?|Number)[ \t.:-]*(?P<value>[\w()./ -]+)", multiple=True
        )
        fields["court"] = _capture(
            first, r"^\s*(?P<value>(?:IN THE )?(?:UNITED STATES|SUPREME|HIGH|DISTRICT)[ \t][^\n]{0,100}\bCOURT\b[^\n]{0,70})$",
            lambda value: re.sub(r"^IN THE\s+", "", value.strip(), flags=re.I),
        )
        jurisdiction = _capture(
            first, r"^[ \t]*(?P<value>(?:IN THE )?SUPREME COURT OF INDIA|UNITED STATES DISTRICT COURT)\b",
            lambda value: "IN" if "INDIA" in value.upper() else "US",
        )
        fields["jurisdiction"] = jurisdiction | {"method": "recognized_court"}
        fields["neutral_citation"] = _capture(
            first, r"(?P<value>(?:\[\d{4}\]|\b\d{4})[ \t]+(?:INSC|UKSC|EWCA Civ|HCA|SCC|NZSC)[ \t]+\d+)", multiple=True
        )
        if kind == "pleading":
            fields["patent_numbers"] = _capture(
                pages, r"\b(?:U\.S\.[ \t]+)?Patent[ \t]+No\.?[ \t]+(?P<value>\d[\d,]{4,14})",
                lambda value: value.replace(",", ""), multiple=True,
            )
        fields["parties"] = _parties(
            first, r"^[ \t]*(?P<name>[^\n]{2,125}?)(?:\n[ \t]*|[ .]{2,})(?P<role>plaintiffs?|defendants?|appellants?|respondents?|petitioners?)(?:[,.)\s]|$)"
        )
        date_key = {"pleading": "filing_date", "motion": "filing_date", "judgment": "decision_date", "transcript": "session_date"}[kind]
        date_label = {"pleading": "filed", "motion": "filed", "judgment": "decided|pronounced|judgment dated", "transcript": "date|taken on"}[kind]
        fields[date_key] = _capture(pages, rf"\b(?:{date_label})[ \t:]+(?P<value>{DATE})", _date_value)
    if kind == "statute":
        fields["instrument_number"] = _capture(first, r"\b(?P<value>(?:Act|Regulation) No\.?[ \t]+\d+(?: of \d{4})?)")
        fields["enactment_date"] = _capture(first, rf"\b(?:enactment date|enacted on)[ \t:]+(?P<value>{DATE})", _date_value)
        fields["commencement_date"] = _capture(pages, rf"\b(?:commencement date|comes into force on|enforcement date)[ \t:]+(?P<value>{DATE})", _date_value)
    if kind == "correspondence":
        for field, heading in (("sender", "From"), ("recipient", "To"), ("sent_date", "Date|Sent"),
                               ("message_id", "Message-ID"), ("in_reply_to", "In-Reply-To")):
            fields[field] = _capture(pages, rf"^(?:{heading}):[ \t]*(?P<value>[^\n]+)$")
    return profile


def _heading(line, kind, layout_kind):
    if match := HEADING.match(line):
        name = match["kind"].lower()
        label = match["label"].rstrip(".")
        if name in {"article", "chapter", "part", "count", "schedule", "exhibit", "annex", "annexure", "appendix"}:
            # In prose, 'Exhibit A is attached' is a reference, not a heading.
            if line.strip().isupper() or layout_kind in {"title", "section_header"}:
                return name, label, 1
        elif name in {"section", "rule"} and (
            line.strip().isupper() or layout_kind == "section_header" or "." in label
        ):
            return name, label, 2
    if match := ROMAN_HEADING.match(line):
        return "heading", match["label"], 1
    if kind in {"agreement", "amendment", "trust", "statute", "unknown"} and (match := DOTTED_NUMBER.match(line)):
        label = match["label"]
        return "clause", label, 2 + label.count(".") + label.count("(")
    if kind in {"agreement", "amendment", "trust", "statute"} and (match := SUBCLAUSE.match(line)):
        return "subclause", match["label"], None
    if kind == "judgment" and (match := JUDGMENT_SUBPARAGRAPH.match(line)):
        label = match["label"]
        return "numbered_paragraph", label, 6 + label.count(".")
    if kind in {"pleading", "motion", "judgment"} and (match := PARAGRAPH_NUMBER.match(line)):
        return "numbered_paragraph", match["label"], 6
    if kind == "judgment":
        if line.strip().casefold().rstrip(":") in JUDGMENT_HEADINGS:
            return "heading", None, 2
        return None
    if layout_kind == "section_header":
        return "heading", None, 2
    return None


def _node_kind(node_type, line, document_type):
    text = line.casefold()
    if document_type in {"pleading", "motion"}:
        if "prayer for relief" in text or "relief requested" in text:
            return "requested_relief"
        if "allegation" in text or node_type == "count":
            return "allegation"
    if document_type == "judgment":
        number = JUDGMENT_SUBPARAGRAPH.match(line) or PARAGRAPH_NUMBER.match(line)
        if number and re.match(r"in the result\b", line[number.end():], re.I):
            return "court_disposition"
        if "submissions" in text or "arguments" in text:
            return "party_submissions"
        if text.strip() in {"order", "conclusion", "directions"}:
            return "court_disposition"
    return None


def _chunk(pages, title, node_path, spans, *, judgment=False):
    sources = [_span(pages[page_number - 1], start, end) for page_number, start, end in spans]
    content = "\n".join(span["text"] for span in sources)
    section_parts = []
    for node in node_path:
        if judgment and node["kind"] == "numbered_paragraph":
            if node["function"] == "court_disposition":
                section_parts.append("Court disposition")
            section_parts.append(f"Paragraph {node['label']}")
        else:
            section_parts.append(node["heading"])
    section = " > ".join(section_parts)
    return {
        "kind": node_path[-1]["kind"] if node_path else "text",
        "section_name": section or None,
        "node_id": node_path[-1]["id"] if node_path else None,
        "content": content, "search_text": "\n".join(x for x in (title, section, content) if x),
        "source_spans": sources,
    }


def _judgment_chunk_spans(pages, spans, *, overlap=False):
    """Split a judgment block, optionally carrying one sentence across a cut."""
    texts = [pages[page - 1].text[start:end] for page, start, end in spans]
    content = "\n".join(texts)
    if len(content) <= CHUNK_CHARACTERS:
        yield spans
        return

    starts, ends = [], []
    offset = 0
    for part in texts:
        starts.append(offset)
        offset += len(part)
        ends.append(offset)
        offset += 1
    sentence_ends = [
        match.end() for match in SENTENCE_END.finditer(content)
        if not SENTENCE_ABBREVIATION.search(content[max(0, match.start() - 8):match.start() + 1])
    ]
    word_ends = [match.start() for match in re.finditer(r"\s+", content)]
    start = new_start = 0
    while new_start < len(content):
        end = min(start + CHUNK_CHARACTERS, len(content))
        if end < len(content):
            sentence = bisect_right(sentence_ends, end)
            if sentence and sentence_ends[sentence - 1] - start >= CHUNK_CHARACTERS // 2:
                end = sentence_ends[sentence - 1]
            else:
                word = bisect_right(word_ends, end)
                if word and word_ends[word - 1] > new_start:
                    end = word_ends[word - 1]
        first = bisect_right(ends, start)
        last = bisect_left(starts, end)
        yield [
            (page, source_start + max(start - starts[index], 0),
             source_start + min(end, ends[index]) - starts[index])
            for index, (page, source_start, _) in enumerate(spans[first:last], first)
        ]
        new_start = end
        while new_start < len(content) and content[new_start].isspace():
            new_start += 1
        next_start = new_start
        if overlap and new_start < len(content):
            sentence = bisect_left(sentence_ends, end)
            if sentence < len(sentence_ends) and sentence_ends[sentence] == end:
                previous_end = sentence_ends[sentence - 1] if sentence else 0
                sentence_start = previous_end
                while content[sentence_start].isspace():
                    sentence_start += 1
                if start <= sentence_start and new_start - sentence_start <= JUDGMENT_OVERLAP_CHARACTERS:
                    next_start = sentence_start
        start = next_start


def build_structure(pages, profile, blocks, *, judgment_overlap=True):
    kind = profile["type"]
    metadata = profile["metadata"]
    title = metadata["title"]["value"]
    judgment = kind == "judgment"
    if judgment:
        citations = metadata.get("neutral_citation", {}).get("value") or []
        title = " ".join(("Judgment", *citations))
    layout_lines = {}
    for block in blocks:
        if block["kind"] not in {"title", "section_header", "page_header", "page_footer"}:
            continue
        for src in block["sources"]:
            layout_lines[(src["page_number"], src["start"])] = block["kind"]

    nodes, stack, chunks, parts = [], [], [], []
    size = 0

    def flush():
        nonlocal size
        if parts:
            overlap = bool(judgment_overlap and stack and stack[-1]["kind"] == "numbered_paragraph")
            groups = _judgment_chunk_spans(pages, parts, overlap=overlap) if judgment else (parts,)
            chunks.extend(_chunk(pages, title, stack, spans, judgment=judgment) for spans in groups)
            parts.clear()
        size = 0

    previous_page = 0
    for page, start, end, line in _lines(pages):
        if page.number > previous_page + 1:
            flush()
        previous_page = page.number
        offset = start + len(line) - len(line.lstrip())
        hint = layout_lines.get((page.number, offset))
        if hint in {"page_header", "page_footer"} or (page.number is not None and PAGE_MARKER.match(line)):
            continue
        detected = _heading(line, kind, hint)
        if detected:
            flush()
            node_kind, label, level = detected
            if level is None:
                ancestor = next((node for node in reversed(stack) if node["kind"] != "subclause"), None)
                level = ancestor["level"] + 1 if ancestor else 6
            while stack and (
                stack[-1]["level"] >= level
                or (judgment and node_kind == stack[-1]["kind"] == "numbered_paragraph"
                    and not label.startswith(f"{stack[-1]['label']}."))
            ):
                stack.pop()
            node = {
                "id": len(nodes) + 1, "kind": node_kind, "label": label,
                "level": level, "parent_id": stack[-1]["id"] if stack else None,
                "heading": line.strip(), "function": _node_kind(node_kind, line, kind),
                "source": _span(page, start, end),
            }
            nodes.append(node)
            stack.append(node)

        while not judgment and end - start > CHUNK_CHARACTERS:
            stop = page.text.rfind(" ", start, start + CHUNK_CHARACTERS)
            stop = stop if stop > start else start + CHUNK_CHARACTERS
            flush()
            parts.append((page.number, start, stop))
            flush()
            start = stop + int(page.text[stop:stop + 1] == " ")
        if end <= start:
            continue
        if not judgment and size + (end - start) + 1 > CHUNK_CHARACTERS:
            flush()
        if parts and parts[-1][0] == page.number and not page.text[parts[-1][2]:start].strip():
            size += end - parts[-1][2]
            parts[-1] = (page.number, parts[-1][1], end)
        else:
            parts.append((page.number, start, end))
            size += end - start + 1

    flush()
    return nodes, chunks


def extract_references(pages, nodes, document_type):
    key_counts = Counter(node["label"].casefold() for node in nodes if node["label"])
    targets = {node["label"].casefold(): node for node in nodes if node["label"]}
    references = []
    for page in pages:
        for match in REFERENCE.finditer(page.text):
            kind, label = match["kind"].lower(), match["label"]
            normalized = label.casefold()
            target = targets.get(normalized)
            if (
                target and target["source"]["page_number"] == page.number
                and not page.text[target["source"]["start"]:match.start()].strip()
                and target["source"]["start"] <= match.start() < target["source"]["end"]
                and target["heading"].casefold().startswith(match[0].casefold())
            ):
                continue
            following = page.text[match.end():match.end() + 75]
            amendment = document_type == "amendment" and re.match(
                r"[ \t]+(?:of .{1,35} )?is[ \t]+(?:hereby[ \t]+)?(?:amended|replaced|deleted|restated)",
                following, re.I,
            )
            if amendment:
                action = re.search(r"\b(amended|replaced|deleted|restated)\b", amendment[0], re.I)[1].lower()
                relation = {"amended": "amends", "replaced": "replaces", "deleted": "deletes", "restated": "restates"}[action]
            else:
                relation = "refers_to"
            target_id = None
            if target and key_counts[normalized] == 1 and not amendment:
                equivalent = {"section": {"section", "clause"}, "article": {"article"}, "clause": {"clause"}}
                if target["kind"] in equivalent.get(kind, {kind}):
                    target_id = target["id"]
            references.append({
                "relation": relation, "target_kind": kind, "target_label": label,
                "target_node_id": target_id, "source": _span(page, match.start(), match.end() + (len(amendment[0]) if amendment else 0)),
            })
    if document_type == "amendment":
        title_and_date = rf"""(?P<title>[A-Z][A-Za-z' -]{{2,65}}?\b(?:Agreement|Contract))[, \t]+(?:dated|effective)(?:[ \t]+as[ \t]+of)?[ \t]+(?P<date>{DATE})"""
        qualified = re.compile(rf"\b(?:to (?:the )?|that certain |the ){title_and_date}", re.I)
        standalone = re.compile(rf"^[ \t]*{title_and_date}", re.I | re.M)
        for page in pages[:3]:
            matches = list(qualified.finditer(page.text)) or list(standalone.finditer(page.text))
            for match in matches:
                references.append({
                    "relation": "amends_document", "target_title": match["title"].strip(),
                    "target_date": _date_value(match["date"]), "target_node_id": None,
                    "source": _span(page, match.start(), match.end()),
                })
    return references
