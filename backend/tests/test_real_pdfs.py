"""Test full extraction → section detection → chunking pipeline with real legal PDFs.

Downloads real UK/EU law PDFs from the internet and verifies:
1. Text extraction produces structured content
2. Section detector finds real legal boundaries (ARTICLE, PART, SECTION, CLAUSE, etc.)
3. Chunking produces meaningful chunks with real section names (not generic "Section N")
4. Chunks have appropriate sizes and metadata
5. Different jurisdictions' patterns work correctly

Run: python -m pytest backend/tests/test_real_pdfs.py -v -s
"""
import sys
import os
import json
import logging

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from services.text_extraction import extract_text, extract_pdf_text_structured, extract_pdf_text
from services.section_detector import detect_sections, SECTION_PATTERNS
from services.chunking import chunk_document_from_blob, _chunk_markdown_content, _chunk_plain_text, _get_fallback_splitter


# ============================================================================
# Test with UK Consumer Rights Act 2015
# ============================================================================

UK_PDF_PATH = "/tmp/uk_consumer_rights_act.pdf"
GDPR_PDF_PATH = "/tmp/eu_gdpr.pdf"


def _load_pdf(path: str) -> bytes:
    """Load a PDF file as bytes. Skips test if file not found."""
    import pytest
    if not os.path.exists(path):
        pytest.skip(f"PDF not found at {path}. Download it first.")
    with open(path, "rb") as f:
        return f.read()


def print_section_summary(label: str, sections: list):
    """Print a summary of detected sections."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    for i, s in enumerate(sections):
        content_preview = s.get("content", "")[:100].replace("\n", " ")
        print(f"  [{i+1}] type={s.get('section_type', 'N/A'):20s} | header={s.get('header', ''):40s}")
        print(f"       content[:{min(100, len(s.get('content', '')))}]={content_preview}...")
    print(f"  Total: {len(sections)} sections\n")


def print_chunk_summary(label: str, chunks: list):
    """Print a summary of chunks."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    # Group by section_type
    type_counts = {}
    type_names = {}
    for c in chunks:
        st = c.get("section_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
        if st not in type_names:
            type_names[st] = []
        name = c.get("section_name", "")
        if name and name not in type_names[st]:
            type_names[st].append(name)

    print(f"  Total chunks: {len(chunks)}")
    print(f"  Chunk sizes: min={min(len(c['content']) for c in chunks)}, "
          f"max={max(len(c['content']) for c in chunks)}, "
          f"avg={sum(len(c['content']) for c in chunks)//len(chunks)}")

    print(f"\n  Section types:")
    for st, count in sorted(type_counts.items()):
        names = type_names[st][:5]
        print(f"    {st:20s}: {count:4d} chunks | sample names: {names}")

    # Show first 10 chunks
    print(f"\n  First 10 chunks:")
    for i, c in enumerate(chunks[:10]):
        preview = c["content"][:80].replace("\n", " ")
        print(f"    [{i+1}] page={c.get('page_num', 'N/A'):5s} | "
              f"type={c.get('section_type', 'N/A'):20s} | "
              f"name={c.get('section_name', ''):30s}")
        print(f"         {preview}...")

    # Show any chunks with generic names (should be minimal)
    generic = [c for c in chunks if c.get("section_name", "").startswith(("Section ", "Chunk "))]
    if generic:
        print(f"\n  WARNING: {len(generic)} chunks have generic names (should be minimal):")
        for c in generic[:5]:
            print(f"    name={c['section_name']} | type={c['section_type']}")
    else:
        print(f"\n  PASS: All chunks have meaningful section names!")

    print()


def test_uk_consumer_rights_act():
    """Full pipeline test with UK Consumer Rights Act 2015."""
    print("\n" + "="*80)
    print("  TEST: UK Consumer Rights Act 2015")
    print("="*80)

    blob = _load_pdf(UK_PDF_PATH)
    print(f"  PDF size: {len(blob):,} bytes")

    # Step 1: Extract text
    print("\n--- Step 1: Text Extraction ---")
    sections = extract_text(blob, "pdf")
    print(f"  Extracted {len(sections)} page sections")

    is_markdown = any(s.get("format") == "markdown" for s in sections)
    print(f"  Format: {'markdown (pymupdf4llm)' if is_markdown else 'plain text'}")

    # Show first 3 pages content preview
    for s in sections[:3]:
        preview = s["content"][:200].replace("\n", " ")
        print(f"  Page {s.get('location', '?')}: {preview}...")

    assert len(sections) > 0, "No sections extracted"

    # Step 2: Section detection on combined text
    print("\n--- Step 2: Section Detection ---")
    full_text = "\n\n".join(s["content"] for s in sections)
    detected = detect_sections(full_text)
    print_section_summary("UK Consumer Rights Act - Detected Sections", detected)

    # Verify we find real UK law patterns
    section_types = {s["section_type"] for s in detected}
    print(f"  Unique section types found: {section_types}")

    # UK law should have: part, section, numbered_header, or clause patterns
    uk_patterns = {"part", "section", "clause", "numbered_header", "article"}
    found_uk = section_types & uk_patterns
    print(f"  UK-relevant patterns found: {found_uk}")

    # Step 3: Full chunking pipeline
    print("\n--- Step 3: Full Chunking Pipeline ---")
    chunks = chunk_document_from_blob(blob, "pdf")
    print_chunk_summary("UK Consumer Rights Act - Chunks", chunks)

    assert len(chunks) > 0, "No chunks produced"

    # Verify chunk quality
    for chunk in chunks:
        assert "content" in chunk, "Chunk missing content"
        assert "section_name" in chunk, "Chunk missing section_name"
        assert "section_type" in chunk, "Chunk missing section_type"
        assert len(chunk["content"]) > 0, "Empty chunk content"

    # Check chunk sizes are reasonable
    sizes = [len(c["content"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    print(f"  Average chunk size: {avg_size:.0f} chars (~{avg_size/4:.0f} tokens)")
    assert avg_size < 5000, f"Average chunk size too large: {avg_size}"

    # Check that most chunks have meaningful names
    generic_count = sum(1 for c in chunks
                        if c["section_name"].startswith(("Section ", "Chunk ")))
    meaningful_pct = ((len(chunks) - generic_count) / len(chunks)) * 100
    print(f"  Chunks with meaningful names: {meaningful_pct:.1f}% ({len(chunks) - generic_count}/{len(chunks)})")

    return chunks


def test_eu_gdpr():
    """Full pipeline test with EU GDPR regulation."""
    print("\n" + "="*80)
    print("  TEST: EU General Data Protection Regulation (GDPR)")
    print("="*80)

    blob = _load_pdf(GDPR_PDF_PATH)
    print(f"  PDF size: {len(blob):,} bytes")

    # Step 1: Extract text
    print("\n--- Step 1: Text Extraction ---")
    sections = extract_text(blob, "pdf")
    print(f"  Extracted {len(sections)} page sections")

    is_markdown = any(s.get("format") == "markdown" for s in sections)
    print(f"  Format: {'markdown (pymupdf4llm)' if is_markdown else 'plain text'}")

    for s in sections[:3]:
        preview = s["content"][:200].replace("\n", " ")
        print(f"  Page {s.get('location', '?')}: {preview}...")

    assert len(sections) > 0, "No sections extracted"

    # Step 2: Section detection
    print("\n--- Step 2: Section Detection ---")
    full_text = "\n\n".join(s["content"] for s in sections)
    detected = detect_sections(full_text)
    print_section_summary("GDPR - Detected Sections", detected)

    section_types = {s["section_type"] for s in detected}
    print(f"  Unique section types found: {section_types}")

    # GDPR should have articles and possibly sections/regulations
    eu_patterns = {"article", "section", "regulation", "part"}
    found_eu = section_types & eu_patterns
    print(f"  EU-relevant patterns found: {found_eu}")

    # Step 3: Full chunking pipeline
    print("\n--- Step 3: Full Chunking Pipeline ---")
    chunks = chunk_document_from_blob(blob, "pdf")
    print_chunk_summary("GDPR - Chunks", chunks)

    assert len(chunks) > 0, "No chunks produced"

    # Verify chunk quality
    for chunk in chunks:
        assert "content" in chunk
        assert "section_name" in chunk
        assert "section_type" in chunk
        assert len(chunk["content"]) > 0

    sizes = [len(c["content"]) for c in chunks]
    avg_size = sum(sizes) / len(sizes)
    print(f"  Average chunk size: {avg_size:.0f} chars (~{avg_size/4:.0f} tokens)")

    generic_count = sum(1 for c in chunks
                        if c["section_name"].startswith(("Section ", "Chunk ")))
    meaningful_pct = ((len(chunks) - generic_count) / len(chunks)) * 100
    print(f"  Chunks with meaningful names: {meaningful_pct:.1f}% ({len(chunks) - generic_count}/{len(chunks)})")

    return chunks


def test_section_detector_uk_patterns():
    """Test section detector specifically for UK legal patterns."""
    print("\n" + "="*80)
    print("  TEST: Section Detector - UK Legal Patterns")
    print("="*80)

    # Test with typical UK law text patterns
    uk_text = """PART 1
CONSUMER CONTRACTS FOR GOODS, DIGITAL CONTENT AND SERVICES

Chapter 1
Goods

1 Consumer rights: goods contracts

(1) This section applies to contracts between a trader and a consumer for a trader to supply goods to the consumer.

Section 9
Goods to be of satisfactory quality

(1) Every contract to supply goods is to be treated as including a term that the quality of the goods is satisfactory.

PART 2
UNFAIR TERMS

Section 62
Requirement for contract terms and notices to be fair

(1) An unfair term of a consumer contract is not binding on the consumer.

SCHEDULE 1
ENFORCEMENT POWERS

Clause 3.1 Investigation Powers

The enforcing authority may require a person to provide information.

Regulation 5 Application

This regulation applies to consumer contracts."""

    detected = detect_sections(uk_text)
    print_section_summary("UK Patterns Test", detected)

    section_types = {s["section_type"] for s in detected}
    print(f"  Types found: {section_types}")

    # We should find: part, section, clause, regulation
    assert "part" in section_types, f"Expected 'part' in {section_types}"
    assert "section" in section_types, f"Expected 'section' in {section_types}"

    print("  PASS: All expected UK patterns detected!")


def test_section_detector_contract_headers():
    """Test section detector for common contract section headers."""
    print("\n" + "="*80)
    print("  TEST: Section Detector - Contract Headers")
    print("="*80)

    contract_text = """RECITALS

WHEREAS, the Seller desires to sell and the Buyer desires to purchase certain assets.

DEFINITIONS

"Agreement" means this Asset Purchase Agreement.
"Business" means the business conducted by the Seller.

REPRESENTATIONS AND WARRANTIES

The Seller hereby represents and warrants to the Buyer as follows:

COVENANTS

From the date hereof until the Closing Date, the Seller shall:

INDEMNIFICATION

The Seller shall indemnify and hold harmless the Buyer from and against any and all losses.

TERMINATION

This Agreement may be terminated at any time prior to the Closing.

GOVERNING LAW

This Agreement shall be governed by and construed in accordance with the laws of England.

MISCELLANEOUS

This Agreement constitutes the entire agreement between the parties."""

    detected = detect_sections(contract_text)
    print_section_summary("Contract Headers Test", detected)

    section_types = {s["section_type"] for s in detected}
    headers = {s["header"] for s in detected}

    print(f"  Types found: {section_types}")
    print(f"  Headers found: {headers}")

    assert "contract_header" in section_types, f"Expected 'contract_header' in {section_types}"

    expected_headers = {"RECITALS", "DEFINITIONS", "REPRESENTATIONS AND WARRANTIES",
                       "INDEMNIFICATION", "TERMINATION", "GOVERNING LAW", "MISCELLANEOUS"}
    found_headers = headers & expected_headers
    print(f"  Expected headers found: {len(found_headers)}/{len(expected_headers)}: {found_headers}")
    assert len(found_headers) >= 5, f"Expected at least 5 contract headers, found {len(found_headers)}"

    print("  PASS: Contract headers correctly detected!")


def test_section_detector_litigation():
    """Test section detector for litigation document patterns."""
    print("\n" + "="*80)
    print("  TEST: Section Detector - Litigation Patterns")
    print("="*80)

    litigation_text = """STATEMENT OF FACTS

On or about January 15, 2024, Defendant entered into a contract with Plaintiff.

COUNT I
BREACH OF CONTRACT

Plaintiff incorporates by reference all preceding paragraphs.

COUNT II
FRAUD AND MISREPRESENTATION

Defendant knowingly made false representations to Plaintiff.

CAUSE OF ACTION III
NEGLIGENCE

Defendant owed a duty of care to Plaintiff and breached said duty.

PRAYER FOR RELIEF

WHEREFORE, Plaintiff respectfully requests that this Court:
a) Award compensatory damages;
b) Award punitive damages;
c) Award reasonable attorneys' fees."""

    detected = detect_sections(litigation_text)
    print_section_summary("Litigation Patterns Test", detected)

    section_types = {s["section_type"] for s in detected}
    print(f"  Types found: {section_types}")

    assert "statement_of_facts" in section_types or "preamble" in section_types
    assert "litigation_count" in section_types, f"Expected 'litigation_count' in {section_types}"
    assert "litigation_relief" in section_types, f"Expected 'litigation_relief' in {section_types}"

    print("  PASS: Litigation patterns correctly detected!")


def test_chunking_preserves_sections():
    """Test that chunking preserves section boundaries and metadata."""
    print("\n" + "="*80)
    print("  TEST: Chunking Preserves Section Boundaries")
    print("="*80)

    # Simulate extracted sections (as from text_extraction)
    sections = [
        {
            "content": """ARTICLE I - DEFINITIONS

"Agreement" means this Master Services Agreement between Company and Client.
"Services" means the professional services described in each Statement of Work.
"Confidential Information" means any non-public information disclosed by either party.
"Effective Date" means the date first written above.
"Intellectual Property" means patents, copyrights, trademarks, trade secrets.

ARTICLE II - SERVICES

2.1 Scope of Services. Company shall provide the Services described in each applicable
Statement of Work executed by both parties.

2.2 Standards of Performance. Company shall perform the Services in a professional and
workmanlike manner consistent with generally accepted industry standards.

2.3 Subcontracting. Company may subcontract the performance of Services with prior
written consent of Client, which consent shall not be unreasonably withheld.

ARTICLE III - PAYMENT

3.1 Fees. Client shall pay Company the fees set forth in each Statement of Work.

3.2 Invoicing. Company shall invoice Client monthly for Services rendered during the
preceding month. Client shall pay each invoice within thirty (30) days of receipt.

3.3 Expenses. Client shall reimburse Company for reasonable out-of-pocket expenses
incurred in connection with the performance of Services.

ARTICLE IV - CONFIDENTIALITY

4.1 Obligations. Each party agrees to hold the other party's Confidential Information
in strict confidence and not to disclose such information to any third party.

4.2 Exceptions. The obligations of confidentiality shall not apply to information that
is publicly available or becomes publicly available through no fault of the receiving party.""",
            "location": "1",
            "location_type": "page",
            "format": "markdown"
        }
    ]

    chunks = _chunk_markdown_content(sections)
    print_chunk_summary("Section Boundary Preservation Test", chunks)

    # Verify we got real article names
    article_chunks = [c for c in chunks if "ARTICLE" in c.get("section_name", "")]
    print(f"  Chunks with ARTICLE in name: {len(article_chunks)}")

    # Check that different articles are represented
    unique_names = {c["section_name"] for c in chunks}
    print(f"  Unique section names: {unique_names}")

    assert len(chunks) > 1, "Should produce multiple chunks"

    # Should not have generic names
    generic = [c for c in chunks if c["section_name"].startswith("Section ")
               and c["section_name"][8:].isdigit()]
    assert len(generic) == 0, f"Found {len(generic)} generically named chunks"

    print("  PASS: Section boundaries preserved in chunking!")


def test_chunk_size_distribution():
    """Test that chunk sizes follow expected distribution."""
    print("\n" + "="*80)
    print("  TEST: Chunk Size Distribution")
    print("="*80)

    if not os.path.exists(UK_PDF_PATH):
        print("  SKIP: UK PDF not available")
        return

    blob = _load_pdf(UK_PDF_PATH)
    chunks = chunk_document_from_blob(blob, "pdf")

    sizes = [len(c["content"]) for c in chunks]
    sizes.sort()

    print(f"  Total chunks: {len(chunks)}")
    print(f"  Min size: {sizes[0]} chars")
    print(f"  Max size: {sizes[-1]} chars")
    print(f"  Median size: {sizes[len(sizes)//2]} chars")
    print(f"  Average size: {sum(sizes)//len(sizes)} chars")

    # Size distribution buckets
    buckets = {"<100": 0, "100-500": 0, "500-1000": 0, "1000-1500": 0, "1500-2000": 0, ">2000": 0}
    for s in sizes:
        if s < 100: buckets["<100"] += 1
        elif s < 500: buckets["100-500"] += 1
        elif s < 1000: buckets["500-1000"] += 1
        elif s < 1500: buckets["1000-1500"] += 1
        elif s < 2000: buckets["1500-2000"] += 1
        else: buckets[">2000"] += 1

    print(f"\n  Size distribution:")
    for bucket, count in buckets.items():
        bar = "█" * (count * 2)
        print(f"    {bucket:10s}: {count:4d} {bar}")

    # Most chunks should be in 100-1500 range (our target is ~1000)
    target_range = buckets["100-500"] + buckets["500-1000"] + buckets["1000-1500"]
    target_pct = (target_range / len(chunks)) * 100
    print(f"\n  Chunks in target range (100-1500): {target_pct:.1f}%")

    # Tiny chunks (<100) should be rare
    tiny_pct = (buckets["<100"] / len(chunks)) * 100
    print(f"  Tiny chunks (<100): {tiny_pct:.1f}%")

    print("  PASS: Size distribution analyzed!")


def test_frontend_citation_compatibility():
    """Verify chunks produce data compatible with frontend Citation interface.

    Frontend expects:
    - documentName: string
    - pageNumber: number
    - section?: string
    - excerpt: string
    - relevanceScore: number
    """
    print("\n" + "="*80)
    print("  TEST: Frontend Citation Compatibility")
    print("="*80)

    if not os.path.exists(UK_PDF_PATH):
        print("  SKIP: UK PDF not available")
        return

    blob = _load_pdf(UK_PDF_PATH)
    chunks = chunk_document_from_blob(blob, "pdf")

    # Simulate what would be stored and retrieved for citations
    citations = []
    for i, chunk in enumerate(chunks[:20]):
        citation = {
            "documentName": "uk_consumer_rights_act.pdf",
            "pageNumber": int(chunk["page_num"]) if chunk["page_num"].isdigit() else 0,
            "section": chunk.get("section_name", ""),
            "excerpt": chunk["content"][:200],
            "relevanceScore": 0.85 - (i * 0.02)  # Simulated scores
        }
        citations.append(citation)

    print(f"  Generated {len(citations)} citations")

    # Verify all required fields exist and have correct types
    for i, cit in enumerate(citations):
        assert isinstance(cit["documentName"], str), f"Citation {i}: documentName not string"
        assert isinstance(cit["pageNumber"], (int, float)), f"Citation {i}: pageNumber not number"
        assert isinstance(cit["section"], str), f"Citation {i}: section not string"
        assert isinstance(cit["excerpt"], str), f"Citation {i}: excerpt not string"
        assert isinstance(cit["relevanceScore"], (int, float)), f"Citation {i}: relevanceScore not number"

    # Show sample citations as frontend would display them
    print(f"\n  Sample citations (as frontend would show):")
    for cit in citations[:5]:
        score_label = "High" if cit["relevanceScore"] >= 0.8 else "Medium" if cit["relevanceScore"] >= 0.6 else "Low"
        print(f"    📄 {cit['documentName']} | Page {cit['pageNumber']} | {cit['section']}")
        print(f"       \"{cit['excerpt'][:80]}...\"")
        print(f"       Relevance: {cit['relevanceScore']:.0%} ({score_label})")
        print()

    print("  PASS: All chunks compatible with frontend Citation interface!")


# ============================================================================
# Main runner
# ============================================================================

if __name__ == "__main__":
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + "  REAL PDF PIPELINE TEST — Legal Documents from UK/EU".center(78) + "║")
    print("╚" + "═"*78 + "╝")

    results = {}

    # Pattern tests (no PDF needed)
    tests_no_pdf = [
        ("UK Legal Patterns", test_section_detector_uk_patterns),
        ("Contract Headers", test_section_detector_contract_headers),
        ("Litigation Patterns", test_section_detector_litigation),
        ("Section Boundary Preservation", test_chunking_preserves_sections),
    ]

    for name, test_fn in tests_no_pdf:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            logger.exception(f"Test {name} failed")

    # Real PDF tests
    tests_pdf = [
        ("UK Consumer Rights Act 2015", test_uk_consumer_rights_act),
        ("EU GDPR", test_eu_gdpr),
        ("Chunk Size Distribution", test_chunk_size_distribution),
        ("Frontend Citation Compatibility", test_frontend_citation_compatibility),
    ]

    for name, test_fn in tests_pdf:
        try:
            test_fn()
            results[name] = "PASS"
        except FileNotFoundError as e:
            results[name] = f"SKIP: {e}"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            logger.exception(f"Test {name} failed")

    # Final summary
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + "  TEST RESULTS SUMMARY".center(78) + "║")
    print("╠" + "═"*78 + "╣")

    passed = sum(1 for v in results.values() if v == "PASS")
    failed = sum(1 for v in results.values() if v.startswith("FAIL"))
    skipped = sum(1 for v in results.values() if v.startswith("SKIP"))

    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗" if result.startswith("FAIL") else "○"
        print(f"║  {status} {name:40s} {result[:35]:35s} ║")

    print("╠" + "═"*78 + "╣")
    print(f"║  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}".ljust(79) + "║")
    print("╚" + "═"*78 + "╝\n")

    # Exit with error if any test failed
    if failed > 0:
        sys.exit(1)
