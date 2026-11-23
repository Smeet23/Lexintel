import unittest

from lexintel_ingest.native_pdf import Page
from lexintel_ingest.structure import CHUNK_CHARACTERS, build_structure, classify, extract_metadata, extract_references


def page(text, number=1):
    return Page(number, None, (0, 0, 0, 0), 0, "text_layer", text, (None,) * len(text))


class MetadataTests(unittest.TestCase):
    def test_agreement_dates_roles_and_clause_links_are_evidence_backed(self):
        document = page(
            'MASTER SERVICES AGREEMENT\n'
            'Made as of January 15, 2025 between Alpha Ltd ("Supplier") and Beta Ltd ("Customer").\n'
            'Effective Date: February 1, 2025\n'
            'ARTICLE VII TERMINATION\n'
            '7.2 Notice. See Section 7.3(b).\n'
            '7.3 Exceptions.\n'
            '7.3(b) Fraud permits immediate termination.\n'
        )
        profile = extract_metadata(classify([document], []), [document])
        self.assertEqual(profile["type"], "agreement")
        self.assertEqual(profile["metadata"]["agreement_date"]["value"], "2025-01-15")
        self.assertEqual(profile["metadata"]["effective_date"]["value"], "2025-02-01")
        self.assertEqual(profile["metadata"]["parties"]["value"][0]["role"], "supplier")
        nodes, chunks = build_structure([document], profile, [])
        clause = next(node for node in nodes if node["label"] == "7.3(b)")
        self.assertEqual(nodes[clause["parent_id"] - 1]["label"], "7.3")
        references = extract_references([document], nodes, profile["type"])
        self.assertEqual(references[0]["target_node_id"], clause["id"])
        for chunk in chunks:
            for source in chunk["source_spans"]:
                self.assertEqual(document.text[source["start"]:source["end"]], source["text"])

    def test_ambiguous_date_is_preserved_as_evidence(self):
        document = page("SERVICES AGREEMENT\nEffective Date: 03/04/2025\n")
        profile = extract_metadata(classify([document], []), [document])
        effective = profile["metadata"]["effective_date"]
        self.assertEqual(effective["status"], "ambiguous")
        self.assertIsNone(effective["value"])
        self.assertIn("03/04/2025", effective["sources"][0]["text"])

    def test_contract_reference_does_not_turn_the_exhibit_into_a_heading(self):
        document = page("SERVICES AGREEMENT\n7.2 Exhibit A is attached.\n")
        profile = extract_metadata(classify([document], []), [document])
        nodes, _ = build_structure([document], profile, [])
        self.assertEqual([node["label"] for node in nodes], ["7.2"])


class JudgmentChunkTests(unittest.TestCase):
    def test_nested_paragraphs_override_layout_fragment_and_keep_the_order_separate(self):
        first = page(
            '2025 INSC 1067\nReportable\nJUDGMENT\n'
            '5. We consider the law.\n'
            '5.6 A bench in Prathvi Raj\nChauhan (supra) said, “The provision applies.”\n'
            '5.6.1 The same case considered an exception.\n'
        )
        second = page(
            '10. In the result, the order dated\n'
            '29.04.2025 is set aside.\n'
            '10.1 The observations are limited to this appeal.\n',
            2,
        )
        pages = [first, second]
        misplaced_heading = first.text.index("Chauhan")
        blocks = [{
            "kind": "section_header",
            "text": "Chauhan (supra) said, “The provision applies.”",
            "sources": [{"page_number": 1, "start": misplaced_heading}],
        }]
        profile = extract_metadata(classify(pages, []), pages)
        nodes, chunks = build_structure(pages, profile, blocks)

        self.assertEqual([node["label"] for node in nodes], ["5", "5.6", "5.6.1", "10", "10.1"])
        self.assertEqual(nodes[2]["parent_id"], nodes[1]["id"])
        self.assertIsNone(nodes[3]["parent_id"])
        self.assertIn("Prathvi Raj\nChauhan", chunks[2]["content"])
        self.assertIn("2025 INSC 1067", chunks[2]["search_text"])
        self.assertEqual(chunks[4]["section_name"], "Court disposition > Paragraph 10")
        self.assertIn("29.04.2025", chunks[4]["content"])

    def test_long_judgment_paragraph_overlaps_one_sentence_across_pages(self):
        sentences = [
            f"The court considered written submission {number} and reviewed the record."
            for number in range(36)
        ]
        first = page("JUDGMENT\n1. " + " ".join(sentences[:19]))
        second = page(" ".join(sentences[19:]), 2)
        pages = [first, second]
        profile = extract_metadata(classify(pages, []), pages)
        _, chunks = build_structure(pages, profile, [])
        paragraphs = [chunk for chunk in chunks if chunk["kind"] == "numbered_paragraph"]

        self.assertEqual(len(paragraphs), 2)
        previous = paragraphs[0]["content"]
        following = paragraphs[1]["content"]
        shared = [sentence for sentence in sentences if sentence in previous and sentence in following]
        self.assertEqual(len(shared), 1)
        self.assertTrue(previous.endswith(shared[0]))
        self.assertTrue(following.startswith(shared[0]))
        for chunk in paragraphs:
            self.assertLessEqual(len(chunk["content"]), CHUNK_CHARACTERS)
            for span in chunk["source_spans"]:
                self.assertEqual(pages[span["page_number"] - 1].text[span["start"]:span["end"]], span["text"])
        self.assertTrue(all(sentence in previous or sentence in following for sentence in sentences))

    def test_long_sentence_is_not_repeated(self):
        sentence = "The court " + "reviewed a submission and " * 32 + "considered the record."
        pages = [page("JUDGMENT\n1. " + " ".join([sentence] * 3))]
        profile = extract_metadata(classify(pages, []), pages)
        _, chunks = build_structure(pages, profile, [])
        paragraphs = [chunk for chunk in chunks if chunk["kind"] == "numbered_paragraph"]

        self.assertEqual(len(paragraphs), 2)
        self.assertEqual(sum(chunk["content"].count(sentence) for chunk in paragraphs), 3)
