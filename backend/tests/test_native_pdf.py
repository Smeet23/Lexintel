import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from reportlab.pdfgen.canvas import Canvas

from lexintel_ingest.native_pdf import inspect_pdf


class NativePdfTests(unittest.TestCase):
    def test_provision_offsets_and_geometry_across_pages(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "agreement.pdf"
            pdf = Canvas(str(path), pagesize=(400, 400))
            pdf.drawString(40, 360, "SERVICES AGREEMENT")
            pdf.drawString(40, 330, "7.2 Notice. Either party may terminate")
            pdf.drawString(40, 310, "with 30 days' written notice.")
            pdf.showPage()
            pdf.drawString(40, 360, "The notice must be delivered by courier.")
            pdf.drawString(40, 330, "7.3 Fees. The customer shall pay the fee.")
            pdf.showPage()
            pdf.rect(40, 40, 100, 100)
            pdf.showPage()
            pdf.drawString(40, 350, "Unclassified text after an unreadable page.")
            pdf.save()

            document = inspect_pdf(path)
            self.assertEqual(document.suggested_type, "agreement")
            self.assertEqual([item.label for item in document.provisions], ["7.2", "7.3"])
            self.assertEqual(document.pages[2].status, "no_text_layer")

            notice = document.provisions[0]
            self.assertEqual([span.page_number for span in notice.spans], [1, 2])
            first_span = notice.spans[0]
            self.assertIn("30 days", first_span.text)
            self.assertEqual(len(first_span.rectangles), 2)
            first_line, second_line = first_span.rectangles
            self.assertAlmostEqual(first_line[0], 40)
            self.assertGreater(first_line[1], second_line[3])
            first_page = document.pages[0]
            self.assertEqual(first_page.text[first_span.start:first_span.end], first_span.text)

            offset = first_page.text.index("30 days")
            phrase = first_page.span(offset, offset + len("30 days"))
            self.assertEqual(phrase.text, "30 days")
            self.assertEqual(len(phrase.rectangles), 1)
            self.assertGreater(phrase.rectangles[0][0], second_line[0])
            self.assertLess(phrase.rectangles[0][2], second_line[2])
            self.assertEqual(len(document.provisions[1].spans), 1)

            result = subprocess.run(
                [sys.executable, "-m", "lexintel_ingest", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["provisions"][0]["spans"][0]["text"], first_span.text)

    def test_pleading_paragraphs_stop_at_count_boundaries(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "complaint.pdf"
            pdf = Canvas(str(path))
            pdf.drawString(40, 700, "UNITED STATES DISTRICT COURT")
            pdf.drawString(40, 680, "COMPLAINT FOR PATENT INFRINGEMENT")
            pdf.drawString(40, 660, "12. Plaintiff alleges infringement.")
            pdf.drawString(40, 640, "COUNT II: SECOND ALLEGATION")
            pdf.drawString(40, 620, "13. Plaintiff seeks damages.")
            pdf.save()

            document = inspect_pdf(path)
            self.assertEqual(document.suggested_type, "pleading")
            self.assertEqual([item.label for item in document.provisions], ["12", "13"])
            first = document.provisions[0]
            self.assertEqual(first.kind, "numbered_paragraph")
            self.assertEqual(first.spans[0].text, "12. Plaintiff alleges infringement.")


if __name__ == "__main__":
    unittest.main()
