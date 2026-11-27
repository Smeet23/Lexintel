import unittest
from datetime import date
from pathlib import Path

from lexintel_ingest.sci_scraper import (
    Judgment,
    JudgmentScraper,
    ScrapeError,
    _parse_date,
    parse_results,
    split_window,
)


RESULTS = """
<div class="distTableContent">
  <table>
    <thead><tr><th>Serial Number</th><th>Diary Number</th><th>Case Number</th>
    <th>Petitioner / Respondent</th><th>Advocate</th><th>Bench</th><th>Judgment By</th><th>Judgment</th></tr></thead>
    <tbody>
      <tr id="abc" data-diary-no="531/2025">
        <td>1</td>
        <td>531/2025</td>
        <td>C.A. No.-006395-006397 - 2025</td>
        <td class="petitioners text-center"><div>A JOHN KENNEDY</div><div>VS</div><div>STATE OF TAMIL NADU</div></td>
        <td><p class="respondents"><div>PRASANNA S.- 2919</div><div><br> <center> __ </center> <br></div></p></td>
        <td></td>
        <td>HON'BLE MR. JUSTICE SANDEEP MEHTA</td>
        <td>
          <a href="https://api.sci.gov.in/supremecourt/2025/531/531_2025_14_1501_76171_Judgement_06-Nov-2026.pdf">06-11-2026(English) <br></a><br/>
          <a href="https://api.sci.gov.in/supremecourt/2025/531/531_2025_14_1501_76171_Judgement_06-Nov-2026.pdf">2026 INSC 1204(English) <br></a><br/>
          <a href="https://api.sci.gov.in/"></a><br/>
        </td>
      </tr>
      <tr id="def" data-diary-no="1966/2022">
        <td>2</td>
        <td>1966/2022</td>
        <td>C.A. No.-000894 - 2022</td>
        <td><div>TWINSTAR TECHNOLOGIES LIMITED</div><div>VS</div><div>BANK OF MAHARASHTRA</div></td>
        <td><p class="respondents"><div>KARUN MEHTA- 1972</div></p></td>
        <td>HON'BLE MR. JUSTICE PAMIDIGHANTAM SRI NARASIMHA<br> HON'BLE MR. JUSTICE ALOK ARADHE</td>
        <td>HON'BLE MR. JUSTICE ALOK ARADHE</td>
        <td>
          <a href="https://api.sci.gov.in/supremecourt/2022/1966/1966_2022_4_1502_75976_Judgement_16-Nov-2026.pdf">16-11-2026(English) <br></a><br/>
          <a href="https://api.sci.gov.in//pdfdate/index1.php?dt=2026-11-16&dno=19662022&filename=supremecourt/2022/1966/1966_2022_4_1502_75976_Judgement_16-Nov-2026.pdf">2026 INSC 1214(English) <br></a>
        </td>
      </tr>
    </tbody>
  </table>
</div>
"""


class SplitWindowTests(unittest.TestCase):
    def test_splits_long_period_into_inclusive_windows_without_gaps(self):
        windows = list(split_window(date(2024, 1, 1), date(2024, 3, 31)))
        self.assertEqual(windows[0], (date(2024, 1, 1), date(2024, 1, 31)))
        self.assertEqual(windows[1], (date(2024, 2, 1), date(2024, 3, 2)))
        self.assertEqual(windows[-1], (date(2024, 3, 3), date(2024, 3, 31)))
        # Every window fits the service limit of 31 inclusive days.
        self.assertTrue(all((end - start).days + 1 <= 31 for start, end in windows))
        for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
            self.assertEqual((next_start - previous_end).days, 1)

    def test_single_day_and_exact_limit_do_not_split(self):
        self.assertEqual(list(split_window(date(2024, 5, 5), date(2024, 5, 5))), [(date(2024, 5, 5), date(2024, 5, 5))])
        self.assertEqual(list(split_window(date(2024, 5, 1), date(2024, 5, 31))), [(date(2024, 5, 1), date(2024, 5, 31))])

    def test_rejects_reversed_period(self):
        with self.assertRaises(ValueError):
            list(split_window(date(2024, 5, 5), date(2024, 5, 4)))


class ParseResultsTests(unittest.TestCase):
    def test_extracts_parties_bench_date_and_deduplicates_pdf_links(self):
        judgments = parse_results(RESULTS)
        self.assertEqual(len(judgments), 2)

        first = judgments[0]
        self.assertEqual(first.diary_number, "531/2025")
        self.assertEqual(first.case_number, "C.A. No.-006395-006397 - 2025")
        self.assertEqual(first.petitioners, ("A JOHN KENNEDY",))
        self.assertEqual(first.respondents, ("STATE OF TAMIL NADU",))
        self.assertEqual(first.advocates, ("PRASANNA S.- 2919",))
        self.assertEqual(first.bench, ())
        self.assertEqual(first.judgment_date, date(2026, 11, 6))
        self.assertEqual(
            first.pdf_urls,
            ("https://api.sci.gov.in/supremecourt/2025/531/531_2025_14_1501_76171_Judgement_06-Nov-2026.pdf",),
        )

    def test_ignores_the_pdf_viewer_link_and_keeps_bench_lines(self):
        second = parse_results(RESULTS)[1]
        self.assertEqual(
            second.bench,
            ("HON'BLE MR. JUSTICE PAMIDIGHANTAM SRI NARASIMHA", "HON'BLE MR. JUSTICE ALOK ARADHE"),
        )
        self.assertEqual(second.judgment_date, date(2026, 11, 16))
        self.assertFalse(any("/pdfdate/" in url for url in second.pdf_urls))
        self.assertEqual(len(second.pdf_urls), 1)

    def test_skips_header_and_rows_without_a_pdf(self):
        header_only = "<table><thead><tr><th>Serial</th></tr></thead><tbody></tbody></table>"
        self.assertEqual(parse_results(header_only), [])
        no_pdf = RESULTS.replace("https://api.sci.gov.in/supremecourt", "https://example.invalid/supremecourt")
        self.assertEqual(parse_results(no_pdf), [])


class DateParsingTests(unittest.TestCase):
    def test_reads_both_label_formats(self):
        self.assertEqual(_parse_date("06-11-2026(English)"), date(2026, 11, 6))
        self.assertEqual(_parse_date("Judgement 06-Nov-2026"), date(2026, 11, 6))

    def test_returns_none_for_unparseable_text(self):
        self.assertIsNone(_parse_date("2026 INSC 1204(English)"))
        self.assertIsNone(_parse_date("31-02-2026"))


class _StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubClient:
    def __init__(self, page, responses):
        self._page = page
        self._responses = list(responses)
        self.requests = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.requests.append(("GET", url))
        return _StubResponse({"text": self._page})

    def post(self, url, data=None, **kwargs):
        self.requests.append(("POST", url, dict(data or {})))
        return self._responses.pop(0)

    def close(self):
        pass


FORM_PAGE = """
<form id="sciapi-services-judgements-judgement-date">
  <input type="text" name="from_date" id="from_date" value="">
  <input type="hidden" name="scid" id="input_siwp_captcha_id_0" value="abc123">
  <input type="hidden" name="tok_deadbeef" id="tok_deadbeef" value="tok">
  <input type="hidden" name="sci_form_nonce" id="sci_form_nonce" value="nonce1">
  <input type="hidden" name="_wp_http_referer" value="/judgements-judgement-date/">
  <input type="hidden" name="_form_time" value="1795716626">
  <input type="hidden" name="_form_signature" value="sig">
  <input type="hidden" name="es_ajax_request" value="1">
  <input type="text" name="siwp_captcha_value" value="">
</form>
"""


def success_response(html):
    return _StubResponse({"success": True, "data": {"pagination": False, "resultsHtml": html}})


class ScraperTests(unittest.TestCase):
    def test_search_submits_form_fields_and_parses_results(self):
        client = _StubClient(FORM_PAGE, [success_response(RESULTS)])
        with JudgmentScraper(client=client) as scraper:
            judgments = scraper.search(date(2026, 11, 5), date(2026, 11, 6))

        self.assertEqual(len(judgments), 2)
        method, url, payload = client.requests[-1]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://www.sci.gov.in/wp-admin/admin-ajax.php")
        self.assertEqual(payload["action"], "get_judgements_judgement_date")
        self.assertEqual(payload["from_date"], "05-11-2026")
        self.assertEqual(payload["to_date"], "06-11-2026")
        self.assertEqual(payload["siwp_captcha_value"], ".")
        self.assertEqual(payload["sci_auto_submit"], "1")
        self.assertEqual(payload["sci_form_nonce"], "nonce1")
        self.assertEqual(payload["_form_signature"], "sig")

    def test_search_rejects_window_at_or_over_the_limit(self):
        with JudgmentScraper(client=_StubClient(FORM_PAGE, [])) as scraper:
            with self.assertRaises(ValueError):
                scraper.search(date(2026, 1, 1), date(2026, 2, 1))

    def test_search_accepts_a_full_31_day_window(self):
        client = _StubClient(FORM_PAGE, [success_response(RESULTS)])
        with JudgmentScraper(client=client) as scraper:
            scraper.search(date(2026, 1, 1), date(2026, 1, 31))
        self.assertEqual(len(client.requests), 2)  # one page GET, one search POST

    def test_missing_nonce_is_reported_once(self):
        client = _StubClient("<html><body>No form here</body></html>", [])
        with JudgmentScraper(client=client) as scraper:
            with self.assertRaises(ScrapeError):
                scraper.search(date(2026, 11, 5), date(2026, 11, 6))

    def test_service_failure_is_raised_as_scrape_error(self):
        failure = _StubResponse({"success": False, "data": '{"message":"Nonce Verification Failed."}'})
        client = _StubClient(FORM_PAGE, [failure])
        with JudgmentScraper(client=client) as scraper:
            with self.assertRaises(ScrapeError):
                scraper.search(date(2026, 11, 5), date(2026, 11, 6))

    def test_download_skips_existing_file_and_writes_pdf(self):
        judgment = Judgment(
            record_id="abc",
            diary_number="531/2025",
            case_number="C.A. No.-0001 - 2025",
            petitioners=("A",),
            respondents=("B",),
            advocates=(),
            bench=(),
            judgment_by="J",
            judgment_date=date(2026, 11, 6),
            pdf_urls=("https://api.sci.gov.in/supremecourt/2025/531/x.pdf",),
        )
        pdf = _StubResponse(b"%PDF-1.7 body")
        pdf.headers = {"content-type": "application/pdf"}
        pdf.content = b"%PDF-1.7 body"
        client = _StubClient(FORM_PAGE, [pdf])
        client.get = lambda url, **kwargs: pdf  # type: ignore[method-assign]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with JudgmentScraper(client=client) as scraper:
                path = scraper.download(judgment, out)
                self.assertTrue(path is not None and path.read_bytes().startswith(b"%PDF"))

                second = scraper.download(judgment, out)
                self.assertEqual(second, path)

    def test_download_rejects_non_pdf_response(self):
        judgment = Judgment(
            record_id="abc",
            diary_number="531/2025",
            case_number="C.A. No.-0001 - 2025",
            petitioners=("A",),
            respondents=("B",),
            advocates=(),
            bench=(),
            judgment_by="J",
            judgment_date=date(2026, 11, 6),
            pdf_urls=("https://api.sci.gov.in/supremecourt/2025/531/x.pdf",),
        )
        not_pdf = _StubResponse(b"<html>select name from x</html>")
        not_pdf.headers = {"content-type": "text/html"}
        not_pdf.content = b"<html>select name from x</html>"
        client = _StubClient(FORM_PAGE, [not_pdf])
        client.get = lambda url, **kwargs: not_pdf  # type: ignore[method-assign]

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with JudgmentScraper(client=client) as scraper:
                self.assertIsNone(scraper.download(judgment, Path(tmp)))
                self.assertEqual(list(Path(tmp).iterdir()), [])


class RecordTests(unittest.TestCase):
    def test_to_dict_round_trips_dates_and_lists(self):
        judgment = Judgment(
            record_id="abc",
            diary_number="531/2025",
            case_number="C.A. No.-0001 - 2025",
            petitioners=("A",),
            respondents=("B",),
            advocates=("C",),
            bench=("D",),
            judgment_by="J",
            judgment_date=date(2026, 11, 6),
            pdf_urls=("https://api.sci.gov.in/x.pdf",),
        )
        payload = judgment.to_dict()
        self.assertEqual(payload["judgment_date"], "2026-11-06")
        self.assertEqual(payload["petitioners"], ["A"])
        self.assertEqual(payload["pdf_urls"], ["https://api.sci.gov.in/x.pdf"])


if __name__ == "__main__":
    unittest.main()
