"""Download Supreme Court of India judgments by judgment date.

The public "Judgment Date" service (https://www.sci.gov.in/judgements-judgement-date/)
is a WordPress form that submits to ``/wp-admin/admin-ajax.php``. That endpoint is
explicitly allowed by the site's ``robots.txt``. A search is capped at a 31 day range,
so a longer period is split into consecutive windows.

Each result row carries a link to the judgment PDF on ``api.sci.gov.in``. Rows may also
link to an ``html`` viewer that wraps the same PDF; only the direct PDF links are used
because the viewer endpoint does not reliably return a document.

The search form is protected by a CAPTCHA that the site's own JavaScript bypasses when
``sci_auto_submit=1`` is sent, so the scraper sends the same flag. The form nonce and
signature are read from the page markup, which is server rendered.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
import json
import logging
from pathlib import Path
import re
import time
from typing import Iterator, Sequence

import httpx

logger = logging.getLogger(__name__)

JUDGEMENT_DATE_URL = "https://www.sci.gov.in/judgements-judgement-date/"
AJAX_URL = "https://www.sci.gov.in/wp-admin/admin-ajax.php"
AJAX_ACTION = "get_judgements_judgement_date"
FORM_ID = "sciapi-services-judgements-judgement-date"
DATE_FORMAT = "%d-%m-%Y"
MAX_WINDOW_DAYS = 31
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
PDF_HOST = "api.sci.gov.in"


class ScrapeError(RuntimeError):
    """The judgment service rejected a request or returned no usable response."""


@dataclass(frozen=True)
class Judgment:
    """One judgment row returned by the judgment date search."""

    diary_number: str
    case_number: str
    petitioners: tuple[str, ...]
    respondents: tuple[str, ...]
    advocates: tuple[str, ...]
    bench: tuple[str, ...]
    judgment_by: str
    judgment_date: date | None
    pdf_urls: tuple[str, ...]
    record_id: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "diary_number": self.diary_number,
            "case_number": self.case_number,
            "petitioners": list(self.petitioners),
            "respondents": list(self.respondents),
            "advocates": list(self.advocates),
            "bench": list(self.bench),
            "judgment_by": self.judgment_by,
            "judgment_date": self.judgment_date.isoformat() if self.judgment_date else None,
            "pdf_urls": list(self.pdf_urls),
        }


@dataclass
class ScrapeStats:
    """Progress counters for one scrape run."""

    windows: int = 0
    judgments: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def _parse_date(value: str) -> date | None:
    """Read a judgment date written as ``DD-MM-YYYY`` or ``DD-Mon-YYYY``."""
    numeric = re.search(r"\b(\d{2})-(\d{2})-(\d{4})\b", value)
    if numeric:
        try:
            return datetime.strptime(numeric.group(0), "%d-%m-%Y").date()
        except ValueError:
            return None
    textual = re.search(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\b", value)
    if textual:
        try:
            return datetime.strptime(textual.group(0), "%d-%b-%Y").date()
        except ValueError:
            return None
    return None


class _FormParser(HTMLParser):
    """Collect hidden inputs and the submit fields of the judgment date form."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, str] = {}
        self._in_form = False
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "form":
            if attributes.get("id") == FORM_ID:
                self._in_form = True
                self._depth = 1
            elif self._in_form:
                self._depth += 1
            return
        if self._in_form and tag == "input":
            name = attributes.get("name")
            if name and attributes.get("type", "text").lower() == "hidden":
                self.fields[name] = attributes.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._in_form:
            self._depth -= 1
            if self._depth <= 0:
                self._in_form = False


class _RowParser(HTMLParser):
    """Extract one judgment record per ``<tr>`` in the results table.

    Block level elements (``div``, ``br``, ``p``) are record separators inside a cell,
    because the service renders each party and each judge in its own block.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._row: dict | None = None
        self._cell: list[str] | None = None
        self._links: list[tuple[str, str]] | None = None
        self._link_text: list[str] | None = None
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._row = {
                "record_id": attributes.get("id", ""),
                "diary_number": (attributes.get("data-diary-no") or "").strip(),
                "cells": [],
            }
            self._cell = []
            self._links = []
        elif tag == "td" and self._row is not None:
            self._cell = []
            self._links = []
        elif tag in ("div", "br", "p") and self._cell is not None:
            self._cell.append("\n")
        elif tag == "a" and self._cell is not None:
            self._href = attributes.get("href") or ""
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._cell is not None and self._link_text is not None:
            if self._href and self._links is not None:
                self._links.append((self._href, "".join(self._link_text).strip()))
            self._href = None
            self._link_text = None
        elif tag in ("div", "p") and self._cell is not None:
            self._cell.append("\n")
        elif tag == "td" and self._row is not None and self._cell is not None:
            self._row["cells"].append("".join(self._cell))
            self._row.setdefault("links", []).extend(self._links or [])
            self._cell = None
            self._links = []
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._link_text is not None:
            self._link_text.append(data)
        if self._cell is not None:
            self._cell.append(data)


def _segments(value: str) -> list[str]:
    """Split a cell into whitespace-normalised block segments."""
    segments = []
    for line in re.split(r"\n+", value):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            segments.append(line)
    return segments


def _split_parties(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a party cell into petitioners and respondents around ``VS``."""
    petitioners: list[str] = []
    respondents: list[str] = []
    target = petitioners
    for segment in _segments(value):
        if segment.upper() == "VS":
            target = respondents
            continue
        target.append(segment)
    return tuple(petitioners), tuple(respondents)


def _parse_advocates(value: str) -> tuple[str, ...]:
    """Read advocate names, ignoring the blank separator blocks."""
    advocates = []
    for segment in _segments(value):
        segment = re.sub(r"^_+\s*|\s*_+$", "", segment).strip()
        if segment and not re.fullmatch(r"[_\s]+", segment):
            advocates.append(segment)
    return tuple(advocates)


def parse_results(html: str) -> list[Judgment]:
    """Parse the results table returned by the judgment date search."""
    parser = _RowParser()
    parser.feed(html)
    judgments: list[Judgment] = []
    for row in parser.rows:
        cells = row["cells"]
        links = row.get("links", [])
        if len(cells) < 8:
            continue
        pdf_urls = tuple(dict.fromkeys(href for href, _ in links if _is_pdf_link(href)))
        if not pdf_urls:
            continue
        petitioners, respondents = _split_parties(cells[3])
        judgments.append(
            Judgment(
                record_id=row["record_id"],
                diary_number=row["diary_number"],
                case_number=re.sub(r"\s+", " ", cells[2]).strip(),
                petitioners=petitioners,
                respondents=respondents,
                advocates=_parse_advocates(cells[4]),
                bench=tuple(_segments(cells[5])),
                judgment_by=re.sub(r"\s+", " ", cells[6]).strip(),
                judgment_date=_judgment_date(links),
                pdf_urls=pdf_urls,
            )
        )
    return judgments


def _judgment_date(links: Sequence[tuple[str, str]]) -> date | None:
    """Read the judgment date from the link labels, which carry ``DD-Mon-YYYY``."""
    for _, label in links:
        parsed = _parse_date(label)
        if parsed:
            return parsed
    return None


def _is_pdf_link(link: str) -> bool:
    return link.lower().endswith(".pdf") and PDF_HOST in link and "/pdfdate/" not in link


def split_window(start: date, end: date, max_days: int = MAX_WINDOW_DAYS) -> Iterator[tuple[date, date]]:
    """Yield inclusive ``(from, to)`` windows of at most ``max_days`` days."""
    if end < start:
        raise ValueError("end date must not be before start date")
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=max_days - 1), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def _backoff_delay(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    return min(cap, base * (2**attempt))


class JudgmentScraper:
    """Search the judgment date service and download judgment PDFs."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        retries: int = 4,
        timeout: float = 60.0,
        politeness_delay: float = 1.0,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        )
        self._retries = retries
        self._politeness_delay = politeness_delay
        self._form_fields: dict[str, str] | None = None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None  # type: ignore[assignment]

    def __enter__(self) -> "JudgmentScraper":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _get(self, url: str, **kwargs: object) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.get(url, **kwargs)
            except httpx.HTTPError as error:
                last_error = error
                logger.warning("GET %s failed (%s), attempt %s", url, error, attempt + 1)
            else:
                if response.status_code < 500:
                    return response
                last_error = ScrapeError(f"GET {url} returned {response.status_code}")
                logger.warning("GET %s returned %s, attempt %s", url, response.status_code, attempt + 1)
            if attempt < self._retries:
                time.sleep(_backoff_delay(attempt))
        raise ScrapeError(f"Unable to reach {url}: {last_error}")

    def _form(self) -> dict[str, str]:
        """Return the hidden form fields, refreshing them once per session."""
        if self._form_fields is None:
            response = self._get(JUDGEMENT_DATE_URL)
            parser = _FormParser()
            parser.feed(response.text)
            if "sci_form_nonce" not in parser.fields:
                raise ScrapeError("Judgment date form nonce was not found on the page")
            self._form_fields = parser.fields
        return dict(self._form_fields)

    def search(self, from_date: date, to_date: date) -> list[Judgment]:
        """Search judgments delivered between two dates (at most 31 days inclusive)."""
        if (to_date - from_date).days >= MAX_WINDOW_DAYS:
            raise ValueError(f"search window must be shorter than {MAX_WINDOW_DAYS} days")
        payload = self._form()
        payload.update(
            {
                "action": AJAX_ACTION,
                "from_date": from_date.strftime(DATE_FORMAT),
                "to_date": to_date.strftime(DATE_FORMAT),
                "siwp_captcha_value": ".",
                "sci_auto_submit": "1",
                "language": "en",
            }
        )
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post(AJAX_URL, data=payload)
            except httpx.HTTPError as error:
                last_error = error
                logger.warning("Search failed (%s), attempt %s", error, attempt + 1)
            else:
                if response.status_code < 500:
                    return self._parse_response(response)
                last_error = ScrapeError(f"Search returned HTTP {response.status_code}")
                logger.warning("Search returned %s, attempt %s", response.status_code, attempt + 1)
            if attempt < self._retries:
                time.sleep(_backoff_delay(attempt))
        raise ScrapeError(f"Unable to search judgments: {last_error}")

    def _parse_response(self, response: httpx.Response) -> list[Judgment]:
        try:
            body = response.json()
        except ValueError as error:
            raise ScrapeError(f"Search returned invalid JSON: {error}") from error
        if not body.get("success"):
            message = body.get("data") or body.get("message") or "Unknown error"
            raise ScrapeError(f"Search failed: {message}")
        data = body.get("data") or {}
        html = data.get("resultsHtml") or ""
        return parse_results(html)

    def iter_judgments(
        self, start: date, end: date, stats: ScrapeStats | None = None
    ) -> Iterator[Judgment]:
        """Yield every judgment in a period, splitting it into valid search windows."""
        stats = stats if stats is not None else ScrapeStats()
        for window_start, window_end in split_window(start, end):
            stats.windows += 1
            try:
                results = self.search(window_start, window_end)
            except ScrapeError as error:
                stats.errors.append(f"{window_start}:{window_end}: {error}")
                logger.error("Window %s to %s failed: %s", window_start, window_end, error)
                continue
            logger.info("Window %s to %s returned %s judgments", window_start, window_end, len(results))
            for judgment in results:
                stats.judgments += 1
                yield judgment
            time.sleep(self._politeness_delay)

    def download(self, judgment: Judgment, directory: Path, stats: ScrapeStats | None = None) -> Path | None:
        """Download the judgment PDF, returning its path or ``None`` when it already exists."""
        directory.mkdir(parents=True, exist_ok=True)
        target = _target_path(judgment, directory)
        if target.exists() and target.stat().st_size > 0:
            if stats is not None:
                stats.skipped += 1
            return target
        url = judgment.pdf_urls[0]
        try:
            response = self._get(url)
        except ScrapeError as error:
            if stats is not None:
                stats.failed += 1
                stats.errors.append(f"{url}: {error}")
            logger.error("Download failed for %s: %s", url, error)
            return None
        content_type = response.headers.get("content-type", "")
        if not response.content.startswith(b"%PDF") or "pdf" not in content_type.lower():
            if stats is not None:
                stats.failed += 1
                stats.errors.append(f"{url}: response is not a PDF")
            logger.error("Skipping %s: response is not a PDF", url)
            return None
        partial = target.with_suffix(target.suffix + ".part")
        partial.write_bytes(response.content)
        partial.replace(target)
        if stats is not None:
            stats.downloaded += 1
        return target

    def scrape(
        self,
        start: date,
        end: date,
        directory: Path,
        download_pdfs: bool = True,
        manifest: Path | None = None,
    ) -> tuple[list[Judgment], ScrapeStats]:
        """Scrape a period into ``directory``, optionally writing a JSON manifest."""
        stats = ScrapeStats()
        judgments: list[Judgment] = []
        seen: set[str] = set()
        for judgment in self.iter_judgments(start, end, stats):
            key = judgment.pdf_urls[0]
            if key in seen:
                continue
            seen.add(key)
            judgments.append(judgment)
            if download_pdfs:
                self.download(judgment, directory, stats)
        if manifest is not None:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "source": JUDGEMENT_DATE_URL,
                        "from_date": start.isoformat(),
                        "to_date": end.isoformat(),
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "judgments": [judgment.to_dict() for judgment in judgments],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return judgments, stats


def _target_path(judgment: Judgment, directory: Path) -> Path:
    filename = judgment.pdf_urls[0].rsplit("/", 1)[-1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    return directory / safe


def _parse_period(value: str) -> date:
    try:
        return datetime.strptime(value, DATE_FORMAT).date()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not a DD-MM-YYYY date") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scraper from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="from_date", required=True, type=_parse_period, help="Start date (DD-MM-YYYY)")
    parser.add_argument("--to", dest="to_date", required=True, type=_parse_period, help="End date (DD-MM-YYYY)")
    parser.add_argument("--out", type=Path, default=Path("judgments"), help="Directory for downloaded PDFs")
    parser.add_argument("--manifest", type=Path, help="Write judgment metadata to this JSON file")
    parser.add_argument("--no-pdf", action="store_true", help="Only collect metadata, do not download PDFs")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between search windows")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    try:
        with JudgmentScraper(politeness_delay=args.delay) as scraper:
            judgments, stats = scraper.scrape(
                args.from_date,
                args.to_date,
                args.out,
                download_pdfs=not args.no_pdf,
                manifest=args.manifest,
            )
    except (ScrapeError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    logger.info(
        "Finished: %s windows, %s judgments, %s downloaded, %s skipped, %s failed",
        stats.windows,
        len(judgments),
        stats.downloaded,
        stats.skipped,
        stats.failed,
    )
    for message in stats.errors:
        logger.error("%s", message)
    return 1 if stats.errors and not judgments else 0


if __name__ == "__main__":
    raise SystemExit(main())
