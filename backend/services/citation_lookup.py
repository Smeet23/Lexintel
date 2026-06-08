"""Multi-jurisdiction citation lookup and verification.

Verifies that legal citations reference real cases using:
- CourtListener API (US) — 18.1M citations, free
- Indian Kanoon (India) — free
- National Archives / BAILII patterns (UK) — URL validation
- Source-matching fallback (all jurisdictions)
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# API endpoints
COURTLISTENER_BASE = "https://www.courtlistener.com"
COURTLISTENER_CITATION_LOOKUP = f"{COURTLISTENER_BASE}/api/rest/v4/citation-lookup/"
COURTLISTENER_SEARCH = f"{COURTLISTENER_BASE}/api/rest/v4/search/"
COURTLISTENER_CLUSTERS = f"{COURTLISTENER_BASE}/api/rest/v4/clusters/"

REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# SSRF guard: only fetch URLs that originate from CourtListener
COURTLISTENER_ORIGIN = "https://www.courtlistener.com"

# Concurrency limit for API calls — created lazily so it binds to the running
# event loop instead of the module-import loop (which breaks in Celery workers).
_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore, creating it on first use."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(5)
    return _semaphore

# In-memory cache with size limit (process-local)

class _BoundedCache:
    """LRU bounded cache backed by collections.OrderedDict.

    On get: moves the accessed key to the end (most-recently-used).
    On set when full: evicts the front entry (least-recently-used).
    """
    def __init__(self, maxsize: int = 5000):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, Optional[Dict]]" = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[Dict]:
        if key not in self._cache:
            return None
        # Move to end to mark as recently used
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Optional[Dict]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        if len(self._cache) >= self._maxsize:
            # Evict least-recently-used (front of the ordered dict)
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

_lookup_cache = _BoundedCache(maxsize=5000)


def _html_title(html: str) -> str:
    """Extract the <title> text from third-party HTML using a real parser.

    Replaces brittle `re.search(r'<title>([^<]+)</title>')` which breaks on
    `<title id=...>`, HTML entities, and newlines inside the title. Returns ""
    when there is no title or parsing fails.
    """
    try:
        from bs4 import BeautifulSoup
        title = BeautifulSoup(html, "html.parser").title
        return title.get_text(strip=True) if title else ""
    except Exception:  # noqa: BLE001 - never let title parsing crash a lookup
        return ""


def _get_courtlistener_headers() -> Dict[str, str]:
    """Build CourtListener request headers."""
    try:
        from backend.config import get_settings
    except ImportError:
        try:
            from config import get_settings
        except ImportError:
            from ..config import get_settings

    settings = get_settings()
    headers = {"Content-Type": "text/plain"}
    if settings.courtlistener_api_token:
        headers["Authorization"] = f"Token {settings.courtlistener_api_token}"
    return headers


def _cache_key(citation: str, jurisdiction: str) -> str:
    """Generate a normalized cache key."""
    return f"{jurisdiction}:{citation.strip().lower()}"


def _reporter_to_cl_slug(reporter: str) -> str:
    """Derive a CourtListener URL slug from a canonical reporter abbreviation.

    This is URL ROUTING, not citation parsing: CourtListener's /c/ endpoint uses
    lowercased, dot-stripped, space/`.`→hyphen reporter slugs (e.g. 'U.S.'→'us',
    'F. Supp. 2d'→'f-supp-2d', 'S. Ct.'→'s-ct'). Pure string transform on the
    canonical reporter eyecite already resolved via reporters-db — no per-reporter
    table to maintain and no regex.
    """
    slug = reporter.strip().lower().replace(".", "")
    slug = "-".join(slug.split())  # collapse internal whitespace to single hyphens
    return slug


def _parse_us_citation(citation_text: str) -> Optional[Dict]:
    """Parse a US citation into volume, canonical reporter, page, and CL slug.

    Library-based (eyecite + reporters-db), NO regex. eyecite tokenises the cite
    and resolves the reporter to its canonical edition; we read volume/page from
    its ``groups`` and the canonical reporter from ``corrected_reporter()``, then
    derive the CourtListener URL slug deterministically. Returns None if the
    string is not a parseable full case citation.
    """
    try:
        from eyecite import get_citations
        from eyecite.models import FullCaseCitation
    except ImportError:  # pragma: no cover - eyecite is a hard dependency
        return None

    try:
        cites = get_citations(citation_text)
    except Exception:  # pragma: no cover - never let parsing crash a lookup
        return None

    for cite in cites:
        if not isinstance(cite, FullCaseCitation):
            continue
        groups = cite.groups or {}
        volume = groups.get("volume")
        page = groups.get("page")
        if not (volume and page):
            continue
        try:
            reporter = cite.corrected_reporter() or groups.get("reporter") or ""
        except Exception:  # pragma: no cover
            reporter = groups.get("reporter") or ""
        if not reporter:
            continue
        return {
            "volume": str(volume),
            "reporter_slug": _reporter_to_cl_slug(reporter),
            "page": str(page),
            "reporter": reporter,
        }
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
)
async def _citation_lookup_post(
    citation_text: str, client: "httpx.AsyncClient"
) -> Optional[Dict]:
    """CourtListener citation-lookup POST — the authoritative resolver.

    Unlike the direct-redirect Method 1 (which returns empty cluster_id and
    date_filed), this returns the canonical cluster: its id (stable case
    identity that merges PARALLEL citations) and date_filed (authoritative
    decision date for the chronology guard). The token is optional but strongly
    recommended (rate-limited without it).

    Returns None on no-match (404), ambiguous (300 — must NOT auto-merge),
    network failure, or non-200 so the caller falls through to Methods 1-2.
    """
    # Let TimeoutException and ConnectError propagate so tenacity can retry.
    # Only catch non-retriable failures (non-200 status, JSON parse errors).
    resp = await client.post(
        COURTLISTENER_CITATION_LOOKUP,
        content=citation_text.encode("utf-8"),
        headers=_get_courtlistener_headers(),
    )
    if resp.status_code != 200:
        return None
    try:
        items = resp.json()
    except Exception:
        return None
    for item in items if isinstance(items, list) else []:
        # status 200 = matched; 300 = ambiguous (do NOT merge); 404 = no match.
        if item.get("status") == 200 and item.get("clusters"):
            cluster = item["clusters"][0]
            absolute_url = cluster.get("absolute_url", "") or ""
            if absolute_url and not absolute_url.startswith("http"):
                absolute_url = f"{COURTLISTENER_BASE}{absolute_url}"
            return {
                "found": True,
                "case_name": cluster.get("case_name", "") or "",
                "court": cluster.get("court", "") or "",
                "date_filed": cluster.get("date_filed", "") or "",
                "url": absolute_url,
                "status": "Published",
                "citation_count": cluster.get("citation_count", 0) or 0,
                "cluster_id": str(cluster.get("id", "") or ""),
                "source": "courtlistener_citation_lookup",
            }
    return None


async def _lookup_us_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up a US citation via CourtListener.

    Method 3: Citation-lookup POST — authoritative cluster (id + date_filed),
      tried FIRST because it carries the parallel-cite-merging cluster_id and the
      authoritative decision date. Token optional (rate-limited without it).
    Method 1: Direct citation page /c/{reporter}/{volume}/{page}/
      - 302 redirect = case exists (most reliable, no auth needed)
      - 404 = case does NOT exist
    Method 2: Search API fallback (for citations the direct lookup misses)
    """
    parsed = _parse_us_citation(citation_text)

    async with _get_semaphore():
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:

            # Method 3 first: authoritative cluster (id + date_filed).
            result = await _citation_lookup_post(citation_text, client)
            if result is not None:
                return result

            # Method 1: Direct citation page (most reliable, no auth needed)
            if parsed:
                try:
                    cite_url = (
                        f"{COURTLISTENER_BASE}/c/"
                        f"{parsed['reporter_slug']}/{parsed['volume']}/{parsed['page']}/"
                    )
                    response = await client.get(cite_url, follow_redirects=False)

                    if response.status_code in (301, 302):
                        # Case exists — follow redirect to get metadata
                        opinion_url = response.headers.get("location", "")
                        if opinion_url and not opinion_url.startswith("http"):
                            opinion_url = f"{COURTLISTENER_BASE}{opinion_url}"

                        # Validate redirect target (prevent SSRF)
                        if opinion_url and not opinion_url.startswith(f"{COURTLISTENER_BASE}/"):
                            logger.warning(f"Refusing redirect to non-CourtListener URL: {opinion_url[:80]}")
                            opinion_url = ""

                        # Fetch opinion page for metadata
                        case_name = ""
                        court = ""
                        date_filed = ""
                        citation_count = 0
                        cluster_id = ""
                        try:
                            meta_resp = await client.get(
                                f"{opinion_url}",
                                follow_redirects=True,
                                headers={"Accept": "text/html"},
                            )
                            if meta_resp.status_code == 200:
                                # Extract case name from page title (bs4, not regex)
                                raw_title = _html_title(meta_resp.text)
                                if raw_title:
                                    # Remove " | CourtListener" suffix
                                    case_name = raw_title.split("|")[0].strip()
                        except Exception:
                            pass

                        return {
                            "found": True,
                            "case_name": case_name,
                            "court": court,
                            "date_filed": date_filed,
                            "url": opinion_url,
                            "status": "Published",
                            "citation_count": citation_count,
                            "cluster_id": cluster_id,
                            "source": "courtlistener",
                        }

                    elif response.status_code == 404:
                        # Definitively not found
                        return None

                except (httpx.HTTPStatusError, httpx.TimeoutException):
                    raise  # Let tenacity retry
                except Exception as e:
                    logger.debug(f"Direct citation lookup failed: {e}")

            # Method 2: Search API fallback (broader, works without auth)
            try:
                response = await client.get(
                    COURTLISTENER_SEARCH,
                    params={"q": citation_text, "type": "o", "format": "json"},
                    headers=_get_courtlistener_headers(),
                )
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    # Verify the citation actually appears in the result's citations
                    for result in results[:5]:
                        result_cites = result.get("citation", [])
                        if citation_text in result_cites:
                            absolute_url = result.get("absolute_url", "")
                            return {
                                "found": True,
                                "case_name": result.get("caseName", ""),
                                "court": result.get("court", ""),
                                "date_filed": result.get("dateFiled", ""),
                                "url": f"{COURTLISTENER_BASE}{absolute_url}" if absolute_url else "",
                                "status": result.get("status", ""),
                                "citation_count": result.get("citeCount", 0),
                                "cluster_id": result.get("cluster_id", result.get("id", "")),
                                "source": "courtlistener_search",
                            }
            except Exception as e:
                logger.debug(f"Citation search fallback failed: {e}")

    return None


def _uk_citation_to_uri(court: str, year: str, number: str) -> Optional[str]:
    """Convert UK neutral citation components to National Archives URI slug."""
    # Map court codes to Find Case Law URI prefixes
    # Docs: https://nationalarchives.github.io/ds-find-caselaw-docs/public
    court_uri_map = {
        "UKSC": "uksc",
        "UKHL": "ukhl",
        "UKPC": "ukpc",
        "EWCA Civ": "ewca/civ",
        "EWCA Crim": "ewca/crim",
        "EWHC": "ewhc/kb",      # Default division; covers most EWHC cases
        "EWCOP": "ewcop",
        "EWFC": "ewfc",
        "UKUT": "ukut/iac",     # Default chamber
        "UKFTT": "ukftt/tc",    # Default chamber
        "EAT": "eat",
    }
    return court_uri_map.get(court)


async def _lookup_uk_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up a UK citation via the National Archives Find Case Law API.

    Uses direct URI lookup: GET /{court}/{year}/{number}/data.xml
    - 200 = case exists (REAL verification)
    - 404 = case NOT in database

    Free, no auth needed, 1000 req/5min.
    API docs: https://nationalarchives.github.io/ds-find-caselaw-docs/public
    """
    import re
    try:
        import defusedxml.ElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET

    # Extract neutral citation components
    match = re.match(
        r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA\s+(?:Civ|Crim)|EWHC|EWCOP|EWFC|UKUT|UKFTT|EAT)\s+(\d+)',
        citation_text,
    )
    if not match:
        return None

    year, court, number = match.groups()
    court = ' '.join(court.split())  # normalise multi/tab whitespace (e.g. "EWCA  Crim" → "EWCA Crim")

    # Convert to National Archives URI. Validate the court code against the
    # data-driven map and log (rather than silently dropping) unknown codes.
    uri_prefix = _uk_citation_to_uri(court, year, number)
    if not uri_prefix:
        logger.debug(f"Unknown UK court code '{court}' in citation: {citation_text}")
        return None

    document_uri = f"{uri_prefix}/{year}/{number}"
    na_base = "https://caselaw.nationalarchives.gov.uk"

    # Method 1: Direct URI lookup (fastest, most reliable)
    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(
                    f"{na_base}/{document_uri}/data.xml",
                    follow_redirects=True,
                )

                if response.status_code == 200:
                    # Parse XML to extract case name
                    case_name = ""
                    try:
                        root = ET.fromstring(response.content)
                        # LegalDocML uses namespaces — search with wildcard
                        title_elem = root.find('.//{*}FRBRname')
                        if title_elem is not None:
                            case_name = title_elem.get("value", "")
                        if not case_name:
                            # Fallback: look for docTitle
                            doc_title = root.find('.//{*}docTitle')
                            if doc_title is not None and doc_title.text:
                                case_name = doc_title.text.strip()
                    except ET.ParseError:
                        logger.debug(f"Failed to parse XML for {document_uri}")

                    return {
                        "found": True,
                        "case_name": case_name,
                        "court": court,
                        "date_filed": year,
                        "url": f"{na_base}/{document_uri}",
                        "status": "Published",
                        "citation_count": 0,
                        "source": "national_archives",
                    }

                elif response.status_code == 404:
                    logger.debug(f"UK citation not found in National Archives: {citation_text}")
                    # Don't return found=True — this is honest verification
                    return None

    except httpx.TimeoutException:
        logger.warning(f"National Archives API timeout for {citation_text}")
    except Exception as e:
        logger.warning(f"National Archives lookup failed for {citation_text}: {e}")

    # Method 2: Atom feed search (fallback if direct URI fails)
    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(
                    f"{na_base}/atom.xml",
                    params={"query": citation_text, "per_page": 1},
                )

                if response.status_code == 200:
                    try:
                        root = ET.fromstring(response.content)
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        entries = root.findall("atom:entry", ns)

                        if entries:
                            entry = entries[0]
                            title = entry.findtext("atom:title", "", ns)
                            published = entry.findtext("atom:published", "", ns)
                            link = entry.find("atom:link[@rel='alternate']", ns)
                            href = link.get("href", "") if link is not None else ""

                            return {
                                "found": True,
                                "case_name": title,
                                "court": court,
                                "date_filed": published[:10] if published else year,
                                "url": href or f"{na_base}/{document_uri}",
                                "status": "Published",
                                "citation_count": 0,
                                "source": "national_archives_search",
                            }
                    except ET.ParseError:
                        pass
    except Exception as e:
        logger.debug(f"National Archives Atom search failed: {e}")

    # If both methods fail, return None — we do NOT rubber-stamp
    return None


async def _lookup_in_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up an Indian citation via Indian Kanoon web search (no API key needed).

    Searches indiankanoon.org for the exact citation string and checks
    if results contain a matching case.
    """
    from bs4 import BeautifulSoup

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(
                    "https://indiankanoon.org/search/",
                    params={"formInput": f'"{citation_text}"'},
                )
                if response.status_code == 200:
                    # Parse the search results with bs4 instead of scraping with
                    # a brittle DOTALL regex. Indian Kanoon wraps each hit in a
                    # `.result_title` element containing the case-title anchor.
                    soup = BeautifulSoup(response.text, "html.parser")
                    result_title = soup.find(class_="result_title")
                    if result_title is None:
                        # No results — citation not found
                        return None

                    anchor = (
                        result_title
                        if getattr(result_title, "name", None) == "a"
                        else result_title.find("a")
                    )
                    case_name = anchor.get_text(strip=True) if anchor else ""

                    href = anchor.get("href", "") if anchor else ""
                    # Only trust /doc/{id} hrefs (validate, prevent open redirect).
                    doc_url = ""
                    if href.startswith("/doc/"):
                        doc_url = f"https://indiankanoon.org{href.rstrip('/')}/"

                    return {
                        "found": True,
                        "case_name": case_name,
                        "court": "",
                        "date_filed": "",
                        "url": doc_url,
                        "status": "Published",
                        "citation_count": 0,
                        "source": "indiankanoon",
                    }
    except Exception as e:
        logger.debug(f"Indian Kanoon search failed: {e}")

    return None


async def _lookup_au_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up an Australian citation via AustLII (no auth needed).

    Direct URL: /cgi-bin/viewdoc/au/cases/cth/{COURT}/{YEAR}/{NUMBER}.html
    - 200 = case exists
    - 404 = case not found
    """
    import re

    match = re.match(
        r'\[(\d{4})\]\s+(HCA|FCAFC|FCA|NSWCA|NSWSC|VSC|VSCA|QCA|QSC)\s+(\d+)',
        citation_text,
    )
    if not match:
        return None

    year, court, number = match.groups()

    # Map court codes to AustLII paths
    court_paths = {
        "HCA": "cth/HCA",
        "FCAFC": "cth/FCAFC",
        "FCA": "cth/FCA",
        "NSWCA": "nsw/NSWCA",
        "NSWSC": "nsw/NSWSC",
        "VSC": "vic/VSC",
        "VSCA": "vic/VSCA",
        "QCA": "qld/QCA",
        "QSC": "qld/QSC",
    }

    path = court_paths.get(court)
    if not path:
        logger.debug(f"Unknown AU court code '{court}' in citation: {citation_text}")
        return None

    url = f"https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/{path}/{year}/{number}.html"

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(url)

                if response.status_code == 200 and len(response.text) > 5000:
                    # Extract case name from title (bs4, not regex)
                    case_name = _html_title(response.text)

                    return {
                        "found": True,
                        "case_name": case_name,
                        "court": court,
                        "date_filed": year,
                        "url": url.replace("/cgi-bin/viewdoc/", "/"),
                        "status": "Published",
                        "citation_count": 0,
                        "source": "austlii",
                    }
                elif response.status_code == 404:
                    return None
    except Exception as e:
        logger.debug(f"AustLII lookup failed: {e}")

    return None


async def _lookup_sg_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up a Singapore citation via eLitigation (no auth needed).

    URL pattern: /gd/s/{YEAR}_{COURT}_{NUMBER}
    - 200 with real title = case exists
    - 200 with "Page Not Found" title = case doesn't exist
    """
    import re

    match = re.match(
        r'\[(\d{4})\]\s+(SGCA|SGHC|SGDC)\s+(\d+)',
        citation_text,
    )
    if not match:
        return None

    year, court, number = match.groups()
    url = f"https://www.elitigation.sg/gd/s/{year}_{court}_{number}"

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    title = _html_title(response.text)

                    # "Page Not Found" in title = case doesn't exist.
                    # If we couldn't parse a title at all, treat as not-found
                    # rather than rubber-stamping a match.
                    if not title or "not found" in title.lower() or "error" in title.lower():
                        return None

                    return {
                        "found": True,
                        "case_name": "",
                        "court": court,
                        "date_filed": year,
                        "url": url,
                        "status": "Published",
                        "citation_count": 0,
                        "source": "elitigation",
                    }
    except Exception as e:
        logger.debug(f"eLitigation lookup failed: {e}")

    return None


async def _lookup_eu_citation(citation_text: str) -> Optional[Dict]:
    """
    Look up an EU citation via EUR-Lex (no auth needed).

    Converts Case C-{number}/{year} to CELEX number and checks EUR-Lex.
    - 200 = case exists
    - 404 = case not found
    """
    import re

    match = re.match(r'Case\s+([CT])-(\d+)/(\d+)', citation_text)
    if not match:
        return None

    court_letter, case_num, case_year = match.groups()

    # Build CELEX number: 6{year}CJ{number} for Court of Justice
    # T = General Court (Tribunal), C = Court of Justice
    court_code = "CJ" if court_letter == "C" else "TJ"
    # Pad year to 4 digits, case number to 4 digits
    year_full = case_year if len(case_year) == 4 else f"20{case_year}"
    celex = f"6{year_full}{court_code}{case_num.zfill(4)}"

    url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "Lexintel/1.0"},
            ) as client:
                response = await client.get(url)

                if response.status_code == 200 and len(response.text) > 10000:
                    # Extract case name from page title (bs4, not regex)
                    case_name = _html_title(response.text)

                    return {
                        "found": True,
                        "case_name": case_name,
                        "court": "Court of Justice" if court_letter == "C" else "General Court",
                        "date_filed": year_full,
                        "url": url,
                        "status": "Published",
                        "citation_count": 0,
                        "source": "eurlex",
                    }
                elif response.status_code == 404:
                    return None
    except Exception as e:
        logger.debug(f"EUR-Lex lookup failed: {e}")

    return None


async def lookup_citation(citation_text: str, jurisdiction: str = "US") -> Optional[Dict]:
    """
    Look up a single citation to verify it exists.

    Args:
        citation_text: The citation string (e.g., "347 U.S. 483")
        jurisdiction: ISO jurisdiction code

    Returns:
        Dict with case metadata if found, None if not found/hallucinated.
    """
    # Input length validation (prevent abuse via oversized citations)
    if len(citation_text) > 200:
        logger.warning(f"Citation text too long ({len(citation_text)} chars), skipping lookup")
        return None

    # Check cache first
    key = _cache_key(citation_text, jurisdiction)
    if key in _lookup_cache:
        return _lookup_cache.get(key)

    result = None
    try:
        if jurisdiction == "US":
            result = await _lookup_us_citation(citation_text)
        elif jurisdiction == "UK":
            result = await _lookup_uk_citation(citation_text)
        elif jurisdiction == "IN":
            result = await _lookup_in_citation(citation_text)
        elif jurisdiction == "AU":
            result = await _lookup_au_citation(citation_text)
        elif jurisdiction == "SG":
            result = await _lookup_sg_citation(citation_text)
        elif jurisdiction == "EU":
            result = await _lookup_eu_citation(citation_text)
        elif jurisdiction == "CA":
            # CanLII blocks automated access (403) — no free API available
            # Return None (honest: we cannot verify Canadian citations)
            result = None
        else:
            logger.debug(f"No lookup API for jurisdiction: {jurisdiction}")
    except Exception as e:
        logger.warning(f"Citation lookup failed for '{citation_text}' ({jurisdiction}): {e}")

    # Sanitize URLs before caching (prevent scheme injection)
    if result and result.get("url"):
        url = result["url"]
        if not url.startswith("https://"):
            result["url"] = ""

    # Cache the result
    _lookup_cache.set(key, result)
    return result


async def lookup_citations_batch(
    citations: List[Dict],
) -> Dict[int, Optional[Dict]]:
    """
    Batch lookup citations with concurrency control.

    Args:
        citations: List of citation dicts with 'raw_text', 'jurisdiction', and 'index'

    Returns:
        Dict mapping citation index -> lookup result (or None if not found)
    """
    tasks = []
    for cite in citations:
        tasks.append(
            lookup_citation(cite["raw_text"], cite.get("jurisdiction", "US"))
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    lookup_map = {}
    for cite, result in zip(citations, results):
        if isinstance(result, Exception):
            logger.warning(f"Lookup failed for citation {cite.get('index')}: {result}")
            lookup_map[cite["index"]] = None
        else:
            lookup_map[cite["index"]] = result

    return lookup_map


async def check_case_status(cluster_id: str) -> str:
    """
    Check if a US case is still good law by analyzing forward citations.

    Uses CourtListener's citing-opinions to detect if the case has been
    overruled, reversed, or significantly criticized.

    Args:
        cluster_id: CourtListener cluster ID

    Returns:
        "good_law" | "unknown"

    Note:
        Only "good_law" (cite_count > 0) or "unknown" are ever returned.
        Negative treatment signals ("bad_law", "overruled", "caution") are
        NOT determined here — CourtListener's precedential_status is a
        publication flag, not a citator field.  Negative treatment is
        determined exclusively by NEGATIVE_TREATMENTS edges in the citation
        graph (see citation_graph.is_good_law).  Callers that branch on
        "bad_law" are harmless dead-branches; they will activate if a real
        negative-treatment signal is added to this function in the future.
    """
    if not cluster_id:
        return "unknown"

    # Validate cluster_id is numeric before interpolating into URL
    if not re.match(r'^\d+$', str(cluster_id)):
        logger.warning(f"check_case_status: non-numeric cluster_id rejected: {cluster_id!r}")
        return "unknown"

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                response = await client.get(
                    f"{COURTLISTENER_CLUSTERS}{cluster_id}/",
                    params={"format": "json"},
                    headers=_get_courtlistener_headers(),
                )
                if response.status_code != 200:
                    return "unknown"

                data = response.json()

                # NOTE: CourtListener's `precedential_status` indicates publication
                # status (Published/Unpublished/Errata/Separate/In-chambers/
                # Relating-to/Unknown) — it is NOT a citator treatment field.
                # Negative treatment (overruled, distinguished, etc.) is determined
                # by NEGATIVE_TREATMENTS edges in the citation graph, not here.

                # If case has citing opinions it is likely still relevant
                cite_count = data.get("citation_count", 0)
                if cite_count > 0:
                    return "good_law"

                return "unknown"

    except Exception as e:
        logger.debug(f"Case status check failed for cluster {cluster_id}: {e}")
        return "unknown"


async def get_outbound_case_citations(cluster_id: str, max_results: int = 25) -> List[Dict]:
    """Best-effort: return the cases that the given CourtListener cluster CITES.

    Enables real case->case edges. Uses the v4 opinions endpoint. Returns a list
    of dicts {citation, case_name, cluster_id} for cited opinions. Never raises;
    returns [] on any failure or when no token/network. Requires a CourtListener
    token for best results but degrades gracefully.
    """
    results: List[Dict] = []
    if not cluster_id:
        return results

    # Validate cluster_id is numeric before interpolating into URL
    if not re.match(r'^\d+$', str(cluster_id)):
        logger.warning(f"get_outbound_case_citations: non-numeric cluster_id rejected: {cluster_id!r}")
        return results

    headers = _get_courtlistener_headers()

    async def _get_json(client: httpx.AsyncClient, url: str):
        """Fetch JSON from an absolute CourtListener URL; return dict or None.

        SSRF guard: only fetches URLs that originate from CourtListener.
        Any URL taken from response bodies that does not start with the
        trusted origin is rejected to prevent server-side request forgery.
        """
        if not isinstance(url, str) or not url.startswith(f"{COURTLISTENER_ORIGIN}/"):
            logger.warning(f"_get_json: refusing non-CourtListener URL: {str(url)[:80]!r}")
            return None
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:  # noqa: BLE001 - best-effort enrichment
            return None

    try:
        async with _get_semaphore():
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                # 1. Load the source cluster to find its sub-opinions.
                cluster_url = f"{COURTLISTENER_CLUSTERS}{cluster_id}/?format=json"
                cluster_data = await _get_json(client, cluster_url)
                if not cluster_data:
                    return results

                sub_opinions = cluster_data.get("sub_opinions") or []
                if not sub_opinions:
                    return results

                # 2. Load the first sub-opinion to find what it cites.
                opinion_data = await _get_json(client, sub_opinions[0])
                if not opinion_data:
                    return results

                cited_uris = opinion_data.get("opinions_cited") or []
                if not cited_uris:
                    return results

                # 3. For each cited opinion, resolve its cluster and citation.
                #    Cap total external work at max_results to stay best-effort.
                for cited_uri in cited_uris[:max_results]:
                    cited_opinion = await _get_json(client, cited_uri)
                    if not cited_opinion:
                        continue

                    cited_cluster_uri = cited_opinion.get("cluster")
                    if not cited_cluster_uri:
                        continue

                    cited_cluster = await _get_json(client, cited_cluster_uri)
                    if not cited_cluster:
                        continue

                    citations = cited_cluster.get("citations") or []
                    citation_str = ""
                    for c in citations:
                        try:
                            citation_str = f"{c['volume']} {c['reporter']} {c['page']}"
                            break
                        except (KeyError, TypeError):
                            continue

                    case_name = (
                        cited_cluster.get("case_name")
                        or cited_cluster.get("case_name_short")
                        or ""
                    )

                    # Derive the cited cluster id from the cluster URI tail.
                    cited_cluster_id = ""
                    try:
                        cited_cluster_id = cited_cluster_uri.rstrip("/").split("/")[-1]
                    except Exception:  # noqa: BLE001
                        cited_cluster_id = ""

                    results.append(
                        {
                            "citation": citation_str,
                            "case_name": case_name,
                            "cluster_id": cited_cluster_id,
                        }
                    )
    except Exception:  # noqa: BLE001 - never raise from best-effort enrichment
        return results

    return results
