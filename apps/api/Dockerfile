FROM ghcr.io/astral-sh/uv:0.9.17 AS uv

FROM python:3.12-slim-bookworm AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_HTTP_TIMEOUT=180
WORKDIR /app

FROM base AS api
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev
RUN useradd --create-home --uid 10001 lexintel && install -d -o lexintel -g lexintel /var/lib/lexintel/files
COPY src ./src
USER lexintel
CMD ["uvicorn", "lexintel_ingest.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS parser-tools
RUN <<'INSTALL'
set -eu
sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources
apt-get update
set -- libreoffice-writer tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin fonts-liberation fonts-dejavu-core
apt-get --print-uris --yes --no-install-recommends install "$@" > /tmp/packages
python - <<'PYTHON'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import shutil
import urllib.request

entries = [shlex.split(line) for line in Path("/tmp/packages").read_text().splitlines() if line.startswith("'")]

def download(entry):
    url, filename, *_ = entry
    destination = Path("/var/cache/apt/archives") / filename
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            return
        except OSError:
            destination.unlink(missing_ok=True)
            if attempt == 2:
                raise

with ThreadPoolExecutor(max_workers=12) as executor:
    list(executor.map(download, entries))
PYTHON
apt-get install -y --no-download --no-install-recommends "$@"
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /tmp/packages
INSTALL

FROM parser-tools AS worker
ENV HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --group parser
RUN useradd --create-home --uid 10001 lexintel \
    && install -d -o lexintel -g lexintel /var/lib/lexintel/files /home/lexintel/.cache/huggingface
COPY src ./src
USER lexintel
CMD ["python", "-m", "lexintel_ingest.worker"]
