FROM python:3.12-slim-bookworm AS download
ARG TARGETARCH
RUN <<'BUILD'
set -eu
export TARGETARCH
python - <<'PY'
import hashlib
import io
import os
from pathlib import Path
import urllib.request
import zipfile

architecture = os.environ["TARGETARCH"]
checksums = {
    "amd64": "ed4410e8b6879a8971cd87fd03853f182e47b85b199c9bfcd9b34f159caba9fb",
    "arm64": "e65f953e99f47c05dde88a031884cf36ea269bcb62377f554b76ddb4d13fdc3e",
}
version = "1.5.1"
url = f"https://github.com/timescale/pg_textsearch/releases/download/v{version}/pg-textsearch-v{version}-pg17-{architecture}.zip"
with urllib.request.urlopen(url, timeout=120) as response:
    package = response.read()
if hashlib.sha256(package).hexdigest() != checksums[architecture]:
    raise SystemExit("pg_textsearch release checksum mismatch")
with zipfile.ZipFile(io.BytesIO(package)) as archive:
    (filename,) = archive.namelist()
    Path("/pg-textsearch.deb").write_bytes(archive.read(filename))
PY
BUILD

FROM pgvector/pgvector:pg17-bookworm
RUN --mount=from=download,source=/pg-textsearch.deb,target=/tmp/pg-textsearch.deb \
    dpkg-deb --extract /tmp/pg-textsearch.deb /
