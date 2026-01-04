FROM python:3.11-slim

WORKDIR /app

# First, install the shared package from packages/shared (with extraction extras for PyPDF2, python-docx)
COPY packages/shared ./packages/shared
RUN pip install --no-cache-dir "./packages/shared[extraction]"

# Then install worker requirements
COPY apps/workers/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker application
COPY apps/workers .

CMD ["python", "-m", "src"]
