FROM python:3.11-slim

WORKDIR /app

# First, install the shared package from packages/shared (with extraction extras for PyPDF2, python-docx)
COPY packages/shared ./packages/shared
RUN pip install --no-cache-dir "./packages/shared[extraction]"

# Then install backend requirements
COPY apps/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application
COPY apps/backend .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
