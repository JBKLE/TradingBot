FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY static/ ./static/
COPY dashboard.py .
COPY dashboard_shared.py .
COPY pages/ ./pages/

# Ensure data directory exists
RUN mkdir -p /app/data/logs

ENV TZ=Europe/Berlin
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the scheduler (set RUN_ONCE=true for a single cycle)
CMD ["python", "-m", "src.main"]
