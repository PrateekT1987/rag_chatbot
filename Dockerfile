# --- Build & run -----------------------------------------------------------
#   docker compose up -d --build          (reads .env automatically)
# or:
#   docker build -t rag-chatbot .
#   docker run -p 8000:8000 --env-file .env -v ./data:/app/data rag-chatbot
#
# The base image is pinned by digest for reproducible builds and matches the
# Python version requirements.lock was generated on (3.14). To update it:
#   docker pull python:3.14-slim
#   docker image inspect python:3.14-slim --format '{{index .RepoDigests 0}}'
FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# requirements.lock pins exact versions for reproducible production builds.
# Generous retries keep builds reliable on slow/flaky networks.
COPY requirements.lock .
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r requirements.lock

COPY backend/ backend/
COPY static/ static/
COPY data/ data/

# Run as non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

# --proxy-headers keeps rate limiting accurate behind a reverse proxy.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
