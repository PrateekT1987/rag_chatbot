# --- Build & run -----------------------------------------------------------
#   docker compose up -d --build          (reads .env automatically)
# or:
#   docker build -t rag-chatbot .
#   docker run -p 8000:8000 --env-file .env -v ./data:/app/data rag-chatbot
#
# The base image is pinned by digest for reproducible builds. To update it:
#   docker pull python:3.12-slim
#   docker image inspect python:3.12-slim --format '{{index .RepoDigests 0}}'
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# requirements.lock pins exact versions for reproducible production builds.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

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
