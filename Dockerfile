# ── Builder stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        clang \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./

# Install dependencies into a prefix so we can copy them to the runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="C++ Modernization Engine" \
      org.opencontainers.image.description="Air-gapped LLM pipeline that transforms legacy C/C++ into C++17" \
      org.opencontainers.image.source="https://github.com/Aditi187/cpp-modernizer"

# Install runtime-only system deps (compiler needed at run time for verification)
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Install the project itself (editable-like, no deps — already installed above)
RUN pip install --no-cache-dir --no-deps -e .

# Create output dir
RUN mkdir -p /app/output

# ── Security ──────────────────────────────────────────────────────────────────
# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────────────────────────────────────
EXPOSE 8000

# Required — set this via docker run -e or docker-compose env_file
# ENV API_AUTH_TOKEN=your_token_here

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
