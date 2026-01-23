# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app

# Builder stage: install dependencies in a venv
FROM base AS builder
# Copy only requirements.txt first for better cache utilization
COPY requirements.txt ./
# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

# Final stage: copy app code and venv, set up non-root user
FROM base AS final
# Create non-root user
RUN useradd -m appuser
USER appuser

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/
COPY scope.py ./scope.py
COPY requirements.txt ./requirements.txt
COPY README.md ./README.md

# Expose port 
EXPOSE 8080

# Set default entrypoint
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
