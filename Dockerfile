# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app

# Builder stage: install dependencies in a venv
FROM base AS builder
# Copy only requirements.txt first for better cache utilization
COPY --link requirements.txt ./
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
COPY --link --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source code
COPY --link src/ ./src/
COPY --link data/ ./data/
COPY --link notebooks/ ./notebooks/
COPY --link scope.py ./scope.py
COPY --link requirements.txt ./requirements.txt
COPY --link README.md ./README.md

# Expose port if needed (uncomment and set correct port if applicable)
# EXPOSE 8000

# Set default entrypoint (adjust as needed, e.g. for API service)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
