# HarFeast OpenEnv - HF Spaces / Docker deployment
# Build: docker build -t harfeast-env .

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy project files
COPY harfeast_env /app/harfeast_env
COPY harfeast_openenv /app/harfeast_openenv
COPY harfeast_world /app/harfeast_world
COPY harfeast_synthetic_world_generator.py /app/

# Generate world if missing (e.g. harfeast_world not committed)
RUN python /app/harfeast_synthetic_world_generator.py --output-dir /app/harfeast_world 2>/dev/null || true

# Install dependencies
RUN pip install --no-cache-dir openenv-core>=0.2.1 fastapi uvicorn

ENV HARFEAST_WORLD_PATH=/app/harfeast_world
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
CMD ["uvicorn", "harfeast_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
