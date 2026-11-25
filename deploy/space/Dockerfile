# Hugging Face Space (Docker SDK): FastAPI + the built React app on one port.
# The build context is assembled by .github/workflows/deploy-space.yml.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir ".[api]"
COPY --chown=user static ./static

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    RUN_DIR=/home/user/model \
    STATIC_DIR=/home/user/app/static \
    CORS_ORIGINS=* \
    PRELOAD_MODEL=1

EXPOSE 7860
CMD ["uvicorn", "fashion_analyser.api:app", "--host", "0.0.0.0", "--port", "7860"]
