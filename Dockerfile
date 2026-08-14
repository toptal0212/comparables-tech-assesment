# syntax=docker/dockerfile:1
#
# Two-stage build. The builder installs dependencies, downloads the embedding
# model and builds the search index; the runtime stage copies only the finished
# artifacts. That keeps build tooling out of the shipped image and, more
# importantly, moves all the slow work to build time.
#
# The index is built here rather than on first boot on purpose. Embedding 50k
# documents takes about six minutes. Doing that at startup would mean every
# restart, autoscale event and redeploy pays it again — against a brief that
# names frequent restarts as a constraint. Prebuilt, a cold start is ~1.5s.
#
# Cost of that choice: a larger image (~900MB). Worth it. Pulling 900MB once per
# deploy beats six minutes of unavailability per container start, and it removes
# the runtime dependency on Hugging Face being reachable.

# ---------------------------------------------------------------------------
# Stage 1 — build
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependencies first, in their own layer, so editing application code does not
# invalidate a ~400MB install.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY sample_dataset/ ./sample_dataset/

ENV PYTHONPATH=/install/lib/python3.12/site-packages \
    PATH=/install/bin:$PATH \
    MODEL_CACHE_DIR=/opt/models \
    DATA_DIR=/build/prebuilt \
    # The Hugging Face Xet download backend is compiled for instruction sets
    # that older and shared cloud CPUs lack, where it aborts the process with
    # SIGILL mid-download. The plain HTTP path is marginally slower and works
    # everywhere.
    HF_HUB_DISABLE_XET=1

# Fetch the model into a fixed location, then build SQLite + FTS + vectors.
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/models'); \
print('model cached')" \
 && python -m scripts.build_index --data-dir /build/prebuilt \
 && python -c "\
import json,pathlib; \
m=json.loads(pathlib.Path('/build/prebuilt/index_meta.json').read_text()); \
assert m['count']>0, m; print('index built:', m)"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Surface tracebacks from native crashes (onnxruntime) in the logs rather
    # than leaving an unexplained exit code.
    PYTHONFAULTHANDLER=1 \
    MODEL_CACHE_DIR=/opt/models \
    DATA_DIR=/data \
    PREBUILT_DIR=/opt/prebuilt \
    HF_HUB_DISABLE_XET=1 \
    ENV=production \
    PORT=8000

# curl is for the HEALTHCHECK below; nothing else is needed at runtime, since
# onnxruntime and numpy ship their own shared libraries in the wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Run unprivileged. /data is chowned because a mounted volume starts empty and
# owned by root.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data /opt/prebuilt /opt/models \
 && chown -R appuser:appuser /data /opt/prebuilt /opt/models

COPY --from=builder /install /usr/local
COPY --from=builder --chown=appuser:appuser /opt/models /opt/models
COPY --from=builder --chown=appuser:appuser /build/prebuilt /opt/prebuilt

WORKDIR /app
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser ui/ ./ui/
COPY --chown=appuser:appuser docker-entrypoint.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER appuser

EXPOSE 8000

# Readiness, not liveness: this must fail while the index is still loading so an
# orchestrator holds traffic back instead of routing into errors.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health/ready" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]
