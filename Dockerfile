# ---- Base image ----
FROM python:3.11-slim-bullseye AS base

# Prevent Python from writing .pyc files; unbuffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Faster, repeatable installs
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

# System deps: tini (init), certs, ps (procps), and bash for scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
      tini ca-certificates procps bash \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g ${GID} app
WORKDIR /app

# ---- Builder layer ----
FROM base AS builder

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN python -m pip install --upgrade pip wheel \
 # Build screed without isolation, against a setuptools that still has pkg_resources
 && python -m pip install "setuptools<81" "setuptools_scm[toml]<6" \
 && python -m pip wheel --no-build-isolation --wheel-dir /wheels "screed>=1.1,<1.2" \
 # Build polycore plus all its deps, reusing the screed wheel just built
 && python -m pip wheel --prefer-binary --find-links /wheels --wheel-dir /wheels /app

# ---- Final runtime image ----
FROM base AS runtime

COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index /wheels/*.whl \
 && rm -rf /wheels

# Drop to non-root
USER app

# Default workdir where users can mount data
WORKDIR /workspace

# Use tini as minimal init to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]
