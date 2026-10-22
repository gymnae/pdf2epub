# syntax=docker/dockerfile:1
#
# Two stages: everything is assembled in a throwaway Debian stage, the runtime is
# distroless (no shell, no package manager, no TeX). torch ships its own CUDA
# libraries; the NVIDIA container toolkit injects the driver (libcuda, NVML) at
# runtime, so no CUDA base image is needed.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- build stage
FROM debian:13-slim AS build
ARG PYTHON_VERSION
ARG PANDOC_VERSION=3.6.4
ARG KEPUBIFY_VERSION=v4.0.4

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Self-contained CPython from uv (Debian 13's 3.13 has no wheels for marker's Pillow 10 pin).
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
RUN uv python install ${PYTHON_VERSION} && uv venv /venv --python ${PYTHON_VERSION}

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /venv/bin/python -r /tmp/requirements.txt

# Static binaries: pandoc (Debian's is older) and kepubify (real Kobo kepubs)
RUN mkdir -p /opt/tools \
    && curl -fsSL https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-linux-amd64.tar.gz \
       | tar -xz -C /tmp \
    && mv /tmp/pandoc-${PANDOC_VERSION}/bin/pandoc /opt/tools/pandoc \
    && curl -fsSL -o /opt/tools/kepubify \
       https://github.com/pgaskin/kepubify/releases/download/${KEPUBIFY_VERSION}/kepubify-linux-64bit \
    && chmod 0755 /opt/tools/*

# marker downloads this font on every converter start unless it already exists
# (marker/util.py: download_font). Baking it in keeps the venv read-only and the
# container free of runtime downloads from models.datalab.to.
ARG MARKER_FONT_URL=https://models.datalab.to/artifacts/GoNotoCurrent-Regular.ttf
RUN font_dir=/venv/lib/python${PYTHON_VERSION}/site-packages/static/fonts \
    && mkdir -p "$font_dir" \
    && curl -fsSL -o "$font_dir/GoNotoCurrent-Regular.ttf" "$MARKER_FONT_URL"

# Writable dirs for the runtime user. They are created *inside* the copied tree:
# COPY --chown reliably applies to copied contents, not to an existing target dir.
# The C++ runtime for torch & co. is copied explicitly rather than relying on what
# the distroless base happens to ship.
RUN mkdir -p /rw/app/library /rw/app/model_cache/home /rw/lib \
    && touch /rw/app/library/.keep /rw/app/model_cache/home/.keep \
    && cp -a /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 /rw/lib/

# -------------------------------------------------------------- runtime stage
FROM gcr.io/distroless/cc-debian13:nonroot
ARG PYTHON_VERSION

COPY --from=build /rw/lib/ /usr/lib/x86_64-linux-gnu/
COPY --from=build /opt/python /opt/python
COPY --from=build /venv /venv
COPY --from=build /opt/tools/ /usr/local/bin/
COPY --from=build --chown=1000:1000 /rw/app /app
COPY --chown=1000:1000 *.py epub.css /app/

ENV PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    HOME=/app/model_cache/home \
    XDG_CACHE_HOME=/app/model_cache/xdg \
    HF_HOME=/app/model_cache/huggingface \
    TORCH_HOME=/app/model_cache/torch \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app
# uid 1000 matches the owner of the bind-mounted ./library and ./model_cache on the host
USER 1000:1000
EXPOSE 7860
ENTRYPOINT ["/venv/bin/python", "app.py"]
