# CUDA base image: provides the NVIDIA runtime hooks (nvidia-smi / NVML for the
# GPU check). torch brings its own CUDA libraries.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# 1. System dependencies (preview-latex-style: one LaTeX run for all formulas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    texlive-latex-base \
    texlive-science \
    preview-latex-style \
    dvisvgm \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# 2. Current pandoc (Ubuntu 22.04 ships 2.9) and kepubify for real Kobo kepubs
ARG PANDOC_VERSION=3.6.4
ARG KEPUBIFY_VERSION=v4.0.4
RUN curl -fsSL -o /tmp/pandoc.deb \
        https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-amd64.deb \
    && dpkg -i /tmp/pandoc.deb && rm /tmp/pandoc.deb \
    && curl -fsSL -o /usr/local/bin/kepubify \
        https://github.com/pgaskin/kepubify/releases/download/${KEPUBIFY_VERSION}/kepubify-linux-64bit \
    && chmod +x /usr/local/bin/kepubify

# 3. Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

RUN useradd -m -u 1000 appuser

RUN mkdir -p /usr/local/lib/python3.10/dist-packages/static && \
    chown -R appuser:appuser /usr/local/lib/python3.10/dist-packages/static

WORKDIR /app

COPY --chown=appuser:appuser . /app/

USER appuser

EXPOSE 7860

CMD ["python3", "app.py"]
