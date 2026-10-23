# pdf2epub

Self-hosted web app that turns PDFs (scientific articles in particular) into reflowable EPUBs for e-readers.

[marker](https://github.com/datalab-to/marker) extracts the text, layout and OCR. The app then rebuilds the document structure: headings, footnotes, figures with captions, tables, lists, quotations, math and references. Each conversion produces two files:

- `.kepub.epub` for Kobo devices
- `.epub` for every other reader

## Run

```sh
docker compose up -d
```

Then open <http://localhost:7860>. You need an NVIDIA GPU with the container toolkit installed. Without a GPU, the app falls back to CPU for PDFs that have a text layer.

Volumes:

- `./library`: cached extractions, edits and outputs, keyed by file hash
- `./model_cache`: model weights

## Use

- **Wizard**: upload a PDF, review and edit the Markdown, then build the EPUB.
- **Bulk Convert**: upload several PDFs and get the EPUBs straight away, plus a zip of all of them. This skips the edit step. Edits you saved earlier in the wizard are still applied.

Repeat runs of the same PDF reuse the cached extraction. Only one marker run happens at a time; other jobs queue first come, first served.

## Configuration

Set these as environment variables in `docker-compose.yml`.

| Variable | Default | Purpose |
|---|---|---|
| `CPU_FALLBACK` | `auto` | `auto` (CPU only for PDFs with a text layer), `always`, `never` |
| `GPU_WAIT_TIMEOUT_S` | `900` | How long to wait for a busy GPU |
| `GPU_SAFETY_MARGIN_GB` | `1.0` | VRAM left free for other processes |
| `MARKER_BASE_VRAM_GB` | `3.5` | marker's base VRAM use; tune it from the "GPU peak" log line |
| `MARKER_CPU_THREADS` | half the cores | CPU threads for marker |
| `MATH_RENDER` | `svg` | `svg` (works on all readers) or `mathml` |

## Development

- `app.py`: Gradio UI, job queue, marker runs
- `structure.py`: turns marker's JSON into Pandoc Markdown
- `pdfanalysis.py`: reads the PDF text layer and geometry
- `epubbuild.py`, `mathrender.py`, `epub.css`: EPUB output
- `gpu.py`: GPU probing and batch sizing

Changes to `structure.py` require bumping `STRUCTURE_VERSION` in `app.py`, so cached documents are restructured.

To check that the built image works:

```sh
docker run --rm --entrypoint /venv/bin/python ghcr.io/gymnae/pdf2epub selftest.py
```

GitHub Actions builds and publishes the image (`.github/workflows/docker.yml`).
