import os
import subprocess
import shutil
import signal
import uuid
import glob
import re
import base64
import io
import hashlib
import json
import queue
import threading
import time
import gradio as gr

import gpu
import epubbuild
import structure
from pdfanalysis import PdfAnalyzer

# Tame Gradio's background telemetry and polling
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

LIBRARY_BASE_DIR = os.environ.get("LIBRARY_DIR", "/app/library")
DELIVERY_BASE_DIR = "/tmp/epub_delivery"
os.makedirs(LIBRARY_BASE_DIR, exist_ok=True)

# Bump when the PDF -> Markdown structure step changes, so cached extractions
# are re-structured (cheap, no GPU) instead of reusing stale Markdown.
STRUCTURE_VERSION = "7"

GPU_WAIT_TIMEOUT_S = int(os.environ.get("GPU_WAIT_TIMEOUT_S", "900"))
GPU_POLL_S = int(os.environ.get("GPU_POLL_S", "15"))
# auto: CPU only for born-digital PDFs (no OCR needed); always: also scanned PDFs; never.
CPU_FALLBACK = os.environ.get("CPU_FALLBACK", "auto").lower()
CPU_THREADS = os.environ.get("MARKER_CPU_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
DEFAULT_MATH_MODE = os.environ.get("MATH_RENDER", "svg").lower()

OOM_PATTERNS = re.compile(r"CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|"
                          r"cudaErrorMemoryAllocation|CUDNN_STATUS_ALLOC_FAILED|out of memory", re.I)


def get_smart_hash(filepath):
    file_size = os.path.getsize(filepath)
    hasher = hashlib.sha256()
    hasher.update(str(file_size).encode())
    with open(filepath, 'rb') as f:
        hasher.update(f.read(1024 * 1024))
        if file_size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, os.SEEK_END)
            hasher.update(f.read(1024 * 1024))
    return hasher.hexdigest()


def fix_mojibake(text):
    if not text: return text
    try:
        return text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"').replace('â€”', '—').replace('â€“', '-')


def enqueue_output(out, q):
    for line in iter(out.readline, ''):
        q.put(line)
    out.close()


def reset_ui():
    return (
        None, None, {},
        gr.update(value=""), gr.update(value=""), gr.update(value=None, visible=False),
        gr.update(value=""), gr.update(value=""),
        gr.update(value=None), gr.update(value=None),
        gr.update(selected=1), gr.update(visible=True, interactive=False),
        gr.update(visible=False), gr.update(interactive=True), gr.update(visible=False)
    )


def enable_start_btn(file): return gr.update(interactive=True) if file is not None else gr.update(interactive=False)


_ACTIVE_PROCS = set()


def cancel_ui():
    for proc in list(_ACTIVE_PROCS):
        _kill_group(proc)
    return (gr.update(visible=True, interactive=True), gr.update(visible=False), gr.update(interactive=True), "🛑 Extraction cancelled.", gr.update(visible=False))


# ------------------------------------------------------------------ preview

_THUMB_CACHE = {}


def _thumb_data_uri(path):
    """Small JPEG data URI for the editor preview, cached by path+mtime."""
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return None
    if key not in _THUMB_CACHE:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((700, 700))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=75)
        _THUMB_CACHE[key] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        if len(_THUMB_CACHE) > 200:
            _THUMB_CACHE.pop(next(iter(_THUMB_CACHE)))
    return _THUMB_CACHE[key]


def render_preview(md_text, work_dir):
    if not md_text or not work_dir: return md_text
    # Pandoc-only syntax the preview widget doesn't know: attributes and fenced divs.
    text = re.sub(r"\)\{[#.][^}\n]*\}", ")", md_text)
    lines, in_div, in_raw = [], 0, False
    for line in text.split("\n"):
        if line.startswith("```{=html}"):
            in_raw = True
            continue
        if in_raw and line.strip() == "```":
            in_raw = False
            continue
        if not in_raw and re.match(r"^:::+", line):
            in_div = in_div + 1 if re.search(r"[\w{]", line) else max(0, in_div - 1)
            continue
        lines.append(("> " + line) if in_div and not line.startswith("<") else line)
    text = "\n".join(lines)
    pattern = r'!\[((?:\\.|[^\]\\])*)\]\((?!http|data:)([^)]+)\)'
    shown = 0

    def repl(m):
        nonlocal shown
        if shown >= 8:
            return "\n\n> 🖼️ *[Image hidden in preview]*\n\n"
        path = os.path.join(work_dir, m.group(2))
        uri = _thumb_data_uri(path) if os.path.exists(path) else None
        if not uri:
            return m.group(0)
        shown += 1
        return f"![{m.group(1)}]({uri})"
    return re.sub(pattern, repl, text)


# ------------------------------------------------------------------ downloads

def _cleanup_deliveries(max_age_s=3600):
    now = time.time()
    for d in glob.glob(os.path.join(DELIVERY_BASE_DIR, "*")):
        try:
            if now - os.path.getmtime(d) > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def prepare_download(filepath, original_basename, extension):
    _cleanup_deliveries()
    dl_dir = os.path.join(DELIVERY_BASE_DIR, uuid.uuid4().hex[:8])
    os.makedirs(dl_dir, exist_ok=True)
    clean_path = os.path.join(dl_dir, f"{original_basename}{extension}")
    shutil.copy(filepath, clean_path)
    return clean_path


# ------------------------------------------------------------------ marker run

def _kill_group(process):
    """Terminate marker and everything it spawned, so no VRAM stays allocated."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, PermissionError):
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _lower_priority():
    try:
        os.nice(10)
    except OSError:
        pass


def _spawn_marker(pdf_path, out_dir, plan, force_ocr, disable_ocr, log):
    """Run marker_single once. Generator yielding log text; returns (returncode, oom, peak_gib)."""
    cfg = dict(plan.config)
    if force_ocr:
        cfg["force_ocr"] = True
    if disable_ocr:
        cfg["disable_ocr"] = True
    cfg_path = os.path.join(out_dir, "marker_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": CPU_THREADS, "MKL_NUM_THREADS": CPU_THREADS,
                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    if plan.device == "cuda":
        env.update({"TORCH_DEVICE": "cuda", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(plan.gpu.index),
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    else:
        env.update({"TORCH_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""})

    cmd = ["marker_single", pdf_path, "--output_dir", out_dir, "--output_format", "json",
           "--config_json", cfg_path]
    if shutil.which("ionice"):
        cmd = ["ionice", "-c2", "-n7"] + cmd
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, env=env, start_new_session=True, preexec_fn=_lower_priority)
    _ACTIVE_PROCS.add(process)
    q = queue.Queue()
    threading.Thread(target=enqueue_output, args=(process.stdout, q), daemon=True).start()

    base_used = gpu.used_bytes(plan.gpu.index) if plan.device == "cuda" else None
    peak = base_used or 0
    oom = False
    try:
        while True:
            try:
                line = q.get(timeout=2.0)
            except queue.Empty:
                if process.poll() is not None and q.empty():
                    break
                if plan.device == "cuda":
                    used = gpu.used_bytes(plan.gpu.index)
                    if used:
                        peak = max(peak, used)
                yield log.tick()
                continue
            if OOM_PATTERNS.search(line):
                oom = True
            yield log.add(line)
        process.wait()
    finally:
        _kill_group(process)
        _ACTIVE_PROCS.discard(process)
    peak_gib = (peak - base_used) / gpu.GiB if base_used is not None else None
    return process.returncode, oom, peak_gib


class _Log:
    """Last lines of the marker log; tqdm progress updates replace each other."""
    def __init__(self, header):
        self.lines = [header]

    def add(self, line):
        line = line.rstrip()
        if not line:
            return self.text()
        prefix = line.split(":")[0][:40]
        if len(self.lines) > 1 and "%|" in line and self.lines[-1].split(":")[0][:40] == prefix:
            self.lines[-1] = line
        else:
            self.lines.append(line)
        self.lines = self.lines[-200:]
        return self.text()

    def tick(self):
        if self.lines[-1].startswith("."):
            self.lines[-1] = self.lines[-1] + "." if len(self.lines[-1]) < 50 else "."
        else:
            self.lines.append(".")
        return self.text()

    def text(self):
        return "\n".join(self.lines[-15:])


def run_marker(pdf_path, out_dir, force_ocr, born_digital):
    """Generator yielding status text. Waits for the GPU when it is occupied by
    something else (no negotiation possible), shrinks batches after an OOM, and
    falls back to CPU where that makes sense."""
    log = _Log("⏳ Heavy extraction (this may take a while)…")
    cpu_allowed = CPU_FALLBACK == "always" or (CPU_FALLBACK == "auto" and born_digital and not force_ocr)
    scale, waited, force_cpu = 1.0, 0, False
    while True:
        plan = gpu.plan_job(scale) if not force_cpu else gpu.Plan("cpu", config=gpu.cpu_batch_config(),
                                                                    reason="GPU unavailable for this job")
        if plan.device == "wait":
            if waited < GPU_WAIT_TIMEOUT_S:
                yield log.add(f"⏸ GPU busy – {plan.reason}. Waiting {waited}/{GPU_WAIT_TIMEOUT_S}s…")
                time.sleep(GPU_POLL_S)
                waited += GPU_POLL_S
                continue
            if not cpu_allowed:
                raise RuntimeError(f"GPU still busy after {GPU_WAIT_TIMEOUT_S}s ({plan.reason}). "
                                   "This PDF needs OCR, which is impractical on CPU – try again later.")
            plan = gpu.Plan("cpu", config=gpu.cpu_batch_config(), reason=plan.reason)
        if plan.device == "cpu" and not cpu_allowed and not force_cpu:
            raise RuntimeError(f"No usable GPU ({plan.reason}) and CPU fallback is not allowed for this PDF.")
        disable_ocr = plan.device == "cpu" and born_digital and not force_ocr
        where = (f"GPU – {plan.reason}" if plan.device == "cuda"
                 else f"CPU ({CPU_THREADS} threads{', text layer only' if disable_ocr else ''}) – {plan.reason}")
        yield log.add(f"▶ Running on {where}")
        rc, oom, peak = yield from _spawn_marker(pdf_path, out_dir, plan, force_ocr, disable_ocr, log)
        if peak is not None:
            yield log.add(f"ℹ GPU memory peak during run: +{peak:.1f} GiB (working budget {plan.budget_gb:.1f} GiB)")
        if rc == 0:
            return
        if oom and plan.device == "cuda":
            if scale > 0.3:
                scale /= 2
                yield log.add("⚠ GPU ran out of memory (another process may have grown). Retrying with smaller batches…")
                continue
            if cpu_allowed:
                force_cpu = True
                yield log.add("⚠ GPU memory still insufficient – falling back to CPU.")
                continue
        if rc in (-9, 137):
            raise RuntimeError("Extractor was killed (host RAM exhausted?).")
        raise RuntimeError(f"Extractor crashed (code {rc}). Last log lines:\n{log.text()}")


# ------------------------------------------------------------------ steps

def _find_marker_json(out_dir):
    hits = [p for p in glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True)
            if not p.endswith(("_meta.json", "marker_config.json", "metadata.json", "structure.json"))
            and "math_cache" not in p]
    return max(hits, key=os.path.getsize) if hits else None


def _structure(json_path, pdf_path, work_dir, base_name):
    """marker JSON -> Pandoc Markdown (cached by STRUCTURE_VERSION)."""
    md_path = os.path.join(work_dir, f"{base_name}.md")
    stamp_path = os.path.join(work_dir, "structure.json")
    try:
        with open(stamp_path) as f:
            stamp = json.load(f)
        if stamp.get("version") == STRUCTURE_VERSION and os.path.exists(md_path):
            with open(md_path, encoding="utf-8") as f:
                return f.read(), stamp.get("title"), stamp.get("stats", {})
    except (OSError, ValueError):
        pass
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)
    shutil.rmtree(os.path.join(work_dir, "images"), ignore_errors=True)
    analyzer = PdfAnalyzer(pdf_path)
    try:
        md, title, stats = structure.convert(doc, analyzer, work_dir)
    finally:
        analyzer.close()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(stamp_path, "w") as f:
        json.dump({"version": STRUCTURE_VERSION, "title": title, "stats": stats}, f)
    return md, title, stats


def _summary(stats):
    labels = [("figures", "figures"), ("tables", "tables"), ("footnotes_linked", "footnotes linked"),
              ("footnotes_anchored_to_paragraph", "unmarked notes"), ("boxes", "info boxes"),
              ("paragraphs_rejoined", "split paragraphs re-joined"), ("styled_runs", "italic/bold runs recovered"),
              ("running_heads_dropped", "running heads removed"), ("cover_sheet_dropped", "cover sheet removed"),
              ("links_dropped", "misaligned links removed"), ("scripts_recovered", "sub/superscripts recovered"),
              ("drop_caps_restored", "drop caps restored"), ("page_notes_detected", "page-foot notes detected"),
              ("decorations_dropped", "logos/badges removed"),
              ("figures_snapped", "figure crops fitted to the drawing"),
              ("text_layer_corrections", "misread characters corrected from the PDF text"),
              ("headings_promoted", "headings recognised from the typography"),
              ("affiliations", "affiliations placed under the authors"),
              ("references_rejoined", "split references re-joined")]
    parts = [f"{stats[k]} {label}" for k, label in labels if stats.get(k)]
    return ", ".join(parts)


def extract_markdown(pdf_file, force_ocr):
    if pdf_file is None:
        yield (None, None, {}, "", "", gr.update(visible=False), "⚠️ No file.", gr.update(selected=1), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=True), gr.update(visible=False))
        return

    base_name = os.path.splitext(os.path.basename(pdf_file.name))[0]
    extracted_metadata = {}

    def busy(msg, work_dir=None):
        return (work_dir, base_name, extracted_metadata, "", "", gr.update(visible=False), msg, gr.update(selected=1), gr.update(interactive=False), gr.update(visible=True), gr.update(interactive=False), gr.update(visible=True))

    try:
        yield busy("⚡ Analyzing PDF & Extracting Metadata...")

        file_hash = get_smart_hash(pdf_file.name)
        session_dir = os.path.join(LIBRARY_BASE_DIR, file_hash)
        work_dir = os.path.join(session_dir, "ocr" if force_ocr else "auto")
        os.makedirs(work_dir, exist_ok=True)
        archive_pdf_path = os.path.join(session_dir, f"{base_name}.pdf")

        if not os.path.exists(archive_pdf_path): shutil.copy(pdf_file.name, archive_pdf_path)

        born_digital = True
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(archive_pdf_path)
            try:
                raw_meta = pdf.get_metadata_dict()
                for key, val in raw_meta.items():
                    if val and str(val).strip():
                        extracted_metadata[key] = fix_mojibake(str(val).strip())
            except Exception:
                pass

            cover_path = os.path.join(session_dir, "cover.jpg")
            if not os.path.exists(cover_path):
                pdf[0].render(scale=2.0).to_pil().convert("RGB").save(cover_path, format="JPEG")
            pdf.close()
            analyzer = PdfAnalyzer(archive_pdf_path)
            born_digital = analyzer.is_born_digital()
            analyzer.close()
        except Exception:
            pass

        edited_md = os.path.join(work_dir, f"{base_name}_edited.md")
        json_path = _find_marker_json(work_dir)
        status = "✅ Cache Hit!"
        if json_path is None:
            runner = run_marker(archive_pdf_path, work_dir, force_ocr, born_digital)
            try:
                for msg in runner:
                    yield busy(msg, work_dir)
            finally:
                runner.close()  # on cancel: tears down the marker process group
            json_path = _find_marker_json(work_dir)
            if json_path is None:
                raise RuntimeError("Extractor finished but produced no JSON output.")
            status = "✅ Done!"

        yield busy("🧩 Rebuilding document structure (footnotes, figures, boxes, styling)…", work_dir)
        md_content, title, stats = _structure(json_path, archive_pdf_path, work_dir, base_name)
        if title and not any(k.lower() == "title" for k in extracted_metadata):
            extracted_metadata["title"] = title
        if os.path.exists(edited_md):
            with open(edited_md, encoding="utf-8") as f:
                md_content = f.read()
            status += " (your saved edits were restored)"
        summary = _summary(stats)
        if summary:
            status += f"\n{summary}"
        md_file = edited_md if os.path.exists(edited_md) else os.path.join(work_dir, f"{base_name}.md")
        yield (work_dir, base_name, extracted_metadata, md_content, render_preview(md_content, work_dir), gr.update(value=prepare_download(md_file, base_name, ".md"), visible=True), status, gr.update(selected=2), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=True), gr.update(visible=False))

    except Exception as e:
        yield (None, None, {}, "", "", gr.update(visible=False), f"❌ ERROR: {str(e)}", gr.update(selected=1), gr.update(interactive=True), gr.update(visible=False), gr.update(interactive=True), gr.update(visible=False))


def build_epub(md_text, metadata_dict, include_toc, math_mode, work_dir, base_name):
    if not work_dir:
        yield gr.update(value=None), gr.update(value=None), "⚠️ No session."
        return
    try:
        yield gr.update(value=None), gr.update(value=None), "⏳ Rendering Math locally & Compiling..."
        session_dir = os.path.dirname(work_dir)

        # Persist edits so a later visit (cache hit) restores them.
        generated_path = os.path.join(work_dir, f"{base_name}.md")
        edited_md_path = os.path.join(work_dir, f"{base_name}_edited.md")
        generated = ""
        if os.path.exists(generated_path):
            with open(generated_path, encoding="utf-8") as f:
                generated = f.read()
        if md_text.replace("\r\n", "\n").strip() != generated.strip():
            with open(edited_md_path, "w", encoding="utf-8") as f: f.write(md_text)
        final_md_path = os.path.join(work_dir, f"{base_name}_final.md")
        with open(final_md_path, "w", encoding="utf-8") as f: f.write(md_text)

        epub_meta = {}
        if not metadata_dict:
            metadata_dict = {}

        if 'Title' not in metadata_dict and 'title' not in metadata_dict:
            epub_meta['title'] = base_name

        key_map = {
            'title': 'title', 'author': 'creator', 'creator': 'creator',
            'subject': 'description', 'description': 'description',
            'keywords': 'keywords', 'creationdate': 'date'
        }

        for k, v in metadata_dict.items():
            k_lower = k.lower()
            target_key = key_map.get(k_lower, k_lower)
            if k_lower == 'creationdate' and str(v).startswith('D:'):
                v = str(v)[2:6]
            epub_meta[target_key] = str(v)

        meta_json_path = os.path.join(work_dir, "metadata.json")
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(epub_meta, f, ensure_ascii=False)

        out_epub_path = os.path.join(work_dir, f"{base_name}.epub")
        out_kepub_path = os.path.join(work_dir, f"{base_name}.kepub.epub")
        css = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epub.css")

        notes = epubbuild.build_epub(
            final_md_path, out_epub_path, [work_dir, session_dir], meta_json_path, css=css,
            cover=os.path.join(session_dir, "cover.jpg"), toc=include_toc,
            math_mode="mathml" if str(math_mode).lower().startswith("mathml") else "svg",
            cache_dir=os.path.join(session_dir, "math_cache"), log=lambda s: None)

        if not epubbuild.kepubify(out_epub_path, out_kepub_path):
            shutil.copy(out_epub_path, out_kepub_path)
            notes.append("kepubify not found – Kobo file is a plain EPUB copy")

        dl_kobo = prepare_download(out_kepub_path, base_name, ".kepub.epub")
        dl_std = prepare_download(out_epub_path, base_name, ".epub")

        msg = "✅ Dual Formats Built Successfully!"
        if notes:
            msg += " (" + "; ".join(notes) + ")"
        yield gr.update(value=dl_kobo), gr.update(value=dl_std), msg
    except Exception as e:
        yield gr.update(value=None), gr.update(value=None), f"❌ Build Error: {str(e)}"


with gr.Blocks(title="PDF to EPUB Wizard") as interface:
    gr.Markdown("# 🧙‍♂️ PDF to EPUB Wizard")
    gr.Markdown("*Self-Hosted Edition: Offline SVG Math, Hidden Metadata Transfer & Dual Outputs.*"
                f" <small>build {os.environ.get('APP_REVISION', 'dev')[:7]}</small>")

    session_state = gr.State()
    basename_state = gr.State()
    metadata_state = gr.State()

    with gr.Tabs() as wizard_tabs:
        with gr.TabItem("1️⃣ Upload", id=1):
            with gr.Row():
                with gr.Column(variant="panel"):
                    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                    force_ocr_cb = gr.Checkbox(label="Force OCR (for PDFs with garbled or missing text)", value=False)
                    start_btn = gr.Button("Start Extraction", variant="primary", interactive=False)
                    stop_btn = gr.Button("🛑 Stop", variant="stop", visible=False)
                    mobile_warn = gr.Markdown("⚠️ **Keep tab open!**", visible=False)
                step1_status = gr.Textbox(label="Logs", interactive=False, lines=7)

        with gr.TabItem("2️⃣ Review & Edit", id=2):
            with gr.Row():
                with gr.Column(scale=1):
                    md_editor = gr.Textbox(label="Editor", lines=25, max_lines=40)
                    md_output = gr.File(label="MD Backup", visible=False)
                    continue_btn = gr.Button("Proceed to Build ➡️", variant="primary")
                md_preview = gr.Markdown(label="Preview", latex_delimiters=[{"left": "$$", "right": "$$", "display": True},{"left": "$", "right": "$", "display": False}])

        with gr.TabItem("3️⃣ Build EPUB", id=3):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("*(PDF Metadata has been captured and will be applied automatically)*")
                    toc_checkbox = gr.Checkbox(label="Generate TOC", value=True)
                    math_radio = gr.Radio(["SVG (works on all readers)", "MathML (smaller; Apple Books, Thorium)"],
                                          value="MathML (smaller; Apple Books, Thorium)" if DEFAULT_MATH_MODE == "mathml" else "SVG (works on all readers)",
                                          label="Math rendering")
                    build_btn = gr.Button("Compile Formats", variant="primary")
                    step2_status = gr.Textbox(label="Status")

            gr.Markdown("### 🎉 Build Outputs")
            with gr.Row():
                kobo_output = gr.File(
                    label="🔵 Kobo Edition (.kepub.epub) — Download this specifically for Kobo devices.",
                    interactive=False
                )
                std_output = gr.File(
                    label="🟢 Standard Edition (.epub) — For Kindle, Apple Books, and all other readers.",
                    interactive=False
                )

    pdf_input.upload(fn=enable_start_btn, inputs=[pdf_input], outputs=[start_btn])

    ext_ev = start_btn.click(
        fn=extract_markdown,
        inputs=[pdf_input, force_ocr_cb],
        outputs=[session_state, basename_state, metadata_state, md_editor, md_preview, md_output, step1_status, wizard_tabs, start_btn, stop_btn, pdf_input, mobile_warn]
    )

    stop_btn.click(fn=cancel_ui, outputs=[start_btn, stop_btn, pdf_input, step1_status, mobile_warn], cancels=[ext_ev])

    pdf_input.change(
        fn=reset_ui,
        outputs=[session_state, basename_state, metadata_state, md_editor, md_preview, md_output, step1_status, step2_status, kobo_output, std_output, wizard_tabs, start_btn, stop_btn, pdf_input, mobile_warn]
    )

    continue_btn.click(fn=lambda: gr.update(selected=3), outputs=[wizard_tabs])

    build_btn.click(
        fn=build_epub,
        inputs=[md_editor, metadata_state, toc_checkbox, math_radio, session_state, basename_state],
        outputs=[kobo_output, std_output, step2_status]
    )

    md_editor.change(fn=render_preview, inputs=[md_editor, session_state], outputs=[md_preview], trigger_mode="always_last")

if __name__ == "__main__":
    print(f"PDF to EPUB Wizard, build {os.environ.get('APP_REVISION', 'dev')}", flush=True)
    # max_size=1 prevents request pile-ups, api_open=False disables background APIs
    interface.queue(max_size=1, api_open=False).launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
