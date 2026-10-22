"""Pandoc Markdown -> EPUB3 (+ real Kobo kepub).

Math is handled on Pandoc's AST instead of with regexes over Markdown, so
currency signs, code and image captions are never mistaken for TeX:
  md --pandoc--> JSON AST --(Math -> inline SVG or MathML)--> pandoc --> EPUB
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

import mathrender

MD_FORMAT = "markdown-smart"
# Sections shorter than this on average are an article's, not a book's chapters:
# a file (= page break) per section would leave half-empty screens.
CHAPTER_MIN_CHARS = int(os.environ.get("CHAPTER_MIN_CHARS", "12000"))
_RAW_MATH = re.compile(r'<math display="(inline|block)">(.*?)</math>', re.S)


def pandoc_version():
    try:
        out = subprocess.run(["pandoc", "--version"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"pandoc\S*\s+(\d+)\.(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    except (OSError, subprocess.SubprocessError):
        return (0, 0)


def _walk(node, fn):
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


def _collect_math(ast):
    found = []

    def visit(n):
        t = n.get("t")
        if t == "Math":
            found.append((n["c"][1], n["c"][0].get("t") == "DisplayMath"))
        elif t in ("RawBlock", "RawInline") and n["c"][0] == "html" and "<math" in n["c"][1]:
            for m in _RAW_MATH.finditer(n["c"][1]):
                found.append((_unescape(m.group(2)), m.group(1) == "block"))
    _walk(ast, visit)
    return found


def _unescape(s):
    import html
    return html.unescape(s)


def _escape(s):
    import html
    return html.escape(s, quote=False)


def _replace_math(ast, svgs, mathml):
    """Math nodes -> SVG (if rendered) else leave for pandoc's --mathml.
    Raw-HTML math (inside tables) -> SVG, else MathML, else plain TeX text."""
    def html_for(tex, display):
        svg = svgs.get((tex, display))
        if svg:
            return f'<span class="math-display">{svg}</span>' if display else svg
        return None

    def visit(n):
        t = n.get("t")
        if t == "Math":
            tex, display = n["c"][1], n["c"][0].get("t") == "DisplayMath"
            h = html_for(tex, display)
            if h:
                n["t"], n["c"] = "RawInline", ["html", h]
        elif t in ("RawBlock", "RawInline") and n["c"][0] == "html" and "<math" in n["c"][1]:
            def sub(m):
                tex, display = _unescape(m.group(2)), m.group(1) == "block"
                return html_for(tex, display) or mathml.get((tex, display)) or \
                    f'<code class="tex">{_escape(tex)}</code>'
            n["c"][1] = _RAW_MATH.sub(sub, n["c"][1])
    _walk(ast, visit)


def _split_level(md_text):
    """2 (one file per ## section) for books, 1 for articles and short documents."""
    sections = len(re.findall(r"^## ", md_text, re.M))
    return 2 if sections and len(md_text) / sections >= CHAPTER_MIN_CHARS else 1


def kepubify(epub_path, out_path):
    """Real Kobo kepub via kepubify (adds koboSpan markup). Returns True on success."""
    exe = shutil.which("kepubify")
    if not exe:
        return False
    with tempfile.TemporaryDirectory(prefix="kepub_") as td:
        try:
            subprocess.run([exe, "-i", "-o", td, epub_path], capture_output=True, timeout=300, check=True)
        except (OSError, subprocess.SubprocessError):
            return False
        produced = [f for f in os.listdir(td) if f.endswith(".epub")]
        if not produced:
            return False
        shutil.move(os.path.join(td, produced[0]), out_path)
    return True


def build_epub(md_path, out_epub, resource_dirs, metadata_file, css=None, cover=None,
               toc=True, math_mode="svg", cache_dir=None, log=print):
    """Returns a list of human-readable notes about the build."""
    notes = []
    resource_path = ":".join(resource_dirs)
    r = subprocess.run(["pandoc", md_path, "-f", MD_FORMAT, "-t", "json"],
                       capture_output=True, text=True, timeout=300, check=True)
    ast = json.loads(r.stdout)
    formulas = _collect_math(ast)
    svgs = {}
    if formulas:
        if math_mode == "svg" and mathrender.tools_available():
            svgs = mathrender.render_svgs(formulas, cache_dir or os.path.join(os.path.dirname(md_path), "math_cache"), log=log)
            failed = sum(1 for f in set(formulas) if not svgs.get(f))
            if failed:
                notes.append(f"{failed} formula(s) fell back to MathML")
        unrendered_raw = [f for f in formulas if not svgs.get(f)]
        mathml = mathrender.tex_to_mathml(unrendered_raw) if unrendered_raw else {}
        _replace_math(ast, svgs, mathml)
    ast_path = md_path + ".ast.json"
    with open(ast_path, "w", encoding="utf-8") as f:
        json.dump(ast, f)
    major, minor = pandoc_version()
    with open(md_path, encoding="utf-8") as f:
        level = _split_level(f.read())
    split_flag = f"--split-level={level}" if (major, minor) >= (3, 0) else f"--epub-chapter-level={level}"
    cmd = ["pandoc", ast_path, "-f", "json", "-o", out_epub, "-t", "epub3", split_flag,
           "--standalone", "--mathml", f"--resource-path={resource_path}",
           f"--metadata-file={metadata_file}"]
    if cover and os.path.exists(cover):
        cmd.append(f"--epub-cover-image={cover}")
    if css and os.path.exists(css):
        cmd.append(f"--css={css}")
    if toc:
        cmd += ["--toc", "--toc-depth=3"]
    try:
        subprocess.run(cmd, cwd=os.path.dirname(md_path), capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError((e.stderr or "").strip()[-800:] or "pandoc failed") from e
    finally:
        try:
            os.remove(ast_path)
        except OSError:
            pass
    return notes
