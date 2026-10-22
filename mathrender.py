"""TeX -> inline SVG, batched.

The old code ran `latex` + `dvisvgm` once per formula (two processes each) and
inlined SVGs whose glyph ids (g0-12 ...) collided between formulas. Here all
formulas of a book go through a few LaTeX runs (one page per formula via the
preview package), in parallel chunks, with:
  * baseline-correct alignment (vertical-align from TeX's box depth),
  * glyphs drawn as paths with no shared ids,
  * aria-label alt text, currentColor (dark mode / sepia themes),
  * a per-formula disk cache, so re-builds after edits only compile new formulas,
  * TeX from untrusted PDFs screened (no file access, no definitions).
Formulas that fail are returned as None; the caller falls back to MathML.
"""
import concurrent.futures as cf
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile

RENDER_VERSION = "2"
CHUNK_SIZE = int(os.environ.get("MATH_CHUNK_SIZE", "150"))
WORKERS = int(os.environ.get("MATH_WORKERS", str(min(4, os.cpu_count() or 1))))
FONT_PT = 12.0

_FORBIDDEN = re.compile(
    r"\\(input|include|openin|openout|read|write|immediate|catcode|csname|def|edef|gdef|xdef|let|"
    r"newcommand|renewcommand|providecommand|usepackage|documentclass|special|loop|end\s*\{document\}|"
    r"begin\s*\{document\}|makeatletter|directlua|pdf\w+|output|everypar|jobname|scantokens)(?![A-Za-z])")
_PACKAGES = ["amsmath", "amssymb", "amsfonts", "mathrsfs", "bm", "upgreek", "mhchem", "textcomp"]
_available_packages = None


def _available():
    global _available_packages
    if _available_packages is None:
        found = []
        for pkg in _PACKAGES:
            try:
                r = subprocess.run(["kpsewhich", f"{pkg}.sty"], capture_output=True, text=True, timeout=10)
                if r.stdout.strip():
                    found.append(pkg)
            except (OSError, subprocess.SubprocessError):
                break
        _available_packages = found
    return _available_packages


def tools_available():
    return bool(shutil.which("latex") and shutil.which("dvisvgm"))


def _balanced(tex):
    depth = 0
    for m in re.finditer(r"\\.|[{}]", tex):
        t = m.group(0)
        if t == "{":
            depth += 1
        elif t == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _acceptable(tex):
    return bool(tex.strip()) and len(tex) < 4000 and _balanced(tex) and not _FORBIDDEN.search(tex)


def _key(tex, display):
    return hashlib.sha1(f"{RENDER_VERSION}|{int(display)}|{tex}".encode()).hexdigest()


def _preamble(preview):
    pk = _available()
    lines = [r"\documentclass[12pt]{article}"]
    for p in pk:
        lines.append(r"\usepackage[version=4]{mhchem}" if p == "mhchem" else rf"\usepackage{{{p}}}")
    if preview:
        lines += [r"\usepackage[active,tightpage]{preview}", r"\setlength\PreviewBorder{0pt}"]
    lines += [r"\pagestyle{empty}", r"\begin{document}"]
    return lines


def _compile_chunk(formulas, workdir, preview=True):
    """formulas: [(key, tex, display)] -> {key: svg_html or None}. One LaTeX run."""
    pre = _preamble(preview)
    lines = list(pre)
    line_of = []   # (first_line, last_line) per formula, 1-based
    for i, (_, tex, display) in enumerate(formulas):
        body = rf"$\displaystyle {tex}$" if display else rf"${tex}$"
        start = len(lines) + 1
        if preview:
            lines.append(rf"\begin{{preview}}\sbox0{{{body}}}\typeout{{MATHDIM:{i}:\the\ht0:\the\dp0:\the\wd0}}\usebox0\end{{preview}}")
        else:
            lines.append(rf"\sbox0{{{body}}}\typeout{{MATHDIM:{i}:\the\ht0:\the\dp0:\the\wd0}}\noindent\usebox0\newpage")
        line_of.append((start, len(lines)))
    lines.append(r"\end{document}")
    tex_path = os.path.join(workdir, "m.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    env = dict(os.environ, openin_any="p", openout_any="p", shell_escape="f")
    try:
        r = subprocess.run(["latex", "-interaction=nonstopmode", "-no-shell-escape", "m.tex"],
                           cwd=workdir, capture_output=True, text=True, timeout=180, env=env,
                           errors="replace")
    except subprocess.TimeoutExpired:
        return None
    log = r.stdout
    try:
        with open(os.path.join(workdir, "m.log"), encoding="utf-8", errors="replace") as f:
            log = f.read()
    except OSError:
        pass
    if not os.path.exists(os.path.join(workdir, "m.dvi")):
        return None
    bad_lines = {int(n) for n in re.findall(r"^l\.(\d+)", log, re.M)}
    bad = {i for i, (a, b) in enumerate(line_of) if any(a <= n <= b for n in bad_lines)}
    dims = {}
    for m in re.finditer(r"MATHDIM:(\d+):(-?[\d.]+)pt:(-?[\d.]+)pt:(-?[\d.]+)pt", log):
        dims[int(m.group(1))] = tuple(float(m.group(k)) for k in (2, 3, 4))
    cmd = ["dvisvgm", "--no-fonts=1", "--page=1-", "--output=f%p", "m.dvi"]
    cmd.insert(1, "--bbox=preview" if preview else "--exact-bbox")
    try:
        subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=300, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    svgs = sorted((f for f in os.listdir(workdir) if re.fullmatch(r"f\d+\.svg", f)),
                  key=lambda f: int(f[1:-4]))
    if len(svgs) != len(formulas):
        return None   # page/formula mismatch: caller bisects
    out = {}
    for i, ((key, tex, display), fname) in enumerate(zip(formulas, svgs)):
        if i in bad or i not in dims:
            out[key] = None
            continue
        with open(os.path.join(workdir, fname), encoding="utf-8") as f:
            out[key] = _finish_svg(f.read(), key, tex, display, dims[i] if preview else None)
    return out


def _finish_svg(svg, key, tex, display, dims):
    m = re.search(r"<svg\b.*?</svg>", svg, re.S)
    if not m:
        return None
    svg = re.sub(r"<!--.*?-->", "", m.group(0), flags=re.S)
    prefix = "m" + key[:8] + "-"
    svg = re.sub(r'\bid="([^"]+)"', lambda g: f'id="{prefix}{g.group(1)}"', svg)
    svg = re.sub(r'href="#([^"]+)"', lambda g: f'href="#{prefix}{g.group(1)}"', svg)
    root_end = svg.index(">") + 1
    svg = re.sub(r"\s(width|height)=\"[^\"]*\"", "", svg[:root_end]) + svg[root_end:]
    vb = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', svg)
    if not vb:
        return None
    w_pt, h_pt = float(vb.group(3)), float(vb.group(4))
    style = "display:inline-block"
    if dims:
        _, depth, _ = dims
        style += f";vertical-align:{-depth / FONT_PT:.3f}em"
    else:
        style += ";vertical-align:middle"
    if display:
        style += ";max-width:100%"
    for pat in (r'fill=[\'"](#000000|#000|black)[\'"]', r'stroke=[\'"](#000000|#000|black)[\'"]'):
        svg = re.sub(pat, lambda g: g.group(0).split("=")[0] + '="currentColor"', svg)
    label = html.escape(tex, quote=True)
    svg = svg.replace("<svg ", f'<svg width="{w_pt / FONT_PT:.3f}em" height="{h_pt / FONT_PT:.3f}em" '
                               f'style="{style}" fill="currentColor" role="img" aria-label="{label}" '
                               f'class="math-{"display" if display else "inline"}" ', 1)
    return svg


def _compile_with_bisect(formulas, preview):
    with tempfile.TemporaryDirectory(prefix="math_") as wd:
        res = _compile_chunk(formulas, wd, preview)
    if res is not None:
        return res
    if len(formulas) == 1:
        return {formulas[0][0]: None}
    mid = len(formulas) // 2
    out = _compile_with_bisect(formulas[:mid], preview)
    out.update(_compile_with_bisect(formulas[mid:], preview))
    return out


def render_svgs(formulas, cache_dir, log=None):
    """formulas: iterable of (tex, display). Returns {(tex, display): svg_html or None}."""
    result, todo = {}, []
    os.makedirs(cache_dir, exist_ok=True)
    seen = set()
    for tex, display in formulas:
        if (tex, display) in seen:
            continue
        seen.add((tex, display))
        k = _key(tex, display)
        path = os.path.join(cache_dir, k + ".json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    result[(tex, display)] = json.load(f).get("svg")
                continue
            except (OSError, ValueError):
                pass
        if not _acceptable(tex):
            result[(tex, display)] = None
            continue
        flat = re.sub(r"\s+", " ", re.sub(r"(?<!\\)%[^\n]*", "", tex)).strip()
        todo.append((k, flat, display))
    if todo and tools_available():
        preview = "preview" in _probe_preview()
        chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
        if log:
            log(f"Rendering {len(todo)} formulas in {len(chunks)} LaTeX run(s)…")
        with cf.ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            compiled = {}
            for part in pool.map(lambda c: _compile_with_bisect(c, preview), chunks):
                compiled.update(part)
        by_key = {_key(t, d): (t, d) for t, d in seen}
        for k, svg in compiled.items():
            result[by_key[k]] = svg
            try:
                with open(os.path.join(cache_dir, k + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"svg": svg}, f)
            except OSError:
                pass
    for tex, display in seen:
        result.setdefault((tex, display), None)
    return result


_preview_state = None


def _probe_preview():
    global _preview_state
    if _preview_state is None:
        try:
            r = subprocess.run(["kpsewhich", "preview.sty"], capture_output=True, text=True, timeout=10)
            _preview_state = "preview" if r.stdout.strip() else "none"
        except (OSError, subprocess.SubprocessError):
            _preview_state = "none"
    return _preview_state


def tex_to_mathml(formulas):
    """Batch TeX -> MathML through one pandoc call. {(tex, display): mathml or None}."""
    formulas = list(dict.fromkeys(formulas))
    if not formulas or not shutil.which("pandoc"):
        return {f: None for f in formulas}
    parts = []
    for i, (tex, display) in enumerate(formulas):
        delim = "$$" if display else "$"
        parts.append(f"MATHSEP{i}\n\n{delim}{tex}{delim}")
    try:
        r = subprocess.run(["pandoc", "-f", "markdown", "-t", "html", "--mathml"],
                           input="\n\n".join(parts), capture_output=True, text=True, timeout=120)
        out = r.stdout
    except (OSError, subprocess.SubprocessError):
        return {f: None for f in formulas}
    result = {}
    for i, f in enumerate(formulas):
        m = re.search(rf"<p>MATHSEP{i}</p>\s*<p>(.*?)</p>", out, re.S)
        mm = re.search(r"<math\b.*?</math>", m.group(1), re.S) if m else None
        result[f] = mm.group(0) if mm else None
    return result
