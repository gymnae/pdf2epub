"""TeX -> inline SVG with ziamath (pure Python, MIT, bundled STIX Two Math font).

Replaces the former LaTeX + dvisvgm pipeline: no TeX Live in the image (smaller,
and no Turing-complete TeX interpreter running on text taken from untrusted PDFs),
no subprocesses, a few milliseconds per formula. Output:
  * glyphs as plain paths, no element ids, so many SVGs can share one page,
  * baseline-correct alignment (ziamath puts the baseline at y=0),
  * aria-label alt text, currentColor (dark mode / sepia themes),
  * a per-formula disk cache, so re-builds after edits only render new formulas.
Common mhchem (\\ce{...}) is translated to plain math first. Formulas that
ziamath cannot render faithfully (unknown commands) are returned as None; the
caller falls back to MathML.
"""
import concurrent.futures as cf
import hashlib
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess

RENDER_VERSION = "3-ziamath"
FONT_PX = 12.0          # rendering size; SVG sizes are converted to em relative to it
CHUNK_SIZE = int(os.environ.get("MATH_CHUNK_SIZE", "50"))
WORKERS = int(os.environ.get("MATH_WORKERS", str(min(4, os.cpu_count() or 1))))
PARALLEL_THRESHOLD = 80  # below this many formulas a process pool is not worth starting
CHUNK_TIMEOUT_S = 120


def tools_available():
    return all(importlib.util.find_spec(m) is not None for m in ("ziamath", "latex2mathml"))


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
    return bool(tex.strip()) and len(tex) < 4000 and _balanced(tex)


def _key(tex, display):
    return hashlib.sha1(f"{RENDER_VERSION}|{int(display)}|{tex}".encode()).hexdigest()


def _finish_svg(svg, tex, display):
    """Size in em, baseline alignment, theme-aware colour, alt text."""
    m = re.search(r"<svg\b.*?</svg>", svg, re.S)
    vb = re.search(r'viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"', svg)
    if not m or not vb:
        return None
    svg = m.group(0)
    _, y, w, h = (float(v) for v in vb.groups())
    depth = h + y            # part of the box below the baseline (y=0)
    root_end = svg.index(">") + 1
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg[:root_end]) + svg[root_end:]
    svg = re.sub(r'(fill|stroke)="(#000000|#000|black)"', r'\1="currentColor"', svg)
    style = f"display:inline-block;vertical-align:{-depth / FONT_PX:.3f}em"
    if display:
        style += ";max-width:100%"
    label = html.escape(tex, quote=True)
    return svg.replace("<svg ", f'<svg width="{w / FONT_PX:.3f}em" height="{h / FONT_PX:.3f}em" '
                               f'style="{style}" fill="currentColor" role="img" aria-label="{label}" '
                               f'class="math-{"display" if display else "inline"}" ', 1)


_CE_ARROWS = [("<=>", r"\rightleftharpoons "), ("<->", r"\leftrightarrow "), ("->", r"\rightarrow "),
              ("<-", r"\leftarrow ")]


def _ce_body(body):
    """Common mhchem notation (H2O, SO4^2-, 2H2 + O2 -> 2H2O) as plain TeX math."""
    out, i = [], 0
    while i < len(body):
        for arrow, tex in _CE_ARROWS:
            if body.startswith(arrow, i):
                out.append(tex)
                i += len(arrow)
                break
        else:
            m = re.match(r"\^\{?([0-9]*[+\-−])\}?|\^([0-9]+)", body[i:])
            if m:
                out.append("^{" + (m.group(1) or m.group(2)) + "}")
                i += m.end()
                continue
            m = re.match(r"([A-Z][a-z]?|[)\]])(\d+)", body[i:])
            if m:
                sym = m.group(1)
                out.append((rf"\mathrm{{{sym}}}" if sym.isalpha() else sym) + "_{" + m.group(2) + "}")
                i += m.end()
                continue
            m = re.match(r"[A-Z][a-z]?", body[i:])
            if m:
                out.append(rf"\mathrm{{{m.group(0)}}}")
                i += m.end()
                continue
            ch = body[i]
            out.append(r"\," if ch == " " else ch)
            i += 1
    return "".join(out)


def _expand_mhchem(tex):
    return re.sub(r"\\ce\s*\{([A-Za-z0-9()\[\]+\-−^{}<>=. ]*)\}", lambda m: "{" + _ce_body(m.group(1)) + "}", tex)


def _render_one(tex, display):
    import latex2mathml.converter
    import ziamath as zm
    zm.config.svg2 = False   # plain paths instead of <symbol>/<use>: no ids to collide
    label_tex, tex = tex, _expand_mhchem(tex)
    probe = latex2mathml.converter.convert(tex)
    if re.search(r">\\[A-Za-z]", probe):
        return None          # unknown command rendered as literal text: let MathML handle it
    return _finish_svg(zm.Math.fromlatex(tex, size=FONT_PX, inline=not display).svg(), label_tex, display)


def _render_chunk(items):
    out = []
    for tex, display in items:
        try:
            out.append(_render_one(tex, display))
        except Exception:
            out.append(None)
    return out


def render_svgs(formulas, cache_dir, log=None):
    """formulas: iterable of (tex, display). Returns {(tex, display): svg_html or None}."""
    result, todo = {}, []
    os.makedirs(cache_dir, exist_ok=True)
    for tex, display in dict.fromkeys(formulas):
        path = os.path.join(cache_dir, _key(tex, display) + ".json")
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
        todo.append((tex, display))
    if todo and tools_available():
        if log:
            log(f"Rendering {len(todo)} formulas…")
        if len(todo) < PARALLEL_THRESHOLD or WORKERS < 2:
            rendered = _render_chunk(todo)
        else:
            chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
            rendered = []
            with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
                for chunk, fut in [(c, pool.submit(_render_chunk, c)) for c in chunks]:
                    try:
                        rendered.extend(fut.result(timeout=CHUNK_TIMEOUT_S))
                    except Exception:
                        rendered.extend([None] * len(chunk))
        for (tex, display), svg in zip(todo, rendered):
            result[(tex, display)] = svg
            try:
                with open(os.path.join(cache_dir, _key(tex, display) + ".json"), "w", encoding="utf-8") as f:
                    json.dump({"svg": svg}, f)
            except OSError:
                pass
    for f in formulas:
        result.setdefault(f, None)
    return result


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
