"""marker JSON -> Pandoc Markdown with semantic structure.

marker's own Markdown renderer flattens everything that makes scientific
layouts hard to read on an e-reader. This module works on marker's block tree
(JSON output) instead and emits Pandoc Markdown that keeps:

  * footnotes as real notes ([^id]) linked to their reference marks,
  * figures as one image (split figures merged) with their caption,
  * tables as real HTML tables (merged cells intact) instead of linearised text,
  * info boxes as ::: {.box} divs, moved out of the running text,
  * paragraphs that a page break, figure, box or footnote split in two, re-joined,
  * italic/bold recovered from the PDF fonts where marker lost it,
  * consistent heading levels, no publisher cover sheet, no running heads,
  * only links whose text matches their target (marker's link offsets drift).

Everything here is pure Python; `analyzer` (pdfanalysis.PdfAnalyzer) is optional.
"""
import base64
import contextvars
import html as htmllib
import os
import re
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import quote

TABLE_IMAGE_MIN_COLS = int(os.environ.get("TABLE_IMAGE_MIN_COLS", "7"))
FIGURE_DPI = int(os.environ.get("FIGURE_DPI", "220"))
FIGURE_MAX_PX = int(os.environ.get("FIGURE_MAX_PX", "1600"))

PARA_TYPES = {"Text", "TextInlineMath", "Handwriting", "ComplexRegion", "ListItem"}
IMAGE_TYPES = {"Figure", "Picture"}
DROP_TYPES = {"PageHeader", "PageFooter", "TableOfContents", "Reference", "Line", "Span"}
REFERENCE_HEADINGS = re.compile(
    r"^\W*(references|bibliography|literature( cited)?|works cited|sources|literatur(verzeichnis)?|"
    r"quellen(verzeichnis)?|références|bibliografía)\W*$", re.I)
FOOTNOTE_SYMBOLS = "*†‡§¶‖"
AFFILIATION = re.compile(r"\b(universit|institut|department|laborator|centre|center|school of|faculty|"
                         r"college|corresponding author|e-?mail|present address)", re.I)
CONTINUATION_WORDS = {
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "by", "for", "with", "from", "as",
    "at", "that", "which", "is", "are", "was", "were", "be", "this", "these", "its", "their",
    "der", "die", "das", "und", "oder", "von", "zu", "mit", "für", "ein", "eine", "des", "dem", "den",
}


# --------------------------------------------------------------------------- HTML tree

class _Node:
    __slots__ = ("tag", "attrs", "children")

    def __init__(self, tag, attrs=None):
        self.tag, self.attrs, self.children = tag, dict(attrs or {}), []


VOID_TAGS = {"br", "img", "hr", "col", "wbr", "input", "meta", "link"}


class _TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(_Node(tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def parse_html(s):
    tb = _TreeBuilder()
    tb.feed(s or "")
    tb.close()
    return tb.root


def plain_text(node):
    if isinstance(node, str):
        return node
    if node.tag in ("content-ref", "img"):
        return ""
    return "".join(plain_text(c) for c in node.children)


def html_plain(s):
    return re.sub(r"\s+", " ", plain_text(parse_html(s))).strip()


# --------------------------------------------------------------------------- text fixes

_SPACING_ACCENTS = {"¨": "\u0308", "´": "\u0301", "ˆ": "\u0302", "˜": "\u0303", "¸": "\u0327",
                    "˚": "\u030a", "ˇ": "\u030c", "˘": "\u0306", "˙": "\u0307", "˝": "\u030b"}
_ACCENT_CLASS = "".join(_SPACING_ACCENTS)
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "\u00ad": ""}


def _compose(base, accent):
    if accent == "´" and base.lower() not in "aeiouy":
        return None  # "don´t" is an apostrophe, not ń
    composed = unicodedata.normalize("NFC", base + _SPACING_ACCENTS[accent])
    return composed if len(composed) == 1 else None


# Some publisher fonts (MathPi-style symbol fonts) map math glyphs to Latin-1 code
# points: "WS ¼ WFloc", "ð1Þ", "WAloc þ". Only repaired when a document shows the pattern.
_MATHPI = contextvars.ContextVar("mathpi", default=False)
_MATHPI_PAIR = re.compile(r"ð([^ðÞ\n]{0,40})Þ")
_MATHPI_MAP = {"¼": "=", "þ": "+"}


def fix_text(s):
    """Repair text-layer artefacts common in TeX-made and older PDFs."""
    if not s:
        return s
    for k, v in _LIGATURES.items():
        s = s.replace(k, v)
    if _MATHPI.get():
        s = _MATHPI_PAIR.sub(r"(\1)", s)
        for k, v in _MATHPI_MAP.items():
            s = s.replace(k, v)
    # "Hovmo¨ller", "c¸a" -> "Hovmöller", "ça" (spacing accent after the letter)
    s = re.sub(rf"([A-Za-z])([{_ACCENT_CLASS}])",
               lambda m: _compose(m.group(1), m.group(2)) or m.group(0), s)
    # "¨uber" -> "über" (accent before the letter, at a word start)
    s = re.sub(rf"(?<![A-Za-z])([{_ACCENT_CLASS}])([A-Za-z])",
               lambda m: _compose(m.group(2), m.group(1)) or m.group(0), s)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e\x0f\x12-\x1f]", "", s)


_MD_ESCAPE = re.compile(r"([\\`*_{}\[\]<>$^~#|@&])")
NOTE_TOKEN = re.compile(r"\x10([^\x11]+)\x11")


def md_escape(s):
    return _MD_ESCAPE.sub(r"\\\1", s)


def escape_line_start(md):
    """Stop a paragraph's first characters from being read as list/div/etc. syntax."""
    m = re.match(r"(\d+|[A-Za-z]|[ivxlcdmIVXLCDM]+)([.)])(\s|$)", md)
    if m:
        return md[:m.end(1)] + "\\" + md[m.end(1):]
    if re.match(r"\(\s*(\d+|[A-Za-z]|[ivxlcdm]+)\s*\)(\s|$)", md):
        return "\\" + md
    if md[:1] in "-+:%=":
        return "\\" + md
    return md


# --------------------------------------------------------------------------- math

_TEXTY_MATH = re.compile(r"[\^_=]|\\(frac|sum|int|sqrt|left|begin|partial|alpha|beta|gamma|delta|"
                         r"lambda|mu|sigma|pi|theta|cdot|times|leq|geq|approx|infty)")


def _looks_like_text(tex):
    bare = re.sub(r"\\[A-Za-z]+", " ", tex)
    words = re.findall(r"\b[A-Za-z]{4,}\b", bare)
    return len(words) >= 5 and not _TEXTY_MATH.search(tex)


def _plainify_tex(tex):
    s = re.sub(r"\\(text|mathrm|textrm|textit|mathit|mbox|operatorname)\s*\{([^{}]*)\}", r"\2", tex)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\([%$&#_{}])", r"\1", s)
    s = re.sub(r"\\[A-Za-z]+", "", s)
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def clean_tex(tex):
    tex = re.sub(r"(?<!\\)%[^\n]*", "", tex or "")
    return re.sub(r"\s+", " ", tex).strip()


def math_md(tex, display):
    tex = clean_tex(tex)
    if not tex:
        return ""
    if _looks_like_text(tex):
        return md_escape(fix_text(_plainify_tex(tex)))
    tex = tex.replace("$", r"\$")
    return f"$${tex}$$" if display else f"${tex}$"


# --------------------------------------------------------------------------- links

def keep_link(href, text):
    """marker's link spans drift on many PDFs: keep a link only if its text matches it."""
    if not href or href.startswith("#"):
        return False
    h = href.lower()
    t = re.sub(r"\s+", "", text.lower()).strip(".,;:()[]")
    if h.startswith("mailto:"):
        return "@" in t
    if len(t) < 4:
        return False
    t = re.sub(r"^(https?://|doi:)", "", t)
    return t in h or (len(t) >= 8 and t[:8] in h)


def _unescape_md(s):
    return re.sub(r"\\(.)", r"\1", s)


_MD_LINK = re.compile(r"\[(?:\\.|[^\]\\])*\]\((?:\\.|[^)\\])*\)")


def _outside_links(md, fn):
    out, pos = [], 0
    for m in _MD_LINK.finditer(md):
        out.append(fn(md[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(md[pos:]))
    return "".join(out)


def _link_dois(md):
    out, pos = [], 0
    while True:
        i = md.find("(doi:", pos)
        if i < 0:
            out.append(md[pos:])
            return "".join(out)
        j, depth = i + 1, 1
        while j < len(md) and depth:
            depth += {"(": 1, ")": -1}.get(md[j], 0)
            j += 1
        inner = md[i + 5:j - 1] if depth == 0 else ""
        doi_md = re.sub(r"\s+", "", inner)
        doi = _unescape_md(doi_md)
        if depth == 0 and re.match(r"10\.\d{4,9}/\S+", doi):
            out.append(md[pos:i] + f"(doi:[{doi_md}](https://doi.org/{quote(doi, safe='/:;()')}))")
        else:
            out.append(md[pos:j])
        pos = j


def _link_urls(md):
    def repl(m):
        url_md = m.group(0).rstrip(".,;:")
        tail = m.group(0)[len(url_md):]
        return f"[{url_md}]({quote(_unescape_md(url_md), safe=':/?#=&%~+@!$;,')}){tail}"
    return re.sub(r"(?<![\w(\[<])https?://(?:\\.|[^\s<>\\\])])+", repl, md)


def autolink(md):
    return _outside_links(_link_dois(md) if "(doi:" in md else md, _link_urls)


# --------------------------------------------------------------------------- inline rendering

@dataclass
class _Ctx:
    style_re: object = None
    styles: dict = None
    styled: bool = False
    heading: bool = False
    stats: dict = None


def _wrap(delim, inner):
    core = inner.strip()
    if not core:
        return inner
    lead = inner[:len(inner) - len(inner.lstrip())]
    trail = inner[len(inner.rstrip()):]
    return f"{lead}{delim}{core}{delim}{trail}"


def _text_md(text, ctx):
    text = re.sub(r"\s+", " ", fix_text(text))
    out = []
    pieces = re.split(r"(\x10[^\x11]+\x11)", text)
    for piece in pieces:
        m = NOTE_TOKEN.fullmatch(piece)
        if m:
            out.append(f"[^{m.group(1)}]")
        elif ctx.style_re is not None and not ctx.styled and not ctx.heading:
            pos = 0
            for sm in ctx.style_re.finditer(piece):
                style = ctx.styles.get(re.sub(r"\s+", " ", sm.group(0)))
                out.append(md_escape(piece[pos:sm.start()]))
                delim = "*" if style == "italic" else "**"
                out.append(f"{delim}{md_escape(sm.group(0))}{delim}")
                if ctx.stats is not None:
                    ctx.stats["styled_runs"] = ctx.stats.get("styled_runs", 0) + 1
                pos = sm.end()
            out.append(md_escape(piece[pos:]))
        else:
            out.append(md_escape(piece))
    return "".join(out)


def render_inline(node, ctx):
    if isinstance(node, str):
        return _text_md(node, ctx)
    tag = node.tag
    if tag in ("content-ref", "img", "hr", "script", "style"):
        return ""
    if tag == "br":
        return " "
    if tag == "math":
        return math_md(plain_text(node), node.attrs.get("display") == "block")
    if tag == "code":
        code = plain_text(node).replace("`", "'")
        return f"`{code}`" if code.strip() else ""

    def inner(**changes):
        sub = _Ctx(**{**ctx.__dict__, **changes})
        return "".join(render_inline(c, sub) for c in node.children)

    if tag in ("i", "em"):
        return _wrap("*", inner(styled=True))
    if tag in ("b", "strong"):
        return inner() if ctx.heading else _wrap("**", inner(styled=True))
    if tag in ("sup", "sub"):
        content = inner(styled=True).strip()
        if not content:
            return ""
        if NOTE_TOKEN.search(content) or "[^" in content:
            return content
        delim = "^" if tag == "sup" else "~"
        return delim + content.replace(" ", "\\ ") + delim
    if tag in ("u", "small", "mark"):
        content = inner()
        cls = {"u": "underline", "small": "small", "mark": "mark"}[tag]
        return f"[{content}]{{.{cls}}}" if content.strip() else content
    if tag == "a":
        content = inner()
        href = node.attrs.get("href", "")
        if keep_link(href, plain_text(node)):
            return f"[{content}]({quote(href, safe=':/?#=&%~+@!$;,()')})"
        if ctx.stats is not None and href and not href.startswith("#"):
            ctx.stats["links_dropped"] = ctx.stats.get("links_dropped", 0) + 1
        return content
    return inner()


def inline_md(html, ctx):
    md = render_inline(parse_html(html), ctx)
    md = re.sub(r"[ \t\r\n]+", " ", md).strip()
    return autolink(md)


# --------------------------------------------------------------------------- safe HTML (tables)

_TABLE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "b", "i", "em",
               "strong", "sup", "sub", "br", "math", "span", "p", "u", "small"}
_KEEP_ATTRS = {"td": ("colspan", "rowspan"), "th": ("colspan", "rowspan"), "math": ("display",)}


def safe_html(node):
    """Re-serialise an HTML subtree as well-formed XHTML with a tag whitelist."""
    if isinstance(node, str):
        return htmllib.escape(fix_text(node), quote=False)
    inner = "".join(safe_html(c) for c in node.children)
    tag = node.tag
    if tag not in _TABLE_TAGS:
        return inner
    if tag == "math":
        return f'<math display="{htmllib.escape(node.attrs.get("display") or "inline")}">' \
               f"{htmllib.escape(clean_tex(plain_text(node)), quote=False)}</math>"
    if tag == "br":
        return "<br/>"
    attrs = "".join(f' {k}="{htmllib.escape(str(node.attrs[k]))}"'
                    for k in _KEEP_ATTRS.get(tag, ()) if node.attrs.get(k))
    return f"<{tag}{attrs}>{inner}</{tag}>"


def _find(node, tag):
    if isinstance(node, str):
        return None
    if node.tag == tag:
        return node
    for c in node.children:
        hit = _find(c, tag)
        if hit is not None:
            return hit
    return None


def _table_columns(table):
    best = 0
    stack = [table]
    while stack:
        n = stack.pop()
        if isinstance(n, str):
            continue
        if n.tag == "tr":
            cols = 0
            for c in n.children:
                if not isinstance(c, str) and c.tag in ("td", "th"):
                    try:
                        cols += int(c.attrs.get("colspan") or 1)
                    except ValueError:
                        cols += 1
            best = max(best, cols)
        else:
            stack.extend(n.children)
    return best


# --------------------------------------------------------------------------- items

@dataclass
class Item:
    kind: str                 # heading para list figure table equation footnote caption box code
    page: int
    bbox: tuple
    html: str = ""
    btype: str = ""
    level: int = 0            # heading: marker's level, then the normalised level
    caption: "Item" = None
    children: list = field(default_factory=list)   # list: li htmls; box: items
    images: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)      # (note_id, html) defined after this item
    md: str = ""

    @property
    def text(self):
        return html_plain(self.html)


def _norm_bbox(bbox, page_bbox):
    px0, py0, px1, py1 = page_bbox
    w, h = (px1 - px0) or 1.0, (py1 - py0) or 1.0
    x0, y0, x1, y1 = bbox
    return ((x0 - px0) / w, (y0 - py0) / h, (x1 - px0) / w, (y1 - py0) / h)


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _contained(inner, outer):
    """Fraction of `inner`'s area that lies inside `outer`."""
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return w * h / area


def _heading_level(html):
    m = re.search(r"<h([1-6])", html or "")
    return int(m.group(1)) if m else 2


class Converter:
    def __init__(self, doc, analyzer=None, out_dir=".", image_subdir="images", drop_cover=True):
        self.doc = doc
        self.an = analyzer
        self.out_dir = out_dir
        self.image_subdir = image_subdir
        self.drop_cover = drop_cover
        self.stats = {}
        self.title = None
        self._style_cache = {}
        self._fig_count = 0

    # ---- collection

    def _collect(self):
        items = []
        self.page_count = 0
        for pidx, page in enumerate(self.doc.get("children") or []):
            m = re.match(r"/page/(\d+)/", page.get("id", ""))
            pnum = int(m.group(1)) if m else pidx
            self.page_count = max(self.page_count, pnum + 1)
            pb = page.get("bbox") or [0, 0, 1, 1]
            for blk in page.get("children") or []:
                items.extend(self._from_block(blk, pnum, pb))
        return items

    def _from_block(self, blk, pnum, pb):
        bt = blk.get("block_type", "")
        html = blk.get("html") or ""
        bbox = _norm_bbox(blk.get("bbox") or pb, pb)
        kids = blk.get("children") or []
        if bt in DROP_TYPES:
            return []
        if bt in ("FigureGroup", "PictureGroup", "TableGroup"):
            main, captions, rest = None, [], []
            for k in kids:
                kt = k.get("block_type")
                if main is None and kt in IMAGE_TYPES | {"Table", "Form"}:
                    main = self._from_block(k, pnum, pb)
                elif kt == "Caption":
                    captions.append(k.get("html") or "")
                else:
                    rest.extend(self._from_block(k, pnum, pb))
            if not main:
                return rest
            head = main[0]
            if captions:
                head.caption = Item("caption", pnum, bbox, " ".join(captions), "Caption")
            return main + rest
        if bt == "ListGroup":
            lis = [k.get("html") or "" for k in kids] or [html]
            return [Item("list", pnum, bbox, " ".join(lis), bt, children=lis)]
        if kids and bt not in PARA_TYPES | IMAGE_TYPES | {"Table", "Form", "Caption", "Footnote",
                                                         "SectionHeader", "Equation", "Code"}:
            out = []
            for k in kids:
                out.extend(self._from_block(k, pnum, pb))
            return out
        if bt in IMAGE_TYPES:
            return [Item("figure", pnum, bbox, html, bt, images=dict(blk.get("images") or {}))]
        if bt in ("Table", "Form"):
            return [Item("table", pnum, bbox, html, bt, images=dict(blk.get("images") or {}))]
        kind = {"SectionHeader": "heading", "Caption": "caption", "Footnote": "footnote",
                "Equation": "equation", "Code": "code"}.get(bt, "para")
        item = Item(kind, pnum, bbox, html, bt)
        if kind == "heading":
            item.level = _heading_level(html)
        if not item.text and kind != "equation":
            return []
        return [item]

    # ---- cleanup passes

    @staticmethod
    def _in_margin(it):
        x0, y0, x1, y1 = it.bbox
        top_bottom = y1 < 0.1 or y0 > 0.9
        side = (x1 < 0.1 or x0 > 0.9) and (x1 - x0) < 0.08   # e.g. vertical "Downloaded from" stamps
        return top_bottom or side

    def _drop_noise(self, items):
        if self.drop_cover and self.an is not None:
            covers = set(self.an.cover_pages())
            if covers:
                items = [it for it in items if it.page not in covers]
                self.stats["cover_sheet_dropped"] = len(covers)
        key = lambda it: re.sub(r"\d+", "#", it.text.lower())
        pages_of = {}
        for it in items:
            if it.kind in ("para", "heading") and len(it.text) <= 250:
                pages_of.setdefault(key(it), set()).add(it.page)
        threshold = max(3, 0.3 * self.page_count)
        repeated = {k for k, pages in pages_of.items() if len(pages) >= threshold}
        out = []
        for it in items:
            if it.kind in ("para", "heading"):
                text = it.text
                sentence = text.endswith(".") and len(text.split()) >= 5
                boilerplate = key(it) in repeated and (self._in_margin(it) or (
                    len(pages_of[key(it)]) >= 0.5 * self.page_count and not sentence))
                page_number = self._in_margin(it) and re.fullmatch(r"(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?", text, re.I)
                junk = re.fullmatch(r"[\d():,;.\-]{8,}", text) is not None  # e.g. Nature's "1234567890():,;"
                if boilerplate or page_number or junk:
                    self.stats["running_heads_dropped"] = self.stats.get("running_heads_dropped", 0) + 1
                    continue
            out.append(it)
        return out

    def _drop_figure_labels(self, items):
        """Axis labels and legends of vector charts that marker emitted as text."""
        figs = [it for it in items if it.kind == "figure"]
        if not figs:
            return items
        out = []
        for it in items:
            if it.kind in ("para", "heading") and any(
                    f.page == it.page and _contained(it.bbox, f.bbox) >= 0.9 for f in figs):
                self.stats["figure_labels_dropped"] = self.stats.get("figure_labels_dropped", 0) + 1
                continue
            out.append(it)
        return out

    def _figures_and_captions(self, items):
        merged = []
        for it in items:
            prev = merged[-1] if merged else None
            if (it.kind == "figure" and prev is not None and prev.kind == "figure"
                    and prev.page == it.page and not (prev.caption and it.caption)
                    and (it.bbox[1] - prev.bbox[3] < 0.05 or prev.bbox[0] < it.bbox[2] and it.bbox[0] < prev.bbox[2])):
                prev.bbox = _union(prev.bbox, it.bbox)
                prev.caption = prev.caption or it.caption
                prev.images.update(it.images)
                self.stats["figures_merged"] = self.stats.get("figures_merged", 0) + 1
                continue
            merged.append(it)
        items = merged
        out = list(items)
        for idx, it in enumerate(items):
            if it.kind != "caption":
                continue
            want = "table" if re.match(r"\W*tab", it.text, re.I) else "figure"
            candidates = [items[j] for j in (idx - 1, idx + 1) if 0 <= j < len(items)]
            candidates += sorted((c for c in items if c.page == it.page),
                                 key=lambda c: abs(c.bbox[1] - it.bbox[1]))
            for c in candidates:
                if c.page == it.page and c.kind in (want, "figure", "table") and c.caption is None:
                    if c.kind != want and any(x.kind == want and x.page == it.page and x.caption is None
                                              for x in candidates):
                        continue
                    c.caption = it
                    out.remove(it)
                    break
        return out

    def _boxes(self, items):
        if self.an is None:
            return items
        pages = sorted({it.page for it in items})
        for p in pages:
            on_page = [it for it in items if it.page == p]
            exclude = [it.bbox for it in on_page if it.kind in ("figure", "table")]
            try:
                rects = self.an.box_regions(p, exclude)
            except Exception:
                continue
            text_items = [it for it in on_page if it.kind in ("para", "heading", "list", "equation")]
            for rect in sorted(rects, key=lambda r: -(r[2] - r[0]) * (r[3] - r[1])):
                members = []
                for it in text_items:
                    cx, cy = (it.bbox[0] + it.bbox[2]) / 2, (it.bbox[1] + it.bbox[3]) / 2
                    if rect[0] - 0.01 <= cx <= rect[2] + 0.01 and rect[1] - 0.01 <= cy <= rect[3] + 0.01:
                        members.append(it)
                if not members or len(members) >= max(2, 0.8 * len(text_items)):
                    continue
                if all(m.kind == "heading" for m in members):
                    continue
                box = Item("box", p, rect, children=members)
                pos = items.index(members[0])
                for m in members:
                    items.remove(m)
                    text_items.remove(m)
                items.insert(min(pos, len(items)), box)
                self.stats["boxes"] = self.stats.get("boxes", 0) + 1
        return items

    def _footnotes(self, items):
        flat = []  # (item, container) for reference search, in reading order
        for it in items:
            if it.kind == "box":
                flat.extend(it.children)
            else:
                flat.append(it)
        note_seq = 0
        linked = unlinked = 0
        first_page = min((it.page for it in flat), default=0)
        for idx, note in enumerate(list(flat)):
            if note.kind != "footnote":
                continue
            html = note.html
            m = re.match(r"\s*(?:<p[^>]*>)?\s*(?:<sup>\s*([^<\s]{1,4})\s*</sup>|([" + re.escape(FOOTNOTE_SYMBOLS)
                         + r"]{1,3})|(\d{1,3})(?=\s))", html)
            mark = (m.group(1) or m.group(2) or m.group(3)) if m else None
            body = html[m.end():] if m else html
            note_seq += 1
            note_id = f"n{note_seq}"
            token = f"\x10{note_id}\x11"
            owner = None
            if mark:
                patterns = [r"<sup>\s*" + re.escape(mark) + r"\s*</sup>"]
                if not mark.isdigit():
                    patterns.append(r"(?<=[\w.,;:)\]’'])" + re.escape(mark)
                                    + r"(?![\w" + re.escape(FOOTNOTE_SYMBOLS) + "])")
                for pattern in patterns:
                    for scope in (note.page, note.page - 1):
                        owner = next((c for c in flat if c.page == scope and self._place_token(c, pattern, token)), None)
                        if owner:
                            break
                    if owner:
                        break
            by_mark = owner is not None
            if owner is None:
                # Unmarked note: anchor it to the paragraph it interrupted.
                if AFFILIATION.search(note.text) and note.page == first_page:
                    # Affiliations / correspondence notes belong to the byline under the title.
                    owner = next((c for c in flat if c.page == note.page and c.kind == "para"), None)
                if owner is None:
                    # The reference goes at the end of that paragraph (after re-joining).
                    owner = next((c for c in reversed(flat[:idx]) if c.kind == "para"), None)
            if owner is None:
                continue
            owner.notes.append((note_id, body, not by_mark))
            note.kind = "consumed"
            linked += by_mark
            unlinked += not by_mark
        self.stats["footnotes_linked"] = linked
        self.stats["footnotes_anchored_to_paragraph"] = unlinked
        out = []
        for it in items:
            if it.kind == "consumed":
                continue
            if it.kind == "box":
                it.children = [c for c in it.children if c.kind != "consumed"]
            out.append(it)
        return out

    @staticmethod
    def _place_token(cand, pattern, token):
        if cand.kind not in ("para", "list", "caption", "heading", "equation"):
            return False
        if cand.kind == "list":
            for i, li in enumerate(cand.children):
                new, n = re.subn(pattern, token, li, count=1)
                if n:
                    cand.children[i] = new
                    cand.html = " ".join(cand.children)
                    return True
            return False
        new, n = re.subn(pattern, token, cand.html, count=1)
        if n:
            cand.html = new
        return bool(n)

    def _heading_levels(self, items):
        heads = [it for it in items if it.kind == "heading"]
        if not heads:
            return

        def kind_of(text):
            if re.match(r"^\d+\.\d+\.\d+\.?\s", text):
                return "n3"
            if re.match(r"^\d+\.\d+\.?\s", text):
                return "n2"
            if re.match(r"^(\d+\.?|[IVX]+\.)\s+\S", text):
                return "n1"
            if re.match(r"^\(\s*[a-z]\s*\)\s", text) or re.match(r"^[a-z]\)\s", text):
                return "alpha"
            if re.match(r"^\(\s*[ivx]+\s*\)\s", text):
                return "roman"
            if re.match(r"^[A-Z]\.\s+\S", text):
                return "Alpha"
            return None

        kinds = [kind_of(h.text) for h in heads]
        numbered = [k for k in kinds if k]
        top_marker = min((h.level for h, k in zip(heads, kinds) if k == "n1"), default=2)
        first = heads[0]
        title_idx = None
        if kinds[0] is None and first.level == min(h.level for h in heads):
            title_idx = 0
            self.title = first.text
        distinct = sorted({h.level for i, h in enumerate(heads) if i != title_idx})
        stack = []
        fixed_depth = {"n1": 0, "n2": 1, "n3": 2}
        for i, (h, k) in enumerate(zip(heads, kinds)):
            if i == title_idx:
                h.level = 1
                continue
            if k:
                if k in stack:
                    del stack[stack.index(k) + 1:]
                elif k in fixed_depth:
                    stack = stack[:fixed_depth[k]] + [k]
                else:
                    stack.append(k)
                h.level = min(6, 1 + len(stack))
            elif numbered:
                if h.level <= top_marker:
                    stack = []
                    h.level = 2
                else:
                    h.level = min(6, 2 + len(stack))
            else:
                h.level = min(6, 2 + distinct.index(h.level))
        if self.title is None:
            self.title = first.text

    # ---- rendering

    def _ctx(self, page, heading=False):
        if self.an is None:
            return _Ctx(heading=heading, stats=self.stats)
        if page not in self._style_cache:
            try:
                phrases = self.an.styled_phrases(page)
            except Exception:
                phrases = {"italic": [], "bold": []}
            styles = {}
            for style in ("bold", "italic"):
                for ph in phrases.get(style, []):
                    styles[re.sub(r"\s+", " ", ph)] = style
            regex = None
            if styles:
                alts = sorted(styles, key=len, reverse=True)
                regex = re.compile(r"(?<!\w)(?:" + "|".join(r"\s+".join(map(re.escape, a.split(" ")))
                                                           for a in alts) + r")(?!\w)")
            self._style_cache[page] = (regex, styles)
        regex, styles = self._style_cache[page]
        return _Ctx(style_re=regex, styles=styles, heading=heading, stats=self.stats)

    def _save_image(self, item, rect, stem):
        os.makedirs(os.path.join(self.out_dir, self.image_subdir), exist_ok=True)
        data = ext = None
        if self.an is not None:
            try:
                from pdfanalysis import encode_image
                img = self.an.render_region(item.page, rect, dpi=FIGURE_DPI, max_px=FIGURE_MAX_PX)
                if img is not None:
                    data, ext = encode_image(img)
            except Exception:
                data = None
        if data is None:
            for b64 in item.images.values():
                try:
                    data = base64.b64decode(b64)
                    ext = "png" if data[:4] == b"\x89PNG" else "jpg"
                    break
                except Exception:
                    continue
        if data is None:
            return None
        rel = f"{self.image_subdir}/{stem}.{ext}"
        with open(os.path.join(self.out_dir, rel), "wb") as f:
            f.write(data)
        return rel

    def _caption_md(self, item):
        if item.caption is None:
            return ""
        return inline_md(item.caption.html, self._ctx(item.caption.page))

    def _render(self, it, in_box=False):
        ctx = self._ctx(it.page)
        if it.kind == "heading":
            text = inline_md(it.html, self._ctx(it.page, heading=True))
            text = re.sub(r"^\\?\(\s*([a-zA-Z]|[ivx]+|\d+)\s*\)", r"(\1)", text)
            if in_box:
                return f"**{text}**"
            return "#" * it.level + " " + text
        if it.kind in ("para", "equation", "caption"):
            md = escape_line_start(inline_md(it.html, ctx))
            return md
        if it.kind == "footnote":   # left over when nothing could be anchored
            return "::: {.footnote-text}\n" + escape_line_start(inline_md(it.html, ctx)) + "\n:::"
        if it.kind == "list":
            lines = []
            for li in it.children:
                md = inline_md(li, ctx)
                md = re.sub(r"^(?:[•●▪◦‣∙·–—]|\\-)\s*", "", md)
                if md:
                    lines.append("- " + md)
            return "\n".join(lines)
        if it.kind == "code":
            code = plain_text(parse_html(it.html)).strip("\n")
            fence = "~~~~" if "```" in code else "```"
            return f"{fence}\n{code}\n{fence}"
        if it.kind == "figure":
            self._fig_count += 1
            rel = self._save_image(it, it.bbox, f"p{it.page + 1}_fig{self._fig_count}")
            if rel is None:
                return self._caption_md(it)
            self.stats["figures"] = self.stats.get("figures", 0) + 1
            return f"![{self._caption_md(it)}]({rel}){{#fig{self._fig_count}}}"
        if it.kind == "table":
            return self._render_table(it)
        if it.kind == "box":
            inner = "\n\n".join(self._with_notes(c, self._render(c, in_box=True)) for c in it.children)
            return "::: {.box}\n" + inner + "\n:::"
        return ""

    def _render_table(self, it):
        tree = parse_html(it.html)
        table = _find(tree, "table")
        parts = []
        cap = self._caption_md(it)
        if cap:
            parts.append(escape_line_start(cap))
        wide = table is not None and _table_columns(table) >= TABLE_IMAGE_MIN_COLS
        if table is None or wide:
            self._fig_count += 1
            rel = self._save_image(it, it.bbox, f"p{it.page + 1}_table{self._fig_count}")
            if rel:
                parts.append(f"![]({rel}){{.table-image}}")
        if table is not None:
            cls = "table-wrap table-wide" if wide else "table-wrap"
            parts.append(f"::: {{.{cls.replace(' ', ' .')}}}\n```{{=html}}\n{safe_html(table)}\n```\n:::")
        self.stats["tables"] = self.stats.get("tables", 0) + 1
        return "::: {.table-block}\n" + "\n\n".join(parts) + "\n:::"

    def _with_notes(self, it, md):
        if not it.notes:
            return md
        defs = []
        for note_id, body, at_end in it.notes:
            if at_end:
                md += f"[^{note_id}]"
            defs.append(f"[^{note_id}]: " + inline_md(body, self._ctx(it.page)))
        return md + "\n\n" + "\n\n".join(defs)

    # ---- paragraph continuation

    @staticmethod
    def _ends_open(md):
        tail = NOTE_TOKEN.sub("", md)
        tail = re.sub(r"\[\^[^\]]+\]$", "", tail).rstrip(" *_^~")
        if not tail:
            return False
        if tail.endswith("-") and len(tail) > 1 and tail[-2].isalpha():
            return True
        if re.search(r"[.!?:;…][\"'”’)\]]*$", tail):
            return False
        return True

    @staticmethod
    def _starts_continuation(md, prev_md):
        head = md.lstrip("*_ ")
        if not head:
            return False
        if head[0].islower() or head[0] in ",;)]":
            return True
        # "... published by the | Royal Society": only when the paragraph really ends
        # on a function word, and only for real paragraphs (not title/byline lines).
        tail = NOTE_TOKEN.sub("", prev_md).rstrip(" *_")
        last = re.search(r"(?<![\w.])([A-Za-zÀ-ÿ]+)$", tail)
        return bool(last) and last.group(1).lower() in CONTINUATION_WORDS and len(tail) >= 80

    def _merge_continuations(self, blocks):
        """blocks: list of [item, md]. Joins paragraphs split by page/column breaks or
        floats (figures, tables, boxes); the floats end up after the joined paragraph."""
        out = []
        open_idx = None
        for blk in blocks:
            it, md = blk
            if it.kind == "para" and open_idx is not None:
                prev_it, prev_md = out[open_idx]
                drop_cap = re.fullmatch(r"\**[A-ZÀ-Þ]\**", prev_md) and md[:1].islower()
                if (drop_cap or self._starts_continuation(md, prev_md)) and it.page - prev_it.page <= 1:
                    if drop_cap:
                        joined = prev_md.strip("*") + md
                    elif prev_md.endswith("-") and md[:1].islower():
                        joined = prev_md[:-1] + md
                    else:
                        joined = prev_md + " " + md
                    prev_it.notes.extend(it.notes)
                    out[open_idx] = [prev_it, joined]
                    self.stats["paragraphs_rejoined"] = self.stats.get("paragraphs_rejoined", 0) + 1
                    if not self._ends_open(joined):
                        open_idx = None
                    continue
            out.append(blk)
            if it.kind == "para":
                open_idx = len(out) - 1 if self._ends_open(md) else None
            elif it.kind not in ("figure", "table", "box", "caption", "footnote"):
                open_idx = None
        return out

    # ---- main

    def convert(self):
        items = self._collect()
        sample = " ".join(it.text for it in items)
        mathpi = len(_MATHPI_PAIR.findall(sample)) >= 2
        token = _MATHPI.set(mathpi)
        try:
            return self._convert(items)
        finally:
            _MATHPI.reset(token)

    def _convert(self, items):
        items = self._drop_noise(items)
        items = self._figures_and_captions(items)
        items = self._drop_figure_labels(items)
        items = self._boxes(items)
        items = self._footnotes(items)
        self._heading_levels(items)
        # Journal lines etc. printed above the title would otherwise become a
        # chapter of their own (pandoc splits at headings): move them below it.
        title_pos = next((i for i, it in enumerate(items) if it.kind == "heading" and it.level == 1), None)
        if title_pos:
            items = [items[title_pos]] + items[:title_pos] + items[title_pos + 1:]

        blocks = []
        for it in items:
            md = self._render(it)
            if md.strip():
                blocks.append([it, md])
        blocks = self._merge_continuations(blocks)

        out, in_refs = [], False
        for it, md in blocks:
            if it.kind == "heading":
                if in_refs:
                    out.append(":::")
                    in_refs = False
                out.append(self._with_notes(it, md))
                if REFERENCE_HEADINGS.match(it.text):
                    out.append("::: {.references}")
                    in_refs = True
                continue
            if in_refs and it.kind == "list":
                md = "\n\n".join(escape_line_start(line[2:]) for line in md.split("\n"))
            out.append(self._with_notes(it, md))
        if in_refs:
            out.append(":::")
        return "\n\n".join(out).strip() + "\n"


def convert(doc, analyzer=None, out_dir=".", drop_cover=True):
    """Returns (markdown, title, stats)."""
    conv = Converter(doc, analyzer, out_dir, drop_cover=drop_cover)
    md = conv.convert()
    return md, conv.title, conv.stats
