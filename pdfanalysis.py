"""Direct PDF analysis with pypdfium2 (no GPU, no torch).

Fills gaps in marker's output:
  * font styling: recovers italic/bold runs from glyph geometry, which works
    even for subset fonts with obfuscated names and empty style flags;
  * info boxes: filled or framed rectangles that contain text;
  * figures: re-renders figure regions as crisp images straight from the PDF;
  * sub/superscripts that the text layer flattens ("CO2", "model36"), from glyph
    size and baseline;
  * drop caps that marker loses;
  * document checks: publisher cover sheets and text-layer quality.

Rectangles are page fractions with a top-left origin, relative to the visible
(rendered) page area, i.e. the CropBox, not the MediaBox.
"""
import ctypes
import io
import math
import re
import statistics
from array import array
from collections import Counter

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

COVER_SHEET_MARKERS = (
    "downloaded from", "this article cites", "email alerting service", "to subscribe to",
    "terms of use", "jstor", "rapid response", "subject collections", "cite this article",
    "stable url", "published by:", "for personal use only", "this content downloaded",
    "citation for published version", "terms and conditions", "you may not further distribute",
    "accepted manuscript", "receive free email alerts", "view the article online",
    "terms of service", "use of this article is subject to", "reprints and permissions",
    "rights and permissions", "article information",
    # ResearchGate
    "see discussions, stats, and author profiles", "all content following this page was uploaded by",
    "requested enhancement of the downloaded file", "view project", "see profile",
)
ITALIC_NAME = re.compile(r"(italic|oblique|[-.,_]it$|[-.,_]i$|ital\b|-ita\b|cmti|cmmi)", re.I)
BOLD_NAME = re.compile(r"(bold|black|heavy|semibd|[-.,_]b$|[-.,_]bd$|cmbx)", re.I)

ITALIC_SLANT_DELTA = 0.09   # italic glyphs lean ~0.15-0.25; upright ones ~0 +- 0.08
BOLD_INK_RATIO = 1.3        # bold strokes cover >= 30% more area than the body font

SCRIPT_SIZE_RATIO = 0.88    # sub/superscript glyphs are <= ~80% of the base glyph height
SCRIPT_SHIFT_RATIO = 0.12   # ... and their baseline moves by >= ~15% of it
SCRIPT_CHARS = re.compile(r"[0-9A-Za-z,+\-−–*†‡§¶‖'′=().α-ωΔ]")
DROP_CAP_RATIO = 1.8        # a drop cap is >= ~2 lines tall
# Ligatures whose glyph a broken ToUnicode map gives as its first letter only.
LIGATURE_TAILS = {"T": ("h",), "f": ("f", "i", "l", "t", "fi", "fl", "b", "h", "j", "k")}
LIGATURE_RARE = {"b", "h", "j", "k"}   # only some fonts have fb/fh/fj/fk: need intact spellings
LIGATURE_WIDE = 1.35        # such a glyph is ~1.7-2x as wide as the letter alone
LIGATURE_FIT = 0.95         # ... and 3-8% narrower than its letters side by side
LIGATURE_SLACK = 0.07       # widths closer than this don't decide (fi/fl; italics vary more)
# Symbol fonts whose ToUnicode map gives the wrong character (Taylor & Francis'
# "Advent" PDFs: "0.01 20.02" for "0.01 −0.02", "28C" for "2°C").
SYMBOL_FONT_CHARS = {"AdvPS7DA6": {"2": "−", "8": "°"}}
INK_DELTA = 24              # grey-level difference from the page background that counts as drawn
CAPTION_START = re.compile(
    r"\s*(fig(ure|ura)?|figs?|tab(le|la|leau|elle)?|plate|chart|map|scheme|exhibit|illustration|"
    r"graph(ique)?|abb(ildung)?|bild|karte|schema|рис(унок)?|табл(ица)?)\.?\s*[0-9IVXA-Z]{1,4}([.\-–][0-9]+)*\b",
    re.I)


def _wide_text(buf_len_fn, fill_fn):
    n = buf_len_fn()
    if n <= 0:
        return ""
    buf = ctypes.create_string_buffer(n * 2 + 2)
    fill_fn(ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort)), n)
    return buf.raw.decode("utf-16-le", errors="ignore").split("\x00", 1)[0]


class PdfAnalyzer:
    def __init__(self, path):
        self.pdf = pdfium.PdfDocument(path)
        self._font_cache = {}
        self._body_style = None
        self._page_text = {}
        self._glyph_cache = {}
        self._frames = {}
        self._body_font = ""
        self._scanned = None
        self._running = None
        self._ligature_table = None

    def close(self):
        try:
            self.pdf.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.pdf)

    def page_size(self, index):
        return self.pdf[index].get_size()

    def page_text(self, index):
        if index not in self._page_text:
            glyphs = self._glyphs(index)
            if any(g[7] in SYMBOL_FONT_CHARS for g in glyphs):
                self._page_text[index] = "".join(g[0] for g in glyphs)    # with the repaired characters
            else:
                self._page_text[index] = self.pdf[index].get_textpage().get_text_bounded()
        return self._page_text[index]

    def rotation(self, index):
        """The page's /Rotate in degrees (clockwise): 90/270 for landscape pages stored sideways."""
        return self._frame(index)[4]

    # ------------------------------------------------------------ coordinates

    def _frame(self, index):
        """(left, top, width, height) of the visible page area in display space (see
        _to_display), plus the page's /Rotate and its unrotated box."""
        if index not in self._frames:              # per glyph in to_frac: loading the page is slow
            page = self.pdf[index]
            left, bottom, right, top = page.get_bbox()
            rot = page.get_rotation()
            if rot:
                # Display space: origin at the displayed page's bottom left, as marker sees it.
                w, h = (top - bottom, right - left) if rot in (90, 270) else (right - left, top - bottom)
                self._frames[index] = (0.0, h, w or 1.0, h or 1.0, rot, (left, bottom, right, top))
            else:
                w, h = page.get_size()
                self._frames[index] = (left, top, w or 1.0, h or 1.0, 0, None)
        return self._frames[index]

    def _to_display(self, index, l, b, r, t):
        """A user-space box as seen on the displayed page. Pages with /Rotate (landscape
        tables) are stored sideways; marker and the renderer work on the displayed page."""
        rot, unrotated = self._frame(index)[4:]
        if not rot:
            return l, b, r, t
        L, B, R, T = unrotated
        pt = {90: lambda x, y: (y - B, R - x), 180: lambda x, y: (R - x, T - y),
              270: lambda x, y: (T - y, x - L)}[rot]
        (x0, y0), (x1, y1) = pt(l, b), pt(r, t)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def visible_frame(self, index):
        """(dx, dy, w, h, media_w, media_h): the visible area within the MediaBox, in
        points from the MediaBox's top-left corner. marker gives boxes in MediaBox
        coordinates, which differ from ours wherever a (possibly inherited) CropBox is set."""
        page = self.pdf[index]
        if page.get_rotation():
            return None
        left, top, w, h = self._frame(index)[:4]
        ml, mb, mr, mt = page.get_mediabox()
        return left - ml, mt - top, w, h, mr - ml, mt - mb

    def to_frac(self, index, l, b, r, t):
        """Display-space box (see _to_display) -> page fractions (top-left origin, visible area)."""
        left, top, w, h = self._frame(index)[:4]
        return ((l - left) / w, (top - t) / h, (r - left) / w, (top - b) / h)

    # ---------------------------------------------------- glyph-level text

    def _glyphs(self, index):
        """[(char, generated, height, origin_x, origin_y, right, (l, b, r, t), font, weight)]
        in text order, display space (_to_display). Height is the loose (font ascent-descent)
        box, so it tracks the effective font size even where FPDFText_GetFontSize ignores
        the text matrix."""
        if index in self._glyph_cache:
            return self._glyph_cache[index]
        tp = self.pdf[index].get_textpage()
        rotated = bool(self._frame(index)[4])
        out = []
        x, y = ctypes.c_double(), ctypes.c_double()
        matrix = pdfium_c.FS_MATRIX()
        name_buf, flags = ctypes.create_string_buffer(128), ctypes.c_int()
        for i in range(tp.count_chars()):
            code = pdfium_c.FPDFText_GetUnicode(tp.raw, i)
            ch = chr(code) if 0 < code < 0x110000 else ""
            gen = pdfium_c.FPDFText_IsGenerated(tp.raw, i) == 1
            l, b, r, t = tp.get_charbox(i, loose=True)
            pdfium_c.FPDFText_GetCharOrigin(tp.raw, i, ctypes.byref(x), ctypes.byref(y))
            if rotated and r - l < 1e-6 and t - b < 1e-6 and ch.strip() \
                    and pdfium_c.FPDFText_GetMatrix(tp.raw, i, ctypes.byref(matrix)):
                # pdfium's loose box collapses to a point for text set sideways: rebuild it
                # from the glyph's em (0.24 em below the baseline, 0.76 above) along the
                # text's up direction, and its ink along the line.
                up = math.hypot(matrix.c, matrix.d)
                em = pdfium_c.FPDFText_GetFontSize(tp.raw, i) * up
                if em > 0:
                    ux, uy = matrix.c / up, matrix.d / up
                    tl, tb, tr, tt = tp.get_charbox(i)
                    xs = [tl, tr, x.value - 0.24 * em * ux, x.value + 0.76 * em * ux]
                    ys = [tb, tt, y.value - 0.24 * em * uy, y.value + 0.76 * em * uy]
                    l, b, r, t = min(xs), min(ys), max(xs), max(ys)
            l, b, r, t = self._to_display(index, l, b, r, t)
            ox, oy = self._to_display(index, x.value, y.value, x.value, y.value)[:2]
            n = pdfium_c.FPDFText_GetFontInfo(tp.raw, i, name_buf, 128, ctypes.byref(flags))
            font = _font_base(name_buf.value.decode("latin1", "replace")) if n else ""
            ch = SYMBOL_FONT_CHARS.get(font, {}).get(ch, ch)
            out.append((ch, gen, t - b, ox, oy, r, (l, b, r, t), font,
                        pdfium_c.FPDFText_GetFontWeight(tp.raw, i)))
        self._glyph_cache[index] = out
        if len(self._glyph_cache) > 4:            # passes revisit nearby pages; keep a few
            self._glyph_cache.pop(next(iter(self._glyph_cache)))
        return out

    def script_runs(self, index):
        """Sub/superscripts on a page as [(prefix, run, kind)], kind 'sup' or 'sub'.

        prefix is the text directly before the run on the same line (the word it
        is attached to). Only runs that are *consistently* scripted on the page
        are returned (every plain-text occurrence of prefix+run is scripted), so
        applying them to marker's text by string matching is safe."""
        glyphs = self._glyphs(index)
        heights = [g[2] for g in glyphs if g[2] > 0 and not g[1] and g[0].strip()]
        if not heights:
            return []
        body_h = statistics.median(heights)
        found = Counter()
        i, n = 0, len(glyphs)
        while i < n:
            ch, gen, h, x, y, *_ = glyphs[i]
            if gen or not ch.strip() or h <= 0 or h > 1.5 * body_h:
                i += 1
                continue
            # i is a normal-size glyph; is it followed by a scripted run?
            j = i + 1
            if j < n and glyphs[j][1] and not glyphs[j][0].strip():
                j += 1                     # one generated space ("Dolan 1")
            run, kind, k = [], None, j
            while k < n:
                c2, gen2, h2, x2, y2, *_ = glyphs[k]
                dy = y2 - y
                if (gen2 or h2 <= 0 or h2 > SCRIPT_SIZE_RATIO * h or abs(dy) < SCRIPT_SHIFT_RATIO * h
                        or abs(dy) > 0.8 * h or not SCRIPT_CHARS.fullmatch(c2 or " ")):
                    break
                k_kind = "sup" if dy > 0 else "sub"
                if kind and k_kind != kind:
                    break
                kind = k_kind
                run.append(c2)
                k += 1
            text = "".join(run).strip(",.()")
            if text and len(run) <= 12 and glyphs[j][3] - glyphs[i][5] < 2 * h:
                p = i
                while p > 0 and i - p < 24:
                    c0, gen0, h0, *_ = glyphs[p - 1]
                    if gen0 or not c0.strip() or abs(h0 - h) > 0.1 * h or not re.match(r"[\w.\-]", c0):
                        break
                    p -= 1
                prefix = "".join(g[0] for g in glyphs[p:i + 1]).lstrip(".-")
                if prefix and "".join(run).startswith(text):
                    found[(prefix, text, kind)] += 1
                i = k
                continue
            i += 1
        page_text = re.sub(r"\s+", " ", self.page_text(index))
        result = []
        for (prefix, run, kind), count in found.items():
            total = len(re.findall(re.escape(prefix) + r" ?" + re.escape(run), page_text))
            if total <= count:
                result.append((prefix, run, kind))
        return result

    def drop_caps(self, index):
        """[(letter, following_text)] for oversized initials that start a paragraph."""
        glyphs = self._glyphs(index)
        heights = [g[2] for g in glyphs if g[2] > 0 and not g[1] and g[0].strip()]
        if not heights:
            return []
        body_h = statistics.median(heights)
        out = []
        for i, (ch, gen, h, *_rest) in enumerate(glyphs):
            if gen or not ch.isupper() or h < DROP_CAP_RATIO * body_h:
                continue
            follow = "".join(g[0] for g in glyphs[i + 1:i + 40])
            follow = re.split(r"\s", follow, maxsplit=1)[0]
            nxt = glyphs[i + 1] if i + 1 < len(glyphs) else None
            if nxt and nxt[2] < h / DROP_CAP_RATIO * 1.2 and follow[:1].islower() and len(follow) >= 2:
                out.append((ch, follow))
        return out

    # ------------------------------------------------------------------ checks

    def text_quality(self, sample_pages=8):
        """Rough text-layer health: (avg chars/page, fraction of bad chars)."""
        n = len(self.pdf)
        if n == 0:
            return 0.0, 1.0
        step = max(1, n // sample_pages)
        chars = bad = pages = 0
        for i in range(0, n, step):
            t = self.page_text(i)
            core = re.sub(r"\s", "", t)
            chars += len(core)
            bad += core.count("�") + 5 * t.count("(cid:")
            bad += sum(1 for ch in core if 0xE000 <= ord(ch) <= 0xF8FF)
            pages += 1
        return chars / max(pages, 1), bad / max(chars, 1)

    def is_born_digital(self):
        per_page, bad_ratio = self.text_quality()
        return per_page >= 200 and bad_ratio < 0.02

    def cover_pages(self):
        """Publisher/repository cover sheets, first page (JSTOR, HighWire, Royal
        Society...) or last page (Science/AAAS "View the article online")."""
        n = len(self.pdf)
        if n < 4:
            return []
        found = []
        for i in (0, n - 1):
            text = self.page_text(i).lower()
            hits = sum(1 for m in COVER_SHEET_MARKERS if m in text)
            if hits >= 3 and (i == 0 or len(text) < 1500):
                found.append(i)
        return found

    def has_cover_sheet(self):
        return 0 in self.cover_pages()

    def _scan(self):
        """Per page, what the document-wide checks need (running heads, ligatures),
        read in one pass: reading glyphs is the costly part on long books. Columns,
        not per-glyph tuples, to stay small."""
        if self._scanned is None:
            self._scanned = []
            for i in range(len(self.pdf)):
                lines = [(r, n, t) for r, n, t in self.text_lines(i)
                         if n >= 3 and not (0.2 < r[1] and r[3] < 0.85)]
                chars, gens, fonts = [], bytearray(), []
                ratios, lefts, rights = array("f"), array("f"), array("f")
                for ch, gen, h, _x, _y, _r, box, font, _w in self._glyphs(i):
                    chars.append(ch)
                    gens.append(gen)
                    fonts.append(font)
                    ratios.append((box[2] - box[0]) / h if h > 0 else 0.0)
                    lefts.append(box[0])
                    rights.append(box[2])
                self._scanned.append({"margin_lines": lines, "chars": "".join(chars) if all(
                    len(c) == 1 for c in chars) else chars, "gens": gens, "fonts": fonts,
                    "ratios": ratios, "lefts": lefts, "rights": rights})
        return self._scanned

    def running_heads(self):
        """{page: [(rect, key)]} of running heads and feet: lines near the top or bottom
        that recur at the same height on >= 3 pages, compared with digits masked
        ("4645 L. Skinner", "Facing future climate change 4631"). marker usually
        labels them, but not always."""
        if self._running is None:
            seen = {}
            for i, page in enumerate(self._scan()):
                for rect, n, text in page["margin_lines"]:
                    key = _running_key(text)
                    if len(key) >= 3:
                        seen.setdefault(key, []).append((i, rect))
            self._running = {}
            for key, hits in seen.items():
                for i, rect in hits:
                    pages = {j for j, r in hits if abs(r[1] - rect[1]) < 0.015}
                    if len(pages) >= 3:
                        self._running.setdefault(i, []).append((rect, key))
        return self._running

    def is_running_head(self, index, rect, text):
        """Whether `text` at `rect` (page fractions) is one of the page's running heads."""
        key = _running_key(text)
        return any(key == k and min(rect[3], r[3]) - max(rect[1], r[1]) > 0
                   for r, k in self.running_heads().get(index, ()))

    # ------------------------------------------------------------- font style

    @staticmethod
    def _glyph_points(font, ch, size=100.0):
        gp = pdfium_c.FPDFFont_GetGlyphPath(font, ord(ch), ctypes.c_float(size))
        if not gp:
            return []
        pts = []
        for i in range(pdfium_c.FPDFGlyphPath_CountGlyphSegments(gp)):
            seg = pdfium_c.FPDFGlyphPath_GetGlyphPathSegment(gp, i)
            x, y = ctypes.c_float(), ctypes.c_float()
            if pdfium_c.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y)):
                pts.append((x.value, y.value, pdfium_c.FPDFPathSegment_GetType(seg)))
        return pts

    def _font_metrics(self, font):
        """(slant, ink) of a font from glyph outlines; None where not measurable."""
        slants, inks = [], []
        for ch in "lIdhbkfT":
            pts = self._glyph_points(font, ch)
            if len(pts) < 4:
                continue
            ys = [p[1] for p in pts]
            lo, hi = min(ys), max(ys)
            h = hi - lo
            if h <= 0:
                continue
            top = [p[0] for p in pts if p[1] > hi - 0.2 * h]
            bot = [p[0] for p in pts if p[1] < lo + 0.2 * h]
            if top and bot:
                slants.append((sum(top) / len(top) - sum(bot) / len(bot)) / h)
        for ch in "nolehI":
            pts = self._glyph_points(font, ch)
            if len(pts) < 3:
                continue
            area, start, prev = 0.0, None, None
            for x, y, t in pts:
                if t == pdfium_c.FPDF_SEGMENT_MOVETO:
                    if prev and start:
                        area += prev[0] * start[1] - start[0] * prev[1]
                    start = (x, y)
                elif prev:
                    area += prev[0] * y - x * prev[1]
                prev = (x, y)
            if prev and start:
                area += prev[0] * start[1] - start[0] * prev[1]
            ys = [p[1] for p in pts]
            h = max(ys) - min(ys)
            if h > 0:
                inks.append(abs(area) / 2 / (h * h))
        med = lambda v: sorted(v)[len(v) // 2] if v else None
        return med(slants), med(inks)

    def _font_key(self, font):
        buf = ctypes.create_string_buffer(256)
        pdfium_c.FPDFFont_GetFontName(font, buf, 256)
        name = buf.value.decode("latin1", errors="replace")
        return name or f"anon-{ctypes.cast(font, ctypes.c_void_p).value}"

    def _text_objects(self, index):
        """(font_key, text, (l, b, r, t)) for each top-level text object of a page."""
        page = self.pdf[index]
        tp = page.get_textpage()
        out = []
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_TEXT]):
            font = pdfium_c.FPDFTextObj_GetFont(obj.raw)
            if not font:
                continue
            key = self._font_key(font)
            if key not in self._font_cache:
                self._font_cache[key] = self._font_metrics(font)
            text = _wide_text(
                lambda: pdfium_c.FPDFTextObj_GetText(obj.raw, tp.raw, None, 0),
                lambda b, n: pdfium_c.FPDFTextObj_GetText(obj.raw, tp.raw, b, n))
            if text:
                out.append((key, text, self._to_display(index, *obj.get_pos())))
        return out

    def _partial_word(self, index, pos):
        """Whether a text object at `pos` is part of a word: a letter or digit adjoins
        it on its line ("at" of "NATURE" set in small caps as separate pieces)."""
        l, b, r, t = pos
        h = max(t - b, 1e-6)
        for ch, gen, gh, x, y, right, box, *_ in self._glyphs(index):
            if gen or not ch.isalnum() or gh <= 0:
                continue
            cy = (box[1] + box[3]) / 2
            if not b - 0.2 * h <= cy <= t + 0.2 * h:
                continue
            if 0 <= l - box[2] < 0.12 * gh or 0 <= box[0] - r < 0.12 * gh:
                return True
        return False

    def _body_metrics(self):
        if self._body_style is None:
            counts = Counter()
            n = len(self.pdf)
            for i in range(0, n, max(1, n // 6)):
                for key, text, _pos in self._text_objects(i):
                    counts[key] += len(text)
            body = None
            for key, _ in counts.most_common():
                slant, ink = self._font_cache.get(key, (None, None))
                if slant is not None and ink is not None:
                    body = (slant, ink)
                    self._body_font = key.split("+")[-1]
                    break
            self._body_style = body or (0.0, None)
        return self._body_style

    def styled_phrases(self, index):
        """{'italic': [...], 'bold': [...]} phrases that are *consistently* styled on a page.

        A phrase is only reported if every occurrence on the page is in that
        style, so applying it to marker's text by string matching is safe.
        """
        body_slant, body_ink = self._body_metrics()
        found = {"italic": Counter(), "bold": Counter()}
        for key, text, pos in self._text_objects(index):
            slant, ink = self._font_cache.get(key, (None, None))
            phrase = re.sub(r"\s+", " ", text).strip()
            if len(phrase) < 2 or not re.search(r"\w\w", phrase):
                continue
            if len(phrase) <= 4 and self._partial_word(index, pos):
                continue
            base = key.split("+")[-1]
            # Relative to the body font: some PDFs set all running text in a bold
            # (or italic) font, and then that is the normal style, not emphasis.
            if ((ITALIC_NAME.search(base) and not ITALIC_NAME.search(self._body_font))
                    or (slant is not None and slant - (body_slant or 0) > ITALIC_SLANT_DELTA)):
                found["italic"][phrase] += 1
            elif ((BOLD_NAME.search(base) and not BOLD_NAME.search(self._body_font))
                  or (ink is not None and body_ink and ink / body_ink > BOLD_INK_RATIO)):
                found["bold"][phrase] += 1
        page_text = re.sub(r"\s+", " ", self.page_text(index))
        result = {}
        for style, counter in found.items():
            keep = []
            for phrase, styled in counter.items():
                total = len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", page_text))
                if 0 < total <= styled:
                    keep.append(phrase)
            result[style] = sorted(keep, key=len, reverse=True)
        return result

    # ------------------------------------------------------- figure extents

    def line_style(self, index, text):
        """(size_pt, font, weight) of the PDF line where `text` starts, or None.
        Size is the median glyph height (half-point steps), font the one used by
        most of its glyphs, so headings of one class share a key."""
        # Long enough to tell "Dissolution of salts" from "Dissolution of gases";
        # a heading broken over two lines still matches on its first line.
        want = _norm(text)[:40]
        if len(want) < 3:
            return None
        cur, styles = [], []
        for g in self._glyphs(index) + [("\n", True, 0, 0, 0, 0, None, "", 0)]:
            if g[0] in ("\r", "\n"):
                line = "".join(x[0] for x in cur)
                norm = _norm(line)
                if norm and (norm.startswith(want) or (len(norm) >= 3 and want.startswith(norm))):
                    glyphs = [x for x in cur if x[0].strip() and x[2] > 0 and not x[1]]
                    if glyphs:
                        size = round(statistics.median(x[2] for x in glyphs) * 2) / 2
                        font, _ = Counter(x[7] for x in glyphs).most_common(1)[0]
                        weight = Counter(x[8] for x in glyphs).most_common(1)[0][0]
                        styles.append((size, font, weight))
                cur = []
                continue
            cur.append(g)
        return styles[0] if len(styles) == 1 else None      # ambiguous: several matches

    def block_styles(self, index, text):
        """(italic, bold) share of the glyphs that set `text` on the page, judged
        against the body font (a bold body font is normal, not bold). None if the
        text cannot be located unambiguously."""
        self._body_metrics()
        glyphs = [g for g in self._glyphs(index) if not g[1] and g[0].strip()]
        stream = "".join(_norm(g[0]) or "\x00" for g in glyphs)
        want = _norm(text)
        if len(want) < 8 or len(stream) != len(glyphs):
            return None
        start = stream.find(want[:24])
        if start < 0 or stream.find(want[:24], start + 1) >= 0:
            return None
        span = glyphs[start:start + len(want)]
        body = self._body_font
        slants = {_font_base(key): m for key, m in self._font_cache.items() if m}
        body_slant = (self._body_style or (0.0, None))[0] or 0.0
        italic = bold = 0
        for g in span:
            font = g[7]
            slant = (slants.get(font) or (None, None))[0]
            if ((ITALIC_NAME.search(font) and not ITALIC_NAME.search(body))
                    or (slant is not None and slant - body_slant > ITALIC_SLANT_DELTA)):
                italic += 1
            if BOLD_NAME.search(font) and not BOLD_NAME.search(body):
                bold += 1
        return italic / len(span), bold / len(span)

    def page_words(self, index):
        """Words of the text layer, soft hyphens and their line breaks removed, lost
        ligature letters restored."""
        text = "".join(g[0] for g in self._glyphs(index))
        text = re.sub(r"[\ufffe\x02]\s*", "", text)
        repairs = dict(self.ligature_repairs(index))
        if not repairs:
            return text.split()
        out = []
        for w in text.split():
            m = re.fullmatch(r"(\W*)(.*?)(\W*)", w)
            out.append(m.group(1) + repairs.get(m.group(2), m.group(2)) + m.group(3))
        return out

    # ------------------------------------------------------------- ligatures

    def ligature_repairs(self, index):
        """[(broken, fixed)] words of a page whose ligature glyphs the text layer maps
        to their first letter only ("Te"/"The", "efects"/"effects", "food"/"flood"), a
        common font-embedding fault (e.g. Springer Nature PDFs). Only words whose every
        occurrence on the page is broken, so replacing them in marker's text is safe."""
        return self._ligatures().get(index, [])

    def _ligatures(self):
        if self._ligature_table is not None:
            return self._ligature_table
        self._ligature_table = {}
        pages, ratios = self._scan(), {}
        for page in pages:
            for ch, gen, ratio, font in zip(page["chars"], page["gens"], page["ratios"], page["fonts"]):
                if ratio > 0 and not gen and ch.isalpha():
                    ratios.setdefault((font, ch), []).append(ratio)
        med = {k: statistics.median(v) for k, v in ratios.items() if len(v) >= 5}
        # A letter rare in its font (a capital in italics): estimate it from the other
        # fonts, scaled by how wide this font sets the letters both have.
        by_char = {}
        for (font, ch), v in ratios.items():
            by_char.setdefault(ch, []).extend(v)
        overall = {ch: statistics.median(v) for ch, v in by_char.items() if len(v) >= 5}
        scale = {}
        for (font, ch), m in med.items():
            scale.setdefault(font, []).append(m / overall[ch])
        for font, ch in set(ratios) | {(f, c) for f in scale for c in "fhilt"}:
            if (font, ch) not in med and font in scale and ch in overall:
                med[(font, ch)] = overall[ch] * statistics.median(scale[font])

        # Over-wide first letters of a ligature whose other letters are missing,
        # grouped by font and exact width: one group is one ligature glyph.
        groups = {}                      # (font, ch, width) -> [(page, glyph index)]
        for p, page in enumerate(pages):
            chars, lefts, rights = page["chars"], page["lefts"], page["rights"]
            for i, (ch, gen, ratio, font) in enumerate(zip(chars, page["gens"], page["ratios"], page["fonts"])):
                if ch not in LIGATURE_TAILS or gen:
                    continue
                normal = med.get((font, ch))
                if not normal or ratio < LIGATURE_WIDE * normal:
                    continue
                nxt = next((k for k in range(i + 1, min(i + 3, len(chars))) if chars[k].strip()), None)
                if nxt is not None and lefts[nxt] < rights[i] - 0.3 * (rights[i] - lefts[i]):
                    continue             # the next letter overlaps: mapped correctly
                groups.setdefault((font, ch, round(ratio, 3)), []).append((p, i))
        if not groups:
            return self._ligature_table

        # Words as glyph-index runs; a soft hyphen joins across its line break.
        tokens, token_of = [], []
        for page in pages:
            toks, cur, joining = [], [], False
            for i, ch in enumerate(page["chars"]):
                if ch in "\ufffe\x02":
                    joining = True
                elif not ch.strip():
                    if cur and not joining:
                        toks.append(cur)
                        cur = []
                else:
                    cur.append(i)
                    joining = False
            if cur:
                toks.append(cur)
            tokens.append(toks)
            token_of.append({i: t for t, idx in enumerate(toks) for i in idx})

        def spell(p, tok, extra=None):
            text = "".join(pages[p]["chars"][i] + (extra or {}).get(i, "") for i in tok)
            return re.sub(r"^\W+|\W+$", "", text)

        flagged = {(p, i) for hits in groups.values() for p, i in hits}
        vocab = Counter()
        for p, toks in enumerate(tokens):
            vocab.update(spell(p, t).lower() for t in toks if not any((p, i) in flagged for i in t))

        # Letter trigrams of the intact words: which spelling looks like this document's
        # language where the width cannot tell (fi/fl and ffi/ffl are equally wide).
        trigrams = Counter()

        def learn(word, n=1):
            w = f"^{word.lower()}$"
            for k in range(len(w) - 2):
                trigrams[w[k:k + 3]] += n

        for w, n in vocab.items():
            learn(w, n)

        def plausible(word):
            w, total = f"^{word.lower()}$", sum(trigrams.values()) or 1
            return sum(math.log((trigrams[w[k:k + 3]] + 0.5) / total) for k in range(len(w) - 2))

        def fit(key, tail):
            font, ch, ratio = key
            return abs(LIGATURE_FIT * (med[(font, ch)] + sum(med[(font, c)] for c in tail)) - ratio)

        # Which letters each glyph stands for: the spelling the document has intact elsewhere.
        scored = []
        for key, hits in groups.items():
            font, ch, ratio = key
            fits = [t for t in LIGATURE_TAILS[ch]
                    if all((font, c) in med for c in t) and fit(key, t) <= 0.15 * ratio]
            spelt = {t: [spell(p, tokens[p][token_of[p][i]], {i: t}) for p, i in hits] for t in fits}
            votes = {t: sum(1 for w in spelt[t] if vocab[w.lower()]) for t in fits}
            scored.append((max(votes.values(), default=0), len(hits), key, fits, votes, spelt))
        # Best-attested glyphs first; their repaired words then inform the rest.
        scored.sort(key=lambda s: (-s[0], -s[1]))
        used, choice, confirmed = set(), {}, set()
        for _, _, key, fits, votes, spelt in scored:
            font, ch, ratio = key
            free = [t for t in fits if (font, ch, t) not in used and (votes[t] or t not in LIGATURE_RARE)]
            if not free:
                continue
            closest = min(fit(key, t) for t in free)
            best = max(free, key=lambda t: (votes[t], fit(key, t) <= closest + LIGATURE_SLACK * ratio,
                                            sum(plausible(w) for w in spelt[t])))
            if votes[best] > 0:
                confirmed.add(font)
            elif font not in confirmed:
                continue                 # no evidence that this font's ligatures are broken
            used.add((font, ch, best))   # each tail is one glyph per font
            choice[key] = best
            for w in spelt[best]:
                learn(w)

        # Per page, the words to replace: only where every occurrence is broken.
        for p, toks in enumerate(tokens):
            extra = {i: choice[k] for k, hits in groups.items() if k in choice for q, i in hits if q == p}
            if not extra:
                continue
            raw = Counter(spell(p, t) for t in toks)
            pairs = Counter((spell(p, t), spell(p, t, extra)) for t in toks if any(i in extra for i in t))
            self._ligature_table[p] = [(old, new) for (old, new), n in pairs.items()
                                       if old and old != new and raw[old] == n]
        return self._ligature_table

    def glyph_height(self, index):
        """Median glyph height on a page in PDF points: the body text's size there."""
        heights = [g[2] for g in self._glyphs(index) if not g[1] and g[0].strip() and g[2] > 0]
        return statistics.median(heights) if heights else None

    def region_words(self, index, rect):
        """Words of the text layer whose glyphs lie inside `rect` (page fractions), in text order."""
        words, cur = [], []
        for g in self._glyphs(index) + [(" ", True, 0, 0, 0, 0, None, "", 0)]:
            ch, box = g[0], g[6]
            inside = False
            if box is not None and ch.strip() and ch not in "￾\x02":
                x0, y0, x1, y1 = self.to_frac(index, *box)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                inside = rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3]
            if inside:
                cur.append(ch)
            elif ch not in "￾\x02" and cur:
                words.append("".join(cur))
                cur = []
        return words

    def text_lines(self, index):
        """[(rect, chars, text)] per text line (rotated text yields one entry per glyph)."""
        lines, cur, prev = [], [], None

        def flush():
            boxes = [g[6] for g in cur if not g[1] and g[0].strip()]
            if boxes:
                l, b = min(x[0] for x in boxes), min(x[1] for x in boxes)
                r, t = max(x[2] for x in boxes), max(x[3] for x in boxes)
                text = "".join(g[0] for g in cur).strip()
                lines.append((self.to_frac(index, l, b, r, t), len(boxes), text))

        for g in self._glyphs(index):
            ch, gen, h, x, y = g[:5]
            if ch in ("\r", "\n"):
                flush()
                cur, prev = [], None
                continue
            if not ch.strip() or h <= 0:
                cur.append(g)
                continue
            if prev is not None and (abs(y - prev[4]) > 0.6 * max(h, prev[2]) or x < prev[3] - 1):
                flush()
                cur = []
            cur.append(g)
            prev = g
        flush()
        return lines

    def _prose_lines(self, index, known_texts=(), lines=None):
        """Boxes of text lines that are running text, captions, headings or running
        heads, i.e. never part of a figure. Running text is wide, horizontal and
        dense (axis tick rows are sparse, axis titles short); a paragraph's short
        last line is recognised by body-size type starting at the text column's
        edge; anything else is excluded when it is part of `known_texts` (text
        blocks marker found on the page, e.g. a heading right above a photo)."""
        lines = self.text_lines(index) if lines is None else lines
        known = [_norm(t) for t in known_texts if t]
        # Running text relative to this page's own type: wide lines whose character
        # density is close to the body text's (large-print books, small formats and
        # dense journals all differ in absolute terms; tick rows are always sparse).
        wide = [(fr, n) for fr, n, _ in lines if n >= 20 and fr[2] - fr[0] >= 0.25 and fr[2] - fr[0] > fr[3] - fr[1]]
        body_density = statistics.median(n / (fr[2] - fr[0]) for fr, n in wide) if wide else 0
        running = [fr for fr, n in wide if n / (fr[2] - fr[0]) >= 0.6 * body_density]
        out = list(running)
        if running:
            body_h = statistics.median(fr[3] - fr[1] for fr in running)
            left = statistics.median(fr[0] for fr in running)
        for fr, n, text in lines:
            if fr in running:
                continue
            w, h = fr[2] - fr[0], fr[3] - fr[1]
            core = _norm(text).rstrip("-")
            if (CAPTION_START.match(text) or fr[3] < 0.07 or fr[1] > 0.93
                    or (len(core) >= 8 and any(core in k for k in known))
                    or (running and n >= 8 and w > h and n / max(w, 1e-6) >= 0.6 * body_density
                        and abs(h - body_h) <= 0.15 * body_h and left - 0.005 <= fr[0] <= left + 0.04)):
                out.append(fr)
        return out, (statistics.median(fr[3] - fr[1] for fr in running) if running else 0)

    def content_rect(self, index, rect, known_texts=(), others=(), reach=0.25, gap_pt=12.0):
        """The visible extent of the figure that `rect` (marker's layout box, which can
        be off by several lines) points at. The page is rendered with running text,
        captions and running heads blanked out; starting from the ink inside `rect`,
        the box grows by all ink within `gap_pt` (axis labels, legends, panel
        letters, the other panels) but at most `reach` (page fraction) beyond `rect`.
        Working on pixels makes clip paths, transforms and stray path geometry
        irrelevant. `others` are the boxes of the page's other figures/tables: if one
        overlaps `rect`, the start is narrowed to the band of ink (split at blank
        rows) that belongs to `rect` rather than to them. Returns page fractions,
        or None if `rect` holds no ink or the page has no text layer (a pure scan: text
        could not be told from figure)."""
        from PIL import ImageDraw
        page = self.pdf[index]
        lines = self.text_lines(index)
        if not lines:
            return None
        W, H = page.get_size()
        img = page.render(scale=1.0).to_pil().convert("L")
        pw, ph = img.size
        hist = img.histogram()
        background = max(range(256), key=hist.__getitem__)    # paper colour, not assumed white
        draw = ImageDraw.Draw(img)
        prose, body_h = self._prose_lines(index, known_texts, lines)
        for fr in prose:
            draw.rectangle([fr[0] * pw - 1, fr[1] * ph - 1, fr[2] * pw + 1, fr[3] * ph + 1], fill=background)
        mask = img.point(lambda v: 255 if abs(v - background) > INK_DELTA else 0)
        gap_pt = max(gap_pt, body_h * H)                       # at least one body line
        px = lambda r: (int(r[0] * pw), int(r[1] * ph), int(r[2] * pw + 0.999), int(r[3] * ph + 0.999))
        region = px((rect[0] - reach, rect[1] - reach, rect[2] + reach, rect[3] + reach))
        region = (max(0, region[0]), max(0, region[1]), min(pw, region[2]), min(ph, region[3]))

        def ink(box):
            box = (max(box[0], region[0]), max(box[1], region[1]), min(box[2], region[2]), min(box[3], region[3]))
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            bb = mask.crop(box).getbbox()
            return None if bb is None else (box[0] + bb[0], box[1] + bb[1], box[0] + bb[2], box[1] + bb[3])

        g = max(1, round(gap_pt * pw / W))

        def bands_of(box, axis):
            """Ink bands inside `box` along an axis (1: rows, 0: columns), split
            where >= the gap of blank lines separates them."""
            lo, hi = box[axis], box[axis + 2]
            bands, start, blank = [], None, 0
            for v in range(lo, hi):
                strip = (box[0], v, box[2], v + 1) if axis == 1 else (v, box[1], v + 1, box[3])
                if mask.crop(strip).getbbox():
                    if start is None:
                        start = v
                    blank = 0
                elif start is not None:
                    blank += 1
                    if blank >= g:
                        bands.append((start, v - blank + 1))
                        start, blank = None, 0
            if start is not None:
                bands.append((start, hi - blank))
            return [(box[0], a, box[2], b) if axis == 1 else (a, box[1], b, box[3]) for a, b in bands]

        def core(box, rivals, trim=True):
            """Ink inside `box` without slivers cut off at its edges (the border of a
            neighbouring table above or beside it, a caption's last line) and, where a
            rival box overlaps, only the part that belongs to `box` rather than to them."""
            found = ink(box)
            near = (box[0] - g, box[1] - g, box[2] + g, box[3] + g)
            rivals = [r for r in rivals if _intersection(r, near) > 0]

            def continues_into_rival(b, axis):
                """The band touches an edge of `box` and its ink runs on, without a
                gap, into a rival's box beyond that edge: it is the rival's."""
                for at_start in (True, False):
                    edge = box[axis] if at_start else box[axis + 2]
                    if abs((b[axis] if at_start else b[axis + 2]) - edge) > 2:
                        continue
                    lo, hi = (edge - g, edge) if at_start else (edge, edge + g)
                    strip = (b[0], lo, b[2], hi) if axis == 1 else (lo, b[1], hi, b[3])
                    for r in rivals:
                        part = (max(strip[0], r[0]), max(strip[1], r[1]), min(strip[2], r[2]), min(strip[3], r[3]))
                        if part[2] > part[0] and part[3] > part[1] and ink(part):
                            return True
                return False

            for axis in (1, 0):                      # rows, then columns
                if found is None:
                    return None
                bands = bands_of(found, axis)
                if rivals and len(bands) > 1:
                    bands = [b for b in bands if not continues_into_rival(b, axis)] or bands
                if trim and len(bands) > 1:
                    total = found[axis + 2] - found[axis]
                    at_edge = lambda b: b[axis] <= box[axis] + 2 or b[axis + 2] >= box[axis + 2] - 2
                    kept = [b for b in bands if not (at_edge(b) and b[axis + 2] - b[axis] < 0.2 * total)]
                    bands = kept or bands
                if rivals:
                    score = lambda b: _intersection(b, box) - max(_intersection(b, r) for r in rivals)
                    bands = [b for b in bands if score(b) > 0.25 * _area(b)] or [max(bands, key=score)]
                union = bands[0]
                for b in bands[1:]:
                    union = _union(union, b)
                found = ink(union) or found
            return found

        own = px(rect)
        rivals = [px(o) for o in others]
        cur = core(own, rivals)
        if cur is None:
            return None
        def grow(box, avoid):
            """Extend `box` side by side by all ink within the gap, never reaching
            further into any of the `avoid` boxes than it already does."""
            changed = True
            while changed:
                changed = False
                for side in range(4):
                    probe = list(box)
                    probe[side] += g if side >= 2 else -g
                    grown = ink(tuple(probe))
                    nxt = _union(box, grown) if grown else box
                    if nxt != box and not any(_intersection(nxt, a) > _intersection(box, a) for a in avoid):
                        box, changed = nxt, True
            return box

        # Neighbouring figures/tables: their full extent (grown like ours, but not
        # into our start) fences the crop in. On the side where a neighbour lies the
        # crop stops just before it; the other directions stay free, so a diagram
        # next to a table can still grow upwards to take in its labels.
        neighbours = [grow(c, [cur]) for c in (core(r, [own], trim=False) for r in rivals) if c]
        fence = [region[0], region[1], region[2], region[3]]
        for c in neighbours:
            dx = ((c[0] + c[2]) - (cur[0] + cur[2])) / max(1, (c[2] - c[0]) + (cur[2] - cur[0]))
            dy = ((c[1] + c[3]) - (cur[1] + cur[3])) / max(1, (c[3] - c[1]) + (cur[3] - cur[1]))
            if abs(dx) >= abs(dy):
                if dx > 0:
                    fence[2] = min(fence[2], c[0])
                else:
                    fence[0] = max(fence[0], c[2])
            elif dy > 0:
                fence[3] = min(fence[3], c[1])
            else:
                fence[1] = max(fence[1], c[3])
        region = tuple(fence)
        cur = ink(cur)                 # ink() clips to the (fenced) region
        if cur is None:
            return None
        cur = grow(cur, neighbours)
        pad_x, pad_y = round(3.0 * pw / W), round(3.0 * ph / H)       # 3 pt, but never past a fence
        cur = (max(region[0], cur[0] - pad_x), max(region[1], cur[1] - pad_y),
               min(region[2], cur[2] + pad_x), min(region[3], cur[3] + pad_y))
        return (cur[0] / pw, cur[1] / ph, cur[2] / pw, cur[3] / ph)

    # ------------------------------------------------------------------ boxes

    def box_regions(self, index, exclude=()):
        """Rectangles (x0, y0, x1, y1 as page fractions, origin top-left) that look
        like info boxes: filled with a non-white colour or framed, reasonably large,
        and not overlapping figures/tables given in `exclude` (same coordinates)."""
        page = self.pdf[index]
        W, H = page.get_size()
        rects = []
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_PATH], max_depth=0):
            l, b, r, t = self._to_display(index, *obj.get_pos())
            w, h = r - l, t - b
            if w < 0.25 * W or h < 0.04 * H or w * h > 0.9 * W * H:
                continue
            if pdfium_c.FPDFPath_CountSegments(obj.raw) > 40:
                continue  # curves / chart lines, not a box
            fill_mode, stroke = ctypes.c_int(), ctypes.c_int()
            pdfium_c.FPDFPath_GetDrawMode(obj.raw, ctypes.byref(fill_mode), ctypes.byref(stroke))
            rgba = [ctypes.c_uint() for _ in range(4)]
            pdfium_c.FPDFPageObj_GetFillColor(obj.raw, *(ctypes.byref(v) for v in rgba))
            red, green, blue, alpha = (v.value for v in rgba)
            filled = fill_mode.value != 0 and alpha > 0 and (red + green + blue) < 735
            if not (filled or stroke.value):
                continue
            rect = self.to_frac(index, l, b, r, t)
            if any(_overlap_fraction(rect, ex) > 0.3 for ex in exclude):
                continue
            if any(_iou(rect, other) > 0.8 for other in rects):
                continue
            rects.append(rect)
        return rects

    # ---------------------------------------------------------------- figures

    def render_region(self, index, rect, dpi=220, max_px=1600, pad=0.004):
        """Render a page region (fractions of the displayed page, top-left origin) to a
        PIL image. pdfium applies the page's /Rotate, so crops are in display space too."""
        page = self.pdf[index]
        W, H = page.get_size()
        x0, y0, x1, y1 = (max(0.0, rect[0] - pad), max(0.0, rect[1] - pad),
                          min(1.0, rect[2] + pad), min(1.0, rect[3] + pad))
        width_pt = (x1 - x0) * W
        if width_pt <= 1 or (y1 - y0) * H <= 1:
            return None
        scale = min(dpi / 72.0, max_px / width_pt)
        crop = (x0 * W, (1 - y1) * H, (1 - x1) * W, y0 * H)
        return page.render(scale=scale, crop=crop).to_pil()


def encode_image(img):
    """PNG for line art (smaller and sharper), JPEG for photos; grayscale when there is
    no colour (a third of the size). Opaque on purpose: transparent black-on-clear art
    disappears on readers that keep images unchanged in dark mode. Returns (bytes, ext)."""
    from PIL import ImageChops
    img = img.convert("RGB")
    r, g, b = img.split()
    if max(ImageChops.difference(r, g).getextrema()[1], ImageChops.difference(g, b).getextrema()[1]) <= 8:
        img = img.convert("L")
    png, jpg = io.BytesIO(), io.BytesIO()
    img.save(png, format="PNG", optimize=True)
    img.save(jpg, format="JPEG", quality=85, optimize=True)
    if png.tell() <= 1.5 * jpg.tell():
        return png.getvalue(), "png"
    return jpg.getvalue(), "jpg"


def _area(r):
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def _intersection(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _overlap_fraction(a, b):
    """Fraction of `a` covered by `b`."""
    return _intersection(a, b) / _area(a) if _area(a) else 0.0


def _font_base(name):
    """"ABCDEF+Times-Bold" / "AdvTTec61d777.BI+26" -> the font itself."""
    return re.sub(r"\+\w{1,3}$", "", re.sub(r"^[A-Z]{6}\+", "", name))


def _norm(text):
    return re.sub(r"\s+", "", text).lower()


def _running_key(text):
    """Running heads compare without page numbers, spacing and punctuation."""
    return re.sub(r"\d+", "#", re.sub(r"[\W_]+", "", text.lower()))


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _iou(a, b):
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0
