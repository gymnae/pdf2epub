"""Direct PDF analysis with pypdfium2 (no GPU, no torch).

Fills gaps in marker's output:
  * font styling: recovers italic/bold runs from glyph geometry, which works
    even for subset fonts with obfuscated names and empty style flags;
  * info boxes: filled or framed rectangles that contain text;
  * figures: re-renders figure regions as crisp images straight from the PDF;
  * document checks: publisher cover sheets and text-layer quality.
"""
import ctypes
import io
import re
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
)
ITALIC_NAME = re.compile(r"(italic|oblique|[-.,_]it$|[-.,_]i$|ital\b|-ita\b|cmti|cmmi)", re.I)
BOLD_NAME = re.compile(r"(bold|black|heavy|semibd|[-.,_]b$|[-.,_]bd$|cmbx)", re.I)

ITALIC_SLANT_DELTA = 0.09   # italic glyphs lean ~0.15-0.25; upright ones ~0 +- 0.08
BOLD_INK_RATIO = 1.3        # bold strokes cover >= 30% more area than the body font


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
            tp = self.pdf[index].get_textpage()
            self._page_text[index] = tp.get_text_bounded()
        return self._page_text[index]

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
        """(font_key, text) for each top-level text object of a page."""
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
                out.append((key, text))
        return out

    def _body_metrics(self):
        if self._body_style is None:
            counts = Counter()
            n = len(self.pdf)
            for i in range(0, n, max(1, n // 6)):
                for key, text in self._text_objects(i):
                    counts[key] += len(text)
            body = None
            for key, _ in counts.most_common():
                slant, ink = self._font_cache.get(key, (None, None))
                if slant is not None and ink is not None:
                    body = (slant, ink)
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
        for key, text in self._text_objects(index):
            slant, ink = self._font_cache.get(key, (None, None))
            phrase = re.sub(r"\s+", " ", text).strip()
            if len(phrase) < 2 or not re.search(r"\w\w", phrase):
                continue
            base = key.split("+")[-1]
            if ITALIC_NAME.search(base) or (slant is not None and slant - (body_slant or 0) > ITALIC_SLANT_DELTA):
                found["italic"][phrase] += 1
            elif BOLD_NAME.search(base) or (ink is not None and body_ink and ink / body_ink > BOLD_INK_RATIO):
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

    # ------------------------------------------------------------------ boxes

    def box_regions(self, index, exclude=()):
        """Rectangles (x0, y0, x1, y1 as page fractions, origin top-left) that look
        like info boxes: filled with a non-white colour or framed, reasonably large,
        and not overlapping figures/tables given in `exclude` (same coordinates)."""
        page = self.pdf[index]
        W, H = page.get_size()
        rects = []
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_PATH], max_depth=0):
            l, b, r, t = obj.get_pos()
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
            rect = (l / W, (H - t) / H, r / W, (H - b) / H)
            if any(_overlap_fraction(rect, ex) > 0.3 for ex in exclude):
                continue
            if any(_iou(rect, other) > 0.8 for other in rects):
                continue
            rects.append(rect)
        return rects

    # ---------------------------------------------------------------- figures

    def render_region(self, index, rect, dpi=220, max_px=1600, pad=0.004):
        """Render a page region (fractions, top-left origin) to a PIL image."""
        page = self.pdf[index]
        if page.get_rotation():
            return None
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
    """PNG for line art (smaller and sharper), JPEG for photos. Returns (bytes, ext)."""
    img = img.convert("RGB")
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


def _iou(a, b):
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0
