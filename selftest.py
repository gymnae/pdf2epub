"""Container self-test (the distroless image has no shell, so checks run in Python).

    docker run --rm --entrypoint /venv/bin/python <image> selftest.py

Checks that every external tool and native library the app relies on is present
and works; needs no GPU and no network.
"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile


def check(name, fn):
    try:
        detail = fn()
        print(f"ok    {name}{': ' + detail if detail else ''}")
        return True
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        return False


def tool(name, *args):
    def run():
        exe = shutil.which(name)
        if not exe:
            raise FileNotFoundError(f"{name} not on PATH")
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60, check=True)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    return run


def on_path(name):
    def run():
        exe = shutil.which(name)
        if not exe:
            raise FileNotFoundError(f"{name} not on PATH")
        return exe
    return run


def imports(*modules):
    def run():
        loaded = [importlib.import_module(m) for m in modules]
        versions = [f"{m} {getattr(mod, '__version__', '')}".strip() for m, mod in zip(modules, loaded)
                    if m in ("torch", "cv2", "gradio")]
        if "torch" in modules:
            versions.append(f"CUDA {loaded[modules.index('torch')].version.cuda}")
        return ", ".join(versions)
    return run


def math_svg():
    import mathrender
    svgs = mathrender.render_svgs([(r"\frac{a}{b}+\ce{H2O}", False)], tempfile.mkdtemp())
    svg = next(iter(svgs.values()))
    if not svg or "<path" not in svg:
        raise RuntimeError("no SVG produced")
    return f"{len(svg)} bytes"


def epub_roundtrip():
    import epubbuild
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "t.md"), "w") as f:
        f.write("# T\n\nText $x^2$ and a note.[^1]\n\n[^1]: Note.\n")
    with open(os.path.join(d, "m.json"), "w") as f:
        json.dump({"title": "T"}, f)
    epubbuild.build_epub(os.path.join(d, "t.md"), os.path.join(d, "t.epub"), [d], os.path.join(d, "m.json"))
    if not epubbuild.kepubify(os.path.join(d, "t.epub"), os.path.join(d, "t.kepub.epub")):
        raise RuntimeError("kepubify failed")
    names = zipfile.ZipFile(os.path.join(d, "t.epub")).namelist()
    return f"{len(names)} files in EPUB"


def marker_font():
    """marker writes this font on every start if missing; the venv is read-only."""
    from marker.settings import settings
    path = settings.FONT_PATH
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    if os.access(os.path.dirname(path) if os.path.isdir(os.path.dirname(path)) else "/", os.W_OK):
        return f"missing, but {os.path.dirname(path)} is writable (downloaded on first run)"
    raise FileNotFoundError(f"{path} missing and not writable – marker would fail at startup")


def writable():
    dirs = [os.environ.get("LIBRARY_DIR", "/app/library"), os.environ.get("HOME", ""), tempfile.gettempdir()]
    bad = [d for d in dirs if d and not os.access(d, os.W_OK)]
    if bad:
        raise PermissionError("not writable: " + ", ".join(bad))
    return ", ".join(d for d in dirs if d)


if __name__ == "__main__":
    results = [
        check("pandoc", tool("pandoc", "--version")),
        check("kepubify", tool("kepubify", "--version")),
        check("marker_single on PATH", on_path("marker_single")),
        check("native libraries", imports("torch", "cv2", "PIL", "pypdfium2", "transformers")),
        check("marker / surya", imports("marker.converters.pdf", "surya")),
        check("app modules", imports("epubbuild", "gpu", "mathrender", "pdfanalysis", "structure", "gradio", "pynvml")),
        check("math → SVG", math_svg),
        check("markdown → EPUB → kepub", epub_roundtrip),
        check("marker font baked in", marker_font),
        check("runtime dirs writable", writable),
    ]
    sys.exit(0 if all(results) else 1)
