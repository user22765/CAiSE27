"""Load a MinerU article with every figure and table image, plus an image manifest for labeling."""
import hashlib
import json
import os
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

from .documents import SKIP, block_text, image_paths, joined

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".avif", ".jp2", ".ppm", ".pgm", ".pbm", ".pnm", ".svg",
}
SUPPLEMENTAL_INDEX_BASE = 1 << 52


def supplemental_index(relative):
    """A stable ID for an image without its own content-list block.

    Hashing the path (instead of appending a counter) keeps IDs stable when files
    are added; the value stays below 2**53, so JavaScript JSON readers keep it exact.
    """
    return SUPPLEMENTAL_INDEX_BASE + int(hashlib.sha256(relative.encode()).hexdigest()[:13], 16)


def model_image(path, directory):
    """Path of an image the model can read; SVGs are rasterized once into a cache folder."""
    if path.suffix.lower() != ".svg":
        return str(path)
    try:
        import cairosvg
    except (ImportError, OSError) as error:
        raise RuntimeError("SVG inputs require CairoSVG and its Cairo library. "
                           "In Colab, install cairosvg before loading articles.") from error
    # Include the location because linked SVG resources can depend on its directory.
    key = hashlib.sha256(str(path).encode() + b"\0" + path.read_bytes()).hexdigest()
    cache = directory / ".semextract_rasterized"
    cache.mkdir(exist_ok=True)
    rendered = cache / f"{key}.png"
    if not rendered.is_file():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".png", delete=False) as handle:
                temporary = Path(handle.name)
            cairosvg.svg2png(url=str(path), write_to=str(temporary), unsafe=False)
            os.replace(temporary, rendered)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return str(rendered)


def convert(directory):
    """Return (article_id, document) with paragraphs, tables, images, image_manifest and diagnostics.

    A chart with no existing text chunk keeps its original content-list index.
    Equation/chart text chunks retain their IDs; their additional visual inputs
    receive supplemental IDs. Images absent from the content list are included
    directly from disk, without guessing a parent table or equation.
    """
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(directory.glob("*_content_list.json"))
    if len(files) > 1:
        raise ValueError(f"Expected at most one *_content_list.json in {directory}; found {len(files)}.")
    diagnostics = []
    if files:
        article_id = files[0].name.removesuffix("_content_list.json")
        entries = json.loads(files[0].read_text(encoding="utf-8"))
        if not isinstance(entries, list) or any(not isinstance(e, dict) or "type" not in e for e in entries):
            raise ValueError(f"Expected a flat list of typed content blocks in {files[0]}")
    else:
        article_id = directory.parent.name if directory.name == "vlm" else directory.name
        entries = []
        diagnostics.append("No content list found; loading image files only.")

    # Every image file on disk under images/.
    inventory, ignored = {}, []
    for path in sorted((directory / "images").rglob("*")):
        if not path.is_file():
            continue
        absolute = path.resolve()
        if not absolute.is_relative_to(directory):
            raise ValueError(f"Image path escapes article directory: {path}")
        if path.suffix.lower() in IMAGE_SUFFIXES:
            inventory.setdefault(absolute, path.relative_to(directory).as_posix())
        else:
            ignored.append(path.relative_to(directory).as_posix())

    # Which content-list blocks link to which image file.
    references, by_index, prepared = defaultdict(list), {}, {}
    for index, entry in enumerate(entries):
        relative, absolute = image_paths(directory, entry)
        if absolute:
            path = Path(absolute)
            references[path].append((index, entry))
            by_index[index] = path
            if not path.is_file():
                diagnostics.append(f"b{index}: referenced image does not exist: {relative}")

    def visual_item(index, entry, path, content_indices):
        kind = entry.get("type", "unlinked_image")
        page_idx = entry.get("page_idx")
        item = dict(index=index, page=int(page_idx) + 1 if page_idx is not None else 0,
                    image_path=None, image_path_abs=None, source_kind=kind,
                    content_indices=content_indices,
                    transcription=joined(entry.get("content")) if kind != "table" else "",
                    caption=joined(entry.get(f"{kind}_caption") or entry.get("image_caption")
                                   or entry.get("caption")),
                    footnote=joined(entry.get(f"{kind}_footnote") or entry.get("image_footnote")
                                    or entry.get("footnote")))
        if path is not None:
            item["image_path"] = inventory.get(path, path.relative_to(directory).as_posix())
            if path.is_file():
                if path not in prepared:
                    prepared[path] = model_image(path, directory)
                item["image_path_abs"] = prepared[path]
        return item

    result = {"full_text": [], "paragraphs": [], "images": [], "tables": []}
    for index, entry in enumerate(entries):
        kind = entry["type"]
        if kind in SKIP:
            continue
        if kind in {"image", "table"}:
            item = visual_item(index, entry, by_index.get(index), [index])
            if kind == "image":
                result["images"].append(item)
                continue  # images are excluded from full_text
            item["html"] = entry.get("table_body", "")
            result["tables"].append(item)
        text = block_text(entry)
        if text:
            result["full_text"].append(text)
            if kind != "table":
                result["paragraphs"].append({"index": index, "page": int(entry.get("page_idx", 0) or 0) + 1,
                                             "text": text})

    # Add every remaining image: files under images/ and linked chart/equation images elsewhere.
    used = {item["index"] for group in ("paragraphs", "images", "tables") for item in result[group]}
    represented = {by_index[item["index"]] for group in ("images", "tables")
                   for item in result[group] if item["image_path_abs"]}
    candidates = dict(inventory)
    for path in references:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.setdefault(path, path.relative_to(directory).as_posix())
    for path, relative in sorted(candidates.items(), key=lambda pair: pair[1]):
        if path in represented:
            continue
        refs = references.get(path, [])
        reusable = [(index, entry) for index, entry in refs if index not in used and entry["type"] not in SKIP]
        if reusable:
            index, entry = reusable[0]
        else:
            index = supplemental_index(relative)
            entry = refs[0][1] if refs else {}
        if index in used:
            raise ValueError(f"Supplemental ID collision for {relative}: b{index}")
        used.add(index)
        result["images"].append(visual_item(index, entry, path, [i for i, _ in refs]))
        represented.add(path)

    result["full_text"] = "\n".join(result["full_text"])
    result["images"].sort(key=lambda item: item["index"])
    result["image_manifest"] = sorted([
        dict(article_id=article_id, block_id=f"b{item['index']}",
             input_kind="table" if group == "tables" else "image",
             source_kind=item["source_kind"], page=item["page"],
             image_path=item["image_path"], image_path_abs=item["image_path_abs"],
             content_indices=item["content_indices"])
        for group in ("images", "tables") for item in result[group] if item["image_path_abs"]
    ], key=lambda item: int(item["block_id"][1:]))
    result["ignored_files"] = ignored
    result["diagnostics"] = diagnostics
    for message in diagnostics:
        warnings.warn(f"{article_id}: {message}", stacklevel=2)
    if not set(inventory) <= represented:
        raise AssertionError("An inventoried image was not included in the document.")
    return article_id, result


def load_article(directory):
    """Return {id, full_text, chunks, image_manifest, ...}; chunks are the pipeline's sources in reading order."""
    article_id, doc = convert(directory)
    chunks = []
    for paragraph in doc["paragraphs"]:
        chunks.append(dict(id=f"b{paragraph['index']}", index=paragraph["index"], kind="paragraph",
                           page=paragraph["page"], caption="", text=paragraph["text"], image=None))
    for group, kind in (("tables", "table"), ("images", "image")):
        for item in doc[group]:
            chunks.append(dict(id=f"b{item['index']}", index=item["index"], kind=kind,
                               page=item["page"], caption="\n".join(
                                   part for part in (item["caption"], item["footnote"]) if part),
                               text=item.get("html", ""), image=item["image_path_abs"],
                               transcription=item.get("transcription", ""), source_kind=item.get("source_kind", kind)))
    chunks.sort(key=lambda item: item["index"])
    return dict(id=article_id, full_text=doc["full_text"], chunks=chunks,
                image_manifest=doc["image_manifest"], ignored_files=doc["ignored_files"],
                diagnostics=doc["diagnostics"])
