"""Source selection and prompt assembly. Nothing here calls a model.

An article is {id, chunks}; each chunk is one source: a paragraph, table or image
with id, kind, page, caption, text (HTML body for tables), image path and
transcription (MinerU's text version of a figure).

Text condition: 'with_html_tables' shows table bodies inside text inputs,
'without_html_tables' shows only their captions. Table setting 'in_text' uses the
former; 'separate' uses the latter and extracts tables as their own modality.
"""
from copy import deepcopy
from pathlib import Path
import json
import re

from .common import LABEL_ALIASES, read_json, recipe
from .documents import Source
from .fusion import normalize_context
from .validation import validate_context

MODALITIES = ("text", "tables", "images")
SINGULAR = {"text": "text", "tables": "tables", "images": "image"}  # as in paradigm names
PARADIGM_MODALITY = {"text": "text", "table": "tables", "tables": "tables", "image": "images", "images": "images"}
SELECT_LABELS = {"tested_model", "hypothesized_model", "context"}
IMAGE_SELECT_LABELS = {"tested_model", "hypothesized_model"}


# --------------------------------------------------------------------------- settings

def condition_for(table_setting):
    """Text condition of a table setting: in_text -> with_html_tables, separate -> without_html_tables."""
    if table_setting not in ("in_text", "separate"):
        raise ValueError("table_setting must be 'in_text' or 'separate'")
    return "with_html_tables" if table_setting == "in_text" else "without_html_tables"


def condition_suffix(condition):
    """Recipe name suffix of a text condition."""
    return "with_tables" if condition == "with_html_tables" else "without_tables"


def default_modalities(table_setting):
    return ("text", "images") if table_setting == "in_text" else ("text", "tables", "images")


def check_modalities(modalities, table_setting):
    modalities = tuple(modalities)
    if (len(modalities) != len(set(modalities)) or not set(modalities) <= set(MODALITIES)
            or not 2 <= len(modalities) <= 3):
        raise ValueError(f"Fusion needs two or three distinct modalities from {MODALITIES}")
    if "tables" in modalities and table_setting != "separate":
        raise ValueError("A table modality requires table_setting='separate'; in_text puts tables into text")
    return modalities


def paradigm_order(paradigm):
    """'text_then_tables_then_image' -> ('text', 'tables', 'images')."""
    parts = paradigm.split("_then_")
    if len(parts) < 2 or any(p not in PARADIGM_MODALITY for p in parts):
        raise ValueError(f"Unknown paradigm {paradigm}")
    return tuple(PARADIGM_MODALITY[p] for p in parts)


# --------------------------------------------------------------------------- contexts and labels

def load_contexts(path, articles):
    """Saved contexts, checked against the articles' source IDs.

    Fields the pipeline no longer uses (e.g. source_scopes in older runs) are dropped.
    """
    saved = read_json(path)
    contexts = {}
    for article in articles:
        if article["id"] not in saved:
            raise ValueError("Context file is missing an article")
        contexts[article["id"]] = normalize_context(saved[article["id"]])
        validate_context(contexts[article["id"]], {c["id"] for c in article["chunks"]})
    return contexts


def context_text(context):
    return "ARTICLE CONTEXT (shared frozen context.json):\n" + json.dumps(normalize_context(context), ensure_ascii=False)


def normalized_labels(value):
    """Classifier answer or saved label(s) -> list of pipeline label names."""
    if isinstance(value, dict):
        value = value.get("label", value.get("labels", []))
    if isinstance(value, str):
        value = [value]
    return [LABEL_ALIASES.get(label, label) for label in value or []]


# --------------------------------------------------------------------------- selecting sources

def is_figure(chunk):
    """A non-table image (figure, chart, equation or unlinked image) with a file."""
    return chunk["kind"] == "image" and bool(chunk.get("image"))


def usable(chunk, mode):
    if mode == "images":
        return is_figure(chunk)
    return bool(chunk.get("text") or chunk.get("caption") or (mode == "joint" and chunk.get("image")))


def select(chunks, mode, labels):
    """Usable chunks whose labels are relevant; an unlabeled chunk counts as relevant; labels=None keeps all."""
    accepted = IMAGE_SELECT_LABELS if mode == "images" else SELECT_LABELS
    return [c for c in chunks if usable(c, mode) and (labels is None or
            bool(set(normalized_labels(labels.get(c["id"], ["tested_model"]))) & accepted))]


def text_chunks(article, text_condition="with_html_tables"):
    """The article as text: figures (and, without HTML tables, tables) keep only their captions."""
    output = []
    for original in article.get("chunks", []):
        chunk = deepcopy(original)
        chunk["image"] = None
        if chunk["kind"] == "image" or (chunk["kind"] == "table" and text_condition == "without_html_tables"):
            chunk["kind"] = "caption"
            chunk["text"] = ""
        if chunk.get("text") or chunk.get("caption") or chunk.get("transcription"):
            output.append(chunk)
    return output


def project_tables(chunks, representation):
    """Show each table one way: its HTML body ('html') or its original image ('image').

    A table without the requested form falls back to the other one; one with
    neither is shown by its caption only. shown_as records what was used
    ('html', 'image' or None).
    """
    if representation not in ("html", "image"):
        raise ValueError("table_representation must be html or image")
    output = []
    for chunk in chunks:
        item = deepcopy(chunk)
        available = [form for form, present in (("html", item.get("text")), ("image", item.get("image"))) if present]
        item["shown_as"] = representation if representation in available else next(iter(available), None)
        if item["shown_as"] != "image":
            item["image"] = None
        if item["shown_as"] != "html":
            item["text"] = ""
        output.append(item)
    return output


def joint_batches(article, chunks, text_condition, include_figures=True, max_images=8):
    """One request per paper: all selected text plus all selected figures, in reading order.

    Only a paper with more figures than max_images is split: every request then
    repeats the full text with the next group of figures (the other figures'
    captions stay as text). Tables enter joint inputs as HTML, never as images.
    """
    selected_ids = {c["id"] for c in chunks}
    base = [c for c in text_chunks(article, text_condition) if c["id"] in selected_ids]
    visuals = [c for c in chunks if include_figures and is_figure(c)]
    if not visuals:
        return [base] if base else []
    output = []
    for start in range(0, len(visuals), max_images):
        group = {c["id"]: c for c in visuals[start:start + max_images]}
        batch = [deepcopy(c) for c in base if c["id"] not in group] + [deepcopy(c) for c in group.values()]
        output.append(sorted(batch, key=lambda c: c.get("index", 0)))
    return output


def batches(article, mode, labels, text_input, text_condition, table_representation="html",
            include_figures=True, baseline=False, image_labels=None, max_images=8):
    """The request batches of one article; each batch is a list of chunks.

    tables/images: one batch per selected table/figure. text: one batch with all
    selected text, or one per source for text_input='elementwise'. joint
    (early fusion, baseline): see joint_batches.
    """
    if mode == "tables":
        tables = select([c for c in article["chunks"] if c["kind"] == "table"], "joint", labels)
        return [[c] for c in project_tables(tables, table_representation)]
    if mode == "images":
        return [[c] for c in select(article["chunks"], "images", image_labels)]
    chunks = text_chunks(article, text_condition) if mode == "text" else article["chunks"]
    chunks = select(chunks, mode, labels)
    if mode == "joint":
        # Images follow their own labels; the baseline keeps every image.
        figures = ([c for c in article["chunks"] if is_figure(c)] if baseline
                   else select(article["chunks"], "images", image_labels))
        chunks = sorted([c for c in chunks if not is_figure(c)] + figures, key=lambda c: c["index"])
    if text_input == "elementwise":
        return [[c] for c in chunks]
    if mode == "text":
        return [chunks] if chunks else []
    return joint_batches(article, chunks, text_condition, include_figures, max_images)


def extraction_task(mode, batch, text_condition, baseline=False):
    """Recipe name of an extraction request."""
    if baseline:
        return "baseline"
    if mode == "tables" or (mode == "text" and len(batch) == 1 and batch[0]["kind"] == "table"):
        return "table"
    if mode == "text":
        return "text_" + condition_suffix(text_condition)
    if mode == "images":
        return "image"
    return "early_" + condition_suffix(text_condition)


# --------------------------------------------------------------------------- prompts

def without_transcripts(text):
    """Remove figure transcriptions (<details> blocks and mermaid diagrams) from a caption or text."""
    text = re.sub(r"<details\b[^>]*>.*?</details>", "", text or "", flags=re.S | re.I)
    return re.sub(r"```\s*mermaid\b.*?```", "", text, flags=re.S | re.I)


def render(chunk, mode, allow_transcriptions=False, text_condition="with_html_tables"):
    """(text, image path or None) of one source as it appears in a prompt."""
    lines = [f"[{chunk['id']}; page {chunk.get('page', 0)}; {chunk['kind']}]"]
    if chunk.get("caption"):
        lines.append(without_transcripts(chunk["caption"]))
    # The table modality always shows its body (HTML or image); in text/joint
    # inputs, HTML table bodies follow the text condition.
    if mode == "tables" or (mode != "images" and (chunk["kind"] != "table" or text_condition == "with_html_tables")):
        lines.append(without_transcripts(chunk.get("text", "")))
    if allow_transcriptions and chunk.get("transcription"):
        lines.append("FIGURE TRANSCRIPTION (same source; converted evidence):\n" + chunk["transcription"])
    image = chunk.get("image") if mode != "text" else None
    return "\n".join(v for v in lines if v), image


def build_prompt(instructions, json_schema, extra, chunks, mode, allow_transcriptions=False,
                 text_condition="with_html_tables", max_images=8):
    """(prompt text, image paths): instructions, schema, extra text, then the numbered sources."""
    parts = [instructions]
    if json_schema is not None:
        parts.append("JSON SCHEMA:\n" + json.dumps(json_schema, ensure_ascii=False))
    if extra:
        parts.append(extra)
    rendered = [(chunk, *render(chunk, mode, allow_transcriptions, text_condition)) for chunk in chunks]
    total = sum(bool(image) for _, _, image in rendered)
    source_texts, images = [], []
    for chunk, text, image in rendered:
        if image:
            images.append(str(image))
            source_texts.append(f"Attached original image {len(images)} of {total} corresponds to {chunk['id']}.")
        source_texts.append(text)
    if len(images) > max_images:
        raise ValueError(f"At most {max_images} original images per prompt")
    if source_texts:
        parts.append("SOURCES:\n" + "\n\n".join(source_texts))
    return "\n\n".join(parts), images


def task_request(task, context, chunks, mode, extra="", text_condition=None, max_images=8):
    """((prompt text, image paths), output schema) for a manifest recipe."""
    spec = recipe(task)
    if spec["requires_context"]:
        if context is None:
            raise ValueError(f"{task} requires the shared context")
        extra = context_text(context) + "\n" + extra
    extra += "\nDeclared text condition: " + (text_condition or spec.get("text_condition", "with_html_tables"))
    request = build_prompt(spec["instructions"], spec["output_schema"], extra, chunks, mode,
                           spec["allows_figure_transcriptions"], text_condition or "with_html_tables", max_images)
    limit = max_images if spec["max_original_images"] is None else spec["max_original_images"]
    if len(request[1]) > limit:
        raise ValueError("Recipe image budget exceeded")
    return request, spec["output_schema"]


def sources(batch, mode, text_condition):
    """The batch's sources for review checks, each with its rendered text."""
    output = []
    for chunk in batch:
        text, image = render(chunk, mode, False, text_condition)
        output.append(Source(chunk["id"], chunk["kind"], chunk.get("page", 0), text, Path(image) if image else None))
    return output


def source_chunks(sources):
    """Sources back as chunks for a review prompt. The prompt renders the already rendered
    text once more (so each source header appears twice); kept for comparability of runs."""
    return [dict(id=s.id, kind=s.kind, page=s.page, caption="", text=s.text,
                 image=str(s.image) if s.image else None, transcription="") for s in sources]
