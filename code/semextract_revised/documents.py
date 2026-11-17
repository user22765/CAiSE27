"""MinerU content-list blocks and the Source record used by reviews."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    """One original source as shown to a review: rendered text and/or its image."""
    id: str
    kind: str
    page: int
    text: str
    image: Path | None = None


# Page furniture and other non-body blocks.
SKIP = {"header", "footer", "page_number", "aside_text", "page_footnote", "ref_text", "discarded"}


def joined(value):
    return "\n".join(map(str, value)) if isinstance(value, list) else str(value or "")


def block_text(entry):
    kind = entry["type"]
    if kind == "table":
        return entry.get("table_body", "")
    if kind == "list":
        return joined(entry.get("list_items"))
    if kind == "code":
        return entry.get("code_body", "")
    return entry.get("text", "")  # text, headings, equations (LaTeX)


def image_paths(directory, entry):
    """(relative, absolute) image path of a block, or (None, None); refuses paths outside the article."""
    directory = Path(directory).resolve()
    relative = entry.get("img_path")
    if not relative:
        return None, None
    if not isinstance(relative, str):
        raise ValueError(f"Expected a string img_path, got {relative!r}")
    absolute = (directory / relative).resolve()
    if not absolute.is_relative_to(directory):
        raise ValueError(f"Image path escapes article directory: {relative}")
    return relative, str(absolute)
