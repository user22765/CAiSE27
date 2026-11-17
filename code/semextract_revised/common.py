"""JSON files, fingerprints, prompt recipes, and name normalization."""
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "prompts_revise"
FORMAT_VERSION = 3

# classify_schema.txt emits *_relations labels; pipeline code selects on *_model.
LABEL_ALIASES = {"tested_relations": "tested_model", "estimated_sem": "tested_model",
                 "path_results": "tested_model", "indirect_results": "tested_model",
                 "moderation_or_groups": "tested_model", "hypothesized_relations": "hypothesized_model"}


def strict_json(text):
    """json.loads that rejects duplicate keys and NaN/Infinity."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return number

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=unique_keys, parse_float=finite_float,
                      parse_constant=reject_constant)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """Write via a temporary file, so an interrupted save never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt(name):
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def schema(name):
    return read_json(PROMPT_DIR / f"{name}_schema.txt")


def recipe(name):
    """A task from prompts_revise/manifest.json, with its assembled instructions and output schema."""
    spec = read_json(PROMPT_DIR / "manifest.json")["recipes"][name]
    result = dict(spec, name=name)
    result["instructions"] = "\n\n".join((PROMPT_DIR / part).read_text(encoding="utf-8")
                                         for part in spec["components"])
    result["output_schema"] = read_json(PROMPT_DIR / spec["schema"]) if spec["schema"] else None
    return result


def prompt_fingerprint():
    return digest({p.name: file_digest(p) for p in sorted(PROMPT_DIR.iterdir())
                   if p.suffix in {".txt", ".json"}})


def classification_labels():
    """The classifier vocabulary from classify_schema.txt, in pipeline label names."""
    return {LABEL_ALIASES.get(label, label) for label in schema("classify")["properties"]["label"]["enum"]}


def label_key(value):
    """Comparison key for construct names, aliases and model/group IDs."""
    value = unicodedata.normalize("NFKC", value or "").casefold()
    return re.sub(r"[\s\-‐‑–—]+", " ", value).strip()


def normalize_name(value):
    """Comparison key used by the evaluation."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", " ", value).strip()


def parse_json_response(text):
    """Parse a model answer, removing thinking blocks, end tokens and ``` fences around the JSON."""
    text = text.strip()
    if text.startswith("<think>") and "</think>" not in text:
        raise ValueError("Generation ended inside a thinking block")
    if not text.startswith(("[", "{", "```")) and "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if text.startswith("<|channel>thought"):
        if "<channel|>" not in text:
            raise ValueError("Generation ended inside a thought channel")
        text = text.split("<channel|>", 1)[1]
    text = re.sub(r"(?:<\|(?:im_end|end|eot_id|turn_end|end_of_text)\|>|</s>|<｜end▁of▁sentence｜>)\s*$",
                  "", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return strict_json(text)
