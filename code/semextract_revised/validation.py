"""Schema and cross-field checks for relations and contexts; nothing is silently repaired."""
import math
from copy import deepcopy

from jsonschema import Draft202012Validator, ValidationError

from .common import label_key, schema
from .utils import normalize_p


def semantic_record(record):
    """The relation fields of a record, without consolidation metadata (candidate_ids, status, ...)."""
    return {key: deepcopy(record[key]) for key in schema("relation")["items"]["properties"]}


def validate_relation(record, strict=True):
    """Raise if a relation is invalid.

    strict=False (pipeline default) checks only the schema, effect_type vs.
    mediator/moderator roles, a finite beta and a possible p; strict=True also
    rejects blank names and IDs.
    """
    contract = schema("relation")["items"]
    # Consolidation metadata (candidate_ids, status, ...) is checked by its own response schema.
    Draft202012Validator(contract).validate({k: v for k, v in record.items() if k in contract["properties"]})
    mediator, moderator, kind = record["mediator"], record["moderator"], record["effect_type"]
    if ((kind == "direct" and (mediator is not None or moderator is not None))
            or (kind == "moderation" and (moderator is None or mediator is not None))
            or (kind == "indirect" and mediator is None)):
        raise ValueError("effect_type and mediator/moderator roles disagree")
    beta = record["beta"]
    if beta is not None and (isinstance(beta, bool) or not math.isfinite(beta)):
        raise ValueError("beta must be a finite number")
    if record["p"] is not None:
        normalize_p(record["p"])  # rejects impossible bounds such as "<0"
    if not strict:
        return record
    for key in ("cause", "effect", "mediator", "moderator", "model_id", "group_id", "hypothesis_id"):
        if isinstance(record.get(key), str) and not record[key].strip():
            raise ValueError(f"Blank {key}")
    return record


def clean_relations(relations):
    """A validated deep copy; no rounding, coercion, or loss of metadata."""
    if not isinstance(relations, list):
        raise ValueError("Relations must be an array")
    output = deepcopy(relations)
    for record in output:
        validate_relation(record, strict=False)
        for alternative in record.get("alternatives", []):
            validate_relation(alternative, strict=False)
    return output


def screen_relations(relations):
    """Check extracted records one by one: return (kept, dropped).

    An impossible p is set to null and the record kept; a record that is
    otherwise invalid is dropped. One bad record never discards its batch.
    """
    kept, dropped = [], []
    for original in relations:
        record = deepcopy(original)
        if record.get("p") is not None:
            try:
                normalize_p(record["p"])
            except ValueError as error:
                dropped.append(dict(relation=deepcopy(record), reason=f"p set to null: {error}", kept=True))
                record["p"] = None
        try:
            validate_relation(record, strict=False)
        except (ValueError, ValidationError) as error:
            dropped.append(dict(relation=record, reason=str(error), kept=False))
            continue
        kept.append(record)
    return kept, dropped


def validate_context(context, allowed_ids=None):
    """Raise if a context is invalid, ambiguous, or cites sources outside allowed_ids."""
    Draft202012Validator(schema("context")).validate(context)
    for catalog, field in (("constructs", "name"), ("models", "id"), ("groups", "id")):
        owners = {}
        for item in context[catalog]:
            for label in [item[field], *item["aliases"]]:
                key = label_key(label)
                if key in owners and owners[key] != item[field]:
                    raise ValueError(f"Ambiguous {catalog} label {label!r}")
                owners[key] = item[field]
    models = {m["id"] for m in context["models"]}
    groups = {g["id"] for g in context["groups"]}
    refs = []
    for entry in context["significance_criteria"]:
        for field, known in (("model_id", models), ("group_id", groups)):
            if entry[field] is not None and entry[field] not in known:
                raise ValueError(f"Unknown {field} in significance_criteria")
        refs.append(entry["source_id"])
        refs.extend(s["source_id"] for s in entry["applies_to"])
        if (entry["scope_kind"] == "article") != (not entry["applies_to"]):
            raise ValueError("Significance criterion scope and applies_to disagree")
    if allowed_ids is not None and not set(refs) <= set(allowed_ids):
        raise ValueError("Context cites an unavailable source")
    return context
