"""Late fusion: candidates from independent modality predictions, and checks of the fused result."""
import json
from copy import deepcopy

from jsonschema import validate

from .common import schema
from .fusion import consolidate_candidates, normalize_context, rule_fuse
from .validation import clean_relations, screen_relations, semantic_record, validate_relation


def fusion_input(context, relations):
    """{context, candidates} for relations = {modality: [records]}; candidate IDs are text_1, images_2, ..."""
    candidates = []
    for modality in ("text", "tables", "images"):
        if relations.get(modality) is None:
            continue
        for index, record in enumerate(clean_relations(relations[modality]), 1):
            candidates.append(dict(candidate_id=f"{modality}_{index}", modality=modality, **semantic_record(record)))
    return dict(context=normalize_context(context), candidates=candidates)


def validate_fusion(result, payload):
    """Raise unless result groups the candidates exactly as the shared rules allow, without changing values."""
    validate(result, schema("late_fusion"))
    expected = consolidate_candidates(payload["candidates"], payload["context"])
    by_members = {frozenset(r["candidate_ids"]): r for r in expected}
    used = []
    for record in result:
        validate_relation(record, strict=False)
        members = frozenset(record["candidate_ids"])
        if members not in by_members:
            raise ValueError("Fusion grouping is not supported by the shared mapping/compatibility rules")
        reference, actual = deepcopy(by_members[members]), deepcopy(record)
        for field in ("candidate_ids", "supported_by"):
            reference[field] = sorted(reference[field])
            actual[field] = sorted(actual[field])
        reference["alternatives"] = sorted(reference["alternatives"], key=lambda v: json.dumps(v, sort_keys=True))
        actual["alternatives"] = sorted(actual["alternatives"], key=lambda v: json.dumps(v, sort_keys=True))
        if reference != actual:
            raise ValueError("Fusion altered candidate values or the conflict policy")
        used.extend(record["candidate_ids"])
    given = [c["candidate_id"] for c in payload["candidates"]]
    # Candidates without beta/p on a path estimated elsewhere are dropped, not covered.
    required = [cid for r in expected for cid in r["candidate_ids"]]
    if sorted(used) != sorted(required) or len(set(given)) != len(given):
        raise ValueError("Every retained candidate must be covered exactly once")
    return deepcopy(result)


def reviewable_fusion(records, originals, context):
    """Plain relations for the original-source review of a fused result.

    Fusion fields (candidate_ids, status, ...) are dropped; an unresolved
    contradiction is expanded into its original candidates.
    """
    output = []
    for record in records:
        members = [originals[cid] for cid in record.get("candidate_ids", []) if cid in originals]
        if record.get("status") == "contradiction" and members:
            output.extend(deepcopy(members))
            continue
        output.append(semantic_record(record))
    kept, _ = screen_relations(output)
    return rule_fuse(kept, context)
