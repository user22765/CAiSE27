"""Reviews and intermediate updates: decide on each candidate with original-source evidence.

The model answers {decisions, additions}: keep/replace/remove for every candidate
R1, R2, ... and new records. A replace/remove decision must cite a current source.
"""
from copy import deepcopy
import json

from jsonschema import validate

from .coefficients import text_diagnostics
from .common import schema
from .fusion import canonicalize, rule_fuse
from .validation import validate_relation


def check_relations(relations, context=None):
    """Strict checks for review edits/additions (extraction batches are only screened)."""
    for record in relations:
        validate_relation(record)
    if context is not None:
        canonicalize(relations, context, strict=True)  # Also checks model/group IDs against the catalog.


def candidate_records(candidates):
    return [dict(candidate_id=f'R{i + 1}', relation=deepcopy(r)) for i, r in enumerate(candidates)]


def decision_provenance(candidates, review):
    """Which fields each decision edited (with the sources it cited) and which it inherited."""
    prior = {f'R{i + 1}': r for i, r in enumerate(candidates)}
    records = []
    for decision in review['decisions']:
        before = prior[decision['candidate_id']]
        after = decision['replacement'] if decision['action'] == 'replace' else before
        fields = {field: dict(origin='current_edit' if after.get(field) != before[field] else 'inherited',
                              before=deepcopy(before[field]), after=deepcopy(after.get(field)))
                  for field in before}
        records.append(dict(candidate_id=decision['candidate_id'], action=decision['action'],
                            current_decision_sources=decision['source_ids'], fields=fields))
    return dict(candidates=records, additions=len(review['additions']))


def apply_revision(candidates, review, sources, context=None):
    validate(review, schema('revision'))
    ids = [f'R{i + 1}' for i in range(len(candidates))]
    decisions = review['decisions']
    if len(decisions) != len(ids) or {d['candidate_id'] for d in decisions} != set(ids):
        raise ValueError('Revision must decide every candidate exactly once')
    current_ids = {s.id for s in sources}
    by_id = {d['candidate_id']: d for d in decisions}
    output = []
    for rid, candidate in zip(ids, candidates):
        decision = by_id[rid]
        support = set(decision['source_ids'])
        if not support <= current_ids:
            raise ValueError('Decision cites an unavailable current source')
        if decision['action'] != 'keep' and not support:
            raise ValueError('Replace/remove requires current original-source support')
        if decision['action'] == 'replace':
            replacement = decision['replacement']
            if replacement is None:
                raise ValueError('Replace requires a complete record')
            check_relations([replacement], context)
            output.append(deepcopy(replacement))
        else:
            if decision['replacement'] is not None:
                raise ValueError('Keep/remove requires replacement=null')
            if decision['action'] == 'keep':
                output.append(deepcopy(candidate))
    check_relations(review['additions'], context)
    output.extend(deepcopy(review['additions']))
    return canonicalize(output, context) if context is not None else output


def combine_updates(prior, updates, context):
    """Combine independently validated decisions against the SAME prior snapshot.

    Silence never vetoes an evidenced edit. Contradictory second-modality edits
    remain alternatives; no iteration order decides which image is authoritative.
    """
    output, conflicts = [], []
    for index, candidate in enumerate(prior, 1):
        cid = f'R{index}'
        choices = [next(d for d in update['decisions'] if d['candidate_id'] == cid) for update in updates]
        replacements = [d['replacement'] for d in choices if d['action'] == 'replace']
        removals = [d for d in choices if d['action'] == 'remove']
        confirmations = [d for d in choices if d['action'] == 'keep' and d['source_ids']]
        if removals and (replacements or confirmations):
            output.extend([deepcopy(candidate), *deepcopy(replacements)])
            conflicts.append(dict(candidate_id=cid, reason='Second-modality sources disagree on removal', decisions=choices))
        elif removals:
            continue
        elif replacements:
            # A source-confirmed keep competes with a contradictory replacement.
            versions = replacements + ([candidate] if confirmations else [])
            combined = rule_fuse(versions, context)
            output.extend(combined)
            if len(combined) > 1:
                conflicts.append(dict(candidate_id=cid, reason='Unresolved second-modality corrections', decisions=choices))
        else:
            output.append(deepcopy(candidate))
    output.extend(deepcopy(r) for update in updates for r in update['additions'])
    return rule_fuse(output, context), conflicts


def revision_request(candidates, sources, context_text, mode, decimal_comma=False,
                     text_condition='with_html_tables', intermediate=False):
    """Return (recipe name, extra prompt text, request metadata) for one review batch."""
    suffix = 'with_tables' if text_condition == 'with_html_tables' else 'without_tables'
    if len(sources) == 1 and sources[0].kind == 'table':
        task = 'review_table'
    elif mode == 'images':
        task = 'review_image'
    elif mode == 'text':
        task = f'review_text_{suffix}'
    else:
        task = f'review_joint_{suffix}'
    extra = context_text + '\nCANDIDATES:\n' + json.dumps(candidate_records(candidates), ensure_ascii=False)
    if intermediate:
        extra += ('\nThis is review of an INTERMEDIATE UPDATE. The current sources are the second modality; '
                  'preserve inherited first-modality provenance.')
    metadata = dict(source_ids=[s.id for s in sources], mode=mode, recipe=task)
    if mode == 'text':
        checks = text_diagnostics(candidates, sources, decimal_comma)
        metadata['text_checks_before'] = checks
        extra += '\nARTICLE COEFFICIENT CHECKS (diagnostic aids only):\n' + json.dumps(checks, ensure_ascii=False)
    return task, extra, metadata
