"""Relation identity and conservative merging, shared by every stage.

Two records describe the same relation when cause, effect, roles, effect type,
model_id and group_id match after mapping names and IDs to the shared context
(null matches null). Merging never invents values: conflicting estimates stay apart.
"""
from copy import deepcopy
from itertools import combinations

from .common import digest, label_key
from .utils import merge_p, p_interval
from .validation import clean_relations, semantic_record, validate_context

CONTEXT_KEYS = ('constructs', 'models', 'groups', 'significance_criteria')
VARIABLE_FIELDS = ('cause', 'effect', 'mediator', 'moderator')
STAT_FIELDS = ('beta', 'p', 'significance', 'hypothesis_id')


def unique(values):
    return list({digest(value): deepcopy(value) for value in values}.values())


def normalize_context(context):
    """Copy the complete context; never rewrite verbatim definitions/items."""
    if not isinstance(context, dict):
        raise ValueError('Context must be an object')
    output = {key: deepcopy(context.get(key, [])) for key in CONTEXT_KEYS}
    for key in ('models', 'groups'):
        for item in output[key]:
            item.setdefault('aliases', [])
    validate_context(output)
    return output


def merge_context(parts):
    """Consolidate attested aliases once, before the shared context is frozen."""
    result = {key: [] for key in CONTEXT_KEYS}
    for part in parts:
        part = normalize_context(part)
        remaps = {}
        for catalog, field in (('constructs', 'name'), ('models', 'id'), ('groups', 'id')):
            remaps[catalog] = {}
            for item in part[catalog]:
                names = {label_key(n) for n in [item[field], *item['aliases']]}
                matches = [entry for entry in result[catalog]
                           if names & {label_key(n) for n in [entry[field], *entry['aliases']]}]
                if len(matches) > 1:
                    raise ValueError(f'Ambiguous {catalog} context merge; reconcile article context first')
                if not matches:
                    result[catalog].append(deepcopy(item))
                    continue
                entry = matches[0]
                all_names = unique([entry[field], *entry['aliases'], item[field], *item['aliases']])
                old = entry[field]
                # Respect an already selected canonical name. Only a fuller
                # canonical name from another context part can supersede it;
                # merely repeating a part must not promote one of its aliases.
                canonical = max([entry[field], item[field]], key=lambda n: (len(n.split()), len(n), n)) if field == 'name' else old
                entry[field] = canonical
                entry['aliases'] = [n for n in all_names if n != canonical]
                remaps[catalog][item[field]] = canonical
                if catalog == 'constructs':
                    entry['items'] = unique(entry['items'] + item['items'])
                    if entry['definition'] is None:
                        entry['definition'] = item['definition']
                    elif item['definition'] not in (None, entry['definition']):
                        # Both are verbatim; retain a complete attested definition.
                        entry['definition'] = max([entry['definition'], item['definition']], key=len)
        for key in CONTEXT_KEYS[3:]:
            for item in part[key]:
                item = deepcopy(item)
                for field, catalog in (('model_id', 'models'), ('group_id', 'groups')):
                    item[field] = remaps[catalog].get(item[field], item[field])
                result[key].append(item)
    for key in CONTEXT_KEYS:
        result[key] = unique(result[key])
    return normalize_context(result)


def catalog_map(context, catalog, field):
    mapping = {}
    for item in context[catalog]:
        for name in [item[field], *item['aliases']]:
            key = label_key(name)
            if key in mapping and mapping[key] != item[field]:
                raise ValueError(f'Ambiguous catalog alias {name!r}')
            mapping[key] = item[field]
    return mapping


def resolve_record(record, context, strict=False):
    """Map construct names and model/group IDs (or their aliases) to the shared catalog.

    An ID not in the catalog is kept as written (strict=False, pipeline default) or rejected (strict=True).
    """
    record = deepcopy(record)
    mapping = catalog_map(context, 'constructs', 'name')
    for field in VARIABLE_FIELDS:
        if record[field] is not None:
            record[field] = mapping.get(label_key(record[field]), record[field])
    for field, catalog in (('model_id', 'models'), ('group_id', 'groups')):
        mapping = catalog_map(context, catalog, 'id')
        value = record[field]
        if value is None:
            continue
        if label_key(value) in mapping:
            record[field] = mapping[label_key(value)]
        elif strict:
            raise ValueError(f'Unattested {field}: {value}')
    return record


def canonicalize(relations, context, strict=False):
    context = normalize_context(context)
    return [resolve_record(r, context, strict) for r in clean_relations(relations)]


def identity(record, include_context=True):
    fields = (*VARIABLE_FIELDS, 'effect_type') + (('model_id', 'group_id') if include_context else ())
    return tuple(label_key(record.get(f)) for f in fields)


def same_relation(a, b):
    return identity(a) == identity(b)


def coefficient_compatible(a, b):
    ta, tb = a['coefficient_type'], b['coefficient_type']
    if ta == tb or {ta, tb} == {'standardized', 'unstandardized'}:
        return True
    unknown = a if ta == 'unknown' else b
    # Hypotheses with no statistics can donate metadata without inventing a test.
    if all(unknown[f] is None for f in ('beta', 'p', 'significance')):
        return True
    return a['beta'] == b['beta'] or a['beta'] is None or b['beta'] is None


def selected_records(records):
    if any(r['coefficient_type'] == 'standardized' and r['beta'] is not None for r in records):
        return [r for r in records if r['coefficient_type'] != 'unstandardized']
    return records


def merge_pair(a, b):
    """The merged record of two compatible records, or None."""
    if not same_relation(a, b) or not coefficient_compatible(a, b):
        return None
    records = selected_records([a, b])
    result = deepcopy(records[0])
    for other in records[1:]:
        for field in STAT_FIELDS:
            if field == 'p':
                okay, value = merge_p(result[field], other[field])
                if not okay:
                    return None
                result[field] = value
            elif result[field] is not None and other[field] is not None and result[field] != other[field]:
                return None
            elif result[field] is None:
                result[field] = other[field]
        if result['coefficient_type'] == 'unknown':
            result['coefficient_type'] = other['coefficient_type']
    # A projected unstandardized estimate may contribute only path metadata.
    for other in (a, b):
        if result['hypothesis_id'] is None:
            result['hypothesis_id'] = other['hypothesis_id']
        elif other['hypothesis_id'] not in (None, result['hypothesis_id']):
            return None
    result['is_control'] = a['is_control'] or b['is_control']
    return result


def ambiguous_indices(records):
    ambiguous = set()
    for i, record in enumerate(records):
        if not any(record[f] is None for f in STAT_FIELDS):
            continue
        matches = [j for j, other in enumerate(records) if i != j and merge_pair(record, other) is not None]
        if any(merge_pair(records[j], records[k]) is None for j, k in combinations(matches, 2)):
            ambiguous.add(i)
    return ambiguous


def unestimated(record):
    return record['beta'] is None and record['p'] is None


def hypothesis_ids(records):
    """{path: hypothesis_id} of the given records, first record per path wins."""
    found = {}
    for record in records:
        if record['hypothesis_id'] is not None:
            found.setdefault(identity(record, include_context=False), record['hypothesis_id'])
    return found


def drop_unestimated(records):
    """Drop records without beta and p whose directed path (cause, effect,
    mediator, moderator, effect_type) has a beta in any record, whatever its
    model or group. Compatible same-context records have already been merged.

    A dropped record's hypothesis_id is kept: estimated records of its path
    without a hypothesis_id take it over.
    """
    estimated = {identity(r, include_context=False) for r in records if r['beta'] is not None}
    dropped = [r for r in records if unestimated(r) and identity(r, include_context=False) in estimated]
    hypotheses = hypothesis_ids(dropped)
    kept = [r for r in records if not any(r is d for d in dropped)]
    for record in kept:
        if record['hypothesis_id'] is None:
            record['hypothesis_id'] = hypotheses.get(identity(record, include_context=False))
    return kept


def rule_fuse(relations, context=None):
    context = normalize_context(context or {})
    records = sorted(canonicalize(relations, context), key=digest)
    ambiguous = ambiguous_indices(records)
    groups = []
    for i, record in enumerate(records):
        if i in ambiguous:
            groups.append(([i], deepcopy(record)))
            continue
        matches = [j for j, (members, combined) in enumerate(groups)
                   if not any(m in ambiguous for m in members)
                   and all(merge_pair(records[m], record) is not None for m in members)
                   and merge_pair(combined, record) is not None]
        if len(matches) == 1:
            j = matches[0]
            members, combined = groups[j]
            groups[j] = (members + [i], merge_pair(combined, record))
        else:
            groups.append(([i], deepcopy(record)))
    return drop_unestimated([r for _, r in groups])


def disjoint_p(a, b):
    la, ha, ca, da = p_interval(a)
    lb, hb, cb, db = p_interval(b)
    return ha < lb or hb < la or (ha == lb and not (da and cb)) or (hb == la and not (db and ca))


def consolidate_candidates(candidates, context):
    """Deterministic consolidation also serves as the LLM merge validator."""
    context = normalize_context(context)
    originals = {c['candidate_id']: deepcopy(c) for c in candidates}
    records = [resolve_record(c, context) for c in candidates]
    ambiguous = ambiguous_indices(records)
    groups = []
    for i in sorted(range(len(records)), key=lambda n: records[n]['candidate_id']):
        record = records[i]
        matching = []
        for group in groups:
            if i in ambiguous or any(j in ambiguous for j in group):
                continue
            if all(same_relation(record, records[j]) and coefficient_compatible(record, records[j]) for j in group):
                ps = [r['p'] for r in selected_records([record, *(records[j] for j in group)]) if r['p'] is not None]
                if any(not merge_p(a, b)[0] and not disjoint_p(a, b) for a, b in combinations(ps, 2)):
                    continue
                matching.append(group)
        if len(matching) == 1:
            matching[0].append(i)
        else:
            groups.append([i])
    # Same rule as rule_fuse, judged on the candidates (a contradiction nulls its beta):
    # a group without beta/p on a path estimated in any other group is dropped.
    estimated = {identity(r, False) for r in records if r['beta'] is not None}
    dropped = {i for group in groups if all(unestimated(records[i]) for i in group)
               and identity(records[group[0]], False) in estimated for i in group}
    dropped_hypotheses = hypothesis_ids(records[i] for i in sorted(dropped, key=lambda n: records[n]['candidate_id']))
    output = []
    for group in groups:
        if group[0] in dropped:
            continue
        members = [records[i] for i in group]
        selected = selected_records(members)
        result = semantic_record(selected[0])
        conflicting = []
        for field in STAT_FIELDS:
            relevant = members if field == 'hypothesis_id' else selected
            values = unique([r[field] for r in relevant if r[field] is not None])
            value = values[0] if values else None
            for other in values[1:]:
                if field == 'p':
                    okay, value = merge_p(value, other)
                    if not okay:
                        conflicting.append(field)
                        value = None
                        break
                elif value != other:
                    conflicting.append(field)
                    value = None
                    break
            result[field] = value
        if result['hypothesis_id'] is None and 'hypothesis_id' not in conflicting:
            # Keep the hypothesis of a dropped hypothesis-only candidate on this path.
            result['hypothesis_id'] = dropped_hypotheses.get(identity(result, False))
        types = {r['coefficient_type'] for r in selected} - {'unknown'}
        if len(types) == 1:
            result['coefficient_type'] = next(iter(types))
        result['is_control'] = any(r['is_control'] for r in members)
        ids = sorted(r['candidate_id'] for r in members)
        modalities = sorted({r['modality'] for r in members})
        result.update(candidate_ids=ids, supported_by=modalities, alternatives=[])
        uncertain = any(i in ambiguous for i in group)
        if len(group) == 1:
            uncertain |= any(j not in group and j not in dropped and identity(records[j], False) == identity(members[0], False)
                             and not any(records[j][f] is not None and members[0][f] is not None
                                         and records[j][f] != members[0][f] for f in ('model_id', 'group_id'))
                             and (not same_relation(records[j], members[0])
                                  or not coefficient_compatible(records[j], members[0])
                                  or (records[j]['p'] and members[0]['p'] and not merge_p(records[j]['p'], members[0]['p'])[0]
                                      and not disjoint_p(records[j]['p'], members[0]['p'])))
                             for j in range(len(records)))
        if conflicting:
            result['status'] = 'contradiction'
            result['alternatives'] = [dict(candidate_id=cid, **semantic_record(originals[cid])) for cid in ids]
        elif uncertain:
            result['status'] = 'uncertain'
        elif len(modalities) == 1:
            result['status'] = 'single_modality'
        else:
            identical = all(all(r[f] == members[0][f] for f in (*STAT_FIELDS, 'coefficient_type', 'is_control')) for r in members)
            result['status'] = 'agreement' if identical else 'complementary'
        output.append(result)
    return output


def conflicts(relations, context=None):
    context = normalize_context(context or {})
    records = canonicalize(relations, context)
    return [{'relation_indices': [i, j], 'reason': 'Unresolved estimates in the same established context'}
            for i, j in combinations(range(len(records)), 2)
            if same_relation(records[i], records[j]) and merge_pair(records[i], records[j]) is None]
