import random
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .common import FORMAT_VERSION, digest, normalize_name, read_json, write_json
from .utils import (strict_missing as missing, normalize_p,
                    evaluation_p_level as p_level, rounded_beta)

EVALUATION_VERSION = 7
SIZES = ["3", "4", "5"]
SCORE_TYPES = ["entities", *SIZES]
NAME_MATCHING_MODES = ("aliases", "normalized", "exact", "relaxed")
RELAXED_TAU_LEV = 0.9
RELAXED_TAU_DICE = 0.65
_STOPWORDS = frozenset({"a", "an", "the", "of", "to", "in", "on", "for", "and", "with",
                        "by", "at", "from", "toward", "towards"})
_R2_PATTERN = re.compile(r"\(?\s*\br\s*(\^\s*2|²|2)\s*[=:]\s*-?\d*\.?\d+\s*\)?", re.IGNORECASE)
_ABBREV_PATTERN = re.compile(r"^[A-Z][A-Z0-9&]{1,7}$")
_EXTENSION = re.compile(r"\.(pdf|json|md|txt|xml|html)$", re.IGNORECASE)


def strip_extension(value):
    return _EXTENSION.sub("", str(value))


class Unmatchable:
    """Stands in for a missing beta/p when match_missing=False; equal only to itself."""
    __slots__ = ()

    def __repr__(self):
        return "missing"


def name_key(value, name_matching="aliases"):
    """Comparison form of a variable name: the exact string (strict) or its normalized form."""
    if name_matching == "exact":
        return value if isinstance(value, str) else ""
    if name_matching == "relaxed":
        return relaxed_normalize_name(value)
    return normalize_name(value)


# --------------------------------------------------------------------------- relaxed names

def relaxed_normalize_name(value):
    """The calibration code's normalization, used only for relaxed matching."""
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", value)
    value = _R2_PATTERN.sub(" ", value).lower().replace("-", " ").replace("_", " ")
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


@lru_cache(maxsize=1)
def _relaxed_stemmer():
    try:
        from nltk.stem.snowball import SnowballStemmer
    except ImportError:
        return None
    return SnowballStemmer("english")


@lru_cache(maxsize=32768)
def _content_tokens(normalized, stemmer=None):
    tokens = [t for t in normalized.split() if t not in _STOPWORDS]
    return tuple(stemmer.stem(t) for t in tokens) if stemmer is not None else tuple(tokens)


def levenshtein_similarity(a, b):
    """1 - edit distance / length of the longer string; empty inputs score zero."""
    if not a or not b:
        return 0.0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / len(a)


def dice(tokens_a, tokens_b):
    """Sørensen–Dice similarity of content-token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    return 2 * len(tokens_a & tokens_b) / (len(tokens_a) + len(tokens_b))


def _name_similarity(gold, predicted, stemmer):
    return (levenshtein_similarity(" ".join(_content_tokens(gold)),
                                   " ".join(_content_tokens(predicted))),
            dice(frozenset(_content_tokens(gold, stemmer)),
                 frozenset(_content_tokens(predicted, stemmer))))


def relaxed_construct_aliases(context, article_id=None):
    """Map each normalized context name/alias to all raw names of that construct.

    Keep all strings, as in the calibration loader. Relation missing-value rules
    do not apply here: labels such as "NA" can be construct abbreviations.
    """
    if not isinstance(context, dict):
        raise ValueError(f"Context for {article_id} must be an object")
    constructs = context.get("constructs") or []
    if not isinstance(constructs, list):
        raise ValueError(f"Context constructs for {article_id} must be a list")
    result = defaultdict(set)
    for index, construct in enumerate(constructs):
        if not isinstance(construct, dict):
            raise ValueError(f"Context constructs for {article_id} must be objects")
        aliases = construct.get("aliases") or []
        if not isinstance(aliases, list):
            raise ValueError(f"Context aliases for {article_id} must be a list")
        names = [construct.get("name"), *aliases]
        for position, name in enumerate(names):
            if not isinstance(name, str):
                field = "name" if position == 0 else f"aliases[{position - 1}]"
                raise ValueError(f"Context constructs[{index}].{field} for {article_id} "
                                 f"must be a string; got {name!r}")
        for name in names:
            result[relaxed_normalize_name(name)].update(names)
    return {key: sorted(names) for key, names in result.items()}


def relaxed_name_alignment(predictions, gold, aliases=None):
    """Injective predicted -> gold mapping using the calibrated name-only algorithm.

    Normalized identities win, then max(Levenshtein, Dice), then the score of the
    original predicted name. Either threshold suffices. Nonidentical uppercase
    abbreviations need an eligible alias to participate.
    """
    stemmer = _relaxed_stemmer()
    if stemmer is None:
        warnings.warn("nltk not installed: relaxed Dice is computed without stemming.", stacklevel=2)
    aliases = aliases or {}
    gold_names = sorted({relaxed_normalize_name(n) for r in gold for n in construct_names(r)})
    predicted_names = {}
    for relation in predictions:
        for raw in construct_names(relation):
            name = relaxed_normalize_name(raw)
            predicted_names[name] = predicted_names.get(name, False) or bool(_ABBREV_PATTERN.fullmatch(raw.strip()))
    candidates = []
    for g in gold_names:
        for v, abbreviated in predicted_names.items():
            identical = g == v
            blocked = not identical and abbreviated
            lev, token_dice = _name_similarity(g, v, stemmer)
            original_score = 1.0 if identical else 0.0 if blocked else max(lev, token_dice)
            if blocked:
                lev = token_dice = 0.0
            for alias in aliases.get(v, ()):
                name = relaxed_normalize_name(alias)
                if name == g or not _ABBREV_PATTERN.fullmatch(alias.strip()):
                    alias_lev, alias_dice = _name_similarity(g, name, stemmer)
                    lev, token_dice = max(lev, alias_lev), max(token_dice, alias_dice)
                    blocked = False
            if identical or (not blocked and (lev >= RELAXED_TAU_LEV or token_dice >= RELAXED_TAU_DICE)):
                score = 1.0 if identical else max(lev, token_dice)
                candidates.append((not identical, -score, -original_score, g, v))
    used_gold, mapping = set(), {}
    for _, _, _, g, v in sorted(candidates):
        if g not in used_gold and v not in mapping:
            used_gold.add(g)
            mapping[v] = g
    return mapping


# --------------------------------------------------------------------------- relations

def adapt_relation(item):
    """Validate and normalize one relation; raises ValueError if it cannot be scored."""
    cause = item.get("cause", item.get("Construct from"))
    effect = item.get("effect", item.get("Construct to"))
    if missing(cause) or missing(effect):
        raise ValueError("Every relation needs a cause and effect")
    if not isinstance(cause, str) or not isinstance(effect, str):
        raise ValueError("Construct names must be strings")
    moderator = item.get("moderator", item.get("Construct Moderator"))
    mediator = item.get("mediator", item.get("Construct Mediator"))
    moderator = None if missing(moderator) else moderator
    mediator = None if missing(mediator) else mediator
    if moderator is not None and not isinstance(moderator, str):
        raise ValueError("Moderator must be a construct name or null")
    if mediator is not None:
        chain = mediator if isinstance(mediator, list) else [mediator]
        if not chain or any(not isinstance(v, str) or missing(v) for v in chain):
            raise ValueError("Mediator must be a name, a nonempty ordered list of names, or null")
    significance = item.get("significance")
    # Legacy significance contains a p-value string, not a boolean decision.
    legacy_p = significance if isinstance(significance, str) and not missing(significance) else None
    p = normalize_p(item.get("p", legacy_p))
    beta = item.get("beta", item.get("Path coefficient"))
    beta = None if missing(beta) else beta
    if beta is not None:
        rounded_beta(beta)  # validates the coefficient
    effect_type = item.get("effect_type") or ("indirect" if mediator else "moderation" if moderator else "direct")
    if effect_type not in {"direct", "indirect", "moderation"}:
        raise ValueError(f"Unsupported effect type: {effect_type}")
    if effect_type == "moderation" and mediator:
        raise ValueError("Moderated mediation must use effect_type=indirect")
    if effect_type == "direct" and (moderator or mediator):
        raise ValueError("Direct relations cannot contain mediation/moderation roles")
    model_id = item.get("model_id", item.get("model"))
    group_id = item.get("group_id")
    return {
        "cause": cause, "effect": effect, "moderator": moderator, "mediator": mediator,
        "beta": beta, "p": p,
        "significance": significance if isinstance(significance, bool) else None,
        "effect_type": effect_type,
        "model_id": None if missing(model_id) else str(model_id),
        "group_id": None if missing(group_id) else str(group_id),
    }


def adapt_relations(relations, article_id=None, exclude_controls=False, stages=None, keep_raw=False):
    """Return (relations, invalid, skipped).

    Total effects are skipped. Relations can be filtered by stage (gold) or by is_control
    (predictions). Relations that cannot be scored are returned in `invalid` with the reason
    (and the raw relation if keep_raw=True); callers decide how to count them.
    """
    if not isinstance(relations, list):
        raise ValueError(f"Relations for {article_id} must be a list")
    output, invalid, skipped = [], [], Counter()
    for index, item in enumerate(relations):
        try:
            if not isinstance(item, dict):
                raise ValueError("Relation must be an object")
            if item.get("effect_type") == "total":
                skipped["total_effect"] += 1
                continue
            if stages is not None and item.get("stage") not in stages:
                skipped["stage"] += 1
                continue
            if exclude_controls and item.get("is_control") is True:
                skipped["control"] += 1
                continue
            output.append(adapt_relation(item))
        except ValueError as error:
            entry = {"index": index, "error": str(error)}
            if keep_raw:
                entry["relation"] = item
            invalid.append(entry)
    return output, invalid, dict(skipped)


def construct_names(relation):
    """All construct names in a relation: cause, effect, moderator, and mediator chain."""
    mediator = relation.get("mediator")
    chain = mediator if isinstance(mediator, list) else [mediator] if mediator else []
    return [v for v in [relation["cause"], relation["effect"], relation.get("moderator"), *chain]
            if v is not None]


# --------------------------------------------------------------------------- loading

def construct_aliases(document, article_id):
    """Map every alias in document.constructs to its construct name."""
    aliases = {}
    for construct in document.get("constructs") or []:
        if not isinstance(construct, dict):
            raise ValueError(f"Constructs in {article_id} must be objects")
        name = construct.get("name")
        names = construct.get("aliases") or []
        if names and (not isinstance(name, str) or missing(name)):
            raise ValueError(f"Construct with aliases has no name in {article_id}")
        for alias in names:
            if not isinstance(alias, str) or missing(alias):
                continue
            if alias in aliases and aliases[alias] != name:
                raise ValueError(f"Alias {alias!r} assigned to multiple constructs in {article_id}")
            aliases[alias] = name
    return aliases


def _is_paper(value):
    """True for an object with document.relations (or relations) as a list."""
    if not isinstance(value, dict):
        return False
    document = value.get("document", value)
    return isinstance(document, dict) and isinstance(document.get("relations"), list)


def find_gold_papers(data):
    """Locate paper entries in a ground-truth file.

    Returns ([(key, entry), ...], ignored_top_level_keys).
    """
    def from_list(items, where):
        papers = []
        for position, item in enumerate(items):
            if not _is_paper(item):
                raise ValueError(f"Entry #{position} in {where} is not a paper with document.relations")
            identifier = item.get("paper_id") or item.get("article_id")
            if not identifier:
                raise ValueError(f"Entry #{position} in {where} has no paper_id")
            papers.append((str(identifier), item))
        return papers

    if isinstance(data, list):
        return from_list(data, "the ground-truth list"), []
    if not isinstance(data, dict) or not data:
        raise ValueError("Ground truth must be a nonempty JSON object or list")

    # Case 1: papers at the top level, possibly next to metadata such as schema_version.
    top = [(str(k), v) for k, v in data.items() if _is_paper(v)]
    if top:
        ignored = [str(k) for k, v in data.items() if not _is_paper(v)]
        for key in ignored:
            if isinstance(data[key], dict) and "paper_id" in data[key]:
                raise ValueError(f"Ground-truth entry {key!r} has a paper_id but no document.relations")
        return top, ignored

    # Case 2: papers inside exactly one container, e.g. "papers": {...} or "papers": [...].
    containers = [k for k, v in data.items()
                  if (isinstance(v, dict) and any(_is_paper(x) for x in v.values()))
                  or (isinstance(v, list) and any(_is_paper(x) for x in v))]
    if not containers:
        raise ValueError(f"Could not locate papers in the ground truth; top-level keys: {list(data)[:10]}")
    if len(containers) > 1:
        raise ValueError(f"Papers found under several top-level keys: {containers}")
    key = containers[0]
    value = data[key]
    if isinstance(value, list):
        papers = from_list(value, repr(key))
    else:
        bad = [k for k, v in value.items() if not _is_paper(v)]
        if bad:
            raise ValueError(f"Entries under {key!r} without document.relations: {bad[:5]}")
        papers = [(str(k), v) for k, v in value.items()]
    return papers, [str(k) for k in data if k != key]


def load_gold(gt_path, stages=None, invalid_gold="skip"):
    """Return (ground-truth articles in file order, ignored top-level keys).

    invalid_gold="skip" leaves relations that cannot be scored out of the gold set and reports
    them per article; invalid_gold="raise" lists all of them in one error.
    """
    if invalid_gold not in {"skip", "raise"}:
        raise ValueError("invalid_gold must be 'skip' or 'raise'")
    papers, ignored = find_gold_papers(read_json(gt_path))
    articles, seen = [], set()
    for key, entry in papers:
        article_id = str(entry.get("paper_id") or strip_extension(key))
        if article_id in seen:
            raise ValueError(f"Duplicate ground-truth paper ID: {article_id}")
        seen.add(article_id)
        document = entry.get("document", entry)
        relations, invalid, skipped = adapt_relations(document["relations"], article_id=article_id,
                                                      stages=stages, keep_raw=True)
        articles.append({"article_id": article_id, "key": key, "relations": relations,
                         "aliases": construct_aliases(document, article_id),
                         "skipped": skipped, "invalid": invalid})
    if not articles:
        raise ValueError("The ground truth contains no papers")
    problems = [(a["article_id"], x) for a in articles for x in a["invalid"]]
    if problems and invalid_gold == "raise":
        listing = "\n".join(f"  {article_id} #{x['index']}: {x['error']}" for article_id, x in problems)
        raise ValueError(f"{len(problems)} invalid ground-truth relations:\n{listing}\n"
                         "Fix them or pass invalid_gold='skip'")
    if problems:
        warnings.warn(f"Skipped {len(problems)} invalid ground-truth relations in "
                      f"{len({article_id for article_id, _ in problems})} papers; "
                      "see 'invalid_gold' in the report", stacklevel=3)
    return articles, ignored


def index_predictions(data, id_map=None):
    """Map paper ID -> (source key, raw relation list or None)."""
    id_map = id_map or {}
    index, sources = {}, set()

    def add(source, relations):
        if source in sources:
            raise ValueError(f"Duplicate prediction entry: {source!r}")
        sources.add(source)
        if relations is not None and not isinstance(relations, list):
            raise ValueError(f"Predictions for {source!r} must be a list of relations")
        identifier = str(id_map.get(source, source))
        for key in {identifier, strip_extension(identifier)}:
            if key in index and index[key][0] != source:
                raise ValueError(f"Prediction entries {index[key][0]!r} and {source!r} map to the same ID")
            index[key] = (source, relations)

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = value.get("relations", (value.get("document") or {}).get("relations"))
            add(str(key), value)
    elif isinstance(data, list):
        for position, entry in enumerate(data):
            identifier = (entry.get("article_id") or entry.get("paper_id")) if isinstance(entry, dict) else None
            if not identifier:
                raise ValueError(f"Prediction entry #{position} has no article_id/paper_id")
            add(str(identifier), entry.get("relations"))
    else:
        raise ValueError("Predictions must be a JSON object or list")
    return index


# --------------------------------------------------------------------------- aliases

def resolve_aliases(values):
    aliases = {}
    for key, target in values.items():
        key, target = normalize_name(key), normalize_name(target)
        if not key or not target:
            raise ValueError("Aliases and targets must be nonempty names")
        if key in aliases and aliases[key] != target:
            raise ValueError(f"Conflicting normalized alias: {key}")
        aliases[key] = target
    for key in aliases:
        target, seen = key, set()
        while target in aliases and aliases[target] != target:
            if target in seen:
                raise ValueError(f"Alias cycle involving {key}")
            seen.add(target)
            target = aliases[target]
        aliases[key] = target
    return aliases


def one_to_one_aliases(relations, aliases):
    """Restrict an alias map so each construct is claimed by at most one surface name.

    Among the distinct normalized names in `relations` that resolve to the same construct,
    the canonical name wins if present; otherwise the most frequently used name, then the
    alphabetically first. The remaining names stay unmapped and cannot match gold names.
    The choice depends only on `relations`, never on the gold relations.
    Returns (restricted_aliases, collisions).
    """
    usage = Counter(normalize_name(v) for r in relations for v in construct_names(r))
    groups = defaultdict(list)
    for surface in usage:
        groups[aliases.get(surface, surface)].append(surface)
    restricted, collisions = {}, {}
    for canonical, surfaces in groups.items():
        winner = canonical if canonical in surfaces else min(surfaces, key=lambda s: (-usage[s], s))
        if winner != canonical:
            restricted[winner] = canonical
        if len(surfaces) > 1:
            collisions[canonical] = {"mapped": winner,
                                     "unmapped": sorted(s for s in surfaces if s != winner)}
    return restricted, collisions


# --------------------------------------------------------------------------- scoring

def tuple_key(relation, size, aliases=None, include_context=False, match_missing=True,
              name_matching="aliases"):
    if size not in {3, 4, 5}:
        raise ValueError("Tuple size must be 3, 4, or 5")
    if name_matching not in NAME_MATCHING_MODES:
        raise ValueError(f"name_matching must be one of {NAME_MATCHING_MODES}")
    aliases = (aliases or {}) if name_matching in {"aliases", "relaxed"} else {}

    def name(value):
        key = name_key(value, name_matching)
        return aliases.get(key, key)

    mediator = relation.get("mediator")
    chain = mediator if isinstance(mediator, list) else [mediator] if mediator else []
    # The middle element is typed and retains both roles for moderated mediation.
    # Roles: mediator chain and moderator; effect_type is not compared.
    role = (tuple(name(v) for v in chain), name(relation.get("moderator")))

    key = (name(relation["cause"]), role, name(relation["effect"]))
    if size >= 4:
        beta = rounded_beta(relation["beta"])
        key += (Unmatchable() if beta is None and not match_missing else beta,)
    if size >= 5:
        level = p_level(relation["p"])
        key += (Unmatchable() if level is None and not match_missing else level,)
    if include_context:
        key += (normalize_name(relation.get("model_id")), normalize_name(relation.get("group_id")))
    return key


def describe_key(key, size, include_context):
    """Readable, JSON-serializable form of a tuple key."""
    cause, (chain, moderator), effect = key[:3]
    described = {"cause": cause, "effect": effect,
                 "mediators": list(chain), "moderator": moderator or None}
    rest = list(key[3:])
    if size >= 4:
        beta = rest.pop(0)
        described["beta"] = None if beta is None else str(beta)
    if size >= 5:
        level = rest.pop(0)
        described["p_level"] = None if level is None else str(level)
    if include_context:
        described["model_id"], described["group_id"] = rest
    return described


def metrics(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * tp / (predicted + gold) if predicted + gold else 0.0
    return dict(tp=tp, predicted=predicted, gold=gold, precision=precision, recall=recall, f1=f1)


def score_entities(predictions, gold, aliases=None, prediction_aliases=None,
                   name_matching="aliases", relaxed_aliases=None):
    """Score distinct variables in adapted relations, independent of their roles.

    Count each name once per article after applying the selected normalization
    and mapping. Manual many-to-one mappings collapse to a single construct.
    Relaxed mode uses the same one-to-one name algorithm as relation scoring.
    Names must come from the valid, filtered relations used for scoring; context
    catalogs only supply aliases and do not add entities to either inventory.
    """
    prediction_aliases = aliases if prediction_aliases is None else prediction_aliases
    if name_matching == "relaxed":
        aliases = {}
        prediction_aliases = relaxed_name_alignment(predictions, gold, relaxed_aliases)
    return _score_aligned_entities(predictions, gold, aliases, prediction_aliases, name_matching)


def _score_aligned_entities(predictions, gold, aliases, prediction_aliases, name_matching):
    if name_matching not in NAME_MATCHING_MODES:
        raise ValueError(f"name_matching must be one of {NAME_MATCHING_MODES}")

    def names(relations, mapping):
        mapping = (mapping or {}) if name_matching in {"aliases", "relaxed"} else {}
        keys = {name_key(name, name_matching) for relation in relations for name in construct_names(relation)}
        return {mapping.get(key, key) for key in keys}

    predicted, expected = names(predictions, prediction_aliases), names(gold, aliases)
    matched = predicted & expected
    spurious, missing_names = predicted - expected, expected - predicted
    return {**metrics(len(matched), len(predicted), len(expected)),
            "fp": len(spurious), "fn": len(missing_names),
            "matched": sorted(matched), "missing": sorted(missing_names), "spurious": sorted(spurious)}


def score_article(predictions, gold, size, aliases=None, include_context=False,
                  prediction_aliases=None, invalid=0, match_missing=True, name_matching="aliases",
                  relaxed_aliases=None):
    """Score one article on its unique tuples: each distinct tuple counts once per side.
    `aliases` applies to gold (and to predictions unless `prediction_aliases` is given);
    aliases are only used with name_matching="aliases". `invalid` predictions count as spurious.
    With name_matching="relaxed", relaxed_aliases is the output of relaxed_construct_aliases();
    one name alignment is selected independently of tuple size, coefficients and p-values.
    """
    prediction_aliases = aliases if prediction_aliases is None else prediction_aliases
    if name_matching == "relaxed":
        aliases = {}
        prediction_aliases = relaxed_name_alignment(predictions, gold, relaxed_aliases)
    return _score_aligned_article(predictions, gold, size, aliases, include_context,
                                  prediction_aliases, invalid, match_missing, name_matching)


def _score_aligned_article(predictions, gold, size, aliases, include_context,
                           prediction_aliases, invalid, match_missing, name_matching):
    predicted = Counter(set(tuple_key(r, size, prediction_aliases, include_context, match_missing, name_matching)
                            for r in predictions))
    expected = Counter(set(tuple_key(r, size, aliases, include_context, match_missing, name_matching)
                           for r in gold))
    matched = predicted & expected
    result = metrics(sum(matched.values()), sum(predicted.values()) + invalid, sum(expected.values()))
    result["invalid"] = invalid
    result["missing"] = [{**describe_key(key, size, include_context), "count": count}
                         for key, count in (expected - predicted).items()]
    result["spurious"] = [{**describe_key(key, size, include_context), "count": count}
                          for key, count in (predicted - expected).items()]
    return result


def summarize(records):
    """Micro and article-macro scores, including entities when present in the records."""
    summary = {}
    score_types = SCORE_TYPES if any("entities" in r["scores"] for r in records) else SIZES
    for size in score_types:
        scores = [r["scores"][size] for r in records]
        micro = metrics(sum(s["tp"] for s in scores), sum(s["predicted"] for s in scores),
                        sum(s["gold"] for s in scores))
        if size == "entities":
            micro.update(fp=micro["predicted"] - micro["tp"], fn=micro["gold"] - micro["tp"])
        macro = {name: sum(s[name] for s in scores) / len(scores) for name in ["precision", "recall", "f1"]}
        summary[size] = {"micro": micro, "macro": macro, "articles": len(scores)}
    return summary


# --------------------------------------------------------------------------- evaluation
def manual_prediction_aliases(mapping, aliases, article_id):
    """Translate a reviewed {extracted name: GT construct} map into normalized prediction aliases.

    Targets are resolved through the gold alias map, so they land on the same canonical key
    as the gold relations. Returns {normalized extracted name: canonical gold key}.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"Name map for {article_id} must be an object")
    result = {}
    for surface, target in mapping.items():
        if not isinstance(surface, str) or not isinstance(target, str) or missing(surface) or missing(target):
            raise ValueError(f"Name map for {article_id} must map construct names to construct names")
        key, goal = normalize_name(surface), normalize_name(target)
        goal = aliases.get(goal, goal)
        if key in result and result[key] != goal:
            raise ValueError(f"Name map for {article_id} maps {surface!r} to several constructs")
        result[key] = goal
    return result


def manual_mapping_report(predictions, mapping, predicted_aliases):
    """Diagnostics: map entries not used by this run, and constructs claimed by several names."""
    used = {normalize_name(v) for r in predictions for v in construct_names(r)}
    claims = defaultdict(set)
    for surface in used:
        claims[predicted_aliases.get(surface, surface)].add(surface)
    return {"entries": len(mapping),
            "unused": sorted(s for s in mapping if normalize_name(s) not in used),
            "many_to_one": {k: sorted(v) for k, v in claims.items() if len(v) > 1}}

            
def evaluate(gt_path, predictions_path, output_path=None, split=None, aliases_path=None,
             include_context=False, limit=None, gold_stages=None, exclude_controls=False,
             match_missing=True, id_map=None, invalid_gold="skip", name_matching="aliases",
             name_map=None, contexts=None):
    """Evaluate one predictions file against one ground-truth file.

    split:            label stored in the report (reports are only compared within one split)
    aliases_path:     optional JSON {article_id: {alias: construct name}} added to the GT aliases
                      (only with name_matching="aliases")
    include_context:  also require matching model_id and group_id
    limit:            evaluate only the first N ground-truth papers
    gold_stages:      e.g. {"tested"} to keep only gold relations with these stage values
    exclude_controls: drop predicted relations with is_control=True
    match_missing:    if False, a missing beta/p never matches (sizes 4/5 become strict)
    id_map:           dict or JSON path {prediction key: ground-truth paper id}
    invalid_gold:     "skip" (report and leave out) or "raise" for gold relations that cannot be scored
    name_matching:    "aliases" (normalized + GT aliases, default), "normalized" (no aliases),
                      "exact" (strict: identical strings, no normalization, no aliases), or
                      "relaxed" (one-to-one names, Levenshtein >= 0.9 OR token Dice >= 0.65)
    name_map:         dict or JSON path {article_id: {extracted name: GT construct name}}, e.g. the
                      construct review tool's output. Predictions then use only these reviewed
                      mappings (plus normalized identity) instead of automatic GT-alias matching;
                      gold names are still resolved through the GT aliases. Several extracted names
                      may map to one construct (identical tuples still count once).
                      Requires name_matching="aliases".
    contexts:         optional dict or JSON path {article_id: {constructs: [...]}} from extraction,
                      used only with name_matching="relaxed". Scores take the best of each
                      predicted name and its aliases; GT aliases and manual maps are not used.
                      Exact normalized names are aligned first, then highest name similarity,
                      with original-name similarity breaking ties. Dice uses English Snowball
                      stemming when nltk is installed (otherwise warns and uses unstemmed tokens).
                      All tuple sizes share this alignment and retain the existing beta/p rules.
    Ground-truth papers without predictions are scored as empty extractions.
    scores/summary["entities"] scores distinct variables in valid, filtered relations,
    regardless of role, direction, coefficient, p-value, model or group. It reuses
    the relation name mapping and counts each resulting name once per article.
    Invalid/skipped relations and context-only constructs do not contribute entities.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if name_matching not in NAME_MATCHING_MODES:
        raise ValueError(f"name_matching must be one of {NAME_MATCHING_MODES}")
    if aliases_path and name_matching != "aliases":
        raise ValueError("aliases_path only applies with name_matching='aliases'")
    if name_map is not None and name_matching != "aliases":
        raise ValueError("name_map requires name_matching='aliases': gold names are resolved through GT aliases")
    if contexts is not None and name_matching != "relaxed":
        raise ValueError("contexts only applies with name_matching='relaxed'")
    use_aliases = name_matching == "aliases"
    stages = set(gold_stages) if gold_stages is not None else None
    matching = {"aliases": "gt_aliases_1to1+file" if aliases_path else "gt_aliases_1to1",
                "normalized": "normalized", "exact": "exact", "relaxed": "relaxed_names_1to1"}[name_matching]
    if name_map is not None:
        matching = "gt_aliases+manual_map" + ("+file" if aliases_path else "")
    settings = {"split": split,
                "matching": matching,
                "name_matching": name_matching,
                "entity_scope": "scored_relations",
                "include_context": include_context,
                "gold_stages": sorted(stages) if stages is not None else None,
                "exclude_controls": exclude_controls,
                "match_missing": match_missing,
                "invalid_gold": invalid_gold}
    if name_matching == "relaxed":
        settings["relaxed_thresholds"] = {"lev": RELAXED_TAU_LEV, "dice": RELAXED_TAU_DICE}
        settings["relaxed_stemming"] = "english_snowball" if _relaxed_stemmer() is not None else None
        settings["relaxed_context_aliases"] = contexts is not None
    contexts_source = None
    if isinstance(contexts, (str, Path)):
        contexts_source = str(Path(contexts).resolve())
        contexts = read_json(contexts)
    if contexts is not None and not isinstance(contexts, dict):
        raise ValueError("contexts must be a JSON object {article_id: {constructs: [...]}}")
    gold_articles, ignored_gt_keys = load_gold(gt_path, stages, invalid_gold)
    name_map_source = None
    if isinstance(name_map, (str, Path)):
        name_map_source = str(Path(name_map).resolve())
        name_map = read_json(name_map)
    if name_map is not None:
        if not isinstance(name_map, dict):
            raise ValueError("name_map must be a JSON object {article_id: {extracted name: GT construct}}")
        unknown = sorted(set(name_map) - {a["article_id"] for a in gold_articles})
        if unknown:
            raise ValueError(f"name_map contains paper IDs that are not in the ground truth: {unknown[:5]}")
    if limit is not None:
        gold_articles = gold_articles[:limit]
    if isinstance(id_map, (str, Path)):
        id_map = read_json(id_map)
    index = index_predictions(read_json(predictions_path), id_map)
    extra_aliases = read_json(aliases_path) if aliases_path else {}

    records, without_predictions, used_sources = [], [], set()
    for article in gold_articles:
        article_id = article["article_id"]
        lookups = (article_id, str(article["key"]), strip_extension(article["key"]))
        candidates = {index[k][0]: index[k][1] for k in lookups if k in index}
        if len(candidates) > 1:
            raise ValueError(f"Several prediction entries match {article_id}: {sorted(candidates)}")
        source, raw = next(iter(candidates.items()), (None, None))
        if source is not None:
            used_sources.add(source)
        if raw is None:
            without_predictions.append(article_id)
            raw = []
        predictions, invalid, skipped = adapt_relations(raw, article_id=article_id,
                                                        exclude_controls=exclude_controls)
        gold = article["relations"]
        manual, mapping = None, None
        if use_aliases:
            aliases = resolve_aliases({**article["aliases"], **extra_aliases.get(article_id, {})})
            if name_map is not None:
                # Reviewed decisions replace automatic alias matching on the prediction side only.
                mapping = name_map.get(article_id, {})
                predicted_aliases = manual_prediction_aliases(mapping, aliases, article_id)
                prediction_collisions = {}
                manual = manual_mapping_report(predictions, mapping, predicted_aliases)
            else:
                # Gold keeps the full many-to-one mapping; predictions may claim each construct with one name only.
                predicted_aliases, prediction_collisions = one_to_one_aliases(predictions, aliases)
            _, gold_collisions = one_to_one_aliases(gold, aliases)
        else:
            aliases, predicted_aliases, prediction_collisions, gold_collisions = {}, {}, {}, {}
        relaxed_aliases = None
        if name_matching == "relaxed":
            context_keys = dict.fromkeys((*lookups, source, strip_extension(source) if source else None))
            context_matches = [key for key in context_keys if key in (contexts or {})]
            if len(context_matches) > 1:
                raise ValueError(f"Several context entries match {article_id}: {context_matches}")
            context = contexts[context_matches[0]] if context_matches else {}
            relaxed_aliases = relaxed_construct_aliases(context, article_id)
            predicted_aliases = relaxed_name_alignment(predictions, gold, relaxed_aliases)
        scores = {"entities": _score_aligned_entities(predictions, gold, aliases, predicted_aliases, name_matching)}
        scores.update({size: _score_aligned_article(predictions, gold, int(size), aliases, include_context,
                                                   predicted_aliases, len(invalid), match_missing, name_matching)
                       for size in SIZES})
        records.append({
            "article_id": article_id,
            "prediction_source": source,
            "scores": scores,
            "invalid_predictions": invalid,
            "invalid_gold": article["invalid"],
            "skipped": {"gold": article["skipped"], "predictions": skipped},
            "prediction_alias_collisions": prediction_collisions,
            "gold_alias_collisions": gold_collisions,
            "manual_mapping": manual,
            "coverage": {"prediction_beta": sum(r["beta"] is not None for r in predictions),
                         "prediction_p": sum(r["p"] is not None for r in predictions),
                         "gold_beta": sum(r["beta"] is not None for r in gold),
                         "gold_p": sum(r["p"] is not None for r in gold)},
            "gold_fingerprint": digest(gold),
            "alias_fingerprint": digest(aliases),
            "name_map_fingerprint": digest(mapping) if mapping is not None else None,
            "prediction_fingerprint": digest(raw),
            **({"relaxed_name_mapping": predicted_aliases,
                "relaxed_alias_fingerprint": digest(relaxed_aliases)} if name_matching == "relaxed" else {}),
        })

    if index and not used_sources:
        gold_examples = [a["article_id"] for a in gold_articles[:3]]
        prediction_examples = sorted({source for source, _ in index.values()})[:3]
        raise ValueError(f"No prediction IDs match ground-truth IDs (ground truth e.g. {gold_examples}, "
                         f"predictions e.g. {prediction_examples}); pass id_map to translate them")

    sources = {"ground_truth": str(Path(gt_path).resolve()),
               "predictions": str(Path(predictions_path).resolve())}
    if name_map is not None:
        sources["name_map"] = name_map_source or "in-memory dict"
    if contexts is not None:
        sources["contexts"] = contexts_source or "in-memory dict"
    result = {"evaluation_version": EVALUATION_VERSION, "format_version": FORMAT_VERSION,
              "settings": settings,
              "sources": sources,
              "summary": summarize(records),
              "invalid_gold_relations": sum(len(r["invalid_gold"]) for r in records),
              "invalid_prediction_relations": sum(len(r["invalid_predictions"]) for r in records),
              "ground_truth_ignored_keys": ignored_gt_keys,
              "articles_without_predictions": without_predictions,
              "predictions_without_gold": sorted({s for s, _ in index.values()} - used_sources),
              "articles": records}
    if output_path:
        write_json(output_path, result)
    return result


def compare_reports(first, second, size="5", samples=10000, seed=42):
    """Paired bootstrap/permutation tests for entity or tuple F1 (second minus first).

    size: "entities", "3", "4", or "5".
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    size = str(size)
    if size not in SCORE_TYPES:
        raise ValueError("Score type must be 'entities', 3, 4, or 5")
    first = read_json(first) if isinstance(first, (str, Path)) else first
    second = read_json(second) if isinstance(second, (str, Path)) else second
    for report in (first, second):
        if report.get("evaluation_version") != EVALUATION_VERSION:
            raise ValueError("Re-run evaluate(): a report was produced by a different evaluation version")
    if first["settings"] != second["settings"]:
        raise ValueError(f"Reports use different settings: {first['settings']} vs {second['settings']}")

    def indexed(report):
        records = {a["article_id"]: a for a in report["articles"]}
        if len(records) != len(report["articles"]):
            raise ValueError("Duplicate articles in evaluation report")
        return records

    a, b = indexed(first), indexed(second)
    if not a or a.keys() != b.keys():
        raise ValueError("Reports must contain the same nonempty set of articles")
    keys = sorted(a)
    for key in keys:
        if (a[key]["gold_fingerprint"], a[key]["alias_fingerprint"]) != (
                b[key]["gold_fingerprint"], b[key]["alias_fingerprint"]):
            raise ValueError(f"Reports use different annotations or alias rules for {key}")
        if a[key]["scores"][size]["gold"] != b[key]["scores"][size]["gold"]:
            raise ValueError(f"Reports disagree on gold counts for {key}")
    pairs = [(a[k]["scores"][size], b[k]["scores"][size]) for k in keys]

    def f1(scores, aggregation):
        if aggregation == "macro":
            return sum(s["f1"] for s in scores) / len(scores)
        return metrics(sum(s["tp"] for s in scores), sum(s["predicted"] for s in scores),
                       sum(s["gold"] for s in scores))["f1"]

    rng = random.Random(seed)
    output = {}
    for aggregation in ["micro", "macro"]:
        observed = f1([p[1] for p in pairs], aggregation) - f1([p[0] for p in pairs], aggregation)
        differences, extreme = [], 0
        for _ in range(samples):
            draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            differences.append(f1([p[1] for p in draw], aggregation) - f1([p[0] for p in draw], aggregation))
            swapped = [p[::-1] if rng.random() < 0.5 else p for p in pairs]
            difference = f1([p[1] for p in swapped], aggregation) - f1([p[0] for p in swapped], aggregation)
            extreme += abs(difference) >= abs(observed) - 1e-12
        differences.sort()
        output[aggregation] = {
            "difference_second_minus_first": observed,
            "bootstrap_95_ci": [differences[int(0.025 * (samples - 1))], differences[int(0.975 * (samples - 1))]],
            "permutation_p_two_sided": (extreme + 1) / (samples + 1)}
    return {"tuple_size": size, "articles": len(pairs), "samples": samples, "seed": seed, "results": output}


def evaluate_revision(gt_path, before_path, after_path, output_path=None, split="val",
                      samples=10000, seed=42, **settings):
    """Paired validation ablation of a revision step: predictions before vs. after revision.

    `settings` are passed to evaluate() (aliases_path, include_context, limit, gold_stages,
    exclude_controls, match_missing, id_map, invalid_gold, name_matching).
    """
    if split != "val":
        raise ValueError("Choose whether to use revision on split=val, then freeze the choice for test")
    before = evaluate(gt_path, before_path, split=split, **settings)
    after = evaluate(gt_path, after_path, split=split, **settings)
    comparisons = {size: compare_reports(before, after, size, samples, seed) for size in SCORE_TYPES}
    changes = [{"article_id": x["article_id"],
                "f1_changes": {s: y["scores"][s]["f1"] - x["scores"][s]["f1"] for s in SCORE_TYPES}}
               for x, y in zip(before["articles"], after["articles"])]
    result = {"split": split, "before": before, "after": after, "comparisons": comparisons,
              "article_changes": changes,
              "interpretation": "Positive differences favor revision; negative differences favor the original. "
              "Select one global policy per pipeline/model on validation data; freeze it before testing."}
    if output_path:
        write_json(output_path, result)
    return result


def evaluate_classification(predictions, annotations):
    """Score human multi-label annotations keyed by article/source ID."""
    if not annotations or predictions.keys() != annotations.keys():
        raise ValueError("Classification evaluation requires exactly the same source IDs")
    counts = defaultdict(lambda: [0, 0, 0])
    from .common import schema
    allowed = set(schema("classification")["properties"]["labels"]["items"]["enum"])
    for source_id, labels in annotations.items():
        predicted_labels = predictions[source_id]
        if not isinstance(predicted_labels, list) or not isinstance(labels, list):
            raise ValueError("Classification labels must be lists")
        predicted, expected = set(predicted_labels), set(labels)
        if not predicted or not expected or not (predicted | expected).issubset(allowed):
            raise ValueError("Classification labels must be nonempty and use the defined vocabulary")
        for label in predicted | expected:
            counts[label][0] += label in predicted & expected
            counts[label][1] += label in predicted
            counts[label][2] += label in expected
    per_label = {label: metrics(*values) for label, values in counts.items()}
    micro = metrics(*(sum(values[i] for values in counts.values()) for i in range(3)))
    macro = {key: sum(m[key] for m in per_label.values()) / len(per_label)
             for key in ["precision", "recall", "f1"]}
    return {"per_label": per_label, "micro": micro, "macro": macro}
