"""The extraction pipeline, as used by the notebooks (import semextract_revised.extraction as ex).

Configure the module settings below (ex.MODEL_NAME = ..., ex.STAGE_DIR = ...),
then run the stages in order:

    contexts = ex.general_pass(llm, processor, T, articles)        # shared article context
    labels = ex.classify(llm, processor, T, articles, contexts)    # relevance of every source
    papers = ex.run_paradigm(llm, processor, T, articles, contexts, labels, 'text')

Every extraction stage follows the same path:
select sources -> extract -> screen -> merge -> review against the originals -> save.
It returns papers, {article_id: paper}; paper['relations_after_revision'] holds the
final relations and paper['relations_before_revision'] those before the review.
Each stage also saves <stage>_papers/_requests/_metadata/_before_revision/
_after_revision.json in STAGE_DIR.

Model requests go through ex.generate(llm, processor, items, json_schema,
temperature, max_tokens). It uses vLLM by default; a notebook may replace it,
e.g. with an HTTP client, by assigning ex.generate = my_generate.
"""
from copy import deepcopy
from pathlib import Path
import json
import time
from uuid import uuid4

from jsonschema import validate

from .common import FORMAT_VERSION, digest, file_digest, prompt_fingerprint, read_json, recipe, schema, write_json
from .fusion import canonicalize, consolidate_candidates, merge_context, rule_fuse
from .inputs import (
    MODALITIES, SINGULAR, batches, build_prompt, check_modalities, condition_for, condition_suffix,
    context_text, default_modalities, extraction_task, is_figure, joint_batches, normalized_labels,
    paradigm_order, project_tables, source_chunks, sources, task_request, text_chunks,
)
from .late_fusion import fusion_input, reviewable_fusion, validate_fusion
from .revision import apply_revision, candidate_records, combine_updates, decision_provenance, revision_request
from .runtime import new_run_dir, vllm_generate
from .validation import clean_relations, screen_relations, validate_context

# --------------------------------------------------------------------------- settings

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
STAGE_DIR = None             # run directory for stage files; created under out/stages on first use if None
MAX_TOKENS = 8192            # output tokens per request
CONTEXT_MAX_TOKENS = 16384   # output tokens per context (general pass) request
CUT_OFF_RETRY_FACTOR = 2     # an answer cut off at its token limit is retried with this many times the limit
MAX_ENGINE_IMAGES = 8        # images per request; papers with more figures are split
VISUAL_TOKEN_BUDGET = 65536  # vLLM: visual tokens per request, shared by its images
ENABLE_THINKING = False      # vLLM chat template option
REVISE_TEXT = REVISE_TABLES = REVISE_IMAGES = REVISE_JOINT = True  # original-source review per modality
TEXT_DECIMAL_COMMA = False   # coefficient checks also read 0,25 as a number
FILTER_CONTENT = True        # run_paradigm: extract only from sources classified relevant
REUSE_STAGES = True          # run_stage: reuse a saved result whose inputs are unchanged

# Logs of this kernel; every stage also saves its share in <stage>_requests.json.
REQUEST_LOG = []        # one entry per request attempt: prompt, images, response, error
GENERATION_ERRORS = []  # error messages
RAW_COMPLETIONS = []    # raw model outputs appended by generate()

TEXT_INPUTS = ("entire", "filtered", "elementwise")
VERSIONS = ("before_revision", "after_revision")
STAGE_FILE_KINDS = ("metadata", "requests", "papers", "answers", "output", "responses", "seed",
                    "before_revision", "after_revision")
_USED_PREFIXES = set()


def settings(temperature=None):
    """The current settings, as saved in stage metadata and cache fingerprints."""
    reviews = dict(text=REVISE_TEXT, tables=REVISE_TABLES, images=REVISE_IMAGES, joint=REVISE_JOINT)
    return dict(
        model_name=MODEL_NAME, temperature=temperature, max_tokens=MAX_TOKENS,
        context_max_tokens=CONTEXT_MAX_TOKENS, cut_off_retry_factor=CUT_OFF_RETRY_FACTOR, max_images=MAX_ENGINE_IMAGES,
        visual_token_budget=VISUAL_TOKEN_BUDGET, enable_thinking=ENABLE_THINKING,
        review_modes=[mode for mode, enabled in reviews.items() if enabled],
        decimal_comma=TEXT_DECIMAL_COMMA, filter_content=FILTER_CONTENT, reuse_stages=REUSE_STAGES,
    )


# --------------------------------------------------------------------------- model requests

def generate(llm, processor, items, json_schema, temperature, max_tokens=None):
    """Answer each (prompt text, image paths) item: parsed JSON, plain text if json_schema is None, or None."""
    if not items:
        return []
    answers, completions = vllm_generate(
        llm, processor, items, json_schema, temperature, max_tokens or MAX_TOKENS,
        MODEL_NAME, ENABLE_THINKING, MAX_ENGINE_IMAGES, VISUAL_TOKEN_BUDGET)
    RAW_COMPLETIONS.extend(completions)
    return answers


def safe_generate(llm, processor, items, json_schema, temperature, max_tokens=None, check=None):
    """generate() with schema validation and logging; failed items are retried once, together.

    An answer cut off at the token limit (finish_reason 'length') is retried with
    CUT_OFF_RETRY_FACTOR times the limit. check(item_index, answer) may reject a
    schema-valid answer by raising; it is then retried as well.
    """
    if not items:
        return []
    max_tokens = max_tokens or MAX_TOKENS
    cut_off = set()

    def attempt(indices, budget):
        batch = [items[i] for i in indices]
        first_completion = len(RAW_COMPLETIONS)
        try:
            answers = generate(llm, processor, batch, json_schema, temperature, budget)
        except Exception as error:
            GENERATION_ERRORS.append(str(error))
            answers = []
        answers = (list(answers) + [None] * len(batch))[:len(batch)]
        completions = RAW_COMPLETIONS[first_completion:]
        output = []
        for position, (item_index, (text, images), answer) in enumerate(zip(indices, batch, answers)):
            raw = completions[position] if position < len(completions) else None
            entry = dict(prompt=text, images=images, response=deepcopy(answer), error=None, raw_completion=raw,
                         schema_fingerprint=digest(json_schema), temperature=temperature, max_tokens=budget)
            if isinstance(raw, dict) and raw.get("finish_reason") == "length":
                cut_off.add(item_index)
            try:
                if answer is None:
                    raise ValueError("No model response")
                if json_schema is not None:
                    validate(answer, json_schema)
                elif not isinstance(answer, str):
                    raise ValueError("Expected a plain-text answer")
                if check is not None:
                    check(item_index, answer)
                output.append(answer)
            except Exception as error:
                entry["error"] = str(error)
                GENERATION_ERRORS.append(str(error))
                output.append(None)
            REQUEST_LOG.append(entry)
        return output

    answers = attempt(range(len(items)), max_tokens)
    failed = [i for i, answer in enumerate(answers) if answer is None]
    other_failures = [i for i in failed if i not in cut_off]
    cut_off_failures = [i for i in failed if i in cut_off]
    for indices, budget in ((other_failures, max_tokens), (cut_off_failures, max_tokens * CUT_OFF_RETRY_FACTOR)):
        if indices:
            for index, answer in zip(indices, attempt(indices, budget)):
                answers[index] = answer
    return answers


def generate_all(llm, processor, temperature, requests, max_tokens=None):
    """Answer (item, json_schema) requests with one safe_generate call per schema, in request order."""
    groups = {}
    for index, (_, json_schema) in enumerate(requests):
        groups.setdefault(digest(json_schema), (json_schema, []))[1].append(index)
    answers = [None] * len(requests)
    for json_schema, indices in groups.values():
        results = safe_generate(llm, processor, [requests[i][0] for i in indices], json_schema, temperature, max_tokens)
        for index, answer in zip(indices, results):
            answers[index] = answer
    return answers


# --------------------------------------------------------------------------- stage files

def new_stage(name, contexts=None):
    """Start a stage: reserve its file prefix in STAGE_DIR (a rerun gets name_attempt2, ...)."""
    global STAGE_DIR
    if STAGE_DIR is None:
        STAGE_DIR = new_run_dir("out/stages")
    root = Path(STAGE_DIR)
    root.mkdir(parents=True, exist_ok=True)
    prefix, number = name, 2
    while (str(root), prefix) in _USED_PREFIXES or any(
            (root / f"{prefix}_{kind}.json").exists() for kind in STAGE_FILE_KINDS):
        prefix = f"{name}_attempt{number}"
        number += 1
    _USED_PREFIXES.add((str(root), prefix))
    return dict(name=name, root=root, prefix=prefix, files=str(root / (prefix + "_*.json")),
                started=time.perf_counter(), first_request=len(REQUEST_LOG),
                first_error=len(GENERATION_ERRORS), context_fingerprint=digest(contexts))


def save_stage_file(stage, kind, value):
    write_json(stage["root"] / f"{stage['prefix']}_{kind}.json", value)


def finish_stage(stage, contexts, papers, temperature, metadata=None, seed=None):
    """Save the stage's papers, both relation versions, requests and metadata; return papers."""
    if digest(contexts) != stage["context_fingerprint"]:
        raise AssertionError("Shared context was mutated")
    for version in VERSIONS:
        save_stage_file(stage, version, {aid: paper["relations_" + version] for aid, paper in papers.items()})
    if seed is not None:
        save_stage_file(stage, "seed", seed)
    requests = REQUEST_LOG[stage["first_request"]:]
    save_stage_file(stage, "papers", papers)
    save_stage_file(stage, "requests", requests)
    save_stage_file(stage, "metadata", dict(
        elapsed_seconds=time.perf_counter() - stage["started"],
        prompt_fingerprint=prompt_fingerprint(), context_fingerprint=stage["context_fingerprint"],
        config=settings(temperature), errors=GENERATION_ERRORS[stage["first_error"]:],
        request_count=len(requests), stage=stage["prefix"], stage_files=stage["files"], **(metadata or {})))
    return papers


def run_stage(name, path, fn, timings, article_ids, *, cache_inputs=None):
    """Return fn(), saved to path; reuse the saved result if cache_inputs and settings are unchanged.

    Without cache_inputs nothing is reused. A replaced result is kept as <name>_previous_<id>.json.
    """
    path = Path(path)
    cache_path = path.with_suffix(path.suffix + ".cache.json")
    fingerprint = digest(dict(format=FORMAT_VERSION, prompts=prompt_fingerprint(), config=settings(),
                              inputs=cache_inputs, articles=list(article_ids)))
    if REUSE_STAGES and cache_inputs is not None and path.exists() and cache_path.exists():
        cache = read_json(cache_path)
        if cache.get("fingerprint") == fingerprint and cache.get("result") == file_digest(path):
            return read_json(path)
    start = time.perf_counter()
    result = fn()
    if path.exists() and read_json(path) != result:
        write_json(path.with_name(path.stem + "_previous_" + uuid4().hex[:8] + path.suffix), read_json(path))
    write_json(path, result)
    write_json(cache_path, dict(fingerprint=fingerprint, result=file_digest(path)))
    timings[name] = dict(seconds=time.perf_counter() - start, n_articles=len(article_ids))
    return result


# --------------------------------------------------------------------------- context and classification

def general_pass(llm, processor, temperature, articles):
    """The shared context of each article, from one request with its entire text (no images)."""
    stage = new_stage("context")
    requests, allowed = [], []
    for article in articles:
        chunks = text_chunks(article, "with_html_tables")
        if not chunks and article.get("full_text"):
            chunks = [dict(id="article_text", kind="paragraph", page=0, text=article["full_text"],
                           caption="", image=None)]
        request, _ = task_request("context", None, chunks, "text", text_condition="with_html_tables",
                                  max_images=MAX_ENGINE_IMAGES)
        requests.append(request)
        allowed.append({c["id"] for c in [*article.get("chunks", []), *chunks]})

    def usable(index, answer):
        """An ambiguous context (one label for two constructs/models/groups) or one citing
        unknown sources is rejected, so its request is retried like a failed one."""
        validate_context(merge_context([answer]), allowed[index])

    answers = safe_generate(llm, processor, requests, schema("context"), temperature, CONTEXT_MAX_TOKENS,
                            check=usable)
    by_article = {article["id"]: answer for article, answer in zip(articles, answers)}
    save_stage_file(stage, "answers", by_article)
    save_stage_file(stage, "requests", REQUEST_LOG[stage["first_request"]:])
    failed = [aid for aid, answer in by_article.items() if answer is None]
    if failed:
        raise ValueError(f"Context generation failed twice for {failed} (errors in {stage['prefix']}_requests.json); "
                         "do not run branches with partial context")
    contexts = {article["id"]: merge_context([answer]) for article, answer in zip(articles, answers)}
    save_stage_file(stage, "output", contexts)
    return contexts


def classify(llm, processor, temperature, articles, contexts, labels=None, mode="joint",
             text_condition="with_html_tables", table_representation="html"):
    """Labels {article_id: {source_id: [label, ...]}} for every source not yet in labels.

    mode 'joint' classifies all sources (tables by their HTML), 'text' the text
    sources, 'tables' the tables, 'images' only figures. A failed answer counts as tested_model.
    """
    if contexts is None:
        raise ValueError("Generate/load shared context before classification")
    output = deepcopy(labels or {})
    jobs = []
    for article in articles:
        aid = article["id"]
        output.setdefault(aid, {})
        if mode == "text":
            chunks = text_chunks(article, text_condition)
        elif mode == "tables":
            chunks = project_tables([c for c in article["chunks"] if c["kind"] == "table"], table_representation)
        else:
            chunks = [dict(c, image=None) if c["kind"] == "table" else c for c in article["chunks"]]
        for chunk in chunks:
            if chunk["id"] in output[aid] or (mode == "images" and not is_figure(chunk)):
                continue
            task = ("classify_table" if chunk["kind"] == "table"
                    else "classify_image" if chunk.get("image") else "classify_text")
            request = task_request(task, contexts[aid], [chunk], mode, text_condition=text_condition,
                                   max_images=MAX_ENGINE_IMAGES)
            jobs.append((aid, chunk["id"], request))
    answers = generate_all(llm, processor, temperature, [request for _, _, request in jobs])
    for (aid, source_id, _), answer in zip(jobs, answers):
        output[aid][source_id] = normalized_labels(answer) if answer is not None else ["tested_model"]
    return output


# --------------------------------------------------------------------------- extraction

def check_options(text_input, text_condition, table_representation):
    if text_input not in TEXT_INPUTS:
        raise ValueError(f"text_input must be one of {TEXT_INPUTS}")
    if text_condition not in ("with_html_tables", "without_html_tables"):
        raise ValueError("text_condition must be with_html_tables or without_html_tables")
    if table_representation not in ("html", "image"):
        raise ValueError("table_representation must be html or image")


def plan_papers(llm, processor, temperature, articles, contexts, mode, labels, text_input, text_condition,
                table_representation="html", include_figures=True, image_labels=None, baseline=False):
    """One plan per article: its request batches, classified first where the input mode needs labels.

    A plan holds article, context, batches, relations (current before-review
    relations) and record (details saved with the paper).
    """
    wants_text = not baseline and (text_input == "filtered" or labels is not None)
    wants_images = not baseline and (mode == "images" or (mode == "joint" and include_figures))
    classified = {}
    if wants_text or wants_images:
        classified = classify(llm, processor, temperature, articles, contexts,
                              labels=labels if labels is not None else image_labels,
                              mode=mode if wants_text else "images", text_condition=text_condition,
                              table_representation=table_representation)
    plans = []
    for article in articles:
        aid = article["id"]
        validate_context(contexts[aid], {chunk["id"] for chunk in article["chunks"]})
        paper_batches = batches(article, mode, classified.get(aid) if wants_text else None, text_input,
                                text_condition, table_representation, include_figures, baseline=baseline,
                                image_labels=classified.get(aid) if wants_images else None,
                                max_images=MAX_ENGINE_IMAGES)
        plans.append(dict(article=article, context=contexts[aid], batches=paper_batches, relations=[], record={}))
    return plans


def extract(llm, processor, temperature, articles, contexts, mode, labels=None, text_input=None,
            text_condition="with_html_tables", table_representation="html", include_figures=True,
            image_labels=None, stage_name=None, baseline=False):
    """Extract one modality (text, tables, images) or all together (joint); return papers.

    labels select the relevant sources; text_input 'entire' ignores them for text.
    Figures are always gated by labels (image_labels, else labels, else classified now).
    """
    if mode not in {*MODALITIES, "joint"}:
        raise ValueError("Unknown extraction mode")
    text_input = text_input or ("filtered" if labels is not None else "entire")
    if baseline:  # unfiltered: everything, with HTML tables
        text_input, text_condition, labels = "entire", "with_html_tables", None
    check_options(text_input, text_condition, table_representation)
    stage = new_stage(stage_name or mode, contexts)
    plans = plan_papers(llm, processor, temperature, articles, contexts, mode, labels, text_input, text_condition,
                        table_representation, include_figures, image_labels, baseline)

    jobs = []
    for plan in plans:
        plan["record"].update(relations_concatenated=[], extractions=[])
        for index, batch in enumerate(plan["batches"]):
            task = extraction_task(mode, batch, text_condition, baseline)
            request = task_request(task, plan["context"], batch, mode, "", text_condition, MAX_ENGINE_IMAGES)
            jobs.append((plan, index, task, request))
    answers = generate_all(llm, processor, temperature, [request for *_, request in jobs])
    for (plan, index, task, _), answer in zip(jobs, answers):
        if answer is None:
            status, error, candidates, screening = "failed", "No valid extraction response", [], []
            GENERATION_ERRORS.append(error)
        else:
            status, error = "completed", None
            kept, screening = screen_relations(answer)
            candidates = canonicalize(kept, plan["context"])
        plan["record"]["relations_concatenated"].extend(candidates)
        plan["record"]["extractions"].append(dict(
            batch=index, recipe=task, source_ids=[c["id"] for c in plan["batches"][index]],
            status=status, error=error, raw_response=answer, relations=candidates, screening=screening))
    for plan in plans:
        plan["relations"] = rule_fuse(plan["record"]["relations_concatenated"], plan["context"])
        plan["record"].update(
            input_mode=text_input, text_condition=text_condition, modality=mode,
            include_figures=include_figures if mode == "joint" else None,
            table_representation=table_representation if mode == "tables" else None)
        if mode == "tables":  # differs from table_representation where a table lacked that form
            plan["record"]["tables_shown_as"] = {batch[0]["id"]: batch[0]["shown_as"] for batch in plan["batches"]}
    return review_and_save(llm, processor, temperature, stage, plans, contexts, mode, text_condition,
                           metadata=dict(text_condition=text_condition, input_mode=text_input))


# --------------------------------------------------------------------------- review

def new_audit(plan, stage_name, index, current_sources):
    return dict(article_id=plan["article"]["id"], stage=stage_name, batch_id=str(index),
                context_fingerprint=digest(plan["context"]), source_ids=sorted({s.id for s in current_sources}),
                status="pending", reason=None, before=deepcopy(plan["relations"]), review=None,
                after=deepcopy(plan["relations"]))


def apply_decision(plan, answer, current_sources, audit):
    """Apply a review/update answer to the plan's relations, recording the outcome in audit.

    Return the field provenance of the decisions, or None if the answer was rejected.
    """
    try:
        if answer is None:
            raise ValueError("Invalid or missing review response")
        revised = apply_revision(plan["relations"], answer, current_sources, plan["context"])
        # Source IDs express a set. Normalize only after validation, preserving
        # their order and leaving the raw model response in REQUEST_LOG intact.
        answer = deepcopy(answer)
        for decision in answer["decisions"]:
            decision["source_ids"] = list(dict.fromkeys(decision["source_ids"]))
        audit.update(status="applied", review=answer, after=revised)
        provenance = decision_provenance(plan["relations"], answer)
    except Exception as error:
        GENERATION_ERRORS.append(str(error))
        audit.update(status="failed", reason=str(error), review=None, after=deepcopy(plan["relations"]))
        provenance = None
    validate(audit, schema("review_audit"))
    return provenance


def review_and_save(llm, processor, temperature, stage, plans, contexts, mode, text_condition,
                    metadata=None, seed=None, review=None, review_stage=None, intermediate=False):
    """Review every batch against its original sources, then save the stage; return papers.

    All reviews of a paper decide on the same before-review relations; their
    decisions are combined afterwards (combine_updates), never applied in sequence.
    """
    reviews = dict(text=REVISE_TEXT, tables=REVISE_TABLES, images=REVISE_IMAGES, joint=REVISE_JOINT)
    enabled = reviews[mode] if review is None else review
    review_stage = review_stage or stage["prefix"]
    states, jobs = {}, []
    for plan in plans:
        state = states[plan["article"]["id"]] = dict(audits=[], review_batches=[], updates=[])
        for index, batch in enumerate(plan["batches"] or [[]]):
            current_sources = sources(batch, mode, text_condition)
            audit = new_audit(plan, review_stage, index, current_sources)
            detail = dict(batch_id=str(index), request_metadata=None, field_provenance=None)
            if not enabled or not batch:
                audit.update(status="skipped", reason="Review disabled" if not enabled else "No sources available")
            else:
                try:
                    task, extra, detail["request_metadata"] = revision_request(
                        plan["relations"], current_sources, context_text(plan["context"]), mode,
                        TEXT_DECIMAL_COMMA, text_condition, intermediate)
                    spec = recipe(task)
                    item = build_prompt(spec["instructions"], spec["output_schema"], extra,
                                        source_chunks(current_sources), "tables" if mode == "tables" else "joint",
                                        False, text_condition, MAX_ENGINE_IMAGES)
                    jobs.append((plan, current_sources, audit, detail, (item, spec["output_schema"])))
                except Exception as error:
                    GENERATION_ERRORS.append(str(error))
                    audit.update(status="failed", reason=str(error))
            state["audits"].append(audit)
            state["review_batches"].append(detail)
    answers = generate_all(llm, processor, temperature, [request for *_, request in jobs])
    for (plan, current_sources, audit, detail, _), answer in zip(jobs, answers):
        provenance = apply_decision(plan, answer, current_sources, audit)
        if provenance is not None:
            detail["field_provenance"] = provenance
            states[plan["article"]["id"]]["updates"].append(audit["review"])

    papers = {}
    for plan in plans:
        state = states[plan["article"]["id"]]
        for audit in state["audits"]:
            validate(audit, schema("review_audit"))
        updated, conflicts = combine_updates(plan["relations"], state["updates"], plan["context"])
        papers[plan["article"]["id"]] = dict(
            plan["record"], context_fingerprint=digest(plan["context"]), stage_files=stage["files"],
            relations_before_revision=plan["relations"], relations_after_revision=rule_fuse(updated, plan["context"]),
            audits=state["audits"], review_batches=state["review_batches"], review_conflicts=conflicts)
    return finish_stage(stage, contexts, papers, temperature, metadata, seed)


# --------------------------------------------------------------------------- fusion

def intermediate_fusion(llm, processor, temperature, articles, contexts, seeds, mode, labels=None,
                        text_input=None, text_condition="with_html_tables", table_representation="html",
                        image_labels=None, stage_name=None, prior_modes=None, input_versions=None):
    """Update fixed prior relations (seeds, {article_id: relations}) with the sources of one more modality.

    Every batch of the new modality decides on the same prior; the decisions are combined afterwards.
    """
    if mode not in MODALITIES:
        raise ValueError("Intermediate target must be text, tables or images")
    text_input = text_input or ("filtered" if labels is not None else "entire")
    check_options(text_input, text_condition, table_representation)
    prior_modes = prior_modes or (["images"] if mode == "text" else ["text"])
    name = stage_name or "_then_".join(SINGULAR[m] for m in [*prior_modes, mode])
    stage = new_stage(name, contexts)
    seed_source = str(seeds) if isinstance(seeds, (str, Path)) else "provided_results"
    seeds = read_json(seeds) if isinstance(seeds, (str, Path)) else deepcopy(seeds)
    input_versions = input_versions or {"seed": "caller_supplied"}
    plans = plan_papers(llm, processor, temperature, articles, contexts, mode, labels, text_input, text_condition,
                        table_representation, image_labels=image_labels)
    jobs, updates = [], {plan["article"]["id"]: [] for plan in plans}
    for plan in plans:
        plan["relations"] = canonicalize(seeds.get(plan["article"]["id"], []), plan["context"])
        plan["record"].update(
            seed_source=seed_source, seed_relations=deepcopy(plan["relations"]),
            source_mode=prior_modes[0] if len(prior_modes) == 1 else list(prior_modes),
            target_mode=mode, conflict_policy="second_modality_evidence_wins", text_condition=text_condition,
            table_representation=table_representation if mode == "tables" else None,
            input_versions=input_versions, updates=[], update_field_provenance=[])
        prior_text = ("PRIOR EXTRACTION (fixed first-modality snapshot):\n"
                      + json.dumps(candidate_records(plan["relations"]), ensure_ascii=False)
                      + "\nPRIOR MODALITIES: " + ", ".join(prior_modes) + "; CURRENT SECOND MODALITY: " + mode)
        for index, batch in enumerate(plan["batches"]):
            if mode == "tables" or (len(batch) == 1 and batch[0]["kind"] == "table"):
                task = "intermediate_table"
            elif mode == "images":
                task = "intermediate_image"
            else:
                task = "intermediate_text_" + condition_suffix(text_condition)
            current_sources = sources(batch, mode, text_condition)
            audit = new_audit(plan, name + "_update", index, current_sources)
            request = task_request(task, plan["context"], batch, mode, prior_text, text_condition, MAX_ENGINE_IMAGES)
            jobs.append((plan, current_sources, audit, request))
    answers = generate_all(llm, processor, temperature, [request for *_, request in jobs])
    for (plan, current_sources, audit, _), answer in zip(jobs, answers):
        provenance = apply_decision(plan, answer, current_sources, audit)
        if provenance is not None:
            updates[plan["article"]["id"]].append(audit["review"])
            plan["record"]["update_field_provenance"].append(dict(batch_id=audit["batch_id"],
                                                                  field_provenance=provenance))
        plan["record"]["updates"].append(audit)
    for plan in plans:
        updated, conflicts = combine_updates(plan["relations"], updates[plan["article"]["id"]], plan["context"])
        plan["relations"] = rule_fuse(updated, plan["context"])
        plan["record"]["update_conflicts"] = conflicts
    return review_and_save(llm, processor, temperature, stage, plans, contexts, mode, text_condition,
                           metadata=dict(input_versions=input_versions), seed=seeds, intermediate=True)


def late_fusion(llm, processor, temperature, articles, contexts, relations, use_llm=False,
                text_condition="with_html_tables", stage_name=None, input_versions=None, review=None):
    """Fuse independent predictions, relations = {modality: {article_id: relations}}; return papers.

    Rule-based fusion groups compatible candidates deterministically. With
    use_llm, the model's grouping is used as answered; a missing answer falls
    back to the rule-based grouping. The result is reviewed against all sources.
    """
    modalities = [mode for mode in MODALITIES if relations.get(mode) is not None]
    if set(relations) - set(MODALITIES) or not modalities:
        raise ValueError(f"Late fusion takes relations keyed by {MODALITIES}")
    review_condition = "with_html_tables" if "tables" in modalities else text_condition
    name = stage_name or ("late_llm" if use_llm else "late_rule")
    stage = new_stage(name, contexts)
    input_versions = input_versions or {"upstream": "caller_supplied"}
    plans, jobs, owners = [], [], []
    for article in articles:
        aid = article["id"]
        records = {mode: clean_relations(relations[mode].get(aid, [])) for mode in modalities}
        payload = fusion_input(contexts[aid], records)
        originals = {f"{mode}_{i}": record for mode in modalities for i, record in enumerate(records[mode], 1)}
        review_batches = joint_batches(article, article["chunks"], review_condition, "images" in modalities,
                                       MAX_ENGINE_IMAGES)
        plan = dict(article=article, context=contexts[aid], batches=review_batches, relations=[], record=dict(
            modalities=modalities, review_text_condition=review_condition, fusion_input=payload,
            candidate_provenance=originals, naming_fallback=None, input_versions=input_versions))
        plans.append(plan)
        owners.append((aid, payload))
        if use_llm and payload["candidates"]:
            extra = ("FUSED MODALITIES: " + ", ".join(modalities) + "\nPREDICTIONS:\n"
                     + json.dumps(payload["candidates"], ensure_ascii=False))
            jobs.append((plan, task_request("late", plan["context"], [], "text", extra, text_condition,
                                            MAX_ENGINE_IMAGES)))
    responses = generate_all(llm, processor, temperature, [request for _, request in jobs])
    answers = {plan["article"]["id"]: answer for (plan, _), answer in zip(jobs, responses)}
    for plan in plans:
        aid, record = plan["article"]["id"], plan["record"]
        fused = answers.get(aid)
        if fused is None:  # rule-based fusion, or the fallback for a missing LLM answer
            if aid in answers:
                record["naming_fallback"] = "Missing or unparsable fusion output after retry"
                GENERATION_ERRORS.append(record["naming_fallback"])
            fused = consolidate_candidates(record["fusion_input"]["candidates"], plan["context"])
            validate_fusion(fused, record["fusion_input"])
        record["fusion_records"] = fused
        plan["relations"] = reviewable_fusion(fused, record["candidate_provenance"], plan["context"])
    save_stage_file(stage, "responses", dict(owners=owners, answers=[answers.get(p["article"]["id"]) for p in plans],
                                             output_schema=schema("late_fusion"), empty_article_ids=[]))
    return review_and_save(llm, processor, temperature, stage, plans, contexts, "joint", review_condition,
                           metadata=dict(modalities=modalities, input_versions=input_versions),
                           review=review, review_stage=name + "_original_source_review")


# --------------------------------------------------------------------------- experiments

def run_paradigm(llm, processor, temperature, articles, contexts, labels, paradigm, text_input=None,
                 table_setting="in_text", table_representation="html", upstream_version="after_revision",
                 modalities=None, branches=None, stage_suffix=""):
    """Run one experiment and return its papers.

    paradigm: 'text', 'tables', 'images' (one modality), 'baseline' (everything,
    unfiltered), 'early' (one joint request), an intermediate order such as
    'text_then_image' or 'text_then_tables_then_image', or 'late_rule'/'late_llm'.
    table_setting 'separate' is required for a table modality.
    branches = {modality: papers} reuses earlier results instead of extracting that
    modality again; fusion then reads their upstream_version relations.
    stage_suffix is appended to every stage name, e.g. '_filtered_in_text'.
    """
    text_condition = condition_for(table_setting)
    if upstream_version not in VERSIONS:
        raise ValueError(f"upstream_version must be one of {VERSIONS}")
    selected_labels = labels if FILTER_CONTENT and text_input != "entire" else None
    input_mode = text_input or ("filtered" if selected_labels is not None else "entire")
    check_options(input_mode, text_condition, table_representation)
    common = dict(text_input=input_mode, text_condition=text_condition, table_representation=table_representation)
    stage = paradigm + stage_suffix

    if paradigm in MODALITIES:
        return extract(llm, processor, temperature, articles, contexts, paradigm, selected_labels,
                       image_labels=labels, stage_name=stage, **common)
    if paradigm == "baseline":
        return extract(llm, processor, temperature, articles, contexts, "joint", baseline=True,
                       stage_name=stage, **common)
    if paradigm == "early":
        chosen = check_modalities(modalities or default_modalities(table_setting), table_setting)
        uses_tables = table_setting == "in_text" or "tables" in chosen
        common["text_condition"] = "with_html_tables" if uses_tables else "without_html_tables"
        # Early fusion classifies the text by default, too.
        if FILTER_CONTENT and text_input != "entire":
            selected_labels = classify(llm, processor, temperature, articles, contexts, selected_labels,
                                       text_condition=common["text_condition"],
                                       table_representation=table_representation)
            common["text_input"] = text_input or "filtered"
        return extract(llm, processor, temperature, articles, contexts, "joint", selected_labels,
                       image_labels=labels, include_figures="images" in chosen, stage_name=stage, **common)

    versions = {}  # where each fused input comes from, saved with the result

    def branch(mode):
        """{article_id: relations} of one modality: reused from branches, or extracted now."""
        if branches is not None and mode in branches:
            papers = branches[mode]
            missing = [article["id"] for article in articles if article["id"] not in papers]
            if missing:
                raise ValueError(f"Reused {mode} results are missing articles: {missing}")
            versions[mode] = dict(version=upstream_version, reused=True, stage_files=sorted(
                {str(papers[article["id"]].get("stage_files")) for article in articles}))
            return {article["id"]: deepcopy(papers[article["id"]]["relations_" + upstream_version])
                    for article in articles}
        papers = extract(llm, processor, temperature, articles, contexts, mode, selected_labels, image_labels=labels,
                         stage_name=f"{paradigm}_{mode}{stage_suffix}", **common)
        versions[mode] = dict(version=upstream_version, stage_files=stage_files(papers))
        return relations_of(papers, upstream_version)

    if paradigm in ("late_rule", "late_llm"):
        chosen = check_modalities(modalities or default_modalities(table_setting), table_setting)
        relations = {mode: branch(mode) for mode in chosen}
        return late_fusion(llm, processor, temperature, articles, contexts, relations,
                           use_llm=paradigm == "late_llm", text_condition=text_condition,
                           stage_name=stage, input_versions=versions)

    order = check_modalities(paradigm_order(paradigm), table_setting)
    if modalities is not None and set(modalities) != set(order):
        raise ValueError(f"{paradigm} conflicts with modalities={modalities}")
    seeds = branch(order[0])
    for position, mode in enumerate(order[1:], 1):
        step = "_then_".join(SINGULAR[m] for m in order[:position + 1]) + stage_suffix
        papers = intermediate_fusion(llm, processor, temperature, articles, contexts, seeds, mode, selected_labels,
                                     image_labels=labels, stage_name=step, prior_modes=list(order[:position]),
                                     input_versions=dict(versions), **common)
        versions[step] = dict(version=upstream_version, stage_files=stage_files(papers))
        seeds = relations_of(papers, upstream_version)
    for paper in papers.values():
        paper["order"] = list(order)
    return papers


def relations_of(papers, version="after_revision"):
    """{article_id: relations} of one version ('before_revision' or 'after_revision')."""
    return {aid: deepcopy(paper["relations_" + version]) for aid, paper in papers.items()}


def stage_files(papers):
    return next((paper["stage_files"] for paper in papers.values()), None)
