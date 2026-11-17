"""Manifest-driven SEM extraction with frozen context and auditable revisions."""
import json
import re
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from jsonschema import validate, ValidationError

from .common import (FORMAT_VERSION, LABEL_ALIASES, digest, file_digest, prompt, schema, recipe,
                     prompt_fingerprint, read_json, write_json, strict_json)
from .documents import Source, convert
from .fusion import (normalize_context, merge_context, canonicalize, rule_fuse,
                     consolidate_candidates, merge_provenance, unique)
from .late_fusion import fusion_input, validate_fusion
from .revision import (revision_request, apply_revision, combine_updates,
                       candidate_records, check_relations, decision_provenance)
from .utils import clean_relations
from .validation import validate_context, validate_relation

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR = Path("data")
OUTDIR = Path("out")  # runs go to OUTDIR/stages/<YYYY-MM-DD_hh-mm>/, flat JSON files only
MAX_TOKENS = 8192
MAX_CHARS = 18000
MAX_IMAGES = 1
MAX_ENGINE_IMAGES = 8  # images per request for baseline, early fusion and joint reviews; more are split
VISUAL_TOKEN_BUDGET = 65536  # total visual tokens per request; per-image resolution shrinks with more images
ENABLE_THINKING = False
REVISE_IMAGES = True
REVISE_TEXT = True
REVISE_TABLES = True
REVISE_JOINT = True
TEXT_DECIMAL_COMMA = False
TEXT_CONDITION = "with_html_tables"
FILTER_CONTENT = True
REUSE_STAGES = True
STAGE_DIR = None  # run directory; None creates OUTDIR/stages/<YYYY-MM-DD_hh-mm> on first use
_RUN_DIR = None
_USED_PREFIXES = set()
CONTEXT_MAX_TOKENS = 16384
GENERATION_ERRORS = []
REQUEST_LOG = []
RAW_COMPLETIONS = []
LAST_STAGE_METADATA = {}
ACTIVE_STAGE_DIRS = {}
MODALITIES = ("text", "tables", "images")
TABLE_SETTINGS = ("in_text", "separate")  # A: HTML tables inside text; B: tables as own modality
_SINGULAR = {"text": "text", "tables": "tables", "images": "image"}
_PARADIGM_MODALITY = {"text": "text", "table": "tables", "tables": "tables", "image": "images", "images": "images"}
SELECT_LABELS = {"tested_model", "hypothesized_model", "context"}
IMAGE_SELECT_LABELS = {"tested_model", "hypothesized_model"}

def vision_settings(model_name, n_images=1):
    """Maximum documented image budgets for the supported checkpoints.

    Gemma: https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html
    Qwen: the checkpoint's preprocessor_config.json (16,384 visual tokens).
    Pixel limits are areas, not side lengths; keep the original aspect ratio.
    """
    name = model_name.lower()
    if "gemma-4" in name or "gemma4" in name:
        return {"max_soft_tokens": 1120}
    # Qwen: one visual token per 32x32 (Qwen3) or 28x28 (Qwen2.5) pixels. With several
    # images, the per-image maximum is reduced so all fit into VISUAL_TOKEN_BUDGET.
    for key, patch, minimum in (("qwen3", 32, 65536), ("qwen2.5", 28, 3136), ("qwen2-vl", 28, 3136)):
        if key in name:
            tokens = min(16384, VISUAL_TOKEN_BUDGET // max(1, n_images))
            return {"min_pixels": minimum, "max_pixels": max(minimum, tokens * patch * patch)}
    return {}

def engine_settings(model_name):
    vision = vision_settings(model_name)
    settings = {
        # Allow room for full-resolution originals and the generated review graph.
        "max_model_len": 128000 if "max_pixels" in vision else 32768,
        "limit_mm_per_prompt": {"image": MAX_ENGINE_IMAGES},
        "mm_processor_kwargs": vision,
    }
    if "max_soft_tokens" in vision:
        settings["hf_overrides"] = {
            "vision_config": {"default_output_length": 1120},
            "vision_soft_tokens_per_image": 1120,
        }
    return settings

def generate(llm, processor, items, json_schema, temperature, max_tokens=None):
    if not items:
        return []
    from PIL import Image
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompts = []
    for text, images in items:
        if len(images) > MAX_ENGINE_IMAGES:
            raise ValueError(f"At most {MAX_ENGINE_IMAGES} original images per request")
        content = []
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        entry = {"prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING)}
        if images:
            loaded = []
            for path in images:
                with Image.open(path) as image:
                    loaded.append(image.convert("RGB"))
            entry["multi_modal_data"] = {"image": loaded}
            # Covers extraction, classification, fusion and image revision.
            entry["mm_processor_kwargs"] = vision_settings(MODEL_NAME, len(images))
        prompts.append(entry)

    budget = max_tokens or MAX_TOKENS
    # Schemas avoid keywords vLLM's grammar cannot compile (e.g. uniqueItems).
    structured = StructuredOutputsParams(json=json_schema) if json_schema is not None else None
    if temperature is None:
        params = SamplingParams(max_tokens=budget, structured_outputs=structured)
    else:
        params = SamplingParams(temperature=temperature, max_tokens=budget, structured_outputs=structured)

    results = []
    for output in llm.generate(prompts, params):
        completion = output.outputs[0]
        RAW_COMPLETIONS.append(dict(text=completion.text, finish_reason=getattr(completion, 'finish_reason', None)))
        if getattr(completion, "finish_reason", "stop") not in {"stop", None}:
            results.append(None)
            continue
        if json_schema is None:
            results.append(completion.text)
            continue
        try:
            results.append(strict_json(completion.text))
        except ValueError:
            results.append(None)
    return (results + [None] * len(items))[:len(items)]

def save_json(path, value):
    write_json(path, value)


def load_json(path):
    return read_json(path)


def context_version(contexts):
    return digest(contexts)[:12]


def run_name(now=None):
    """Run directory name: date and time of the start, e.g. 2026-11-04_14-05."""
    return (now or datetime.now()).strftime('%Y-%m-%d_%H-%M')


def new_run_dir(root=None, now=None):
    """Create out/stages/<YYYY-MM-DD_hh-mm>; a second run in the same minute gets _2, _3, ..."""
    base = Path(root) if root is not None else OUTDIR / 'stages'
    name = run_name(now);path = base / name;number = 2
    while path.exists():
        path = base / f'{name}_{number}';number += 1
    path.mkdir(parents=True)
    return path


def _root():
    global _RUN_DIR
    if STAGE_DIR is not None:
        return Path(STAGE_DIR)
    if _RUN_DIR is None:
        _RUN_DIR = new_run_dir()
    return _RUN_DIR


class StageFiles:
    """Flat JSON files of one stage in the run directory: <prefix>_<kind>.json."""

    def __init__(self, root, prefix):
        self.root, self.prefix = Path(root), prefix

    def __call__(self, kind):
        return self.root / f'{self.prefix}_{kind}.json'

    def __str__(self):
        return str(self.root / (self.prefix + '_*.json'))


def _begin_stage(stage, root=None):
    """Reserve a file prefix; repeating a stage in the same run gets <stage>_attempt2, ..."""
    root = Path(root) if root is not None else _root();root.mkdir(parents=True, exist_ok=True)
    prefix, number = stage, 2
    while (str(root), prefix) in _USED_PREFIXES or any(
            (root / f'{prefix}_{kind}.json').exists() for kind in ('metadata', 'requests', 'papers')):
        prefix = f'{stage}_attempt{number}';number += 1
    _USED_PREFIXES.add((str(root), prefix))
    files = StageFiles(root, prefix);ACTIVE_STAGE_DIRS[stage] = files
    return files


def stage_snapshot(stage, version, results, temperature):
    if stage not in ACTIVE_STAGE_DIRS:
        _begin_stage(stage)
    path = ACTIVE_STAGE_DIRS[stage](version)
    if path.exists() and read_json(path) != results:
        raise ValueError(f'Refusing to overwrite an immutable stage snapshot: {path}')
    save_json(path, results)
    return path


def save_result(path, result):
    """Preserve provenance and export comparable relation versions beside the result."""
    path = Path(path)
    save_json(path, result)
    for key in ('revision', 'text_revision', 'image_revision', 'table_revision', 'joint_revision'):
        review = result.get(key)
        if not isinstance(review, dict):
            continue
        for suffix, field in (('before_revision', 'relations_before_revision'),
                              ('after_revision', 'relations_after_revision')):
            if field in review:
                snapshot = {k: deepcopy(v) for k, v in result.items()
                            if k not in ('relations', 'revision', 'text_revision', 'image_revision',
                                         'table_revision', 'joint_revision', 'intermediate_fusion')}
                snapshot.update(relations=review[field], result_version=suffix, review_stage=key)
                save_json(path.with_name(f'{path.stem}_{key}_{suffix}{path.suffix}'), snapshot)


def run_stage(name, path, fn, timings, article_ids, *, cache_inputs=None):
    """Reuse only explicitly fingerprinted inputs, never article-ID-only caches."""
    path = Path(path); meta_path = path.with_suffix(path.suffix + '.cache.json')
    fingerprint = digest(dict(format=FORMAT_VERSION, prompts=prompt_fingerprint(), model=MODEL_NAME,
        inputs=cache_inputs, articles=list(article_ids), text_condition=TEXT_CONDITION,
        reviews=[REVISE_TEXT, REVISE_IMAGES, REVISE_TABLES, REVISE_JOINT],
        max_tokens=MAX_TOKENS, max_chars=MAX_CHARS, filter_content=FILTER_CONTENT,
        thinking=ENABLE_THINKING))
    if REUSE_STAGES and cache_inputs is not None and path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta.get('fingerprint') == fingerprint and meta.get('result') == file_digest(path):
            return read_json(path)
    start = time.perf_counter(); result = fn()
    # New configuration gets a separate recoverable previous output.
    if path.exists() and read_json(path) != result:
        save_json(path.with_name(path.stem + '_previous_' + uuid4().hex[:8] + path.suffix), read_json(path))
    save_json(path, result)
    save_json(meta_path, dict(fingerprint=fingerprint, result=file_digest(path)))
    timings[name] = dict(seconds=time.perf_counter() - start, n_articles=len(article_ids))
    return result


def safe_generate(llm, processor, items, json_schema, temperature, max_tokens=None):
    """Validate complete response shapes; retain raw attempts for failure analysis."""
    if not items:
        return []
    def call(requests):
        raw_start = len(RAW_COMPLETIONS)
        try:
            responses = list(generate(llm, processor, requests, json_schema, temperature, max_tokens))
        except Exception as error:
            GENERATION_ERRORS.append(str(error));responses = [None] * len(requests)
        responses = (responses + [None] * len(requests))[:len(requests)]
        output = []
        raw = RAW_COMPLETIONS[raw_start:]
        for index, (request, answer) in enumerate(zip(requests, responses)):
            entry = dict(prompt=request[0], images=request[1], response=deepcopy(answer), error=None,
                         raw_completion=raw[index] if index < len(raw) else None,
                         schema_fingerprint=digest(json_schema), temperature=temperature)
            try:
                if answer is None:
                    raise ValueError('No model response')
                if json_schema is not None:
                    validate(answer, json_schema)
                elif not isinstance(answer, str):
                    raise ValueError('Scaffold must be plain text')
                output.append(answer)
            except Exception as error:
                entry['error'] = str(error);GENERATION_ERRORS.append(str(error));output.append(None)
            REQUEST_LOG.append(entry)
        return output
    answers = call(items)
    failed = [i for i, answer in enumerate(answers) if answer is None]
    if failed:  # one batched retry, not one request per failure
        for i, answer in zip(failed, call([items[i] for i in failed])):
            answers[i] = answer
    return answers


def generate_all(llm, processor, temperature, requests, max_tokens=None):
    """Generate [(item, schema), ...] together: one batched vLLM call per distinct
    schema across all papers. Answers keep the input order."""
    answers, groups = [None] * len(requests), {}
    for index, (_, contract) in enumerate(requests):
        groups.setdefault(digest(contract), (contract, []))[1].append(index)
    for contract, indices in groups.values():
        output = safe_generate(llm, processor, [requests[i][0] for i in indices], contract, temperature, max_tokens)
        for i, answer in zip(indices, output):
            answers[i] = answer
    return answers


def _without_transcripts(text):
    text = re.sub(r'<details\b[^>]*>.*?</details>', '', text or '', flags=re.S | re.I)
    return re.sub(r'```\s*mermaid\b.*?```', '', text, flags=re.S | re.I)


def caption_of(item):
    return '\n'.join(v for v in (item.get('caption'), item.get('footnote')) if v)


def load_article(directory):
    article_id, doc = convert(directory)
    chunks = [dict(id=f"b{p['index']}", index=p['index'], kind='paragraph', page=p['page'],
                   caption='', text=p['text'], image=None, transcription='') for p in doc['paragraphs']]
    for group, kind in (('tables', 'table'), ('images', 'image')):
        for item in doc[group]:
            chunks.append(dict(id=f"b{item['index']}", index=item['index'], kind=kind, page=item['page'],
                               caption=caption_of(item), text=item.get('html', ''), image=item['image_path_abs'],
                               transcription=item.get('transcription', ''), source_kind=item.get('source_kind', kind)))
    chunks.sort(key=lambda c: c['index'])
    return dict(id=article_id, full_text=doc['full_text'], chunks=chunks)


def render(chunk, mode, allow_transcriptions=False, text_condition=None):
    condition = text_condition or TEXT_CONDITION
    lines = [f"[{chunk['id']}; page {chunk.get('page', 0)}; {chunk['kind']}]"]
    if chunk.get('caption'):
        lines.append(_without_transcripts(chunk['caption']))
    # The table modality always shows its projected body (HTML or image); in
    # text/joint inputs, HTML table bodies follow the text condition.
    if mode == 'tables' or (mode != 'images' and (chunk['kind'] != 'table' or condition == 'with_html_tables')):
        lines.append(_without_transcripts(chunk.get('text', '')))
    if allow_transcriptions and chunk.get('transcription'):
        lines.append('FIGURE TRANSCRIPTION (same source; converted evidence):\n' + chunk['transcription'])
    image = chunk.get('image') if mode != 'text' else None
    return '\n'.join(v for v in lines if v), image


def is_figure(chunk):
    """Any non-table image (figure, chart, equation or unlinked image) with a file.

    Tables belong to text or the table modality. Which images the image pipeline
    actually processes is decided by classification (see _image_labels).
    """
    return chunk['kind'] == 'image' and bool(chunk.get('image'))


def _image_labels(llm, processor, temperature, articles, contexts, mode, labels, input_mode, condition,
                  table_representation='html', baseline=False, include_figures=True, image_labels=None):
    """Return (text_labels, image_labels).

    Images are always classified, independent of FILTER_CONTENT: only non-table
    images labelled tested or hypothesized relations enter the image pipeline.
    Text is filtered only for filtered/scaffolded input or when labels are given.
    The unfiltered baseline classifies nothing.
    """
    wants_text = input_mode in ('filtered', 'scaffolded') or labels is not None
    wants_images = not baseline and (mode == 'images' or (mode == 'joint' and include_figures))
    if not (wants_text or wants_images):
        return None, None
    seed = labels if labels is not None else image_labels
    classified = classify(llm, processor, temperature, articles, seed, contexts,
                          mode if wants_text else 'images', condition, table_representation)
    return (classified if wants_text else None), (classified if wants_images else None)


def usable(chunk, mode):
    if mode == 'images':
        return is_figure(chunk)
    return bool(chunk.get('text') or chunk.get('caption') or (mode == 'joint' and chunk.get('image')))


def _project_tables(chunks, representation):
    """One table representation per request: HTML body or original table image."""
    if representation not in ('html', 'image'):
        raise ValueError('table_representation must be html or image')
    output = []
    for chunk in chunks:
        item = deepcopy(chunk)
        if representation == 'html':
            if not item.get('text'):
                raise ValueError(f"HTML table representation unavailable for {item['id']}")
            item['image'] = None
        else:
            if not item.get('image'):
                raise ValueError(f"Table image unavailable for {item['id']}")
            item['text'] = ''
        output.append(item)
    return output


def text_chunks(article, text_condition=None):
    condition = text_condition or TEXT_CONDITION
    output = []
    for original in article.get('chunks', []):
        chunk = deepcopy(original);chunk['image'] = None
        if chunk['kind'] == 'image' or (chunk['kind'] == 'table' and condition == 'without_html_tables'):
            chunk['kind'] = 'caption';chunk['text'] = ''
        if chunk.get('text') or chunk.get('caption') or chunk.get('transcription'):
            output.append(chunk)
    return output


def build_prompt(instructions, json_schema, extra, chunks, mode,
                 allow_transcriptions=False, text_condition=None):
    parts = [instructions]
    if json_schema is not None:
        parts.append('JSON SCHEMA:\n' + json.dumps(json_schema, ensure_ascii=False))
    if extra:
        parts.append(extra)
    source_texts, images = [], []
    total = sum(bool(render(c, mode, allow_transcriptions, text_condition)[1]) for c in chunks)
    for chunk in chunks:
        value, image = render(chunk, mode, allow_transcriptions, text_condition)
        if image:
            images.append(str(image))
            source_texts.append(f"Attached original image {len(images)} of {total} corresponds to {chunk['id']}.")
        source_texts.append(value)
    if len(images) > MAX_ENGINE_IMAGES:
        raise ValueError(f'At most {MAX_ENGINE_IMAGES} original images per prompt')
    if source_texts:
        parts.append('SOURCES:\n' + '\n\n'.join(source_texts))
    return '\n\n'.join(parts), images


def group_chunks(chunks, mode, max_images=1, allow_transcriptions=False, text_condition=None):
    if max_images > 1:
        raise ValueError('Image limit is one')
    groups, group, size, n_images = [], [], 0, 0
    for chunk in chunks:
        text, image = render(chunk, mode, allow_transcriptions, text_condition)
        if group and (size + len(text) > MAX_CHARS or n_images + bool(image) > max_images):
            groups.append(group);group=[];size=0;n_images=0
        group.append(chunk);size += len(text);n_images += bool(image)
    if group:
        groups.append(group)
    return groups


def context_text(context):
    return 'ARTICLE CONTEXT (shared frozen context.json):\n' + json.dumps(normalize_context(context), ensure_ascii=False)


def normalized_labels(value):
    if isinstance(value, dict):
        value = value.get('label', value.get('labels', []))
    if isinstance(value, str):
        value = [value]
    return [LABEL_ALIASES.get(label, label) for label in value or []]


def select(chunks, mode, labels):
    accepted = IMAGE_SELECT_LABELS if mode == 'images' else SELECT_LABELS
    return [c for c in chunks if usable(c, mode) and (labels is None or
            bool(set(normalized_labels(labels.get(c['id'], ['tested_model']))) & accepted))]


def _condition(value=None):
    value = value or TEXT_CONDITION
    if value not in ('with_html_tables', 'without_html_tables'):
        raise ValueError('text_condition must be with_html_tables or without_html_tables')
    return value


def _suffix(condition):
    return 'with_tables' if condition == 'with_html_tables' else 'without_tables'


def _joint_batches(article, chunks, condition, transcriptions=False, include_figures=True):
    """One request per paper: all selected text plus all selected figures.

    Each figure appears at its position in the article, with its original image.
    Only a paper with more than MAX_ENGINE_IMAGES figures is split: every request
    then repeats the full text with the next group of figures (the other figures'
    captions stay as text), so no figure is dropped. Tables enter joint inputs as
    HTML (per text condition), never as images.
    """
    selected_ids = {c['id'] for c in chunks}
    base = [c for c in text_chunks(article, condition) if c['id'] in selected_ids]
    visuals = [c for c in chunks if include_figures and is_figure(c)]
    if not visuals:
        return [base] if base else []
    order = lambda c: c.get('index', 0)
    output = []
    for start in range(0, len(visuals), MAX_ENGINE_IMAGES):
        group = {c['id']: c for c in visuals[start:start + MAX_ENGINE_IMAGES]}
        batch = [deepcopy(c) for c in base if c['id'] not in group] + [deepcopy(c) for c in group.values()]
        output.append(sorted(batch, key=order))
    return output


def _sources(chunks, mode, condition):
    sources = []
    for chunk in chunks:
        text, image = render(chunk, mode, False, condition)
        sources.append(Source(chunk['id'], chunk['kind'], chunk.get('page', 0), text,
                              Path(image) if image else None))
    return sources


def _source_chunks(sources):
    return [dict(id=s.id, kind=s.kind, page=s.page, caption='', text=s.text,
                 image=str(s.image) if s.image else None, transcription='') for s in sources]


def _task_request(task, context, chunks, mode, extra='', condition=None):
    spec = recipe(task)
    if spec['requires_context']:
        if context is None:
            raise ValueError(f'{task} requires the shared context')
        extra = context_text(context) + '\n' + extra
    extra += '\nDeclared text condition: ' + (condition or spec.get('text_condition', TEXT_CONDITION))
    request = build_prompt(spec['instructions'], spec['output_schema'], extra, chunks, mode,
                           spec['allows_figure_transcriptions'], condition)
    limit = MAX_ENGINE_IMAGES if spec['max_original_images'] is None else spec['max_original_images']
    if len(request[1]) > limit:
        raise ValueError('Recipe image budget exceeded')
    return request, spec['output_schema']


def general_pass(llm, processor, temperature, articles):
    """One request per article with the entire converted article and no images:
    prose, HTML tables, captions/footnotes and figure transcriptions."""
    files = _begin_stage('context');start_log = len(REQUEST_LOG)
    requests, allowed = [], []
    for article in articles:
        chunks = text_chunks(article, 'with_html_tables')
        if not chunks and article.get('full_text'):
            chunks = [dict(id='article_text', kind='paragraph', page=0, text=article['full_text'], caption='', image=None)]
        requests.append(_task_request('context', None, chunks, 'text', condition='with_html_tables')[0])
        allowed.append({c['id'] for c in [*article.get('chunks', []), *chunks]})
    answers = safe_generate(llm, processor, requests, schema('context'), temperature, CONTEXT_MAX_TOKENS)
    by_article = {article['id']: answer for article, answer in zip(articles, answers)}
    save_json(files('answers'), by_article)
    save_json(files('requests'), REQUEST_LOG[start_log:])
    failed = [aid for aid, answer in by_article.items() if answer is None]
    if failed:
        raise ValueError(f"Context generation failed for {failed}; do not run branches with partial context")
    output = {}
    for article, answer, ids in zip(articles, answers, allowed):
        context = merge_context([answer])
        validate_context(context, ids)
        output[article['id']] = context
    save_json(files('output'), output)
    return output


def load_contexts(path, articles=None):
    contexts = read_json(path)
    if articles is not None:
        for article in articles:
            if article['id'] not in contexts:
                raise ValueError('Context file is missing an article')
            validate_context(contexts[article['id']], {c['id'] for c in article['chunks']})
    else:
        for context in contexts.values():
            validate_context(context)
    return deepcopy(contexts)


def classify(llm, processor, temperature, articles, labels=None, contexts=None, mode='joint', text_condition=None,
             table_representation='html'):
    if contexts is None:
        raise ValueError('Generate/load shared context before classification')
    condition = _condition(text_condition);output = deepcopy(labels or {});jobs = []
    for article in articles:
        aid = article['id'];output.setdefault(aid, {})
        if mode == 'text':
            chunks = text_chunks(article, condition)
        elif mode == 'tables':
            chunks = _project_tables([c for c in article['chunks'] if c['kind'] == 'table'], table_representation)
        else:
            # Joint inputs carry tables as HTML only, so classify them that way.
            chunks = [dict(c, image=None) if c['kind'] == 'table' else c for c in article['chunks']]
        for chunk in chunks:
            if chunk['id'] in output[aid]:
                continue
            if mode == 'images' and not is_figure(chunk):
                continue
            input_mode = mode if mode in ('images', 'text', 'tables') else 'joint'
            task = 'classify_table' if chunk['kind'] == 'table' else 'classify_image' if chunk.get('image') else 'classify_text'
            jobs.append((aid, chunk['id'], *_task_request(task, contexts[aid], [chunk], input_mode, condition=condition)))
    answers = generate_all(llm, processor, temperature, [(request, contract) for _, _, request, contract in jobs])
    for (aid, chunk_id, _, _), answer in zip(jobs, answers):
        output[aid][chunk_id] = normalized_labels(answer) if answer is not None else ['tested_model']
    return output


def _batches(article, mode, labels, input_mode, condition, baseline=False, table_representation='html',
             include_figures=True, image_labels=None):
    if mode == 'tables':
        tables = select([c for c in article['chunks'] if c['kind'] == 'table'], 'joint', labels)
        return [[c] for c in _project_tables(tables, table_representation)]
    if mode == 'images':
        return [[c] for c in select(article['chunks'], 'images', image_labels)]
    chunks = text_chunks(article, condition) if mode == 'text' else article['chunks']
    chunks = select(chunks, mode, labels)
    if mode == 'joint':
        # Images follow their own labels; the baseline keeps every image.
        figures = [c for c in article['chunks'] if is_figure(c)] if baseline else \
            select(article['chunks'], 'images', image_labels)
        chunks = sorted([c for c in chunks if not is_figure(c)] + figures, key=lambda c: c['index'])
    if input_mode == 'elementwise':
        return [[c] for c in chunks]
    if mode == 'text':
        return [chunks] if chunks else []
    return _joint_batches(article, chunks, condition, baseline, include_figures=include_figures)


def _extraction_task(mode, group, input_mode, condition, baseline=False):
    if baseline:
        return 'baseline'
    if mode == 'tables' or (mode == 'text' and len(group) == 1 and group[0]['kind'] == 'table'):
        return 'table'
    if mode == 'text':
        return ('scaffolded_' if input_mode == 'scaffolded' else 'text_') + _suffix(condition)
    if mode == 'images':
        return 'image'
    return 'early_' + _suffix(condition)


def screen_relations(relations):
    """Drop single records whose beta is not finite or whose effect_type disagrees
    with mediator/moderator; keep everything else of the batch.

    An impossible p (e.g. "<0" or "1.5") is set to null and noted, since it
    cannot be compared. Returns (kept, dropped)."""
    from .statistics import normalize_p
    kept, dropped = [], []
    for record in relations:
        record = deepcopy(record)
        if record.get('p') is not None:
            try:
                normalize_p(record['p'])
            except ValueError as error:
                dropped.append(dict(relation=deepcopy(record), reason=f'p set to null: {error}', kept=True))
                record['p'] = None
        try:
            validate_relation(record, strict=False)
        except (ValueError, ValidationError) as error:
            dropped.append(dict(relation=record, reason=str(error), kept=False))
            continue
        kept.append(record)
    return kept, dropped


def postprocess(records, context):
    """The shared deterministic clean-up applied to every saved version
    (before and after review): canonical names/IDs, rule_fuse merging and the
    drop of unestimated duplicates of estimated paths."""
    return rule_fuse(records, context)


def _review_enabled(mode):
    return {'text': REVISE_TEXT, 'images': REVISE_IMAGES, 'tables': REVISE_TABLES, 'joint': REVISE_JOINT}[mode]


def _review(llm, processor, temperature, candidates, batches, context, aid, stage,
            mode, condition, enabled=True, intermediate=False):
    """Review one paper; see _review_all."""
    return _review_all(llm, processor, temperature, [dict(
        candidates=candidates, batches=batches, context=context, aid=aid, stage=stage,
        mode=mode, condition=condition, enabled=enabled, intermediate=intermediate)])[0]


def _review_all(llm, processor, temperature, jobs):
    """Review every paper's fixed snapshot with one batched generation call.

    Each job is one paper: its batches all review the same snapshot, and their
    decisions are combined without source-order priority. Writes no files: the
    caller stores the result in its stage's papers file. Returns
    [(after, audits, details)]; details holds conflicts, request metadata and
    field provenance per batch.
    """
    states, prepared = [], []
    for number, job in enumerate(jobs):
        before = deepcopy(job['candidates']);context = job['context']
        entries, batches = [], []
        for index, group in enumerate(job['batches'] or [[]]):
            sources = _sources(group, job['mode'], job['condition'])
            audit = dict(article_id=job['aid'], stage=job['stage'], batch_id=str(index), context_fingerprint=digest(context),
                         source_ids=sorted({s.id for s in sources}), status='pending', reason=None,
                         before=deepcopy(before), review=None, after=deepcopy(before))
            batch = dict(batch_id=str(index), request_metadata=None, field_provenance=None)
            validate(audit, schema('review_audit'))
            if not job['enabled'] or not group:
                audit.update(status='skipped', reason='Review disabled' if not job['enabled'] else 'No sources available')
            else:
                try:
                    request, preparation = revision_request(before, sources, job['stage'], index, context_text(context),
                        None, mode=job['mode'], decimal_comma=TEXT_DECIMAL_COMMA,
                        text_condition=job['condition'], context=context, intermediate=job.get('intermediate', False))
                    batch['request_metadata'] = preparation
                    # Sources are already rendered for this modality; tables keep their body.
                    item = build_prompt(request['instructions'], request['schema'], request['extra'], _source_chunks(sources),
                                        'tables' if job['mode'] == 'tables' else 'joint', False, job['condition'])
                    prepared.append((number, audit, batch, request, item))
                except Exception as error:
                    GENERATION_ERRORS.append(str(error))
                    audit.update(status='failed', reason=str(error), review=None, after=deepcopy(before))
            entries.append(audit);batches.append(batch)
        states.append(dict(before=before, entries=entries, batches=batches, updates=[]))
    answers = generate_all(llm, processor, temperature, [(item, request['schema']) for *_, request, item in prepared])
    for (number, audit, batch, request, _), answer in zip(prepared, answers):
        before = states[number]['before']
        try:
            audit['review'] = deepcopy(answer)
            if answer is None:
                raise ValueError('Invalid or missing review response')
            revised = request['check'](answer)
            audit.update(status='applied', after=revised)
            batch['field_provenance'] = decision_provenance(before, answer)
            states[number]['updates'].append(answer)
        except Exception as error:
            GENERATION_ERRORS.append(str(error))
            # Raw invalid payloads stay in REQUEST_LOG; audit remains schema-valid.
            audit.update(status='failed', reason=str(error), review=None, after=deepcopy(before))
    results = []
    for job, state in zip(jobs, states):
        for audit in state['entries']:
            validate(audit, schema('review_audit'))
        after, conflicts = (combine_updates(state['before'], state['updates'], job['context'])
                            if state['updates'] else (deepcopy(state['before']), []))
        after = postprocess(after, job['context'])
        results.append((after, state['entries'], dict(review_conflicts=conflicts, review_batches=state['batches'])))
    return results


def _finish_metadata(files, start, start_log, errors, context_fingerprint, metadata=None, papers=None):
    value = dict(elapsed_seconds=time.perf_counter() - start, prompt_fingerprint=prompt_fingerprint(),
                 context_fingerprint=context_fingerprint, errors=GENERATION_ERRORS[errors:],
                 request_count=len(REQUEST_LOG) - start_log, stage=files.prefix, stage_files=str(files),
                 **(metadata or {}))
    if papers is not None:
        save_json(files('papers'), papers)
    save_json(files('requests'), REQUEST_LOG[start_log:]);save_json(files('metadata'), value)
    LAST_STAGE_METADATA.clear();LAST_STAGE_METADATA.update(value)
    return value


def extract(llm, processor, temperature, articles, contexts, mode, labels,
            revision_dir=None, image_revision=None, text_revision_dir=None, text_revision=None,
            prior_relations=None, text_input=None, stage_name=None, text_condition=None,
            table_revision=None, joint_revision=None, table_representation='html', include_figures=True,
            image_labels=None, baseline=None):
    """image_labels: labels to reuse for gating images when `labels` is None (no text filter).
    baseline: unfiltered baseline (default: stage_name == 'baseline')."""
    condition = _condition(text_condition)
    if mode not in ('text', 'images', 'joint', 'tables'):
        raise ValueError('Unknown extraction mode')
    if table_representation not in ('html', 'image'):
        raise ValueError('table_representation must be html or image')
    if prior_relations is not None:
        return intermediate_fusion(llm, processor, temperature, articles, contexts, prior_relations,
            mode, labels, revision_dir, image_revision, text_revision_dir, text_revision,
            text_input=text_input, text_condition=condition, table_revision=table_revision,
            table_representation=table_representation, image_labels=image_labels)
    stage = stage_name or mode;baseline = stage == 'baseline' if baseline is None else baseline
    if baseline:
        condition = 'with_html_tables';labels = None
    custom_root = text_revision_dir if mode == 'text' and text_revision_dir is not None else revision_dir
    files = _begin_stage(stage, custom_root);start=time.perf_counter();start_log=len(REQUEST_LOG);error_start=len(GENERATION_ERRORS)
    frozen = digest(contexts)
    for article in articles:
        validate_context(contexts[article['id']], {c['id'] for c in article['chunks']})
    # The baseline is unfiltered whatever text input mode the caller uses elsewhere.
    input_mode = 'entire' if baseline else text_input or ('filtered' if labels is not None else 'entire')
    if input_mode not in ('entire', 'filtered', 'elementwise', 'scaffolded'):
        raise ValueError('Unknown text input mode')
    labels, image_labels = _image_labels(llm, processor, temperature, articles, contexts, mode, labels, input_mode,
                                         condition, table_representation, baseline, include_figures, image_labels)
    plans=[]
    for article in articles:
        aid=article['id'];groups=_batches(article,mode,labels.get(aid) if labels is not None else None,input_mode,condition,
            baseline,table_representation,include_figures,image_labels.get(aid) if image_labels is not None else None)
        plans.append(dict(aid=aid,context=contexts[aid],groups=groups,extras=['']*len(groups),scaffolds=[],
                          tasks=[_extraction_task(mode,g,input_mode,condition,baseline) for g in groups]))
    if input_mode == 'scaffolded' and mode == 'text':
        jobs=[(plan,index,_task_request('hypotheses_'+_suffix(condition),plan['context'],group,'text',condition=condition)[0])
              for plan in plans for index,group in enumerate(plan['groups'])]
        drafts=safe_generate(llm,processor,[item for *_,item in jobs],None,temperature)
        for (plan,index,_),scaffold in zip(jobs,drafts):
            plan['scaffolds'].append(dict(batch=index,source_ids=[c['id'] for c in plan['groups'][index]],draft=scaffold))
            plan['extras'][index]='HYPOTHESIS DRAFT (provisional; reread original sources):\n'+str(scaffold or 'No usable scaffold; extract from original sources.')
    owners=[(plan,index) for plan in plans for index in range(len(plan['groups']))]
    answers=generate_all(llm,processor,temperature,[_task_request(plan['tasks'][i],plan['context'],plan['groups'][i],mode,
                                                                  plan['extras'][i],condition) for plan,i in owners])
    for plan in plans:
        plan['records']=[];plan['batch_records']=[]
    for (plan,index),answer in zip(owners,answers):
        group=plan['groups'][index];status='completed';error=None;dropped=[]
        if answer is None:
            candidates=[];status='failed';error='No valid extraction response';GENERATION_ERRORS.append(error)
        else:
            kept,dropped=screen_relations(answer)
            candidates=canonicalize(kept,plan['context'])
        plan['records'].extend(candidates)
        plan['batch_records'].append(dict(batch=index,recipe=plan['tasks'][index],source_ids=[c['id'] for c in group],
                                          status=status,error=error,raw_response=answer,relations=candidates,
                                          screening=dropped))
    for plan in plans:
        plan['before']=postprocess(plan['records'],plan['context'])
    # The review sees the same source coverage, without any transcriptions.
    reviews=_review_all(llm,processor,temperature,[dict(candidates=plan['before'],batches=plan['groups'],context=plan['context'],
        aid=plan['aid'],stage=stage,mode=mode,condition=condition,
        enabled=_review_enabled(mode)) for plan in plans])
    outputs={};before_all={};after_all={};papers={}
    for plan,(after,audits,details) in zip(plans,reviews):
        aid=plan['aid'];before=plan['before']
        paper=dict(input_mode=input_mode,text_condition=condition,modality=mode,include_figures=include_figures if mode=='joint' else None,
                   context_fingerprint=digest(plan['context']),
                   stage_files=str(files), table_representation=table_representation if mode=='tables' else None,
                   relations_concatenated=plan['records'],relations_before_revision=before,relations_after_revision=after,
                   extractions=plan['batch_records'],scaffolds=plan['scaffolds'],audits=audits,**details)
        papers[aid]=paper
        target={'text':text_revision,'images':image_revision,'tables':table_revision,'joint':joint_revision}[mode]
        if target is not None:
            target[aid]=deepcopy(paper)
        outputs[aid]=after;before_all[aid]=before;after_all[aid]=after
    if digest(contexts)!=frozen:
        raise AssertionError('Shared context was mutated')
    stage_snapshot(stage,'before_revision',before_all,temperature)
    stage_snapshot(stage,'after_revision',after_all,temperature)
    _finish_metadata(files,start,start_log,error_start,frozen,dict(text_condition=condition,input_mode=input_mode),papers)
    return outputs
def intermediate_fusion(llm, processor, temperature, articles, contexts, seeds, mode, labels,
                        revision_dir=None, image_revision=None, text_revision_dir=None,
                        text_revision=None, intermediate=None, text_input=None, text_condition=None,
                        input_versions=None, table_revision=None, table_representation='html',
                        stage_name=None, prior_modes=None, image_labels=None):
    """Update a fixed prior with one further modality (text, tables or images).

    prior_modes names the modality/modalities the seeds came from; a chain of
    three modalities calls this once per later modality.
    """
    if mode not in ('text', 'images', 'tables'):
        raise ValueError('Intermediate target must be text, tables or images')
    condition=_condition(text_condition);input_mode=text_input or ('filtered' if labels is not None else 'entire')
    if input_mode not in ('entire','filtered','elementwise','scaffolded'):
        raise ValueError('Unknown text input mode')
    if prior_modes is None:
        prior_modes=['images'] if mode=='text' else ['text']
    direction=stage_name or '_then_'.join(_SINGULAR[m] for m in [*prior_modes, mode])
    custom_root = text_revision_dir if mode == 'text' and text_revision_dir is not None else revision_dir
    files=_begin_stage(direction,custom_root);start=time.perf_counter();start_log=len(REQUEST_LOG);errors=len(GENERATION_ERRORS)
    if isinstance(seeds,(str,Path)):
        seed_source=str(seeds);seeds=read_json(seeds)
    else:
        seed_source='provided_results'
    seeds=deepcopy(seeds);frozen=digest(contexts);result={};before_all={};details={}
    labels,image_labels=_image_labels(llm,processor,temperature,articles,contexts,mode,labels,input_mode,condition,
                                      table_representation,image_labels=image_labels)
    plans=[]
    for article in articles:
        aid=article['id'];context=normalize_context(contexts[aid]);prior=canonicalize(seeds.get(aid,[]),context)
        groups=_batches(article,mode,labels.get(aid) if labels is not None else None,input_mode,condition,
                        table_representation=table_representation,
                        image_labels=image_labels.get(aid) if image_labels is not None else None)
        prior_text=('PRIOR EXTRACTION (fixed first-modality snapshot):\n'+json.dumps(candidate_records(prior),ensure_ascii=False)
                    +'\nPRIOR MODALITIES: '+', '.join(prior_modes)+'; CURRENT SECOND MODALITY: '+mode)
        plans.append(dict(aid=aid,context=context,prior=prior,groups=groups,extras=[prior_text]*len(groups),
                          updates=[],audits=[],scaffolds=[],update_provenance=[]))
    # Scaffolded text also remains available as a recorded intermediate input variant.
    if input_mode=='scaffolded' and mode=='text':
        jobs=[(plan,index,_task_request('hypotheses_'+_suffix(condition),plan['context'],group,'text',condition=condition)[0])
              for plan in plans for index,group in enumerate(plan['groups'])]
        drafts=safe_generate(llm,processor,[item for *_,item in jobs],None,temperature)
        for (plan,index,_),scaffold in zip(jobs,drafts):
            plan['scaffolds'].append(dict(batch=index,draft=scaffold))
            plan['extras'][index]+='\nPROVISIONAL HYPOTHESIS SCAFFOLD (not evidence):\n'+str(scaffold or '')
    owners,requests=[],[]
    for plan in plans:
        for index,group in enumerate(plan['groups']):
            task='intermediate_table' if mode=='tables' or (len(group)==1 and group[0]['kind']=='table') else (
                'intermediate_image' if mode=='images' else 'intermediate_text_'+_suffix(condition))
            sources=_sources(group,mode,condition)
            audit=dict(article_id=plan['aid'],stage=direction+'_update',batch_id=str(index),context_fingerprint=digest(plan['context']),
                source_ids=sorted({s.id for s in sources}),status='pending',reason=None,before=deepcopy(plan['prior']),
                review=None,after=deepcopy(plan['prior']))
            owners.append((plan,sources,audit))
            requests.append(_task_request(task,plan['context'],group,mode,plan['extras'][index],condition))
    answers=generate_all(llm,processor,temperature,requests)
    for (plan,sources,audit),answer in zip(owners,answers):
        try:
            if answer is None:
                raise ValueError('Invalid intermediate response; retaining prior')
            revised=apply_revision(plan['prior'],answer,sources,plan['context'])
            audit.update(review=answer,after=revised,status='applied');plan['updates'].append(answer)
            plan['update_provenance'].append(dict(batch_id=audit['batch_id'],
                                                  field_provenance=decision_provenance(plan['prior'], answer)))
        except Exception as error:
            audit.update(status='failed',reason=str(error));GENERATION_ERRORS.append(str(error))
        validate(audit,schema('review_audit'));plan['audits'].append(audit)
    for plan in plans:
        plan['updated'],plan['conflicts']=combine_updates(plan['prior'],plan['updates'],plan['context'])
        plan['updated']=postprocess(plan['updated'],plan['context'])
    reviews=_review_all(llm,processor,temperature,[dict(candidates=plan['updated'],batches=plan['groups'],context=plan['context'],
        aid=plan['aid'],stage=direction,mode=mode,condition=condition,
        enabled=_review_enabled(mode),intermediate=True) for plan in plans])
    papers={}
    for plan,(after,review_audits,review_details) in zip(plans,reviews):
        aid=plan['aid']
        paper=dict(seed_source=seed_source,seed_relations=plan['prior'],
            source_mode=prior_modes[0] if len(prior_modes)==1 else list(prior_modes),
            target_mode=mode,conflict_policy='second_modality_evidence_wins',text_condition=condition,
            table_representation=table_representation if mode=='tables' else None,
            input_versions=input_versions or {'seed':'caller_supplied'},stage_files=str(files),
            scaffolds=plan['scaffolds'],updates=plan['audits'],update_field_provenance=plan['update_provenance'],
            update_conflicts=plan['conflicts'],
            relations_before_revision=plan['updated'],relations_after_revision=after,audits=review_audits,**review_details)
        papers[aid]=paper
        if intermediate is not None:
            intermediate[aid]=deepcopy(paper)
        branch_review={'text':text_revision,'images':image_revision,'tables':table_revision}[mode]
        if branch_review is not None:
            branch_review[aid]=deepcopy(paper)
        result[aid]=after;before_all[aid]=plan['updated'];details[aid]=paper
    if digest(contexts)!=frozen:
        raise AssertionError('Context mutated')
    stage_snapshot(direction,'seed',seeds,temperature)
    stage_snapshot(direction,'before_revision',before_all,temperature)
    stage_snapshot(direction,'after_revision',result,temperature)
    _finish_metadata(files,start,start_log,errors,frozen,dict(input_versions=input_versions or {'seed':'caller_supplied'}),papers)
    return result


def early_fusion(llm,processor,temperature,articles,contexts,labels=None,**kwargs):
    condition=_condition(kwargs.get('text_condition'))
    kwargs.setdefault('stage_name','early')
    if FILTER_CONTENT and kwargs.get('text_input')!='entire':
        labels=classify(llm,processor,temperature,articles,labels,contexts,'joint',condition)
    elif labels is not None:
        kwargs.setdefault('image_labels',labels);labels=None  # images are still gated by their labels
    return extract(llm,processor,temperature,articles,contexts,'joint',labels,**kwargs)


def extract_tables(llm, processor, temperature, articles, contexts, labels=None,
                   representation='html', **kwargs):
    """Standalone table extraction; extraction and review use the same representation."""
    return extract(llm, processor, temperature, articles, contexts, 'tables', labels,
                   table_representation=representation, **kwargs)


def _reviewable_fusion(records, originals, context):
    """Fused records as plain relations with their candidates' original evidence
    (no quotations enter prediction-only fusion), postprocessed like every stage.

    Also accepts unchecked LLM fusion output: unknown candidate IDs are ignored,
    and records failing the relation screen (non-finite beta, role mismatch) are dropped.
    """
    output=[]
    for record in records:
        members=[originals[cid] for cid in record.get('candidate_ids',[]) if cid in originals]
        if record.get('status')=='contradiction' and members:
            output.extend(deepcopy(members))
            continue
        value={k:deepcopy(record[k]) for k in schema('relation')['items']['properties'] if k!='evidence' and k in record}
        value['evidence']=unique([e for member in members for e in member.get('evidence',[])])
        output.append(value)
    kept,_=screen_relations(output)
    return postprocess(kept,context)


def _late(llm,processor,temperature,articles,contexts,relations,use_llm,
          fusion_dir=None,fusion_metadata=None,text_condition=None,input_versions=None,review=None,stage_name=None):
    """Prediction-only fusion of two or three modalities: {'text', 'tables', 'images'} -> {aid: records}."""
    modalities=[m for m in MODALITIES if relations.get(m) is not None]
    unknown=set(relations)-set(MODALITIES)
    if unknown or not modalities:
        raise ValueError(f'Late fusion takes relations keyed by {MODALITIES}')
    condition=_condition(text_condition);stage=stage_name or ('late_llm' if use_llm else 'late_rule')
    # The final original-source review covers every fused modality: table bodies
    # when tables are a fused modality, figures only when images are.
    review_condition='with_html_tables' if 'tables' in modalities else condition
    files=_begin_stage(stage,fusion_dir);start=time.perf_counter();start_log=len(REQUEST_LOG);errors=len(GENERATION_ERRORS)
    frozen=digest(contexts);before_all={};results={};papers={};owners=[];plans=[]
    for article in articles:
        aid=article['id'];context=contexts[aid]
        records={m:clean_relations(relations[m].get(aid,[])) for m in modalities}
        payload=fusion_input(context,**{_FUSION_ARGUMENT[m]:records[m] for m in modalities})
        originals={f'{m}_{i}':r for m in modalities for i,r in enumerate(records[m],1)}
        owners.append((aid,payload))
        plans.append(dict(article=article,aid=aid,context=context,payload=payload,originals=originals,answer=None,fallback=None))
    asked=[plan for plan in plans if use_llm and plan['payload']['candidates']]
    answers_llm=generate_all(llm,processor,temperature,[_task_request('late',plan['context'],[],'text',
        'FUSED MODALITIES: '+', '.join(modalities)+'\nPREDICTIONS:\n'+json.dumps(plan['payload']['candidates'],ensure_ascii=False),
        condition) for plan in asked])
    for plan,answer in zip(asked,answers_llm):
        plan['answer']=answer
    asked_ids={id(plan) for plan in asked}
    for plan in plans:
        if id(plan) in asked_ids and plan['answer'] is not None:
            # The LLM merge is used as answered: no comparison with the rule-based consolidation.
            fused=plan['answer']
        else:
            if id(plan) in asked_ids:
                plan['fallback']='Missing or unparsable fusion output after retry';GENERATION_ERRORS.append(plan['fallback'])
            fused=consolidate_candidates(plan['payload']['candidates'],plan['context'])
            validate_fusion(fused,plan['payload'])
        plan['fused']=fused
        # Both saved versions are plain, postprocessed relations.
        plan['before']=_reviewable_fusion(fused,plan['originals'],plan['context'])
        before_all[plan['aid']]=deepcopy(plan['before'])
    answers=[plan['answer'] for plan in plans]
    # Original-source review is a distinct, saved stage, never part of fusion input.
    enabled=REVISE_JOINT if review is None else review
    reviews=_review_all(llm,processor,temperature,[dict(
        candidates=plan['before'],
        batches=_joint_batches(plan['article'],plan['article']['chunks'],review_condition,include_figures='images' in modalities),
        context=plan['context'],aid=plan['aid'],
        stage=stage+'_original_source_review',mode='joint',condition=review_condition,enabled=enabled) for plan in plans])
    for plan,(revised,audits,details) in zip(plans,reviews):
        after=revised if enabled else plan['before']
        papers[plan['aid']]=dict(modalities=modalities,review_text_condition=review_condition,stage_files=str(files),
                                 fusion_input=plan['payload'],candidate_provenance=plan['originals'],
                                 fusion_records=plan['fused'],
                                 relations_before_revision=plan['before'],relations_after_revision=after,audits=audits,
                                 naming_fallback=plan['fallback'],input_versions=input_versions or {'upstream':'caller_supplied'},
                                 **details)
        results[plan['aid']]=after
    # Checkpoint for late_fusion.recover_fusion(source=<this file>).
    save_json(files('responses'),dict(owners=owners,answers=answers,output_schema=schema('late_fusion'),empty_article_ids=[]))
    stage_snapshot(stage,'before_revision',before_all,temperature);stage_snapshot(stage,'after_revision',results,temperature)
    if digest(contexts)!=frozen:
        raise AssertionError('Context mutated')
    metadata=_finish_metadata(files,start,start_log,errors,frozen,dict(modalities=modalities,
        input_versions=input_versions or {'upstream':'caller_supplied'}),papers)
    if fusion_metadata is not None:
        fusion_metadata.update(metadata=metadata,articles=papers)
    return results


_FUSION_ARGUMENT={'text':'text_relations','tables':'table_relations','images':'image_relations'}


def late_rule_fusion(articles,contexts,text_relations,image_relations,temperature,
                     llm=None,processor=None,table_relations=None,**kwargs):
    relations=dict(text=text_relations,tables=table_relations,images=image_relations)
    return _late(llm,processor,temperature,articles,contexts,relations,False,**kwargs)


def llm_fuse(llm,processor,temperature,articles,contexts,text_relations,image_relations,
             fusion_dir=None,validate_output=True,labels=None,fusion_metadata=None,table_relations=None,**kwargs):
    relations=dict(text=text_relations,tables=table_relations,images=image_relations)
    return _late(llm,processor,temperature,articles,contexts,relations,True,
                 fusion_dir=fusion_dir,fusion_metadata=fusion_metadata,**kwargs)


def fuse_rules(relations,context):
    return rule_fuse(relations,context)


def resolve_table_setting(table_setting=None, text_condition=None):
    """Setting A (in_text): text carries HTML tables. Setting B (separate): text
    without HTML tables plus a table modality. Images are figures in both."""
    if table_setting is None:
        table_setting='in_text' if _condition(text_condition)=='with_html_tables' else 'separate'
    if table_setting not in TABLE_SETTINGS:
        raise ValueError(f'table_setting must be one of {TABLE_SETTINGS}')
    required='with_html_tables' if table_setting=='in_text' else 'without_html_tables'
    if text_condition is not None and text_condition!=required:
        raise ValueError(f'table_setting={table_setting} requires text_condition={required}')
    return table_setting,required


def default_modalities(table_setting):
    return ('text','images') if table_setting=='in_text' else ('text','tables','images')


def _check_modalities(modalities, table_setting):
    modalities=tuple(modalities)
    if len(modalities)!=len(set(modalities)) or not set(modalities)<=set(MODALITIES) or not 2<=len(modalities)<=3:
        raise ValueError(f'Fusion needs two or three distinct modalities from {MODALITIES}')
    if 'tables' in modalities and table_setting!='separate':
        raise ValueError("A table modality requires table_setting='separate'; in_text puts tables into text")
    return modalities


def paradigm_order(paradigm):
    """'text_then_tables_then_image' -> ('text', 'tables', 'images')."""
    parts=paradigm.split('_then_')
    if len(parts)<2 or any(p not in _PARADIGM_MODALITY for p in parts):
        raise ValueError(f'Unknown paradigm {paradigm}')
    return tuple(_PARADIGM_MODALITY[p] for p in parts)


def run_paradigm(llm,processor,temperature,articles,contexts,labels,paradigm,
                 revision_dir=None,image_revision=None,text_revision_dir=None,text_revision=None,
                 intermediate=None,text_input=None,fusion_dir=None,text_condition=None,
                 table_revision=None,joint_revision=None,fusion_metadata=None,upstream_version='after_revision',
                 table_representation='html',table_setting=None,modalities=None,branches=None,stage_suffix=''):
    """Run one paradigm under table setting A ('in_text') or B ('separate').

    Late fusion fuses `modalities` (default: text+images in A, text+tables+images
    in B). Intermediate paradigms name their order, e.g. text_then_image or
    text_then_tables_then_image (B only). Early fusion and the baseline read text
    with HTML tables plus figures in both settings; early fusion includes figures
    only when images are among the modalities.

    branches={'text': {aid: paper}, ...} reuses single-modality papers (the
    text_revision/image_revision/table_revision stores of earlier runs) instead
    of extracting those modalities again, so every fusion paradigm consumes the
    same upstream results.

    text_input='entire' never filters text by labels (images stay gated).
    stage_suffix is appended to every stage file prefix (e.g. '_filtered_in_text'),
    so several variants of one paradigm can share a run directory.
    """
    if upstream_version not in ('before_revision','after_revision'):
        raise ValueError('upstream_version must be before_revision or after_revision')
    table_setting,condition=resolve_table_setting(table_setting,text_condition)
    selected_labels=labels if FILTER_CONTENT and text_input!='entire' else None
    name=lambda stage:stage+stage_suffix
    # Images are always gated by the shared labels, also when text is not filtered.
    options=dict(revision_dir=revision_dir,image_revision=image_revision,text_revision_dir=text_revision_dir,
                 text_revision=text_revision,text_input=text_input,text_condition=condition,
                 table_revision=table_revision,joint_revision=joint_revision,table_representation=table_representation,
                 image_labels=labels)
    if paradigm in ('text','images','tables'):
        return extract(llm,processor,temperature,articles,contexts,paradigm,selected_labels,
                       stage_name=name(paradigm),**options)
    if paradigm=='baseline':
        return extract(llm,processor,temperature,articles,contexts,'joint',None,stage_name='baseline',
                       baseline=True,**options)
    if paradigm=='early':
        chosen=_check_modalities(modalities or default_modalities(table_setting),table_setting)
        joint_condition='with_html_tables' if table_setting=='in_text' or 'tables' in chosen else 'without_html_tables'
        return early_fusion(llm,processor,temperature,articles,contexts,selected_labels,stage_name=name('early'),
                            **dict(options,text_condition=joint_condition),include_figures='images' in chosen)
    versions={}
    sinks={'text':text_revision,'images':image_revision,'tables':table_revision}
    def branch(mode):
        if branches is not None and mode in branches:
            store=branches[mode];missing=[a['id'] for a in articles if a['id'] not in store]
            if missing:
                raise ValueError(f'Reused {mode} results are missing articles: {missing}')
            versions[mode]=dict(version=upstream_version,reused=True,
                                stage_files=sorted({str(store[a['id']].get('stage_files')) for a in articles}))
            return {a['id']:deepcopy(store[a['id']]['relations_'+upstream_version]) for a in articles}
        store={}
        stores={m:(store if m==mode else sinks[m]) for m in MODALITIES}
        extract(llm,processor,temperature,articles,contexts,mode,selected_labels,
            image_revision=stores['images'],text_revision=stores['text'],table_revision=stores['tables'],
            text_input=text_input,text_condition=condition,stage_name=name(f'{paradigm}_{mode}'),
            table_representation=table_representation,image_labels=labels)
        if sinks[mode] is not None:
            sinks[mode].update(store)
        versions[mode]=dict(version=upstream_version,stage_files=LAST_STAGE_METADATA['stage_files'])
        return {aid:paper['relations_'+upstream_version] for aid,paper in store.items()}
    if paradigm in ('late_rule','late_llm'):
        chosen=_check_modalities(modalities or default_modalities(table_setting),table_setting)
        relations={m:branch(m) for m in chosen}
        return _late(llm,processor,temperature,articles,contexts,relations,paradigm=='late_llm',
            fusion_dir=fusion_dir,fusion_metadata=fusion_metadata,text_condition=condition,input_versions=versions,
            stage_name=name(paradigm))
    order=_check_modalities(paradigm_order(paradigm),table_setting)
    if modalities is not None and set(modalities)!=set(order):
        raise ValueError(f'{paradigm} fuses {order}; it conflicts with modalities={tuple(modalities)}')
    seeds=branch(order[0]);steps={}
    for position,mode in enumerate(order[1:],1):
        stage=name('_then_'.join(_SINGULAR[m] for m in order[:position+1]));detail={}
        result=intermediate_fusion(llm,processor,temperature,articles,contexts,seeds,mode,selected_labels,
            image_revision=image_revision,text_revision=text_revision,table_revision=table_revision,
            intermediate=detail,text_input=text_input,text_condition=condition,input_versions=dict(versions),
            table_representation=table_representation,stage_name=stage,prior_modes=list(order[:position]),
            image_labels=labels)
        steps[stage]=detail
        versions[stage]=dict(version=upstream_version,stage_files=LAST_STAGE_METADATA['stage_files'])
        seeds={aid:paper['relations_'+upstream_version] for aid,paper in detail.items()}
    if intermediate is not None:
        for aid,paper in detail.items():
            intermediate[aid]=dict(deepcopy(paper),order=list(order),
                previous_steps={s:deepcopy(d[aid]) for s,d in steps.items() if d is not detail})
    return result
