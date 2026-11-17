"""vLLM engine settings and generation, and run directories. Importing this loads no model."""
from datetime import datetime
from pathlib import Path

from .common import strict_json


def vision_settings(model_name, n_images=1, visual_token_budget=65536):
    """Image resolution settings; with more images per request, each image gets fewer visual tokens."""
    name = model_name.lower()
    if "gemma-4" in name or "gemma4" in name:
        return {"max_soft_tokens": 1120}
    for key, patch, minimum in (("qwen3", 32, 65536), ("qwen2.5", 28, 3136), ("qwen2-vl", 28, 3136)):
        if key in name:
            tokens = min(16384, visual_token_budget // max(1, n_images))
            return {"min_pixels": minimum, "max_pixels": max(minimum, tokens * patch * patch)}
    return {}


def engine_settings(model_name, max_images=8, visual_token_budget=65536):
    """Keyword arguments for vllm.LLM(model=model_name, **settings)."""
    vision = vision_settings(model_name, visual_token_budget=visual_token_budget)
    settings = dict(
        max_model_len=128000 if "max_pixels" in vision else 32768,
        limit_mm_per_prompt={"image": max_images},
        mm_processor_kwargs=vision,
    )
    if "max_soft_tokens" in vision:
        settings["hf_overrides"] = {
            "vision_config": {"default_output_length": 1120},
            "vision_soft_tokens_per_image": 1120,
        }
    return settings


def vllm_generate(llm, processor, items, json_schema, temperature, max_tokens, model_name,
                  enable_thinking=False, max_images=8, visual_token_budget=65536):
    """Run one vLLM batch. items are (prompt text, image paths).

    Return (answers, completions): an answer is parsed JSON (plain text if
    json_schema is None) or None for a truncated/unparsable output; completions
    are the raw texts with their finish reasons.
    """
    from PIL import Image
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    prompts = []
    for text, images in items:
        if len(images) > max_images:
            raise ValueError(f"At most {max_images} original images per request")
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        entry = {"prompt": processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )}
        if images:
            loaded = []
            for path in images:
                with Image.open(path) as image:
                    loaded.append(image.convert("RGB"))
            entry["multi_modal_data"] = {"image": loaded}
            entry["mm_processor_kwargs"] = vision_settings(model_name, len(images), visual_token_budget)
        prompts.append(entry)

    structured = StructuredOutputsParams(json=json_schema) if json_schema is not None else None
    sampling = dict(max_tokens=max_tokens, structured_outputs=structured)
    if temperature is not None:
        sampling["temperature"] = temperature
    answers, completions = [], []
    for output in llm.generate(prompts, SamplingParams(**sampling)):
        completion = output.outputs[0]
        finish = getattr(completion, "finish_reason", None)
        completions.append(dict(text=completion.text, finish_reason=finish))
        if finish not in {"stop", None}:
            answers.append(None)
        elif json_schema is None:
            answers.append(completion.text)
        else:
            try:
                answers.append(strict_json(completion.text))
            except ValueError:
                answers.append(None)
    return answers, completions


def new_run_dir(root="out/stages", now=None):
    """Create root/<YYYY-MM-DD_hh-mm> (with _2, _3, ... if it exists) and return it."""
    root = Path(root)
    name = (now or datetime.now()).strftime("%Y-%m-%d_%H-%M")
    path, number = root / name, 2
    while path.exists():
        path = root / f"{name}_{number}"
        number += 1
    path.mkdir(parents=True)
    return path
