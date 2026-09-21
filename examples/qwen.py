"""Run text or image decisions with Qwen3.5-4B."""

import argparse
import json
from pathlib import Path

import torch

from s1kit import SystemOne

MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def bind_qwen(model, processor, *, max_tokens=4096):
    """Images are RGB PIL objects prepared by the caller; no media autodetection."""
    tokenizer = processor.tokenizer
    device = model.get_input_embeddings().weight.device

    def preprocess(messages, *, answer="", images=()):
        if images:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": messages[0]["content"]}]},
                {"role": "user", "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": messages[1]["content"]},
                ]},
            ]
            prompt = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = dict(processor(text=[prompt + answer], images=list(images), return_tensors="pt").to(device))
        else:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = tokenizer.encode(prompt + answer, add_special_tokens=False)
            length = len(ids)
            inputs = {
                "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, length), dtype=torch.long, device=device),
                # Qwen text positions must not inherit a previous image's RoPE offsets.
                "position_ids": torch.arange(length, device=device)[None, None, :].expand(3, 1, -1),
            }
        return {**inputs, "use_cache": False, "return_dict": True}

    return SystemOne(
        preprocess=preprocess, backbone=model.model,
        head=model.get_output_embeddings(), tokenizer=tokenizer, max_tokens=max_tokens,
    )


def load_engine(*, cache_dir=None, device="cpu", quantization="none", local_files_only=True, max_tokens=4096):
    """Load this example's pinned model; callers can also bind an existing model."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    if device == "cpu" and quantization != "none":
        raise ValueError("CPU requires quantization='none'")
    if quantization not in ("none", "4bit"):
        raise ValueError("quantization must be 'none' or '4bit'")
    common = dict(revision=REVISION, cache_dir=cache_dir, local_files_only=local_files_only)
    processor = AutoProcessor.from_pretrained(MODEL_ID, **common)
    processor.image_processor.size["longest_edge"] = 262144
    processor.image_processor.size["shortest_edge"] = min(processor.image_processor.size["shortest_edge"], 262144)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kwargs = dict(dtype=dtype, attn_implementation="sdpa", device_map={"": device})
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=["lm_head"],
        )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, **common, **kwargs)
    return bind_qwen(model, processor, max_tokens=max_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", nargs="?", type=Path, help="Decision request JSON; defaults to a text or image example")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    args = parser.parse_args()
    if args.device == "cpu" and args.quantization != "none":
        parser.error("CPU requires --quantization none")
    from PIL import Image

    request_path = args.request or Path(__file__).with_name("image_request.json" if args.image else "request.json")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    engine = load_engine(cache_dir=args.cache_dir, device=args.device,
                         quantization=args.quantization, local_files_only=not args.allow_download)
    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    result = engine.decide(
        request, images=images,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
