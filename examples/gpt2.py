"""Use a custom text prompt with GPT-2, which has no built-in chat template."""

import argparse
import json
from pathlib import Path

from s1kit import SystemOne


def bind_gpt2(model, tokenizer):
    device = model.get_input_embeddings().weight.device

    def preprocess(messages, *, answer=""):
        prompt = "\n\n".join(message["content"] for message in messages) + "\nAnswer:"
        inputs = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt")
        return {**inputs.to(device), "use_cache": False, "return_dict": True}

    return SystemOne.from_transformers(
        model, tokenizer, preprocess=preprocess,
        max_tokens=model.config.n_positions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local model directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Model commit; required when using a model ID")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        parser.error("Provide --revision for a Hugging Face model ID")

    from transformers import AutoTokenizer, GPT2LMHeadModel

    common = dict(revision=args.revision, local_files_only=not args.allow_download)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    model = GPT2LMHeadModel.from_pretrained(args.model, **common)
    result = bind_gpt2(model, tokenizer).noul(
        "PDF export fails, but CSV export still works.",
        "Is a workaround available?",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
