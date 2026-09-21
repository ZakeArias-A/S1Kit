"""Classify text with a causal language model that provides a chat template."""

import argparse
import json
from pathlib import Path

from s1kit import SystemOne


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local model directory or Hugging Face model ID")
    parser.add_argument("--revision", help="Model commit; required when using a model ID")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--customize", action="store_true", help="Run structured state, score and system prompt examples")
    args = parser.parse_args()
    if not Path(args.model).is_dir() and not args.revision:
        parser.error("Provide --revision for a Hugging Face model ID")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    common = dict(revision=args.revision, local_files_only=not args.allow_download)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **common).to(args.device)
    engine = SystemOne.from_transformers(model, tokenizer, max_tokens=args.max_tokens)

    if args.customize:
        if __package__:
            from .customization import run
        else:
            from customization import run
        print(json.dumps(run(engine), indent=2, ensure_ascii=False, allow_nan=False))
        return

    answer = engine.choice(
        "PDF export fails, but CSV export still works.",
        "Classify the main intent.",
        {"bug": "Reporting a broken feature", "feature": "Requesting a new feature"},
    )
    print(json.dumps(answer, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
