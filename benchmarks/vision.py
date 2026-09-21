"""Run image fixtures and one text-only control per group."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from PIL import Image
import torch

from benchmarks.run import load_factory, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cases", type=Path, default=Path("examples/vision/cases.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    engine = load_factory(config)
    rows, controls = [], {}
    for case in cases:
        if case["group"] not in controls:
            record = engine.decide(case["request"])["questions"]["decision"]
            controls[case["group"]] = record["answer"]
        pictures, hashes, sizes = [], {}, {}
        for filename in case["images"]:
            path = args.cases.parent / filename
            with Image.open(path) as original:
                pictures.append(original.convert("RGB"))
                sizes[filename] = list(original.size)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        start = time.perf_counter()
        record = engine.decide(case["request"], images=pictures)["questions"]["decision"]
        elapsed = time.perf_counter() - start
        probs = probabilities(record["answer"])
        prediction = max(sorted(probs), key=probs.get)
        row = {"id": case["id"], "expected": case["expected"], "predicted": prediction,
               "correct": prediction == case["expected"], "answer": record["answer"],
               "input_tokens": record["input_tokens"], "seconds": elapsed,
               "image_sha256": hashes, "image_sizes": sizes}
        rows.append(row)
        print(f"{case['id']}: expected={case['expected']} predicted={prediction}", flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
              "dtype": str(engine.head.weight.dtype), "device": str(engine.head.weight.device),
              "torch": torch.__version__, "threads": torch.get_num_threads(), "temperature": 1.0,
              "transformers": importlib.metadata.version("transformers"),
              "pillow": importlib.metadata.version("Pillow"),
              "cases_hash_format": "UTF-8 with LF line endings",
              "cases_sha256": hashlib.sha256(args.cases.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
              "cases": rows, "text_only_controls": controls}
    if engine.head.weight.is_cuda:
        report["gpu"] = torch.cuda.get_device_name(engine.head.weight.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
