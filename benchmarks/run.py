"""Evaluate the pinned public JevBench splits with S1Kit or TypeSafe."""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

JEVBENCH_REVISION = "15028ef00ed3949e05c5821492253ed1f212d681"
SPLITS = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}


def load_suite(checkout):
    """Verify the dataset and scoring code before importing the upstream harness."""
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != JEVBENCH_REVISION:
        raise ValueError(f"JevBench must be checked out at {JEVBENCH_REVISION}")
    changes = subprocess.check_output(
        ["git", "-C", str(checkout), "status", "--porcelain", "--", "jevbench", "datasets/public"], text=True,
    ).strip()
    if changes:
        raise ValueError("JevBench code or public datasets have local modifications")
    sys.path.insert(0, str(checkout.resolve()))
    from jevbench.tasks import dataset_hash, load_jsonl

    tasks, hashes = [], {}
    for tier, filename in SPLITS.items():
        split = load_jsonl(str(checkout / "datasets/public" / filename))
        if any(t.split != "public" or t.expected is None for t in split):
            raise ValueError("Only public tasks with gold labels are supported")
        hashes[tier] = dataset_hash(split)
        tasks.extend((tier, task) for task in split)
    return tasks, hashes


def make_request(task):
    # The expected answer and other benchmark metadata never reach the model.
    return {"state": task.state, "questions": {"decision": task.question}}


def probabilities(answer):
    if "noul" in answer:
        p = answer["noul"]
        return {"no": 1 - p, "yes": p}
    return answer["probabilities"]


def load_factory(config):
    module, name = config["factory"].split(":", 1)
    return getattr(importlib.import_module(module), name)(**config.get("kwargs", {}))


def summarize(rows):
    result = {}
    for tier in [*SPLITS, "all"]:
        items = [r for r in rows if tier == "all" or r["tier"] == tier]
        if not items:
            continue
        correct = sum(r["scoring"]["correct"] is True for r in items)
        result[tier] = {"count": len(items), "correct": correct, "accuracy": correct / len(items),
                        "invalid": sum(not r["scoring"]["valid"] for r in items),
                        "renormalized": sum(r["scoring"].get("renormalized", False) for r in items)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jevbench", required=True, type=Path)
    parser.add_argument("--backend", choices=("s1kit", "typesafe"), required=True)
    parser.add_argument("--config", type=Path, help="S1Kit factory and keyword arguments, as JSON")
    parser.add_argument("--output", required=True, type=Path, help="New directory for this run")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    tasks, hashes = load_suite(args.jevbench)
    from jevbench.scoring import score_task

    config = None
    if args.backend == "s1kit":
        if not args.config:
            parser.error("--config is required for s1kit")
        config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "backend": args.backend,
                "jevbench_revision": JEVBENCH_REVISION, "dataset_hashes": hashes,
                "python": platform.python_version(), "platform": platform.platform(),
                "config": config, "warmup_requests": 1, "retries": 0,
                "timing": "Synchronous wall time including preprocessing, excluding loading; API includes network"}
    rows = []
    try:
        if args.backend == "typesafe":
            from jevbench.adapters.typesafe import TypeSafeAdapter

            if not os.environ.get("TYPESAFE_API_KEY"):
                os.environ["TYPESAFE_API_KEY"] = getpass.getpass("TypeSafe API key: ").strip()
            adapter = TypeSafeAdapter(endpoint="https://api.typesafe.ai", model="jev-1.13.0", timeout_s=120)

            def evaluate(task):
                response = adapter.run(task)
                # Retain only structured benchmark fields, never headers or raw error bodies.
                result = {"probs": response.probs if response.ok else {}, "model": response.model,
                          "http_status": response.status, "usage": response.usage}
                if not response.ok:
                    result["error"] = f"API failure (HTTP {response.status}); response omitted"
                return result
        else:
            import torch

            torch.set_num_threads(args.threads)
            engine = load_factory(config)
            metadata.update(torch=torch.__version__, transformers=importlib.metadata.version("transformers"),
                            threads=torch.get_num_threads(), temperature=1.0,
                            device=str(engine.head.weight.device), dtype=str(engine.head.weight.dtype))
            if engine.head.weight.is_cuda:
                metadata["gpu"] = torch.cuda.get_device_name(engine.head.weight.device)

            def evaluate(task):
                record = engine.decide(make_request(task), temperature=1.0)["questions"]["decision"]
                return {"probs": probabilities(record["answer"]), "input_tokens": record["input_tokens"],
                        "input_ids_sha256": record["input_ids_sha256"], "option_logits": record["option_logits"]}

        print(f"Warmup, then {len(tasks)} public tasks", flush=True)
        warmup = evaluate(tasks[0][1])
        if "error" in warmup:
            raise RuntimeError(warmup["error"])
        metadata["source_sha256"] = {
            str(p).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted(Path("src/s1kit").glob("*.py")), Path(__file__).relative_to(Path.cwd())]
        }
        with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for index, (tier, task) in enumerate(tasks, 1):
                start = time.perf_counter()
                try:
                    result = evaluate(task)
                except (ValueError, RuntimeError, KeyError) as exc:
                    result = {"probs": {}, "error": type(exc).__name__}
                elapsed = time.perf_counter() - start
                row = {"id": task.id, "tier": tier, "family": task.family, "expected": task.expected,
                       "result": result, "seconds": elapsed, "scoring": score_task(result["probs"], task)}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                if index % 10 == 0 or index == len(tasks):
                    print(f"{index}/{len(tasks)} {tier}: {sum(r['scoring']['correct'] is True for r in rows)} correct", flush=True)
                if result.get("http_status") in (401, 403):
                    raise RuntimeError("API authorization failed; stopping the run")
    finally:
        if args.backend == "typesafe":
            os.environ.pop("TYPESAFE_API_KEY", None)
        metadata["completed_tasks"] = len(rows)
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["summary"] = summarize(rows)
        (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
