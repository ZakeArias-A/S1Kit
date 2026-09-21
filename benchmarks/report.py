"""Combine two complete runs into compact, inspectable publication data."""

import argparse
import csv
import json
from pathlib import Path
import statistics

from benchmarks.run import summarize


def read_run(directory):
    metadata = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 231 or len({r["id"] for r in rows}) != 231:
        raise ValueError("Publication requires all 231 distinct public tasks")
    if metadata["summary"] != summarize(rows):
        raise ValueError("Summary does not match predictions")
    return metadata, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1kit", required=True, type=Path)
    parser.add_argument("--typesafe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    runs = {name: read_run(path) for name, path in [("s1kit", args.s1kit), ("jev", args.typesafe)]}
    left, right = runs["s1kit"][0], runs["jev"][0]
    if left["dataset_hashes"] != right["dataset_hashes"] or left["jevbench_revision"] != right["jevbench_revision"]:
        raise ValueError("Cannot compare different dataset versions")
    if [r["id"] for r in runs["s1kit"][1]] != [r["id"] for r in runs["jev"][1]]:
        raise ValueError("Task order differs between runs")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name, (meta, rows) in runs.items():
        seconds = sorted(r["seconds"] for r in rows)
        meta["latency_seconds"] = {"p50": statistics.median(seconds), "p95": seconds[int(0.95 * (len(seconds) - 1))]}
        meta["family_accuracy"] = {
            family: {"correct": sum(r["scoring"]["correct"] is True for r in rows if r["family"] == family),
                     "count": sum(r["family"] == family for r in rows)}
            for family in sorted({r["family"] for r in rows})
        }
        if name == "jev":
            meta["models_returned"] = sorted({r["result"]["model"] for r in rows})
            meta["input_tokens"] = sum(r["result"]["usage"].get("input_tokens", 0) for r in rows)
        else:
            meta["input_tokens"] = sum(r["result"].get("input_tokens", 0) for r in rows)
            meta["max_input_tokens"] = max(r["result"].get("input_tokens", 0) for r in rows)
        metadata[name] = meta
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "tier", "family", "expected", "s1kit_predicted", "jev_predicted",
                         "s1kit_probabilities", "jev_probabilities", "s1kit_seconds", "jev_seconds"])
        for a, b in zip(runs["s1kit"][1], runs["jev"][1], strict=True):
            if a["expected"] != b["expected"]:
                raise ValueError("Gold labels differ between runs")
            writer.writerow([a["id"], a["tier"], a["family"], a["expected"],
                             a["scoring"]["predicted"], b["scoring"]["predicted"],
                             json.dumps(a["result"]["probs"]), json.dumps(b["result"]["probs"]),
                             a["seconds"], b["seconds"]])


if __name__ == "__main__":
    main()
