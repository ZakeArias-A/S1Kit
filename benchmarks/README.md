# Benchmarks

S1Kit is an inference wrapper. Scores describe the particular base model, prompt, and precision used here. They do not describe every model that can be bound to S1Kit.

## Measured results — 2026-09-21

| Public split | S1Kit + Qwen3.5-4B | Jev 1.13.0 |
|---|---|---|
| Easy | 48/48 (100%) | 48/48 (100%) |
| Standard | 64/72 (88.9%) | 70/72 (97.2%) |
| Hard | 61/111 (55.0%) | 79/111 (71.2%) |
| All public tasks | 173/231 (74.9%) | 197/231 (85.3%) |

Both systems returned valid distributions on all tasks; no renormalization was needed. See [per-task predictions and probabilities](results/2026-09-21/predictions.csv) and [metadata, dataset hashes, and results by family](results/2026-09-21/summary.json).

Local hardware: NVIDIA RTX 4060 Laptop GPU (8 GB), Intel i9-13900H, 64 GB RAM, Windows 11. Software: Python 3.12.10, PyTorch 2.11.0+cu130, Transformers 5.17.0. Qwen uses NF4 double quantization, BF16 computation, an unquantized output head, SDPA, and a 262,144-pixel budget per image. The optional causal-conv1d and flash-linear-attention kernels were unavailable; the model used their PyTorch fallbacks. Maximum measured text input: 4,030 tokens.

Observed p50/p95 wall time: local 0.204/2.040 s; Jev API 0.575/0.636 s. These are single-run observations with different hardware and timing boundaries, not a speed ranking. An earlier partial CPU run is excluded; all published local text and image results use the CUDA configuration.

### Image results

The six simple image cases scored **5/6**. Both inventory charts and both service dashboards were classified correctly. The unpaid invoice was incorrectly marked paid, with `noul=0.977`; a concentrated probability is not evidence of correctness. [All six results and image-free controls](results/2026-09-21/vision.json).

The complex set scored **1/5**:

| Case | Expected | Actual | Probability of actual choice |
|---|---|---|---|
| No ordered SKU stock | Refund | Manual review | 39.3% |
| Ordered SKU in stock | Replace | Replace | 58.4% |
| Photo matches ordered color | Reject mismatch claim | Replace | 75.7% |
| Outside 14-day window | Manual review | Replace | 54.5% |
| Customer photo removed | Request photo | Replace | 57.6% |

These results do not demonstrate reliable cross-image policy reasoning. The test fixes resolution, precision, prompts, and options; it does not isolate which factor caused a mistake. [All complex results, image hashes, and controls](results/2026-09-21/vision-complex.json).

A draft complex run was excluded after a rendering bug made two intended variants identical. The published run uses corrected fixtures, checked for distinct processed pixels and complete image grids. Every final case is retained, including failures; prompts, model settings, and intended gold labels were unchanged.

## Text: JevBench public splits

We evaluate all **231 public tasks** from [JevBench](https://github.com/fstandhartinger/jevbench/tree/15028ef00ed3949e05c5821492253ed1f212d681): Easy (48), Standard (72), and Hard (111). Private tasks and the private judge split are unavailable. This is a public-subset comparison, not an official JevBench leaderboard score.

The runner imports the pinned upstream task loader, TypeSafe adapter, and `score_task` function. Each request contains only `state` and one question. Gold labels and provenance stay outside the prompt. Both systems see the same original question, criteria, and option order.

- Each system receives one warmup request, followed by one request per task, in dataset order. No retries, sample selection, temperature tuning, or result cache.
- S1Kit uses temperature 1 and one backbone forward per question. No generation, training, or calibration.
- Accuracy uses the highest-probability label, including for Score questions; it does not round the returned expected score. Ties follow upstream lexicographic ordering.
- Malformed responses count as incorrect. Upstream permits renormalization within its 0.02 rounding tolerance and separately records strict validity at 0.001.
- This public dataset has previously been inspected during development. It is a reproducibility check, not a blind held-out evaluation.
- Latency includes preprocessing and label checks locally, and network time for Jev. Model loading is excluded. Different hardware and serving conditions prevent a direct speed ranking.

### Reproduce

From the S1Kit repository root:

```sh
uv sync --locked --extra cu130 --extra qwen
git clone https://github.com/fstandhartinger/jevbench .cache/jevbench
git -C .cache/jevbench checkout 15028ef00ed3949e05c5821492253ed1f212d681

uv run --no-sync python -m benchmarks.run --jevbench .cache/jevbench --backend typesafe --output runs/jev
uv run --no-sync python -m benchmarks.run --jevbench .cache/jevbench --backend s1kit --config benchmarks/qwen-cuda.json --output runs/s1kit
uv run --no-sync python -m benchmarks.report --s1kit runs/s1kit --typesafe runs/jev --output runs/comparison
```

The TypeSafe command reads `TYPESAFE_API_KEY`, or prompts without echoing it. It calls only `https://api.typesafe.ai/v1/systemone`, with model `jev-1.13.0`. Keys, headers, and raw server error bodies are not recorded. API usage may incur charges.

The local example loads cached `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, NF4 4-bit with double quantization and BF16 computation, on CUDA. The output head stays unquantized. CPU preprocessing uses 4 PyTorch threads. Set `HF_HUB_CACHE` to an existing cache, or add `"local_files_only": false` to the factory's `kwargs` to allow the initial download. `max_tokens` is 8192; longer inputs fail rather than being truncated. For an unquantized CPU run, use `benchmarks/qwen-cpu.json`; its results are a separate configuration.

For another model, change the config's `factory` to an importable `module:function` that returns a bound `SystemOne` instance, and supply its arguments under `kwargs`. The runner has no model-family dispatch.

Each run creates `summary.json` with versions and semantic dataset hashes, plus `predictions.jsonl` with per-task outcomes. The report command refuses incomplete or mismatched runs and produces a compact summary and CSV for review.

## Images: paired application examples

The six [image fixtures](../examples/vision/preview.png) cover invoice payment checks (`noul`), inventory replenishment (`choice`), and service incident severity (`score`). Each pair uses exactly the same request; only the image changes. Gold labels are stored separately in [cases.json](../examples/vision/cases.json).

These are small, synthetic application examples, not an established vision benchmark. The runner also records one image-free control per pair. Controls have no visible evidence and are not included in accuracy. Jev's [documented input is text only](https://docs.typesafe.ai/models), so there is no Jev vision score.

```sh
uv run --no-sync python -m examples.vision.render
uv run --no-sync python -m benchmarks.vision --config benchmarks/qwen-cuda.json --output runs/vision.json
```

The render script creates all fixtures deterministically before inference. The vision runner records every case, including failures, image hashes, full probability distributions, and controls. It uses the same model precision as the text run. Images enter the processor as pixels; filenames, expected answers, and drawing parameters are not sent to the model. There is no external OCR step.

### Complex case: browser + document + real photograph

[complex/cases.json](../examples/vision/complex/cases.json) adds four three-image returns decisions and a missing-photo ablation. The three inputs have different sizes: a 1040 × 680 mock browser page, an 820 × 1080 policy document, and the unchanged 489 × 500 user-provided photograph of a yellow object. Browser and policy data are fictional.

The order is for a green variant, delivered on September 9 and claimed on September 21 (12 days). The current policy permits a color-mismatch remedy within 14 days: replace if that SKU is in stock, otherwise refund. Stock for a different blue SKU is irrelevant. The document includes a clearly superseded policy as a distractor. The photograph is needed to establish the actual color; its straps and shadows are not evidence of product damage.

| Case | Evidence change | Expected next action |
|---|---|---|
| `return-no-stock` | Ordered SKU has 0 units | Refund |
| `return-in-stock` | Ordered SKU has 8 units; everything else unchanged | Replace |
| `return-color-matches` | Ordered variant is yellow, matching the photo | Reject the color-mismatch claim |
| `return-outside-window` | Delivery was September 1, making the interval 20 days | Manual review |
| `return-no-photo` | Remove the real photograph from the first case | Request a clear photo |

```sh
uv run --no-sync python -m examples.vision.complex.render
uv run --no-sync python -m benchmarks.vision --config benchmarks/qwen-cuda.json --cases examples/vision/complex/cases.json --output runs/vision-complex.json
```

This tests cross-image reading, color comparison, date arithmetic, version selection, and conditional action selection. It does not establish general performance on real-world returns. The unchanged real photograph is included separately; the renderer only creates the browser and document fixtures.

## Sources

- [JevBench protocol and datasets](https://github.com/fstandhartinger/jevbench/tree/15028ef00ed3949e05c5821492253ed1f212d681), Florian Standhartinger and contributors. The dataset stays in its own checkout, with upstream license and attribution.
- [TypeSafe API reference](https://docs.typesafe.ai/api) and [versioned model reference](https://docs.typesafe.ai/models).
