# Examples

[Documentation](../docs/README.md) · [Visual case gallery](vision/README.md) · [Benchmark results](../benchmarks/README.md)

Run these scripts from the repository root after installing S1Kit. Models must already be available locally unless `--allow-download` is passed. For a Hugging Face model ID, provide a pinned commit with `--revision`.

| Example | What it demonstrates | Output |
|---|---|---|
| [text.py](text.py) | Bind a causal LM with a chat template | One Choice answer |
| [customization.py](customization.py) | Structured state, criteria, system prompt and independent questions | Full decision records for Choice, Score and Noul |
| [gpt2.py](gpt2.py) | Supply preprocessing without a chat template | One Noul answer |
| [qwen.py](qwen.py) | Bind a native multimodal processor and backbone | Decision records for a JSON request |
| [vision](vision/README.md) | Compare image evidence and cross-image policy decisions | Fixtures, expected answers and measured results |

## Text with a chat template

```sh
uv sync --locked --extra cpu --extra transformers
uv run --no-sync python examples/text.py ./my-model
```

Uses `SystemOne.from_transformers()` and `choice()`. The model must support the default system/user chat messages and a native linear output head. CPU is the default; for BF16 inference, sync with `--extra cu130 --extra transformers` and use `--device cuda`. Set `--max-tokens` within the model's supported context length.

For structured state, structured score levels, custom system prompts and multiple independent questions:

```sh
uv run --no-sync python examples/text.py ./my-model --customize
```

[customization.py](customization.py) also exposes `run(engine)` for any already bound model. See the [input guide](../docs/inputs.md) and [documentation index](../docs/README.md).

## Custom text preprocessing

```sh
uv run --no-sync python examples/gpt2.py ./my-gpt2-model
```

Shows how to supply a prompt for a model without a chat template, then call `noul()`. This demonstrates integration; GPT-2 is not instruction-tuned.

## Images with Qwen3.5

```sh
uv sync --locked --extra cu130 --extra qwen
uv run --no-sync python examples/qwen.py --image photo.png --cache-dir ./models
```

Uses pinned Qwen3.5-4B weights, CUDA/NF4 4-bit loading and BF16 computation. Pass `--allow-download` to download the weights into the chosen cache. For CPU, first run `uv sync --locked --extra cpu --extra qwen`, then use `--device cpu --quantization none`.

With an image, the default request compares red, green and blue. Without one, the script runs the three text questions in [request.json](request.json). To use different questions:

```sh
uv run --no-sync python examples/qwen.py my-request.json --image photo.png --cache-dir ./models
```

The `bind_qwen()` function shows native processor use and explicit text positions when alternating text and image requests. It is a model-specific example, not part of the generic engine.

## Paired image scenarios

[vision/cases.json](vision/cases.json) contains six invoice, inventory, and service-dashboard cases. The PNG files are ready to use; [vision/render.py](vision/render.py) regenerates them with Pillow. Expected answers are stored outside each request.

```python
import json
from pathlib import Path
from PIL import Image
from examples.qwen import load_engine

engine = load_engine()  # Cached Qwen3.5-4B, CPU / FP32
case = json.loads(Path("examples/vision/cases.json").read_text())[0]
with Image.open(Path("examples/vision") / case["images"][0]) as image:
    result = engine.decide(case["request"], images=[image.convert("RGB")])
print(result["questions"]["decision"]["answer"])
```

Run all six plus image-free controls with the [benchmark commands](../benchmarks/README.md).

The [complex cases](vision/README.md#多图交叉判断售后处理) combine a browser page, a policy document and a real photograph. Requests and expected answers are in [complex/cases.json](vision/complex/cases.json); use the benchmark runner to evaluate all five variants.

```sh
uv run --no-sync python -m benchmarks.vision --config benchmarks/qwen-cuda.json --cases examples/vision/complex/cases.json --output runs/vision-complex.json
```

This uses cached weights with the pinned CUDA configuration. Set `HF_HUB_CACHE` to the model cache, or configure loading as described in [benchmarks](../benchmarks/README.md). The result file includes each prediction, its probabilities and image-free controls; it does not modify the published results.
