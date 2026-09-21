import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.run import make_request, probabilities, summarize
from s1kit.schema import parse_request


@pytest.mark.parametrize("answer,expected", [
    ({"noul": 0.8}, {"no": 0.2, "yes": 0.8}),
    ({"choice": "b", "probabilities": {"a": 0.1, "b": 0.9}}, {"a": 0.1, "b": 0.9}),
    ({"score": 0.8, "probabilities": {"0": 0.2, "1": 0.8}}, {"0": 0.2, "1": 0.8}),
])
def test_native_answers_map_to_benchmark_labels(answer, expected):
    assert probabilities(answer) == pytest.approx(expected)


def test_request_does_not_leak_gold_or_provenance():
    task = SimpleNamespace(state="A customer asks for a refund.",
                           question={"type": "noul", "instructions": "Is this a billing issue?"},
                           expected="yes", labels=["no", "yes"], provenance={"notes": "gold reasoning"})
    request = make_request(task)
    assert request == {"state": task.state, "questions": {"decision": task.question}}
    parse_request(request)


def test_failed_tasks_remain_in_accuracy_denominator():
    rows = [{"tier": "hard", "scoring": {"correct": True, "valid": True}},
            {"tier": "hard", "scoring": {"correct": False, "valid": False}}]
    result = summarize(rows)["all"]
    assert result["count"] == 2
    assert result["accuracy"] == 0.5
    assert result["invalid"] == 1


def test_vision_pairs_change_only_image_evidence():
    directory = Path(__file__).resolve().parents[1] / "examples/vision"
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 6
    for first, second in zip(cases[::2], cases[1::2], strict=True):
        assert first["group"] == second["group"]
        assert first["request"] == second["request"]
        assert first["expected"] != second["expected"]
        assert (directory / first["images"][0]).read_bytes() != (directory / second["images"][0]).read_bytes()
        parse_request(first["request"])


def test_complex_case_requires_different_sizes_and_missing_photo_control():
    from PIL import Image

    directory = Path(__file__).resolve().parents[1] / "examples/vision/complex"
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 5
    assert all(case["request"] == cases[0]["request"] for case in cases)
    assert cases[0]["images"][1:] == cases[1]["images"][1:]
    assert cases[-1]["images"] == cases[0]["images"][:2]
    assert [c["expected"] for c in cases] == ["refund", "replace", "reject", "manual_review", "request_photo"]
    # Each counterfactual must change the rendered order, not just its declared gold label.
    assert len({(directory / case["images"][0]).read_bytes() for case in cases[:4]}) == 4
    sizes = []
    for name in cases[0]["images"]:
        with Image.open(directory / name) as image:
            sizes.append(image.size)
    assert len(set(sizes)) == 3
    parse_request(cases[0]["request"])
