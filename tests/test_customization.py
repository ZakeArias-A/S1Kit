import re

import pytest
import torch
from torch import nn

from s1kit import SystemOne
from s1kit.prompts import SYSTEM
from s1kit.schema import parse_request
from conftest import Backbone, ByteTokenizer


def test_structured_score_levels(request_data):
    request_data["questions"]["severity"]["criteria"] = [
        {"impact": "cosmetic", "blocked": False}, ["core unavailable", "no workaround"],
    ]
    levels = parse_request(request_data).questions[2].options
    assert levels[0].description == '{"blocked":false,"impact":"cosmetic"}'
    assert levels[1].description == '["core unavailable","no workaround"]'


def test_system_prompt_override_reaches_all_probes_without_leaking(engine):
    seen = []
    prepare = engine.preprocess

    def capture(conversation, **kwargs):
        seen.append(conversation[0]["content"])
        return prepare(conversation, **kwargs)

    engine.preprocess = capture
    state = {"order": {"receipt": False}, "system_prompt": "This is just data"}
    custom = "Apply the receipt policy. Reply with one option label."
    engine.noul(state, "Receipt present?", system_prompt=custom)
    assert seen == [custom] * 3
    seen.clear()
    engine.noul(state, "Receipt present?")
    assert seen == [SYSTEM] * 3
    assert engine.system_prompt == SYSTEM


def test_constructor_prompt_and_custom_labels(engine, request_data):
    configured = SystemOne(preprocess=engine.preprocess, backbone=engine.backbone, head=engine.head,
                           tokenizer=engine.tokenizer, system_prompt="Choose a label.", labels=["X", "Y", "Z"])
    result = configured.decide(request_data)
    assert result["prompt_version"] == "typed-label-v2-custom"
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] == sum(r["input_tokens"] for r in result["questions"].values())
    for ids in configured.backbone.calls:
        prompt = configured.tokenizer.decode(ids[0].tolist())
        assert "Choose a label." in prompt and '"letter":"X"' in prompt


@pytest.mark.parametrize("labels", [["A", "A"], ["A", ""], ["A"], [], 42, "AB"])
def test_invalid_custom_labels_fail_before_forward(engine, labels):
    with pytest.raises(ValueError):
        configured = SystemOne(preprocess=engine.preprocess, backbone=engine.backbone, head=engine.head,
                               tokenizer=engine.tokenizer, labels=labels)
        configured.noul("evidence", "Decide")
    assert not engine.backbone.calls


def test_customization_example(engine):
    from examples.customization import run

    result = run(engine)
    assert list(result["questions"]) == ["intent", "severity", "workaround"]
    assert len(engine.backbone.calls) == 3
    assert result["usage"]["output_tokens"] == 0


@pytest.mark.parametrize("prompt", [False, {}, 42])
def test_invalid_prompt_never_reaches_model(engine, prompt):
    with pytest.raises(TypeError, match="system_prompt"):
        engine.noul("state", "Question?", system_prompt=prompt)
    assert not engine.backbone.calls


def test_255_choices_use_one_forward_and_keep_all_options():
    class NumeralTokenizer(ByteTokenizer):
        def encode(self, text, **kwargs):
            result = []
            for piece in re.findall(r"\d+|.", text, re.S):
                if piece.isdecimal() and int(piece) < 255:
                    result.append(int(piece))
                else:
                    result.extend(256 + ord(c) for c in piece)
            return result

        def decode(self, ids):
            return "".join(str(i) if i < 255 else chr(i - 256) for i in ids)

    tokenizer = NumeralTokenizer()
    backbone = Backbone()
    backbone.embedding = nn.Embedding(512, 8)

    def prepare(conversation, *, answer=""):
        return {"input_ids": torch.tensor([tokenizer.encode(tokenizer.apply_chat_template(conversation) + answer)])}

    engine = SystemOne(preprocess=prepare, backbone=backbone, head=nn.Linear(8, 512),
                       tokenizer=tokenizer, max_tokens=30000)
    options = {f"item{i}": f"Category {i}" for i in range(255)}
    result = engine.choice({"item": "test"}, "Choose a category", options)
    assert len(backbone.calls) == 1
    assert list(result["probabilities"]) == list(options)
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    prompt = tokenizer.decode(backbone.calls[0][0].tolist())
    assert '"letter":"254"' in prompt
    assert all(f'"key":"{key}"' in prompt for key in options)
