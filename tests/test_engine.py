from copy import deepcopy

import pytest
import torch
from torch import nn

from s1kit import SystemOne
from s1kit.prompts import messages
from s1kit.schema import parse_request
from conftest import Backbone, ByteTokenizer


def test_selected_rows_match_full_head_with_bias(engine, request_data):
    result = engine.decide(request_data)
    assert len(engine.backbone.calls) == 3  # Boundary probes never run the model.
    parsed = parse_request(request_data)
    with torch.inference_mode():
        for q, ids in zip(parsed.questions, list(engine.backbone.calls)):
            hidden = engine.backbone(input_ids=ids).last_hidden_state
            full = engine.head(hidden)[0, -1, 65:65 + len(q.options)]
            assert result["questions"][q.id]["option_logits"] == pytest.approx(full.tolist(), abs=1e-4)


def test_all_candidates_visible_and_other_questions_absent(engine, request_data):
    engine.decide(request_data)
    for ids, q in zip(engine.backbone.calls, parse_request(request_data).questions):
        text = engine.tokenizer.decode(ids[0].tolist())
        assert q.instructions in text
        for option in q.options:
            assert option.description in text
        for other in parse_request(request_data).questions:
            if q.id != other.id:
                assert other.instructions not in text


def test_order_subset_and_repeated_requests_are_independent(engine, request_data):
    original = engine.decide(request_data)
    changed = deepcopy(request_data)
    changed["questions"] = dict(reversed(list(changed["questions"].items())))
    assert engine.decide(changed)["questions"] == original["questions"]
    changed["questions"] = {"severity": changed["questions"]["severity"]}
    assert engine.decide(changed)["questions"]["severity"] == original["questions"]["severity"]
    changed["state"] = "unrelated"
    engine.decide(changed)
    assert engine.decide(request_data) == original


def test_media_is_forwarded_and_expanded_boundary_checked(engine, request_data):
    plain = engine.decide(request_data)
    media = engine.decide(request_data, image=torch.ones(8))
    for key in plain["questions"]:
        assert media["questions"][key]["input_tokens"] == plain["questions"][key]["input_tokens"] + 3
        assert media["questions"][key]["option_logits"] != plain["questions"][key]["option_logits"]


@pytest.mark.parametrize("kind,key", [("choice", "intent"), ("score", "severity"), ("noul", "workaround")])
@pytest.mark.parametrize("with_media", [False, True])
def test_shortcuts_match_full_decisions(engine, request_data, kind, key, with_media):
    question = request_data["questions"][key]
    media = {"image": torch.ones(8)} if with_media else {}
    expected = engine.decide(request_data, temperature=1.7, **media)["questions"][key]["answer"]
    engine.backbone.calls.clear()
    args = [request_data["state"], question["instructions"]]
    if kind != "noul":
        args.append(question["criteria"])
    assert getattr(engine, kind)(*args, temperature=1.7, **media) == expected
    assert len(engine.backbone.calls) == 1


def test_noul_shortcut_preserves_custom_descriptions(engine):
    criteria = {"true": "Present", "false": "Absent"}
    engine.noul("Evidence", "Present?", criteria=criteria)
    prompt = engine.tokenizer.decode(engine.backbone.calls[-1][0].tolist())
    assert '"description":"Absent","key":"false","letter":"A"' in prompt
    assert '"description":"Present","key":"true","letter":"B"' in prompt


@pytest.mark.parametrize("call", [
    lambda e: e.choice("state", "Choose", {"only": "one"}),
    lambda e: e.score("state", "Rate", ["low", ""]),
    lambda e: e.noul("state", "Present?", criteria={"yes": "wrong"}),
    lambda e: e.choice("state", "Choose", {"a": "A", "b": "B"}, temperature=0),
])
def test_shortcuts_use_existing_validation(engine, call):
    with pytest.raises(ValueError):
        call(engine)
    assert not engine.backbone.calls


def test_custom_output_extractor(engine, request_data):
    class TupleBackbone(nn.Module):
        def forward(self, input_ids):
            return (torch.ones(1, input_ids.shape[1], 8),)

    bound = SystemOne(preprocess=engine.preprocess, backbone=TupleBackbone(), head=engine.head,
                      tokenizer=engine.tokenizer, hidden=lambda output: output[0])
    assert bound.decide(request_data)["questions"]


@pytest.mark.parametrize("temperature", [0, -1, True, None, float("nan"), float("inf")])
def test_temperature_rejected_before_forward(engine, request_data, temperature):
    with pytest.raises(ValueError, match="temperature"):
        engine.decide(request_data, temperature=temperature)
    assert not engine.backbone.calls


def test_token_budget_boundary(engine, request_data):
    request_data["questions"] = {"intent": request_data["questions"]["intent"]}
    result = engine.decide(request_data)
    length = result["questions"]["intent"]["input_tokens"]
    engine.max_tokens = length
    assert engine.decide(request_data) == result  # Probes may be one token longer.
    engine.max_tokens -= 1
    with pytest.raises(ValueError, match="no truncation"):
        engine.decide(request_data)


@pytest.mark.parametrize("problem", ["merge", "truncate", "image_expansion"])
def test_actual_input_boundary_failure(engine, request_data, problem):
    prepare = engine.preprocess

    def broken(conversation, *, answer=""):
        inputs = prepare(conversation, answer=answer)
        if answer:
            if problem == "merge":
                inputs["input_ids"][0, -2] = 0
            elif problem == "truncate":
                inputs["input_ids"] = inputs["input_ids"][:, :-1]
            else:
                inputs["input_ids"] = torch.cat([torch.tensor([[1]]), inputs["input_ids"]], dim=1)
        return inputs

    engine.preprocess = broken
    with pytest.raises(ValueError, match="answer boundary"):
        engine.decide(request_data)
    assert not engine.backbone.calls


@pytest.mark.parametrize("problem", ["multiple", "decode", "vocab"])
def test_invalid_label_tokens(engine, request_data, problem):
    class BrokenTokenizer(ByteTokenizer):
        def encode(self, text, **kwargs):
            if text == "A":
                return [65, 66] if problem == "multiple" else [999] if problem == "vocab" else [65]
            return super().encode(text, **kwargs)

        def decode(self, ids):
            if ids == [999]:
                return "A"
            return "wrong" if problem == "decode" else super().decode(ids)

    engine.tokenizer = BrokenTokenizer()
    with pytest.raises(ValueError):
        engine.decide(request_data)
    assert not engine.backbone.calls


@pytest.mark.parametrize("inputs", [
    [], {}, {"input_ids": torch.tensor([1, 2])},
    {"input_ids": torch.ones(2, 3, dtype=torch.long)},
    {"input_ids": torch.ones(1, 3)}, {"input_ids": torch.empty(1, 0, dtype=torch.long)},
    {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 0]])},
    {"input_ids": torch.tensor([[1, 2]]), "past_key_values": object()},
])
def test_invalid_prepared_inputs(engine, request_data, inputs):
    engine.preprocess = lambda *args, **kwargs: inputs
    with pytest.raises((TypeError, ValueError)):
        engine.decide(request_data)
    assert not engine.backbone.calls


def test_invalid_hidden_shape(engine, request_data):
    engine.hidden = lambda output: output.last_hidden_state[:, -1, :]
    with pytest.raises(ValueError, match="hidden"):
        engine.decide(request_data)


def test_preprocessor_must_not_reuse_mutable_input_dict(engine, request_data):
    prepare = engine.preprocess
    shared = {}

    def reused(conversation, *, answer=""):
        shared.update(prepare(conversation, answer=answer))
        return shared

    engine.preprocess = reused
    with pytest.raises(ValueError, match="return fresh inputs"):
        engine.decide(request_data)
    assert not engine.backbone.calls


def test_quantized_or_custom_heads_rejected(engine):
    class CustomHead(nn.Linear):
        def forward(self, value):
            return super().forward(value) * 2

    with pytest.raises(TypeError, match="native dense"):
        SystemOne(preprocess=engine.preprocess, backbone=engine.backbone,
                  head=CustomHead(8, 256), tokenizer=engine.tokenizer)


def test_prompt_is_canonical_and_matches_reference_contract(request_data):
    question = parse_request(request_data).questions[0]
    conversation = messages({"z": True, "a": {"z": 2, "a": 1}}, question)
    assert conversation[1]["content"].startswith('State:\n{"a":{"a":1,"z":2},"z":true}\n\nQuestion:\n')
    assert '"letter":"A"' in conversation[1]["content"]
