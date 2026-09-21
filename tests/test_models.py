"""Real model classes with random tiny weights. No downloads or quality claims."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig,
                          LlamaForCausalLM, Qwen3_5ForConditionalGeneration)
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)

from examples.gpt2 import bind_gpt2
from examples.qwen import bind_qwen
from s1kit import SystemOne
from s1kit.prompts import messages
from s1kit.readout import make_answer
from s1kit.schema import parse_request
from conftest import ByteTokenizer


@pytest.fixture(scope="module", params=["qwen", "gpt2", "llama"])
def bound_model(request):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260921)
        tokenizer = ByteTokenizer()
        if request.param == "qwen":
            text = Qwen3_5TextConfig(
                vocab_size=272, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=4096,
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2]},
                attention_dropout=0.0, pad_token_id=0,
            )
            vision = Qwen3_5VisionConfig(
                depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
                spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=32, num_position_embeddings=16,
            )
            model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
                text_config=text, vision_config=vision, image_token_id=260, video_token_id=261,
                vision_start_token_id=262, vision_end_token_id=263,
            )).eval()
            engine = bind_qwen(model, SimpleNamespace(tokenizer=tokenizer))
        elif request.param == "gpt2":
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=256, n_positions=4096, n_embd=32, n_layer=2, n_head=2,
                resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
            )).eval()
            engine = bind_gpt2(model, tokenizer)
        else:
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
            )).eval()
            tokenizer.chat_template = "byte-test-template"
            engine = SystemOne.from_transformers(model, tokenizer)
    return model, engine


def test_native_full_logits_match_selected_readout(bound_model, request_data):
    model, engine = bound_model
    result = engine.decide(request_data)
    with torch.inference_mode():
        for question in parse_request(request_data).questions:
            inputs = engine.preprocess(messages(request_data["state"], question))
            # Independent oracle: full native model, including the full vocabulary head.
            output = model(**inputs)
            logits = output.logits[0, -1, 65:65 + len(question.options)].float().tolist()
            actual = result["questions"][question.id]
            assert actual["option_logits"] == pytest.approx(logits, abs=1e-6, rel=0)
            expected = make_answer(question, logits)
            if question.type == "noul":
                assert actual["answer"]["noul"] == pytest.approx(expected["noul"], abs=1e-6)
            else:
                assert actual["answer"]["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-6)


def test_real_backbones_question_order_and_repetition(bound_model, request_data):
    _, engine = bound_model
    first = engine.decide(request_data)
    reverse = {**request_data, "questions": dict(reversed(list(request_data["questions"].items())))}
    assert engine.decide(reverse)["questions"] == first["questions"]
    engine.decide({**request_data, "state": "different context"})
    assert engine.decide(request_data) == first


def test_shortcuts_match_on_real_model_classes(bound_model, request_data):
    _, engine = bound_model
    full = engine.decide(request_data)
    for key, question in request_data["questions"].items():
        args = [request_data["state"], question["instructions"]]
        if question["type"] != "noul":
            args.append(question["criteria"])
        assert getattr(engine, question["type"])(*args) == full["questions"][key]["answer"]


def test_factory_with_custom_preprocess_matches_explicit_binding(bound_model, request_data):
    model, explicit = bound_model
    simple = SystemOne.from_transformers(model, explicit.tokenizer, preprocess=explicit.preprocess)
    assert simple.decide(request_data) == explicit.decide(request_data)
    assert simple.backbone is model.base_model
    assert simple.head is model.get_output_embeddings()


def test_generic_text_factory_against_native_model(bound_model, request_data):
    model, explicit = bound_model
    tokenizer = explicit.tokenizer
    tokenizer.chat_template = "byte-test-template"
    simple = SystemOne.from_transformers(model, tokenizer)
    actual = simple.decide(request_data)
    with torch.inference_mode():
        for question in parse_request(request_data).questions:
            prompt = tokenizer.apply_chat_template(messages(request_data["state"], question))
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            expected = model(**inputs, use_cache=False, return_dict=True).logits[0, -1, 65:65 + len(question.options)]
            assert actual["questions"][question.id]["option_logits"] == pytest.approx(expected.tolist(), abs=1e-6)


def test_factory_requires_template_unless_preprocessor_supplied(bound_model):
    model, explicit = bound_model
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="no chat template"):
        SystemOne.from_transformers(model, tokenizer)
    assert SystemOne.from_transformers(model, tokenizer, preprocess=explicit.preprocess)
