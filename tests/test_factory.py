from types import SimpleNamespace

import pytest
from torch import nn

from s1kit import SystemOne
from conftest import ByteTokenizer


class Model(nn.Module):
    """An arbitrary module layout exposing only the standard HF accessors."""

    def __init__(self):
        super().__init__()
        self.arbitrary_name = nn.Embedding(256, 8)
        self.output_layer = nn.Linear(8, 256)
        self.calls = 0

    @property
    def base_model(self):
        return self  # Explicit backbone is required in this particular fixture.

    def get_input_embeddings(self):
        return self.arbitrary_name

    def get_output_embeddings(self):
        return self.output_layer

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        assert not use_cache and return_dict
        return SimpleNamespace(last_hidden_state=self.arbitrary_name(input_ids))


class ChatTokenizer(ByteTokenizer):
    chat_template = "test"

    def apply_chat_template(self, messages, **kwargs):
        self.template_options = kwargs
        return super().apply_chat_template(messages, **kwargs)


def test_explicit_backbone_override_and_template_options():
    model = Model()
    tokenizer = ChatTokenizer()
    engine = SystemOne.from_transformers(
        model, tokenizer, backbone=model, template_kwargs={"enable_thinking": False, "custom": 42},
    )
    assert engine.backbone is model
    engine.noul("state", "Question?")
    assert tokenizer.template_options == {
        "tokenize": False, "add_generation_prompt": True, "enable_thinking": False, "custom": 42,
    }
    assert model.calls == 1


def test_backbone_is_not_guessed_from_module_names():
    with pytest.raises(ValueError, match="No separate backbone"):
        SystemOne.from_transformers(Model(), ChatTokenizer())


@pytest.mark.parametrize("name", ["tokenize", "add_generation_prompt", "return_tensors", "return_dict"])
def test_reserved_template_options_rejected(name):
    model = Model()
    with pytest.raises(ValueError, match="Reserved template options"):
        SystemOne.from_transformers(model, ChatTokenizer(), backbone=model, template_kwargs={name: False})


def test_custom_preprocess_does_not_silently_ignore_template_options():
    model = Model()
    with pytest.raises(ValueError, match="custom preprocess"):
        SystemOne.from_transformers(model, ChatTokenizer(), backbone=model,
                                    preprocess=lambda messages: {}, template_kwargs={})


def test_default_text_preprocess_rejects_media():
    model = Model()
    engine = SystemOne.from_transformers(model, ChatTokenizer(), backbone=model)
    with pytest.raises(TypeError, match="images"):
        engine.noul("state", "Question?", images=[object()])
    assert model.calls == 0


def test_factory_keeps_input_budget():
    model = Model()
    engine = SystemOne.from_transformers(model, ChatTokenizer(), backbone=model, max_tokens=1)
    with pytest.raises(ValueError, match="limit 1"):
        engine.noul("state", "Question?")
    assert model.calls == 0


def test_factory_accepts_prompt_and_label_configuration():
    model = Model()
    engine = SystemOne.from_transformers(model, ChatTokenizer(), backbone=model,
                                         system_prompt="Reply with an option label.", labels=["X", "Y"])
    assert engine.system_prompt == "Reply with an option label."
    assert engine.labels == ("X", "Y")
    assert "noul" in engine.noul({"nested": ["evidence"]}, {"question": "Is it present?"})
