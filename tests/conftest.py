import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from s1kit import SystemOne


@pytest.fixture
def request_data():
    path = Path(__file__).resolve().parents[1] / "examples/request.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ByteTokenizer:
    """UTF-8 tokens for structural tests only, not a pretrained vocabulary."""

    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"

    def __call__(self, text, **kwargs):
        from transformers import BatchEncoding
        ids = torch.tensor([self.encode(text)])
        return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 8)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, pixels=None):
        assert not self.training and not torch.is_grad_enabled()
        self.calls.append(input_ids.clone())
        hidden = self.embedding(input_ids).cumsum(dim=1)
        if pixels is not None:
            hidden = hidden + pixels
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def engine():
    tokenizer = ByteTokenizer()

    def preprocess(conversation, *, answer="", image=None):
        ids = tokenizer.encode(tokenizer.apply_chat_template(conversation) + answer)
        if image is not None:
            # Emulate processor expansion of an image placeholder.
            ids = [1, 2, 3] + ids
        inputs = {"input_ids": torch.tensor([ids])}
        if image is not None:
            inputs["pixels"] = image
        return inputs

    with torch.random.fork_rng():
        torch.manual_seed(7)
        return SystemOne(preprocess=preprocess, backbone=Backbone(), head=nn.Linear(8, 256), tokenizer=tokenizer)
