import json
from types import SimpleNamespace

from PIL import Image
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from examples import gpt2, qwen, text
from conftest import ByteTokenizer


@pytest.mark.parametrize("example,customize", [(text, False), (gpt2, False), (text, True)])
def test_text_examples_use_local_model(monkeypatch, tmp_path, capsys, example, customize):
    import transformers

    tokenizer = ByteTokenizer()
    tokenizer.chat_template = "test"
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=256, n_positions=1024, n_embd=16, n_layer=1, n_head=2,
    ))
    calls = []

    def load_model(path, **kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    model_class = transformers.AutoModelForCausalLM if example is text else transformers.GPT2LMHeadModel
    monkeypatch.setattr(model_class, "from_pretrained", load_model)
    monkeypatch.setattr("sys.argv", [example.__file__, str(tmp_path)] + (["--customize"] if customize else []))
    example.main()
    answer = json.loads(capsys.readouterr().out)
    if customize:
        assert list(answer["questions"]) == ["intent", "severity", "workaround"]
    else:
        assert ("choice" if example is text else "noul") in answer
    assert calls[0]["local_files_only"] is True


@pytest.mark.parametrize("example", [text, gpt2])
def test_model_id_requires_revision(monkeypatch, example):
    monkeypatch.setattr("sys.argv", [example.__file__, "owner/model"])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2


@pytest.mark.parametrize("with_image", [False, True])
def test_qwen_selects_matching_default_request(monkeypatch, tmp_path, capsys, with_image):
    import transformers

    processor = SimpleNamespace(image_processor=SimpleNamespace(size={"shortest_edge": 1024}))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    monkeypatch.setattr(transformers.Qwen3_5ForConditionalGeneration, "from_pretrained", lambda *args, **kwargs: object())
    seen = []

    def decide(request, *, images):
        seen.append((request, images))
        return {"questions": list(request["questions"])}

    monkeypatch.setattr(qwen, "bind_qwen", lambda *args, **kwargs: SimpleNamespace(decide=decide))
    argv = [qwen.__file__, "--device", "cpu", "--quantization", "none"]
    if with_image:
        path = tmp_path / "red.png"
        Image.new("RGB", (16, 16), "red").save(path)
        argv += ["--image", str(path)]
    monkeypatch.setattr("sys.argv", argv)
    qwen.main()
    result = json.loads(capsys.readouterr().out)
    assert result["questions"] == (["color"] if with_image else ["intent", "workaround", "severity"])
    assert len(seen[0][1]) == int(with_image)
