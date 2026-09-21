"""Independent causal-model decisions using caller-owned PyTorch modules."""

from collections.abc import Callable, Mapping
import hashlib
import operator

import torch
from torch import nn
from torch.nn import functional as F

from .prompts import SYSTEM, PROMPT_VERSION, answer_labels, canonical_json, label_tokens, messages
from .readout import make_answer, validate_temperature
from .schema import parse_request


class SystemOne:
    """Bind preprocessing, a causal backbone, and its native dense output head.

    ``preprocess(messages, *, answer='', **media)`` returns model kwargs with
    unpadded ``input_ids`` of shape [1, sequence]. It renders the chat template
    and appends ``answer`` before encoding, without truncation. The callback
    handles devices, multimodal positions, and model-specific forward options.
    ``hidden(output)`` must return [1, sequence, hidden_size].

    The supplied modules are put in eval mode. Calls are serial and inference
    only; the caller must not share mutable model state with concurrent work.
    """

    def __init__(self, *, preprocess: Callable, backbone: nn.Module, head: nn.Linear,
                 tokenizer, hidden: Callable = operator.attrgetter("last_hidden_state"),
                 max_tokens: int = 4096, system_prompt: str = SYSTEM, labels=None):
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        # Selected weight rows bypass custom forward logic, adapters and quantizers.
        # Restrict this first implementation to an ordinary, dense linear head.
        if type(head) is not nn.Linear or type(head.weight) is not nn.Parameter:
            raise TypeError("head must be a native dense torch.nn.Linear without custom forward logic")
        if not callable(preprocess) or not callable(hidden):
            raise TypeError("preprocess and hidden must be callable")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.preprocess = preprocess
        self.backbone = backbone.eval()
        self.head = head.eval()
        self.tokenizer = tokenizer
        self.hidden = hidden
        self.max_tokens = max_tokens
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        self.system_prompt = system_prompt
        if labels is not None:
            answer_labels(2, labels)
        self.labels = None if labels is None else tuple(labels)

    @classmethod
    def from_transformers(cls, model, tokenizer, *, preprocess=None, backbone=None,
                          template_kwargs=None, max_tokens=4096, system_prompt=SYSTEM, labels=None):
        """Bind a loaded causal LM using Transformers' public model interfaces.

        The default preprocessor is text-only and requires a chat template.
        For multimodal or model-specific positions, supply ``preprocess``.
        Models with nonstandard outputs/heads can use the explicit constructor.
        No model names, module-name searches or modality detection are used.
        """
        if preprocess is not None and template_kwargs is not None:
            raise ValueError("With custom preprocess, configure its template there instead of template_kwargs")
        if backbone is None:
            backbone = model.base_model
            if backbone is model:
                raise ValueError("No separate backbone exposed; pass backbone explicitly or use SystemOne(...)")
        if preprocess is None:
            if not getattr(tokenizer, "chat_template", None):
                raise ValueError("Tokenizer has no chat template; supply preprocess or configure tokenizer.chat_template")
            options = {"enable_thinking": False, **(template_kwargs or {})}
            reserved = {"tokenize", "add_generation_prompt", "return_tensors", "return_dict"} & options.keys()
            if reserved:
                raise ValueError(f"Reserved template options: {', '.join(sorted(reserved))}")
            device = model.get_input_embeddings().weight.device

            def preprocess(messages, *, answer=""):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **options,
                )
                inputs = tokenizer(prompt + answer, add_special_tokens=False, return_tensors="pt")
                return {**inputs.to(device), "use_cache": False, "return_dict": True}

        return cls(preprocess=preprocess, backbone=backbone, head=model.get_output_embeddings(),
                   tokenizer=tokenizer, max_tokens=max_tokens, system_prompt=system_prompt, labels=labels)

    def choice(self, state, instructions, options: dict, *, temperature=1.0, system_prompt=None, **media) -> dict:
        """Choose from an ordered key-to-description mapping; return the answer dict."""
        return self._ask(state, "choice", instructions, options, temperature, system_prompt, media)

    def score(self, state, instructions, levels: list, *, temperature=1.0, system_prompt=None, **media) -> dict:
        """Return the expected zero-based level index, probabilities and confidence."""
        return self._ask(state, "score", instructions, levels, temperature, system_prompt, media)

    def noul(self, state, instructions, *, criteria=None, temperature=1.0, system_prompt=None, **media) -> dict:
        """Return {'noul': true_probability}; criteria may describe false and true."""
        return self._ask(state, "noul", instructions, criteria, temperature, system_prompt, media)

    def _ask(self, state, kind, instructions, criteria, temperature, system_prompt, media):
        request = {"state": state, "questions": {
            "result": {"type": kind, "instructions": instructions, "criteria": criteria},
        }}
        return self.decide(request, temperature=temperature, system_prompt=system_prompt, **media)["questions"]["result"]["answer"]

    @staticmethod
    def _input_ids(inputs):
        if not isinstance(inputs, Mapping):
            raise TypeError("preprocess must return a mapping of model kwargs")
        ids = inputs.get("input_ids")
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1
                or ids.shape[1] == 0 or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("input_ids must be a nonempty integer tensor of shape [1, sequence]")
        mask = inputs.get("attention_mask")
        if mask is not None and (
            not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or not (mask == 1).all().item()
        ):
            raise ValueError("Use one unpadded sequence with an all-ones attention_mask")
        if inputs.get("past_key_values") is not None:
            raise ValueError("Independent decisions do not accept past_key_values")
        return ids[0].tolist()

    def decide(self, request: dict, *, temperature: float = 1.0, system_prompt=None, **media) -> dict:
        """Score each question separately; media kwargs go only to the preprocessor.

        Preprocessing runs once for the input and once per legal label to verify
        the actual answer boundary, including image-token expansion. Only the
        original input is forwarded through the model; probes never run inference.
        """
        parsed = parse_request(request)
        temperature = validate_temperature(temperature)
        system_prompt = self.system_prompt if system_prompt is None else system_prompt
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if "answer" in media or "messages" in media:
            raise ValueError("answer and messages are reserved preprocessing arguments")
        records = {}
        with torch.inference_mode():
            for question in parsed.questions:
                labels = answer_labels(len(question.options), self.labels)
                slots = label_tokens(self.tokenizer, len(question.options), labels)
                if min(slots) < 0 or max(slots) >= self.head.out_features:
                    raise ValueError("Answer label token is outside the output head vocabulary")
                def conversation():
                    return messages(parsed.state, question, system_prompt=system_prompt, labels=self.labels)

                inputs = self.preprocess(conversation(), answer="", **media)
                ids = self._input_ids(inputs)
                if len(ids) > self.max_tokens:
                    raise ValueError(f"Input has {len(ids)} tokens; limit {self.max_tokens}; no truncation")
                for letter, slot in zip(labels, slots):
                    probe = self.preprocess(conversation(), answer=letter, **media)
                    if self._input_ids(probe) != ids + [slot]:
                        raise ValueError(f"Preprocessing changed the answer boundary for {letter!r}")
                    del probe
                if self._input_ids(inputs) != ids:
                    raise ValueError("preprocess mutated an earlier input; return fresh inputs for each call")
                output = self.backbone(**inputs)
                hidden = self.hidden(output)
                if (not isinstance(hidden, torch.Tensor)
                        or tuple(hidden.shape) != (1, len(ids), self.head.in_features)):
                    raise ValueError("hidden(output) must return [1, input_length, head.in_features]")
                index = torch.tensor(slots, device=self.head.weight.device)
                weight = self.head.weight.index_select(0, index)
                bias = self.head.bias
                bias = None if bias is None else bias.index_select(0, index)
                logits = F.linear(hidden[:, -1, :].to(weight), weight, bias)[0].float().cpu().tolist()
                records[question.id] = {
                    "answer": make_answer(question, logits, temperature),
                    "option_logits": logits,
                    "input_tokens": len(ids),
                    # Token fingerprint only, not a multimodal cache identity.
                    "input_ids_sha256": hashlib.sha256(canonical_json(ids).encode()).hexdigest(),
                }
                del output, hidden, inputs
        return {
            "schema_version": "s1kit-result-v1",
            "prompt_version": PROMPT_VERSION if system_prompt == SYSTEM and self.labels is None
                              and all(len(q.options) <= 26 for q in parsed.questions) else "typed-label-v2-custom",
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "usage": {"input_tokens": sum(r["input_tokens"] for r in records.values()), "output_tokens": 0},
            "mode": "independent", "temperature": temperature,
            "probability_status": "uncalibrated scores conditional on the declared options",
            "questions": records,
        }
