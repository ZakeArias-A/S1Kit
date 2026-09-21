"""Validate typed decisions; model loading and media belong to the caller."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any


@dataclass(frozen=True)
class Option:
    key: str
    description: str


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    instructions: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class DecisionRequest:
    state: Any
    questions: tuple[Question, ...]


def _validate_json(value: Any, path: str, ancestors: set[int] | None = None) -> None:
    """Reject Python-only values, non-string object keys, NaN and cycles."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{path} must contain only JSON values")
    ancestors = set() if ancestors is None else ancestors
    if id(value) in ancestors:
        raise ValueError(f"{path} must not contain a circular reference")
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{path} object keys must be strings")
                _validate_json(child, f"{path}.{key}", ancestors)
        else:
            for index, child in enumerate(value):
                _validate_json(child, f"{path}[{index}]", ancestors)
    finally:
        ancestors.remove(id(value))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _unknown_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown fields: {', '.join(sorted(unknown))}")


def _parse_question(question_id: str, payload: Any) -> Question:
    path = f"questions.{question_id}"
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    _unknown_fields(payload, {"type", "instructions", "criteria"}, path)
    question_type = payload.get("type")
    if question_type not in ("choice", "noul", "score"):
        raise ValueError(f"{path}.type must be choice, noul or score")
    instructions = payload.get("instructions")
    if instructions is None or instructions == [] or instructions == {} or (
        isinstance(instructions, str) and not instructions.strip()
    ):
        raise ValueError(f"{path}.instructions must be a nonempty JSON value")
    criteria = payload.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"{path}.criteria must be an object with 2 to 255 choices")
        options = tuple(
            Option(
                _nonempty_string(key, f"{path}.criteria key"),
                key if description is None else _text(description),
            )
            for key, description in criteria.items()
        )
    elif question_type == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"{path}.criteria must be a list of 2 to 10 ordered labels")
        options = []
        for index, label in enumerate(criteria):
            if not isinstance(label, (str, dict, list)) or not label or (isinstance(label, str) and not label.strip()):
                raise ValueError(f"{path}.criteria[{index}] must be a nonempty string, object or array")
            options.append(Option(str(index), _text(label)))
        options = tuple(options)
    else:
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise ValueError(f"{path}.criteria must be an object with true/false descriptions")
        _unknown_fields(criteria, {"false", "true"}, f"{path}.criteria")
        options = tuple(
            Option(key, key if criteria.get(key) is None else _text(criteria[key]))
            for key in ("false", "true")
        )
    return Question(question_id, question_type, _text(instructions), options)


def parse_request(payload: dict[str, Any]) -> DecisionRequest:
    """Validate a decoded JSON request, preserving question and choice order.

    Non-string instructions and choice/noul descriptions are rendered as compact
    sorted JSON. Score criteria are ordered strings, objects or arrays. Media is passed to
    the caller-provided preprocessor separately, not serialized into the state.
    """
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    _validate_json(payload, "request")
    _unknown_fields(payload, {"state", "questions"}, "request")
    if "state" not in payload:
        raise ValueError("request.state is required")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("request.questions must be a nonempty object")
    parsed_questions = tuple(
        _parse_question(_nonempty_string(key, "question id"), value)
        for key, value in questions.items()
    )
    return DecisionRequest(payload["state"], parsed_questions)
