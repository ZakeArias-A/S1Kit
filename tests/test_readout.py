"""Numerical stability and public answer semantics, without model dependencies."""

import math

import pytest

from s1kit.readout import make_answer
from s1kit.schema import Option, Question, parse_request


def question(kind, criteria=None):
    data = {"type": kind, "instructions": "Decide"}
    if criteria is not None:
        data["criteria"] = criteria
    return parse_request({"state": "", "questions": {"q": data}}).questions[0]


def test_choice_probabilities_and_confidence_use_original_keys():
    answer = make_answer(question("choice", {"accept": None, "review": None}), [math.log(.82), math.log(.18)])
    assert answer["choice"] == "accept"
    assert answer["probabilities"] == pytest.approx({"accept": .82, "review": .18})
    assert answer["confidence"] == pytest.approx(.64)


def test_uniform_distribution_has_zero_confidence_and_stable_tie():
    answer = make_answer(question("choice", {"z": None, "a": None, "m": None}), [99, 99, 99])
    assert answer["choice"] == "z"
    assert answer["confidence"] == pytest.approx(0)
    assert sum(answer["probabilities"].values()) == pytest.approx(1)


def test_noul_is_true_probability_without_extra_confidence():
    answer = make_answer(question("noul"), [math.log(.2), math.log(.8)])
    assert answer == pytest.approx({"noul": .8})


def test_score_uses_expected_index_and_modal_distance_confidence():
    labels = ["none", "slight", "moderate", "high", "extreme"]
    probabilities = [.01, .02, .07, .3, .6]
    answer = make_answer(question("score", labels), [math.log(value) for value in probabilities])
    assert answer["score"] == pytest.approx(3.46)
    assert answer["confidence"] == pytest.approx(.55)
    assert answer["legend"] == dict(zip(map(str, range(5)), labels))
    assert answer["probabilities"] == pytest.approx(dict(zip(map(str, range(5)), probabilities)))


def test_score_uniform_confidence_and_expected_midpoint():
    answer = make_answer(question("score", ["low", "medium", "high"]), [0, 0, 0])
    assert answer["score"] == pytest.approx(1)
    assert answer["confidence"] == pytest.approx(0)


@pytest.mark.parametrize("logits", [[10000, 9999], [-10000, -10001], [1e308, -1e308]])
def test_extreme_finite_logits_produce_valid_distribution(logits):
    answer = make_answer(question("choice", {"a": None, "b": None}), logits)
    assert answer["choice"] == "a"
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in answer["probabilities"].values())
    assert sum(answer["probabilities"].values()) == pytest.approx(1)


def test_temperature_changes_concentration_not_order():
    q = question("choice", {"a": None, "b": None})
    cold = make_answer(q, [2, 0], temperature=.5)
    warm = make_answer(q, [2, 0], temperature=2)
    assert cold["choice"] == warm["choice"] == "a"
    assert cold["confidence"] > warm["confidence"]
    assert make_answer(q, [2, 0], temperature=1e-300)["probabilities"] == {"a": 1, "b": 0}


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), None, True])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        make_answer(question("noul"), [0, 1], temperature)


@pytest.mark.parametrize("logits", [[], [0], [0, 1, 2], [0, float("nan")], [float("inf"), 0], [None, 0]])
def test_invalid_logits_rejected(logits):
    with pytest.raises(ValueError, match="logits"):
        make_answer(question("noul"), logits)


@pytest.mark.parametrize("invalid", [
    Question("q", "noul", "Decide", (Option("true", "yes"), Option("false", "no"))),
    Question("q", "score", "Rate", (Option("1", "low"), Option("2", "high"))),
    Question("q", "choice", "Choose", (Option("a", "first"), Option("a", "second"))),
])
def test_manually_constructed_invalid_questions_rejected(invalid):
    with pytest.raises(ValueError):
        make_answer(invalid, [0, 1])
