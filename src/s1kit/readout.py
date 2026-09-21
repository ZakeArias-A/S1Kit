"""Convert finite candidate logits into typed, explicitly uncalibrated answers.

Confidence is a distribution statistic, not a probability of correctness.
Formulas follow the public TypeSafe adapter at commit
adffc2eab300a4fa3c0e92252d4ffd6ceaa53700, _utils/confidence_metrics.py.
Temperature must be fitted on held-out data before claiming calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from .schema import Question


def validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool):
        raise ValueError("temperature must be a positive finite number")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be a positive finite number") from exc
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number")
    return temperature


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    temperature = validate_temperature(temperature)
    try:
        values = [float(value) for value in logits]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("logits must be finite numbers") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("logits must be a nonempty sequence of finite numbers")
    peak = max(values)
    # Subtract before division to avoid positive overflow at tiny temperatures.
    weights = [math.exp((value - peak) / temperature) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def make_answer(question: Question, logits: Sequence[float], temperature: float = 1.0) -> dict:
    """Return a primitive answer without generating or parsing model text.

    Ties choose the first option in request order. Score is the expected index,
    not the most likely level and not a normalized value on [0, 1].
    """
    count = len(question.options)
    if question.type not in ("choice", "score", "noul"):
        raise ValueError("unsupported question type")
    if count < 2 or len({option.key for option in question.options}) != count:
        raise ValueError("question must contain at least two uniquely keyed options")
    if question.type == "noul" and tuple(option.key for option in question.options) != ("false", "true"):
        raise ValueError("noul options must be ordered false, true")
    if question.type == "score" and tuple(option.key for option in question.options) != tuple(map(str, range(count))):
        raise ValueError("score option keys must be consecutive indices starting at zero")
    probabilities = _softmax(logits, temperature)
    if len(probabilities) != count:
        raise ValueError("number of logits must equal number of options")
    if question.type == "noul":
        return {"noul": probabilities[1]}
    probability_map = {option.key: probability for option, probability in zip(question.options, probabilities)}
    mode = max(range(count), key=probabilities.__getitem__)
    if question.type == "choice":
        uniform = 1.0 / count
        confidence = (probabilities[mode] - uniform) / (1.0 - uniform)
        return {
            "choice": question.options[mode].key,
            "probabilities": probability_map,
            "confidence": min(1.0, max(0.0, confidence)),
        }
    expected_score = math.fsum(index * probability for index, probability in enumerate(probabilities))
    distance_from_mode = math.fsum(probability * abs(index - mode) for index, probability in enumerate(probabilities))
    uniform_center = (count - 1) / 2
    uniform_deviation = math.fsum(abs(index - uniform_center) for index in range(count)) / count
    confidence = max(0.0, 1.0 - distance_from_mode / uniform_deviation)
    return {
        "score": expected_score,
        "legend": {option.key: option.description for option in question.options},
        "probabilities": probability_map,
        "confidence": confidence,
    }
