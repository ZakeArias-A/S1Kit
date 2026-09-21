from copy import deepcopy

import pytest

from s1kit.schema import parse_request


def test_order_and_json_descriptions(request_data):
    q = request_data["questions"]["intent"]
    q["instructions"] = {"z": ["中文"], "a": False}
    q["criteria"] = {"z": None, "a": {"z": 1, "a": 2}}
    parsed = parse_request(request_data)
    assert [q.id for q in parsed.questions] == ["intent", "workaround", "severity"]
    assert parsed.questions[0].instructions == '{"a":false,"z":["中文"]}'
    assert [(o.key, o.description) for o in parsed.questions[0].options] == [
        ("z", "z"), ("a", '{"a":2,"z":1}'),
    ]
    assert [o.key for o in parsed.questions[1].options] == ["false", "true"]
    assert [o.key for o in parsed.questions[2].options] == ["0", "1", "2"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), (1, 2), {1: "key"}])
def test_invalid_state(request_data, value):
    request_data["state"] = value
    with pytest.raises(ValueError):
        parse_request(request_data)


def test_cycle(request_data):
    cycle = []
    cycle.append(cycle)
    request_data["state"] = cycle
    with pytest.raises(ValueError, match="circular"):
        parse_request(request_data)


@pytest.mark.parametrize("instructions", [None, "", "  ", [], {}])
def test_empty_instructions(request_data, instructions):
    request_data["questions"]["intent"]["instructions"] = instructions
    with pytest.raises(ValueError, match="instructions"):
        parse_request(request_data)


@pytest.mark.parametrize("count,valid", [(1, False), (2, True), (26, True), (27, True), (255, True), (256, False)])
def test_choice_limit(request_data, count, valid):
    request_data["questions"]["intent"]["criteria"] = {str(i): None for i in range(count)}
    if valid:
        assert len(parse_request(request_data).questions[0].options) == count
    else:
        with pytest.raises(ValueError):
            parse_request(request_data)


@pytest.mark.parametrize("criteria", [["one"], ["x"] * 11, ["x", ""], ["x", 1], {"a": "b"}])
def test_score_limit(request_data, criteria):
    request_data["questions"]["severity"]["criteria"] = criteria
    with pytest.raises(ValueError):
        parse_request(request_data)


@pytest.mark.parametrize("change", [
    lambda r: r.update(model="unused"),
    lambda r: r.update(images=["pass to preprocess separately"]),
    lambda r: r.pop("state"),
    lambda r: r.update(questions={}),
    lambda r: r["questions"]["intent"].update(instruction="typo"),
    lambda r: r["questions"]["intent"].update(type="unknown"),
    lambda r: r["questions"]["workaround"].update(criteria={"yes": "typo"}),
])
def test_bad_contract(request_data, change):
    payload = deepcopy(request_data)
    change(payload)
    with pytest.raises(ValueError):
        parse_request(payload)
