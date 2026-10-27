from types import SimpleNamespace as NS

from llm_classifier import TOOL_NAME, parse_response

USAGE = NS(input_tokens=100, output_tokens=20, cache_creation_input_tokens=None, cache_read_input_tokens=50)


def resp(content, stop_reason="tool_use", stop_details=None):
    return NS(content=content, stop_reason=stop_reason, stop_details=stop_details, usage=USAGE)


def tool_use(inp, name=TOOL_NAME):
    return NS(type="tool_use", name=name, input=inp)


def test_ok_after_thinking_block():
    r = parse_response(resp([NS(type="thinking", thinking=""),
                             tool_use({"label": "phishing", "phishing_probability": 87})]))
    assert r["status"] == "ok" and r["label"] == "phishing" and r["prob"] == 87
    assert r["input_tokens"] == 100 and r["cache_read_input_tokens"] == 50
    assert r["cache_creation_input_tokens"] == 0          # None from the API becomes 0


def test_probability_as_numeric_string_is_accepted():
    r = parse_response(resp([tool_use({"label": "legitimate", "phishing_probability": "12.6"})]))
    assert r["status"] == "ok" and r["prob"] == 13


def test_refusal_records_category():
    r = parse_response(resp([], stop_reason="refusal", stop_details=NS(category="cyber")))
    assert r["status"] == "refusal" and r["refusal_category"] == "cyber"


def test_missing_tool_call():
    r = parse_response(resp([NS(type="text", text="It is phishing.")], stop_reason="end_turn"))
    assert r["status"] == "unparseable" and r["stop_reason"] == "end_turn"


def test_wrong_tool_name_is_missing():
    r = parse_response(resp([tool_use({"label": "phishing", "phishing_probability": 90}, name="other")]))
    assert r["status"] == "unparseable"


def test_bad_label():
    r = parse_response(resp([tool_use({"label": "spam", "phishing_probability": 90})]))
    assert r["status"] == "unparseable"


def test_probability_out_of_range_or_invalid():
    for prob in (101, -1, "high", None, True, float("nan")):
        r = parse_response(resp([tool_use({"label": "phishing", "phishing_probability": prob})]))
        assert r["status"] == "unparseable", prob
