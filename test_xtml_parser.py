"""Unit tests for the AirWrapperLLM XTML output parser.

Run with:  python3 test_xtml_parser.py

Uses only the standard library. Verifies the parser logic in
airwrapper_xtml.py independent of any ML framework or model.
"""
import sys
sys.path.insert(0, ".")
from airwrapper_xtml import parse_xtml_output

O = "<|open|>"
C = "<|close|>"
S = "<|sep|>"


def check(name, raw, exp_reasoning=None, exp_content=None, exp_tools=None):
    r = parse_xtml_output(raw)
    ok = True
    if exp_reasoning is not None:
        ok = ok and (r["reasoning"] == exp_reasoning), f"[{name}] reasoning mismatch"
    if exp_content is not None:
        ok = ok and (r["content"] == exp_content), f"[{name}] content mismatch"
    if exp_tools is not None:
        got = [(t["function"]["name"], t["function"]["arguments"]) for t in r["tool_calls"]]
        ok = ok and (got == exp_tools), f"[{name}] tools mismatch"
    print(("PASS" if ok else "FAIL"), name)
    return ok


# Test 1: thinking + response
check(
    "think_and_response",
    f"{O}think{S}Let me compute 2+2.{C}think{S}{O}response{S}The answer is 4.{C}response{S}",
    exp_reasoning="Let me compute 2+2.",
    exp_content="The answer is 4.",
)

# Test 2: response only (no thinking)
check(
    "response_only",
    f"{O}response{S}Hello world{C}response{S}",
    exp_content="Hello world",
)

# Test 3: tool call
check(
    "tool_call",
    f"{O}response{S}{C}response{S}{O}tools{S}{O}call tool=\"get_weather\" index=\"1\"{S}{O}json type=\"object\"{S}{{\"city\": \"Tokyo\"}}{C}json{S}{C}call{S}{C}tools{S}",
    exp_content="",
    exp_tools=[("get_weather", '{"city": "Tokyo"}')],
)

# Test 4: tool call with multi-arg JSON
check(
    "tool_call_multiargs",
    f"{O}response{S}Sure.{C}response{S}{O}tools{S}{O}call tool=\"search\" index=\"1\"{S}{O}json type=\"object\"{S}{{\"query\": \"cats\", \"limit\": 5}}{C}json{S}{C}call{S}{C}tools{S}",
    exp_content="Sure.",
    exp_tools=[("search", '{"query": "cats", "limit": 5}')],
)

# Test 5: plain text fallback (no structural tags)
check(
    "plain_text_fallback",
    "Just some plain text without tags.",
    exp_content="Just some plain text without tags.",
)

# Test 6: truncated (think opened but not closed)
check(
    "truncated_thinking",
    f"{O}think{S}Half a thought...",
    exp_reasoning="Half a thought...",
)

# Test 7: two tool calls
check(
    "two_tool_calls",
    f"{O}tools{S}{O}call tool=\"a\" index=\"1\"{S}{O}json type=\"object\"{S}{{\"x\": 1}}{C}json{S}{C}call{S}{O}call tool=\"b\" index=\"2\"{S}{O}json type=\"object\"{S}{{\"y\": 2}}{C}json{S}{C}call{S}{C}tools{S}",
    exp_tools=[("a", '{"x": 1}'), ("b", '{"y": 2}')],
)

print("\nAll tests done.")