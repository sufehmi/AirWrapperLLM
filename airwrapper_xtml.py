"""AirWrapperLLM XTML output parser — pure Python, no ML dependencies.

Parses the Kimi K3 XTML structural-token output format into structured fields:
  - reasoning (the <think> channel)
  - content   (the <response> channel)
  - tool_calls (the <tools>/<call> channel)

XTML uses special tokens:
  <|open|> ... <|sep|>      opening tag, e.g. <|open|>think<|sep|>
  <|close|>tag<|sep|>       closing tag
  <|end_of_msg|>            message terminator

Channels:
  <|open|>think<|sep|> ... <|close|>think<|sep|>
  <|open|>response<|sep|> ... <|close|>response<|sep|>
  <|open|>tools<|sep|>
    <|open|>call tool="name" index="1"<|sep|>
      <|open|>json type="object"<|sep|>{...}<|close|>json<|sep|>
    <|close|>call<|sep|>
  <|close|>tools<|sep|>

Reusable outside this project: import parse_xtml_output.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

OPEN_TOK = "<|open|>"
CLOSE_TOK = "<|close|>"
SEP_TOK = "<|sep|>"
EOM_TOK = "<|end_of_msg|>"

_TAG_RE = re.compile(
    re.escape(OPEN_TOK) + r"(.*?)" + re.escape(SEP_TOK) + r"|(.*?)" +
    re.escape(CLOSE_TOK) + r"(.*?)" + re.escape(SEP_TOK) + r"|" +
    re.escape(EOM_TOK),
    re.DOTALL,
)


def _tokenize_xtml(text: str) -> list[tuple[str, str | None, str | None]]:
    """Split raw text into a sequence of structural events.

    Returns a list of:
      ("open",  tagname, None)    - opening tag
      ("close", tagname, None)    - closing tag
      ("text",  text,    None)    - literal text content
    """
    events: list[tuple[str, str | None, str | None]] = []
    pos = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > pos:
            events.append(("text", text[pos:m.start()], None))
        if m.group(1) is not None:          # OPEN
            events.append(("open", m.group(1), None))
        elif m.group(2) is not None:        # CLOSE
            events.append(("close", m.group(3), None))
        pos = m.end()
    if pos < len(text):
        events.append(("text", text[pos:], None))
    return events


def _parse_attrs(tagbody: str) -> tuple[str, dict[str, str]]:
    """'call tool="get_weather" index="1"' -> ('call', {'tool':'get_weather','index':'1'})"""
    parts = tagbody.strip().split(None, 1)
    name = parts[0] if parts else ""
    attrs: dict[str, str] = {}
    if len(parts) > 1:
        for am in re.finditer(r'(\w+)="([^"]*)"', parts[1]):
            attrs[am.group(1)] = am.group(2)
    return name, attrs


def _safe_json(s: str, fallback: Any = None) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return fallback


def parse_xtml_output(raw: str) -> dict[str, Any]:
    """Parse a Kimi K3 XTML generation into structured fields.

    Returns ``{reasoning, content, tool_calls, raw}``.
    """
    events = _tokenize_xtml(raw)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    channel: Optional[str] = None        # 'think' | 'response' | None
    in_tools = False
    cur_call: Optional[dict[str, Any]] = None
    json_buf: Optional[str] = None

    for kind, a, _ in events:
        if kind == "text":
            txt = a or ""
            if txt.strip() == "":
                pass
            elif channel == "think":
                reasoning_parts.append(txt)
            elif channel == "response":
                content_parts.append(txt)
            elif in_tools and json_buf is not None:
                json_buf += txt
        elif kind == "open":
            tagname, attrs = _parse_attrs(a or "")
            if tagname == "think":
                channel = "think"
            elif tagname == "response":
                channel = "response"
            elif tagname == "tools":
                in_tools = True
            elif tagname == "call":
                cur_call = {"name": attrs.get("tool", ""), "_index": attrs.get("index", "")}
            elif tagname == "json":
                json_buf = ""
        elif kind == "close":
            tagname = (a or "").strip()
            if tagname == "think":
                channel = None
            elif tagname == "response":
                channel = None
            elif tagname == "tools":
                in_tools = False
            elif tagname == "call":
                if cur_call is not None:
                    tool_calls.append(cur_call)
                    cur_call = None
            elif tagname == "json":
                if cur_call is not None and json_buf is not None:
                    cur_call["arguments"] = _safe_json(json_buf.strip(), fallback=json_buf.strip())
                json_buf = None

    # Build OpenAI-style tool_calls list
    oai_tool_calls = []
    for tc in tool_calls:
        args = tc.get("arguments")
        if isinstance(args, dict):
            args_str = json.dumps(args, ensure_ascii=False)
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)
        oai_tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": args_str,
            },
        })

    reasoning = "".join(reasoning_parts).strip()
    content = "".join(content_parts).strip()
    # If the model never emitted a response channel (truncated), fall back to
    # whatever text exists outside think/tools.
    if not content and not reasoning and not oai_tool_calls:
        leftover = "".join((e[1] or "") for e in events if e[0] == "text").strip()
        content = leftover

    return {
        "reasoning": reasoning,
        "content": content,
        "tool_calls": oai_tool_calls,
        "raw": raw,
    }