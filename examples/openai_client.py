"""Minimal OpenAI SDK client for AirWrapperLLM.

Usage:
    pip install openai
    AIRWRAPPER_BASE_URL=http://localhost:20002/v1 \\
    AIRWRAPPER_API_KEY=air-XXXXXXXXXXXXXXXXXXXXXXXXXXXX \\
    python examples/openai_client.py
"""
import os

from openai import OpenAI

base_url = os.environ.get("AIRWRAPPER_BASE_URL", "http://localhost:20002/v1")
api_key = os.environ.get("AIRWRAPPER_API_KEY", "")
if not api_key:
    raise SystemExit("Set AIRWRAPPER_API_KEY environment variable")

client = OpenAI(base_url=base_url, api_key=api_key)

# ── 1. Plain chat ────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[
        {"role": "user", "content": "What is 17 * 23? Think step by step."}
    ],
    max_tokens=300,
    temperature=0.6,
    extra_body={"reasoning": {"effort": "auto"}},
)
print("=== Reasoning ===")
print(resp.choices[0].message.reasoning)
print("\n=== Answer ===")
print(resp.choices[0].message.content)

# ── 2. Multi-turn ────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[
        {"role": "user", "content": "My name is Alice. Remember it."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ],
    max_tokens=100,
    temperature=0.6,
)
print("\n=== Multi-turn answer ===")
print(resp.choices[0].message.content)

# ── 3. Tool calling ──────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    max_tokens=300,
)
print("\n=== Tool calls ===")
for tc in resp.choices[0].message.tool_calls or []:
    print(f"  {tc.function.name}({tc.function.arguments})")