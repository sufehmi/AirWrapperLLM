#!/usr/bin/env bash
# ── AirWrapperLLM curl examples ──────────────────────────────────────
# Run the server first, copy the API key from its stdout, then:
#   API_KEY=air-XXXXX ./examples/curl_examples.sh

set -euo pipefail
API_KEY="${API_KEY:-air-CHANGEME}"
BASE="${BASE_URL:-http://localhost:20002}"

echo
echo "=== 1. List models ==="
curl -s "$BASE/v1/models" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

echo
echo "=== 2. Health (no auth) ==="
curl -s "$BASE/health" | python3 -m json.tool

echo
echo "=== 3. Chat with thinking ==="
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role":"user","content":"What is 17 * 23?"}],
    "max_tokens": 200,
    "temperature": 0.6,
    "reasoning": {"effort": "auto"}
  }' | python3 -m json.tool

echo
echo "=== 4. Multi-turn ==="
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [
      {"role":"user","content":"My name is Alice. Remember it."},
      {"role":"assistant","content":"Nice to meet you, Alice!"},
      {"role":"user","content":"What is my name?"}
    ],
    "max_tokens": 50
  }' | python3 -m json.tool

echo
echo "=== 5. Tool calling ==="
curl -s "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role":"user","content":"Weather in Tokyo?"}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "description":"Get current weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    }}],
    "max_tokens": 300
  }' | python3 -m json.tool

echo
echo "=== 6. Streaming ==="
curl -s -N "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role":"user","content":"Count to 5."}],
    "max_tokens": 50,
    "stream": true
  }'