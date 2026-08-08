# AirWrapperLLM

An **OpenAI-compatible FastAPI server that wraps [AirLLM](https://github.com/lyogavin/airllm)**, exposing memory-frugal disk-streaming inference as a standard HTTP API.

AirLLM lets you run models far larger than your VRAM by streaming layers / experts from disk into the GPU one at a time. The catch is that **AirLLM does not ship with a server**. AirWrapperLLM fills that gap.

```
┌─────────────────────────────┐         ┌────────────────────────────────┐
│  OpenAI-style client        │         │  AirWrapperLLM                 │
│  / curl / Python / OpenAI   │  HTTPS  │  ┌──────────────────────────┐  │
│  SDK / etc.                 ├────────►│  │ FastAPI (uvicorn)        │  │
└─────────────────────────────┘         │  └────────────┬─────────────┘  │
                                         │               │                │
                                         │       ┌───────▼────────┐       │
                                         │       │  AirLLM model  │       │
                                         │       │  + XTML parser │       │
                                         │       └───────┬────────┘       │
                                         │               │ disk-stream    │
                                         └───────────────▼────────────────┘
                                                   1.5TB+ weights on NVMe
                                                   <16GB resident on GPU
```

---

## What you get

* `POST /v1/chat/completions` — full OpenAI schema, both `stream=true` and non-streaming
* `GET /v1/models` — model listing
* `GET /health` — liveness check
* **Bearer-token API-key auth** (auto-generated on first start)
* **Multi-turn conversations** — pass message history
* **Thinking / reasoning mode** — model `<think>` trace split into `message.reasoning`
* **Tool / function calling** — OpenAI-format `tools=[...]`, model returns structured `tool_calls`
* **Thinking-effort control** — `chat_template_kwargs={"thinking_effort": "low"|"high"|"max"}`
* Single-process, single-GPU; serializes generation through a global lock (AirLLM is single-tenant by design)

The XTML output parser lives in a dependency-free module (`airwrapper_xtml.py`) with unit tests — easy to reuse elsewhere.

---

## ⚠️ Hardware & speed — read this first

**AirLLM is a memory-frugal streaming engine, not a fast one.** Speed comes from the size of your model vs. your disk bandwidth vs. your GPU, in roughly that order.

| Scenario | Realistic throughput |
|---|---|
| 2.8T-param Kimi K3 on 1× RTX 5060 Ti (16 GB) | **~0.5–1 tok/s** (many seconds per token) |
| 671B DeepSeek-V3 on 1× RTX 5090 (32 GB) | **~2–5 tok/s** |
| 405B Llama 3.1 on 1× A100 (80 GB) | **~5–10 tok/s** |

The point of AirLLM is **"make the impossible model fit,"** not "make it fast." If you need normal-speed inference you want **vLLM + a model that actually fits in your GPU's VRAM** — AirLLM is the fallback for when the model simply won't fit any other way.

Disk space: the model weights must live on a fast NVMe (over ~1 GB/s read). Expect **1–2× the model size on disk** during the initial layer-split transform.

---

## Requirements

* Linux (any recent distro — Ubuntu 22.04/24.04 tested)
* CUDA-capable GPU with **any** VRAM size — AirLLM streams the rest from disk
* Python 3.10+
* ~3× the model size in **disk** (e.g. Kimi K3 needs ~3 TB free during the transform)
* ~16 GB system RAM (more is better — pinned-memory layer prefetch helps)
* Hugging Face account with access to the model you want (most are public)

---

## Installation

```bash
git clone https://github.com/sufehmi/AirWrapperLLM.git
cd AirWrapperLLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned requirements include:
* `airllm` (the streaming engine)
* `torch >= 2.4` with CUDA wheels matching your driver
* `transformers >= 4.56,<4.57` (some AirLLM model variants — e.g. Kimi K3 — need this range)
* `tiktoken` (required by Kimi K3's tokenizer)
* `fla-core` (required by Kimi K3's remote modeling code)
* `compressed-tensors`, `bitsandbytes` (for AirLLM's `compression` mode)
* `fastapi`, `uvicorn` (the server)

If your model uses **flash-attn**, install it with the CUDA toolkit that matches your `torch` CUDA build:
```bash
CUDA_HOME=/usr/local/cuda-12.8 pip install flash-attn --no-build-isolation
```
(Use whatever CUDA major version your `torch` wheel is built against.)

---

## Usage

### 1. Download a model

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8', local_dir='./models/k3', max_workers=16)
"
```

### 2. Start the server

```bash
python airwrapper.py \
  --model ./models/k3 \
  --host 0.0.0.0 \
  --port 20002 \
  --max-seq-len 32768 \
  --compression 4bit \
  --delete-original
```

Output on a successful start:
```
================================================================
AirWrapperLLM — OpenAI-compatible server for AirLLM
================================================================
API key : air-XXXXXXXXXXXXXXXXXXXXXXXXXXXX
Model   : ./models/k3
Listen  : 0.0.0.0:20002
Compress: 4bit
Delete original after split: True
Max seq : 32768
================================================================
...
[AirWrapperLLM] Model loaded in 1500s.
[AirWrapperLLM] Starting uvicorn on 0.0.0.0:20002
INFO:     Uvicorn running on http://0.0.0.0:20002
```

The first start is slow — AirLLM splits the checkpoint into per-layer shards on disk (and optionally 4bit/8bit-compresses them). Subsequent starts are fast because the shards are cached.

### 3. Talk to it

```bash
API_KEY=air-XXXXXXXXXXXXXXXXXXXXXXXXXXXX   # from server stdout

curl http://localhost:20002/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 200,
    "temperature": 0.6,
    "reasoning": {"effort": "auto"}
  }'
```

Python with the official OpenAI SDK:
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:20002/v1",
    api_key="air-XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
)
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    extra_body={"reasoning": {"effort": "auto"}},
)
print(resp.choices[0].message.reasoning)  # thinking trace (if any)
print(resp.choices[0].message.content)     # final answer
```

### 4. Tool calling

```bash
curl http://localhost:20002/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k3",
    "messages": [{"role":"user","content":"What is the weather in Tokyo?"}],
    "tools": [{"type":"function","function":{
      "name":"get_weather",
      "description":"Get current weather for a city",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
    }}]
  }'
```

Response includes structured `tool_calls`:
```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123...",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}
      }]
    }
  }]
}
```

### 5. Multi-turn

Just pass the full conversation history:
```python
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[
        {"role": "user", "content": "My name is Alice. Remember it."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ],
    max_tokens=50,
)
```

---

## API reference

All endpoints require `Authorization: Bearer <API_KEY>` unless noted.

### `POST /v1/chat/completions`

OpenAI-compatible request body. Notable fields:

| Field | Type | Notes |
|---|---|---|
| `messages` | array | OpenAI format. Assistant messages may carry `reasoning` or `tool_calls` for history replay. |
| `max_tokens` | int | Hard cap on tokens generated. Keep small — AirLLM is slow. |
| `temperature` | float | 0.6 default. |
| `top_p` | float | 0.95 default. |
| `stream` | bool | SSE stream of delta chunks. |
| `tools` | array | OpenAI tool schema. |
| `reasoning` | object | `{"effort": "low"\|"medium"\|"high"\|"max"}` — enables thinking mode + controls effort. |
| `chat_template_kwargs` | object | Lower-level passthrough: `{"enable_thinking": true, "thinking_effort": "high"}`. |

### `GET /v1/models`

Lists the served model with `max_model_len`.

### `GET /health`

Returns `{"status": "ready", "model": ..., "uptime": ...}`. **No auth required** — handy for liveness probes.

---

## Configuration

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--model` | env `AIRWRAPPER_MODEL` or `/workspace/kimi-k3` | Path to downloaded model |
| `--host` | env `AIRWRAPPER_HOST` or `0.0.0.0` | Bind address |
| `--port` | env `AIRWRAPPER_PORT` or `20002` | HTTP port |
| `--compression` | env `AIRWRAPPER_COMPRESSION` or empty | `4bit` / `8bit` / empty (no compression) |
| `--delete-original` | env `AIRWRAPPER_DELETE_ORIGINAL` or `false` | Delete HF snapshot after layer-split to save disk |
| `--dtype` | env `AIRWRAPPER_DTYPE` or `auto` | `bfloat16` / `float16` |
| `--max-seq-len` | env `AIRWRAPPER_MAX_SEQ_LEN` or `1048576` | Context window cap |

### Environment variables

* `AIRWRAPPER_API_KEY_FILE` — where to read/write the API key (default `/workspace/.airwrapper_api_key`)
* `AIRWRAPPER_DEVICE` — CUDA device (default `cuda:0`)
* The same names as the flags above (env wins if flag omitted)

### Security

* The API key is auto-generated on first start and persisted to `AIRWRAPPER_API_KEY_FILE` with `chmod 600`.
* Auth is enforced on all endpoints except `GET /health`.
* **Bind to `127.0.0.1` if exposing via a reverse proxy / tunnel** — the server has no rate limiting or per-request isolation.

---

## How it works

```
                         ┌─────────────────────────────────┐
   chat message ────────►│ tokenize + render chat template │
                         └────────────────┬────────────────┘
                                          │ text
                                          ▼
                         ┌─────────────────────────────────┐
                         │ AirLLM AutoModel.from_pretrained│
                         │  - splits ckpt into per-layer   │
                         │    safetensors on first load    │
                         │  - instantiates model on meta   │
                         │  - hooks each layer to stream   │
                         │    disk→GPU→free per token      │
                         └────────────────┬────────────────┘
                                          │ tokens
                                          ▼
                         ┌─────────────────────────────────┐
                         │ decode(skip_special_tokens=False)│
                         └────────────────┬────────────────┘
                                          │ raw text w/ structural tokens
                                          ▼
                         ┌─────────────────────────────────┐
                         │ airwrapper_xtml.parse_xtml_output│
                         │  - splits <|open|>think<|sep|>… │
                         │    <|open|>response<|sep|>…      │
                         │    <|open|>tools<|sep|><|open|>… │
                         │  - returns reasoning / content /│
                         │    tool_calls                    │
                         └────────────────┬────────────────┘
                                          │ structured
                                          ▼
                         OpenAI-style JSON response
```

The XTML format is Kimi K3's chat-protocol extension. `airwrapper_xtml.py` is **model-agnostic over that grammar** — any model that emits `<|open|>` / `<|close|>` / `<|sep|>` / `<|end_of_msg|>` structural markers with `think` / `response` / `tools` channel tags will parse cleanly.

---

## Limitations

* **Single request at a time.** AirLLM has no batched-inference path; a global lock serializes generation. Two requests in flight will queue.
* **No streaming token-by-token.** AirLLM returns the full generated sequence only after the last token is produced. We simulate streaming by emitting the parsed result as SSE deltas after generation completes.
* **No persistent batching / prefix cache.** State is recreated per request.
* **No LoRA / adapter support.** AirLLM doesn't expose this.
* **XTML parser is Kimi-K3-aware** but may not handle every model's structural-token variant. If you swap models, you may need to adjust the tag parser.

---

## File layout

```
AirWrapperLLM/
├── README.md                # this file
├── LICENSE                  # MIT
├── airwrapper.py            # FastAPI server entry point
├── airwrapper_xtml.py       # pure-Python XTML parser (reusable)
├── test_xtml_parser.py      # parser unit tests (stdlib only)
├── launch_docker.sh         # convenience launch wrapper
├── requirements.txt         # pinned dependencies
├── .env.example             # config template
├── .gitignore
└── examples/
    ├── openai_client.py     # Python OpenAI SDK usage
    └── curl_examples.sh     # all curl examples
```

---

## License

MIT — see [LICENSE](LICENSE).

`airwrapper.py` is original work for this repo. `airwrapper_xtml.py` is also original work. AirLLM itself is Apache-2.0; see https://github.com/lyogavin/airllm.

---

## Acknowledgements

* [AirLLM](https://github.com/lyogavin/airllm) by Gavin Li — the disk-streaming engine that makes impossible-size models run on small GPUs.
* [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) by Moonshot AI — the 2.8T-param hybrid MoE+Mamba-2 model used as the primary test target.
* Built with [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/).