#!/usr/bin/env python3
"""AirWrapperLLM — OpenAI-compatible FastAPI server for AirLLM.

Wraps a memory-frugal AirLLM model (built for streaming-from-disk inference of
models far larger than VRAM) behind a standard OpenAI Chat Completions API:
  - Bearer API-key auth
  - /v1/chat/completions  (stream + non-stream)
  - /v1/models
  - /health (unauthenticated liveness)
  - multi-turn conversations (caller passes message history)
  - thinking/reasoning mode  (parsed from the model's <think> channel)
  - tool/function calling   (parsed from the model's <tools> channel)

Because AirLLM streams layers/experts from disk one at a time, throughput is
naturally low (single-digit tok/s on most hardware). The server is therefore
strictly sequential: one request at a time, guarded by a global lock. This
matches the underlying engine, which cannot benefit from concurrent batching.

The chat rendering is performed by the tokenizer's own apply_chat_template;
the server only needs a *decoder* for the generated tokens — see
airwrapper_xtml.py for that.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from typing import Any, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from airwrapper_xtml import parse_xtml_output  # pure-Python XTML decoder

# ─────────────────────────── globals ───────────────────────────

AIR_MODEL = None          # the AirLLM model, lazily loaded
TOKENIZER = None          # AirLLM exposes the tokenizer at model.tokenizer
DEVICE = os.environ.get("AIRWRAPPER_DEVICE", "cuda:0")
MODEL_ID = "airwrapper"   # overridden to the served model name below
LOAD_LOCK = threading.Lock()
GEN_LOCK = threading.Lock()    # AirLLM is single-tenant; serialise generation
START_TIME = time.time()

# Semantic version. Bump on release. Keep in sync with the README header
# and the git tag (e.g. `git tag -a v1.0.0 -m "v1.0 release"`).
__version__ = "1.0.0"

# ─────────────────────────── auth ───────────────────────────

API_KEY_FILE = os.environ.get("AIRWRAPPER_API_KEY_FILE", "/tmp/.airwrapper_api_key")


def load_api_key() -> str:
    """Load the API key; generate+persist one if missing."""
    try:
        with open(API_KEY_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith("AIRWRAPPER_API_KEY="):
                    return line.split("=", 1)[1]
                if line.startswith("air-"):
                    return line
    except FileNotFoundError:
        pass
    key = "air-" + os.urandom(24).hex()
    os.makedirs(os.path.dirname(API_KEY_FILE) or ".", exist_ok=True)
    with open(API_KEY_FILE, "w") as f:
        f.write(f"AIRWRAPPER_API_KEY={key}\n")
    os.chmod(API_KEY_FILE, 0o600)
    return key


API_KEY = load_api_key()


async def verify_api_key(request: Request) -> None:
    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else ""
    if token != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide 'Authorization: Bearer <key>'.",
        )


# ─────────────────────────── request models ───────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None
    tool_calls: Optional[list[Any]] = None
    tool_call_id: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_content: Optional[str] = None


class Tool(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Any] = None
    # accept but ignore many OpenAI passthrough fields
    thinking: Optional[Any] = None
    reasoning: Optional[Any] = None
    stop: Optional[Any] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    chat_template_kwargs: Optional[dict[str, Any]] = None

    model_config = {"populate_by_name": True, "extra": "allow"}


# ─────────────────────────── model loading ───────────────────────────


def load_model(model_path: str, compression: Optional[str], delete_original: bool,
               dtype: Optional[str], max_seq_len: int) -> None:
    """Load the AirLLM model + tokenizer into the globals."""
    global AIR_MODEL, TOKENIZER
    import torch  # deferred
    from airllm import AutoModel

    dt = getattr(torch, dtype, None) if dtype else None

    print(f"[AirWrapperLLM] Loading AirLLM model from {model_path} ...", flush=True)
    t0 = time.time()
    AIR_MODEL = AutoModel.from_pretrained(
        model_path,
        device=DEVICE,
        dtype=dt,
        max_seq_len=max_seq_len,
        compression=compression,
        delete_original=delete_original,
    )
    TOKENIZER = AIR_MODEL.tokenizer
    elapsed = time.time() - t0
    print(f"[AirWrapperLLM] Model loaded in {elapsed:.1f}s.", flush=True)


# ─────────────────────────── generation core ───────────────────────────


def _normalise_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    msg_dicts = [m.model_dump(exclude_none=True) for m in messages]
    for m in msg_dicts:
        # OpenAI allows content as list-of-parts; K3 expects plain string.
        c = m.get("content")
        if isinstance(c, list):
            texts = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            m["content"] = "".join(texts)
        # carry reasoning into reasoning_content for the encoder
        if m.get("reasoning") and not m.get("reasoning_content"):
            m["reasoning_content"] = m.pop("reasoning")
    return msg_dicts


def _chat_apply(messages: list[ChatMessage], tools: Optional[list[Tool]],
                thinking: bool, thinking_effort: Optional[str]) -> str:
    """Render messages -> prompt string using the tokenizer's chat template."""
    msg_dicts = _normalise_messages(messages)
    kw: dict[str, Any] = dict(
        tokenize=False,
        add_generation_prompt=True,
        thinking=thinking,
    )
    if thinking_effort:
        kw["thinking_effort"] = thinking_effort
    tools_list = [t.model_dump() for t in tools] if tools else None

    return TOKENIZER.apply_chat_template(
        msg_dicts,
        tools=tools_list,
        **kw,
    )


def _generate(raw_prompt: str, max_new_tokens: int, temperature: float,
              top_p: float) -> str:
    """Run a single AirLLM generation. Caller holds GEN_LOCK."""
    import torch  # deferred
    enc = TOKENIZER(raw_prompt, return_tensors="pt",
                    return_attention_mask=True, truncation=True,
                    max_length=AIR_MODEL.max_seq_len, padding=False)
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        out = AIR_MODEL.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            use_cache=True,
            pad_token_id=getattr(TOKENIZER, "eos_token_id", None),
            return_dict_in_generate=True,
        )
    seq = out.sequences[0]
    new_tokens = seq[input_ids.shape[1]:]
    return TOKENIZER.decode(new_tokens, skip_special_tokens=False)


# ─────────────────────────── FastAPI app ───────────────────────────


app = FastAPI(title="AirWrapperLLM", version=__version__)


@app.get("/")
async def root():
    return {"service": "AirWrapperLLM", "version": __version__, "status": "ok"}


@app.get("/health")
async def health():
    ready = AIR_MODEL is not None
    return {"status": "ready" if ready else "loading",
            "version": __version__,
            "model": MODEL_ID,
            "uptime": time.time() - START_TIME}


@app.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(START_TIME),
            "owned_by": "airwrapper",
            "max_model_len": getattr(AIR_MODEL, "max_seq_len", None) if AIR_MODEL else None,
        }],
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _new_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


async def _stream_chat(req: ChatCompletionRequest, prompt: str,
                       gen_params: dict[str, Any]):
    """Run the blocking generation on a thread; emit the parsed result as SSE deltas.

    AirLLM is synchronous and not token-iterative, so we run it in a worker and
    emit the deltas after completion. Keep-alive comments prevent client timeouts.
    """
    import asyncio

    resp_id = _new_response_id()
    created = int(time.time())

    yield _sse("message", {
        "id": resp_id, "object": "chat.completion.chunk", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    result_text: list[str] = []

    def _run():
        try:
            txt = _generate(prompt, **gen_params)
            result_text.append(txt)
        except Exception as e:  # noqa: BLE001
            result_text.append(f"__AIRWRAPPER_ERROR__::{type(e).__name__}: {e}")

    th = threading.Thread(target=_run, daemon=True)
    with GEN_LOCK:
        th.start()
        while th.is_alive():
            await asyncio.sleep(2.0)
            yield ": keep-alive\n\n"
        th.join()

    raw = result_text[0] if result_text else ""
    if raw.startswith("__AIRWRAPPER_ERROR__"):
        err = raw.split("::", 1)[1] if "::" in raw else "generation error"
        yield _sse("message", {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
            "error": {"message": err},
        })
        yield "data: [DONE]\n\n"
        return

    parsed = parse_xtml_output(raw)
    if parsed["reasoning"]:
        yield _sse("message", {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"reasoning": parsed["reasoning"]},
                         "finish_reason": None}],
        })
    if parsed["content"]:
        yield _sse("message", {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"content": parsed["content"]},
                         "finish_reason": None}],
        })
    if parsed["tool_calls"]:
        yield _sse("message", {
            "id": resp_id, "object": "chat.completion.chunk", "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {"tool_calls": parsed["tool_calls"]},
                         "finish_reason": None}],
        })

    finish = "tool_calls" if parsed["tool_calls"] else "stop"
    yield _sse("message", {
        "id": resp_id, "object": "chat.completion.chunk", "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
    })
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _=Depends(verify_api_key)):
    if AIR_MODEL is None:
        raise HTTPException(503, "Model is still loading. Retry shortly.")

    # Determine thinking mode + effort
    thinking = True
    thinking_effort: Optional[str] = None
    ctk = req.chat_template_kwargs or {}
    if "enable_thinking" in ctk:
        thinking = bool(ctk["enable_thinking"])
    if "thinking_effort" in ctk:
        thinking_effort = ctk["thinking_effort"]
    effort_src = req.reasoning or req.thinking
    if isinstance(effort_src, dict):
        eff = effort_src.get("effort")
        if eff in ("low", "medium", "high", "max"):
            # K3 accepts low/high/max; coerce "medium" to closest supported value.
            thinking_effort = eff if eff != "medium" else "high"
            thinking = True

    try:
        prompt = _chat_apply(req.messages, req.tools, thinking, thinking_effort)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Failed to render chat template: {e}")

    max_new = req.max_tokens or req.max_completion_tokens or 1024
    gen_params = {
        "max_new_tokens": max_new,
        "temperature": req.temperature if req.temperature is not None else 0.6,
        "top_p": req.top_p if req.top_p is not None else 0.95,
    }

    if req.stream:
        return StreamingResponse(
            _stream_chat(req, prompt, gen_params),
            media_type="text/event-stream",
        )

    with GEN_LOCK:
        try:
            raw = _generate(prompt, **gen_params)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Generation failed: {e}")

    parsed = parse_xtml_output(raw)
    finish = "tool_calls" if parsed["tool_calls"] else "stop"
    resp_id = _new_response_id()
    created = int(time.time())

    message: dict[str, Any] = {"role": "assistant", "content": parsed["content"] or None}
    if parsed["reasoning"]:
        message["reasoning"] = parsed["reasoning"]
        message["reasoning_content"] = parsed["reasoning"]
    if parsed["tool_calls"]:
        message["tool_calls"] = parsed["tool_calls"]
        message["content"] = None

    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": _count_tokens(prompt),
            "completion_tokens": _count_tokens(raw),
            "total_tokens": _count_tokens(prompt) + _count_tokens(raw),
        },
    }


def _count_tokens(text: str) -> int:
    if TOKENIZER is None:
        return max(1, len(text) // 4)
    try:
        return len(TOKENIZER.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ─────────────────────────── entrypoint ───────────────────────────


def main():
    p = argparse.ArgumentParser(description="AirWrapperLLM — OpenAI-compatible server for AirLLM")
    p.add_argument("--model", default=os.environ.get("AIRWRAPPER_MODEL", "/workspace/kimi-k3"))
    p.add_argument("--host", default=os.environ.get("AIRWRAPPER_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("AIRWRAPPER_PORT", "20002")))
    p.add_argument("--compression", default=os.environ.get("AIRWRAPPER_COMPRESSION"),
                   choices=[None, "4bit", "8bit"])
    p.add_argument("--delete-original", action="store_true",
                   default=os.environ.get("AIRWRAPPER_DELETE_ORIGINAL") == "1")
    p.add_argument("--dtype", default=os.environ.get("AIRWRAPPER_DTYPE"))
    p.add_argument("--max-seq-len", type=int,
                   default=int(os.environ.get("AIRWRAPPER_MAX_SEQ_LEN", "1048576")))
    args = p.parse_args()

    global MODEL_ID
    # Use the directory name as the served model id (e.g. "kimi-k3" for /workspace/kimi-k3).
    MODEL_ID = os.path.basename(os.path.normpath(args.model)) or "airwrapper"

    print("=" * 64, flush=True)
    print("AirWrapperLLM — OpenAI-compatible server for AirLLM", flush=True)
    print("=" * 64, flush=True)
    print(f"API key : {API_KEY}", flush=True)
    print(f"Model   : {args.model}", flush=True)
    print(f"Listen  : {args.host}:{args.port}", flush=True)
    print(f"Compress: {args.compression or 'none'}", flush=True)
    print(f"Delete original after split: {args.delete_original}", flush=True)
    print(f"Dtype   : {args.dtype or 'auto'}", flush=True)
    print(f"Max seq : {args.max_seq_len}", flush=True)
    print("=" * 64, flush=True)

    # Load model before serving (so /health reports ready truthfully).
    with LOAD_LOCK:
        load_model(args.model, args.compression, args.delete_original,
                   args.dtype, args.max_seq_len)

    print(f"[AirWrapperLLM] Starting uvicorn on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()