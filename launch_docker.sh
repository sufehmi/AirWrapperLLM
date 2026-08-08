# ── AirWrapperLLM convenience launcher ─────────────────────────────────
# Sets up the environment and execs airwrapper.py. Use this from supervisor
# or systemd, or just run it directly:
#   ./launch_docker.sh

#!/bin/bash
set -euo pipefail

# --- Optional CUDA toolkit location for flash-attn etc. ---
# Edit this to match the CUDA major version your torch wheel is built against
# (e.g. /usr/local/cuda-12.8 for torch cu128).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# --- AirWrapperLLM config (override via env vars before launching) ---
export AIRWRAPPER_MODEL="${AIRWRAPPER_MODEL:-/workspace/kimi-k3}"
export AIRWRAPPER_HOST="${AIRWRAPPER_HOST:-0.0.0.0}"
export AIRWRAPPER_PORT="${AIRWRAPPER_PORT:-20002}"
export AIRWRAPPER_API_KEY_FILE="${AIRWRAPPER_API_KEY_FILE:-/workspace/.airwrapper_api_key}"
export AIRWRAPPER_DEVICE="${AIRWRAPPER_DEVICE:-cuda:0}"
export AIRWRAPPER_COMPRESSION="${AIRWRAPPER_COMPRESSION:-4bit}"
export AIRWRAPPER_DELETE_ORIGINAL="${AIRWRAPPER_DELETE_ORIGINAL:-1}"
export AIRWRAPPER_DTYPE="${AIRWRAPPER_DTYPE:-bfloat16}"
export AIRWRAPPER_MAX_SEQ_LEN="${AIRWRAPPER_MAX_SEQ_LEN:-32768}"

# Resolve to the directory this script lives in, so it works from anywhere.
cd "$(dirname "$(readlink -f "$0")")"

exec python3 -u airwrapper.py "$@"