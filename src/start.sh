#!/bin/bash
# Instrumented start.sh — prints diagnostics before launching the handler so
# failures produce useful output instead of a bare "exit code 1". Output is
# unbuffered (`python3 -u`) so anything the handler writes makes it to
# RunPod's log stream before the process dies.
set -ex

echo "==== start.sh: env diagnostics ===="
echo "MODEL_NAME=${MODEL_NAME:-<unset>}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-<unset>}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-<unset>}"
echo "TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-<unset>}"
echo "REASONING_PARSER=${REASONING_PARSER:-<unset>}"
echo "ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-<unset>}"
echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-<unset>}"

echo "==== start.sh: python + vllm sanity ===="
python3 --version
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>&1 || echo "WARN: vllm import failed"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('cuda available:', torch.cuda.is_available())" 2>&1 || echo "WARN: torch import failed"

if [ -n "${TRANSFORMERS_VERSION}" ]; then
    echo "Installing transformers==${TRANSFORMERS_VERSION}"
    uv pip install --system "transformers==${TRANSFORMERS_VERSION}"
fi

echo "==== start.sh: launching handler.py ===="
exec python3 -u /src/handler.py
