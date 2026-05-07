#!/bin/bash
# Run the /ping sidecar in the background, then exec vLLM in the foreground.
# vLLM args come through "$@" — RunPod's template-level docker-start-cmd is
# passed directly to this script.
set -e

echo "==== entrypoint.sh: starting /ping sidecar on PORT_HEALTH=${PORT_HEALTH:-5000} ===="
python3 /ping_sidecar.py &

echo "==== entrypoint.sh: starting vLLM with args: $@ ===="
exec python3 -m vllm.entrypoints.openai.api_server "$@"
