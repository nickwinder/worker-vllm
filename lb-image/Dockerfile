# Tiny wrapper around vllm/vllm-openai for use as a RunPod Serverless
# load-balancing endpoint. The official image exposes /health on its main
# port but NOT /ping (vLLM gates /ping behind SageMaker mode and a different
# entrypoint). RunPod's load-balancer health-checks /ping specifically and
# kills any worker that returns 404 on it. This image solves that with a
# 20-line FastAPI sidecar on a separate port that proxies /ping to vLLM's
# /health, leaving the main vLLM serving path untouched.
#
# Resulting image is ~50MB on top of the base — the base does all the heavy
# lifting (CUDA 12.9, vLLM nightly, model code).

FROM vllm/vllm-openai:cu129-nightly

# Sidecar deps: small.
RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY ping_sidecar.py /ping_sidecar.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# vLLM listens on 8000 (PORT), sidecar on 5000 (PORT_HEALTH).
EXPOSE 8000 5000

ENTRYPOINT ["/entrypoint.sh"]
