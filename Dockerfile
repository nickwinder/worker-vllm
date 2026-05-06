FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip curl \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"

RUN ldconfig /usr/local/cuda-12.9/compat/

# Two-stage vLLM install:
#  (1) Install vLLM 0.19.1 first to seed all worker-handler dependencies in
#      a known-good cu129 configuration (this is upstream's recipe).
#  (2) Upgrade vLLM to 0.20.0 — first release with day-0 Qwen3.6 / DSv4 /
#      MiniMax-M2.7 tool-call parsers. We pin --extra-index-url to cu129 so
#      torch stays on the matching wheel.
#  (3) Install transformers from main — Qwen3.6 ships custom modeling code
#      that only works with a recent transformers; PyPI is often too old.
#
# IMPORTANT: must use the explicit cu129 extra-index, not `--torch-backend=auto`.
# Per uv's docs, `auto` queries the build host for a CUDA driver and falls
# back to CPU-only PyTorch when none is found. GitHub Actions runners don't
# have GPUs, so `auto` silently gives us CPU-only torch and the worker
# crashes on import at runtime.
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.19.1" --extra-index-url https://download.pytorch.org/whl/cu129 && \
    uv pip install --system -U "vllm[flashinfer]==0.20.0" --extra-index-url https://download.pytorch.org/whl/cu129 && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    uv pip install --system git+https://github.com/huggingface/transformers.git

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

ENV PYTHONPATH="/:/vllm-workspace"

# VLLM_NIGHTLY block removed — the base install above already pins vLLM 0.20.0
# with auto-selected CUDA wheels. Re-add a similar block here if you ever need
# a true nightly (e.g. for unreleased model architecture support); just be
# aware the nightly wheels currently target CUDA 13 and won't load on
# CUDA 12.x driver hosts without further pinning.

COPY src /src
RUN chmod +x /src/start.sh
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]
