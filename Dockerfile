FROM nvidia/cuda:12.9.1-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip curl \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"

RUN ldconfig /usr/local/cuda-12.9/compat/

# Install vLLM 0.20.0 — first version with day-0 Qwen3.6, DeepSeek-V4, and
# MiniMax-M2.7 tool-call parsers. We use `--torch-backend=auto` so uv inspects
# the installed CUDA driver at build time and picks the matching wheel
# variant (vLLM 0.20+ ships separate wheels for cu128 / cu129 / cu130). This
# avoids the pitfall where `pip install vllm==0.20.0` defaults to the cu130
# wheel and then fails on RunPod's cu129 hosts with `libcudart.so.13: cannot
# open shared object file`.
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.20.0" --torch-backend=auto

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
