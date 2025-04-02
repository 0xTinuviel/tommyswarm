# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt update -y && \
    apt install -y \
    sudo \
    make \
    gcc \
    lld \
    libncurses-dev \
    libffi-dev \
    liblzma-dev \
    zlib1g zlib1g-dev \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    xz-utils \
    tk-dev \
    libxml2-dev libxmlsec1-dev \
    git curl jq neovim \
    python3 \
    python3-dev \
    python3-venv \
    kmod \
    && apt-get clean

# Create non-root user
RUN useradd -m -s /bin/bash gensyn

# Set up Python environment
WORKDIR /home/gensyn
USER gensyn

# Create and activate virtual environment
RUN python3 -m venv .venv
ENV PATH="/home/gensyn/.venv/bin:$PATH"

# Copy requirements files
COPY requirements.txt requirements-hivemind.txt requirements_gpu.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-hivemind.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements_gpu.txt || true \
    && pip install --no-cache-dir 'protobuf<5.28.0' numpy \
    && pip install --no-cache-dir --no-build-isolation vllm==0.7.0

# Copy the RL Swarm code
COPY --chown=gensyn . /home/gensyn/rl_swarm
WORKDIR /home/gensyn/rl_swarm

# Environment variables
ENV CONNECT_TO_TESTNET=True
ENV ORG_ID=a1257f1c-ca13-4850-97f1-bbf5b292ef28
ENV HUGGINGFACE_ACCESS_TOKEN=None

# Run the swarm
CMD ["python", "-m", "hivemind_exp.gsm8k.train_single_gpu", \
     "--hf_token", "None", \
     "--identity_path", "/home/gensyn/rl_swarm/swarm.pem", \
     "--modal_org_id", "a1257f1c-ca13-4850-97f1-bbf5b292ef28", \
     "--config", "hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"]
