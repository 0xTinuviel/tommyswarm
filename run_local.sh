#!/bin/bash

# Set HF_TOKEN to your token value
export HF_TOKEN="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"
export HUGGINGFACE_ACCESS_TOKEN="$HF_TOKEN"
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Set the root directory
ROOT=$PWD

# Configure for peering test
export HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/40000"  # Use a high port unlikely to have conflicts
# Use the coordinator node from the original script
export PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"
export PUB_MULTI_ADDRS=""

# Identity file path
IDENTITY_PATH="$ROOT/local_swarm.pem"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r "$ROOT"/requirements-hivemind.txt
pip install -r "$ROOT"/requirements.txt

# Detect if you have a GPU and choose the appropriate config
if command -v nvidia-smi &> /dev/null; then
    pip install -r "$ROOT"/requirements_gpu.txt
    CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
else
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
fi

echo "Starting training with peer connection to Gensyn coordinator..."

# Enable verbose logging for DHT
export HIVEMIND_DHT_LOG_LEVEL=DEBUG

# Run the training
python -m hivemind_exp.gsm8k.train_single_gpu \
    --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
    --identity_path "$IDENTITY_PATH" \
    --host_maddr "$HOST_MULTI_ADDRS" \
    --initial_peers "$PEER_MULTI_ADDRS" \
    --config "$CONFIG_PATH"