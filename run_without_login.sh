#!/bin/bash

# Set HF_TOKEN to None to avoid Hugging Face authentication
export HF_TOKEN="None"
# Allow online traffic but don't authenticate
export HUGGINGFACE_ACCESS_TOKEN="None"
# Set timeout to be longer for model downloads
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Set the root directory
ROOT=$PWD

# Set required environment variables
export CONNECT_TO_TESTNET=True
export ORG_ID=a1257f1c-ca13-4850-97f1-bbf5b292ef28  # Using the default org ID from Dockerfile

# Set up multi-address configuration with forwarded ports
export HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/31253"  # Your node's listening port
export PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30417/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ"  # Coordinator with forwarded port
export PUB_MULTI_ADDRS=""  # Leave empty for auto-configuration

# Set up identity path
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements (with output suppression like original script)
echo "Getting requirements..."
pip install -r "$ROOT"/requirements-hivemind.txt > /dev/null
pip install -r "$ROOT"/requirements.txt > /dev/null

# Check for GPU and set appropriate config
if ! which nvidia-smi > /dev/null 2>&1; then
   # You don't have a NVIDIA GPU
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
elif [ -n "$CPU_ONLY" ]; then
   # ... or we don't want to use it
   CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
else
   # NVIDIA GPU found
   pip install -r "$ROOT"/requirements_gpu.txt > /dev/null
   CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-0.5b-deepseek-r1.yaml"
fi

echo ">> Done!"
echo ""
echo "Starting training..."

# Run the training directly
python -m hivemind_exp.gsm8k.train_single_gpu \
    --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
    --identity_path "$IDENTITY_PATH" \
    --public_maddr "$PUB_MULTI_ADDRS" \
    --initial_peer "$PEER_MULTI_ADDRS" \
    --host_maddr "$HOST_MULTI_ADDRS" \
    --config "$CONFIG_PATH" 