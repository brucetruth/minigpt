#!/bin/bash
set -e

echo "=== MiniGPT Advanced Demo ==="
echo "Training a larger GPT model with Shakespeare text"
echo

# Create data directory
mkdir -p data

# Download Shakespeare text (if not already present)
if [ ! -f data/shakespeare.txt ]; then
    echo "Downloading Shakespeare dataset..."
    curl -s https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt > data/shakespeare.txt
    echo "Downloaded Shakespeare dataset ($(wc -c < data/shakespeare.txt) bytes)"
fi

echo
echo "Building minigpt..."
cd ..
make build
cd examples

echo
echo "=== Training Configuration ==="
echo "  - Model: 4 layers, 256 embedding, 4 heads"
echo "  - Steps: 1000"
echo "  - Batch size: 16"
echo "  - Block size: 128"
echo "  - Learning rate: 3e-4 with cosine scheduling"
echo "  - Warmup: 100 steps"
echo "  - Gradient clipping: 1.0"
echo

# Train model
echo "Training model (this may take a few minutes)..."
../minigpt train \
  --text data/shakespeare.txt \
  --steps 1000 \
  --batch 16 \
  --block 128 \
  --emb 256 \
  --layers 4 \
  --heads 4 \
  --lr 3e-4 \
  --lr-min 1e-5 \
  --warmup 100 \
  --max-grad-norm 1.0 \
  --ckpt-interval 250 \
  --seed 42 \
  --out shakespeare_model

echo
echo "=== Training Complete! ==="
echo

# Generate with different sampling strategies
echo "=== Text Generation Examples ==="
echo

echo "1. Greedy (temperature=0.8):"
../minigpt generate \
  --ckpt shakespeare_model \
  --prompt "ROMEO:" \
  --tokens 100 \
  --emb 256 \
  --layers 4 \
  --heads 4 \
  --block 128 \
  --temperature 0.8 \
  --seed 42

echo
echo

echo "2. With top-k=40:"
../minigpt generate \
  --ckpt shakespeare_model \
  --prompt "JULIET:" \
  --tokens 100 \
  --emb 256 \
  --layers 4 \
  --heads 4 \
  --block 128 \
  --temperature 1.0 \
  --top-k 40 \
  --seed 43

echo
echo

echo "3. With top-p=0.9 (nucleus sampling):"
../minigpt generate \
  --ckpt shakespeare_model \
  --prompt "To be or not to be," \
  --tokens 100 \
  --emb 256 \
  --layers 4 \
  --heads 4 \
  --block 128 \
  --temperature 1.0 \
  --top-p 0.9 \
  --seed 44

echo
echo
echo "=== Demo Complete ==="
echo
echo "Model saved to: shakespeare_model/"
echo "Try your own prompts with:"
echo "./minigpt generate --ckpt shakespeare_model \\"
echo "  --prompt 'HAMLET:' --tokens 150 \\"
echo "  --emb 256 --layers 4 --heads 4 --block 128 \\"
echo "  --temperature 0.8 --top-k 40"
