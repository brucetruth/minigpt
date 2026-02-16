#!/bin/bash
set -e

echo "=== MiniGPT Quick Shakespeare Demo ==="
echo "Training a GPT model with Shakespeare text (reduced steps for demo)"
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
echo "  - Model: 3 layers, 192 embedding, 4 heads"
echo "  - Steps: 100 (quick demo)"
echo "  - Batch size: 8"
echo "  - Block size: 64"
echo "  - Learning rate: 5e-4 with cosine scheduling"
echo "  - Warmup: 10 steps"
echo "  - Gradient clipping: 1.0"
echo

# Train model with fewer steps for quick demo
echo "Training model..."
../minigpt train \
  --text data/shakespeare.txt \
  --steps 100 \
  --batch 8 \
  --block 64 \
  --emb 192 \
  --layers 3 \
  --heads 4 \
  --lr 5e-4 \
  --lr-min 1e-5 \
  --warmup 10 \
  --max-grad-norm 1.0 \
  --ckpt-interval 50 \
  --seed 42 \
  --out shakespeare_quick

echo
echo "=== Training Complete! ==="
echo

# Generate with different sampling strategies
echo "=== Text Generation Examples ==="
echo

echo "1. Temperature sampling (0.8):"
../minigpt generate \
  --ckpt shakespeare_quick \
  --prompt "ROMEO:" \
  --tokens 80 \
  --emb 192 \
  --layers 3 \
  --heads 4 \
  --block 64 \
  --temperature 0.8 \
  --seed 42

echo
echo

echo "2. Top-k sampling (k=30):"
../minigpt generate \
  --ckpt shakespeare_quick \
  --prompt "JULIET:" \
  --tokens 80 \
  --emb 192 \
  --layers 3 \
  --heads 4 \
  --block 64 \
  --temperature 1.0 \
  --top-k 30 \
  --seed 43

echo
echo

echo "3. Top-p sampling (p=0.9):"
../minigpt generate \
  --ckpt shakespeare_quick \
  --prompt "To be or not to be," \
  --tokens 80 \
  --emb 192 \
  --layers 3 \
  --heads 4 \
  --block 64 \
  --temperature 1.0 \
  --top-p 0.9 \
  --seed 44

echo
echo
echo "=== Quick Demo Complete ==="
echo
echo "For better results, train longer with shakespeare_demo.sh (1000 steps)"
