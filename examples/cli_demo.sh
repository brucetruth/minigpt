#!/bin/bash
set -e

echo "=== MiniGPT CLI Demo ==="
echo

# Create dummy data
echo "Creating sample training data..."
cat > data.txt << 'EOF'
The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree. The fox was very clever and quick. The tree provided shade on a sunny day. The sunny day made everyone happy. Everyone loves a happy ending.
EOF

echo "Training data created: data.txt"
echo

# Build minigpt
echo "Building minigpt..."
cd ..
make build
cd examples

echo "Build complete."
echo

# Train a tiny model
echo "Training a tiny GPT model (100 steps)..."
../minigpt train \
  --text data.txt \
  --steps 100 \
  --batch 4 \
  --block 16 \
  --emb 32 \
  --layers 1 \
  --heads 2 \
  --lr 0.001 \
  --seed 42 \
  --out checkpoints

echo
echo "Training complete! Checkpoint saved to checkpoints/"
echo

# Generate text
echo "Generating text from trained model..."
echo "Prompt: 'The quick'"
echo
../minigpt generate \
  --ckpt checkpoints \
  --prompt "The quick" \
  --tokens 20 \
  --emb 32 \
  --layers 1 \
  --heads 2 \
  --block 16

echo
echo
echo "=== Demo Complete ==="
echo "Note: The model is very small and trained on minimal data,"
echo "so the generated text may not be very coherent."
