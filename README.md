# MiniGPT in Go

A minimal, hackable, CPU-first GPT implementation in pure Go with **advanced training features**.

## Features
- **Full Transformer Architecture**: Attention + MLP layers with GELU activation
- **Advanced Sampling**: Temperature, top-k, and top-p (nucleus) sampling
- **Modern Training**: Gradient clipping, cosine LR scheduling with warmup, residual dropout
- **End-to-end pipeline**: BPE tokenizer, training, generation, checkpointing
- **Deterministic execution**: Fixed seeds for reproducible results
- **Pure Go**: No CGo required by default
- **Optional acceleration**: OpenBLAS support (`-tags openblas`)
- **Manual backpropagation**: Educational, hackable implementation

## Installation

```bash
git clone https://github.com/brucetruth/minigpt
cd minigpt
go mod tidy
make build
```

## Quick Start

Train a model on Shakespeare:
```bash
cd examples
./shakespeare_demo.sh
```

## Usage

### Training

```bash
./minigpt train \
  --text data/input.txt \
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
  --ckpt-interval 250
```

**New training flags:**
- `--lr-min`: Minimum learning rate for cosine scheduling
- `--warmup`: Number of warmup steps for LR schedule
- `--max-grad-norm`: Gradient clipping threshold (0 = disabled)
- `--ckpt-interval`: Save checkpoints every N steps

### Generation

```bash
./minigpt generate \
  --ckpt checkpoints \
  --prompt "To be or not to be," \
  --tokens 100 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

**Sampling options:**
- `--temperature`: Sampling temperature (higher = more random, default: 1.0)
- `--top-k`: Top-k sampling (0 = disabled)
- `--top-p`: Top-p/nucleus sampling (1.0 = disabled)
- `--seed`: Random seed for reproducible generation

### Examples

See the [`examples/`](examples/) directory for:
- **Simple demo**: [`cli_demo.sh`](examples/cli_demo.sh) - Quick start with minimal config
- **Shakespeare demo**: [`shakespeare_demo.sh`](examples/shakespeare_demo.sh) - Larger model with advanced sampling
- **API demo**: [`api_demo/`](examples/api_demo/) - Programmatic usage in Go

### Benchmark
```bash
./minigpt bench --size 512
```

## Development

Run tests:
```bash
make test
# or
go test ./...
```

## Architecture

### Core Components
- **`llm/tensor`**: NDArray, operators, GELU activation & backward pass, sampling functions
- **`llm/nn`**: Layers (Linear, LayerNorm, Embedding, MLP), Loss, Dropout
- **`llm/transformer`**: GPT model, Multi-head attention, Transformer blocks
- **`llm/optim`**: AdamW optimizer with gradient clipping, LR scheduler
- **`llm/data`**: Dataset loader
- **`llm/tokenizer`**: BPE tokenizer
- **`llm/io`**: Checkpoint saving/loading
- **`cmd/minigpt`**: CLI interface

### Model Architecture
```
Input Tokens
    ↓
Token Embedding + Position Embedding
    ↓
Dropout
    ↓
[Transformer Block] × N
    ├─ LayerNorm
    ├─ Multi-Head Causal Self-Attention
    ├─ Residual + Dropout
    ├─ LayerNorm
    ├─ MLP (Linear → GELU → Linear)
    └─ Residual + Dropout
    ↓
LayerNorm
    ↓
LM Head (Linear to vocab size)
```

## Advanced Features

### Sampling Strategies
- **Temperature**: Controls randomness (0.7-0.9 for coherent text)
- **Top-k**: Limits sampling to k most likely tokens (typically 20-50)
- **Top-p**: Limits sampling to tokens with cumulative probability p (typically 0.9-0.95)

### Training Improvements
- **Gradient Clipping**: Prevents exploding gradients during training
- **LR Scheduling**: Cosine annealing with linear warmup for stable training
- **Residual Dropout**: Applied after attention and MLP for regularization

## Examples

**Train on your own text:**
```bash
./minigpt train --text mydata.txt --steps 2000 --emb 512 --layers 6 --heads 8
```

**Generate with different creativity levels:**
```bash
# Conservative (more coherent)
./minigpt generate --ckpt checkpoints --prompt "Once upon a time" --temperature 0.7

# Balanced
./minigpt generate --ckpt checkpoints --prompt "Once upon a time" --temperature 1.0 --top-k 40

# Creative (more diverse)
./minigpt generate --ckpt checkpoints --prompt "Once upon a time" --temperature 1.2 --top-p 0.95
```

## Performance Notes
- Training speed scales with model size; expect ~1-2 it/s for 256-dim 4-layer models on CPU
- Use smaller batch sizes if running out of memory
- GPU acceleration not yet implemented (contributions welcome!)

## License
MIT
