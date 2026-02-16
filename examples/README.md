# MiniGPT Examples

This directory contains examples demonstrating how to use MiniGPT.

## Quick Start Demo

See [`cli_demo.sh`](cli_demo.sh) - A minimal example that:
- Creates dummy training data
- Trains a tiny GPT model
- Generates text from the trained model

Run it with:
```bash
cd examples
./cli_demo.sh
```

## Shakespeare Demo (Advanced)

See [`shakespeare_demo.sh`](shakespeare_demo.sh) - A comprehensive example featuring:
- Training on real Shakespeare text
- Larger model (4 layers, 256 dim)
- Advanced training features (LR scheduling, gradient clipping)
- Multiple sampling strategies (temperature, top-k, top-p)

Run it with:
```bash
cd examples
./shakespeare_demo.sh
```

⚠️ **Note**: This demo downloads ~1MB of Shakespeare text and takes several minutes to train.

## API Demo

See [`api_demo/main.go`](api_demo/main.go) for a Go program that demonstrates:
- Programmatically creating a GPT model
- Training on a small dataset
- Generating text

Run it with:
```bash
cd examples/api_demo
go run main.go
```

## Testing

See [`../test/integration_test.go`](../test/integration_test.go) for automated end-to-end tests.

Run with:
```bash
cd ..
go test ./test/...
```

## Sampling Strategies Demo

After training with `shakespeare_demo.sh`, try different sampling strategies:

```bash
# Temperature sampling (more conservative)
../minigpt generate --ckpt shakespeare_model --prompt "ROMEO:" \
  --tokens 100 --emb 256 --layers 4 --heads 4 --block 128 \
  --temperature 0.7

# Top-k sampling (typical values: 20-50)
../minigpt generate --ckpt shakespeare_model --prompt "JULIET:" \
  --tokens 100 --emb 256 --layers 4 --heads 4 --block 128 \
  --temperature 1.0 --top-k 40

# Top-p/nucleus sampling (typical values: 0.9-0.95)
../minigpt generate --ckpt shakespeare_model --prompt "To be or not to be," \
  --tokens 100 --emb 256 --layers 4 --heads 4 --block 128 \
  --temperature 1.0 --top-p 0.9
```
