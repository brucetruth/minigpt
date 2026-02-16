package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	llmio "github.com/brucetruth/minigpt/llm/io"
	"github.com/brucetruth/minigpt/llm/tensor"
	"github.com/brucetruth/minigpt/llm/tokenizer"
	"github.com/brucetruth/minigpt/llm/transformer"
)

func generateCmd(args []string) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	prompt := fs.String("prompt", "Hello", "Input prompt")
	tokens := fs.Int("tokens", 50, "Number of tokens to generate")
	ckpt := fs.String("ckpt", "checkpoints", "Checkpoint directory")
	temperature := fs.Float64("temperature", 1.0, "Sampling temperature (higher = more random)")
	topK := fs.Int("top-k", 0, "Top-k sampling (0 = disabled)")
	topP := fs.Float64("top-p", 1.0, "Top-p (nucleus) sampling (1.0 = disabled)")
	seed := fs.Int64("seed", -1, "Random seed (-1 for random)")

	// ... config flags if we can't load config from ckpt ...
	// For simplicity, we hardcode config or expect args matching training.
	// In real impl, config should be saved.
	embDim := fs.Int("emb", 128, "")
	nLayer := fs.Int("layers", 2, "")
	nHead := fs.Int("heads", 2, "")
	blockSize := fs.Int("block", 64, "")

	fs.Parse(args)

	// Set random seed
	if *seed >= 0 {
		rand.Seed(*seed)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	// Load Tokenizer
	tok, err := tokenizer.Load(*ckpt + "/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Load Checkpoint Metadata & Weights
	// First, try to load metadata to get config
	// We need to initialize model first? No, we need config first.
	// io.LoadCheckpoint takes a model.
	// So we should LoadMetadata first?
	// io.LoadCheckpoint does both.
	// Let's rely on manual config for now or peeking at meta.json?
	// Ideally we'd have a LoadRequest/LoadConfig function.
	// But let's look at io/checkpoint.go again. It reads file "meta.json".
	// Let's just peek at it or try to load.

	// For now, let's assume we initialize model with flags (or default) and then LoadCheckpoint.
	// BUT if dimensions mismatch, LoadCheckpoint might warn but not resizing.
	// Actually, let's try to read meta.json manually here to get config, then init model, then load weights.

	// Config
	var cfg transformer.Config
	metaPath := *ckpt + "/meta.json"
	if _, err := os.Stat(metaPath); err == nil {
		// Load config from meta
		data, err := os.ReadFile(metaPath)
		if err == nil {
			var meta llmio.CheckpointMetadata
			if err := json.Unmarshal(data, &meta); err == nil {
				cfg = meta.Config
				fmt.Printf("Loaded config from checkpoint: %+v\n", cfg)
			}
		}
	}

	// Override/Fallback with flags if config is zero (failed load)
	if cfg.NEmb == 0 {
		cfg = transformer.Config{
			VocabSize: tok.VocabSize,
			BlockSize: *blockSize,
			NLayer:    *nLayer,
			NHead:     *nHead,
			NEmb:      *embDim,
			PDrop:     0.0,
		}
	} else {
		// Ensure vocab size matches tokenizer (in case it changed or was different)
		cfg.VocabSize = tok.VocabSize
		cfg.PDrop = 0.0 // Disable dropout for generation
	}

	// Init Model
	log.Println("Initializing model...")
	model := transformer.NewGPT(cfg)

	// Load Weights
	log.Println("Loading weights...")
	if _, err := llmio.LoadCheckpoint(*ckpt, model); err != nil {
		fmt.Printf("Error loading weights: %v. Using random weights.\n", err)
	} else {
		fmt.Println("Weights loaded successfully.")
	}

	// Encode prompt
	ids := tok.Encode(*prompt)

	// Generate loop
	for i := 0; i < *tokens; i++ {
		// Crop context
		ctx := ids
		if len(ctx) > cfg.BlockSize {
			ctx = ctx[len(ctx)-cfg.BlockSize:]
		}

		// Prepare input tensor
		b := 1
		t := len(ctx)
		x := tensor.New(b, t)
		for j, id := range ctx {
			x.Data[j] = float32(id)
		}

		// Forward
		logits := model.Forward(x)

		// Get last token logits: [1, T, V] -> [1, V]
		vocabSize := logits.Shape[2]
		lastLogits := tensor.New(vocabSize)
		offset := (t - 1) * vocabSize
		for j := 0; j < vocabSize; j++ {
			lastLogits.Data[j] = logits.Data[offset+j]
		}

		// Sample next token using temperature, top-k, top-p
		nextID := tensor.SampleFromLogits(lastLogits, float32(*temperature), *topK, float32(*topP))

		// To decode:
		fmt.Print(tok.Decoder[nextID])

		ids = append(ids, nextID)
	}
	fmt.Println()
}
