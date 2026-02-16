package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/brucetruth/minigpt/llm/data"
	llmio "github.com/brucetruth/minigpt/llm/io"
	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/optim"
	"github.com/brucetruth/minigpt/llm/tokenizer"
	"github.com/brucetruth/minigpt/llm/transformer"
)

func trainCmd(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	textPath := fs.String("text", "data/input.txt", "Path to input text")
	steps := fs.Int("steps", 100, "Number of training steps")
	batchSize := fs.Int("batch", 8, "Batch size")
	blockSize := fs.Int("block", 64, "Block size (context length)")
	embDim := fs.Int("emb", 128, "Embedding dimension")
	nLayer := fs.Int("layers", 2, "Number of layers")
	nHead := fs.Int("heads", 2, "Number of heads")
	lr := fs.Float64("lr", 1e-3, "Peak learning rate")
	lrMin := fs.Float64("lr-min", 1e-4, "Minimum learning rate")
	warmupSteps := fs.Int("warmup", 10, "Warmup steps")
	maxGradNorm := fs.Float64("max-grad-norm", 1.0, "Max gradient norm (0 = no clipping)")
	ckptInterval := fs.Int("ckpt-interval", 100, "Save checkpoint every N steps")
	seed := fs.Int64("seed", 42, "Random seed")
	outDir := fs.String("out", "checkpoints", "Output directory")

	fs.Parse(args)

	// Determinism
	rand.Seed(*seed)

	// Load Text
	content, err := os.ReadFile(*textPath)
	if err != nil {
		log.Fatalf("Failed to read file: %v", err)
	}
	text := string(content)

	// Tokenizer
	log.Println("Training tokenizer...")
	tok := tokenizer.New()
	tok.Train(text, 1000) // Small vocab for testing/speed
	// Encode
	ids := tok.Encode(text)
	log.Printf("Encoded %d chars to %d tokens\n", len(text), len(ids))

	// Dataset
	ds := data.NewTextDataset(ids, *blockSize)

	// Config
	cfg := transformer.Config{
		VocabSize: tok.VocabSize,
		BlockSize: *blockSize,
		NLayer:    *nLayer,
		NHead:     *nHead,
		NEmb:      *embDim,
		PDrop:     0.1,
	}

	// Model
	log.Println("Initializing model...")
	model := transformer.NewGPT(cfg)

	// Optimizer
	opt := optim.NewAdamW(model.Parameters(), float32(*lr))
	criterion := nn.NewCrossEntropyLoss()

	// LR Scheduler
	scheduler := optim.NewCosineScheduleWithWarmup(*warmupSteps, *steps, float32(*lr), float32(*lrMin))

	// Loop
	start := time.Now()
	var loss float32
	for step := 0; step < *steps; step++ {
		// Update learning rate
		currentLR := scheduler.GetLR(step)
		opt.SetLR(currentLR)

		// 1. Batch
		x, y := ds.GetBatch(*batchSize)

		// 2. Forward
		logits := model.Forward(x)

		// 3. Loss
		b, t, v := logits.Shape[0], logits.Shape[1], logits.Shape[2]
		logitsFlat, _ := logits.View(b*t, v)
		loss = criterion.Forward(logitsFlat, y)

		if step%10 == 0 {
			fmt.Printf("Step %d | Loss: %.4f | LR: %.6f | Time: %v\n", step, loss, currentLR, time.Since(start))
			start = time.Now()
		}

		// 4. Backward
		opt.ZeroGrad()

		dLogitsFlat := criterion.Backward(logitsFlat, y)
		dLogits, _ := dLogitsFlat.View(b, t, v)
		model.Backward(dLogits)

		// Gradient clipping
		if *maxGradNorm > 0 {
			opt.ClipGradNorm(float32(*maxGradNorm))
		}

		// 5. Step
		opt.Step()

		// Save checkpoint periodically
		if *ckptInterval > 0 && (step+1)%*ckptInterval == 0 {
			fmt.Printf("Saving checkpoint at step %d...\n", step+1)
			meta := llmio.CheckpointMetadata{
				Step:   step + 1,
				Loss:   loss,
				Config: cfg,
			}
			if err := llmio.SaveCheckpoint(*outDir, model, meta); err != nil {
				log.Printf("Failed to save checkpoint: %v", err)
			}
			tok.Save(*outDir + "/tokenizer.json")
		}
	}

	// Save final checkpoint
	fmt.Println("Saving final checkpoint...")
	meta := llmio.CheckpointMetadata{
		Step:   *steps,
		Loss:   loss,
		Config: cfg,
	}
	if err := llmio.SaveCheckpoint(*outDir, model, meta); err != nil {
		log.Printf("Failed to save checkpoint: %v", err)
	}

	// Save tokenizer
	tok.Save(*outDir + "/tokenizer.json")

	fmt.Println("Training complete.")
}
