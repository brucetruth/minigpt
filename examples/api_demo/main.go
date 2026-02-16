package main

import (
	"fmt"
	"math/rand"

	"github.com/brucetruth/minigpt/llm/data"
	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/optim"
	"github.com/brucetruth/minigpt/llm/tensor"
	"github.com/brucetruth/minigpt/llm/tokenizer"
	"github.com/brucetruth/minigpt/llm/transformer"
)

func main() {
	fmt.Println("=== MiniGPT API Demo ===")
	fmt.Println()

	// Set seed for reproducibility
	rand.Seed(42)

	// Sample text
	text := "The quick brown fox jumps over the lazy dog. " +
		"The dog was sleeping under a tree. " +
		"The fox was very clever and quick. " +
		"The tree provided shade on a sunny day. " +
		"The sunny day made everyone happy. " +
		"Everyone loves a happy ending."

	fmt.Println("Training text:", text)
	fmt.Println()

	// Train tokenizer
	fmt.Println("Training tokenizer...")
	tok := tokenizer.New()
	tok.Train(text, 100) // Small vocab

	ids := tok.Encode(text)
	fmt.Printf("Tokenized: %d chars -> %d tokens\n", len(text), len(ids))
	fmt.Println()

	// Create dataset
	blockSize := 16
	ds := data.NewTextDataset(ids, blockSize)

	// Configure model
	cfg := transformer.Config{
		VocabSize: tok.VocabSize,
		BlockSize: blockSize,
		NLayer:    1,
		NHead:     2,
		NEmb:      32,
		PDrop:     0.0, // No dropout for demo
	}

	fmt.Printf("Model config: vocab=%d, block=%d, layers=%d, heads=%d, emb=%d\n",
		cfg.VocabSize, cfg.BlockSize, cfg.NLayer, cfg.NHead, cfg.NEmb)
	fmt.Println()

	// Create model
	model := transformer.NewGPT(cfg)

	// Create optimizer
	lr := float32(0.01)
	opt := optim.NewAdamW(model.Parameters(), lr)
	criterion := nn.NewCrossEntropyLoss()

	// Training loop
	fmt.Println("Training for 50 steps...")
	steps := 50
	batchSize := 4

	for step := 0; step < steps; step++ {
		// Get batch
		x, y := ds.GetBatch(batchSize)

		// Forward
		logits := model.Forward(x)

		// Loss
		b, t, v := logits.Shape[0], logits.Shape[1], logits.Shape[2]
		logitsFlat, _ := logits.View(b*t, v)
		loss := criterion.Forward(logitsFlat, y)

		if step%10 == 0 {
			fmt.Printf("Step %d | Loss: %.4f\n", step, loss)
		}

		// Backward
		opt.ZeroGrad()
		dLogitsFlat := criterion.Backward(logitsFlat, y)
		dLogits, _ := dLogitsFlat.View(b, t, v)
		model.Backward(dLogits)

		// Update
		opt.Step()
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Println()

	// Generate
	fmt.Println("Generating text with prompt: 'The quick'")
	prompt := "The quick"
	promptIds := tok.Encode(prompt)

	// Simple greedy generation
	generated := make([]int, len(promptIds))
	copy(generated, promptIds)

	for i := 0; i < 10; i++ {
		// Prepare context
		ctx := generated
		if len(ctx) > blockSize {
			ctx = ctx[len(ctx)-blockSize:]
		}

		// Create input tensor
		x := make([]float32, len(ctx))
		for j, id := range ctx {
			x[j] = float32(id)
		}

		// Import tensor package for creating input
		input := tensor.NewFromData(x, 1, len(ctx))

		// Forward
		logits := model.Forward(input)

		// Get last token logits
		vocabSize := logits.Shape[2]
		lastLogits := make([]float32, vocabSize)
		offset := (len(ctx) - 1) * vocabSize
		for j := 0; j < vocabSize; j++ {
			lastLogits[j] = logits.Data[offset+j]
		}

		// Greedy sample (argmax)
		maxIdx := 0
		maxVal := lastLogits[0]
		for j := 1; j < vocabSize; j++ {
			if lastLogits[j] > maxVal {
				maxVal = lastLogits[j]
				maxIdx = j
			}
		}

		generated = append(generated, maxIdx)
	}

	// Decode
	result := ""
	for _, id := range generated {
		if s, ok := tok.Decoder[id]; ok {
			result += s
		}
	}

	fmt.Println("Generated:", result)
	fmt.Println()
	fmt.Println("=== Demo Complete ===")
}
