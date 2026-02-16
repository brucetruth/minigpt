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
	fmt.Println("=== MiniGPT Advanced Demo ===")
	fmt.Println()

	// Set seed for reproducibility
	rand.Seed(42)

	// Larger training corpus
	text := `Once upon a time in a magical forest, there lived a wise old wizard named Merlin. 
Merlin had a young apprentice named Arthur who was eager to learn magic.
One day, Arthur asked, "Master, how can I become a great wizard like you?"
Merlin smiled and said, "Young Arthur, magic comes from within. You must believe in yourself."
Arthur practiced every day. He learned to cast spells, brew potions, and talk to animals.
The forest animals became his friends. The rabbits, the birds, and even the wise old owl.
One morning, Arthur discovered a hidden cave deep in the forest.
Inside the cave, he found a magical crystal that glowed with blue light.
The crystal spoke to him: "You have a pure heart, young wizard. Use my power wisely."
Arthur returned to Merlin with the crystal. Merlin was proud of his apprentice.
"You have learned well," said Merlin. "Now you are ready for your first quest."
Arthur set off on his adventure, carrying the magical crystal and the wisdom of his master.`

	fmt.Println("Training text length:", len(text), "characters")
	fmt.Println()

	// Train tokenizer
	fmt.Println("Training tokenizer...")
	tok := tokenizer.New()
	tok.Train(text, 256)
	tokens := tok.Encode(text)
	fmt.Printf("Tokenized: %d chars -> %d tokens\n", len(text), len(tokens))
	fmt.Println()

	// Create dataset
	ds := data.NewTextDataset(tokens, 32)

	// Larger model configuration
	cfg := transformer.Config{
		VocabSize: tok.VocabSize,
		BlockSize: 32,
		NLayer:    3,
		NHead:     4,
		NEmb:      128,
		PDrop:     0.1,
	}

	fmt.Printf("Model config: vocab=%d, block=%d, layers=%d, heads=%d, emb=%d\n",
		cfg.VocabSize, cfg.BlockSize, cfg.NLayer, cfg.NHead, cfg.NEmb)
	fmt.Println()

	// Initialize model
	model := transformer.NewGPT(cfg)

	// Optimizer with learning rate scheduling
	opt := optim.NewAdamW(model.Parameters(), 3e-3)
	scheduler := optim.NewCosineScheduleWithWarmup(20, 200, 3e-3, 1e-4)
	criterion := nn.NewCrossEntropyLoss()

	// Training loop
	fmt.Println("Training for 200 steps with LR scheduling and gradient clipping...")
	for step := 0; step < 200; step++ {
		// Update learning rate
		lr := scheduler.GetLR(step)
		opt.SetLR(lr)

		// Get batch
		x, y := ds.GetBatch(8)

		// Forward
		logits := model.Forward(x)

		// Loss
		b, t, v := logits.Shape[0], logits.Shape[1], logits.Shape[2]
		logitsFlat, _ := logits.View(b*t, v)
		loss := criterion.Forward(logitsFlat, y)

		// Print progress
		if step%20 == 0 {
			fmt.Printf("Step %d | Loss: %.4f | LR: %.6f\n", step, loss, lr)
		}

		// Backward
		opt.ZeroGrad()
		dLogitsFlat := criterion.Backward(logitsFlat, y)
		dLogits, _ := dLogitsFlat.View(b, t, v)
		model.Backward(dLogits)

		// Gradient clipping
		opt.ClipGradNorm(1.0)

		// Update
		opt.Step()
	}

	fmt.Println()
	fmt.Println("Training complete!")
	fmt.Println()

	// Generate with different sampling strategies
	prompts := []string{
		"Once upon a time",
		"Arthur",
		"Merlin said",
	}

	strategies := []struct {
		name string
		temp float32
		topK int
		topP float32
	}{
		{"Greedy (temp=0.8)", 0.8, 0, 1.0},
		{"Top-k=20 (temp=1.0)", 1.0, 20, 1.0},
		{"Top-p=0.9 (temp=1.0)", 1.0, 0, 0.9},
	}

	for _, prompt := range prompts {
		fmt.Println("─────────────────────────────────────")
		fmt.Printf("Prompt: '%s'\n", prompt)
		fmt.Println()

		for i, strat := range strategies {
			// Reset seed for different samples
			rand.Seed(42 + int64(i))

			ids := tok.Encode(prompt)

			// Generate 30 tokens
			for j := 0; j < 30; j++ {
				// Context window
				ctx := ids
				if len(ctx) > cfg.BlockSize {
					ctx = ctx[len(ctx)-cfg.BlockSize:]
				}

				// Prepare input [1, T]
				x := tensor.New(1, len(ctx))
				for k, id := range ctx {
					x.Data[k] = float32(id)
				}

				// Forward
				logits := model.Forward(x)

				// Get last token logits
				vocabSize := logits.Shape[2]
				lastLogits := tensor.New(vocabSize)
				offset := (len(ctx) - 1) * vocabSize
				for k := 0; k < vocabSize; k++ {
					lastLogits.Data[k] = logits.Data[offset+k]
				}

				// Sample with strategy
				nextID := tensor.SampleFromLogits(lastLogits, strat.temp, strat.topK, strat.topP)
				ids = append(ids, nextID)
			}

			// Decode and print
			generated := tok.Decode(ids)
			fmt.Printf("  [%s]\n", strat.name)
			fmt.Printf("  %s\n\n", generated)
		}
	}

	fmt.Println("=== Demo Complete ===")
	fmt.Println()
	fmt.Println("Key improvements demonstrated:")
	fmt.Println("  ✓ Larger model (3 layers, 128 dim, 4 heads)")
	fmt.Println("  ✓ MLP layers with GELU activation")
	fmt.Println("  ✓ LR scheduling (warmup + cosine decay)")
	fmt.Println("  ✓ Gradient clipping")
	fmt.Println("  ✓ Temperature / top-k / top-p sampling")
}
