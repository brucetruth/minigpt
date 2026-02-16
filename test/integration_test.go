package test

import (
	"math/rand"
	"testing"

	"github.com/brucetruth/minigpt/llm/data"
	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/optim"
	"github.com/brucetruth/minigpt/llm/tensor"
	"github.com/brucetruth/minigpt/llm/tokenizer"
	"github.com/brucetruth/minigpt/llm/transformer"
)

func TestEndToEndTraining(t *testing.T) {
	// Set seed for reproducibility
	rand.Seed(123)

	// Small training corpus
	text := "hello world. world is great. great things happen. happen to be here. here we go. go world go."

	// Train tokenizer
	tok := tokenizer.New()
	tok.Train(text, 50)

	ids := tok.Encode(text)
	if len(ids) == 0 {
		t.Fatal("Failed to tokenize text")
	}

	t.Logf("Tokenized %d chars to %d tokens", len(text), len(ids))

	// Create dataset
	blockSize := 8
	ds := data.NewTextDataset(ids, blockSize)

	// Tiny model config
	cfg := transformer.Config{
		VocabSize: tok.VocabSize,
		BlockSize: blockSize,
		NLayer:    1,
		NHead:     2,
		NEmb:      16,
		PDrop:     0.0,
	}

	// Create model
	model := transformer.NewGPT(cfg)

	// Optimizer
	lr := float32(0.01)
	opt := optim.NewAdamW(model.Parameters(), lr)
	criterion := nn.NewCrossEntropyLoss()

	// Training
	steps := 30
	batchSize := 2

	var initialLoss, finalLoss float32

	for step := 0; step < steps; step++ {
		x, y := ds.GetBatch(batchSize)

		logits := model.Forward(x)

		b, timeSteps, v := logits.Shape[0], logits.Shape[1], logits.Shape[2]
		logitsFlat, _ := logits.View(b*timeSteps, v)
		loss := criterion.Forward(logitsFlat, y)

		if step == 0 {
			initialLoss = loss
			t.Logf("Initial loss: %.4f", loss)
		}

		if step == steps-1 {
			finalLoss = loss
			t.Logf("Final loss: %.4f", loss)
		}

		opt.ZeroGrad()
		dLogitsFlat := criterion.Backward(logitsFlat, y)
		dLogits, _ := dLogitsFlat.View(b, timeSteps, v)
		model.Backward(dLogits)
		opt.Step()
	}

	// Verify loss decreased
	if finalLoss >= initialLoss {
		t.Errorf("Loss did not decrease: initial=%.4f, final=%.4f", initialLoss, finalLoss)
	}

	// Test generation (just verify it doesn't crash)
	prompt := "hello"
	promptIds := tok.Encode(prompt)

	if len(promptIds) == 0 {
		t.Fatal("Failed to tokenize prompt")
	}

	// Generate a few tokens
	ctx := promptIds
	if len(ctx) > blockSize {
		ctx = ctx[len(ctx)-blockSize:]
	}

	x := tensor.New(1, len(ctx))
	for j, id := range ctx {
		x.Data[j] = float32(id)
	}

	logits := model.Forward(x)

	// Verify output shape
	expectedShape := []int{1, len(ctx), cfg.VocabSize}
	if len(logits.Shape) != 3 {
		t.Errorf("Expected 3D output, got shape %v", logits.Shape)
	}
	for i := 0; i < 3; i++ {
		if logits.Shape[i] != expectedShape[i] {
			t.Errorf("Output shape mismatch at dim %d: expected %d, got %d",
				i, expectedShape[i], logits.Shape[i])
		}
	}

	t.Log("End-to-end test passed!")
}
