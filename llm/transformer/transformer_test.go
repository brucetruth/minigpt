package transformer

import (
	"math"
	"math/rand"
	"testing"

	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/tensor"
)

func TestTransformerGradient(t *testing.T) {
	// Tiny Config
	cfg := Config{
		VocabSize: 20,
		BlockSize: 4,
		NLayer:    1,
		NHead:     2,
		NEmb:      8,
		PDrop:     0.0,
	}

	gpt := NewGPT(cfg)

	// Input
	batch := 2
	seqLen := 4

	x := tensor.New(batch, seqLen)
	// Random tokens
	for i := range x.Data {
		x.Data[i] = float32(rand.Intn(cfg.VocabSize))
	}

	// Forward
	logits := gpt.Forward(x) // [B, T, V]

	// Create dummy targets
	targets := make([]int, batch*seqLen)
	for i := range targets {
		targets[i] = rand.Intn(cfg.VocabSize)
	}

	// Loss
	lossFn := nn.NewCrossEntropyLoss()

	// Flatten for loss
	b, timeSteps, v := logits.Shape[0], logits.Shape[1], logits.Shape[2]
	logitsFlat, _ := logits.View(b*timeSteps, v)

	loss := lossFn.Forward(logitsFlat, targets)
	t.Logf("Initial Loss: %f", loss)

	// Backward
	dLogitsFlat := lossFn.Backward(logitsFlat, targets)
	dLogits, _ := dLogitsFlat.View(b, timeSteps, v)
	gpt.Backward(dLogits)

	// Check Gradient for a weight: WTE.Weight
	param := gpt.WTE.Weight
	idxToCheck := 0

	analyticalGrad := param.Grad.Data[idxToCheck]

	// Finite Difference
	epsilon := float32(1e-3)

	origVal := param.Data.Data[idxToCheck]

	// f(x + eps)
	param.Data.Data[idxToCheck] = origVal + epsilon
	outPlus := gpt.Forward(x)
	outPlusFlat, _ := outPlus.View(b*timeSteps, v)
	lossPlus := lossFn.Forward(outPlusFlat, targets) // lPlus was unused

	// f(x - eps)
	param.Data.Data[idxToCheck] = origVal - epsilon
	outMinus := gpt.Forward(x)
	outMinusFlat, _ := outMinus.View(b*timeSteps, v)
	lossMinus := lossFn.Forward(outMinusFlat, targets)

	// Restore
	param.Data.Data[idxToCheck] = origVal

	numericalGrad := (lossPlus - lossMinus) / (2 * epsilon)

	t.Logf("Analytical: %f, Numerical: %f", analyticalGrad, numericalGrad)

	diff := math.Abs(float64(analyticalGrad - numericalGrad))
	if diff > 1e-4 { // Slightly loose tolerance for float32
		t.Errorf("Gradient check failed! Analytical: %f, Numerical: %f", analyticalGrad, numericalGrad)
	}
}

func TestDeterministicTraining(t *testing.T) {
	// Random tokens check
	tok := float32(rand.Intn(100))

	rand.Seed(42)
	cfg := Config{
		VocabSize: 100,
		BlockSize: 8,
		NLayer:    2,
		NHead:     2,
		NEmb:      16,
		PDrop:     0.1,
	}

	gpt := NewGPT(cfg)
	// Mock optimize step or check just forward consistency

	x := tensor.New(1, 8)
	x.Data[0] = tok

	out1 := gpt.Forward(x)
	val1 := out1.Data[0]

	// Reset seed
	rand.Seed(42)
	gpt2 := NewGPT(cfg)
	x2 := tensor.New(1, 8)
	x2.Data[0] = tok
	out2 := gpt2.Forward(x2)
	val2 := out2.Data[0]

	if val1 != val2 {
		t.Errorf("Non-deterministic! %f != %f", val1, val2)
	}
}
