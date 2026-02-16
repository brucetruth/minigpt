package nn

import (
	"math"
	"testing"

	"github.com/brucetruth/minigpt/llm/tensor"
)

func TestLayerNorm(t *testing.T) {
	ln := NewLayerNorm(4)
	x := tensor.NewFromData([]float32{1, 2, 3, 4}, 1, 4)
	out := ln.Forward(x)
	
	// Mean of 1,2,3,4 is 2.5
	// Var is ((1-2.5)^2 + ... + (4-2.5)^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 5/4 = 1.25
	// Std = sqrt(1.25) approx 1.118
	// Expected out[0] = (1-2.5)/1.118 = -1.341
	
	expected := float32((1.0 - 2.5) / math.Sqrt(1.25))
	if math.Abs(float64(out.Data[0] - expected)) > 1e-4 {
		t.Errorf("Expected %f, got %f", expected, out.Data[0])
	}
}

func TestSoftmaxStability(t *testing.T) {
	// Large values to test stability
	x := tensor.NewFromData([]float32{1000, 1001}, 1, 2)
	out := tensor.Softmax(x)
	
	// exp(1000-1001) / (exp(1000-1001) + exp(1001-1001))
	// = exp(-1) / (exp(-1) + 1) vs exp(0) / ...
	// max is 1001.
	// exp(1000-1001) = exp(-1) approx 0.3678
	// exp(1001-1001) = 1
	// sum = 1.3678
	// out[1] = 1 / 1.3678 approx 0.731
	
	expected := float32(1.0 / (1.0 + math.Exp(-1)))
	if math.Abs(float64(out.Data[1] - expected)) > 1e-4 {
		t.Errorf("Expected %f, got %f", expected, out.Data[1])
	}
}
