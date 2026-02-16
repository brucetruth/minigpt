package nn

import (
	"math"

	"github.com/brucetruth/minigpt/llm/tensor"
)

// CrossEntropyLoss combines LogSoftmax and NLLLoss.
// Expects logits [B*T, Vocab] and targets [B*T].
type CrossEntropyLoss struct {
}

func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

// Forward returns scalar loss.
// logits: [N, C]
// targets: [N] (indices)
func (l *CrossEntropyLoss) Forward(logits *tensor.NDArray, targets []int) float32 {
	// 1. Softmax
	probs := tensor.Softmax(logits)
	
	// 2. NLL: -log(probs[target])
	var totalLoss float32
	batchSize := logits.Shape[0]
	vocabSize := logits.Shape[1]
	
	for i := 0; i < batchSize; i++ {
		targetErr := targets[i]
		prob := probs.Data[i*vocabSize + targetErr]
		// Clip for stability
		if prob < 1e-10 {
			prob = 1e-10
		}
		totalLoss += -float32(math.Log(float64(prob)))
	}
	
	return totalLoss / float32(batchSize)
}

// Backward returns gradients for logits.
// dL/dz_i = p_i - y_i
func (l *CrossEntropyLoss) Backward(logits *tensor.NDArray, targets []int) *tensor.NDArray {
	probs := tensor.Softmax(logits)
	dLogits := tensor.New(logits.Shape...)
	
	batchSize := logits.Shape[0]
	vocabSize := logits.Shape[1]
	scale := 1.0 / float32(batchSize)
	
	copy(dLogits.Data, probs.Data)
	
	for i := 0; i < batchSize; i++ {
		target := targets[i]
		idx := i*vocabSize + target
		dLogits.Data[idx] -= 1.0
	}
	
	// Scale by 1/BatchSize
	for i := range dLogits.Data {
		dLogits.Data[i] *= scale
	}
	
	return dLogits
}
