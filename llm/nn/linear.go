package nn

import (
	"fmt"

	"github.com/brucetruth/minigpt/llm/tensor"
)

// Linear layer: Y = X W^T + B
type Linear struct {
	W *Parameter // [Out, In]
	B *Parameter // [Out]

	// Cache for backward
	input *tensor.NDArray
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	// Initialize weights: He init or similar
	// For simplicity: uniform random scaled by 1/sqrt(in)
	w := tensor.NewRandom(outFeatures, inFeatures)
	scale := float32(1.0 / float32(inFeatures)) // simplified scale
	for i := range w.Data {
		w.Data[i] = (w.Data[i] - 0.5) * scale // Center around 0
	}

	return &Linear{
		W: &Parameter{Data: w, Grad: tensor.New(outFeatures, inFeatures), Name: "weight"},
		B: &Parameter{Data: tensor.New(outFeatures), Grad: tensor.New(outFeatures), Name: "bias"},
	}
}

func (l *Linear) Forward(x *tensor.NDArray) *tensor.NDArray {
	l.input = x
	// X: [B, ..., In], W: [Out, In] -> W^T: [In, Out]
	// Broadcast W over X.

	// Reshape X to [N, In]
	shape := x.Shape
	rank := len(shape)
	inDim := shape[rank-1]

	if inDim != l.W.Data.Shape[1] {
		panic(fmt.Sprintf("linear shape mismatch: input %v, weight %v", shape, l.W.Data.Shape))
	}

	numElements := x.Size / inDim
	xFlat, _ := x.View(numElements, inDim)

	wt := tensor.Transpose(l.W.Data)
	outFlat := tensor.MatMul(xFlat, wt) // [N, Out]

	// Reshape back
	outShape := make([]int, rank)
	copy(outShape, shape)
	outShape[rank-1] = l.W.Data.Shape[0] // OutDim

	out, _ := outFlat.View(outShape...)

	// Add bias
	stride := l.B.Data.Size
	total := out.Size

	for i := 0; i < total; i++ {
		bIdx := i % stride
		out.Data[i] += l.B.Data.Data[bIdx]
	}
	return out
}

func (l *Linear) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	// gradOutput: [B, ..., Out]
	// dW = gradOutput^T * input  (summed over batch)
	// dB = sum(gradOutput, axis=0..N-2)
	// dInput = gradOutput * W

	// Reshape to 2D for easier math: [Batch*Time, In/Out]
	// But let's keep it robust.

	// 1. dInput = gradOutput * W
	// [B, Out] * [Out, In] -> [B, In]

	// Reshape gradOutput to [N, Out]
	gradShape := gradOutput.Shape
	rank := len(gradShape)
	outDim := gradShape[rank-1]

	numElements := gradOutput.Size / outDim
	gradFlat, _ := gradOutput.View(numElements, outDim)

	dInputFlat := tensor.MatMul(gradFlat, l.W.Data) // [N, In]

	// Reshape dInput back
	inDim := l.W.Data.Shape[1]
	dInputShape := make([]int, rank)
	copy(dInputShape, gradShape)
	dInputShape[rank-1] = inDim

	dInput, _ := dInputFlat.View(dInputShape...)

	// 2. dW
	// We need input^T * gradOutput.
	// If input is [B, T, In] and grad is [B, T, Out].
	// Reshape to [B*T, In] and [B*T, Out].
	// Then (In, B*T) * (B*T, Out) -> (In, Out). Then transpose to (Out, In)?
	// Or (Out, B*T) * (B*T, In) -> (Out, In).
	// Let's rely on iterating.

	// Implementing gradient accumulation for W and B
	// dW: [Out, In]

	batchSize := l.input.Size / l.W.Data.Shape[1] // Total tokens
	// inDim is already defined
	outDim = l.W.Data.Shape[0]

	// Accumulate into l.W.Grad
	for i := 0; i < batchSize; i++ {
		offsetIn := i * inDim
		offsetGrad := i * outDim

		for r := 0; r < outDim; r++ {
			gradVal := gradOutput.Data[offsetGrad+r]
			for c := 0; c < inDim; c++ {
				val := gradVal * l.input.Data[offsetIn+c]
				l.W.Grad.Data[r*inDim+c] += val
			}
		}

		// Accumulate bias
		for r := 0; r < outDim; r++ {
			l.B.Grad.Data[r] += gradOutput.Data[offsetGrad+r]
		}
	}

	return dInput
}

func (l *Linear) Parameters() []*Parameter {
	return []*Parameter{l.W, l.B}
}
