package nn

import (
	"github.com/brucetruth/minigpt/llm/tensor"
)

// Embedding: Lookup table.
// Inputs: indices [B, T], Output: [B, T, Emb]
type Embedding struct {
	Weight *Parameter // [Vocab, Emb]
	
	inputIndices []int // Cache for backward
}

func NewEmbedding(vocabSize, embDim int) *Embedding {
	return &Embedding{
		Weight: &Parameter{
			Data: tensor.NewRandom(vocabSize, embDim), // Usually normal(0, 1)
			Grad: tensor.New(vocabSize, embDim),
			Name: "embedding",
		},
	}
}

func (e *Embedding) ForwardIndices(indices []int, batch, seqLen int) *tensor.NDArray {
	e.inputIndices = indices
	embDim := e.Weight.Data.Shape[1]
	
	out := tensor.New(batch, seqLen, embDim)
	
	for i, idx := range indices {
		offsetOut := i * embDim
		offsetEmb := idx * embDim
		
		for j := 0; j < embDim; j++ {
			out.Data[offsetOut+j] = e.Weight.Data.Data[offsetEmb+j]
		}
	}
	return out
}

func (e *Embedding) Backward(gradOutput *tensor.NDArray) {
	// gradOutput: [B, T, Emb]
	// Accumulate into Weight.Grad based on indices.
	// This is sparse update.
	
	embDim := e.Weight.Data.Shape[1]
	
	for i, idx := range e.inputIndices {
		offsetGrad := i * embDim
		offsetParam := idx * embDim
		
		for j := 0; j < embDim; j++ {
			e.Weight.Grad.Data[offsetParam+j] += gradOutput.Data[offsetGrad+j]
		}
	}
}

func (e *Embedding) Parameters() []*Parameter {
	return []*Parameter{e.Weight}
}
