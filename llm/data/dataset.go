package data

import (
	"math/rand"

	"github.com/brucetruth/minigpt/llm/tensor"
)

type TextDataset struct {
	Tokens    []int
	BlockSize int
}

func NewTextDataset(tokens []int, blockSize int) *TextDataset {
	return &TextDataset{
		Tokens:    tokens,
		BlockSize: blockSize,
	}
}

// GetBatch returns (x, y) tensors.
// x: [B, T], y: [B, T] (shifted)
func (ds *TextDataset) GetBatch(batchSize int) (*tensor.NDArray, []int) {
	x := tensor.New(batchSize, ds.BlockSize)
	y := make([]int, batchSize*ds.BlockSize)

	maxOffset := len(ds.Tokens) - ds.BlockSize - 1
	if maxOffset <= 0 {
		return x, y // Empty or error
	}

	for b := 0; b < batchSize; b++ {
		offset := rand.Intn(maxOffset)

		// Fill x and y
		for t := 0; t < ds.BlockSize; t++ {
			token := ds.Tokens[offset+t]
			target := ds.Tokens[offset+t+1]

			x.Data[b*ds.BlockSize+t] = float32(token) // Embedding expects float indices if using Gather? No, we use Ints usually
			// Our Embedding.ForwardIndices takes []int.
			// However, our standard pipeline might pass NDArray.
			// We made GPT.Forward take NDArray[float32], but internally cast to int.

			y[b*ds.BlockSize+t] = target
		}
	}

	return x, y
}
