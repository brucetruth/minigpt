package transformer

import (
	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/tensor"
)

type GPT struct {
	Config Config
	
	WTE    *nn.Embedding
	WPE    *nn.Embedding
	Drop   *nn.Dropout
	Blocks []*Block
	LNF    *nn.LayerNorm
	LMHead *nn.Linear
}

func NewGPT(cfg Config) *GPT {
	gpt := &GPT{
		Config: cfg,
		WTE:    nn.NewEmbedding(cfg.VocabSize, cfg.NEmb),
		WPE:    nn.NewEmbedding(cfg.BlockSize, cfg.NEmb),
		Drop:   nn.NewDropout(cfg.PDrop),
		LNF:    nn.NewLayerNorm(cfg.NEmb),
		LMHead: nn.NewLinear(cfg.NEmb, cfg.VocabSize),
	}
	
	gpt.Blocks = make([]*Block, cfg.NLayer)
	for i := 0; i < cfg.NLayer; i++ {
		gpt.Blocks[i] = NewBlock(cfg)
	}
	
	return gpt
}

func (gpt *GPT) Forward(idx *tensor.NDArray) *tensor.NDArray {
	// idx: [B, T]
	B, T := idx.Shape[0], idx.Shape[1]
	
	// Embeddings
	// Make pos indices [0, 1, ..., T-1]
	posIdx := make([]int, T)
	for i := range posIdx {
		posIdx[i] = i
	}
	// For each batch, use same pos indices
	// Need to check how Embedding expects indices.
	// Embedding.ForwardIndices expects raw []int slice.
	// We need to extract data from NDArray idx.
	
	// Flatten idx.Data to []int
	flatIdx := make([]int, B*T)
	for i := range flatIdx {
		flatIdx[i] = int(idx.Data[i])
	}
	
	tokEmb := gpt.WTE.ForwardIndices(flatIdx, B, T) // [B, T, C]
	
	// Pos Embedding
	// Repeat posIdx for batch?
	// WPE ForwardIndices expects flat list.
	flatPos := make([]int, B*T)
	for b := 0; b < B; b++ {
		for t := 0; t < T; t++ {
			flatPos[b*T + t] = t
		}
	}
	posEmb := gpt.WPE.ForwardIndices(flatPos, B, T)
	
	x := tensor.Add(tokEmb, posEmb)
	x = gpt.Drop.Forward(x)
	
	// Blocks
	for _, block := range gpt.Blocks {
		x = block.Forward(x)
	}
	
	x = gpt.LNF.Forward(x)
	
	// LM Head
	logits := gpt.LMHead.Forward(x)
	return logits
}

func (gpt *GPT) Backward(gradOutput *tensor.NDArray) {
	// gradOutput: [B, T, Vocab]
	
	dHead := gpt.LMHead.Backward(gradOutput)
	dLNF := gpt.LNF.Backward(dHead)
	
	dout := dLNF
	for i := len(gpt.Blocks) - 1; i >= 0; i-- {
		dout = gpt.Blocks[i].Backward(dout)
	}
	
	dDrop := gpt.Drop.Backward(dout)
	
	// dDrop splits to dWTE and dWPE (add node)
	gpt.WTE.Backward(dDrop)
	gpt.WPE.Backward(dDrop)
}

func (gpt *GPT) Parameters() []*nn.Parameter {
	var params []*nn.Parameter
	params = append(params, gpt.WTE.Parameters()...)
	params = append(params, gpt.WPE.Parameters()...)
	for _, b := range gpt.Blocks {
		params = append(params, b.Parameters()...)
	}
	params = append(params, gpt.LNF.Parameters()...)
	params = append(params, gpt.LMHead.Parameters()...)
	return params
}
