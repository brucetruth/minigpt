package transformer

import (
	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/tensor"
)

type Block struct {
	LN1   *nn.LayerNorm
	Attn  *CausalSelfAttention
	LN2   *nn.LayerNorm
	MLP   *nn.MLP
	Drop1 *nn.Dropout // Residual dropout after attention
	Drop2 *nn.Dropout // Residual dropout after MLP
}

func NewBlock(cfg Config) *Block {
	return &Block{
		LN1:   nn.NewLayerNorm(cfg.NEmb),
		Attn:  NewCausalSelfAttention(cfg),
		LN2:   nn.NewLayerNorm(cfg.NEmb),
		MLP:   nn.NewMLP(cfg.NEmb, cfg.PDrop),
		Drop1: nn.NewDropout(cfg.PDrop), // Residual dropout
		Drop2: nn.NewDropout(cfg.PDrop), // Residual dropout
	}
}

func (b *Block) Forward(x *tensor.NDArray) *tensor.NDArray {
	// x = x + dropout(attn(ln1(x)))
	normalized := b.LN1.Forward(x)
	attnOut := b.Attn.Forward(normalized)
	attnOut = b.Drop1.Forward(attnOut)
	x = tensor.Add(x, attnOut)

	// x = x + dropout(mlp(ln2(x)))
	normalized2 := b.LN2.Forward(x)
	mlpOut := b.MLP.Forward(normalized2)
	mlpOut = b.Drop2.Forward(mlpOut)
	x = tensor.Add(x, mlpOut)

	return x
}

func (b *Block) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	// Backward through second residual: x = x + drop2(mlp(ln2(x)))
	// gradOutput flows to both branches

	// Branch 2: MLP path
	dDrop2 := b.Drop2.Backward(gradOutput)
	dMLP := b.MLP.Backward(dDrop2)
	dLN2 := b.LN2.Backward(dMLP)

	// Combine gradient at x (residual adds gradients)
	dx_mid := tensor.Add(gradOutput, dLN2)

	// Branch 1: Attention path
	dDrop1 := b.Drop1.Backward(dx_mid)
	dAttn := b.Attn.Backward(dDrop1)
	dLN1 := b.LN1.Backward(dAttn)

	// Final dx = dx_mid + dLN1
	return tensor.Add(dx_mid, dLN1)
}

func (b *Block) Parameters() []*nn.Parameter {
	p := b.LN1.Parameters()
	p = append(p, b.Attn.Parameters()...)
	p = append(p, b.LN2.Parameters()...)
	p = append(p, b.MLP.Parameters()...)
	return p
}
