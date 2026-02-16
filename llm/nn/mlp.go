package nn

import (
	"github.com/brucetruth/minigpt/llm/tensor"
)

// MLP (Multi-Layer Perceptron / Feed-Forward Network)
// Standard transformer MLP: Linear -> GELU -> Linear -> Dropout
type MLP struct {
	FC1  *Linear
	FC2  *Linear
	Drop *Dropout

	// Cache for backward
	input     *tensor.NDArray
	fc1Out    *tensor.NDArray
	geluOut   *tensor.NDArray
	geluCache *tensor.NDArray // For GELU backward
}

func NewMLP(nEmb int, dropoutP float32) *MLP {
	return &MLP{
		FC1:  NewLinear(nEmb, 4*nEmb),
		FC2:  NewLinear(4*nEmb, nEmb),
		Drop: NewDropout(dropoutP),
	}
}

func (m *MLP) Forward(x *tensor.NDArray) *tensor.NDArray {
	m.input = x

	// Linear 1
	m.fc1Out = m.FC1.Forward(x)

	// GELU activation
	m.geluOut, m.geluCache = tensor.GELUWithCache(m.fc1Out)

	// Linear 2
	fc2Out := m.FC2.Forward(m.geluOut)

	// Dropout
	out := m.Drop.Forward(fc2Out)

	return out
}

func (m *MLP) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	// Backward through dropout
	dFC2 := m.Drop.Backward(gradOutput)

	// Backward through FC2
	dGELU := m.FC2.Backward(dFC2)

	// Backward through GELU
	dFC1 := tensor.GELUBackward(dGELU, m.geluCache)

	// Backward through FC1
	dInput := m.FC1.Backward(dFC1)

	return dInput
}

func (m *MLP) Parameters() []*Parameter {
	params := m.FC1.Parameters()
	params = append(params, m.FC2.Parameters()...)
	return params
}
