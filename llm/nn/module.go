package nn

import (
	"github.com/brucetruth/minigpt/llm/tensor"
)

// Parameter represents a trainable tensor.
type Parameter struct {
	Data *tensor.NDArray
	Grad *tensor.NDArray
	Name string
}

// NewParameter creates a parameter with shape.
func NewParameter(name string, shape ...int) *Parameter {
	return &Parameter{
		Data: tensor.NewRandom(shape...), // Initialize randomly
		Grad: tensor.New(shape...),       // Initialize zero grad
		Name: name,
	}
}

// ZeroGrad zeroes out the gradient.
func (p *Parameter) ZeroGrad() {
	p.Grad = tensor.NewFull(0.0, p.Data.Shape...)
}

// Module interface for all neural network layers.
type Module interface {
	Forward(input *tensor.NDArray) *tensor.NDArray
	Backward(gradOutput *tensor.NDArray) *tensor.NDArray
	Parameters() []*Parameter
}
