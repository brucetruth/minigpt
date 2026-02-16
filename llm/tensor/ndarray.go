package tensor

import (
	"fmt"
	"math/rand"
)

// DType is float32 as requested
type DType float32

// NDArray represents a multi-dimensional array.
// We assume row-major contiguous storage for simplicity in this implementation,
// but shape and strides are kept for potential view support.
type NDArray struct {
	Data    []float32
	Shape   []int
	Strides []int
	Size    int
}

// New creates a new NDArray with shape, initialized to 0.
func New(shape ...int) *NDArray {
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	strides := computeStrides(shape)
	return &NDArray{
		Data:    data,
		Shape:   makeCopy(shape),
		Strides: strides,
		Size:    size,
	}
}

// NewRandom creates a new NDArray with shape, initialized to random values in [0, 1).
// Deterministic if rand.Seed is set globally (which we will control).
func NewRandom(shape ...int) *NDArray {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = rand.Float32()
	}
	return t
}

// NewFull creates a new NDArray with shape, filled with val.
func NewFull(val float32, shape ...int) *NDArray {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = val
	}
	return t
}

// NewFromData creates a new NDArray wrapping existing data.
// It copies the data to ensure ownership.
func NewFromData(data []float32, shape ...int) *NDArray {
	t := New(shape...)
	copy(t.Data, data)
	return t
}

func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

func makeCopy(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

// View returns a view of the tensor with new shape.
// Fails if size mismatch.
func (t *NDArray) View(shape ...int) (*NDArray, error) {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if size != t.Size {
		return nil, fmt.Errorf("view size %d does not match tensor size %d", size, t.Size)
	}
	return &NDArray{
		Data:    t.Data, // Share data
		Shape:   makeCopy(shape),
		Strides: computeStrides(shape),
		Size:    size,
	}, nil
}

// Clone returns a deep copy.
func (t *NDArray) Clone() *NDArray {
	c := New(t.Shape...)
	copy(c.Data, t.Data)
	return c
}

// String returns a summary string.
func (t *NDArray) String() string {
	return fmt.Sprintf("NDArray%v", t.Shape)
}
