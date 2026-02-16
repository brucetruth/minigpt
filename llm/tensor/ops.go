package tensor

import (
	"math"
	"math/rand"
)

// Add performs element-wise addition: out = a + b
func Add(a, b *NDArray) *NDArray {
	out := New(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// Sub performs element-wise subtraction: out = a - b
func Sub(a, b *NDArray) *NDArray {
	out := New(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

// Mul performs element-wise multiplication: out = a * b
func Mul(a, b *NDArray) *NDArray {
	if len(a.Data) != len(b.Data) {
		panic("shape mismatch in Mul")
	}
	out := New(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

// Div performs element-wise division: out = a / b
func Div(a, b *NDArray) *NDArray {
	out := New(a.Shape...)
	for i := range out.Data {
		out.Data[i] = a.Data[i] / b.Data[i]
	}
	return out
}

// MatMulImpl is the function pointer for MatMul. Defaults to PureGoMatMul.
var MatMulImpl = PureGoMatMul

// MatMul performs generic matrix multiplication using the configured backend.
func MatMul(a, b *NDArray) *NDArray {
	return MatMulImpl(a, b)
}

// PureGoMatMul performs generic matrix multiplication in pure Go.
// Supports broadcasting for last two dims if rank > 2 (batch matmul).
func PureGoMatMul(a, b *NDArray) *NDArray {
	// Assume A is [..., M, K], B is [..., K, N]
	// If rank 2: [M, K] * [K, N] -> [M, N]
	// If rank 3: [B, M, K] * [B, K, N] -> [B, M, N]
	rank := len(a.Shape)
	if rank < 2 {
		panic("matmul requires rank >= 2")
	}

	m := a.Shape[rank-2]
	k := a.Shape[rank-1]

	if b.Shape[rank-2] != k {
		panic("matmul shape mismatch inner dim")
	}
	n := b.Shape[rank-1]

	// Output shape
	outShape := make([]int, rank)
	copy(outShape, a.Shape)
	outShape[rank-1] = n

	out := New(outShape...)

	// Batch size calculation
	batch := 1
	for i := 0; i < rank-2; i++ {
		batch *= a.Shape[i]
	}

	// Loop over batches
	strideA := m * k
	strideB := k * n
	strideC := m * n

	for bIdx := 0; bIdx < batch; bIdx++ {
		offsetA := bIdx * strideA
		offsetB := bIdx * strideB
		offsetC := bIdx * strideC

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum float32
				for l := 0; l < k; l++ {
					valA := a.Data[offsetA+i*k+l]
					valB := b.Data[offsetB+l*n+j]
					sum += valA * valB
				}
				out.Data[offsetC+i*n+j] = sum
			}
		}
	}

	return out
}

// Transpose swaps the last two dimensions.
// For [B, M, N], returns [B, N, M].
// Returns a COPY of the data with new layout.
func Transpose(t *NDArray) *NDArray {
	rank := len(t.Shape)
	if rank < 2 {
		panic("transpose requires rank >= 2")
	}

	m := t.Shape[rank-2]
	n := t.Shape[rank-1]

	outShape := make([]int, rank)
	copy(outShape, t.Shape)
	outShape[rank-2] = n
	outShape[rank-1] = m

	out := New(outShape...)

	batch := 1
	for i := 0; i < rank-2; i++ {
		batch *= t.Shape[i]
	}

	strideIn := m * n
	strideOut := n * m

	for bIdx := 0; bIdx < batch; bIdx++ {
		offsetIn := bIdx * strideIn
		offsetOut := bIdx * strideOut

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				val := t.Data[offsetIn+i*n+j]
				out.Data[offsetOut+j*m+i] = val
			}
		}
	}

	return out
}

// Softmax along the last dimension.
// Numerical stability: subtract max.
func Softmax(t *NDArray) *NDArray {
	out := New(t.Shape...)

	rank := len(t.Shape)
	lastDim := t.Shape[rank-1]
	stride := lastDim

	total := t.Size

	for offset := 0; offset < total; offset += stride {
		// Find max for stability
		maxVal := float32(-math.MaxFloat32)
		for i := 0; i < stride; i++ {
			if t.Data[offset+i] > maxVal {
				maxVal = t.Data[offset+i]
			}
		}

		// Exp and sum
		var sum float32
		for i := 0; i < stride; i++ {
			v := float32(math.Exp(float64(t.Data[offset+i] - maxVal)))
			out.Data[offset+i] = v
			sum += v
		}

		// Normalize
		for i := 0; i < stride; i++ {
			out.Data[offset+i] /= sum
		}
	}

	return out
}

// Exp
func Exp(t *NDArray) *NDArray {
	out := New(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Exp(float64(v)))
	}
	return out
}

// Log
func Log(t *NDArray) *NDArray {
	out := New(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(math.Log(float64(v)))
	}
	return out
}

// Sum over last axis
func Sum(t *NDArray) *NDArray {
	rank := len(t.Shape)
	midShape := t.Shape[:rank-1]
	stride := t.Shape[rank-1]

	out := New(midShape...)

	for i := 0; i < out.Size; i++ {
		var s float32
		offset := i * stride
		for j := 0; j < stride; j++ {
			s += t.Data[offset+j]
		}
		out.Data[i] = s
	}
	return out
}

// Max value
func Max(t *NDArray) float32 {
	m := float32(-math.MaxFloat32)
	for _, v := range t.Data {
		if v > m {
			m = v
		}
	}
	return m
}

// ArgMax over last axis
func ArgMax(t *NDArray) []int {
	rank := len(t.Shape)
	stride := t.Shape[rank-1]
	batch := t.Size / stride

	out := make([]int, batch)

	for i := 0; i < batch; i++ {
		offset := i * stride
		maxVal := float32(-math.MaxFloat32)
		idx := 0
		for j := 0; j < stride; j++ {
			if t.Data[offset+j] > maxVal {
				maxVal = t.Data[offset+j]
				idx = j
			}
		}
		out[i] = idx
	}
	return out
}

// GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func GELU(t *NDArray) *NDArray {
	out := New(t.Shape...)
	c1 := float32(math.Sqrt(2.0 / math.Pi))
	c2 := float32(0.044715)

	for i, x := range t.Data {
		cube := x * x * x
		inner := c1 * (x + c2*cube)
		tanh := float32(math.Tanh(float64(inner)))
		out.Data[i] = 0.5 * x * (1.0 + tanh)
	}
	return out
}

// GELUWithCache computes GELU and returns cache needed for backward pass
func GELUWithCache(t *NDArray) (*NDArray, *NDArray) {
	out := New(t.Shape...)
	cache := New(t.Shape...) // Store input for backward

	c1 := float32(math.Sqrt(2.0 / math.Pi))
	c2 := float32(0.044715)

	for i, x := range t.Data {
		cache.Data[i] = x // Store input

		cube := x * x * x
		inner := c1 * (x + c2*cube)
		tanh := float32(math.Tanh(float64(inner)))
		out.Data[i] = 0.5 * x * (1.0 + tanh)
	}
	return out, cache
}

// GELUBackward computes gradient of GELU
// dL/dx = dL/dy * dy/dx
// dy/dx = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
// where z = sqrt(2/pi) * (x + 0.044715 * x^3)
// dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
func GELUBackward(gradOutput *NDArray, cache *NDArray) *NDArray {
	gradInput := New(cache.Shape...)

	c1 := float32(math.Sqrt(2.0 / math.Pi))
	c2 := float32(0.044715)

	for i, x := range cache.Data {
		cube := x * x * x
		inner := c1 * (x + c2*cube)
		tanh := float32(math.Tanh(float64(inner)))

		// sech^2(z) = 1 - tanh^2(z)
		sech2 := 1.0 - tanh*tanh

		// dz/dx
		dzdx := c1 * (1.0 + 3.0*c2*x*x)

		// dy/dx
		dydx := 0.5*(1.0+tanh) + 0.5*x*sech2*dzdx

		gradInput.Data[i] = gradOutput.Data[i] * dydx
	}

	return gradInput
}

// Clip values
func Clip(t *NDArray, min, max float32) {
	for i, v := range t.Data {
		if v < min {
			t.Data[i] = min
		} else if v > max {
			t.Data[i] = max
		}
	}
}

// Dropout with fixed seed for determinism.
// Returns (output, mask)
func Dropout(t *NDArray, p float32) (*NDArray, *NDArray) {
	out := New(t.Shape...)
	mask := New(t.Shape...)
	scale := 1.0 / (1.0 - p)

	for i := range t.Data {
		if rand.Float32() > p {
			mask.Data[i] = 1.0
			out.Data[i] = t.Data[i] * scale
		} else {
			mask.Data[i] = 0.0
			out.Data[i] = 0.0
		}
	}
	return out, mask
}
