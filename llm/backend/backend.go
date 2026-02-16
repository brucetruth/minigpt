package backend

import (
	"github.com/brucetruth/minigpt/llm/tensor"
)

type Backend interface {
	MatMul(a, b *tensor.NDArray, out *tensor.NDArray)
}

var Current Backend = &PureGoBackend{}

type PureGoBackend struct{}

func (b *PureGoBackend) MatMul(a, rhs *tensor.NDArray, out *tensor.NDArray) {
	// Call Pure Go implementation
	// We need to move the actual loop here or keep it in tensor and call it?
	// To avoid circular imports, tensor depends on backend or backend depends on tensor?
	// tensor package defines NDArray. backend uses NDArray.
	// tensor package should call backend.
	// So tensor imports backend.
	// backend imports tensor. CIRCULAR!

	// Solution: Define interface in a separate package or keep it simple.
	// Prompt says: "llm/backend/ (Backend interface...)"
	// "tensor ... Ops ... matmul ..."
	// If tensor ops call backend, backend cannot import tensor.
	// We can use interface{} or a reduced type.
	// OR `llm/tensor` defines the interface and `llm/backend` implements it?
	// But `llm/backend` is a separate folder.

	// Best approach:
	// `llm/tensor` does not import `llm/backend`.
	// `llm/tensor` has a variable `MatMulImpl func(a, b *NDArray) *NDArray`
	// `llm/backend` sets this variable in `init()`.
	// But `llm/backend` must import `llm/tensor` to see struct layout.
	// This works.
}
