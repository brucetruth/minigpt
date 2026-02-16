package tensor

import (
	"math"
	"testing"
)

func TestMatMul(t *testing.T) {
	// A: 2x3, B: 3x2
	a := NewFromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := NewFromData([]float32{7, 8, 9, 10, 11, 12}, 3, 2)
	
	c := MatMul(a, b)
	
	if c.Shape[0] != 2 || c.Shape[1] != 2 {
		t.Fatalf("Expected shape [2, 2], got %v", c.Shape)
	}
	
	// Expected:
	// ROW 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// ROW 0,1: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	// ROW 1,0: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// ROW 1,1: 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	
	expected := []float32{58, 64, 139, 154}
	for i, v := range c.Data {
		if math.Abs(float64(v - expected[i])) > 1e-5 {
			t.Errorf("At %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestTranspose(t *testing.T) {
	a := NewFromData([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := Transpose(a)
	
	if b.Shape[0] != 3 || b.Shape[1] != 2 {
		t.Fatalf("Expected shape [3, 2], got %v", b.Shape)
	}
	
	// Expected:
	// 1 4
	// 2 5
	// 3 6
	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range b.Data {
		if v != expected[i] {
			t.Errorf("At %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestSoftmax(t *testing.T) {
	a := NewFromData([]float32{0, 0}, 1, 2)
	s := Softmax(a)
	
	// Expected: 0.5, 0.5
	if math.Abs(float64(s.Data[0] - 0.5)) > 1e-5 {
		t.Errorf("Expected 0.5, got %f", s.Data[0])
	}
}

func TestGELU(t *testing.T) {
	a := NewFromData([]float32{0}, 1)
	g := GELU(a)
	// GELU(0) = 0
	if math.Abs(float64(g.Data[0])) > 1e-5 {
		t.Errorf("Expected 0, got %f", g.Data[0])
	}
}
