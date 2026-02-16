package nn

import (
	"math"

	"github.com/brucetruth/minigpt/llm/tensor"
)

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
type LayerNorm struct {
	Gamma *Parameter // [Dim]
	Beta  *Parameter // [Dim]
	Eps   float32
	
	// Cache
	input *tensor.NDArray
	mean  []float32 // [Batch]
	rstd  []float32 // [Batch]
}

func NewLayerNorm(dim int) *LayerNorm {
	ln := &LayerNorm{
		Gamma: &Parameter{Data: tensor.NewFull(1.0, dim), Grad: tensor.New(dim), Name: "ln_gamma"},
		Beta:  &Parameter{Data: tensor.NewFull(0.0, dim), Grad: tensor.New(dim), Name: "ln_beta"},
		Eps:   1e-5,
	}
	return ln
}

func (ln *LayerNorm) Forward(x *tensor.NDArray) *tensor.NDArray {
	ln.input = x
	dim := x.Shape[len(x.Shape)-1]
	batch := x.Size / dim
	
	out := tensor.New(x.Shape...)
	ln.mean = make([]float32, batch)
	ln.rstd = make([]float32, batch)
	
	gamma := ln.Gamma.Data.Data
	beta := ln.Beta.Data.Data
	
	for b := 0; b < batch; b++ {
		offset := b * dim
		
		// Mean
		var sum float32
		for i := 0; i < dim; i++ {
			sum += x.Data[offset+i]
		}
		mean := sum / float32(dim)
		ln.mean[b] = mean
		
		// Var
		var sumSq float32
		for i := 0; i < dim; i++ {
			diff := x.Data[offset+i] - mean
			sumSq += diff * diff
		}
		variance := sumSq / float32(dim)
		rstd := float32(1.0 / math.Sqrt(float64(variance)+float64(ln.Eps)))
		ln.rstd[b] = rstd
		
		// Normalize and Scale
		for i := 0; i < dim; i++ {
			normalized := (x.Data[offset+i] - mean) * rstd
			out.Data[offset+i] = normalized*gamma[i] + beta[i]
		}
	}
	
	return out
}

func (ln *LayerNorm) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	dim := ln.input.Shape[len(ln.input.Shape)-1]
	batch := ln.input.Size / dim
	
	dInput := tensor.New(ln.input.Shape...)
	
	gamma := ln.Gamma.Data.Data
	
	for b := 0; b < batch; b++ {
		offset := b * dim
		mean := ln.mean[b]
		rstd := ln.rstd[b]
		
		// 1. Calculate intermediate gradients
		// dL/d(x_hat) = dL/dy * gamma
		// Also accumulate dGamma, dBeta
		
		var sumDxHat float32
		var sumDxHatXHat float32
		
		dxHat := make([]float32, dim)
		xHat := make([]float32, dim)

		for i := 0; i < dim; i++ {
			dy := gradOutput.Data[offset+i]
			x_h := (ln.input.Data[offset+i] - mean) * rstd
			xHat[i] = x_h

			// Grads for params
			ln.Gamma.Grad.Data[i] += dy * x_h
			ln.Beta.Grad.Data[i] += dy
			
			// dx_hat
			dx_h := dy * gamma[i]
			dxHat[i] = dx_h
			
			sumDxHat += dx_h
			sumDxHatXHat += dx_h * x_h
		}
		
		// 2. Calculate dInput
		// dx = (1/N) * rstd * (N*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
		
		scale := rstd / float32(dim)
		term2 := sumDxHat
		term3 := sumDxHatXHat
		
		for i := 0; i < dim; i++ {
			dx := scale * (float32(dim)*dxHat[i] - term2 - xHat[i]*term3)
			dInput.Data[offset+i] = dx
		}
	}
	return dInput
}

func (ln *LayerNorm) Parameters() []*Parameter {
	return []*Parameter{ln.Gamma, ln.Beta}
}

// Dropout Layer
type Dropout struct {
	P    float32
	mask *tensor.NDArray
}

func NewDropout(p float32) *Dropout {
	return &Dropout{P: p}
}

func (d *Dropout) Forward(x *tensor.NDArray) *tensor.NDArray {
	if d.P == 0 {
		return x
	}
	out, mask := tensor.Dropout(x, d.P)
	d.mask = mask
	return out
}

func (d *Dropout) Backward(gradOutput *tensor.NDArray) *tensor.NDArray {
	if d.P == 0 {
		return gradOutput
	}
	return tensor.Mul(gradOutput, d.mask)
}

func (d *Dropout) Parameters() []*Parameter {
	return nil
}
