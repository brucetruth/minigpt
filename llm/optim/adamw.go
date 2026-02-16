package optim

import (
	"math"

	"github.com/brucetruth/minigpt/llm/nn"
)

// AdamW Optimizer.
type AdamW struct {
	Params      []*nn.Parameter
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32

	step int
	m    []float32 // First moment
	v    []float32 // Second moment
}

func NewAdamW(params []*nn.Parameter, lr float32) *AdamW {
	totalSize := 0
	for _, p := range params {
		totalSize += p.Data.Size
	}

	return &AdamW{
		Params:      params,
		LR:          lr,
		Beta1:       0.9,
		Beta2:       0.999,
		Eps:         1e-8,
		WeightDecay: 0.01,
		step:        0,
		m:           make([]float32, totalSize), // Flattened view
		v:           make([]float32, totalSize),
	}
}

func (opt *AdamW) Step() {
	opt.step++
	// Bias correction
	biasCorrection1 := 1.0 - float32(math.Pow(float64(opt.Beta1), float64(opt.step)))
	biasCorrection2 := 1.0 - float32(math.Pow(float64(opt.Beta2), float64(opt.step)))

	offset := 0
	for _, p := range opt.Params {
		for i := 0; i < p.Data.Size; i++ {
			grad := p.Grad.Data[i]
			data := p.Data.Data[i]

			// Weight Decay
			data -= opt.LR * opt.WeightDecay * data

			// Adam Update
			m := opt.m[offset+i]
			v := opt.v[offset+i]

			m = opt.Beta1*m + (1-opt.Beta1)*grad
			v = opt.Beta2*v + (1-opt.Beta2)*grad*grad

			opt.m[offset+i] = m
			opt.v[offset+i] = v

			// Apply
			denom := (float32(math.Sqrt(float64(v))) / float32(math.Sqrt(float64(biasCorrection2)))) + opt.Eps
			stepSize := opt.LR / biasCorrection1

			p.Data.Data[i] = data - stepSize*(m/denom)
		}
		offset += p.Data.Size
	}
}

func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.Params {
		p.ZeroGrad()
	}
}

// ClipGradNorm clips gradient norms to a maximum value
func (opt *AdamW) ClipGradNorm(maxNorm float32) {
	if maxNorm <= 0 {
		return
	}

	// Calculate total norm
	totalNorm := float32(0.0)
	for _, p := range opt.Params {
		for i := 0; i < p.Data.Size; i++ {
			grad := p.Grad.Data[i]
			totalNorm += grad * grad
		}
	}
	totalNorm = float32(math.Sqrt(float64(totalNorm)))

	// Clip if needed
	if totalNorm > maxNorm {
		clipCoef := maxNorm / (totalNorm + 1e-6)
		for _, p := range opt.Params {
			for i := 0; i < p.Data.Size; i++ {
				p.Grad.Data[i] *= clipCoef
			}
		}
	}
}

// SetLR updates the learning rate
func (opt *AdamW) SetLR(lr float32) {
	opt.LR = lr
}
