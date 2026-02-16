package optim

import (
	"math"
)

// LRScheduler interface for learning rate schedulers
type LRScheduler interface {
	GetLR(step int) float32
}

// CosineScheduleWithWarmup implements cosine annealing with linear warmup
type CosineScheduleWithWarmup struct {
	WarmupSteps int
	MaxSteps    int
	LRMax       float32
	LRMin       float32
}

func NewCosineScheduleWithWarmup(warmupSteps, maxSteps int, lrMax, lrMin float32) *CosineScheduleWithWarmup {
	return &CosineScheduleWithWarmup{
		WarmupSteps: warmupSteps,
		MaxSteps:    maxSteps,
		LRMax:       lrMax,
		LRMin:       lrMin,
	}
}

func (s *CosineScheduleWithWarmup) GetLR(step int) float32 {
	// Linear warmup
	if step < s.WarmupSteps {
		return s.LRMax * float32(step) / float32(s.WarmupSteps)
	}

	// Cosine annealing
	if step > s.MaxSteps {
		return s.LRMin
	}

	progress := float32(step-s.WarmupSteps) / float32(s.MaxSteps-s.WarmupSteps)
	cosineDecay := 0.5 * (1.0 + float32(math.Cos(math.Pi*float64(progress))))
	return s.LRMin + (s.LRMax-s.LRMin)*cosineDecay
}
