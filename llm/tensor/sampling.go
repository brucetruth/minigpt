package tensor

import (
	"math"
	"math/rand"
	"sort"
)

// ApplyTemperature scales logits by temperature
// Higher temperature = more random, lower = more deterministic
func ApplyTemperature(logits *NDArray, temperature float32) *NDArray {
	if temperature == 1.0 {
		return logits
	}

	out := New(logits.Shape...)
	for i, val := range logits.Data {
		out.Data[i] = val / temperature
	}
	return out
}

// SampleMultinomial samples an index from a probability distribution
func SampleMultinomial(probs *NDArray) int {
	r := rand.Float32()
	cumSum := float32(0.0)

	for i, p := range probs.Data {
		cumSum += p
		if r < cumSum {
			return i
		}
	}
	return len(probs.Data) - 1
}

// TopK filters logits to keep only top-k values
// Sets all other logits to -inf
func TopK(logits *NDArray, k int) *NDArray {
	if k <= 0 || k >= len(logits.Data) {
		return logits
	}

	// Create index-value pairs
	type pair struct {
		idx int
		val float32
	}

	pairs := make([]pair, len(logits.Data))
	for i, v := range logits.Data {
		pairs[i] = pair{idx: i, val: v}
	}

	// Sort by value descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].val > pairs[j].val
	})

	// Create output with top-k kept, rest set to -inf
	out := New(logits.Shape...)
	minVal := float32(math.Inf(-1))

	for i := range out.Data {
		out.Data[i] = minVal
	}

	for i := 0; i < k; i++ {
		out.Data[pairs[i].idx] = logits.Data[pairs[i].idx]
	}

	return out
}

// TopP (nucleus sampling) filters logits to keep smallest set with cumulative prob >= p
func TopP(logits *NDArray, p float32) *NDArray {
	if p >= 1.0 {
		return logits
	}

	// First apply softmax to get probabilities
	probs := Softmax(logits)

	// Create index-prob pairs
	type pair struct {
		idx  int
		prob float32
	}

	pairs := make([]pair, len(probs.Data))
	for i, prob := range probs.Data {
		pairs[i] = pair{idx: i, prob: prob}
	}

	// Sort by probability descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})

	// Find cutoff where cumulative prob >= p
	cumSum := float32(0.0)
	cutoff := len(pairs)
	for i, pair := range pairs {
		cumSum += pair.prob
		if cumSum >= p {
			cutoff = i + 1
			break
		}
	}

	// Create output with top-p kept, rest set to -inf
	out := New(logits.Shape...)
	minVal := float32(math.Inf(-1))

	for i := range out.Data {
		out.Data[i] = minVal
	}

	for i := 0; i < cutoff; i++ {
		out.Data[pairs[i].idx] = logits.Data[pairs[i].idx]
	}

	return out
}

// SampleFromLogits samples from logits with optional temperature, top-k, top-p
func SampleFromLogits(logits *NDArray, temperature float32, topK int, topP float32) int {
	// Apply temperature
	if temperature != 1.0 {
		logits = ApplyTemperature(logits, temperature)
	}

	// Apply top-k filtering
	if topK > 0 {
		logits = TopK(logits, topK)
	}

	// Apply top-p filtering
	if topP < 1.0 {
		logits = TopP(logits, topP)
	}

	// Convert to probabilities
	probs := Softmax(logits)

	// Sample
	return SampleMultinomial(probs)
}
