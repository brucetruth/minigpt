package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type Tokenizer struct {
	Vocab    map[int]string // id -> token (bytes)
	Merges   map[string]int // pair "u,v" -> rank
	Encoder  map[string]int // token (bytes) -> id
	Decoder  map[int]string // id -> string
	VocabSize int
}

func New() *Tokenizer {
	return &Tokenizer{
		Vocab:    make(map[int]string),
		Merges:   make(map[string]int),
		Encoder:  make(map[string]int),
		Decoder:  make(map[int]string),
		VocabSize: 256, // Start with bytes
	}
}

// Train logic: simplified
func (t *Tokenizer) Train(text string, vocabSize int) {
	// 1. Initialize with all bytes
	for i := 0; i < 256; i++ {
		b := string([]byte{byte(i)})
		t.Encoder[b] = i
		t.Decoder[i] = b
	}
	
	// Convert text to list of integers (bytes)
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = int(text[i])
	}
	
	numMerges := vocabSize - 256
	for i := 0; i < numMerges; i++ {
		stats := getStats(ids)
		if len(stats) == 0 {
			break
		}
		
		pair := getMax(stats)
		idx := 256 + i
		
		// Record merge
		key := fmt.Sprintf("%d,%d", pair.a, pair.b)
		t.Merges[key] = idx
		
		// New token
		tokenBytes := t.Decoder[pair.a] + t.Decoder[pair.b]
		t.Decoder[idx] = tokenBytes
		t.Encoder[tokenBytes] = idx
		
		// Apply merge
		ids = merge(ids, pair, idx)
	}
	t.VocabSize = 256 + len(t.Merges)
}

func (t *Tokenizer) Encode(text string) []int {
	if len(t.Encoder) == 0 {
		return nil
	}
	
	ids := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		ids[i] = int(text[i])
	}
	
	for {
		stats := getStats(ids)
		if len(stats) == 0 {
			break
		}
		
		// Find lowest rank pair
		minRank := 10000000
		var bestPair pair
		found := false
		
		for p := range stats {
			key := fmt.Sprintf("%d,%d", p.a, p.b)
			rank, ok := t.Merges[key]
			if ok {
				if rank < minRank {
					minRank = rank
					bestPair = p
					found = true
				}
			}
		}
		
		if !found {
			break
		}
		
		ids = merge(ids, bestPair, minRank)
	}
	
	return ids
}

func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(t.Decoder[id])
	}
	return sb.String()
}

func (t *Tokenizer) Save(path string) error {
	data, err := json.Marshal(t)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t := New()
	if err := json.Unmarshal(data, t); err != nil {
		return nil, err
	}
	return t, nil
}

// Helpers
type pair struct {
	a, b int
}

func getStats(ids []int) map[pair]int {
	counts := make(map[pair]int)
	for i := 0; i < len(ids)-1; i++ {
		p := pair{ids[i], ids[i+1]}
		counts[p]++
	}
	return counts
}

func getMax(stats map[pair]int) pair {
	maxVal := -1
	var best pair
	for p, v := range stats {
		if v > maxVal {
			maxVal = v
			best = p
		}
	}
	return best
}

func merge(ids []int, p pair, idx int) []int {
	newIds := make([]int, 0, len(ids))
	i := 0
	for i < len(ids) {
		if i < len(ids)-1 && ids[i] == p.a && ids[i+1] == p.b {
			newIds = append(newIds, idx)
			i += 2
		} else {
			newIds = append(newIds, ids[i])
			i++
		}
	}
	return newIds
}
