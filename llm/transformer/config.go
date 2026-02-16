package transformer

type Config struct {
	VocabSize int
	BlockSize int
	NLayer    int
	NHead     int
	NEmb      int
	PDrop     float32
}

func DefaultConfig() Config {
	return Config{
		VocabSize: 50257,
		BlockSize: 1024,
		NLayer:    12,
		NHead:     12,
		NEmb:      768,
		PDrop:     0.1,
	}
}
