package io

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"

	"github.com/brucetruth/minigpt/llm/nn"
	"github.com/brucetruth/minigpt/llm/transformer"
)

type CheckpointMetadata struct {
	Step   int
	Loss   float32
	Config transformer.Config
}

func SaveCheckpoint(path string, model *transformer.GPT, meta CheckpointMetadata) error {
	os.MkdirAll(path, 0755)

	// 1. Save Metadata
	metaData, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	if err := os.WriteFile(path+"/meta.json", metaData, 0644); err != nil {
		return err
	}

	// 2. Save Weights
	// Format: simple concatenation of all float32 data
	// Or name -> data mapping?
	// Prompt: "checkpoint: metadata JSON + weights .bin float32 LE"
	// We'll just dump all parameters in a defined order or map.
	// To be safe, let's use a simple binary format:
	// [NameLen(4)][Name(N)][ShapeLen(4)][Shape(N*4)][DataLen(4)][Data(N*4)]...

	f, err := os.Create(path + "/weights.bin")
	if err != nil {
		return err
	}
	defer f.Close()

	params := model.Parameters()
	for _, p := range params {
		if err := writeParam(f, p); err != nil {
			return err
		}
	}
	return nil
}

func writeParam(f *os.File, p *nn.Parameter) error {
	// Name
	nameBytes := []byte(p.Name)
	if err := binary.Write(f, binary.LittleEndian, int32(len(nameBytes))); err != nil {
		return err
	}
	if _, err := f.Write(nameBytes); err != nil {
		return err
	}

	// Shape
	shape := p.Data.Shape
	if err := binary.Write(f, binary.LittleEndian, int32(len(shape))); err != nil {
		return err
	}
	for _, s := range shape {
		if err := binary.Write(f, binary.LittleEndian, int32(s)); err != nil {
			return err
		}
	}

	// Data
	data := p.Data.Data
	if err := binary.Write(f, binary.LittleEndian, int32(len(data))); err != nil {
		return err
	}
	if err := binary.Write(f, binary.LittleEndian, data); err != nil {
		return err
	}
	return nil
}

func LoadCheckpoint(path string, model *transformer.GPT) (*CheckpointMetadata, error) {
	// Load Metadata
	metaData, err := os.ReadFile(path + "/meta.json")
	if err != nil {
		return nil, err
	}
	var meta CheckpointMetadata
	if err := json.Unmarshal(metaData, &meta); err != nil {
		return nil, err
	}

	// Load Weights
	f, err := os.Open(path + "/weights.bin")
	if err != nil {
		return &meta, err
	}
	defer f.Close()

	// Map existing params by name for easy loading
	paramMap := make(map[string]*nn.Parameter)
	for _, p := range model.Parameters() {
		paramMap[p.Name] = p
	}

	for {
		name, _, data, err := readParam(f)
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, err
		}

		if p, ok := paramMap[name]; ok {
			// Verify shape matches?
			// Copy data
			if len(p.Data.Data) != len(data) {
				// Warn or error
				fmt.Printf("Warning: size mismatch for %s\n", name)
				continue
			}
			copy(p.Data.Data, data)
		}
	}

	return &meta, nil
}

func readParam(f *os.File) (string, []int, []float32, error) {
	var nameLen int32
	if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
		return "", nil, nil, err
	}
	nameBytes := make([]byte, nameLen)
	if _, err := f.Read(nameBytes); err != nil {
		return "", nil, nil, err
	}
	name := string(nameBytes)

	var shapeLen int32
	if err := binary.Read(f, binary.LittleEndian, &shapeLen); err != nil {
		return "", nil, nil, err
	}
	shape := make([]int, shapeLen)
	for i := 0; i < int(shapeLen); i++ {
		var s int32
		if err := binary.Read(f, binary.LittleEndian, &s); err != nil {
			return "", nil, nil, err
		}
		shape[i] = int(s)
	}

	var dataLen int32
	if err := binary.Read(f, binary.LittleEndian, &dataLen); err != nil {
		return "", nil, nil, err
	}
	data := make([]float32, dataLen)
	if err := binary.Read(f, binary.LittleEndian, data); err != nil {
		return "", nil, nil, err
	}

	return name, shape, data, nil
}
