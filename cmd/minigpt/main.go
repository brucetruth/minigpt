package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		help()
		return
	}
	
	switch os.Args[1] {
	case "train":
		trainCmd(os.Args[2:])
	case "generate":
		generateCmd(os.Args[2:])
	case "bench":
		benchCmd(os.Args[2:])
	case "tokenize":
		tokenizeCmd(os.Args[2:])
	default:
		help()
	}
}

func help() {
	fmt.Println("Usage: minigpt [train|generate|bench|tokenize] [args]")
}
