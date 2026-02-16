package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/brucetruth/minigpt/llm/tensor"
)

func benchCmd(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	size := fs.Int("size", 256, "Matrix size N for NxN matmul")
	iter := fs.Int("iter", 10, "Iterations")
	fs.Parse(args)
	
	fmt.Printf("Benchmarking MatMul %dx%d for %d iters...\n", *size, *size, *iter)
	
	a := tensor.NewRandom(*size, *size)
	b := tensor.NewRandom(*size, *size)
	
	start := time.Now()
	for i := 0; i < *iter; i++ {
		_ = tensor.MatMul(a, b)
	}
	dur := time.Since(start)
	
	fmt.Printf("Total time: %v\n", dur)
	fmt.Printf("Avg time: %v\n", dur / time.Duration(*iter))
	
	ops := 2.0 * float64(*size) * float64(*size) * float64(*size) // 2*N^3
	gflops := (ops * float64(*iter)) / dur.Seconds() / 1e9
	fmt.Printf("GFLOPS: %.4f\n", gflops)
}

func tokenizeCmd(args []string) {
	// TODO
}
