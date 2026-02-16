all: build

build:
	go build -o minigpt ./cmd/minigpt

build-blas:
	go build -tags openblas -o minigpt-blas ./cmd/minigpt

test:
	go test -v ./llm/...

clean:
	rm -f minigpt minigpt-blas
	rm -rf checkpoints/
