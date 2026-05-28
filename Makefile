NVCC = /usr/local/cuda-12.6/bin/nvcc
ARCH = -arch=sm_89
NVCCFLAGS = $(ARCH) -O3 -std=c++17 -Iinclude
LDFLAGS = -L/usr/local/cuda-12.6/lib64 -lcudart -lm

test: test_correctness
	./test_correctness

test_correctness: tests/test_correctness.cu src/*.cu include/*.cuh
	$(NVCC) $(NVCCFLAGS) tests/test_correctness.cu src/*.cu -o $@ $(LDFLAGS)

clean:
	rm -f test_correctness

.PHONY: test clean
