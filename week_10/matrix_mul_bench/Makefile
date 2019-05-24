
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -march=native -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: mat_mul_bench

mat_mul_bench: $(COMMON_DEPENDENCIES) mat_mul_bench.c cl_utils.h
	@$(CC) $(CC_FLAGS) mat_mul_bench.c -o mat_mul_bench -lOpenCL -lm -fopenmp

.PHONEY: clean
clean:
	@rm mat_mul_bench
	
run: all
	@echo "Running benchmark .."
	@./mat_mul_bench


