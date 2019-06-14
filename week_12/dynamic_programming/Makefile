
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -march=native -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: dynamic_programming_seq dynamic_programming_omp dynamic_programming_blocked_seq dynamic_programming_blocked_omp

dynamic_programming_seq: $(COMMON_DEPENDENCIES) dynamic_programming_seq.c
	@$(CC) $(CC_FLAGS) dynamic_programming_seq.c -o dynamic_programming_seq -lm

dynamic_programming_omp: $(COMMON_DEPENDENCIES) dynamic_programming_omp.c
	@$(CC) $(CC_FLAGS) dynamic_programming_omp.c -o dynamic_programming_omp -lm -fopenmp

dynamic_programming_blocked_seq: $(COMMON_DEPENDENCIES) dynamic_programming_blocked_omp.c
	@$(CC) $(CC_FLAGS) dynamic_programming_blocked_omp.c -o dynamic_programming_blocked_seq -lm

dynamic_programming_blocked_omp: $(COMMON_DEPENDENCIES) dynamic_programming_blocked_omp.c
	@$(CC) $(CC_FLAGS) dynamic_programming_blocked_omp.c -o dynamic_programming_blocked_omp -lm -fopenmp


.PHONEY: clean
clean:
	@rm -f dynamic_programming_seq dynamic_programming_omp dynamic_programming_blocked_seq dynamic_programming_blocked_omp
