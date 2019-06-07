
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -march=native -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: dynamic_programming_seq

dynamic_programming_seq: $(COMMON_DEPENDENCIES) dynamic_programming_seq.c
	@$(CC) $(CC_FLAGS) dynamic_programming_seq.c -o dynamic_programming_seq -lm

.PHONEY: clean
clean:
	@rm -f dynamic_programming_seq
