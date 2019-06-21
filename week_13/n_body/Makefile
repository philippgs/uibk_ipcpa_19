
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -march=native -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -lm -Werror -pedantic

COMMON_DEPENDENCIES=Makefile

all: n_body_seq 

n_body_seq: $(COMMON_DEPENDENCIES) n_body_seq.c
	@$(CC) $(CC_FLAGS) n_body_seq.c -o n_body_seq

.PHONEY: clean
clean:
	@rm -f n_body_seq
