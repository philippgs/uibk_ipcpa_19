
OCL_HOME=/scratch/c703/c703429/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: vec_add_seq vec_add_omp vec_add_ocl

vec_add_seq: $(COMMON_DEPENDENCIES) vec_add_seq.c
	@$(CC) $(CC_FLAGS) vec_add_seq.c -o vec_add_seq

vec_add_omp: $(COMMON_DEPENDENCIES) vec_add_omp.c
	@$(CC) $(CC_FLAGS) vec_add_omp.c -o vec_add_omp -fopenmp

vec_add_ocl: $(COMMON_DEPENDENCIES) vec_add_ocl.c
	@$(CC) $(CC_FLAGS) vec_add_ocl.c -o vec_add_ocl -lOpenCL

.PHONEY: clean
clean:
	@rm vec_add_seq vec_add_omp vec_add_ocl
	
run: all
	@echo "Sequential:"
	@./vec_add_seq
	@echo
	@echo "OpenMP:"
	@./vec_add_omp
	@echo
	@echo "OpenCL:"
	@./vec_add_ocl


