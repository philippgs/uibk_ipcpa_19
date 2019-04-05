
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: heat_stencil_seq heat_stencil_omp heat_stencil_ocl

heat_stencil_seq: $(COMMON_DEPENDENCIES) heat_stencil_seq.c
	@$(CC) $(CC_FLAGS) heat_stencil_seq.c -o heat_stencil_seq

heat_stencil_omp: $(COMMON_DEPENDENCIES) heat_stencil_omp.c
	@$(CC) $(CC_FLAGS) heat_stencil_omp.c -o heat_stencil_omp -fopenmp

heat_stencil_ocl: $(COMMON_DEPENDENCIES) heat_stencil_ocl.c
	@$(CC) $(CC_FLAGS) heat_stencil_ocl.c -o heat_stencil_ocl -lOpenCL

.PHONEY: clean
clean:
	@rm heat_stencil_seq heat_stencil_omp heat_stencil_ocl
	
run: all
	@echo "Sequential:"
	@./heat_stencil_seq
	@echo
	@echo "OpenMP:"
	@./heat_stencil_omp
	@echo
	@echo "OpenCL:"
	@./heat_stencil_ocl


