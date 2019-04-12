
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: reduction_seq reduction_omp reduction_ocl

reduction_seq: $(COMMON_DEPENDENCIES) reduction_seq.c
	@$(CC) $(CC_FLAGS) reduction_seq.c -o reduction_seq

reduction_omp: $(COMMON_DEPENDENCIES) reduction_omp.c
	@$(CC) $(CC_FLAGS) reduction_omp.c -o reduction_omp -fopenmp

reduction_ocl: $(COMMON_DEPENDENCIES) reduction_ocl.c
	@$(CC) $(CC_FLAGS) reduction_ocl.c -o reduction_ocl -lOpenCL

.PHONEY: clean
clean:
	@rm reduction_seq reduction_omp reduction_ocl
	
run: all
	@echo "Sequential:"
	@./reduction_seq
	@echo
	@echo "OpenMP:"
	@./reduction_omp
	@echo
	@echo "OpenCL:"
	@./reduction_ocl


