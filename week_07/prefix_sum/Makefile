
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h

all: prefix_sum_seq hillissteele_ocl downsweep_ocl prefixglobal_ocl prefixglobal_in_place_ocl

prefix_sum_seq: $(COMMON_DEPENDENCIES) prefix_sum_seq.c
	@$(CC) $(CC_FLAGS) prefix_sum_seq.c -o prefix_sum_seq

hillissteele_ocl: $(COMMON_DEPENDENCIES) hillissteele.c
	@$(CC) $(CC_FLAGS) hillissteele.c -o hillissteele_ocl -lOpenCL

downsweep_ocl: $(COMMON_DEPENDENCIES) downsweep.c
	@$(CC) $(CC_FLAGS) downsweep.c -o downsweep_ocl -lOpenCL

prefixglobal_ocl: $(COMMON_DEPENDENCIES) prefixglobal.c
	@$(CC) $(CC_FLAGS) prefixglobal.c -o prefixglobal_ocl -lOpenCL

prefixglobal_in_place_ocl: $(COMMON_DEPENDENCIES) prefixglobal_in_place.c
	@$(CC) $(CC_FLAGS) prefixglobal_in_place.c -o prefixglobal_in_place_ocl -lOpenCL

.PHONEY: clean
clean:
	@rm -f prefix_sum_seq hillissteele_ocl downsweep_ocl prefixglobal_ocl prefixglobal_in_place_ocl


