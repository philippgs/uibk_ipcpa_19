
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -pedantic

COMMON_DEPENDENCIES=Makefile utils.h stb/image.h stb/image_write.h

all: auto_level_seq

auto_level_seq: $(COMMON_DEPENDENCIES) auto_level_seq.c
	@$(CC) $(CC_FLAGS) auto_level_seq.c  -lm -o auto_level_seq -fopenmp

.PHONEY: clean
clean:
	@rm -f auto_level_seq
