
OCL_HOME=/scratch/c703/c7031057/opencl

CC=gcc
CC_FLAGS=-O3 -std=c11 -I$(OCL_HOME)/include -L$(OCL_HOME)/lib -Werror -Wall -Wextra -pedantic

COMMON_DEPENDENCIES=Makefile people.h people_list.h

all: list_gen list_sort

list_gen: $(COMMON_DEPENDENCIES) list_gen.c
	@$(CC) $(CC_FLAGS) list_gen.c -o list_gen

list_sort: $(COMMON_DEPENDENCIES) list_sort.c people_sort.h
	@$(CC) $(CC_FLAGS) list_sort.c -o list_sort

.PHONEY: clean
clean:
	@rm -f list_gen list_sort


