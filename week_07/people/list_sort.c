#include <stdio.h>
#include "people_list.h"
#include "people_sort.h"

int main(int argc, char** argv) {

	if(argc != 3) {
		printf("Usage: %s <N> <seed>\n", argv[0]);
		return 1;
	}

	const int N = atoi(argv[1]);
	const int seed = atoi(argv[2]);

	person_t* people = generate_list(N, seed);

	printf("List of unsorted people:\n");
	print_list(people, N);

//	person_t* sorted_people = malloc(sizeof(person_t) * N);

	person_t* sorted_people = generate_sorted_list(people, N);

	printf("\nList of sorted people:\n");
	print_list(sorted_people, N);

	free_list(people);
	free_list(sorted_people);

	return 0;
}
