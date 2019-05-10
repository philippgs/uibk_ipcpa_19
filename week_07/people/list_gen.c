#include <stdio.h>
#include "people_list.h"

int main(int argc, char** argv) {

	if(argc != 3) {
		printf("Usage: %s <N> <seed>\n", argv[0]);
		return 1;
	}

	const int N = atoi(argv[1]);
	const int seed = atoi(argv[2]);

	person_t* people = generate_list(N, seed);

	printf("List of people:\n");
	print_list(people, N);

	free_list(people);

	return 0;
}
