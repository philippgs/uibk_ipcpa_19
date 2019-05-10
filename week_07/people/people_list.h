#pragma once

#include <stdio.h>
#include "people.h"

person_t* generate_list(int N, int rand_seed) {
	srand(rand_seed);
	person_t* people = malloc(sizeof(person_t) * N);

	for(int i = 0; i < N; ++i) {
		people[i].age = rand() % MAX_AGE;
		gen_name(people[i].name);
	}
	return people;
}

void print_list(const person_t* people, int N) {
	for(int i = 0; i < N; ++i) {
		printf("%3d | %s\n", people[i].age, people[i].name);
	}
}

void free_list(person_t* people) {
	free(people);
}
