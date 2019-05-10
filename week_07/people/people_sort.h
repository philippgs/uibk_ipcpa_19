#pragma once

#include <stdio.h>
#include "people.h"

person_t* generate_sorted_list(const person_t* original_list, int N) {

	// Count sort algorithm:
	
	// step 1: create histogram
	int histogram[MAX_AGE] = { 0 };
	for(int i = 0; i < N; ++i) {
		histogram[original_list[i].age]++;
	}

	// step 2: compute prefix sum shifted by one
	int prefix_sum[MAX_AGE] = { histogram[0] };
	for(int i = 1; i < MAX_AGE; ++i) {
		prefix_sum[i] = prefix_sum[i-1] + histogram[i-1];
	}

	person_t* sorted_list = malloc(sizeof(person_t) * N);

	// step 3: copy to target
	for(int i = 0; i < N; ++i) {
		// copy person entry to respective index position and increment index afterwards
		memcpy(&sorted_list[prefix_sum[original_list[i].age]++], &original_list[i], sizeof(person_t));
	}

	return sorted_list;
}
