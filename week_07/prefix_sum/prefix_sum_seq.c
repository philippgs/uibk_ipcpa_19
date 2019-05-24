
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

int main(int argc, char** argv) {

    int N = 20;
    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    
    printf("Running sequential prefix sum on %d elements.\n", N);

    // generate a list of values to be 'prefix-summed'
    int* A = malloc(N*sizeof(int));
    for(int i=0; i<N; i++) {
        A[i] = i+1;
    }
    
    // compute prefix sums (off-by-one, out-of-place)
    int* S = malloc(N*sizeof(int));
    double start = now();
    {
        S[0] = 0;
        for(int i=1; i<N; i++) {
            S[i] = S[i-1] + A[i-1];
        }
    }
    double end = now();
    printf("Computation took %.1lfms\n", (end-start)*1000);
    
    // check result
    bool fail = false;
    for(int i=0; i<N; i++) {
        int should = (i * (i+1))/2;
        if (S[i] != should) {
            printf("\tERROR at index %i, should %d, is %d\n", i, should, S[i]);
            fail = true;
        }
    }

    if (!fail) {
        printf("Result OK!\n");
    }
    
    // cleanup
    free(A);
    free(S);

    return (fail) ? EXIT_FAILURE : EXIT_SUCCESS;
}
