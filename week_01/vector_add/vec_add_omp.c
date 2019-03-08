#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

typedef float value_t;

int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    long long N = 100*1000*1000;
    if (argc > 1) {
        N = atoll(argv[1]);
    }
    printf("Computing vector-add with N=%lld\n", N);

    
    // ---------- setup ----------

    // create two input vectors (on heap!)
    value_t* a = malloc(sizeof(value_t)*N);
    value_t* b = malloc(sizeof(value_t)*N);
    
    // fill vectors
    for(long long i = 0; i<N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }
    
    // ---------- compute ----------
    
    value_t* c = malloc(sizeof(value_t)*N);

    timestamp begin = now();
    #pragma omp parallel for
    for(long long i = 0; i<N; i++) {
        c[i] = a[i] + b[i];
    }
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        if (c[i] == a[i] + b[i]) continue;
        success = false;
        break;
    }
    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    free(a);
    free(b);
    free(c);
    
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}
