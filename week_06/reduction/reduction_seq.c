#include <stdio.h>
#include <stdlib.h>

#include "utils.h"


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    size_t N = (1<<30) / sizeof(int); // = 1GiB by default
    if (argc > 1) {
        N = atol(argv[1]);
    }
    printf("Computing reduction of N=%lu random values\n", N);

    
    // ---------- setup ----------

    // create a buffer for storing random values
    printf("Generating random data (%.1fGiB) ...\n", N*sizeof(int)/(1024.0f*1024*1024));
    int* data = (int*)malloc(N*sizeof(int));
    if (!data) {
        printf("Unable to allocate enough memory\n");
        return EXIT_FAILURE;
    }
    
    // initializing random value buffer
    for(int i=0; i<N; i++) {
        data[i] = rand() % 2;
    }

    // ---------- compute ----------


    printf("Counting ...\n");
    
    timestamp begin = now();
    int count = 0;
    for(int i=0; i<N; i++) {
        if (data[i] == 1) count++;
    }
    timestamp end = now();
    printf("\ttook: %.3f ms\n", (end-begin)*1000);


    // -------- print result -------

    printf("Number of 1s: %d - %.2f%%\n", count, (count/(float)N)*100);

    // ---------- cleanup ----------
    
    free(data);
    
    // done
    return EXIT_SUCCESS;
}

