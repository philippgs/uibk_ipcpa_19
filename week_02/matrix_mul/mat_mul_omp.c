#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 1000;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    printf("Computing matrix-matrix product with N=%d\n", N);

    
    // ---------- setup ----------

    // create two input matrices (on heap!)
    Matrix A = createMatrix(N,N);
    Matrix B = createMatrix(N,N);
    
    // fill matrices
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = i*j;             // some arbitrary matrix - note: flattend indexing!
            B[i*N+j] = (i==j) ? 1 : 0;  // identity
        }
    }
    
    // ---------- compute ----------
    
    Matrix C = createMatrix(N,N);

    timestamp begin = now();

    // The i and j loop do not carry any dependencies, the k loop does.
    // Thus, i and j can be parallelized.
    // For thread-level parallelism (OpenMP) outer-most parallelism is more
    // beneficial to avoid synchronization overhead.
    
    #pragma omp parallel for
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t sum = 0;
            for(long long k = 0; k<N; k++) {
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
    
    
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            if (C[i*N+j] == i*j) continue;
            success = false;
            break;
        }
    }
    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    releaseMatrix(A);
    releaseMatrix(B);
    releaseMatrix(C);
    
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}


Matrix createMatrix(int N, int M) {
    // create data and index vector
    return malloc(sizeof(value_t)*N*M);
}

void releaseMatrix(Matrix m) {
    free(m);
}

