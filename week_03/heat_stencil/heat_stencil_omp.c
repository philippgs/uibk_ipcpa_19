#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

void printTemperature(Matrix m, int N, int M);

// ----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    int N = 500;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    int T = N*100;
    printf("Computing heat-distribution for room size N=%d for T=%d timesteps\n", N, T);

    
    // ---------- setup ----------

    // create a buffer for storing temperature fields
    Matrix A = createMatrix(N,N);
    
    // set up initial conditions in A
    for(int i = 0; i<N; i++) {
        for(int j = 0; j<N; j++) {
            A[i*N+j] = 273;             // temperature is 0° C everywhere (273 K)
        }
    }

    // and there is a heat source in one corner
    int source_x = N/4;
    int source_y = N/4;
    A[source_x*N+source_y] = 273 + 60;

    printf("Initial:\n");
    printTemperature(A,N,N);
    
    // ---------- compute ----------

    // create a second buffer for the computation    
    Matrix B = createMatrix(N,N);

    timestamp begin = now();

    // -- BEGIN ASSIGNMENT --
    
    // TODO: parallelize the following computation using OpenMP
    

    // for each time step ..
    for(int t=0; t<T; t++) {

        // .. we propagate the temperature 
        for(long long i = 0; i<N; i++) {
            for(long long j = 0; j<N; j++) {

                // center stays constant (the heat is still on)
                if (i == source_x && j == source_y) {
                    B[i*N+j] = A[i*N+j];
                    continue;
                }

                // get current temperature at (i,j)
                value_t tc = A[i*N+j];

                // get temperatures left/right and up/down
                value_t tl = ( j !=  0  ) ? A[i*N+(j-1)] : tc;
                value_t tr = ( j != N-1 ) ? A[i*N+(j+1)] : tc;
                value_t tu = ( i !=  0  ) ? A[(i-1)*N+j] : tc;
                value_t td = ( i != N-1 ) ? A[(i+1)*N+j] : tc;

                // update temperature at current point
                B[i*N+j] = tc + 0.2 * (tl + tr + tu + td + (-4*tc));
            }
        }

        // swap matrices (just pointers, not content)
        Matrix H = A;
        A = B;
        B = H;

        // show intermediate step
        if (!(t%1000)) {
            printf("Step t=%d:\n", t);
            printTemperature(A,N,N);
        }
    }


    // -- END ASSIGNMENT --
    

    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    releaseMatrix(B);


    // ---------- check ----------    

    printf("Final:\n");
    printTemperature(A,N,N);
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        for(long long j = 0; j<N; j++) {
            value_t temp = A[i*N+j];
            if (273 <= temp && temp <= 273+60) continue;
            success = false;
            break;
        }
    }
    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    releaseMatrix(A);
    
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

void printTemperature(Matrix m, int N, int M) {
    const char* colors = " .-:=+*#%@";
    const int numColors = 10;

    // boundaries for temperature (for simplicity hard-coded)
    const value_t max = 273 + 30;
    const value_t min = 273 + 0;

    // set the 'render' resolution
    int H = 30;
    int W = 50;

    // step size in each dimension
    int sH = N/H;
    int sW = M/W;


    // upper wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

    // room
    for(int i=0; i<H; i++) {
        // left wall
        printf("X");
        // actual room
        for(int j=0; j<W; j++) {

            // get max temperature in this tile
            value_t max_t = 0;
            for(int x=sH*i; x<sH*i+sH; x++) {
                for(int y=sW*j; y<sW*j+sW; y++) {
                    max_t = (max_t < m[x*N+y]) ? m[x*N+y] : max_t;
                }
            }
            value_t temp = max_t;

            // pick the 'color'
            int c = ((temp - min) / (max - min)) * numColors;
            c = (c >= numColors) ? numColors-1 : ((c < 0) ? 0 : c);

            // print the average temperature
            printf("%c",colors[c]);
        }
        // right wall
        printf("X\n");
    }

    // lower wall
    for(int i=0; i<W+2; i++) {
        printf("X");
    }
    printf("\n");

}

