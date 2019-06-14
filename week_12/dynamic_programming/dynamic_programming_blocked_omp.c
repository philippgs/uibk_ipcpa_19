
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"

int main(int argc, char** argv) {

  int minSize = 10;
  int maxSize = 20;

  // read problem size
  int N = 2000;
  if (argc > 1) {
    N = atoi(argv[1]);
  }

  int S = N+1;
  printf("Computing minimum cost for multiplying %d matrices ...\n",N);

  // generate random matrix sizes
  srand(0);
  int* l = (int*)malloc(sizeof(int)*S);
  for(int i=0; i<S; i++) {
    l[i] = ((rand() /  (float)RAND_MAX) * (maxSize - minSize)) + minSize;
  }

  // compute minimum costs
  int* C = (int*)malloc(sizeof(int)*N*N);

  double start = now();

  int B = 22;           // < the block size (obtained through linear search)
  int NB = N/B;         // < the number of blocks in each dimension
  if (N%B != 0) NB++;   // < increase by 1 if there is an extra block

  // iterate through blocks in wave-front order
  for(int bd = 0; bd<NB; bd++) {
    #pragma omp parallel for          // < this loop can be parallelized
    for(int bi=0; bi<NB-bd; bi++) {
      int bj = bi + bd;

      // get lower-left corner of current blocks
      int ci = (bi+1)*B-1;
      int cj = bj*B;

      // process current block in wave-front order
      int count = 0;
      for(int d=0; d<2*B-1;d++) {
        int li=(d >= B ? B-1 : d);
        int lj=(d < B ? 0 : d-B+1);
        for(;li>=0 && lj<B; lj++,li--) {

          // get coordinated in C
          int i = ci - li;
          int j = cj + lj;

          // check whether the current cell still of interest
          if (i > j || i >=N || j >=N ) continue;

          // for main diagonal
          if (i == j) {
            C[i*N+j] = 0;
            continue;
          }

          // find cheapest cut between i and j
          int min = INT_MAX;
          for(int k=i; k<j; k++) {
            int costs = C[i*N+k] + C[(k+1)*N+j] + l[i] * l[k+1] * l[j+1];
            min = (costs < min) ? costs : min;
          }
          C[i*N+j] = min;

        }
      }
    }
  }

  double end = now();

  printf("Minimal costs: %d FLOPS\n", C[0*N+N-1]);
  printf("Total time: %.3fs\n", (end-start));

  // clean
  free(C);
  free(l);

  // done
  return EXIT_SUCCESS;
}
