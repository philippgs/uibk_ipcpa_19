
__kernel void sum_scan(
    __global int* data,      // the vector to be 'prefix-sumed'
    __global int* result,    // the result vector
    __local  int* scratch,   // a local scratch memory buffer (twice the size of the work group)
    long N                   // the length of the input vector
) {

    // get Ids
    int i = get_global_id(0);
    int s = get_global_size(0);
    int s2 = 2*s;
    
    // -- load data --
    
    // copy input data into scratch (each thread 2 elements, with upper limit, shift right by one)
    scratch[ 2*i ] = ((2*i-1) < N) ? data[ 2*i ] : 0;
    scratch[2*i+1] = ( (2*i)  < N) ? data[2*i+1] : 0;
    
    
    // -- compute scan --
    int offset = 1;
    
    // build sums in place up the tree
    for(int d = s2 >> 1; d > 0; d >>= 1) {
        // sync on local memory state
        barrier(CLK_LOCAL_MEM_FENCE);

        if( i < d ) {
            int ai = offset * (2*i+1) - 1;
            int bi = offset * (2*i+2) - 1;
            scratch[bi] += scratch[ai];
        }
        
        offset *= 2;
    }
    
    // clear the last element
    if (i == 0) {
        scratch[s2-1] = 0;
    }
    
    // traverse down the tree and build scan result
    for(int d = 1; d < s2 ; d *= 2) {
        offset >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if ( i < d ) {
            int ai = offset * (2*i+1) - 1;
            int bi = offset * (2*i+2) - 1;
            int t = scratch[ai];
            scratch[ai] = scratch[bi];
            scratch[bi] += t;
        }
    }
    
    
    // -- save result --

    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // write result back to device memory (two elements per thread)
    if ( 2*i  < N) result[ 2*i ] = scratch[ 2*i ];
    if (2*i+1 < N) result[2*i+1] = scratch[2*i+1];
}

