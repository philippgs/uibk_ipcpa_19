
__kernel void sum_scan_reduce(
    __global int* data,      // the vector to be 'prefix-sumed' -- in and out buffer
    __global int* sum,       // the result vector for storing the total sum of a work group
    __local  int* scratch,   // a local scratch memory buffer (twice the size of the work group)
    long N                   // the length of the input vector
) {

    // get Ids
    int gi = get_global_id(0);
    int li = get_local_id(0);
    int gid = get_group_id(0);
    
    // get the local size - to know the target range of this operation    
    int s = get_local_size(0);
    int s2 = 2*s;
    
    // -- load data --
    
    // copy input data into scratch (each thread 2 elements, with upper limit, shift right by one)
    scratch[ 2*li ] = ((2*gi-1) < N) ? data[ 2*gi ] : 0;
    scratch[2*li+1] = ( (2*gi)  < N) ? data[2*gi+1] : 0;

    // -- compute scan --
    int offset = 1;
    
    // build sums in place up the tree
    for(int d = s2 >> 1; d > 0; d >>= 1) {
        // sync on local memory state
        barrier(CLK_LOCAL_MEM_FENCE);

        if( li < d ) {
            int ai = offset * (2*li+1) - 1;
            int bi = offset * (2*li+2) - 1;
            scratch[bi] += scratch[ai];
        }
        
        offset *= 2;
    }
    
    // clear the last element
    if (li == 0) {
        scratch[s2-1] = 0;
    }
    
    // traverse down the tree and build scan result
    for(int d = 1; d < s2 ; d *= 2) {
        offset >>= 1;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if ( li < d ) {
            int ai = offset * (2*li+1) - 1;
            int bi = offset * (2*li+2) - 1;
            int t = scratch[ai];
            scratch[ai] = scratch[bi];
            scratch[bi] += t;
        }
    }
    
    
    // -- save result --

    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // save total sum to sum vector
    if (li == (s-1)) {
        // the total is the last element + the last entry of the input
        sum[gid] = scratch[s2-1] + ((2*gi+1<N) ? data[2*gi+1] : 0);
    }
    
    // write result back to device memory (two elements per thread)
    if ( 2*gi  < N) data[ 2*gi ] = scratch[ 2*li ];
    if (2*gi+1 < N) data[2*gi+1] = scratch[2*li+1];
}

__kernel void sum_scan_expand(
    __global int* result,    // the result vector - storing partial sums per group
    __global int* sum,       // the result vector for storing the total sum of a work group
    long N                   // the length of the input vector
) {

    // get Ids
    int gi = get_global_id(0);
    int gid = get_group_id(0);

    // not the first ...
    if (gid == 0) return;
    
    // ... add sum of previous group to local result
    if ( 2*gi  < N) result[2*gi  ] += sum[gid];
    if (2*gi+1 < N) result[2*gi+1] += sum[gid];
}

