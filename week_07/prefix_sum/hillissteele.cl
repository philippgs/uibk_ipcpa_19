
__kernel void sum_scan(
    __global int* data,      // the vector to be 'prefix-sumed'
    __global int* result,    // the result vector
    __local  int* scratch,   // a local scratch memory buffer (twice the size of the work group)
    long N                   // the length of the input vector
) {

    // get Ids
    int i = get_local_id(0);
    int s = get_local_size(0);
    
    int pout = 0, pin = 1;
    
    // -- load data --
    
    // copy input data into scratch (with upper limit + shift right by one)
    scratch[i] = (i > 0 && i < N) ? data[i-1] : 0;
    
    // -- compute scan --
    
    // sync on local memory state
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform step-wise prefix-sum computation
    for(int offset = 1; offset < s; offset <<=1) {
        // spaw buffers
        pin = pout;
        pout = pin - 1;
        
        // second half of threads performs aggregation, first just a copy
        if (i >= offset) {
            scratch[pout * s + i] = scratch[pin * s + i] + scratch[pin * s + i - offset];
        } else {
            scratch[pout * s + i] = scratch[pin * s + i];
        }
        
        // and sync threads and memory state
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // -- save result --
    
    // copy all output data at once
    result[i] = scratch[pout * s + i];
}

