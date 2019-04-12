
__kernel void sum(
    __global int* data,      // the vector to be reduced
    __global int* result,    // the result vector
    __local  int* scratch,   // a local scratch memory buffer
    long N                   // the length of the vector
) {

    // get Ids
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);

    // load data into local memory
    if (global_index < N) {
        scratch[local_index] = data[global_index];
    } else {
        scratch[local_index] = 0;       // neutral element
    }
    
    // wait for all in group to flush results to local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // perform reduction
    for(int offset = get_local_size(0)/2; offset > 0; offset >>=1 ) {

        // perform reduction with remaining elements        
        if (local_index < offset) {
            scratch[local_index] += scratch[local_index + offset];
        }
    
        // sync on local memory state
        barrier(CLK_LOCAL_MEM_FENCE);
    }    

    
    // write result to global result buffer
    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}

