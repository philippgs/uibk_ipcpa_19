
__kernel void vec_add(
    __global float* c, 
    __global const float* a, 
    __global const float* b,
    int N
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(0);

    // if beyond boundaries => skip this one
    if (i >= N) return;
    
    // compute C := A + B
    c[i] = a[i] + b[i];
}
