
typedef float value_t;

__kernel void stencil(
    __global const value_t* A, 
    __global value_t* B,
    int source_x,
    int source_y,
    int N
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    // center stays constant (the heat is still on)
    if (i == source_x && j == source_y) {
        B[i*N+j] = A[i*N+j];
        return;
    }

    // get current temperature at (i,j)
    value_t tc = A[i*N+j];

    // get temperatures left/right and up/down
    value_t tl = ( j !=  0  ) ? A[i*N+(j-1)] : tc;
    value_t tr = ( j != N-1 ) ? A[i*N+(j+1)] : tc;
    value_t tu = ( i !=  0  ) ? A[(i-1)*N+j] : tc;
    value_t td = ( i != N-1 ) ? A[(i+1)*N+j] : tc;

    // update temperature at current point
    B[i*N+j] = tc + 0.2f * (tl + tr + tu + td + (-4.0f*tc));

}
