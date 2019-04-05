
typedef float value_t;

__kernel void stencil(
    __global const value_t* A, 
    __global value_t* B,
    int source_x,
    int source_y,
    int N,
    __local value_t* L		// local memory to speed up computation
) {
    // obtain position of this 'thread'
    size_t i = get_global_id(1);
    size_t j = get_global_id(0);

    size_t li = get_local_id(1);
    size_t lj = get_local_id(0);

    size_t mi = get_local_size(1);
    size_t mj = get_local_size(0);
    
    // the width of the local buffer
    const size_t LN = mj + 2;
    
    #define G(X,Y) A[   (X) * N +   (Y)  ]
    #define L(X,Y) L[((X)+1)*LN + ((Y)+1)]
    
    // load part of input buffer B into local memory
    if( i < N && j < N ) {
        // load central box
        L(li,lj) = G(i,j);

        // load boundaries
        if (li ==   0  && i !=  0)  L(li-1,lj) = G(i-1,j);
        if (li == mi-1 && i != N-1) L(li+1,lj) = G(i+1,j);
    
        if (lj ==   0  && j !=  0)  L(li,lj-1) = G(i,j-1);
        if (lj == mj-1 && j != N-1) L(li,lj+1) = G(i,j+1);
    }
    
    // finally: memory fence
    barrier(CLK_LOCAL_MEM_FENCE); // WARNING: same barrier must be reached by all work items
    
    // now we are allowed to kill the excessive work items
    if ( i >= N || j >=N ) return;
    
    // finally update elements using data from local memory

    // center stays constant (the heat is still on)
    if (i == source_x && j == source_y) {
        B[i*N+j] = L(li,lj);
        return;
    }

    // get current temperature at (i,j)
    value_t tc = L(li,lj);

    // get temperatures left/right and up/down
    value_t tl = ( j !=  0  ) ? L(li,lj-1) : tc;
    value_t tr = ( j != N-1 ) ? L(li,lj+1) : tc;
    value_t tu = ( i !=  0  ) ? L(li-1,lj) : tc;
    value_t td = ( i != N-1 ) ? L(li+1,lj) : tc;

    // update temperature at current point
    B[i*N+j] = tc + 0.2f * (tl + tr + tu + td + (-4.0f*tc));

}
