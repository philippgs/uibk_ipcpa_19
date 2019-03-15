#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"

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
    
    {
        // -- solution with CL utils --

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devMatA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix A");
        cl_mem devMatB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix B");
        cl_mem devMatC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * N * sizeof(value_t), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for matrix C");

        // Part 3: fill memory buffers
        err = clEnqueueWriteBuffer(command_queue, devMatA, CL_FALSE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write matrix A to device");
        err = clEnqueueWriteBuffer(command_queue, devMatB, CL_TRUE, 0,  N * N * sizeof(value_t), B, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write matrix B to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "mat_mul.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "mat_mul", &err);
        CLU_ERRCHECK(err, "Failed to create mat_mul kernel from program");

        // Part 5: set arguments and execute kernel
        size_t size[2] = {N, N}; // two dimensional range
        cluSetKernelArguments(kernel, 4,
            sizeof(cl_mem), (void *)&devMatC,
            sizeof(cl_mem), (void *)&devMatA,
            sizeof(cl_mem), (void *)&devMatB,
            sizeof(int), &N
        );
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, size, NULL, 0, NULL, NULL), "Failed to enqueue 2D kernel");

        // Part 6: copy results back to host
        err = clEnqueueReadBuffer(command_queue, devMatC, CL_TRUE, 0, N * N * sizeof(value_t), C, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed reading back result");

        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

        // free device memory
        CLU_ERRCHECK(clReleaseMemObject(devMatA), "Failed to release Matrix A");
        CLU_ERRCHECK(clReleaseMemObject(devMatB), "Failed to release Matrix B");
        CLU_ERRCHECK(clReleaseMemObject(devMatC), "Failed to release Matrix C");

        // free management resources
        CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
        CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
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

