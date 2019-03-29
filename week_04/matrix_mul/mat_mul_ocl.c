#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// -- profile utilities --

unsigned long long getElapsed(cl_event event);

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
    
    cl_event event_run_kernel;
    cl_event event_write_a;
    cl_event event_write_b;
    cl_event event_read_res;
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
        err = clEnqueueWriteBuffer(command_queue, devMatA, CL_FALSE, 0, N * N * sizeof(value_t), A, 0, NULL, &event_write_a);
        CLU_ERRCHECK(err, "Failed to write matrix A to device");
        err = clEnqueueWriteBuffer(command_queue, devMatB, CL_TRUE, 0,  N * N * sizeof(value_t), B, 0, NULL, &event_write_b);
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
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, size, NULL, 0, NULL, &event_run_kernel), "Failed to enqueue 2D kernel");

        // Part 6: copy results back to host
        err = clEnqueueReadBuffer(command_queue, devMatC, CL_TRUE, 0, N * N * sizeof(value_t), C, 0, NULL, &event_read_res);
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

    // compute performance of individual steps
    printf("Individual times: write a: %f ms, write b: %f ms, run kernel: %f ms, read c: %f ms\n", getElapsed(event_write_a)/1e6, getElapsed(event_write_b)/1e6, getElapsed(event_run_kernel)/1e6, getElapsed(event_read_res)/1e6);
    double num_mflop = (((double)2*N-1)*N*N)/1e6;
    double input_data_mbytes = ((double)sizeof(value_t)*N*N)/1024/1024;
    double output_data_mbytes = ((double)sizeof(value_t)*N*N)/1024/1024;
    printf("Throughput write a: %f MB/s\n", input_data_mbytes/(getElapsed(event_write_a)/1e9));
    printf("Throughput write b: %f MB/s\n", input_data_mbytes/(getElapsed(event_write_b)/1e9));
    printf("Performance kernel: %f MFLOP/s\n", num_mflop/(getElapsed(event_run_kernel)/1e9));
    printf("Throughput read res: %f MB/s\n", output_data_mbytes/(getElapsed(event_read_res)/1e9));

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

unsigned long long getElapsed(cl_event event) {
    cl_ulong starttime = 0, endtime = 0;
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL), "Failed to get profiling information");
    CLU_ERRCHECK(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL), "Failed to get profiling information");
	return (endtime-(unsigned long long)starttime);
}

