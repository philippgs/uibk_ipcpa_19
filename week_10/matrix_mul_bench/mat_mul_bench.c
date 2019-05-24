#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "cl_utils.h"

typedef float value_t;


// -- matrix utilities --

typedef value_t* Matrix;

Matrix createMatrix(int N, int M);

void releaseMatrix(Matrix m);

// ----------------------

typedef struct _cl_mm_environment {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;    
} cl_mm_environment;

cl_mm_environment createMMEnvironment();

void destroyMMEnvironment(cl_mm_environment);

int roundUpToMultiple(int N, int B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}

// ----------------------

int SIZES[] = { 500, 734, 1024, 1493, 2345, 4001 };
int NUM_SIZES = 6;
int NUM_REPETITION = 3;

// ----------------------


int main(int argc, char** argv) {


    // ---------- setup ----------

    cl_mm_environment env = createMMEnvironment();

    
    // ------ benchmarking -------

    srand(0);
    printf("Start benchmarking ...\n");

    // the best performance
    double mflops[NUM_SIZES];
    bool allValid = true;

    // for each size ...
    for(int i=0; i<NUM_SIZES; i++) {

        // --- setup benchmark ---
        
        int N = SIZES[i];
        mflops[i] = 0;
        
        printf("\nSetting up N=%d ..\n", N);
        
        // create input
        restrict Matrix A = createMatrix(N,N);
        restrict Matrix B = createMatrix(N,N);
        restrict Matrix C = createMatrix(N,N);
        restrict Matrix R = createMatrix(N,N);

        // fill matrix
        for(int i = 0; i<N; i++) {
            for(int j = 0; j<N; j++) {
                A[i*N+j] = rand() / (float)RAND_MAX + 0.5;      // some matrix
                B[i*N+j] = rand() / (float)RAND_MAX + 0.5;      // some other matrix
            }
        }

        // compute reference results
        double cpu_start = now();
        #pragma omp parallel for
        for(int i = 0; i<N; i++) {
            // a slightly optimized CPU version of MM
            for(int j = 0; j<N; j++) {
                R[i*N+j] = 0;
            }
            for(int k=0; k<N; k++) {
                for(int j=0; j<N; j++) {
                    R[i*N+j] += A[i*N+k] * B[k*N+j];
                }
            }
        }
        double cpu_end = now();
        double cpu_duration = cpu_end - cpu_start;
        printf("\tCPU setup took %2.3fs / %5.3f GFLOPS\n", cpu_duration, (2.0*N*N*N) / cpu_duration / 1e9);

        // repeat X times ..
        for(int r=0; r<NUM_REPETITION; r++) {

            // clear result
            memset(C,0,sizeof(value_t) * N * N);
    
            // create buffer on device
            cl_int err;
            cl_mem devMatA = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
            CLU_ERRCHECK(err, "Failed to create buffer for matrix A");
            cl_mem devMatB = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, N * N * sizeof(value_t), NULL, &err);
            CLU_ERRCHECK(err, "Failed to create buffer for matrix B");
            cl_mem devMatC = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, N * N * sizeof(value_t), NULL, &err);

            // transfer data
            err = clEnqueueWriteBuffer(env.queue, devMatA, CL_TRUE, 0, N * N * sizeof(value_t), A, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed to write matrix A to device");
            err = clEnqueueWriteBuffer(env.queue, devMatB, CL_TRUE, 0,  N * N * sizeof(value_t), B, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed to write matrix B to device");


            // --- perform benchmark ---


            // -- run computation --

            // set arguments and execute kernel
            size_t S = roundUpToMultiple(N,32);
            size_t size[2] = {S, S};
            cluSetKernelArguments(env.kernel, 4,
                sizeof(cl_mem), (void *)&devMatC,
                sizeof(cl_mem), (void *)&devMatA,
                sizeof(cl_mem), (void *)&devMatB,
                sizeof(int), &N
            );

            // submit kernel
            cl_event event;
            CLU_ERRCHECK(clEnqueueNDRangeKernel(env.queue, env.kernel, 2, NULL, size, NULL, 0, NULL, &event), "Failed to enqueue 2D kernel");

            // wait for kernel
            clWaitForEvents(1,&event);
            
            // test whether kernel finished successfully
            cl_int status;
            clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
            if (status < 0) {
                CLU_ERRCHECK(-status, "Kernel failed to execute succesfully.");
                exit(1);
            }
            
            // get execution time
            cl_ulong start, end, duration;
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            duration = end - start;
   
            // release event
            CLU_ERRCHECK(clReleaseEvent(event), "Failed to release event");

            // copy results back to host
            err = clEnqueueReadBuffer(env.queue, devMatC, CL_TRUE, 0, N * N * sizeof(value_t), C, 0, NULL, NULL);
            CLU_ERRCHECK(err, "Failed reading back result");

            // check result
            bool success = true;
            for(int i = 0; i<N; i++) {
                for(int j = 0; j<N; j++) {
                    // if result is close enough, we are fine
                    if (fabsf(C[i*N+j]-R[i*N+j]) < 1e-10) continue;
                    //printf("Wrong result for (%d,%d): %f vs. %f\n", i,j,C[i*N+j],R[i*N+j]);
                    success = false;
                }
            }
            
            
            double seconds = duration / 1e9;
            double curMflops = (2.0*N*N*N) / seconds / 1e9;
            printf("\tDuration: %2.3fs, GFLOPS: %5.3f, Verification: %s\n", seconds, curMflops, (success)?"OK":"FAILED");
            
            // keep track of overall success
            if (!success) allValid = false;
            
            // record best performance
            if (mflops[i] < curMflops) mflops[i] = curMflops;

            // free device memory
            CLU_ERRCHECK(clReleaseMemObject(devMatA), "Failed to release Matrix A");
            CLU_ERRCHECK(clReleaseMemObject(devMatB), "Failed to release Matrix B");
            CLU_ERRCHECK(clReleaseMemObject(devMatC), "Failed to release Matrix C");

        }
        
        printf("\t\t\t\tPerformance result for N=%d: %5.3f\n", N, mflops[i]);

        // --- cleanup ---


        // free host memory
        releaseMatrix(A);
        releaseMatrix(B);
        releaseMatrix(C);
        releaseMatrix(R);

    }

    // cleanup
    
    destroyMMEnvironment(env);

    // finally: report overall result
    printf("\n");
    printf("-------------------------------------------------\n");
        
    if (!allValid) {
        
        printf("Invalid results encountered, failed!\n");
        
    } else {
     
        // overall score: geometric mean of individual best   
        double prod = 1;
        for(int i=0; i<NUM_SIZES; i++) {
            prod *= mflops[i];
        }
        double score = pow(prod,1.0/NUM_SIZES);
        printf("Overall result: %5.3f GFLOPS\n", score);
        
    }
    printf("-------------------------------------------------\n");
    
    // done
    return EXIT_SUCCESS;
}


Matrix createMatrix(int N, int M) {
    // create data and index vector
    return malloc(sizeof(value_t)*N*M);
}

void releaseMatrix(Matrix m) {
    free(m);
}

cl_mm_environment createMMEnvironment() {

    cl_mm_environment res;
    
    // ocl initialization
    cl_device_id device_id = cluInitDeviceWithProperties(0, &res.context, &res.queue, CL_QUEUE_PROFILING_ENABLE);

    // create kernel from source
    cl_int err;
    res.program = cluBuildProgramFromFile(res.context, device_id, "mat_mul.cl", NULL);
    res.kernel = clCreateKernel(res.program, "mat_mul", &err);
    CLU_ERRCHECK(err, "Failed to create mat_mul kernel from program");

    // done
    return res;
}

void destroyMMEnvironment(cl_mm_environment env) {

    // wait for completed operations (there should be none)
    CLU_ERRCHECK(clFlush(env.queue),            "Failed to flush command queue");
    CLU_ERRCHECK(clFinish(env.queue),           "Failed to wait for command queue completion");
    CLU_ERRCHECK(clReleaseKernel(env.kernel),   "Failed to release kernel");
    CLU_ERRCHECK(clReleaseProgram(env.program), "Failed to release program");

    // free management resources
    CLU_ERRCHECK(clReleaseCommandQueue(env.queue), "Failed to release command queue");
    CLU_ERRCHECK(clReleaseContext(env.context),    "Failed to release OpenCL context");
}


