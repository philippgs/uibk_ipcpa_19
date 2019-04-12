#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"


long long roundUpToMultiple(long long N, long long B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    size_t N = (1<<30) / sizeof(int); // = 1GiB by default
    if (argc > 1) {
        N = atol(argv[1]);
    }
    printf("Computing reduction of N=%ld random values\n", N);

    
    // ---------- setup ----------

    // create a buffer for storing random values
    printf("Generating random data (%.1fGiB) ...\n", N*sizeof(int)/(1024.0f*1024*1024));
    int* data = (int*)malloc(N*sizeof(int));
    if (!data) {
        printf("Unable to allocate enough memory\n");
        return EXIT_FAILURE;
    }
    
    // initializing random value buffer
    for(int i=0; i<N; i++) {
        data[i] = rand() % 2;
    }

    // ---------- compute ----------


    printf("Counting ...\n");
        
    int count = 0;
    timestamp begin = now();
    {
        // - setup -
        
        size_t work_group_size = 32;

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create memory buffers
        cl_int err;
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, (roundUpToMultiple(N,work_group_size)/work_group_size) *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");


        // Part 3: fill memory buffers (transferring A is enough, B can be anything)
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(int), data, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");

        // Part 4: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "reduction.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "sum", &err);
        CLU_ERRCHECK(err, "Failed to create reduction kernel from program");

        // Part 5: perform multi-step reduction
        clFinish(command_queue);
        timestamp begin_reduce = now();
        size_t curLength = N;
        int numStages = 0;
        while(curLength > 1) {
        
            // perform one stage of the reduction
            size_t global_size = roundUpToMultiple(curLength,work_group_size);
            
            // for debugging:
            printf("CurLength: %lu, Global: %lu, WorkGroup: %lu\n", curLength, global_size, work_group_size);
        
            // update kernel parameters
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
            clSetKernelArg(kernel, 2, work_group_size * sizeof(int), NULL);
            clSetKernelArg(kernel, 3, sizeof(size_t), &curLength);
        
            // submit kernel
            CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
        
            // update curLength
            curLength = global_size / work_group_size;
            
            // swap buffers
            cl_mem tmp = devDataA;
            devDataA = devDataB;
            devDataB = tmp;
            
            // count number of steps
            numStages++;
        }
        clFinish(command_queue);
        timestamp end_reduce = now();
        printf("\t%d stages of reductions took: %.3f ms\n", numStages, (end_reduce-begin_reduce)*1000);

        
        // download result from device
        err = clEnqueueReadBuffer(command_queue, devDataA, CL_TRUE, 0, sizeof(int), &count, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to download result from device");

        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
        CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");

        // free device memory
        CLU_ERRCHECK(clReleaseMemObject(devDataA), "Failed to release data buffer A");
        CLU_ERRCHECK(clReleaseMemObject(devDataB), "Failed to release data buffer B");

        // free management resources
        CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
        CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
    }
    timestamp end = now();
    printf("\ttook: %.3f ms\n", (end-begin)*1000);


    // -------- print result -------

    printf("Number of 1s: %d - %.2f%%\n", count, (count/(float)N)*100);

    // ---------- cleanup ----------
    
    free(data);
    
    // done
    return EXIT_SUCCESS;
}

