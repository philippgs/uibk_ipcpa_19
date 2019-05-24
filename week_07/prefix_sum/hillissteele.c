
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"

int main(int argc, char** argv) {

    int N = 20;
    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    
    printf("Running single-workgroup prefix sum on %d elements.\n", N);

    // generate a list of values to be 'prefix-summed'
    int* A = malloc(N*sizeof(int));
    for(int i=0; i<N; i++) {
        A[i] = i+1;
    }
    
    // --- Begin of OpenCL part ---
    
    // compute prefix sums (off-by-one, out-of-place)
    int* S = malloc(N*sizeof(int));
    double start = now();
    {
        // - setup -
        cl_int err;
                
        size_t work_group_size = N;
        printf("Input size of %d requires work group size %lu\n", N, work_group_size);

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "hillissteele.cl", NULL);
        cl_kernel kernel = clCreateKernel(program, "sum_scan", &err);
        CLU_ERRCHECK(err, "Failed to create kernel from program");
        
        // check that work group size is valid
        size_t max_work_group_size = 0;
        clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
        
        printf("Max work group size: %lu\n", max_work_group_size);
        if (work_group_size > max_work_group_size) {
            printf("Input too large for single work group, max group size: %lu\n", max_work_group_size);
            
            // clean-up and exit
            CLU_ERRCHECK(clReleaseKernel(kernel),   "Failed to release kernel");
            CLU_ERRCHECK(clReleaseProgram(program), "Failed to release program");
            CLU_ERRCHECK(clReleaseCommandQueue(command_queue), "Failed to release command queue");
            CLU_ERRCHECK(clReleaseContext(context),            "Failed to release OpenCL context");
            
            return EXIT_FAILURE;
        }
        

        // Part 3: create memory buffers
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for output array");

        // Part 4: fill input buffer
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(int), A, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");
        
        // Part 5: compute prefix sum
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &devDataA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &devDataB);
        clSetKernelArg(kernel, 2, 2 * work_group_size * sizeof(int), NULL);     // two times the work group size!
        clSetKernelArg(kernel, 3, sizeof(size_t), &N);    
        
        // submit kernel
        CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &work_group_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
    
        // Part 6: download result from device
        err = clEnqueueReadBuffer(command_queue, devDataB, CL_TRUE, 0, N * sizeof(int), S, 0, NULL, NULL);
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
    double end = now();
    printf("Computation took %.1lfms\n", (end-start)*1000);
    
    // --- End of OpenCL part ---
    
    // check result
    bool fail = false;
    for(int i=0; i<N; i++) {
        int should = (i * (i+1))/2;
        if (S[i] != should) {
            printf("\tERROR at index %i, should %d, is %d\n", i, should, S[i]);
            fail = true;
        }
    }

    if (!fail) {
        printf("Result OK!\n");
    }
    
    // cleanup
    free(A);
    free(S);

    return (fail) ? EXIT_FAILURE : EXIT_SUCCESS;
}

