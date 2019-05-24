
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "cl_utils.h"

int roundUpToPowerOfTwo(int N);
int roundUpToMultiple(int N, int B);

// a function computing the prefix sum of a given on-device data buffer
void prefixSum(
    cl_context context,         // < the OpenCL context to run the reduction in
    cl_command_queue queue,     // < the command queue to use
    cl_kernel reduce,           // < the kernel computing local prefix sums and total sums
    cl_kernel expand,           // < the kernel adding offsets to partial results
    size_t work_group_size,     // < the work group size to be used for the operations
    cl_mem result,              // < the on-device result buffer
    cl_mem data,                // < the on-device input buffer
    size_t size                 // < the number of elements in the input buffer
);

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

        // Part 1: ocl initialization
        cl_context context;
        cl_command_queue command_queue;
        cl_device_id device_id = cluInitDevice(0, &context, &command_queue);

        // Part 2: create kernel from source
        cl_program program = cluBuildProgramFromFile(context, device_id, "prefixglobal.cl", NULL);
        cl_kernel reduce = clCreateKernel(program, "sum_scan_reduce", &err);
        CLU_ERRCHECK(err, "Failed to create kernel from program");
        
        cl_kernel expand = clCreateKernel(program, "sum_scan_expand", &err);
        CLU_ERRCHECK(err, "Failed to create kernel from program");
        
        // check that work group size is valid
        size_t reduce_work_group_size = 0;
        size_t expand_work_group_size = 0;
        clGetKernelWorkGroupInfo(reduce, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &reduce_work_group_size, NULL);
        clGetKernelWorkGroupInfo(expand, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &expand_work_group_size, NULL);
        size_t work_group_size = reduce_work_group_size < expand_work_group_size ? reduce_work_group_size : expand_work_group_size;
        printf("Using work group size: %lu\n", work_group_size);

        // Part 3: create memory buffers
        cl_mem devDataA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for input array");

        cl_mem devDataB = clCreateBuffer(context, CL_MEM_READ_WRITE, N *sizeof(int), NULL, &err);
        CLU_ERRCHECK(err, "Failed to create buffer for output array");

        // Part 4: fill input buffer
        err = clEnqueueWriteBuffer(command_queue, devDataA, CL_TRUE, 0, N * sizeof(int), A, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to write data to device");
        
        // Part 5: compute prefix sum
        prefixSum(context,command_queue,reduce,expand,work_group_size,devDataB,devDataA,N);

        // Part 6: download result from device
        err = clEnqueueReadBuffer(command_queue, devDataB, CL_TRUE, 0, N * sizeof(int), S, 0, NULL, NULL);
        CLU_ERRCHECK(err, "Failed to download result from device");

        // Part 7: cleanup
        // wait for completed operations (there should be none)
        CLU_ERRCHECK(clFlush(command_queue),    "Failed to flush command queue");
        CLU_ERRCHECK(clFinish(command_queue),   "Failed to wait for command queue completion");
        CLU_ERRCHECK(clReleaseKernel(reduce),   "Failed to release reduce kernel");
        CLU_ERRCHECK(clReleaseKernel(expand),   "Failed to release expand kernel");
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
        int should = ((long)i * (i+1))/2;
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

int roundUpToPowerOfTwo(int N) {
    int res = 1;
    while (res < N) res = res << 1;
    return res;
}

int roundUpToMultiple(int N, int B) {
    if ((N % B) == 0) return N;
    return N + (B - (N%B));
}

void prefixSum(cl_context context, cl_command_queue queue, cl_kernel reduce, cl_kernel expand, size_t work_group_size, cl_mem result, cl_mem data, size_t size) {
    
    // compute the global size
    size_t global_size = size/2 + size%2;     // each kernel thread covers 2 elements
    
    // adapt to a multipe of the work group size
    global_size = roundUpToMultiple(global_size,work_group_size);
    printf("Running reduction of %lu elements using %lu threads ...\n", size, global_size);

    // get the number of work groups to start
    size_t num_groups = global_size / work_group_size;

    // -- reduction step --

    // add buffer for temporary sum array
    cl_int err;
    cl_mem devDataSum = clCreateBuffer(context, CL_MEM_READ_WRITE, num_groups * sizeof(int), NULL, &err);
    CLU_ERRCHECK(err, "Failed to create buffer for sum array");

    // set kernel arguments
    clSetKernelArg(reduce, 0, sizeof(cl_mem), &data);
    clSetKernelArg(reduce, 1, sizeof(cl_mem), &result);
    clSetKernelArg(reduce, 2, sizeof(cl_mem), &devDataSum);
    clSetKernelArg(reduce, 3, 2 * work_group_size * sizeof(int), NULL);     // two times the work group size!
    clSetKernelArg(reduce, 4, sizeof(size_t), &size);    
    
    // submit kernel
    CLU_ERRCHECK(clEnqueueNDRangeKernel(queue, reduce, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");

    // -- recursive step --

    // see whether this reduction invocation was the last required
    if (size <= 2*work_group_size) {
        CLU_ERRCHECK(clReleaseMemObject(devDataSum), "Failed to release sum data buffer.");    
        return;
    }
    
    // run recursive prefix-sum
    cl_mem devDataRes = clCreateBuffer(context, CL_MEM_READ_WRITE, num_groups * sizeof(int), NULL, &err);
    prefixSum(context,queue,reduce,expand,work_group_size,devDataRes,devDataSum,num_groups);
    
    // -- expansion step --
    
    // extend result
    clSetKernelArg(expand, 0, sizeof(cl_mem), &result);
    clSetKernelArg(expand, 1, sizeof(cl_mem), &devDataRes);
    clSetKernelArg(expand, 2, sizeof(size_t), &size);    
    
    printf("Running expansion from %lu to %lu elements using %lu threads...\n", num_groups, size, global_size);
    
    // submit kernel
    CLU_ERRCHECK(clEnqueueNDRangeKernel(queue, expand, 1, NULL, &global_size, &work_group_size, 0, NULL, NULL), "Failed to enqueue reduction kernel");
    
    // release data buffer
    CLU_ERRCHECK(clReleaseMemObject(devDataSum), "Failed to release sum data buffer.");    
    CLU_ERRCHECK(clReleaseMemObject(devDataRes), "Failed to release res data buffer.");    
}

