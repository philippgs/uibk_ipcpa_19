#include <stdio.h>
#include <stdlib.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "utils.h"

typedef float value_t;

// -- kernel code utils --

typedef struct kernel_code {
    const char* code;
    size_t size;
} kernel_code;

kernel_code loadCode(const char* filename);

void releaseCode(kernel_code code);

// -----------------------


int main(int argc, char** argv) {

    // 'parsing' optional input parameter = problem size
    long long N = 100*1000*1000;
    if (argc > 1) {
        N = atoll(argv[1]);
    }
    printf("Computing vector-add with N=%lld\n", N);

    
    // ---------- setup ----------

    // create two input vectors (on heap!)
    value_t* a = malloc(sizeof(value_t)*N);
    value_t* b = malloc(sizeof(value_t)*N);
    
    // fill vectors
    for(long long i = 0; i<N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }
    
    // ---------- compute ----------
    
    value_t* c = malloc(sizeof(value_t)*N);

    // --- OpenCL part ---
    timestamp begin = now();
    {
        // OpenCL reference pages:
        // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
        
        // some local state variables
        cl_platform_id platform_id = NULL;
        cl_device_id device_id = NULL;
        cl_context context = NULL;
        cl_command_queue command_queue = NULL;
        cl_program program = NULL;
        cl_kernel kernel = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret;

        // TODO: all return codes should be checked!
    
        // Part A - resource management
    
        // 1) get platform
        ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        
        // 2) get device
        ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

        // 3) create context
        context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
        
        // 4) create command queue
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);



        // Part B - data management
        
        // 5) create memory buffers on device
        size_t vec_size = sizeof(value_t) * N;
        cl_mem devVecA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, vec_size, NULL, &ret);
        cl_mem devVecB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY, vec_size, NULL, &ret);
        cl_mem devVecC = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, vec_size, NULL, &ret);

        // 6) transfer input data from host to device (synchronously)
        ret = clEnqueueWriteBuffer(command_queue, devVecA, CL_TRUE, 0, vec_size, &a[0], 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, devVecB, CL_TRUE, 0, vec_size, &b[0], 0, NULL, NULL);



        // Part C - computation

        // 6) load kernel code from file
        kernel_code code = loadCode("vec_add.cl");
        
        // 7) compile kernel program from source
        program = clCreateProgramWithSource(context, 1, &code.code,
				                      (const size_t *)&code.size, &ret);

        // 8) build program (compile + link for device architecture)
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        
        // report kernel build errors
        if (ret != CL_SUCCESS) {

            // create a temporary buffer for the message
            size_t size = 1<<20;    // 1MB
            char* msg = malloc(size);
            size_t msg_size;

            // retrieve the error message
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, size, msg, &msg_size);

            // print the error message
            printf("Build Error:\n%s",msg);
            exit(1);
        }

        // 9) create OpenCL kernel
        kernel = clCreateKernel(program, "vec_add", &ret);

        // 10) set arguments
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &devVecC);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &devVecA);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &devVecB);
        ret = clSetKernelArg(kernel, 3, sizeof(int), &N);

        // 11) schedule kernel
        size_t global_work_offset = 0;
        size_t global_work_size = N;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 
                    1, &global_work_offset, &global_work_size, NULL, 
                    0, NULL, NULL
        );

        // 12) transfer data back to host
        ret = clEnqueueReadBuffer(command_queue, devVecC, CL_TRUE, 0, vec_size, &c[0], 0, NULL, NULL);
        
        // Part D - cleanup
        
        // wait for completed operations (should all have finished already)
        ret = clFlush(command_queue);
        ret = clFinish(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        
        // free device memory
        ret = clReleaseMemObject(devVecA);
        ret = clReleaseMemObject(devVecB);
        ret = clReleaseMemObject(devVecC);
        
        // free management resources
        ret = clReleaseCommandQueue(command_queue);
        ret = clReleaseContext(context);

    }
    timestamp end = now();
    printf("Total time: %.3f ms\n", (end-begin)*1000);

    // ---------- check ----------    
    
    bool success = true;
    for(long long i = 0; i<N; i++) {
        if (c[i] == a[i] + b[i]) continue;
        success = false;
        break;
    }
    
    printf("Verification: %s\n", (success)?"OK":"FAILED");
    
    // ---------- cleanup ----------
    
    free(a);
    free(b);
    free(c);
    
    // done
    return (success) ? EXIT_SUCCESS : EXIT_FAILURE;
}

kernel_code loadCode(const char* filename) {
    size_t MAX_SOURCE_SIZE = 0x100000;

    FILE* fp;

    // load the source code containing the kernel
    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel from file %s\n", filename);
        exit(1);
    }
    
    kernel_code res;
    res.code = (char*)malloc(MAX_SOURCE_SIZE);
    res.size = fread((char*)res.code, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    return res;
}

void releaseCode(kernel_code code) {
    free((char*)code.code);
}
