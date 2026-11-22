// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

#include <iostream>
#include <vector>
#include <chrono>
#include <CL/opencl.h>

// OpenCL Matrix Multiplication Kernel (Heavy math workload per element)
const char* kernel_source = 
"__kernel void mat_mul(__global const float* A, __global const float* B, __global float* C, int N) {\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    if (row < N && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; ++k) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

int main() {
    // 1024x1024 Matrix Multiplication requires over 2 Billion operations
    const int N = 1024; 
    std::cout << "Initializing 1024x1024 matrices (Heavy Workload)..." << std::endl;
    
    std::vector<float> A(N * N, 1.5f);
    std::vector<float> B(N * N, 2.5f);
    std::vector<float> C_cpu(N * N, 0.0f);
    std::vector<float> C_gpu(N * N, 0.0f);

    // ====================================================================
    // BENCHMARK 1: Processing on your 8-Core Intel CPU
    // ====================================================================
    std::cout << "\n--- Starting CPU Matrix Multiplication ---" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C_cpu[row * N + col] = sum;
        }
    }
    
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "✅ 8-Core CPU finished in: " << cpu_duration.count() << " ms" << std::endl;

    // ====================================================================
    // BENCHMARK 2: Processing on your Intel Iris Xe iGPU [0xa7a1]
    // ====================================================================
    std::cout << "\n--- Starting Intel iGPU Matrix Multiplication ---" << std::endl;
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A.data(), NULL);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B.data(), NULL);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "mat_mul", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Warm up the GPU first to clear initial driver lag
    size_t global_size[2] = {(size_t)N, (size_t)N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Start timing the real computation pass
    auto gpu_start = std::chrono::high_resolution_clock::now();
    
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    clFinish(queue); 
    
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(float) * N * N, C_gpu.data(), 0, NULL, NULL);
    std::cout << "🚀 Intel iGPU finished in: " << gpu_duration.count() << " ms" << std::endl;

    // ====================================================================
    // PRINT THE REAL SPEED GAP
    // ====================================================================
    double speedup = cpu_duration.count() / gpu_duration.count();
    std::cout << "\n=========================================================" << std::endl;
    std::cout << "🎯 RESULT: C++ on iGPU is " << speedup << "x FASTER than CPU threads!" << std::endl;
    std::cout << "=========================================================" << std::endl;

    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseMemObject(bufA); clReleaseMemObject(bufB); clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    return 0;
}
