#include "header.cuh"

__global__ void relu_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

void relu(const float* input, float* output, int n, int threads_per_block) {
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (n + threads_per_block - 1) / threads_per_block;
    relu_kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
