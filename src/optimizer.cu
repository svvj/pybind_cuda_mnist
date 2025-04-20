#include "header.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sgd_kernel(float* param, const float* grad, float learning_rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= learning_rate * grad[idx];
    }
}

void sgd(float* param, const float* grad, float learning_rate, int n, int threads_per_block) {
    // 디바이스 메모리 할당
    float *d_param, *d_grad;
    cudaMalloc(&d_param, n * sizeof(float));
    cudaMalloc(&d_grad, n * sizeof(float));
    
    // 호스트 → 디바이스 복사
    cudaMemcpy(d_param, param, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, grad, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // 커널 실행
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    sgd_kernel<<<blocks, threads_per_block>>>(d_param, d_grad, learning_rate, n);
    
    // 디바이스 → 호스트 복사 (업데이트된 파라미터)
    cudaMemcpy(param, d_param, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 메모리 해제
    cudaFree(d_param);
    cudaFree(d_grad);
    
    // 오류 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA SGD error: %s\n", cudaGetErrorString(err));
    }
}