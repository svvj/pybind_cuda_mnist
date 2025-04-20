#include "header.cuh"

__global__ void vector_add_kernel(float *a, float *b, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

void vector_add(float *a, float *b, float *result, int n, int threads_per_block) {
    // 그리드 크기 계산
    int gridSize = (n + threads_per_block - 1) / threads_per_block;

    // 커널 실행
    vector_add_kernel<<<gridSize, threads_per_block>>>(a, b, result, n);

    // GPU 작업 완료 대기 (디버깅 및 동기화용)
    cudaDeviceSynchronize();
}
