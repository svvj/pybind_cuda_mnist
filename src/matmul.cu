#include <stdio.h>

extern "C" __global__ void matmul_kernel(const float* A, const float* B, float* C, 
    int M, int N, int K,
    bool transA, bool transB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a = transA ? A[i * M + row] : A[row * K + i];
            float b = transB ? B[col * K + i] : B[i * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C,
            int M, int N, int K,
            int threads_x, int threads_y,
            bool transA=false, bool transB=false)
{
    // 포인터 검증
    if (A == nullptr || B == nullptr || C == nullptr) {
        printf("Error: null pointer in matmul\n");
        return;
    }
    
    // 차원 검증
    if (M <= 0 || N <= 0 || K <= 0) {
        printf("Error: invalid dimensions in matmul: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }

    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // 블록 및 그리드 계산 로직
    dim3 block(threads_x, threads_y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    size_t sharedMemSize = 0;  // 필요한 공유 메모리 크기 계산
    matmul_kernel<<<grid, block, sharedMemSize>>>(d_A, d_B, d_C, M, N, K, transA, transB);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 디바이스 메모리에 있는 행렬에 대한 matmul 함수
void matmul_device(const float* d_A, const float* d_B, float* d_C,
            int M, int N, int K,
            int threads_x, int threads_y,
            bool transA=false, bool transB=false)
{
    // 포인터가 이미 디바이스에 있다고 가정
    dim3 block(threads_x, threads_y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 커널 실행
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, transA, transB);
    
    // 오류 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
}
