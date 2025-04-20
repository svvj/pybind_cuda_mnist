#include "header.cuh"
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            return; \
        } \
    } while(0)

__global__ void softmax_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= n) return;

    // 각 블록 내 최대값 계산
    float max_val = -FLT_MAX;
    for (int i = tid; i < n; i += blockDim.x)
        max_val = fmaxf(max_val, input[i]);
    shared[tid] = max_val;
    __syncthreads();

    // 전체 블록에서 최댓값 계산
    float max_all = shared[0];
    for (int i = 1; i < blockDim.x && i < n; ++i)
        max_all = fmaxf(max_all, shared[i]);
    __syncthreads();

    // 각 값에 대해 exp(x - max)
    float val = expf(input[idx] - max_all);
    shared[tid] = val;
    __syncthreads();

    // 총합 계산
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += shared[i];
    shared[tid] = sum;
    __syncthreads();

    // 전체 합 계산
    float total = shared[0];
    for (int i = 1; i < blockDim.x && i < n; ++i)
        total += shared[i];
    __syncthreads();

    // 정규화
    output[idx] = val / total;
}

void softmax(const float* input, float* output, int n, int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks, threads_per_block, threads_per_block * sizeof(float)>>>(input, output, n);
    cudaDeviceSynchronize();
}


// softmax_batch (batch-wise)
__global__ void softmax_batch_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 스레드 수가 클래스 수보다 많을 경우 early return
    if (tid >= num_classes) return;

    int offset = row * num_classes;
    
    // 각 thread가 해당 row에서 하나의 값 계산
    float val = input[offset + tid];

    // 1. shared memory에 각 값 저장
    extern __shared__ float buffer[];  // 크기: 2 * num_classes
    float* vals = buffer;              // 앞쪽은 원래 값
    float* exps = buffer + num_classes; // 뒤쪽은 exp 계산 후 값

    // 범위 내의 스레드만 값을 기록
    vals[tid] = val;
    __syncthreads();

    // 2. max 계산
    float max_val = -FLT_MAX;
    for (int i = 0; i < num_classes; ++i) {
        max_val = fmaxf(max_val, vals[i]);
    }
    __syncthreads();

    // 3. exp 계산
    float exp_val = expf(val - max_val);
    exps[tid] = exp_val;
    __syncthreads();

    // 4. sum 계산
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        sum += exps[i];
    }
    __syncthreads();

    // 5. softmax 결과 저장
    output[offset + tid] = exp_val / sum;
}


void softmax_batch(const float* input, float* output, int batch_size, int num_classes, int threads_per_block) {
    // threads_per_block이 num_classes보다 크면 조정
    threads_per_block = min(threads_per_block, num_classes);
    
    // shared memory 사이즈 지정: 2 * num_classes * sizeof(float) 
    softmax_batch_kernel<<<batch_size, threads_per_block, 2 * num_classes * sizeof(float)>>>(input, output, batch_size, num_classes);
}


// softmax_backward.cu
__global__ void softmax_cross_entropy_backward_kernel(
    const float* probs, const int* labels, float* dx,
    int batch_size, int num_classes) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * num_classes) {
        int sample = idx / num_classes;
        int class_id = idx % num_classes;

        float grad = probs[idx];
        if (class_id == labels[sample]) {
            grad -= 1.0f;
        }
        dx[idx] = grad / batch_size;
    }
}

void softmax_cross_entropy_backward(
    const float* probs, const int* labels, float* dx,
    int batch_size, int num_classes, int threads_per_block) {

    int total = batch_size * num_classes;
    int blocks = (total + threads_per_block - 1) / threads_per_block;

    softmax_cross_entropy_backward_kernel<<<blocks, threads_per_block>>>(
        probs, labels, dx, batch_size, num_classes);
}
