#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "header.cuh"
#include <stdio.h>


//// MLP forward pass
__global__ void bias_relu_forward_kernel(float* x, const float* b, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * dim) return;

    int col = idx % dim;
    x[idx] += b[col];
    x[idx] = fmaxf(x[idx], 0.0f);  // ReLU
}

__global__ void bias_only_kernel(float* x, const float* b, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * dim) return;

    int col = idx % dim;
    x[idx] += b[col];  // 바이어스만 적용, ReLU 없음
}

void mlp_forward(const float* x, const float* W1, const float* b1,
    const float* W2, const float* b2,
    float* out, float* hidden,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    int threads_per_block)
{
    // 디바이스 메모리 할당
    float *d_x, *d_W1, *d_b1, *d_W2, *d_b2, *d_out, *d_hidden;
    size_t size_x = batch_size * input_dim * sizeof(float);
    size_t size_W1 = input_dim * hidden_dim * sizeof(float);
    size_t size_b1 = hidden_dim * sizeof(float);
    size_t size_W2 = hidden_dim * output_dim * sizeof(float);
    size_t size_b2 = output_dim * sizeof(float);
    size_t size_hidden = batch_size * hidden_dim * sizeof(float);
    size_t size_out = batch_size * output_dim * sizeof(float);
    
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_W1, size_W1);
    cudaMalloc(&d_b1, size_b1);
    cudaMalloc(&d_W2, size_W2);
    cudaMalloc(&d_b2, size_b2);
    cudaMalloc(&d_hidden, size_hidden);
    cudaMalloc(&d_out, size_out);
    
    // 데이터 복사
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, size_W1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, size_b1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, size_W2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, size_b2, cudaMemcpyHostToDevice);
    
    // Linear1: d_hidden = d_x @ d_W1
    matmul_device(d_x, d_W1, d_hidden, batch_size, hidden_dim, input_dim, 16, 16, false, false);
    
    // bias + ReLU
    int hidden_total = batch_size * hidden_dim;
    int blocks1 = (hidden_total + threads_per_block - 1) / threads_per_block;
    bias_relu_forward_kernel<<<blocks1, threads_per_block>>>(d_hidden, d_b1, batch_size, hidden_dim);
    
    // Linear2: d_out = d_hidden @ d_W2  
    matmul_device(d_hidden, d_W2, d_out, batch_size, output_dim, hidden_dim, 16, 16);
    
    // bias2 (ReLU 없음)
    int out_total = batch_size * output_dim;
    int blocks2 = (out_total + threads_per_block - 1) / threads_per_block;
    bias_only_kernel<<<blocks2, threads_per_block>>>(d_out, d_b2, batch_size, output_dim);
    
    // 결과를 호스트로 복사
    cudaMemcpy(hidden, d_hidden, size_hidden, cudaMemcpyDeviceToHost);
    cudaMemcpy(out, d_out, size_out, cudaMemcpyDeviceToHost);
    
    // 메모리 해제
    cudaFree(d_x);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_hidden);
    cudaFree(d_out);
}

////////////// ReLU backward ///////////////
__global__ void relu_backward_kernel(const float* hidden, float* d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hidden[idx] = hidden[idx] > 0.0f ? d_hidden[idx] : 0.0f;
    }
}

__global__ void bias_grad_kernel(const float* d_out, float* db, int batch, int dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;

    float sum = 0.0f;
    for (int i = 0; i < batch; ++i) {
        sum += d_out[i * dim + col];
    }
    db[col] = sum;
}


void mlp_backward(const float* x, const float* hidden, const float* d_out,
    const float* W2, float* dW1, float* db1, float* dW2, float* db2,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    int threads_per_block)
{
    // 디바이스 메모리 할당
    float *d_x, *d_hidden, *d_dout, *d_W2;
    float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dhidden;
    
    size_t size_x = batch_size * input_dim * sizeof(float);
    size_t size_hidden = batch_size * hidden_dim * sizeof(float);
    size_t size_dout = batch_size * output_dim * sizeof(float);
    size_t size_W2 = hidden_dim * output_dim * sizeof(float);
    size_t size_dW1 = input_dim * hidden_dim * sizeof(float);
    size_t size_db1 = hidden_dim * sizeof(float);
    size_t size_dW2 = hidden_dim * output_dim * sizeof(float);
    size_t size_db2 = output_dim * sizeof(float);
    
    cudaMalloc(&d_x, size_x);
    cudaMalloc(&d_hidden, size_hidden);
    cudaMalloc(&d_dout, size_dout);
    cudaMalloc(&d_W2, size_W2);
    cudaMalloc(&d_dW1, size_dW1);
    cudaMalloc(&d_db1, size_db1);
    cudaMalloc(&d_dW2, size_dW2);
    cudaMalloc(&d_db2, size_db2);
    cudaMalloc(&d_dhidden, size_hidden);
    
    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden, size_hidden, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dout, d_out, size_dout, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, size_W2, cudaMemcpyHostToDevice);
    cudaMemset(d_dW1, 0, size_dW1);
    cudaMemset(d_db1, 0, size_db1);
    cudaMemset(d_dW2, 0, size_dW2);
    cudaMemset(d_db2, 0, size_db2);
    cudaMemset(d_dhidden, 0, size_hidden);
    
    // db2 계산 (dout 합계)
    // TODO: 커널 구현 또는 라이브러리 함수 사용
    
    // dW2 계산 (hidden^T @ dout)
    matmul_device(d_hidden, d_dout, d_dW2, hidden_dim, output_dim, batch_size, 16, 16, true, false);
    
    // d_hidden 계산 (dout @ W2^T)
    matmul_device(d_dout, d_W2, d_dhidden, batch_size, hidden_dim, output_dim, 16, 16, false, true);
    
    // ReLU 역전파
    int num_elements = batch_size * hidden_dim;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    relu_backward_kernel<<<blocks, threads_per_block>>>(d_hidden, d_dhidden, num_elements);
    
    // db1 계산 (d_hidden 합계)
    // TODO: 커널 구현 또는 라이브러리 함수 사용
    
    // dW1 계산 (x^T @ d_hidden)
    matmul_device(d_x, d_dhidden, d_dW1, input_dim, hidden_dim, batch_size, 16, 16, true, false);
    
    // 결과를 호스트로 복사
    cudaMemcpy(dW1, d_dW1, size_dW1, cudaMemcpyDeviceToHost);
    cudaMemcpy(db1, d_db1, size_db1, cudaMemcpyDeviceToHost);
    cudaMemcpy(dW2, d_dW2, size_dW2, cudaMemcpyDeviceToHost);
    cudaMemcpy(db2, d_db2, size_db2, cudaMemcpyDeviceToHost);
    
    // 메모리 해제
    cudaFree(d_x);
    cudaFree(d_hidden);
    cudaFree(d_dout);
    cudaFree(d_W2);
    cudaFree(d_dW1);
    cudaFree(d_db1);
    cudaFree(d_dW2);
    cudaFree(d_db2);
    cudaFree(d_dhidden);
}