#include <cuda_runtime.h>
#include <math_constants.h>  // for CUDART_INF_F
#include <math.h>

__global__ void cross_entropy_loss_kernel(const float* probs, const int* labels, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        int label = labels[idx];
        float p = probs[idx * num_classes + label];
        loss[idx] = -logf(fmaxf(p, 1e-8f));  // log(0) 방지
    }
}

void cross_entropy_loss(const float* probs, const int* labels, float* loss, int batch_size, int num_classes, int threads_per_block) {
    int grid_size = (batch_size + threads_per_block - 1) / threads_per_block;
    cross_entropy_loss_kernel<<<grid_size, threads_per_block>>>(probs, labels, loss, batch_size, num_classes);
    cudaDeviceSynchronize();
}
