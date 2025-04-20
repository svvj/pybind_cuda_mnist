#ifndef TEST_CUH
#define TEST_CUH

// C++ 래퍼 함수 선언
void vector_add(float *a, float *b, float *result, int n, int threads_per_block);
void matmul(const float* A, const float* B, float* C,
    int M, int N, int K,
    int threads_x, int threads_y,
    bool transA = false, bool transB = false);
void matmul_device(const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K,
    int threads_x, int threads_y,
    bool transA = false, bool transB = false);
void relu(const float* input, float* output, int n, int threads_per_block);
void softmax(const float* x, float* y, int n, int threads_per_block);
void softmax_batch(const float* x, float* y, int batch_size, int num_classes, int threads_per_block);

// MLP Forward
void mlp_forward(const float* x, const float* W1, const float* b1,
    const float* W2, const float* b2,
    float* out, float* hidden,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    int threads_per_block);

// MLP Backward
void mlp_backward(const float* x, const float* hidden, const float* dout,
     const float* W2,
     float* dW1, float* db1,
     float* dW2, float* db2,
     int batch_size, int input_dim, int hidden_dim, int output_dim,
     int threads_per_block);

void cross_entropy_loss(const float* probs, const int* labels, float* loss, int batch_size, int num_classes, int threads_per_block);

void softmax_cross_entropy_backward(
    const float* probs,
    const int* labels,
    float* grad,
    int batch_size,
    int num_classes,
    int threads_per_block);

void sgd(float* param, const float* grad, float learning_rate, int n, int threads_per_block);

#endif // TEST_CUH