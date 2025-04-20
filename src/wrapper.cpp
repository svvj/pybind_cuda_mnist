#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include "header.cuh"

namespace py = pybind11;

template <typename T>
class CudaBuffer {
public:
    T* ptr;
    size_t size;

    CudaBuffer(size_t count) : ptr(nullptr), size(count) {
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            std::string msg = "cudaMalloc failed: ";
            msg += cudaGetErrorString(err);  // << 이 부분 추가
            throw std::runtime_error(msg);
        }
    }

    ~CudaBuffer() {
        if (ptr) cudaFree(ptr);
    }

    // 복사 금지
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // 이동 허용
    CudaBuffer(CudaBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }
};



py::array_t<float> vector_add_wrapper(py::array_t<float> a, py::array_t<float> b,
                                int threads_per_block = 256)
{
    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.ndim != 1 || b_info.ndim != 1)
        throw std::runtime_error("벡터의 차원은 1이어야 합니다");
    if (a_info.shape[0] != b_info.shape[0])
        throw std::runtime_error("입력 벡터의 크기가 일치해야 합니다");

    size_t n = a_info.shape[0];
    py::array_t<float> result(n);
    py::buffer_info result_info = result.request();

    CudaBuffer<float> d_a(n), d_b(n), d_result(n);

    cudaMemcpy(d_a.ptr, a_info.ptr, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.ptr, b_info.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    vector_add(d_a.ptr, d_b.ptr, d_result.ptr, n, threads_per_block);

    cudaMemcpy(result_info.ptr, d_result.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}


py::array_t<float> matmul_wrapper(py::array_t<float> a, py::array_t<float> b,
                                int M, int N, int K,
                                py::tuple threads = py::make_tuple(16, 16))
{
    int threads_x = threads[0].cast<int>();
    int threads_y = threads[1].cast<int>();

    py::buffer_info a_info = a.request();
    py::buffer_info b_info = b.request();

    if (a_info.ndim != 1 || b_info.ndim != 1)
        throw std::runtime_error("입력은 1차원이어야 합니다 (flatten된 배열)");

    if (a_info.size != M * K || b_info.size != K * N)
        throw std::runtime_error("행렬 크기가 일치하지 않습니다.");

    py::array_t<float> result(M * N);
    py::buffer_info r_info = result.request();

    CudaBuffer<float> d_A(M * K);
    CudaBuffer<float> d_B(K * N);
    CudaBuffer<float> d_C(M * N);

    cudaMemcpy(d_A.ptr, a_info.ptr, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.ptr, b_info.ptr, K * N * sizeof(float), cudaMemcpyHostToDevice);

    matmul(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K, threads_x, threads_y);

    cudaMemcpy(r_info.ptr, d_C.ptr, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    result.resize({M, N});
    return result;
}

py::array_t<float> relu_wrapper(py::array_t<float> x, int threads_per_block = 256) {
    py::buffer_info x_info = x.request();

    if (x_info.ndim != 1)
        throw std::runtime_error("ReLU 입력은 1차원이어야 합니다.");

    size_t n = x_info.shape[0];
    py::array_t<float> result(n);
    py::buffer_info r_info = result.request();

    relu(static_cast<float*>(x_info.ptr),
         static_cast<float*>(r_info.ptr),
         static_cast<int>(n),
         threads_per_block);

    return result;
}


py::array_t<float> softmax_wrapper(py::array_t<float> x, int threads_per_block = 256) {
    py::buffer_info x_info = x.request();

    if (x_info.ndim != 1)
        throw std::runtime_error("Softmax 입력은 1차원이어야 합니다.");

    size_t n = x_info.shape[0];
    py::array_t<float> result(n);
    py::buffer_info r_info = result.request();

    CudaBuffer<float> d_input(n), d_output(n);

    cudaMemcpy(d_input.ptr, x_info.ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    softmax(d_input.ptr, d_output.ptr, static_cast<int>(n), threads_per_block);

    cudaMemcpy(r_info.ptr, d_output.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

py::array_t<float> softmax_batch_wrapper(py::array_t<float> x,
                                         int batch_size, int num_classes,
                                         int threads_per_block = 256)
{
    py::buffer_info x_info = x.request();

    if (x_info.ndim != 2)
        throw std::runtime_error("Softmax 입력은 2차원(batch_size, num_classes)이어야 합니다.");
    if (x_info.shape[0] != batch_size || x_info.shape[1] != num_classes)
        throw std::runtime_error("입력 크기와 batch_size/num_classes가 일치하지 않습니다.");

    py::array_t<float> result({batch_size, num_classes});
    py::buffer_info result_info = result.request();

    CudaBuffer<float> d_input(batch_size * num_classes);
    CudaBuffer<float> d_output(batch_size * num_classes);

    cudaMemcpy(d_input.ptr, x_info.ptr, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);

    softmax_batch(d_input.ptr, d_output.ptr, batch_size, num_classes, threads_per_block);
    // 커널 실행 직후 에러 확인
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in softmax_batch: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(result_info.ptr, d_output.ptr, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

py::float_ cross_entropy_loss_wrapper(py::array_t<float> probs, py::array_t<int> labels,
    int batch_size, int num_classes,
    int threads_per_block = 256)
{
    py::buffer_info probs_info = probs.request();
    py::buffer_info labels_info = labels.request();

    if (probs_info.ndim != 2 || labels_info.ndim != 1)
        throw std::runtime_error("입력 차원이 올바르지 않습니다.");
    if (probs_info.shape[0] != batch_size || probs_info.shape[1] != num_classes)
        throw std::runtime_error("probs shape mismatch");
    if (labels_info.shape[0] != batch_size)
        throw std::runtime_error("labels shape mismatch");

    py::array_t<float> result(batch_size);
    py::buffer_info result_info = result.request();

    CudaBuffer<float> d_probs(batch_size * num_classes);
    CudaBuffer<int> d_labels(batch_size);
    CudaBuffer<float> d_loss(batch_size);

    cudaMemcpy(d_probs.ptr, probs_info.ptr, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels.ptr, labels_info.ptr, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    cross_entropy_loss(d_probs.ptr, d_labels.ptr, d_loss.ptr, batch_size, num_classes, threads_per_block);

    cudaMemcpy(result_info.ptr, d_loss.ptr, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0f;
    float* loss_data = static_cast<float*>(result_info.ptr);
    for (int i = 0; i < batch_size; ++i) {
        total_loss += loss_data[i];
    }
    float avg_loss = total_loss / batch_size;

    return py::float_(avg_loss);
}

py::array_t<float> softmax_cross_entropy_backward_wrapper(py::array_t<float> probs,
    py::array_t<int> labels,
    int batch_size, int num_classes,
    int threads_per_block = 256) 
{
    py::buffer_info probs_info = probs.request();
    py::buffer_info labels_info = labels.request();

    if (probs_info.ndim != 2 || labels_info.ndim != 1)
        throw std::runtime_error("입력 차원이 올바르지 않습니다.");
    if (probs_info.shape[0] != batch_size || probs_info.shape[1] != num_classes)
        throw std::runtime_error("probs shape mismatch");
    if (labels_info.shape[0] != batch_size)
        throw std::runtime_error("labels shape mismatch");

    py::array_t<float> grad(batch_size * num_classes);
    py::buffer_info grad_info = grad.request();

    CudaBuffer<float> d_probs(batch_size * num_classes);
    CudaBuffer<int> d_labels(batch_size);
    CudaBuffer<float> d_grad(batch_size * num_classes);

    cudaMemcpy(d_probs.ptr, probs_info.ptr, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels.ptr, labels_info.ptr, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    softmax_cross_entropy_backward(d_probs.ptr, d_labels.ptr, d_grad.ptr,
    batch_size, num_classes, threads_per_block);

    cudaMemcpy(grad_info.ptr, d_grad.ptr, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost);

    grad.resize({batch_size, num_classes});
    return grad;
}


py::tuple mlp_forward_wrapper(
    py::array_t<float> x,
    py::array_t<float> W1,
    py::array_t<float> b1,
    py::array_t<float> W2,
    py::array_t<float> b2,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    int threads_per_block = 256)
{
    // buffer_info 추출
    py::buffer_info x_info = x.request();
    py::buffer_info W1_info = W1.request();
    py::buffer_info b1_info = b1.request();
    py::buffer_info W2_info = W2.request();
    py::buffer_info b2_info = b2.request();

    py::array_t<float> out({batch_size, output_dim});
    py::array_t<float> hidden({batch_size, hidden_dim});
    py::buffer_info out_info = out.request();
    py::buffer_info hidden_info = hidden.request();

    mlp_forward(static_cast<float*>(x_info.ptr),
                static_cast<float*>(W1_info.ptr),
                static_cast<float*>(b1_info.ptr),
                static_cast<float*>(W2_info.ptr),
                static_cast<float*>(b2_info.ptr),
                static_cast<float*>(out_info.ptr),
                static_cast<float*>(hidden_info.ptr),
                batch_size, input_dim, hidden_dim, output_dim,
                threads_per_block);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "MLP Forward Error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    return py::make_tuple(out, hidden);
}


void mlp_backward_wrapper(
    py::array_t<float> x,
    py::array_t<float> hidden,
    py::array_t<float> d_out,
    py::array_t<float> W2,
    py::array_t<float> dW1,
    py::array_t<float> db1,
    py::array_t<float> dW2,
    py::array_t<float> db2,
    int batch_size, int input_dim, int hidden_dim, int output_dim,
    int threads_per_block = 256)
{
    // buffer_info 추출
    py::buffer_info x_info = x.request();
    py::buffer_info hidden_info = hidden.request();
    py::buffer_info d_out_info = d_out.request();
    py::buffer_info W2_info = W2.request();
    py::buffer_info dW1_info = dW1.request();
    py::buffer_info db1_info = db1.request();
    py::buffer_info dW2_info = dW2.request();
    py::buffer_info db2_info = db2.request();

    // MLP backward pass 호출
    mlp_backward(
        static_cast<float*>(x_info.ptr),
        static_cast<float*>(hidden_info.ptr),
        static_cast<float*>(d_out_info.ptr),
        static_cast<float*>(W2_info.ptr),
        static_cast<float*>(dW1_info.ptr),
        static_cast<float*>(db1_info.ptr),
        static_cast<float*>(dW2_info.ptr),
        static_cast<float*>(db2_info.ptr),
        batch_size, input_dim, hidden_dim, output_dim, threads_per_block
    );
}

void sgd_wrapper(
    py::array_t<float> param,
    py::array_t<float> grad,
    float learning_rate,
    int size,
    int threads_per_block = 256)
{
    py::buffer_info param_info = param.request();
    py::buffer_info grad_info = grad.request();
    
    if (param_info.size != size || grad_info.size != size)
        throw std::runtime_error("Parameter and gradient size mismatch");
        
    sgd(
        static_cast<float*>(param_info.ptr),
        static_cast<float*>(grad_info.ptr),
        learning_rate,
        size,
        threads_per_block
    );
}


///////////////////// Python 모듈 정의 ////////////////////////
PYBIND11_MODULE(my_cuda_module, m) {
    m.doc() = "pybind11을 사용한 CUDA 연산";
    
    m.def("vector_add", &vector_add_wrapper, 
          "CUDA를 사용하여 두 벡터를 더합니다",
          py::arg("a"), py::arg("b"), py::arg("threads_per_block"));

    m.def("matmul", &matmul_wrapper, 
        "CUDA 행렬 곱", 
        py::arg("a"), py::arg("b"), py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("threads"));

    m.def("relu", &relu_wrapper,
        "CUDA ReLU 활성화 함수",
        py::arg("x"), py::arg("threads_per_block") = 256);
      
    m.def("softmax", &softmax_wrapper,
        "CUDA softmax 연산",
        py::arg("x"), py::arg("threads_per_block") = 256);
    
    // m.def("softmax", &softmax_batch_wrapper,
    //     "batch softmax",
    //     py::arg("x"), py::arg("batch_size"), py::arg("num_classes"), py::arg("threads_per_block") = 256);

    m.def("softmax_batch", &softmax_batch_wrapper,
        "Batch-wise softmax (2D input)",
        py::arg("x"), py::arg("batch_size"), py::arg("num_classes"),
        py::arg("threads_per_block") = 256);

    m.def("cross_entropy_loss", &cross_entropy_loss_wrapper,
        "Softmax 후의 cross entropy loss 계산",
        py::arg("probs"), py::arg("labels"),
        py::arg("batch_size"), py::arg("num_classes"),
        py::arg("threads_per_block") = 256);
      
    m.def("softmax_cross_entropy_backward", &softmax_cross_entropy_backward_wrapper,
        "Softmax + CrossEntropy Backward 계산",
        py::arg("probs"), py::arg("labels"),
        py::arg("batch_size"), py::arg("num_classes"),
        py::arg("threads_per_block") = 256);
    
    m.def("mlp_forward", &mlp_forward_wrapper,
        "MLP forward pass",
        py::arg("x"), py::arg("W1"), py::arg("b1"),
        py::arg("W2"), py::arg("b2"),
        py::arg("batch_size"), py::arg("input_dim"),
        py::arg("hidden_dim"), py::arg("output_dim"),
        py::arg("threads_per_block") = 256);
      
    // PYBIND11_MODULE 내부:
    m.def("mlp_backward", &mlp_backward_wrapper,
        "MLP backward pass",
        py::arg("x"), py::arg("hidden"), py::arg("d_out"), py::arg("W2"),
        py::arg("dW1"), py::arg("db1"), py::arg("dW2"), py::arg("db2"),
        py::arg("batch_size"), py::arg("input_dim"), 
        py::arg("hidden_dim"), py::arg("output_dim"),
        py::arg("threads_per_block") = 256);

    m.def("sgd", &sgd_wrapper, "SGD optimizer",
        py::arg("param"), py::arg("grad"), py::arg("lr"), py::arg("size"), 
        py::arg("threads_per_block") = 256);
}