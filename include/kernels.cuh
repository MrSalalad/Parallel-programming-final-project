#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

namespace NaiveKernels {

    // --- 1. CONVOLUTION ---
    
    // Forward: Tính Output
    __global__ void conv2d_forward_kernel(const float* input, float* output, const float* weights, const float* bias,
                                          int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                          int out_h, int out_w, int k_size, int padding, int stride);

    // Backward 1: Tính đạo hàm theo Input (dL/dX) -> Truyền lỗi về lớp trước
    __global__ void conv2d_backward_input_kernel(const float* grad_output, const float* weights, float* grad_input,
                                                 int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                                 int out_h, int out_w, int k_size, int padding, int stride);

    // Backward 2: Tính đạo hàm theo Weights & Bias (dL/dW, dL/db) -> Để update
    __global__ void conv2d_backward_weight_kernel(const float* input, const float* grad_output, 
                                                  float* grad_weights, float* grad_bias,
                                                  int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                                  int out_h, int out_w, int k_size, int padding, int stride);

    // --- 2. RELU ---
    __global__ void relu_forward_kernel(float* data, int size);
    
    __global__ void relu_backward_kernel(const float* input, const float* grad_output, 
                                         float* grad_input, int size);

    // --- 3. MAX POOLING ---
    __global__ void max_pool_forward_kernel(const float* input, float* output,
                                            int batch_size, int c, int in_h, int in_w, int out_h, int out_w);

    __global__ void max_pool_backward_kernel(const float* input, const float* grad_output, float* grad_input,
                                             int batch_size, int c, int in_h, int in_w, int out_h, int out_w);

    // --- 4. UPSAMPLING ---
    __global__ void upsample_forward_kernel(const float* input, float* output,
                                            int batch_size, int c, int in_h, int in_w, int out_h, int out_w);

    __global__ void upsample_backward_kernel(const float* grad_output, float* grad_input,
                                             int batch_size, int c, int in_h, int in_w, int out_h, int out_w);

    // --- 5. LOSS & UPDATE ---
    
    // MSE Forward: Tính bình phương lỗi từng phần tử (cần gọi thêm reduce ở ngoài hoặc atomic bên trong)
    // Ở Phase Naive, ta dùng atomicAdd cho đơn giản
    __global__ void mse_loss_kernel(const float* pred, const float* target, float* total_loss, int size);

    // MSE Backward: Tính dL/dOutput
    __global__ void mse_backward_kernel(const float* pred, const float* target, float* grad_input, int size);

    // SGD Update: W = W - lr * grad
    __global__ void sgd_update_kernel(float* weights, const float* grads, float lr, int size);
}

#endif