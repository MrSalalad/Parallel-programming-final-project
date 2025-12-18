#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Namespace chứa các kernel ngây thơ (Naive Implementation)
namespace NaiveKernels {

    // 1. Convolution 2D Kernel
    // Mỗi thread tính 1 pixel output
    __global__ void conv2d_kernel(const float* input, float* output,
                                  const float* weights, const float* bias,
                                  int batch_size, int in_channels, int out_channels,
                                  int in_height, int in_width);

    // 2. ReLU Activation Kernel
    // Mỗi thread xử lý 1 phần tử
    __global__ void relu_kernel(float* data, int size);
    
    // Gradient của ReLU (cho Backward)
    __global__ void relu_backward_kernel(const float* input, const float* grad_output, 
                                         float* grad_input, int size);

    // 3. Max Pooling Kernel
    // Mỗi thread tính 1 pixel output (từ vùng 2x2)
    __global__ void max_pool_kernel(const float* input, float* output,
                                    int batch_size, int channels, 
                                    int in_height, int in_width);
                                    
    // Gradient của MaxPool
    __global__ void max_pool_backward_kernel(const float* input, const float* grad_output,
                                             float* grad_input,
                                             int batch_size, int channels, 
                                             int in_height, int in_width);

    // 4. Upsampling Kernel (Nearest Neighbor)
    __global__ void upsample_kernel(const float* input, float* output,
                                    int batch_size, int channels, 
                                    int in_height, int in_width);

    // Gradient của Upsample
    __global__ void upsample_backward_kernel(const float* grad_output, float* grad_input,
                                             int batch_size, int channels, 
                                             int in_height, int in_width);

    // 5. MSE Loss (Sẽ dùng kết hợp với reduction sau này, tạm thời tính diff trước)
    __global__ void mse_loss_kernel(const float* output, const float* target, 
                                    float* diff_sum, int size);
                                    
    // Gradient của MSE
    __global__ void mse_backward_kernel(const float* output, const float* target,
                                        float* grad_input, int size);
}

#endif