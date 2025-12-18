#include "../../include/kernels.cuh"
#include <cstdio>

namespace NaiveKernels {

    // --- CONV2D KERNEL ---
    // Grid Setup: Có thể dùng 3D Grid hoặc 1D linearized Grid.
    // Để đơn giản cho naive: Mỗi thread xử lý 1 pixel của 1 channel output của 1 ảnh.
    // Tổng số thread cần = batch_size * out_channels * height * width
    __global__ void conv2d_kernel(const float* input, float* output,
                                  const float* weights, const float* bias,
                                  int batch_size, int in_channels, int out_channels,
                                  int in_height, int in_width) {
        
        // Tính chỉ số thread tuyến tính
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * out_channels * in_height * in_width;

        if (idx >= total_elements) return;

        // Giải mã idx -> (b, oc, h, w)
        // idx = b * (OC*H*W) + oc * (H*W) + h * W + w
        int w = idx % in_width;
        int h = (idx / in_width) % in_height;
        int oc = (idx / (in_width * in_height)) % out_channels;
        int b = idx / (in_width * in_height * out_channels);

        float sum = bias[oc];
        int kernel_size = 3;
        int padding = 1;

        // Duyệt qua Input Channels (Vòng lặp này thread vẫn phải tự chạy)
        for (int ic = 0; ic < in_channels; ++ic) {
            // Duyệt qua Kernel 3x3
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    
                    int in_h = h + kh - padding;
                    int in_w = w + kw - padding;

                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                        // Tính index input phẳng
                        int in_idx = b * (in_channels * in_height * in_width) + 
                                     ic * (in_height * in_width) + 
                                     in_h * in_width + in_w;
                        
                        // Tính index weight phẳng: [OC, IC, 3, 3]
                        int w_idx = oc * (in_channels * 9) + 
                                    ic * 9 + 
                                    kh * 3 + kw;
                        
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }

    // --- RELU KERNEL ---
    __global__ void relu_kernel(float* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            if (data[idx] < 0.0f) data[idx] = 0.0f;
        }
    }

    // --- MAX POOL KERNEL ---
    __global__ void max_pool_kernel(const float* input, float* output,
                                    int batch_size, int channels, 
                                    int in_height, int in_width) {
        
        int out_h = in_height / 2;
        int out_w = in_width / 2;
        int total_threads = batch_size * channels * out_h * out_w;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_threads) return;

        // Giải mã index output
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % channels;
        int b = idx / (out_w * out_h * channels);

        float max_val = -1e9;

        // Duyệt vùng 2x2
        for (int ph = 0; ph < 2; ++ph) {
            for (int pw = 0; pw < 2; ++pw) {
                int in_h_idx = h * 2 + ph;
                int in_w_idx = w * 2 + pw;
                
                int in_idx = b * (channels * in_height * in_width) + 
                             c * (in_height * in_width) + 
                             in_h_idx * in_width + in_w_idx;
                
                float val = input[in_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[idx] = max_val;
    }

    // --- UPSAMPLE KERNEL ---
    __global__ void upsample_kernel(const float* input, float* output,
                                    int batch_size, int channels, 
                                    int in_height, int in_width) {
        
        // Thread xử lý Output (lớn)
        int out_h = in_height * 2;
        int out_w = in_width * 2;
        int total_threads = batch_size * channels * out_h * out_w;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_threads) return;

        // Tọa độ trên Output
        int out_x = idx % out_w;
        int out_y = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % channels;
        int b = idx / (out_w * out_h * channels);

        // Map về Input (Nearest Neighbor)
        int in_x = out_x / 2;
        int in_y = out_y / 2;

        int in_idx = b * (channels * in_height * in_width) + 
                     c * (in_height * in_width) + 
                     in_y * in_width + in_x;

        output[idx] = input[in_idx];
    }
}