#include "../include/kernels.cuh"
#include <cstdio>

namespace NaiveKernels {

    // =================================================================================
    // PHẦN 1: FORWARD KERNELS
    // =================================================================================

    // 1. CONV2D FORWARD
    // SỬA: Tính index dựa trên kích thước OUTPUT (quan trọng)
    __global__ void conv2d_forward_kernel(const float* input, float* output, const float* weights, const float* bias,
                                          int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                          int out_h, int out_w, int k_size, int padding, int stride) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_elements = batch_size * out_c * out_h * out_w; // Sửa: Dùng kích thước Output

        if (idx >= total_elements) return;

        // Giải mã index: idx -> (b, oc, h, w) của OUTPUT
        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int oc = (idx / (out_w * out_h)) % out_c;
        int b = idx / (out_w * out_h * out_c);

        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    
                    int in_row = h * stride + kh - padding;
                    int in_col = w * stride + kw - padding;

                    if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                         // Index Input: [Batch, In_Channel, Height, Width]
                        int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + in_row * in_w + in_col;
                        
                        // Index Weight: [Out_Channel, In_Channel, K_H, K_W]
                        int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + kh * k_size + kw;
                        
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }

    // 2. RELU FORWARD
    __global__ void relu_forward_kernel(float* data, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = (data[idx] > 0.0f) ? data[idx] : 0.0f;
        }
    }

    // 3. MAX POOL FORWARD
    __global__ void max_pool_forward_kernel(const float* input, float* output,
                                            int batch_size, int c, int in_h, int in_w, int out_h, int out_w) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * c * out_h * out_w;
        
        if (idx >= total_threads) return;

        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % c;
        int b = idx / (out_w * out_h * c);

        float max_val = -1e9;

        for (int ph = 0; ph < 2; ++ph) {
            for (int pw = 0; pw < 2; ++pw) {
                int in_row = h * 2 + ph;
                int in_col = w * 2 + pw;
                
                int in_idx = b * (c * in_h * in_w) + ch * (in_h * in_w) + in_row * in_w + in_col;
                float val = input[in_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[idx] = max_val;
    }

    // 4. UPSAMPLE FORWARD
    __global__ void upsample_forward_kernel(const float* input, float* output,
                                            int batch_size, int c, int in_h, int in_w, int out_h, int out_w) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * c * out_h * out_w; // Kích thước Output lớn

        if (idx >= total_threads) return;

        int out_x = idx % out_w;
        int out_y = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % c;
        int b = idx / (out_w * out_h * c);

        // Nearest Neighbor
        int in_x = out_x / 2;
        int in_y = out_y / 2;

        int in_idx = b * (c * in_h * in_w) + ch * (in_h * in_w) + in_y * in_w + in_x;
        output[idx] = input[in_idx];
    }

    // 5. MSE LOSS FORWARD
    __global__ void mse_loss_kernel(const float* output, const float* target, float* diff_sum, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            float diff = output[idx] - target[idx];
            atomicAdd(diff_sum, diff * diff);
        }
    }

    // =================================================================================
    // PHẦN 2: BACKWARD KERNELS
    // =================================================================================

    // 6. CONV2D BACKWARD INPUT (MỚI: Tách ra để khớp với header)
    __global__ void conv2d_backward_input_kernel(const float* grad_output, const float* weights, float* grad_input,
                                                 int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                                 int out_h, int out_w, int k_size, int padding, int stride) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * in_c * in_h * in_w; // Duyệt theo INPUT

        if (idx >= total_threads) return;

        int w = idx % in_w;
        int h = (idx / in_w) % in_h;
        int ic = (idx / (in_w * in_h)) % in_c;
        int b = idx / (in_w * in_h * in_c);

        float sum = 0.0f;

        // Duyệt qua tất cả output channel và kernel position
        for (int oc = 0; oc < out_c; ++oc) {
            for (int kh = 0; kh < k_size; ++kh) {
                for (int kw = 0; kw < k_size; ++kw) {
                    // Mapping ngược: in = out*stride + k - pad => out = (in + pad - k)/stride
                    int h_diff = h + padding - kh;
                    int w_diff = w + padding - kw;

                    if (h_diff % stride == 0 && w_diff % stride == 0) {
                        int out_r = h_diff / stride;
                        int out_c_idx = w_diff / stride;

                        if (out_r >= 0 && out_r < out_h && out_c_idx >= 0 && out_c_idx < out_w) {
                            int g_out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + out_r * out_w + out_c_idx;
                            int w_idx = oc * (in_c * k_size * k_size) + ic * (k_size * k_size) + kh * k_size + kw;
                            sum += grad_output[g_out_idx] * weights[w_idx];
                        }
                    }
                }
            }
        }
        grad_input[idx] = sum;
    }

    // 7. CONV2D BACKWARD WEIGHTS (MỚI: Tách ra để khớp với header)
    __global__ void conv2d_backward_weight_kernel(const float* input, const float* grad_output, 
                                                  float* grad_weights, float* grad_bias,
                                                  int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                                  int out_h, int out_w, int k_size, int padding, int stride) {
        
        // Mỗi thread tính 1 trọng số (Weight Element)
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_weights = out_c * in_c * k_size * k_size;

        if (idx >= total_weights) return;

        int kw = idx % k_size;
        int kh = (idx / k_size) % k_size;
        int ic = (idx / (k_size * k_size)) % in_c;
        int oc = idx / (k_size * k_size * in_c);

        float w_sum = 0.0f;

        // Duyệt qua toàn bộ batch và không gian output
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < out_h; ++h) {
                for (int w = 0; w < out_w; ++w) {
                    int in_row = h * stride + kh - padding;
                    int in_col = w * stride + kw - padding;

                    if (in_row >= 0 && in_row < in_h && in_col >= 0 && in_col < in_w) {
                        int in_idx = b*(in_c*in_h*in_w) + ic*(in_h*in_w) + in_row*in_w + in_col;
                        int out_idx = b*(out_c*out_h*out_w) + oc*(out_h*out_w) + h*out_w + w;
                        
                        w_sum += input[in_idx] * grad_output[out_idx];
                    }
                }
            }
        }
        grad_weights[idx] = w_sum;

        // Tính Bias (Chỉ cần tính 1 lần cho mỗi Output Channel)
        // Check nếu đây là weight đầu tiên của kernel (để tránh race condition hoặc tính thừa)
        if (ic == 0 && kh == 0 && kw == 0) {
            float b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        int out_idx = b*(out_c*out_h*out_w) + oc*(out_h*out_w) + h*out_w + w;
                        b_sum += grad_output[out_idx];
                    }
                }
            }
            grad_bias[oc] = b_sum;
        }
    }

    // 8. RELU BACKWARD
    __global__ void relu_backward_kernel(const float* input, const float* grad_output, 
                                         float* grad_input, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
        }
    }

    // 9. MAX POOL BACKWARD
    __global__ void max_pool_backward_kernel(const float* input, const float* grad_output, float* grad_input,
                                             int batch_size, int c, int in_h, int in_w, int out_h, int out_w) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * c * out_h * out_w; // Duyệt Output

        if (idx >= total_threads) return;

        int w = idx % out_w;
        int h = (idx / out_w) % out_h;
        int ch = (idx / (out_w * out_h)) % c;
        int b = idx / (out_w * out_h * c);

        int max_idx = -1;
        float max_val = -1e9;

        for (int ph = 0; ph < 2; ++ph) {
            for (int pw = 0; pw < 2; ++pw) {
                int in_row = h * 2 + ph;
                int in_col = w * 2 + pw;
                int in_idx = b*(c*in_h*in_w) + ch*(in_h*in_w) + in_row*in_w + in_col;
                
                if (input[in_idx] > max_val) {
                    max_val = input[in_idx];
                    max_idx = in_idx;
                }
            }
        }
        if (max_idx != -1) {
            atomicAdd(&grad_input[max_idx], grad_output[idx]);
        }
    }

    // 10. UPSAMPLE BACKWARD
    __global__ void upsample_backward_kernel(const float* grad_output, float* grad_input,
                                             int batch_size, int c, int in_h, int in_w, int out_h, int out_w) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_threads = batch_size * c * in_h * in_w; // Duyệt Input (ảnh nhỏ)

        if (idx >= total_threads) return;

        int w = idx % in_w;
        int h = (idx / in_w) % in_h;
        int ch = (idx / (in_w * in_h)) % c;
        int b = idx / (in_w * in_h * c);

        float sum = 0.0f;
        // Cộng 4 pixel tương ứng ở output
        int out_base = b*(c*out_h*out_w) + ch*(out_h*out_w);
        
        sum += grad_output[out_base + (h*2) * out_w + (w*2)];
        sum += grad_output[out_base + (h*2) * out_w + (w*2+1)];
        sum += grad_output[out_base + (h*2+1) * out_w + (w*2)];
        sum += grad_output[out_base + (h*2+1) * out_w + (w*2+1)];

        grad_input[idx] = sum;
    }

    // 11. MSE BACKWARD
    __global__ void mse_backward_kernel(const float* pred, const float* target, float* grad_input, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            grad_input[idx] = (2.0f / size) * (pred[idx] - target[idx]); // Đã thêm hệ số 2/N
        }
    }

    // 12. SGD UPDATE (Thêm vào cho đủ bộ)
    __global__ void sgd_update_kernel(float* weights, const float* grads, float lr, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            weights[idx] -= lr * grads[idx];
        }
    }
}