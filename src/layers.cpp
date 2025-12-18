#include "../include/layers.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace CPULayers {

    // Helper: Lấy index mảng 1 chiều từ tọa độ NCHW (Batch, Channel, Height, Width)
    inline int get_idx(int b, int c, int h, int w, int channels, int height, int width) {
        return b * (channels * height * width) + c * (height * width) + h * width + w;
    }

    // =========================================================================
    // PHẦN 1: FORWARD PASS (TÍNH XUÔI)
    // =========================================================================

    // 1. Convolution 2D (Padding=1, Stride=1, Kernel 3x3)
    void conv2d(const std::vector<float>& input, std::vector<float>& output,
                const std::vector<float>& weights, const std::vector<float>& bias,
                int batch_size, int in_channels, int out_channels, 
                int in_height, int in_width) {
        
        int out_height = in_height; // Padding=1 giữ nguyên kích thước
        int out_width = in_width;
        int kernel_size = 3;
        int padding = 1;

        // Reset output về 0
        std::fill(output.begin(), output.end(), 0.0f);

        // Nested loops (Rất chậm trên CPU -> Đây là cơ sở để so sánh với GPU)
        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        
                        float sum = bias[oc]; // Khởi tạo bằng bias

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    
                                    int in_h = h + kh - padding;
                                    int in_w = w + kw - padding;

                                    // Kiểm tra biên (Padding zero)
                                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                        int in_idx = get_idx(b, ic, in_h, in_w, in_channels, in_height, in_width);
                                        // Weight layout: [Out_C, In_C, K_H, K_W]
                                        int w_idx = oc * (in_channels * 9) + ic * 9 + kh * 3 + kw;
                                        
                                        sum += input[in_idx] * weights[w_idx];
                                    }
                                }
                            }
                        }
                        
                        int out_idx = get_idx(b, oc, h, w, out_channels, out_height, out_width);
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }

    // 2. ReLU Activation: max(0, x)
    void relu(std::vector<float>& data) {
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] < 0.0f) data[i] = 0.0f;
        }
    }

    // 3. Max Pooling 2x2 (Stride=2)
    void max_pool(const std::vector<float>& input, std::vector<float>& output,
                  int batch_size, int channels, int in_height, int in_width) {
        
        int out_height = in_height / 2;
        int out_width = in_width / 2;

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        
                        float max_val = -1e9;
                        for (int ph = 0; ph < 2; ++ph) {
                            for (int pw = 0; pw < 2; ++pw) {
                                int idx = get_idx(b, c, h * 2 + ph, w * 2 + pw, channels, in_height, in_width);
                                if (input[idx] > max_val) max_val = input[idx];
                            }
                        }
                        
                        output[get_idx(b, c, h, w, channels, out_height, out_width)] = max_val;
                    }
                }
            }
        }
    }

    // 4. Upsampling 2x2 (Nearest Neighbor)
    void upsample(const std::vector<float>& input, std::vector<float>& output,
                  int batch_size, int channels, int in_height, int in_width) {
        
        int out_height = in_height * 2;
        int out_width = in_width * 2;

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        float val = input[get_idx(b, c, h, w, channels, in_height, in_width)];
                        
                        // Copy giá trị ra 4 ô ở output
                        output[get_idx(b, c, h*2,   w*2,   channels, out_height, out_width)] = val;
                        output[get_idx(b, c, h*2,   w*2+1, channels, out_height, out_width)] = val;
                        output[get_idx(b, c, h*2+1, w*2,   channels, out_height, out_width)] = val;
                        output[get_idx(b, c, h*2+1, w*2+1, channels, out_height, out_width)] = val;
                    }
                }
            }
        }
    }

    // 5. MSE Loss Forward
    float mse_loss(const std::vector<float>& output, const std::vector<float>& target) {
        float sum = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            float diff = output[i] - target[i];
            sum += diff * diff;
        }
        return sum / output.size();
    }

    // =========================================================================
    // PHẦN 2: BACKWARD PASS (TÍNH NGƯỢC ĐỂ TRAIN)
    // =========================================================================

    // 1. MSE Backward: dL/dOutput
    void mse_loss_backward(const std::vector<float>& output, const std::vector<float>& target, 
                           std::vector<float>& grad_input) {
        size_t n = output.size();
        grad_input.resize(n);
        float scale = 2.0f / n; // Đạo hàm của MSE = 2/N * (Y_pred - Y_true)
        for (size_t i = 0; i < n; ++i) {
            grad_input[i] = scale * (output[i] - target[i]);
        }
    }

    // 2. ReLU Backward
    void relu_backward(const std::vector<float>& input, const std::vector<float>& grad_output, 
                       std::vector<float>& grad_input) {
        grad_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            // Nếu input > 0 thì gradient truyền qua, ngược lại bằng 0
            grad_input[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
        }
    }

    // 3. Upsample Backward (Cộng dồn Gradient từ lớn về nhỏ)
    void upsample_backward(const std::vector<float>& grad_output, std::vector<float>& grad_input,
                           int batch_size, int channels, int in_height, int in_width) {
        
        int out_h = in_height * 2;
        int out_w = in_width * 2;
        
        // Reset gradient input (kích thước nhỏ) về 0
        size_t input_size = (size_t)batch_size * channels * in_height * in_width;
        grad_input.assign(input_size, 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        // Cộng 4 gradient pixel từ ảnh lớn về 1 pixel ảnh nhỏ
                        float sum_grad = 0.0f;
                        sum_grad += grad_output[get_idx(b, c, h*2,   w*2,   channels, out_h, out_w)];
                        sum_grad += grad_output[get_idx(b, c, h*2,   w*2+1, channels, out_h, out_w)];
                        sum_grad += grad_output[get_idx(b, c, h*2+1, w*2,   channels, out_h, out_w)];
                        sum_grad += grad_output[get_idx(b, c, h*2+1, w*2+1, channels, out_h, out_w)];
                        
                        grad_input[get_idx(b, c, h, w, channels, in_height, in_width)] = sum_grad;
                    }
                }
            }
        }
    }

    // 4. MaxPool Backward (Truyền Gradient về đúng vị trí Max)
    void max_pool_backward(const std::vector<float>& input, const std::vector<float>& grad_output, 
                           std::vector<float>& grad_input,
                           int batch_size, int channels, int in_height, int in_width) {
        
        int out_h = in_height / 2;
        int out_w = in_width / 2;

        // Reset gradient input (kích thước lớn) về 0
        grad_input.assign(input.size(), 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        
                        // Phải tìm lại vị trí max trong ô 2x2
                        int max_idx = -1;
                        float max_val = -1e9;
                        
                        for (int ph = 0; ph < 2; ++ph) {
                            for (int pw = 0; pw < 2; ++pw) {
                                int idx = get_idx(b, c, h*2 + ph, w*2 + pw, channels, in_height, in_width);
                                if (input[idx] > max_val) {
                                    max_val = input[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        
                        // Truyền gradient từ output về đúng index đó
                        if (max_idx != -1) {
                            grad_input[max_idx] += grad_output[get_idx(b, c, h, w, channels, out_h, out_w)];
                        }
                    }
                }
            }
        }
    }

    // 5. Conv2D Backward (Tính Gradient cho Input, Weights và Bias)
    void conv2d_backward(const std::vector<float>& input, const std::vector<float>& grad_output, 
                         const std::vector<float>& weights,
                         std::vector<float>& grad_input, std::vector<float>& grad_weights, std::vector<float>& grad_bias,
                         int batch_size, int in_channels, int out_channels, 
                         int in_height, int in_width) {
        
        int out_height = in_height; 
        int out_width = in_width;
        int kernel_size = 3;
        int padding = 1;

        // Reset Gradients
        grad_input.assign(input.size(), 0.0f);
        grad_weights.assign(weights.size(), 0.0f);
        grad_bias.assign(out_channels, 0.0f);

        for (int b = 0; b < batch_size; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int h = 0; h < out_height; ++h) {
                    for (int w = 0; w < out_width; ++w) {
                        
                        // Gradient từ lớp sau (dL/dY)
                        float d_out = grad_output[get_idx(b, oc, h, w, out_channels, out_height, out_width)];
                        
                        // 1. Tính grad cho Bias: Sum(dL/dY)
                        grad_bias[oc] += d_out;

                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    
                                    int in_h = h + kh - padding;
                                    int in_w = w + kw - padding;

                                    if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                                        int in_idx = get_idx(b, ic, in_h, in_w, in_channels, in_height, in_width);
                                        int w_idx = oc * (in_channels * 9) + ic * 9 + kh * 3 + kw;

                                        // 2. Tính grad cho Weights: dL/dW += Input * dL/dY
                                        grad_weights[w_idx] += input[in_idx] * d_out;

                                        // 3. Tính grad cho Input: dL/dX += Weight * dL/dY
                                        grad_input[in_idx] += weights[w_idx] * d_out;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}