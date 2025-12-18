#ifndef LAYERS_H
#define LAYERS_H

#include <vector>
#include <cmath>

namespace CPULayers {
    // --- FORWARD (Đã có) ---
    void conv2d(const std::vector<float>& input, std::vector<float>& output,
                const std::vector<float>& weights, const std::vector<float>& bias,
                int batch_size, int in_channels, int out_channels, 
                int in_height, int in_width);

    void relu(std::vector<float>& data);

    void max_pool(const std::vector<float>& input, std::vector<float>& output,
                  int batch_size, int channels, int in_height, int in_width);

    void upsample(const std::vector<float>& input, std::vector<float>& output,
                  int batch_size, int channels, int in_height, int in_width);

    float mse_loss(const std::vector<float>& output, const std::vector<float>& target);

    // --- BACKWARD (MỚI - Cần cho Phase 1.3 & 1.4) ---
    
    // 1. MSE Loss Backward: Tính đạo hàm của Loss theo Output dự đoán
    // d_loss/d_output = 2 * (output - target) / N
    void mse_loss_backward(const std::vector<float>& output, 
                           const std::vector<float>& target, 
                           std::vector<float>& grad_input);

    // 2. ReLU Backward: Nếu x > 0 thì grad = grad_output, ngược lại = 0
    void relu_backward(const std::vector<float>& input, 
                       const std::vector<float>& grad_output, 
                       std::vector<float>& grad_input);

    // 3. Conv2D Backward: Tính gradient cho Input (để truyền về trước) và Weights/Bias (để update)
    void conv2d_backward(const std::vector<float>& input, 
                         const std::vector<float>& grad_output, 
                         const std::vector<float>& weights,
                         std::vector<float>& grad_input, 
                         std::vector<float>& grad_weights, 
                         std::vector<float>& grad_bias,
                         int batch_size, int in_channels, int out_channels, 
                         int in_height, int in_width);

    // 4. MaxPool Backward: Truyền gradient về đúng vị trí max lúc forward
    // Lưu ý: Cần chạy lại logic tìm max để biết index (hoặc lưu mask từ forward - nhưng ở đây ta tính lại cho đơn giản bộ nhớ)
    void max_pool_backward(const std::vector<float>& input, 
                           const std::vector<float>& grad_output, 
                           std::vector<float>& grad_input,
                           int batch_size, int channels, int in_height, int in_width);

    // 5. Upsample Backward: Cộng dồn gradient từ 4 pixel output về 1 pixel input
    void upsample_backward(const std::vector<float>& grad_output, 
                           std::vector<float>& grad_input,
                           int batch_size, int channels, int in_height, int in_width);
}
#endif