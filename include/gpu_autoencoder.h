#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_H

#include <vector>
#include <cuda_runtime.h>

class GPUAutoencoder {
public:
    int batch_size;

    // --- 1. WEIGHTS & BIAS POINTERS (DEVICE) ---
    // Encoder
    float *d_conv1_w, *d_conv1_b; 
    float *d_conv2_w, *d_conv2_b; 
    // Decoder
    float *d_conv3_w, *d_conv3_b; 
    float *d_conv4_w, *d_conv4_b; 
    float *d_conv5_w, *d_conv5_b; 

    // --- 2. GRADIENTS POINTERS (MỚI - BẮT BUỘC CÓ) ---
    // Để lưu dW, db tính được từ Backward pass
    float *d_conv1_dw, *d_conv1_db;
    float *d_conv2_dw, *d_conv2_db;
    float *d_conv3_dw, *d_conv3_db;
    float *d_conv4_dw, *d_conv4_db;
    float *d_conv5_dw, *d_conv5_db;

    // --- 3. ACTIVATION BUFFERS (DEVICE) ---
    float *d_input;       // 32x32x3
    
    float *d_conv1_out;   // 32x32x256
    float *d_pool1_out;   // 16x16x256
    float *d_conv2_out;   // 16x16x128
    float *d_encoded;     // 8x8x128 (Latent Space)

    float *d_conv3_out;   // 8x8x128
    float *d_up1_out;     // 16x16x128
    float *d_conv4_out;   // 16x16x256
    float *d_up2_out;     // 32x32x256
    float *d_output;      // 32x32x3

    // --- METHODS ---
    GPUAutoencoder(int batch_size);
    ~GPUAutoencoder();

    // Copy weights từ CPU xuống GPU
    void loadWeights(
        const std::vector<float>& h_conv1_w, const std::vector<float>& h_conv1_b,
        const std::vector<float>& h_conv2_w, const std::vector<float>& h_conv2_b,
        const std::vector<float>& h_conv3_w, const std::vector<float>& h_conv3_b,
        const std::vector<float>& h_conv4_w, const std::vector<float>& h_conv4_b,
        const std::vector<float>& h_conv5_w, const std::vector<float>& h_conv5_b
    );

    // --- CÁC HÀM XỬ LÝ CHÍNH (QUAN TRỌNG - ĐÃ BỔ SUNG) ---
    
    // 1. Forward Pass: Tính toán từ Input -> Output
    void forward(float* d_batch_data);

    // 2. Compute Loss: Tính MSE Loss trên GPU và trả về CPU
    float compute_loss(float* d_target);

    // 3. Backward Pass: Tính Gradients
    void backward(float* d_target);

    // 4. Update Weights: W = W - lr * Grad
    void update(float learning_rate);
};

#endif // GPU_AUTOENCODER_H