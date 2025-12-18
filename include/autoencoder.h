#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <vector>
#include <string>

class Autoencoder {
public:
    int input_h = 32;
    int input_w = 32;
    int input_c = 3;

    // --- Weights & Biases ---
    std::vector<float> w1, b1; // Encoder
    std::vector<float> w2, b2;
    std::vector<float> w3, b3; // Decoder
    std::vector<float> w4, b4;
    std::vector<float> w5, b5;

    // --- Gradients (MỚI: Để lưu đạo hàm) ---
    std::vector<float> dw1, db1;
    std::vector<float> dw2, db2;
    std::vector<float> dw3, db3;
    std::vector<float> dw4, db4;
    std::vector<float> dw5, db5;

    // --- Intermediate Activations (Đã có) ---
    std::vector<float> conv1_out, pool1_out;
    std::vector<float> conv2_out, pool2_out; 
    std::vector<float> conv3_out, up1_out;
    std::vector<float> conv4_out, up2_out;
    std::vector<float> output;

    // Constructor & Init
    Autoencoder();
    void init_weights();

    // --- Core Functions ---
    // 1. Forward (Đã có)
    void forward(const std::vector<float>& input_batch, int batch_size);
    
    // 2. Backward (MỚI): Tính toán Gradients
    void backward(const std::vector<float>& input_batch, int batch_size);
    
    // 3. Update (MỚI): Cập nhật trọng số: W = W - learning_rate * dW
    void update(float learning_rate);

    // Save/Load
    void save_weights(const std::string& filepath);
    void load_weights(const std::string& filepath);
};

#endif