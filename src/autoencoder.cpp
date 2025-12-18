#include "../include/autoencoder.h"
#include "../include/layers.h"
#include <iostream>
#include <random>
#include <fstream>

// Helper init (Giữ nguyên)
void init_param_vec(std::vector<float>& vec, int size, float scale = 0.05f) {
    vec.resize(size);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, scale);
    for(int i=0; i<size; ++i) vec[i] = distribution(generator);
}

Autoencoder::Autoencoder() {
    init_weights();
}

void Autoencoder::init_weights() {
    std::cout << "Initializing Autoencoder Weights..." << std::endl;
    
    // Helper lambda để init cả W, b và resize luôn dW, db
    auto init_layer = [&](std::vector<float>& w, std::vector<float>& b,
                          std::vector<float>& dw, std::vector<float>& db,
                          int count_w, int count_b) {
        init_param_vec(w, count_w);
        init_param_vec(b, count_b, 0.0f);
        // Resize sẵn gradient vectors
        dw.resize(count_w, 0.0f);
        db.resize(count_b, 0.0f);
    };

    // Layer 1
    init_layer(w1, b1, dw1, db1, 3 * 3 * 3 * 256, 256);
    // Layer 2
    init_layer(w2, b2, dw2, db2, 256 * 3 * 3 * 128, 128);
    // Layer 3
    init_layer(w3, b3, dw3, db3, 128 * 3 * 3 * 128, 128);
    // Layer 4
    init_layer(w4, b4, dw4, db4, 128 * 3 * 3 * 256, 256);
    // Layer 5
    init_layer(w5, b5, dw5, db5, 256 * 3 * 3 * 3, 3);
}

void Autoencoder::forward(const std::vector<float>& input, int batch_size) {
    // (Giữ nguyên code forward cũ của bạn)
    // Encoder
    conv1_out.resize(batch_size * 256 * 32 * 32);
    CPULayers::conv2d(input, conv1_out, w1, b1, batch_size, 3, 256, 32, 32);
    CPULayers::relu(conv1_out);

    pool1_out.resize(batch_size * 256 * 16 * 16);
    CPULayers::max_pool(conv1_out, pool1_out, batch_size, 256, 32, 32);

    conv2_out.resize(batch_size * 128 * 16 * 16);
    CPULayers::conv2d(pool1_out, conv2_out, w2, b2, batch_size, 256, 128, 16, 16);
    CPULayers::relu(conv2_out);

    pool2_out.resize(batch_size * 128 * 8 * 8);
    CPULayers::max_pool(conv2_out, pool2_out, batch_size, 128, 16, 16);

    // Decoder
    conv3_out.resize(batch_size * 128 * 8 * 8);
    CPULayers::conv2d(pool2_out, conv3_out, w3, b3, batch_size, 128, 128, 8, 8);
    CPULayers::relu(conv3_out);

    up1_out.resize(batch_size * 128 * 16 * 16);
    CPULayers::upsample(conv3_out, up1_out, batch_size, 128, 8, 8);

    conv4_out.resize(batch_size * 256 * 16 * 16);
    CPULayers::conv2d(up1_out, conv4_out, w4, b4, batch_size, 128, 256, 16, 16);
    CPULayers::relu(conv4_out);

    up2_out.resize(batch_size * 256 * 32 * 32);
    CPULayers::upsample(conv4_out, up2_out, batch_size, 256, 16, 16);

    output.resize(batch_size * 3 * 32 * 32);
    CPULayers::conv2d(up2_out, output, w5, b5, batch_size, 256, 3, 32, 32);
}

// --- MỚI: BACKWARD PASS ---
void Autoencoder::backward(const std::vector<float>& input_batch, int batch_size) {
    // Các biến tạm để lưu gradient giữa các layer
    std::vector<float> d_output, d_up2, d_conv4_relu, d_conv4, d_up1, d_conv3_relu, d_conv3;
    std::vector<float> d_pool2, d_conv2_relu, d_conv2, d_pool1, d_conv1_relu, d_conv1, d_input;

    // 1. Loss Backward: MSE(Output, Input_Batch) -> d_output
    CPULayers::mse_loss_backward(output, input_batch, d_output);

    // 2. Layer 5 (Conv): Output -> Up2_out
    CPULayers::conv2d_backward(up2_out, d_output, w5, d_up2, dw5, db5, batch_size, 256, 3, 32, 32);

    // 3. Layer Up2: Up2_out -> Conv4_out (ReLUed)
    CPULayers::upsample_backward(d_up2, d_conv4_relu, batch_size, 256, 16, 16);

    // 4. Layer 4 (ReLU + Conv): Conv4_out -> Up1_out
    CPULayers::relu_backward(conv4_out, d_conv4_relu, d_conv4); // Back qua ReLU
    CPULayers::conv2d_backward(up1_out, d_conv4, w4, d_up1, dw4, db4, batch_size, 128, 256, 16, 16);

    // 5. Layer Up1: Up1_out -> Conv3_out (ReLUed)
    CPULayers::upsample_backward(d_up1, d_conv3_relu, batch_size, 128, 8, 8);

    // 6. Layer 3 (ReLU + Conv): Conv3_out -> Pool2_out
    CPULayers::relu_backward(conv3_out, d_conv3_relu, d_conv3);
    CPULayers::conv2d_backward(pool2_out, d_conv3, w3, d_pool2, dw3, db3, batch_size, 128, 128, 8, 8);

    // 7. Layer Pool2: Pool2_out -> Conv2_out (ReLUed)
    CPULayers::max_pool_backward(conv2_out, d_pool2, d_conv2_relu, batch_size, 128, 16, 16);

    // 8. Layer 2 (ReLU + Conv): Conv2_out -> Pool1_out
    CPULayers::relu_backward(conv2_out, d_conv2_relu, d_conv2);
    CPULayers::conv2d_backward(pool1_out, d_conv2, w2, d_pool1, dw2, db2, batch_size, 256, 128, 16, 16);

    // 9. Layer Pool1: Pool1_out -> Conv1_out (ReLUed)
    CPULayers::max_pool_backward(conv1_out, d_pool1, d_conv1_relu, batch_size, 256, 32, 32);

    // 10. Layer 1 (ReLU + Conv): Conv1_out -> Input
    CPULayers::relu_backward(conv1_out, d_conv1_relu, d_conv1);
    CPULayers::conv2d_backward(input_batch, d_conv1, w1, d_input, dw1, db1, batch_size, 3, 256, 32, 32);
}

// --- MỚI: UPDATE WEIGHTS ---
void Autoencoder::update(float lr) {
    // Helper lambda để update 1 vector
    auto update_vec = [&](std::vector<float>& w, const std::vector<float>& dw) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] -= lr * dw[i];
        }
    };

    update_vec(w1, dw1); update_vec(b1, db1);
    update_vec(w2, dw2); update_vec(b2, db2);
    update_vec(w3, dw3); update_vec(b3, db3);
    update_vec(w4, dw4); update_vec(b4, db4);
    update_vec(w5, dw5); update_vec(b5, db5);
}

void Autoencoder::save_weights(const std::string& filepath) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        std::cerr << "Cannot save weights to " << filepath << std::endl;
        return;
    }
    // Lưu đơn giản: Kích thước -> Dữ liệu
    auto save_vec = [&](const std::vector<float>& v) {
        size_t size = v.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        out.write(reinterpret_cast<const char*>(v.data()), size * sizeof(float));
    };
    save_vec(w1); save_vec(b1);
    save_vec(w2); save_vec(b2);
    save_vec(w3); save_vec(b3);
    save_vec(w4); save_vec(b4);
    save_vec(w5); save_vec(b5);
    out.close();
    std::cout << "Model saved to " << filepath << std::endl;
}

void Autoencoder::load_weights(const std::string& filepath) {
    // (Optional: Implement nếu cần dùng cho SVM sau này)
}