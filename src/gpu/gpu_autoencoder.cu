#include "../include/gpu_autoencoder.h"
#include <iostream>
#include <cstdio>

// Macro check lỗi CUDA (Cực kỳ quan trọng để debug)
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

GPUAutoencoder::GPUAutoencoder(int b_size) : batch_size(b_size) {
    std::cout << "Allocating GPU Memory (Batch Size: " << batch_size << ")..." << std::endl;

    // --- 1. ALLOCATE WEIGHTS & GRADIENTS ---
    // Helper lambda để đỡ viết lặp lại code
    auto allocate_layer = [&](float** w, float** b, float** dw, float** db, int size_w, int size_b) {
        CHECK(cudaMalloc(w, size_w * sizeof(float)));
        CHECK(cudaMalloc(b, size_b * sizeof(float)));
        CHECK(cudaMalloc(dw, size_w * sizeof(float))); // Cấp phát Gradient
        CHECK(cudaMalloc(db, size_b * sizeof(float))); // Cấp phát Gradient
    };

    // Layer 1: 3 -> 256
    allocate_layer(&d_conv1_w, &d_conv1_b, &d_conv1_dw, &d_conv1_db, 256*3*3*3, 256);
    // Layer 2: 256 -> 128
    allocate_layer(&d_conv2_w, &d_conv2_b, &d_conv2_dw, &d_conv2_db, 128*256*3*3, 128);
    // Layer 3: 128 -> 128
    allocate_layer(&d_conv3_w, &d_conv3_b, &d_conv3_dw, &d_conv3_db, 128*128*3*3, 128);
    // Layer 4: 128 -> 256
    allocate_layer(&d_conv4_w, &d_conv4_b, &d_conv4_dw, &d_conv4_db, 256*128*3*3, 256);
    // Layer 5: 256 -> 3
    allocate_layer(&d_conv5_w, &d_conv5_b, &d_conv5_dw, &d_conv5_db, 3*256*3*3, 3);

    // --- 2. ALLOCATE ACTIVATIONS (BUFFER) ---
    // Tính số lượng phần tử (Elements)
    int size_input = batch_size * 3 * 32 * 32;
    int size_32x32_256 = batch_size * 256 * 32 * 32;
    int size_16x16_256 = batch_size * 256 * 16 * 16;
    int size_16x16_128 = batch_size * 128 * 16 * 16;
    int size_8x8_128   = batch_size * 128 * 8 * 8;
    int size_output    = batch_size * 3 * 32 * 32;

    CHECK(cudaMalloc(&d_input,     size_input * sizeof(float)));
    CHECK(cudaMalloc(&d_conv1_out, size_32x32_256 * sizeof(float)));
    CHECK(cudaMalloc(&d_pool1_out, size_16x16_256 * sizeof(float)));
    CHECK(cudaMalloc(&d_conv2_out, size_16x16_128 * sizeof(float)));
    CHECK(cudaMalloc(&d_encoded,   size_8x8_128 * sizeof(float)));
    CHECK(cudaMalloc(&d_conv3_out, size_8x8_128 * sizeof(float)));
    CHECK(cudaMalloc(&d_up1_out,   size_16x16_128 * sizeof(float)));
    CHECK(cudaMalloc(&d_conv4_out, size_16x16_256 * sizeof(float)));
    CHECK(cudaMalloc(&d_up2_out,   size_32x32_256 * sizeof(float)));
    CHECK(cudaMalloc(&d_output,    size_output * sizeof(float)));

    std::cout << "GPU Memory Allocated Successfully." << std::endl;
}

GPUAutoencoder::~GPUAutoencoder() {
    // Free Weights & Gradients
    cudaFree(d_conv1_w); cudaFree(d_conv1_b); cudaFree(d_conv1_dw); cudaFree(d_conv1_db);
    cudaFree(d_conv2_w); cudaFree(d_conv2_b); cudaFree(d_conv2_dw); cudaFree(d_conv2_db);
    cudaFree(d_conv3_w); cudaFree(d_conv3_b); cudaFree(d_conv3_dw); cudaFree(d_conv3_db);
    cudaFree(d_conv4_w); cudaFree(d_conv4_b); cudaFree(d_conv4_dw); cudaFree(d_conv4_db);
    cudaFree(d_conv5_w); cudaFree(d_conv5_b); cudaFree(d_conv5_dw); cudaFree(d_conv5_db);

    // Free Activations
    cudaFree(d_input);
    cudaFree(d_conv1_out); cudaFree(d_pool1_out);
    cudaFree(d_conv2_out); cudaFree(d_encoded);
    cudaFree(d_conv3_out); cudaFree(d_up1_out);
    cudaFree(d_conv4_out); cudaFree(d_up2_out);
    cudaFree(d_output);

    std::cout << "GPU Memory Released." << std::endl;
}

void GPUAutoencoder::loadWeights(
    const std::vector<float>& h_conv1_w, const std::vector<float>& h_conv1_b,
    const std::vector<float>& h_conv2_w, const std::vector<float>& h_conv2_b,
    const std::vector<float>& h_conv3_w, const std::vector<float>& h_conv3_b,
    const std::vector<float>& h_conv4_w, const std::vector<float>& h_conv4_b,
    const std::vector<float>& h_conv5_w, const std::vector<float>& h_conv5_b
) {
    std::cout << "Copying weights from CPU to GPU..." << std::endl;
    
    auto copy_to_gpu = [](float* d_ptr, const std::vector<float>& h_vec) {
        CHECK(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    copy_to_gpu(d_conv1_w, h_conv1_w); copy_to_gpu(d_conv1_b, h_conv1_b);
    copy_to_gpu(d_conv2_w, h_conv2_w); copy_to_gpu(d_conv2_b, h_conv2_b);
    copy_to_gpu(d_conv3_w, h_conv3_w); copy_to_gpu(d_conv3_b, h_conv3_b);
    copy_to_gpu(d_conv4_w, h_conv4_w); copy_to_gpu(d_conv4_b, h_conv4_b);
    copy_to_gpu(d_conv5_w, h_conv5_w); copy_to_gpu(d_conv5_b, h_conv5_b);

    // Reset Gradients về 0 (Rất quan trọng)
    auto reset_grad = [](float* d_ptr, int size) {
        CHECK(cudaMemset(d_ptr, 0, size * sizeof(float)));
    };
    // (Bạn có thể gọi reset_grad cho các d_convX_dw tại đây nếu muốn chắc chắn)
}