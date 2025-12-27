#include "../../include/gpu_autoencoder.h"
#include "../../include/kernels.cuh"
#include <iostream>
#include <cstdio>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

dim3 get_grid_size(int total_threads, int block_size = 256) {
    return dim3((total_threads + block_size - 1) / block_size);
}

GPUAutoencoder::GPUAutoencoder(int b_size) : batch_size(b_size) {
    std::cout << "Allocating GPU Memory (Batch Size: " << batch_size << ")..." << std::endl;

    // 1. ALLOCATE WEIGHTS & GRADIENTS
    auto allocate_layer = [&](float** w, float** b, float** dw, float** db, int size_w, int size_b) {
        CHECK(cudaMalloc(w, size_w * sizeof(float)));
        CHECK(cudaMalloc(b, size_b * sizeof(float)));
        CHECK(cudaMalloc(dw, size_w * sizeof(float)));
        CHECK(cudaMalloc(db, size_b * sizeof(float)));
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

    // 2. ALLOCATE ACTIVATIONS
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
    cudaFree(d_conv1_w); cudaFree(d_conv1_b); cudaFree(d_conv1_dw); cudaFree(d_conv1_db);
    cudaFree(d_conv2_w); cudaFree(d_conv2_b); cudaFree(d_conv2_dw); cudaFree(d_conv2_db);
    cudaFree(d_conv3_w); cudaFree(d_conv3_b); cudaFree(d_conv3_dw); cudaFree(d_conv3_db);
    cudaFree(d_conv4_w); cudaFree(d_conv4_b); cudaFree(d_conv4_dw); cudaFree(d_conv4_db);
    cudaFree(d_conv5_w); cudaFree(d_conv5_b); cudaFree(d_conv5_dw); cudaFree(d_conv5_db);

    cudaFree(d_input);
    cudaFree(d_conv1_out); cudaFree(d_pool1_out);
    cudaFree(d_conv2_out); cudaFree(d_encoded);
    cudaFree(d_conv3_out); cudaFree(d_up1_out);
    cudaFree(d_conv4_out); cudaFree(d_up2_out);
    cudaFree(d_output);
}

void GPUAutoencoder::loadWeights(
    const std::vector<float>& h_conv1_w, const std::vector<float>& h_conv1_b,
    const std::vector<float>& h_conv2_w, const std::vector<float>& h_conv2_b,
    const std::vector<float>& h_conv3_w, const std::vector<float>& h_conv3_b,
    const std::vector<float>& h_conv4_w, const std::vector<float>& h_conv4_b,
    const std::vector<float>& h_conv5_w, const std::vector<float>& h_conv5_b
) {
    auto copy_to_gpu = [](float* d_ptr, const std::vector<float>& h_vec) {
        CHECK(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    copy_to_gpu(d_conv1_w, h_conv1_w); copy_to_gpu(d_conv1_b, h_conv1_b);
    copy_to_gpu(d_conv2_w, h_conv2_w); copy_to_gpu(d_conv2_b, h_conv2_b);
    copy_to_gpu(d_conv3_w, h_conv3_w); copy_to_gpu(d_conv3_b, h_conv3_b);
    copy_to_gpu(d_conv4_w, h_conv4_w); copy_to_gpu(d_conv4_b, h_conv4_b);
    copy_to_gpu(d_conv5_w, h_conv5_w); copy_to_gpu(d_conv5_b, h_conv5_b);
}

// 1. FORWARD PASS
void GPUAutoencoder::forward(float* d_batch_data) {
    // Copy input batch vào buffer d_input
    int input_size = batch_size * 3 * 32 * 32;
    CHECK(cudaMemcpy(d_input, d_batch_data, input_size * sizeof(float), cudaMemcpyDeviceToDevice));

    int block_size = 256;

    // 1. Conv1: 3 -> 256
    int size_c1 = batch_size * 256 * 32 * 32;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_c1), block_size>>>(
        d_input, d_conv1_out, d_conv1_w, d_conv1_b, 
        batch_size, 3, 256, 32, 32, 32, 32, 3, 1, 1);
    
    // ReLU & Pool1
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c1), block_size>>>(d_conv1_out, size_c1);
    int size_p1 = batch_size * 256 * 16 * 16;
    NaiveKernels::max_pool_forward_kernel<<<get_grid_size(size_p1), block_size>>>(
        d_conv1_out, d_pool1_out, batch_size, 256, 32, 32, 16, 16);

    // 2. Conv2: 256 -> 128
    int size_c2 = batch_size * 128 * 16 * 16;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_c2), block_size>>>(
        d_pool1_out, d_conv2_out, d_conv2_w, d_conv2_b,
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1);
    
    // ReLU & Pool2
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c2), block_size>>>(d_conv2_out, size_c2);
    int size_p2 = batch_size * 128 * 8 * 8;
    NaiveKernels::max_pool_forward_kernel<<<get_grid_size(size_p2), block_size>>>(
        d_conv2_out, d_encoded, batch_size, 128, 16, 16, 8, 8);

    // Decoder
    // 3. Conv3: 128 -> 128
    int size_c3 = batch_size * 128 * 8 * 8;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_c3), block_size>>>(
        d_encoded, d_conv3_out, d_conv3_w, d_conv3_b,
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);
    
    // ReLU & Upsample1
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c3), block_size>>>(d_conv3_out, size_c3);
    int size_up1 = batch_size * 128 * 16 * 16;
    NaiveKernels::upsample_forward_kernel<<<get_grid_size(size_up1), block_size>>>(
        d_conv3_out, d_up1_out, batch_size, 128, 8, 8, 16, 16);

    // 4. Conv4: 128 -> 256
    int size_c4 = batch_size * 256 * 16 * 16;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_c4), block_size>>>(
        d_up1_out, d_conv4_out, d_conv4_w, d_conv4_b,
        batch_size, 128, 256, 16, 16, 16, 16, 3, 1, 1);
    
    // ReLU & Upsample2
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c4), block_size>>>(d_conv4_out, size_c4);
    int size_up2 = batch_size * 256 * 32 * 32;
    NaiveKernels::upsample_forward_kernel<<<get_grid_size(size_up2), block_size>>>(
        d_conv4_out, d_up2_out, batch_size, 256, 16, 16, 32, 32);

    // 5. Conv5 (Output): 256 -> 3
    int size_out = batch_size * 3 * 32 * 32;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_out), block_size>>>(
        d_up2_out, d_output, d_conv5_w, d_conv5_b,
        batch_size, 256, 3, 32, 32, 32, 32, 3, 1, 1);

    CHECK(cudaDeviceSynchronize());
}

void GPUAutoencoder::forward_phase3_ver1() {
    int block_size = 256;

    dim3 block_shared(16, 16); 
        
    // LAYER 1: Conv1 (32x32)
    // Grid: (W/16, H/16, Batch * Out_Channel)
    dim3 grid_c1((32 + 15)/16, (32 + 15)/16, batch_size * 256);
        
    // Gọi Shared Mem Kernel
    Phase3Kernels::conv2d_shared_mem_kernel<<<grid_c1, block_shared, 0, stream_compute>>>(
        d_input, d_conv1_out, d_conv1_w, d_conv1_b,
        batch_size, 3, 256, 32, 32, 32, 32 // Input: 32x32, Output: 32x32
    );
        
    int size_c1 = batch_size * 256 * 32 * 32;
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c1), block_size, 0, stream_compute>>>(d_conv1_out, size_c1);
        
    int size_p1 = batch_size * 256 * 16 * 16;
    NaiveKernels::max_pool_forward_kernel<<<get_grid_size(size_p1), block_size, 0, stream_compute>>>(
        d_conv1_out, d_pool1_out, batch_size, 256, 32, 32, 16, 16);

    // LAYER 2: Conv2 (16x16)
    dim3 grid_c2((16 + 15)/16, (16 + 15)/16, batch_size * 128);
        
    Phase3Kernels::conv2d_shared_mem_kernel<<<grid_c2, block_shared, 0, stream_compute>>>(
        d_pool1_out, d_conv2_out, d_conv2_w, d_conv2_b,
        batch_size, 256, 128, 16, 16, 16, 16
    );

    int size_c2 = batch_size * 128 * 16 * 16;
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c2), block_size, 0, stream_compute>>>(d_conv2_out, size_c2);
        
    int size_p2 = batch_size * 128 * 8 * 8;
    NaiveKernels::max_pool_forward_kernel<<<get_grid_size(size_p2), block_size, 0, stream_compute>>>(
        d_conv2_out, d_encoded, batch_size, 128, 16, 16, 8, 8);

    // DECODER (Conv3, Conv4, Conv5)
    int size_c3 = batch_size * 128 * 8 * 8;
    NaiveKernels::conv2d_forward_kernel<<<get_grid_size(size_c3), block_size, 0, stream_compute>>>(
        d_encoded, d_conv3_out, d_conv3_w, d_conv3_b,
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c3), block_size, 0, stream_compute>>>(d_conv3_out, size_c3);
    
    // Upsample 1
    int size_up1 = batch_size * 128 * 16 * 16;
    NaiveKernels::upsample_forward_kernel<<<get_grid_size(size_up1), block_size, 0, stream_compute>>>(
        d_conv3_out, d_up1_out, batch_size, 128, 8, 8, 16, 16);

    // Conv4 (16x16)
    dim3 grid_c4((16 + 15)/16, (16 + 15)/16, batch_size * 256);
    Phase3Kernels::conv2d_shared_mem_kernel<<<grid_c4, block_shared, 0, stream_compute>>>(
        d_up1_out, d_conv4_out, d_conv4_w, d_conv4_b,
        batch_size, 128, 256, 16, 16, 16, 16
    );
    int size_c4 = batch_size * 256 * 16 * 16;
    NaiveKernels::relu_forward_kernel<<<get_grid_size(size_c4), block_size, 0, stream_compute>>>(d_conv4_out, size_c4);
        
    // Upsample 2
    int size_up2 = batch_size * 256 * 32 * 32;
    NaiveKernels::upsample_forward_kernel<<<get_grid_size(size_up2), block_size, 0, stream_compute>>>(
        d_conv4_out, d_up2_out, batch_size, 256, 16, 16, 32, 32);

    // Conv5 (Output 32x32) -> Dùng Shared Mem được
    dim3 grid_c5((32 + 15)/16, (32 + 15)/16, batch_size * 3);
    Phase3Kernels::conv2d_shared_mem_kernel<<<grid_c5, block_shared, 0, stream_compute>>>(
        d_up2_out, d_output, d_conv5_w, d_conv5_b,
        batch_size, 256, 3, 32, 32, 32, 32
    );

    
    CHECK(cudaStreamSynchronize(stream_compute));
}

// 2. COMPUTE LOSS
float GPUAutoencoder::compute_loss(float* d_target) {
    float* d_loss_sum;
    CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));
    CHECK(cudaMemset(d_loss_sum, 0, sizeof(float)));

    int size_out = batch_size * 3 * 32 * 32;
    NaiveKernels::mse_loss_kernel<<<get_grid_size(size_out), 256>>>(d_output, d_target, d_loss_sum, size_out);

    float total_loss = 0.0f;
    CHECK(cudaMemcpy(&total_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_loss_sum));

    return total_loss / size_out;
}

// 3. BACKWARD PASS
void GPUAutoencoder::backward(float* d_target) {
    int size_32x32 = batch_size * 3 * 32 * 32;
    int block = 256;
    float *d_grad_buffer; 
    CHECK(cudaMalloc(&d_grad_buffer, batch_size * 256 * 32 * 32 * sizeof(float)));

    // 1. MSE Backward
    NaiveKernels::mse_backward_kernel<<<get_grid_size(size_32x32), block>>>(
        d_output, d_target, d_grad_buffer, size_32x32);

    // 2. Layer 5 Backward
    CHECK(cudaMemset(d_conv5_dw, 0, 3 * 256 * 9 * sizeof(float)));
    CHECK(cudaMemset(d_conv5_db, 0, 3 * sizeof(float)));

    int n_w5 = 3 * 256 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w5), block>>>(
        d_up2_out, d_grad_buffer, d_conv5_dw, d_conv5_db,
        batch_size, 256, 3, 32, 32, 32, 32, 3, 1, 1);

    float *d_grad_next;
    CHECK(cudaMalloc(&d_grad_next, batch_size * 256 * 32 * 32 * sizeof(float)));
    
    int size_up2 = batch_size * 256 * 32 * 32;
    NaiveKernels::conv2d_backward_input_kernel<<<get_grid_size(size_up2), block>>>(
        d_grad_buffer, d_conv5_w, d_grad_next,
        batch_size, 256, 3, 32, 32, 32, 32, 3, 1, 1);
    
    // 3. Up2 Backward
    int size_conv4 = batch_size * 256 * 16 * 16;
    NaiveKernels::upsample_backward_kernel<<<get_grid_size(size_conv4), block>>>(
        d_grad_next, d_grad_buffer, batch_size, 256, 16, 16, 32, 32);

    // 4. Layer 4 Backward
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv4), block>>>(
        d_conv4_out, d_grad_buffer, d_grad_buffer, size_conv4);

    CHECK(cudaMemset(d_conv4_dw, 0, 256 * 128 * 9 * sizeof(float)));
    CHECK(cudaMemset(d_conv4_db, 0, 256 * sizeof(float)));
    int n_w4 = 256 * 128 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w4), block>>>(
        d_up1_out, d_grad_buffer, d_conv4_dw, d_conv4_db,
        batch_size, 128, 256, 16, 16, 16, 16, 3, 1, 1);

    int size_up1 = batch_size * 128 * 16 * 16;
    NaiveKernels::conv2d_backward_input_kernel<<<get_grid_size(size_up1), block>>>(
        d_grad_buffer, d_conv4_w, d_grad_next,
        batch_size, 128, 256, 16, 16, 16, 16, 3, 1, 1);

    // 5. Up1 Backward
    int size_conv3 = batch_size * 128 * 8 * 8;
    NaiveKernels::upsample_backward_kernel<<<get_grid_size(size_conv3), block>>>(
        d_grad_next, d_grad_buffer, batch_size, 128, 8, 8, 16, 16);

    // 6. Layer 3 Backward
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv3), block>>>(
        d_conv3_out, d_grad_buffer, d_grad_buffer, size_conv3);

    CHECK(cudaMemset(d_conv3_dw, 0, 128 * 128 * 9 * sizeof(float)));
    CHECK(cudaMemset(d_conv3_db, 0, 128 * sizeof(float)));
    int n_w3 = 128 * 128 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w3), block>>>(
        d_encoded, d_grad_buffer, d_conv3_dw, d_conv3_db,
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);

    int size_enc = batch_size * 128 * 8 * 8;
    NaiveKernels::conv2d_backward_input_kernel<<<get_grid_size(size_enc), block>>>(
        d_grad_buffer, d_conv3_w, d_grad_next, 
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);

    // 7. Pool2 Backward
    int size_conv2 = batch_size * 128 * 16 * 16;
    CHECK(cudaMemset(d_grad_buffer, 0, size_conv2 * sizeof(float)));
    NaiveKernels::max_pool_backward_kernel<<<get_grid_size(size_enc), block>>>(
        d_conv2_out, d_grad_next, d_grad_buffer,
        batch_size, 128, 16, 16, 8, 8);

    // 8. Layer 2 Backward
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv2), block>>>(
        d_conv2_out, d_grad_buffer, d_grad_buffer, size_conv2);
    
    CHECK(cudaMemset(d_conv2_dw, 0, 128 * 256 * 9 * sizeof(float)));
    CHECK(cudaMemset(d_conv2_db, 0, 128 * sizeof(float)));
    int n_w2 = 128 * 256 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w2), block>>>(
        d_pool1_out, d_grad_buffer, d_conv2_dw, d_conv2_db,
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1);

    int size_pool1 = batch_size * 256 * 16 * 16;
    NaiveKernels::conv2d_backward_input_kernel<<<get_grid_size(size_pool1), block>>>(
        d_grad_buffer, d_conv2_w, d_grad_next,
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1);

    // 9. Pool1 Backward
    int size_conv1 = batch_size * 256 * 32 * 32;
    CHECK(cudaMemset(d_grad_buffer, 0, size_conv1 * sizeof(float))); 
    NaiveKernels::max_pool_backward_kernel<<<get_grid_size(size_pool1), block>>>(
        d_conv1_out, d_grad_next, d_grad_buffer,
        batch_size, 256, 32, 32, 16, 16);

    // 10. Layer 1 Backward
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv1), block>>>(
        d_conv1_out, d_grad_buffer, d_grad_buffer, size_conv1);

    CHECK(cudaMemset(d_conv1_dw, 0, 256 * 3 * 9 * sizeof(float)));
    CHECK(cudaMemset(d_conv1_db, 0, 256 * sizeof(float)));
    int n_w1 = 256 * 3 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w1), block>>>(
        d_input, d_grad_buffer, d_conv1_dw, d_conv1_db,
        batch_size, 3, 256, 32, 32, 32, 32, 3, 1, 1);

    CHECK(cudaFree(d_grad_buffer));
    CHECK(cudaFree(d_grad_next));
    CHECK(cudaDeviceSynchronize());
}

void GPUAutoencoder::backward_phase3_ver1(float* d_target) {
    // Config chung
    int size_32x32 = batch_size * 3 * 32 * 32;
    int block = 256;
    dim3 block_shared(16, 16);

    float *d_grad_buffer; 
    CHECK(cudaMalloc(&d_grad_buffer, batch_size * 256 * 32 * 32 * sizeof(float)));

    // 1. MSE Backward (Loss Gradient)
    NaiveKernels::mse_backward_kernel<<<get_grid_size(size_32x32), block, 0, stream_compute>>>(
        d_output, d_target, d_grad_buffer, size_32x32);

    // LAYER 5 BACKWARD (Output -> Up2)
    CHECK(cudaMemsetAsync(d_conv5_dw, 0, 3 * 256 * 9 * sizeof(float), stream_compute));
    CHECK(cudaMemsetAsync(d_conv5_db, 0, 3 * sizeof(float), stream_compute));

    // Weight Gradient
    int n_w5 = 3 * 256 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w5), block, 0, stream_compute>>>(
        d_up2_out, d_grad_buffer, d_conv5_dw, d_conv5_db,
        batch_size, 256, 3, 32, 32, 32, 32, 3, 1, 1);

    float *d_grad_next;
    CHECK(cudaMalloc(&d_grad_next, batch_size * 256 * 32 * 32 * sizeof(float)));
    
    //  Input Gradient (Layer 5)
    
    // Phase 3 mới: Dùng Shared Memory
    // Grid tính theo Input Size (32x32) và Channel Input (256)
    dim3 grid_b5((32 + 15)/16, (32 + 15)/16, batch_size * 256);
    Phase3Kernels::conv2d_backward_input_shared_mem_kernel<<<grid_b5, block_shared, 0, stream_compute>>>(
        d_grad_buffer, d_conv5_w, d_grad_next,
        batch_size, 256, 3, 32, 32, 32, 32 
    );

    // LAYER 4 BACKWARD (Up2 -> Conv4 -> Up1)
    // 1. Upsample Backward (Up2)
    int size_up2 = batch_size * 256 * 32 * 32;
    int size_conv4 = batch_size * 256 * 16 * 16;
    
    float *d_grad_conv4;
    CHECK(cudaMalloc(&d_grad_conv4, size_conv4 * sizeof(float)));

    NaiveKernels::upsample_backward_kernel<<<get_grid_size(size_conv4), block, 0, stream_compute>>>(
        d_grad_next, d_grad_conv4, batch_size, 256, 16, 16, 32, 32);

    // 2. ReLU Backward
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv4), block, 0, stream_compute>>>(
        d_conv4_out, d_grad_conv4, d_grad_conv4, size_conv4);

    // 3. Conv4 Gradients
    CHECK(cudaMemsetAsync(d_conv4_dw, 0, 256 * 128 * 9 * sizeof(float), stream_compute));
    CHECK(cudaMemsetAsync(d_conv4_db, 0, 256 * sizeof(float), stream_compute));
    int n_w4 = 256 * 128 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w4), block, 0, stream_compute>>>(
        d_up1_out, d_grad_conv4, d_conv4_dw, d_conv4_db,
        batch_size, 128, 256, 16, 16, 16, 16, 3, 1, 1);

    float *d_grad_up1;
    CHECK(cudaMalloc(&d_grad_up1, batch_size * 128 * 16 * 16 * sizeof(float)));

    // OPTIMIZATION: Input Gradient (Layer 4)
    // Grid: 16x16, Channel 128
    dim3 grid_b4((16 + 15)/16, (16 + 15)/16, batch_size * 128);
    Phase3Kernels::conv2d_backward_input_shared_mem_kernel<<<grid_b4, block_shared, 0, stream_compute>>>(
        d_grad_conv4, d_conv4_w, d_grad_up1,
        batch_size, 128, 256, 16, 16, 16, 16 
    );

    // LAYER 3 (Up1 -> Conv3 -> Encoded)
    int size_conv3 = batch_size * 128 * 8 * 8;
    float *d_grad_conv3;
    CHECK(cudaMalloc(&d_grad_conv3, size_conv3 * sizeof(float)));

    NaiveKernels::upsample_backward_kernel<<<get_grid_size(size_conv3), block, 0, stream_compute>>>(
        d_grad_up1, d_grad_conv3, batch_size, 128, 8, 8, 16, 16);
    
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv3), block, 0, stream_compute>>>(
        d_conv3_out, d_grad_conv3, d_grad_conv3, size_conv3);

    CHECK(cudaMemsetAsync(d_conv3_dw, 0, 128 * 128 * 9 * sizeof(float), stream_compute));
    CHECK(cudaMemsetAsync(d_conv3_db, 0, 128 * sizeof(float), stream_compute));
    int n_w3 = 128 * 128 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w3), block, 0, stream_compute>>>(
        d_encoded, d_grad_conv3, d_conv3_dw, d_conv3_db,
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);

    float *d_grad_encoded;
    CHECK(cudaMalloc(&d_grad_encoded, size_conv3 * sizeof(float)));

    // Layer 3 nhỏ (8x8)
    NaiveKernels::conv2d_backward_input_kernel<<<get_grid_size(size_conv3), block, 0, stream_compute>>>(
        d_grad_conv3, d_conv3_w, d_grad_encoded,
        batch_size, 128, 128, 8, 8, 8, 8, 3, 1, 1);

    // LAYER 2 (Encoded -> Pool2 -> Conv2 -> Pool1)
    int size_conv2 = batch_size * 128 * 16 * 16;
    float *d_grad_conv2;
    CHECK(cudaMalloc(&d_grad_conv2, size_conv2 * sizeof(float)));
    CHECK(cudaMemsetAsync(d_grad_conv2, 0, size_conv2 * sizeof(float), stream_compute));

    NaiveKernels::max_pool_backward_kernel<<<get_grid_size(size_conv3), block, 0, stream_compute>>>(
        d_conv2_out, d_grad_encoded, d_grad_conv2,
        batch_size, 128, 16, 16, 8, 8); // Pool: Output 8x8 -> Input 16x16

    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv2), block, 0, stream_compute>>>(
        d_conv2_out, d_grad_conv2, d_grad_conv2, size_conv2);

    CHECK(cudaMemsetAsync(d_conv2_dw, 0, 128 * 256 * 9 * sizeof(float), stream_compute));
    CHECK(cudaMemsetAsync(d_conv2_db, 0, 128 * sizeof(float), stream_compute));
    int n_w2 = 128 * 256 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w2), block, 0, stream_compute>>>(
        d_pool1_out, d_grad_conv2, d_conv2_dw, d_conv2_db,
        batch_size, 256, 128, 16, 16, 16, 16, 3, 1, 1);

    float *d_grad_pool1;
    CHECK(cudaMalloc(&d_grad_pool1, batch_size * 256 * 16 * 16 * sizeof(float)));

    // OPTIMIZATION: Input Gradient (Layer 2)
    // Grid 16x16, Channel 256
    dim3 grid_b2((16 + 15)/16, (16 + 15)/16, batch_size * 256);
    Phase3Kernels::conv2d_backward_input_shared_mem_kernel<<<grid_b2, block_shared, 0, stream_compute>>>(
        d_grad_conv2, d_conv2_w, d_grad_pool1,
        batch_size, 256, 128, 16, 16, 16, 16
    );

    // LAYER 1 (Pool1 -> Conv1 -> Input)
    int size_conv1 = batch_size * 256 * 32 * 32;
    float *d_grad_conv1;
    CHECK(cudaMalloc(&d_grad_conv1, size_conv1 * sizeof(float)));
    CHECK(cudaMemsetAsync(d_grad_conv1, 0, size_conv1 * sizeof(float), stream_compute));

    NaiveKernels::max_pool_backward_kernel<<<get_grid_size(batch_size * 256 * 16 * 16), block, 0, stream_compute>>>(
        d_conv1_out, d_grad_pool1, d_grad_conv1,
        batch_size, 256, 32, 32, 16, 16);
    
    NaiveKernels::relu_backward_kernel<<<get_grid_size(size_conv1), block, 0, stream_compute>>>(
        d_conv1_out, d_grad_conv1, d_grad_conv1, size_conv1);

    CHECK(cudaMemsetAsync(d_conv1_dw, 0, 256 * 3 * 9 * sizeof(float), stream_compute));
    CHECK(cudaMemsetAsync(d_conv1_db, 0, 256 * sizeof(float), stream_compute));
    int n_w1 = 256 * 3 * 9;
    NaiveKernels::conv2d_backward_weight_kernel<<<get_grid_size(n_w1), block, 0, stream_compute>>>(
        d_input, d_grad_conv1, d_conv1_dw, d_conv1_db,
        batch_size, 3, 256, 32, 32, 32, 32, 3, 1, 1);

    CHECK(cudaFree(d_grad_buffer));
    CHECK(cudaFree(d_grad_next));
    CHECK(cudaFree(d_grad_conv4));
    CHECK(cudaFree(d_grad_up1));
    CHECK(cudaFree(d_grad_conv3));
    CHECK(cudaFree(d_grad_encoded));
    CHECK(cudaFree(d_grad_conv2));
    CHECK(cudaFree(d_grad_pool1));
    CHECK(cudaFree(d_grad_conv1));

    CHECK(cudaStreamSynchronize(stream_compute));
}

// 4. UPDATE
void GPUAutoencoder::update(float lr) {
    int block = 256;
    auto update_w = [&](float* w, float* dw, int size) {
        NaiveKernels::sgd_update_kernel<<<get_grid_size(size), block>>>(w, dw, lr, size);
    };

    update_w(d_conv1_w, d_conv1_dw, 256 * 3 * 9); update_w(d_conv1_b, d_conv1_db, 256);
    update_w(d_conv2_w, d_conv2_dw, 128 * 256 * 9); update_w(d_conv2_b, d_conv2_db, 128);
    update_w(d_conv3_w, d_conv3_dw, 128 * 128 * 9); update_w(d_conv3_b, d_conv3_db, 128);
    update_w(d_conv4_w, d_conv4_dw, 256 * 128 * 9); update_w(d_conv4_b, d_conv4_db, 256);
    update_w(d_conv5_w, d_conv5_dw, 3 * 256 * 9); update_w(d_conv5_b, d_conv5_db, 3);
    
    CHECK(cudaDeviceSynchronize());
}