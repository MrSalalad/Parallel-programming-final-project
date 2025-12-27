#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>
#include <cstring> // Cho memcpy

#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"     
#include "../include/gpu_autoencoder.h" 
#include <fstream> 

// Hàm phụ trợ: Copy từ GPU về CPU rồi ghi ra file
void save_gpu_layer(std::ofstream& file, float* d_data, int size) {
    std::vector<float> h_data(size);
    // Dùng copy đồng bộ ở đây vì ta cần dữ liệu ngay để lưu
    cudaMemcpy(h_data.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(h_data.data()), size * sizeof(float));
}

void save_gpu_model(GPUAutoencoder& model, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file for save: " << filename << std::endl;
        return;
    }
    
    std::cout << "[SAVE] Saving model to " << filename << "..." << std::endl;

    // Thứ tự lưu phải khớp y hệt lúc đọc (w1, b1, w2, b2...)
    save_gpu_layer(file, model.d_conv1_w, 256*3*3*3);
    save_gpu_layer(file, model.d_conv1_b, 256);
    
    save_gpu_layer(file, model.d_conv2_w, 128*256*3*3);
    save_gpu_layer(file, model.d_conv2_b, 128);
    
    save_gpu_layer(file, model.d_conv3_w, 128*128*3*3);
    save_gpu_layer(file, model.d_conv3_b, 128);
    
    save_gpu_layer(file, model.d_conv4_w, 256*128*3*3);
    save_gpu_layer(file, model.d_conv4_b, 256);
    
    save_gpu_layer(file, model.d_conv5_w, 3*256*3*3);
    save_gpu_layer(file, model.d_conv5_b, 3);
    
    std::cout << "[SAVE] Done." << std::endl;
    file.close();
}

// Hàm đo VRAM
void print_gpu_memory_usage() {
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte / (1024.0 * 1024.0);
    double total_db = (double)total_byte / (1024.0 * 1024.0);
    double used_db = total_db - free_db;
    
    std::cout << "   - VRAM Used : " << std::fixed << std::setprecision(2) << used_db << " MB" << std::endl;
    std::cout << "   - VRAM Total: " << total_db << " MB" << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "    PHASE 3: OPTIMIZED GPU TRAINING (Version 1)   " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Setup Data
    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // 2. Config
    int batch_size = 64;       
    int target_epochs = 20;    
    float learning_rate = 0.001f;
    size_t input_size_bytes = batch_size * 3 * 32 * 32 * sizeof(float);

    std::cout << "[CONFIG] Target Epochs: " << target_epochs << std::endl;
    std::cout << "[CONFIG] Batch Size   : " << batch_size << std::endl;
    
    // 3. Init Weights
    std::cout << "[INIT] Generating random weights..." << std::endl;
    Autoencoder cpu_helper; 
    
    std::cout << "[INIT] Booting up GPU..." << std::endl;
    GPUAutoencoder gpu_model(batch_size);

    // Copy trọng số ngẫu nhiên xuống GPU
    gpu_model.loadWeights(
        cpu_helper.w1, cpu_helper.b1, cpu_helper.w2, cpu_helper.b2,
        cpu_helper.w3, cpu_helper.b3, cpu_helper.w4, cpu_helper.b4, cpu_helper.w5, cpu_helper.b5
    );

    // PHASE 3 CHANGE: PINNED MEMORY
    // Thay vì dùng std::vector hay malloc thường, dùng cudaMallocHost
    // Bộ nhớ này nằm trên RAM nhưng được "ghim" để GPU truy cập trực tiếp (DMA)
    float *h_pinned_input;
    cudaMallocHost((void**)&h_pinned_input, input_size_bytes);
    
    // Buffer tạm để nhận dữ liệu từ Dataset class
    std::vector<float> h_batch_buffer; 

    std::cout << "[INFO] Training Started (Optimized)..." << std::endl;

    // Biến đo tổng thời gian
    double total_gpu_time = 0.0;

    // VÒNG LẶP TRAIN
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle_data();
        int batch_count = 0;
        float epoch_loss = 0.0f;

        while (dataset.get_next_batch(batch_size, h_batch_buffer)) {
            // Skip batch cuối nếu không đủ size
            if (h_batch_buffer.size() / (3 * 32 * 32) != batch_size) continue;

            // 1. Copy từ Vector sang Pinned Memory (Trên CPU - rất nhanh)
            memcpy(h_pinned_input, h_batch_buffer.data(), input_size_bytes);

            // 2. Async Copy từ Host (Pinned) -> Device
            // Sử dụng stream_copy (hoặc stream_compute nếu bạn dùng chung)
            cudaMemcpyAsync(gpu_model.d_input, h_pinned_input, input_size_bytes, cudaMemcpyHostToDevice, gpu_model.stream_compute);
            
            // 3. GPU xử lý (Optimized Kernels)
            // Gọi hàm forward với tham số true để dùng Optimized Kernel
            gpu_model.forward_phase3_ver1(); 

            // 4. Tính Loss & Backward & Update
            // Các hàm này cũng nên chạy trên cùng stream
            float loss = gpu_model.compute_loss(gpu_model.d_input); // Lưu ý: hàm này thường trả về float nên có thể gây sync
            epoch_loss += loss;

            gpu_model.backward_phase3_ver1(gpu_model.d_input);
            gpu_model.update(learning_rate);

            batch_count++;
        }
        
        // Đồng bộ GPU sau mỗi Epoch để lấy thời gian chính xác
        cudaDeviceSynchronize();
        
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = epoch_end - epoch_start;
        double epoch_sec = duration.count();
        total_gpu_time += epoch_sec;

        // In kết quả từng Epoch
        std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << target_epochs 
                  << " | Time: " << std::fixed << std::setprecision(2) << epoch_sec << "s"
                  << " | Avg Loss: " << std::setprecision(5) << epoch_loss / batch_count << std::endl;
    }

    // BÁO CÁO KẾT QUẢ
    double avg_time = total_gpu_time / target_epochs;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               GPU RESULT REPORT                  " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "1. PERFORMANCE:" << std::endl;
    std::cout << "   - Total Time (20 Epochs): " << total_gpu_time << " seconds (" << total_gpu_time/60.0 << " min)" << std::endl;
    std::cout << "   - Avg Time per Epoch    : " << avg_time << " seconds" << std::endl;
    
    std::cout << "\n2. MEMORY:" << std::endl;
    print_gpu_memory_usage();
    
    std::cout << "==================================================" << std::endl;

    save_gpu_model(gpu_model, "./output/model_gpu_phase3_ver1.bin");
    
    // Giải phóng Pinned Memory
    cudaFreeHost(h_pinned_input);
    
    std::cout << "Done." << std::endl;
    return 0;
}