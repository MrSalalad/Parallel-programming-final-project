#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>

#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"     
#include "../include/gpu_autoencoder.h" 

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
    std::cout << "       PHASE 2: GPU TRAINING (20 EPOCHS)          " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Setup Data
    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // 2. Config
    int batch_size = 64;       
    int target_epochs = 20;    // Chạy đủ 20 vòng
    float learning_rate = 0.001f;

    std::cout << "[CONFIG] Target Epochs: " << target_epochs << std::endl;
    
    // 3. Init Weights
    // Vẫn dùng Autoencoder class để sinh số ngẫu nhiên ban đầu (chỉ mất <1s)
    std::cout << "[INIT] Generating random weights..." << std::endl;
    Autoencoder cpu_helper; 
    
    std::cout << "[INIT] Booting up GPU..." << std::endl;
    GPUAutoencoder gpu_model(batch_size);

    // Copy trọng số ngẫu nhiên xuống GPU
    gpu_model.loadWeights(
        cpu_helper.w1, cpu_helper.b1, cpu_helper.w2, cpu_helper.b2,
        cpu_helper.w3, cpu_helper.b3, cpu_helper.w4, cpu_helper.b4, cpu_helper.w5, cpu_helper.b5
    );

    // Cấp phát bộ nhớ tạm
    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));
    std::vector<float> h_batch_data;

    std::cout << "[INFO] Training Started..." << std::endl;

    // Biến đo tổng thời gian
    double total_gpu_time = 0.0;

    // --- VÒNG LẶP TRAIN ---
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        dataset.shuffle_data();
        int batch_count = 0;
        float epoch_loss = 0.0f;

        while (dataset.get_next_batch(batch_size, h_batch_data)) {
            // Skip batch cuối nếu không đủ size (để code gọn)
            if (h_batch_data.size() / (3 * 32 * 32) != batch_size) continue;

            // Copy Data xuống GPU
            cudaMemcpy(d_batch_data, h_batch_data.data(), batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
            
            // GPU xử lý toàn bộ
            gpu_model.forward(d_batch_data);
            float loss = gpu_model.compute_loss(d_batch_data);
            epoch_loss += loss;
            gpu_model.backward(d_batch_data);
            gpu_model.update(learning_rate);

            batch_count++;
        }
        
        // Đồng bộ GPU để đo thời gian chính xác
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

    // --- BÁO CÁO KẾT QUẢ ---
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

    cudaFree(d_batch_data);
    std::cout << "Done." << std::endl;
    return 0;
}