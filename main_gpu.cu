#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>

// Include các header
#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"     // Để lấy weight khởi tạo từ CPU
#include "../include/gpu_autoencoder.h" // Model GPU

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   PHASE 2: GPU ACCELERATION & BENCHMARKING       " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Setup Data
    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // 2. Cấu hình
    // GPU thường dùng batch lớn hơn CPU để tận dụng song song
    int batch_size = 64;       
    int target_epochs = 1;     // Chạy thử 1 epoch để lấy số liệu báo cáo (sau đó sửa thành 20)
    float learning_rate = 0.001f;

    std::cout << "\n[CONFIG] Device: NVIDIA Tesla T4 (Google Colab)" << std::endl;
    std::cout << "[CONFIG] Batch Size: " << batch_size 
              << " | Learning Rate: " << learning_rate 
              << " | Target Epochs: " << target_epochs << std::endl;

    // 3. Khởi tạo & Cấp phát bộ nhớ
    std::cout << "[INIT] Initializing Weights on CPU..." << std::endl;
    Autoencoder cpu_model; // Tạo model CPU để sinh weights ngẫu nhiên

    std::cout << "[INIT] Allocating GPU Memory..." << std::endl;
    GPUAutoencoder gpu_model(batch_size);

    std::cout << "[INIT] Transferring Weights to GPU..." << std::endl;
    gpu_model.loadWeights(
        cpu_model.w1, cpu_model.b1, 
        cpu_model.w2, cpu_model.b2,
        cpu_model.w3, cpu_model.b3, 
        cpu_model.w4, cpu_model.b4, 
        cpu_model.w5, cpu_model.b5
    );

    // Buffer tạm trên GPU để chứa input batch
    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));

    std::vector<float> h_batch_data; // Buffer trên CPU

    std::cout << "[INFO] Starting Training Loop..." << std::endl;

    // --- EPOCH LOOP ---
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        
        std::cout << "\n>>> STARTING EPOCH " << epoch + 1 << "/" << target_epochs << " <<<" << std::endl;
        
        // Bắt đầu bấm giờ Epoch
        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        dataset.shuffle_data();
        int batch_count = 0;
        float epoch_loss = 0.0f;

        // --- BATCH LOOP ---
        while (dataset.get_next_batch(batch_size, h_batch_data)) {
            // Nếu batch cuối không đủ size thì bỏ qua (để đơn giản hóa tính toán)
            if (h_batch_data.size() / (3 * 32 * 32) != batch_size) continue;

            // Đo giờ batch (để so sánh với 20s của CPU)
            auto t_batch_start = std::chrono::high_resolution_clock::now();

            // 1. Copy Data: RAM -> VRAM
            cudaMemcpy(d_batch_data, h_batch_data.data(), 
                       batch_size * 3 * 32 * 32 * sizeof(float), 
                       cudaMemcpyHostToDevice);

            // 2. Forward
            gpu_model.forward(d_batch_data);

            // 3. Loss (Target chính là Input)
            float loss = gpu_model.compute_loss(d_batch_data);
            epoch_loss += loss;

            // 4. Backward
            gpu_model.backward(d_batch_data);

            // 5. Update
            gpu_model.update(learning_rate);

            // [QUAN TRỌNG] Đồng bộ hóa để đảm bảo GPU chạy xong mới bấm giờ
            cudaDeviceSynchronize();

            auto t_batch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t_diff = t_batch_end - t_batch_start;

            batch_count++;

            // In log mỗi 50 batch (vì GPU chạy rất nhanh, in nhiều sẽ bị lag console)
            if (batch_count % 50 == 0) {
                std::cout << "Epoch " << std::setw(2) << epoch + 1 
                          << " | Batch " << std::setw(4) << batch_count 
                          << " | Time: " << std::fixed << std::setprecision(4) << t_diff.count() << "s"
                          << " | Loss: " << std::setprecision(5) << loss << std::endl;
            }
        }

        // Kết thúc bấm giờ Epoch
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;
        double epoch_seconds = epoch_duration.count();
        double est_20_epochs = epoch_seconds * 20.0;

        // IN BẢNG BÁO CÁO (Y hệt Phase 1)
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "FINISHED EPOCH " << epoch + 1 << std::endl;
        std::cout << "Avg Loss: " << epoch_loss / batch_count << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        
        std::cout << ">>> TIME REPORT FOR EPOCH " << epoch + 1 << " (GPU):" << std::endl;
        std::cout << "Actual Epoch Time      : " << epoch_seconds << " seconds (" 
                  << epoch_seconds / 60.0 << " minutes)" << std::endl;
        
        std::cout << "Estimated Full 20 Epochs: " << est_20_epochs / 60.0 << " minutes (" 
                  << est_20_epochs / 3600.0 << " hours)" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        // Save weights (Tuỳ chọn: Vì trên Colab save file binary cũng không xem được ngay)
        // Nếu muốn lưu để tải về:
        // gpu_model.save_weights... (cần implement hàm copy device->host rồi save)
    }

    // Cleanup
    cudaFree(d_batch_data);
    std::cout << "Training Complete!" << std::endl;
    return 0;
}