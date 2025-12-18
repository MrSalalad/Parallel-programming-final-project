#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h>

// Include các header cần thiết
#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"      // CPU Model (để lấy weight init ban đầu)
#include "../include/gpu_autoencoder.h"  // GPU Model

int main() {
    // =========================================================================
    // 1. SETTINGS (PHASE 2.5)
    // =========================================================================
    std::string data_path = "./data/cifar-10-batches-bin";
    
    // Đề bài yêu cầu batch size lớn hơn cho GPU
    int batch_size = 64;       
    int epochs = 20;           
    float learning_rate = 0.001f; 

    std::cout << "===========================================" << std::endl;
    std::cout << "   PHASE 2: GPU TRAINING (NAIVE KERNELS)   " << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;

    // =========================================================================
    // 2. INITIALIZATION
    // =========================================================================
    
    // A. Load Data (CPU)
    std::cout << "[1/4] Loading Data..." << std::endl;
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // B. Init CPU Model (để tạo weights ngẫu nhiên ban đầu)
    std::cout << "[2/4] Initializing Weights (CPU)..." << std::endl;
    Autoencoder cpu_model; 

    // C. Init GPU Model & Allocate Memory
    std::cout << "[3/4] Allocating GPU Memory..." << std::endl;
    GPUAutoencoder gpu_model(batch_size);

    // D. Copy Weights CPU -> GPU
    std::cout << "[4/4] Transferring Weights to GPU..." << std::endl;
    gpu_model.loadWeights(
        cpu_model.w1, cpu_model.b1, 
        cpu_model.w2, cpu_model.b2,
        cpu_model.w3, cpu_model.b3, 
        cpu_model.w4, cpu_model.b4, 
        cpu_model.w5, cpu_model.b5
    );

    // Buffer tạm trên GPU để chứa Input Batch
    // (Lý do: Dataset trả về vector trên RAM, ta cần copy xuống VRAM trước khi đưa vào mạng)
    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));

    // Biến để đo thời gian bằng CUDA Events (Chính xác hơn std::chrono)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // =========================================================================
    // 3. TRAINING LOOP
    // =========================================================================
    std::vector<float> h_batch_data; // Host buffer

    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataset.shuffle_data();
        float total_loss = 0.0f;
        int batch_count = 0;

        // Bắt đầu đo giờ epoch
        cudaEventRecord(start);

        while (dataset.get_next_batch(batch_size, h_batch_data)) {
            // Nếu batch cuối không đủ 64 ảnh, ta bỏ qua để đơn giản hóa việc tính toán index
            // (Trong thực tế sẽ padding, nhưng ở Phase Naive này ta skip cho code gọn)
            if (h_batch_data.size() / (3 * 32 * 32) != batch_size) continue;

            // 1. Copy Batch: RAM -> VRAM
            cudaMemcpy(d_batch_data, h_batch_data.data(), 
                       batch_size * 3 * 32 * 32 * sizeof(float), 
                       cudaMemcpyHostToDevice);

            // 2. Forward
            gpu_model.forward(d_batch_data);

            // 3. Compute Loss
            // Lưu ý: d_batch_data đóng vai trò là Target luôn (Autoencoder: Input == Target)
            float loss = gpu_model.compute_loss(d_batch_data);
            total_loss += loss;

            // 4. Backward
            gpu_model.backward(d_batch_data);

            // 5. Update
            gpu_model.update(learning_rate);

            batch_count++;
            
            // In tiến độ (Log mỗi 50 batch để đỡ rối mắt)
            if (batch_count % 50 == 0) {
                std::cout << "\rBatch " << batch_count << " | Loss: " << loss << std::flush;
            }
        }

        // Dừng đo giờ
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float avg_loss = total_loss / batch_count;
        std::cout << "\n>>> Epoch " << epoch + 1 << "/" << epochs 
                  << " | Time: " << std::fixed << std::setprecision(3) << milliseconds / 1000.0 << "s"
                  << " | Avg Loss: " << std::setprecision(5) << avg_loss << std::endl;
    }

    // =========================================================================
    // 4. CLEANUP
    // =========================================================================
    // Save weights (Tuỳ chọn: Bạn cần implement hàm save trên GPU hoặc copy về CPU để save)
    // gpu_model.save_weights("gpu_model.bin"); 

    cudaFree(d_batch_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "Training Complete!" << std::endl;
    return 0;
}