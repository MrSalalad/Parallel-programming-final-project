#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <cuda_runtime.h>

#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"     
#include "../include/gpu_autoencoder.h" 

// Macro kiểm tra lỗi GPU cực gắt
#define CHECK_ERR(msg) { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "❌ CUDA ERROR at " << msg << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "          DEBUG MODE: KIỂM TRA LỖI                " << std::endl;
    std::cout << "==================================================" << std::endl;

    // 1. Kiểm tra Data
    std::string data_path = "./data"; 
    std::cout << "[1] Loading Data from: " << data_path << " ..." << std::endl;
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // Lấy thử 1 batch để soi
    std::vector<float> debug_batch;
    if (dataset.get_next_batch(10, debug_batch)) {
        std::cout << "   ✅ Loaded batch successfully." << std::endl;
        std::cout << "   🔎 Inspecting 1st Pixel value: " << debug_batch[0] << std::endl;
        
        float sum = 0;
        for(float x : debug_batch) sum += x;
        std::cout << "   🔎 Sum of batch (Check if 0): " << sum << std::endl;
        
        if (sum == 0) {
            std::cout << "   ❌ CẢNH BÁO: DỮ LIỆU TOÀN SỐ 0! KIỂM TRA LẠI FILE DATA." << std::endl;
            return 1;
        }
    } else {
        std::cout << "   ❌ LỖI: Không load được batch nào!" << std::endl;
        return 1;
    }

    // 2. Kiểm tra Weights
    std::cout << "\n[2] Initializing Weights..." << std::endl;
    Autoencoder cpu_helper;
    std::cout << "   🔎 Inspecting 1st Weight value (w1): " << cpu_helper.w1[0] << std::endl;
    if (cpu_helper.w1[0] == 0) {
        std::cout << "   ❌ CẢNH BÁO: TRỌNG SỐ TOÀN 0! HÀM KHỞI TẠO BỊ LỖI." << std::endl;
    }

    // 3. Setup GPU
    std::cout << "\n[3] Setting up GPU..." << std::endl;
    int batch_size = 64;
    GPUAutoencoder gpu_model(batch_size);
    CHECK_ERR("GPU Constructor");

    gpu_model.loadWeights(
        cpu_helper.w1, cpu_helper.b1, cpu_helper.w2, cpu_helper.b2,
        cpu_helper.w3, cpu_helper.b3, cpu_helper.w4, cpu_helper.b4, cpu_helper.w5, cpu_helper.b5
    );
    CHECK_ERR("Load Weights");

    float *d_batch_data;
    cudaMalloc(&d_batch_data, batch_size * 3 * 32 * 32 * sizeof(float));
    CHECK_ERR("Malloc Input");

    // 4. Test Run 1 Batch
    std::cout << "\n[4] Test Running 1 Batch on GPU..." << std::endl;
    
    // Copy data thật vào
    std::vector<float> h_batch_data;
    dataset.reset_iterator(); // Reset để lấy lại từ đầu
    dataset.get_next_batch(batch_size, h_batch_data);
    
    cudaMemcpy(d_batch_data, h_batch_data.data(), batch_size * 3 * 32 * 32 * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_ERR("Memcpy HostToDevice");

    gpu_model.forward(d_batch_data);
    CHECK_ERR("Forward Pass");
    
    float loss = gpu_model.compute_loss(d_batch_data);
    CHECK_ERR("Compute Loss");

    std::cout << "   📊 Single Batch Loss: " << loss << std::endl;

    if (loss == 0.0f) {
         std::cout << "   ❌ KẾT LUẬN: Loss vẫn = 0. Có thể Kernel không chạy hoặc tính sai." << std::endl;
    } else {
         std::cout << "   ✅ KẾT LUẬN: Loss > 0 (" << loss << "). Có vẻ ổn. Hãy chạy lại code Training Full." << std::endl;
    }

    cudaFree(d_batch_data);
    return 0;
}