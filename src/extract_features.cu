#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm> // Cho std::min
#include <cmath>     // Cho std::abs

#include "../include/cifar10_dataset.h"
#include "../include/autoencoder.h"
#include "../include/gpu_autoencoder.h"

// Đường dẫn model đã train
const std::string TRAINED_MODEL_PATH = "output/model_gpu_phase3_ver1.bin";

// --- [HELPER] HÀM KIỂM TRA VECTOR CÓ PHẢI TOÀN SỐ 0 KHÔNG ---
void check_non_zero(const std::vector<float>& vec, std::string name) {
    float sum = 0.0f;
    float max_val = 0.0f;
    // Check 1000 phần tử đầu tiên để tiết kiệm thời gian
    int limit = std::min((int)vec.size(), 1000);
    for(int i=0; i<limit; i++) {
        sum += std::abs(vec[i]);
        if(std::abs(vec[i]) > max_val) max_val = std::abs(vec[i]);
    }
    std::cout << "   [DEBUG] " << name << " -> Sum(first 1k): " << sum << " | Max: " << max_val << std::endl;
    
    if (max_val == 0.0f) {
        std::cerr << "\n❌ CẢNH BÁO ĐỎ: " << name << " TOÀN SỐ 0!!!" << std::endl;
        std::cerr << "   => Nguyên nhân: File bin rỗng hoặc Model chưa học được gì.\n" << std::endl;
    }
}

// Hàm đọc 1 mảng float từ file binary
void read_layer_weights(std::ifstream& file, std::vector<float>& vec, int size) {
    vec.resize(size);
    file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    if (!file) {
        std::cerr << "[ERROR] Failed to read weight layer of size " << size << std::endl;
        exit(1);
    }
}

// Hàm load toàn bộ trọng số từ file bin vào CPU vectors
void load_trained_weights(const std::string& filepath, Autoencoder& cpu_placeholder) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open trained model file: " << filepath << std::endl;
        exit(1);
    }

    std::cout << "[LOAD] Loading weights from " << filepath << "..." << std::endl;

    // Load theo đúng thứ tự đã lưu ở Phase 3
    read_layer_weights(file, cpu_placeholder.w1, 256*3*3*3);

    check_non_zero(cpu_placeholder.w1, "Weight Layer 1 (CPU)");

    read_layer_weights(file, cpu_placeholder.b1, 256);
    
    read_layer_weights(file, cpu_placeholder.w2, 128*256*3*3);
    read_layer_weights(file, cpu_placeholder.b2, 128);
    
    read_layer_weights(file, cpu_placeholder.w3, 128*128*3*3);
    read_layer_weights(file, cpu_placeholder.b3, 128);
    
    read_layer_weights(file, cpu_placeholder.w4, 256*128*3*3);
    read_layer_weights(file, cpu_placeholder.b4, 256);
    
    read_layer_weights(file, cpu_placeholder.w5, 3*256*3*3);
    read_layer_weights(file, cpu_placeholder.b5, 3);

    file.close();
    std::cout << "[LOAD] Weights loaded successfully." << std::endl;
}

// Hàm trích xuất và lưu features
void process_dataset(GPUAutoencoder& gpu_model, 
                     const std::vector<float>& images, 
                     const std::vector<unsigned char>& labels,
                     const std::string& prefix) {
    
    int num_samples = images.size() / 3072; 
    int batch_size = 100; // Giảm batch size xuống 100 cho an toàn bộ nhớ
    int feature_dim = 128 * 8 * 8; // 8192

    std::string feat_out = "output/" + prefix + "_features.bin";
    std::string label_out = "output/" + prefix + "_labels.bin";

    std::ofstream f_feat(feat_out, std::ios::binary);
    std::ofstream f_label(label_out, std::ios::binary);

    std::cout << "--> Extracting " << prefix << " set (" << num_samples << " images)..." << std::endl;

    std::vector<float> h_features(batch_size * feature_dim);
    
    for (int i = 0; i < num_samples; i += batch_size) {
        int current_batch_size = std::min(batch_size, num_samples - i);
        size_t offset = (size_t)i * 3072;
        
        // 1. Copy Batch lên GPU
        cudaMemcpy(gpu_model.d_input, &images[offset], current_batch_size * 3072 * sizeof(float), cudaMemcpyHostToDevice);

        // 2. Chạy Forward
        gpu_model.forward_phase3_ver1(); 

        // 3. Lấy Output của Encoder (d_encoded)
        cudaMemcpy(h_features.data(), gpu_model.d_encoded, current_batch_size * feature_dim * sizeof(float), cudaMemcpyDeviceToHost);

        if (i == 0) {
            check_non_zero(h_features, "Extracted Features (Batch 0)");
        }

        // 4. Ghi ra file
        f_feat.write(reinterpret_cast<char*>(h_features.data()), current_batch_size * feature_dim * sizeof(float));
        
        // 5. Ghi Labels
        f_label.write(reinterpret_cast<const char*>(&labels[i]), current_batch_size * sizeof(unsigned char));

        if ((i + batch_size) % 10000 == 0) 
            std::cout << "    Processed " << (i + batch_size) << "/" << num_samples << std::endl;
    }

    f_feat.close();
    f_label.close();
    std::cout << "[DONE] Saved to " << feat_out << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   PHASE 4 PREP: FEATURE EXTRACTION (DEBUG)       " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string data_path = "./data";
    CIFAR10Dataset dataset(data_path);
    
    std::cout << "[DATA] Loading Training Set..." << std::endl;
    dataset.load_data(); 
    
    // Load Test Data Fix
    if (dataset.test_images.empty()) {
        std::cout << "[DATA] Loading Test Set (test_batch.bin)..." << std::endl;
        std::string test_file = data_path + "/test_batch.bin";
        std::ifstream file(test_file, std::ios::binary);
        if (file.is_open()) {
            int num_test_images = 10000;
            dataset.test_images.resize(num_test_images * 3072);
            dataset.test_labels.resize(num_test_images);
            
            for (int i = 0; i < num_test_images; ++i) {
                char label;
                file.read(&label, 1);
                dataset.test_labels[i] = (unsigned char)label;
                std::vector<unsigned char> buffer(3072);
                file.read(reinterpret_cast<char*>(buffer.data()), 3072);
                for (int j = 0; j < 3072; ++j) dataset.test_images[i * 3072 + j] = buffer[j] / 255.0f;
            }
            file.close();
        } 
    }

    int batch_size = 100; 
    GPUAutoencoder gpu_model(batch_size);
    Autoencoder cpu_weights_holder; 
    
    // Load weights vào CPU
    load_trained_weights(TRAINED_MODEL_PATH, cpu_weights_holder);

    // Load weights xuống GPU
    gpu_model.loadWeights(
        cpu_weights_holder.w1, cpu_weights_holder.b1,
        cpu_weights_holder.w2, cpu_weights_holder.b2,
        cpu_weights_holder.w3, cpu_weights_holder.b3,
        cpu_weights_holder.w4, cpu_weights_holder.b4,
        cpu_weights_holder.w5, cpu_weights_holder.b5
    );

    if (!dataset.train_images.empty()) process_dataset(gpu_model, dataset.train_images, dataset.train_labels, "train");
    if (!dataset.test_images.empty()) process_dataset(gpu_model, dataset.test_images, dataset.test_labels, "test");

    return 0;
}