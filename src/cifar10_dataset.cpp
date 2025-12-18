#include "cifar10_dataset.h"
#include <fstream>
#include <iostream>
#include <algorithm> // cho std::shuffle
#include <random>    // cho std::default_random_engine, std::chrono
#include <chrono>
#include <numeric>   // cho std::iota

CIFAR10Dataset::CIFAR10Dataset(const std::string& dir) 
    : data_dir(dir), current_batch_index(0) {}

void CIFAR10Dataset::load_data() {
    // ... (Giữ nguyên phần load code cũ của bạn ở đây) ...
    // ... Phần reserve và loop đọc file ...
    // Code cũ:
    std::cout << "Loading training data..." << std::endl;
    size_t total_train_floats = 50000 * (size_t)IMG_SIZE; 
    train_images.reserve(total_train_floats);
    train_labels.reserve(50000);

    for (int i = 1; i <= 5; ++i) {
        std::string path = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        read_batch(path, train_images, train_labels);
    }
    
    // --- MỚI: Khởi tạo indices cho việc shuffle ---
    indices.resize(train_labels.size());
    // Điền giá trị từ 0 đến 49999 vào indices
    std::iota(indices.begin(), indices.end(), 0);
    std::cout << "Initialized " << indices.size() << " indices for shuffling." << std::endl;

    // Load test set (giữ nguyên code cũ)
    size_t total_test_floats = 10000 * (size_t)IMG_SIZE;
    test_images.reserve(total_test_floats);
    test_labels.reserve(10000);
    std::string test_path = data_dir + "/test_batch.bin";
    read_batch(test_path, test_images, test_labels);
}

// ... (Giữ nguyên hàm read_batch cũ) ...
void CIFAR10Dataset::read_batch(const std::string& filename, 
                                std::vector<float>& images, 
                                std::vector<unsigned char>& labels) {
    // Paste lại code read_batch cũ của bạn vào đây
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }
    int num_images = 10000;
    std::vector<unsigned char> buffer(1 + IMG_SIZE);
    for (int i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        if (!file) break;
        labels.push_back(buffer[0]);
        for (int j = 0; j < IMG_SIZE; ++j) {
            images.push_back(static_cast<float>(buffer[1 + j]) / 255.0f);
        }
    }
    file.close();
}

// --- CÀI ĐẶT CÁC HÀM MỚI ---

void CIFAR10Dataset::shuffle_data() {
    // Dùng seed theo thời gian thực để mỗi lần chạy shuffle khác nhau
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    
    // Reset lại vị trí đọc batch về đầu
    current_batch_index = 0; 
}

void CIFAR10Dataset::reset_iterator() {
    current_batch_index = 0;
}

bool CIFAR10Dataset::get_next_batch(int batch_size, std::vector<float>& batch_data) {
    // Kiểm tra nếu đã hết dữ liệu trong epoch này
    if (current_batch_index >= indices.size()) {
        return false;
    }

    // Tính toán kích thước batch thực tế (batch cuối có thể nhỏ hơn batch_size)
    size_t end_index = std::min(current_batch_index + batch_size, indices.size());
    size_t actual_batch_size = end_index - current_batch_index;

    // Resize vector output
    batch_data.resize(actual_batch_size * IMG_SIZE);

    // Copy dữ liệu
    for (size_t i = 0; i < actual_batch_size; ++i) {
        // Lấy index thực từ mảng đã shuffle
        int original_idx = indices[current_batch_index + i];

        // Copy 3072 floats của ảnh đó vào batch_data
        // Vị trí bắt đầu trong mảng gốc: original_idx * 3072
        size_t start_pos = original_idx * IMG_SIZE;
        
        // Copy sang batch_data
        for (int j = 0; j < IMG_SIZE; ++j) {
            batch_data[i * IMG_SIZE + j] = train_images[start_pos + j];
        }
    }

    // Cập nhật con trỏ
    current_batch_index += actual_batch_size;
    return true;
}