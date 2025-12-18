#ifndef CIFAR10_DATASET_H
#define CIFAR10_DATASET_H

#include <vector>
#include <string>

const int IMG_WIDTH = 32;
const int IMG_HEIGHT = 32;
const int IMG_CHANNELS = 3;
const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS; // 3072

class CIFAR10Dataset {
public:
    std::vector<float> train_images;
    std::vector<unsigned char> train_labels;
    std::vector<float> test_images;
    std::vector<unsigned char> test_labels;

    CIFAR10Dataset(const std::string& data_dir);
    void load_data();

    // --- MỚI: Các hàm hỗ trợ Training ---
    
    // 1. Xáo trộn thứ tự index
    void shuffle_data();

    // 2. Lấy batch tiếp theo
    // Trả về true nếu lấy thành công, false nếu hết dữ liệu (hết epoch)
    bool get_next_batch(int batch_size, 
                        std::vector<float>& batch_data); // Autoencoder chỉ cần ảnh, không cần label để train

    // 3. Reset iterator về đầu (dùng sau mỗi epoch)
    void reset_iterator();

    // Lấy số lượng ảnh train
    size_t get_num_train() const { return train_labels.size(); }

private:
    std::string data_dir;
    
    // --- MỚI: Quản lý shuffle và batch ---
    std::vector<int> indices; // Mảng lưu chỉ số [0, 1, ..., 49999]
    size_t current_batch_index; // Con trỏ hiện tại đang đọc đến đâu

    void read_batch(const std::string& filename, 
                    std::vector<float>& images, 
                    std::vector<unsigned char>& labels);
};

#endif