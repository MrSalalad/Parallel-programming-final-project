#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <numeric> 

#include "include/cifar10_dataset.h"
#include "include/autoencoder.h"
#include "include/layers.h"

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "   PHASE 1: CPU FULL TRAINING & TIME ESTIMATION   " << std::endl;
    std::cout << "==================================================" << std::endl;

    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();
    
    Autoencoder model;
    
    // Cấu hình
    int batch_size = 32;
    int target_epochs = 1; // Để là 1 để chạy lấy số liệu báo cáo, sau đó thích thì sửa thành 20
    float learning_rate = 0.001f;
    
    std::cout << "\n[CONFIG] Batch Size: " << batch_size 
              << " | Learning Rate: " << learning_rate 
              << " | Target Epochs: " << target_epochs << std::endl;

    std::vector<float> batch_data;

    // --- EPOCH LOOP ---
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        
        std::cout << "\n>>> STARTING EPOCH " << epoch + 1 << "/" << target_epochs << " <<<" << std::endl;
        
        // 1. BẮT ĐẦU BẤM GIỜ CHO CẢ EPOCH
        auto epoch_start_time = std::chrono::high_resolution_clock::now();

        dataset.shuffle_data(); 
        int batch_count = 0;    
        float epoch_loss = 0.0f;

        // BATCH LOOP
        while (dataset.get_next_batch(batch_size, batch_data)) {
            int current_batch_size = batch_data.size() / (3 * 32 * 32);
            
            // Đo giờ từng batch
            auto t_batch_start = std::chrono::high_resolution_clock::now();

            model.forward(batch_data, current_batch_size);
            float loss = CPULayers::mse_loss(model.output, batch_data);
            epoch_loss += loss;
            model.backward(batch_data, current_batch_size);
            model.update(learning_rate);

            auto t_batch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t_diff = t_batch_end - t_batch_start;
            
            batch_count++;

            // Log mỗi batch
            if (batch_count % 1 == 0) {
                std::cout << "Epoch " << std::setw(2) << epoch + 1 
                          << " | Batch " << std::setw(4) << batch_count 
                          << " | Time: " << std::fixed << std::setprecision(2) << t_diff.count() << "s"
                          << " | Loss: " << std::setprecision(5) << loss << std::endl;
            }
        }
        
        // 2. KẾT THÚC BẤM GIỜ EPOCH
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;
        double epoch_seconds = epoch_duration.count();

        // 3. TÍNH TOÁN ƯỚC LƯỢNG 20 EPOCHS
        double est_20_epochs = epoch_seconds * 20.0;

        // Tổng kết Epoch
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "FINISHED EPOCH " << epoch + 1 << std::endl;
        std::cout << "Avg Loss: " << epoch_loss / batch_count << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        
        // IN RA THÔNG TIN THỜI GIAN
        std::cout << ">>> TIME REPORT FOR EPOCH " << epoch + 1 << ":" << std::endl;
        std::cout << "Actual Epoch Time      : " << epoch_seconds / 3600.0 << " hours (" 
                  << epoch_seconds / 60.0 << " minutes)" << std::endl;
        
        std::cout << "Estimated Full 20 Epochs: " << est_20_epochs / 3600.0 << " hours (" 
                  << est_20_epochs / (3600.0 * 24.0) << " days)" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;

        // Save Model
        std::string save_path = "./output/model_epoch_" + std::to_string(epoch + 1) + ".bin";
        // system("mkdir -p output"); 
        model.save_weights(save_path);
    }

    std::cout << "Training Complete!" << std::endl;
    return 0;
}