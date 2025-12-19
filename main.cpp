#include <iostream>
#include <vector>
#include <chrono> 
#include <iomanip> 
#include <numeric> 

#include "include/cifar10_dataset.h"
#include "include/autoencoder.h"
#include "include/layers.h"

int main() {
    std::cout << "==========================" << std::endl;
    std::cout << "   PHASE 1: CPU BASELINE     " << std::endl;
    std::cout << "==========================" << std::endl;

    std::string data_path = "./data"; 
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();
    
    Autoencoder model;
    
    // C?u h?nh ch?y th?t
    int batch_size = 32;
    int target_epochs = 1;      // B?n mu?n ch?y th? 1 epoch trý?c th? ð? 1, mu?n ch?y h?t th? ð? 20
    float learning_rate = 0.001f;
    
    // [QUAN TR?NG] Ð? t?t gi?i h?n test ð? ch?y full
    // int MAX_BATCHES_TO_TEST = 10; 

    std::cout << "\n[CONFIG] Batch Size: " << batch_size 
              << " | Learning Rate: " << learning_rate 
              << " | Target Epochs: " << target_epochs << std::endl;
    std::cout << "[INFO] Starting full training loop..." << std::endl;

    std::vector<float> batch_data;

    // --- EPOCH LOOP ---
    for (int epoch = 0; epoch < target_epochs; ++epoch) {
        
        std::cout << "\nSTARTING EPOCH " << epoch + 1 << "/" << target_epochs << std::endl;
        
        dataset.shuffle_data(); // Xáo tr?n d? li?u ð?u m?i epoch
        int batch_count = 0;    // Reset ð?m batch cho epoch m?i
        float epoch_loss = 0.0f;

        // --- BATCH LOOP ---
        while (dataset.get_next_batch(batch_size, batch_data)) {
            int current_batch_size = batch_data.size() / (3 * 32 * 32);
            auto t_start = std::chrono::high_resolution_clock::now();

            // 1. Forward
            model.forward(batch_data, current_batch_size);

            // 2. Compute Loss
            float loss = CPULayers::mse_loss(model.output, batch_data);
            epoch_loss += loss;

            // 3. Backward
            model.backward(batch_data, current_batch_size);

            // 4. Update
            model.update(learning_rate);

            auto t_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t_diff = t_end - t_start;
            
            batch_count++;

            if (batch_count % 1 == 0) {
                std::cout << "Epoch " << std::setw(2) << epoch + 1 
                          << " | Batch " << std::setw(4) << batch_count 
                          << " | Time: " << std::fixed << std::setprecision(2) << t_diff.count() << "s"
                          << " | Loss: " << std::setprecision(5) << loss << std::endl;
            }
        }
        
        // T?ng k?t Epoch
        std::cout << "------------------" << std::endl;
        std::cout << "FINISHED EPOCH " << epoch + 1 
                  << " | Avg Loss: " << epoch_loss / batch_count << std::endl;
        std::cout << "------------------" << std::endl;

        // Save Model sau m?i epoch 
        std::string save_path = "./output/model_epoch_" + std::to_string(epoch + 1) + ".bin";
        // system("mkdir -p output"); // B? d?ng này n?u ch?y trên Windows b? l?i mkdir
        model.save_weights(save_path);
    }

    std::cout << "Training Complete!" << std::endl;
    return 0;
}
