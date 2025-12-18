#include <iostream>
#include <vector>
#include <chrono> // Đo thời gian
#include "include/cifar10_dataset.h"
#include "include/autoencoder.h"
#include "include/layers.h"

int main() {
    std::cout << "--- Phase 1.4: CPU Training Loop ---" << std::endl;

    // 1. Setup
    CIFAR10Dataset dataset("./data");
    dataset.load_data();
    
    Autoencoder model;
    
    // Hyperparameters
    int batch_size = 32;
    int epochs = 1;         // Test với 1 epoch
    float learning_rate = 0.001f;
    int max_batches_to_run = 5; // CHỈ CHẠY 5 BATCH ĐỂ TEST (Bỏ dòng này nếu muốn chạy full)

    std::cout << "Start Training (CPU Baseline)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        dataset.shuffle_data();
        std::vector<float> batch_data;
        float total_loss = 0.0f;
        int batch_count = 0;

        while (dataset.get_next_batch(batch_size, batch_data)) {
            // 1. Forward
            model.forward(batch_data, batch_size);

            // 2. Compute Loss
            float loss = CPULayers::mse_loss(model.output, batch_data);
            total_loss += loss;

            // 3. Backward
            model.backward(batch_data, batch_size);

            // 4. Update
            model.update(learning_rate);

            // Logging
            if (batch_count % 1 == 0) { // Log mỗi batch
                std::cout << "Epoch " << epoch + 1 << " | Batch " << batch_count 
                          << " | Loss: " << loss << std::endl;
            }

            batch_count++;
            
            // --- TEST LIMIT: Dừng sau 5 batch để không phải chờ ---
            if (batch_count >= max_batches_to_run) {
                std::cout << "Stopping early for verification (Remove limit to run full dataset)" << std::endl;
                break;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    std::cout << "Training finished in " << duration.count() << " seconds." << std::endl;

    // Save Model
    model.save_weights("./output/model_cpu.bin");

    return 0;
}