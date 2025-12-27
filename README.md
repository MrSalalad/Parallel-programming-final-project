CUDA Convolutional Autoencoder for CIFAR-10
1. Hardware RequirementsTo run the GPU implementation (Phase 2 & 3), ensure your environment meets the following: GPU: NVIDIA GPU with Compute Capability >= 3.5. Recommended: NVIDIA Tesla T4, P100, V100, or RTX Series.
VRAM: Minimum 4GB (Project uses ~500MB - 1GB depending on batch size).
Host RAM: 8GB+.
Disk Space: ~200MB (for Dataset and Output files).

2. Setup Instructions
Dependencies & Libraries
OS: Linux (Ubuntu 18.04/20.04/22.04 recommended).
CUDA Toolkit: Version 11.0 or higher (contains nvcc).
C++ Compiler: g++ (supports C++11 standard).

Data Setup:
The project requires the CIFAR-10 Binary Version.
# 1. Download the dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# 2. Extract and organize
tar -xzvf cifar-10-binary.tar.gz
mkdir -p data
mv cifar-10-batches-bin/*.bin data/

# Clean up
rm -rf cifar-10-batches-bin cifar-10-binary.tar.gz

3. Compilation Commands
# 1. Compile GPU Training (Phase 2):
!nvcc -arch=sm_75 -o train_gpu \
    main_gpu.cu \
    src/gpu/gpu_autoencoder.cu \
    src/gpu/kernels_naive.cu \
    src/cifar10_dataset.cpp \
    src/autoencoder.cpp \
    src/layers.cpp \
    -I./include \
    src/gpu/kernels_phase3.cu \
    -O3 

# 2. Compile GPU Training (Phase 3):
!nvcc -arch=sm_75 -o train_gpu_optimized_memory \
    main_gpu_version1.cu \
    src/gpu/gpu_autoencoder.cu \
    src/gpu/kernels_naive.cu \
    src/gpu/kernels_phase3.cu \
    src/cifar10_dataset.cpp \
    src/autoencoder.cpp \
    src/layers.cpp \
    -I./include \
    -O3

4. Execution
# 1. CPU (Phase 1):
Just need to point to project root directory on the terminal than run build_run.bat
# 2. GPU (Phase 2): 
!./train_gpu
# 3. GPU (Phase 3):
!./train_gpu_optimized_memory

5. Expected Outputs
output/model_cpu.bin
output/model_gpu_phase2.bin: The binary file containing trained weights (~1MB).
output/model_gpu_phase3_ver1.bin