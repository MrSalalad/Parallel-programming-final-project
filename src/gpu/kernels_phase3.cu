#include "../../include/kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>

#define TILE_WIDTH 16
#define K_SIZE 3
#define BLOCK_WIDTH (TILE_WIDTH + K_SIZE - 1) 

namespace Phase3Kernels {

    // =====================================================================
    // 1. FORWARD: SHARED MEMORY (Đã thêm __restrict__ và #pragma unroll)
    // =====================================================================
    __global__ void conv2d_shared_mem_kernel(const float* __restrict__ input, 
                                             float* __restrict__ output, 
                                             const float* __restrict__ weights, 
                                             const float* __restrict__ bias,
                                             int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                             int out_h, int out_w) {
        
        __shared__ float s_input[BLOCK_WIDTH][BLOCK_WIDTH];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row_o = blockIdx.y * TILE_WIDTH + ty;
        int col_o = blockIdx.x * TILE_WIDTH + tx;
        
        int b_oc_idx = blockIdx.z;
        int b = b_oc_idx / out_c;
        int oc = b_oc_idx % out_c;

        float sum = (bias != nullptr) ? bias[oc] : 0.0f;

        // Loop channel
        for (int ic = 0; ic < in_c; ++ic) {
            // 1. Load Shared Mem
            int start_r = blockIdx.y * TILE_WIDTH - 1; 
            int start_c = blockIdx.x * TILE_WIDTH - 1;

            // Unroll loops load
            #pragma unroll
            for(int i = ty; i < BLOCK_WIDTH; i += TILE_WIDTH) {
                #pragma unroll
                for(int j = tx; j < BLOCK_WIDTH; j += TILE_WIDTH) {
                    int r_in = start_r + i;
                    int c_in = start_c + j;
                    if (r_in >= 0 && r_in < in_h && c_in >= 0 && c_in < in_w) {
                         // Coalesced access pattern optimization
                        int input_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + r_in * in_w + c_in;
                        s_input[i][j] = input[input_idx];
                    } else {
                        s_input[i][j] = 0.0f;
                    }
                }
            }
            __syncthreads();

            // 2. Compute
            if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
                #pragma unroll
                for (int i = 0; i < K_SIZE; ++i) {
                    #pragma unroll
                    for (int j = 0; j < K_SIZE; ++j) {
                        int w_idx = oc * (in_c * K_SIZE * K_SIZE) + ic * (K_SIZE * K_SIZE) + i * K_SIZE + j;
                        sum += s_input[ty + i][tx + j] * weights[w_idx];
                    }
                }
            }
            __syncthreads();
        }

        if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < out_h && col_o < out_w) {
            int out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + row_o * out_w + col_o;
            output[out_idx] = sum;
        }
    }

    // =====================================================================
    // 2. BACKWARD INPUT: SHARED MEMORY
    // =====================================================================
    // Tính dL/dX: Về bản chất là Convolution giữa Grad_Output (padded) và Weights (rotated 180)
    __global__ void conv2d_backward_input_shared_mem_kernel(const float* __restrict__ grad_output, 
                                                            const float* __restrict__ weights, 
                                                            float* __restrict__ grad_input,
                                                            int batch_size, int in_c, int out_c, int in_h, int in_w, 
                                                            int out_h, int out_w) {
        
        // Input của phép này là Grad_Output, Output là Grad_Input
        // Ta map thread theo Grad_Input (Input gốc của forward)
        __shared__ float s_grad_out[BLOCK_WIDTH][BLOCK_WIDTH];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        
        // Toạ độ trên Grad_Input
        int row_i = blockIdx.y * TILE_WIDTH + ty;
        int col_i = blockIdx.x * TILE_WIDTH + tx;

        int b_ic_idx = blockIdx.z;
        int b = b_ic_idx / in_c;
        int ic = b_ic_idx % in_c; // Output channel của backward là Input channel của forward

        float sum = 0.0f;

        // Loop qua Out Channels (vốn là Input Channels của phép Conv này)
        for (int oc = 0; oc < out_c; ++oc) {
            
            // Padding logic cho Backward Input (Full Convolution)
            // Forward pad=1, k=3 => Backward Input tương đương pad=1
            int start_r = blockIdx.y * TILE_WIDTH - 1; 
            int start_c = blockIdx.x * TILE_WIDTH - 1;

            // Load Tile từ Grad_Output
            #pragma unroll
            for(int i = ty; i < BLOCK_WIDTH; i += TILE_WIDTH) {
                #pragma unroll
                for(int j = tx; j < BLOCK_WIDTH; j += TILE_WIDTH) {
                    int r_out = start_r + i;
                    int c_out = start_c + j;
                    
                    if (r_out >= 0 && r_out < out_h && c_out >= 0 && c_out < out_w) {
                        int go_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + r_out * out_w + c_out;
                        s_grad_out[i][j] = grad_output[go_idx];
                    } else {
                        s_grad_out[i][j] = 0.0f;
                    }
                }
            }
            __syncthreads();

            // Compute
            if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_i < in_h && col_i < in_w) {
                #pragma unroll
                for (int i = 0; i < K_SIZE; ++i) {
                    #pragma unroll
                    for (int j = 0; j < K_SIZE; ++j) {
                        // ROTATE WEIGHTS 180 độ: Lấy index ngược
                        // Forward weight: [oc][ic][i][j]
                        // Backward match: [oc][ic][k-1-i][k-1-j]
                        int w_idx = oc * (in_c * K_SIZE * K_SIZE) + ic * (K_SIZE * K_SIZE) + (K_SIZE - 1 - i) * K_SIZE + (K_SIZE - 1 - j);
                        sum += s_grad_out[ty + i][tx + j] * weights[w_idx];
                    }
                }
            }
            __syncthreads();
        }

        if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_i < in_h && col_i < in_w) {
            int in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + row_i * in_w + col_i;
            grad_input[in_idx] = sum;
        }
    }
}