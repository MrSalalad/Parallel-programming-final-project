@echo off
echo Compiling Phase 2 (Naive GPU)...

REM Đảm bảo đường dẫn include
REM nvcc biên dịch code .cu và liên kết với code .cpp (cần compile riêng hoặc include)
REM Ở đây mình compile chung cho đơn giản

nvcc -O3 -I./include src/gpu/kernels_naive.cu -c -o src/gpu/kernels_naive.o
if %ERRORLEVEL% NEQ 0 goto error

echo Kernel compilation successful.
echo (Chua co main_gpu.cu nen dung tai day de kiem tra moi truong)

goto end

:error
echo Compilation Failed!

:end
pause