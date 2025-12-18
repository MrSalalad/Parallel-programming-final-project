@echo off
echo Compiling Phase 1.3...
g++ -std=c++11 -O3 -I./include main.cpp src/cifar10_dataset.cpp src/layers.cpp src/autoencoder.cpp -o phase1.exe

if %ERRORLEVEL% EQU 0 (
    echo Compilation Successful! Running...
    echo -----------------------------------
    phase1.exe
) else (
    echo Compilation Failed!
)
pause