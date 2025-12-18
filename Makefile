# Makefile for Windows (MinGW)
CXX = g++
# Lưu ý: -I./include để tìm file header
CXXFLAGS = -std=c++11 -O3 -I./include 

# File nguồn
SRC = main.cpp src/cifar10_dataset.cpp
# Tên file chạy
TARGET = phase1.exe

all: $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	del $(TARGET)