CXX      = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
OMP      = -fopenmp

TARGET   = nmdc_solver
SRC      = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(OMP) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) background.dat power_spectrum.dat

.PHONY: all clean
