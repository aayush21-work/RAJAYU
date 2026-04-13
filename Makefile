CXX = g++
CXXFLAGS = -O3

TARGET = solver
SRCS = main.cpp

all: $(TARGET)
	./$(TARGET)
	python plot_all.py background_nmdc.dat

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) background_nmdc.dat

.PHONY: all clean
