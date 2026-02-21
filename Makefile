CXX      = g++
CXXFLAGS = -O3 -Wall -Wextra
SRC = src
OUT = output

all: 
	$(CXX) $(SRC)/background.cpp $(CXXFLAGS) -o background
	$(CXX) $(SRC)/perturbation.cpp $(CXXFLAGS) -o perturbation

background: 
	$(CXX) $(CXXFLAGS) background.cpp -o background

perturbation: 
	$(CXX) $(CXXFLAGS) -o perturbation perturbation.cpp

clean:
	rm -f  $(SRC)/background
	rm -f $(SRC)/perturbation

run:
	./background
	./perturbation
	mv *.dat $(OUT)/


