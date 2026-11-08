CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall

infer: infer.cpp
	$(CXX) $(CXXFLAGS) infer.cpp -o infer

run: infer
	./infer

clean:
	rm -f infer

.PHONY: run clean
