CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic
LDFLAGS  ?= -ldl

.PHONY: all test clean

all: kern-run

kern-run: run.cpp executor.hpp manifest.hpp cuda_api.hpp json.hpp
	$(CXX) $(CXXFLAGS) -o $@ run.cpp $(LDFLAGS)

test_manifest: test_manifest.cpp executor.hpp manifest.hpp cuda_api.hpp json.hpp
	$(CXX) $(CXXFLAGS) -o $@ test_manifest.cpp $(LDFLAGS)

test: test_manifest kern-run
	./test_manifest

clean:
	rm -f kern-run test_manifest
