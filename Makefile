#TODO: Unhack these
python='${HOME}/software/anaconda3/bin/python'
# Compiler
CXX = g++

# Directories
src_dir = cache
build_dir = $(src_dir)/build
# Files
src = $(wildcard $(src_dir)/*.cpp)
lib = $(src:$(src_dir)/%.cpp=$(build_dir)/%.so)
# Paths to includes
include_paths = ./ eigen/
# Compiler flags
optimization = -O3
flags = $(foreach dir, $(include_paths), -I$(dir)) -std=c++17 $(optimization) \
		-Wall -shared -fPIC $(shell $(python) -m pybind11 --includes)
# Libraries and locations
ldlibs =
# Useful variables
empty =

.PHONY: all
all: directories $(lib) $(build_dir)/interior_face_residual.so $(build_dir)/roe.so $(build_dir)/forced_interface.so

# This is purely for testing purposes
.PHONY: print
print:
	$(info $(patsubst $(test_obj_dir)/%_test.o,$(obj_dir)/%.o,$(test_obj)))

.PHONY: directories
directories:
	if [ ! -d $(build_dir) ]; then mkdir $(build_dir); fi

.PHONY: clean
clean:
	rm -rf $(src_dir) $(build_dir)

$(build_dir)/%.so: $(src_dir)/%.cpp
	$(CXX) $(flags) -o $@ $^

$(build_dir)/interior_face_residual.so: interior_face_residual.cpp
	$(CXX) $(flags) -o $@ $^

$(build_dir)/roe.so: roe.cpp
	$(CXX) $(flags) -o $@ $^

$(build_dir)/forced_interface.so: forced_interface.cpp
	$(CXX) $(flags) -o $@ $^
