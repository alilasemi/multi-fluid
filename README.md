# Prerequisites

The following Python packages are needed:

TODO: Add versions for these!
- Numpy
- Matplotlib
- Scipy
- Sympy
- Pybind11
- Rich

To download Pybind, use the following:

    conda install -c conda-forge pybind11

TODO: Add instructions (automated?) for adding Eigen support to gdb (the
.gdbinit file)

# Installing

Clone the repository:

    git clone git@github.com:alilasemi/level-set.git

Get the submodules:

    git submodule update --init

Compile:

    mkdir build
    cd build
    cmake ..
    make -j8

# Getting Started

Run the code with the command:

    python main.py
