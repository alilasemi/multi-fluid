#ifndef defines_h
#define defines_h

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using Eigen::placeholders::all;
using Eigen::seq;

// Custom types for 1D vectors and 2D matrices in Eigen. It is important to
// specify RowMajor since the default of Eigen is column major
template <class T> using vector = Eigen::Vector<
        T, Eigen::Dynamic>;
template <class T> using matrix = Eigen::Matrix<
        T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// Custom type for 2D matrices using Eigen maps
template <class T> using matrix_map = Eigen::Map<
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
// Custom type for matrices using Eigen::Ref. This must be used when passing in
// Numpy arrays as function arguments.
template <class T> using matrix_ref = Eigen::Ref<matrix<T>>;
// Custom type for Numpy arrays
template <class T> using np_array = py::array_t<T, py::array::c_style>;

// Convert a Numpy array into a mapped Eigen matrix.
template <class T>
matrix_map<T> numpy_to_eigen(np_array<T> A) {
    // Get pointer
    T* A_ptr = (T*) A.request().ptr;
    // Get shape
    auto shape = A.request().shape;
    // Check if 1D or 2D
    int rows = shape[0];
    int cols = 1;
    if (shape.size() == 2) {
        cols = shape[1];
    }
    // Turn into Eigen map
    matrix_map<T> A_map(A_ptr, rows, cols);
    return A_map;
}

#endif
