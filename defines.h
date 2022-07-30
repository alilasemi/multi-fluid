#ifndef defines_h
#define defines_h

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
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

#endif
