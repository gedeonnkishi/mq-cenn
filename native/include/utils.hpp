#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

/**
 * Check whether a 1D float64 NumPy array contains only finite values.
 */
bool is_finite_1d(ArrayD x);

/**
 * Check whether a 2D float64 NumPy array contains only finite values.
 */
bool is_finite_2d(ArrayD x);

/**
 * Return the shape of a 1D NumPy array.
 */
py::tuple shape_1d(ArrayD x);

/**
 * Return the shape of a 2D NumPy array.
 */
py::tuple shape_2d(ArrayD x);

}  // namespace mq_cenn_native
