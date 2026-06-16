#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;


static bool all_finite(const double* data, py::ssize_t size) {
    for (py::ssize_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }

    return true;
}


bool is_finite_1d(ArrayD x) {
    auto info = x.request();

    if (info.ndim != 1) {
        throw std::invalid_argument("x must be a 1D float64 array.");
    }

    const auto* ptr = static_cast<const double*>(info.ptr);

    return all_finite(ptr, info.shape[0]);
}


bool is_finite_2d(ArrayD x) {
    auto info = x.request();

    if (info.ndim != 2) {
        throw std::invalid_argument("x must be a 2D float64 array.");
    }

    const auto* ptr = static_cast<const double*>(info.ptr);
    const py::ssize_t size = info.shape[0] * info.shape[1];

    return all_finite(ptr, size);
}


py::tuple shape_1d(ArrayD x) {
    auto info = x.request();

    if (info.ndim != 1) {
        throw std::invalid_argument("x must be a 1D float64 array.");
    }

    return py::make_tuple(info.shape[0]);
}


py::tuple shape_2d(ArrayD x) {
    auto info = x.request();

    if (info.ndim != 2) {
        throw std::invalid_argument("x must be a 2D float64 array.");
    }

    return py::make_tuple(info.shape[0], info.shape[1]);
}

}  // namespace mq_cenn_native
