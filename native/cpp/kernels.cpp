#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;


static void require_2d(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 2) {
        throw std::invalid_argument(name + " must be a 2D float64 array.");
    }
}


static void require_1d(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 1) {
        throw std::invalid_argument(name + " must be a 1D float64 array.");
    }
}


py::array_t<double> rff_cosine_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    double scale
) {
    auto xb = X.request();
    auto wb = W.request();
    auto bb = b.request();

    require_2d(xb, "X");
    require_2d(wb, "W");
    require_1d(bb, "b");

    const py::ssize_t n = xb.shape[0];
    const py::ssize_t d = xb.shape[1];
    const py::ssize_t wd = wb.shape[0];
    const py::ssize_t m = wb.shape[1];

    if (d != wd) {
        throw std::invalid_argument("X.shape[1] must match W.shape[0].");
    }

    if (bb.shape[0] != m) {
        throw std::invalid_argument("b.shape[0] must match W.shape[1].");
    }

    auto x = X.unchecked<2>();
    auto w = W.unchecked<2>();
    auto bias = b.unchecked<1>();

    py::array_t<double> Z({n, m});
    auto z = Z.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < m; ++j) {
            double acc = bias(j);

            for (py::ssize_t k = 0; k < d; ++k) {
                acc += x(i, k) * w(k, j);
            }

            z(i, j) = scale * std::cos(acc);
        }
    }

    return Z;
}


py::array_t<double> polynomial_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    int degree
) {
    auto xb = X.request();
    auto wb = W.request();
    auto bb = b.request();

    require_2d(xb, "X");
    require_2d(wb, "W");
    require_1d(bb, "b");

    const py::ssize_t n = xb.shape[0];
    const py::ssize_t d = xb.shape[1];
    const py::ssize_t wd = wb.shape[0];
    const py::ssize_t m = wb.shape[1];

    if (d != wd) {
        throw std::invalid_argument("X.shape[1] must match W.shape[0].");
    }

    if (bb.shape[0] != m) {
        throw std::invalid_argument("b.shape[0] must match W.shape[1].");
    }

    if (degree < 1) {
        degree = 1;
    }

    auto x = X.unchecked<2>();
    auto w = W.unchecked<2>();
    auto bias = b.unchecked<1>();

    py::array_t<double> Z({n, m});
    auto z = Z.mutable_unchecked<2>();

    const double norm = std::sqrt(static_cast<double>(m));

    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < m; ++j) {
            double acc = bias(j);

            for (py::ssize_t k = 0; k < d; ++k) {
                acc += x(i, k) * w(k, j);
            }

            double value = std::tanh(acc);

            if (degree > 1) {
                const double sign = value >= 0.0 ? 1.0 : -1.0;
                value = sign * std::pow(std::abs(value), degree);
            }

            z(i, j) = value / norm;
        }
    }

    return Z;
}

}  // namespace mq_cenn_native
