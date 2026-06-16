#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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


static std::vector<double> solve_linear_system(
    std::vector<double> A,
    std::vector<double> b,
    py::ssize_t n
) {
    const double eps = 1e-14;

    for (py::ssize_t col = 0; col < n; ++col) {
        py::ssize_t pivot = col;
        double best = std::abs(A[col * n + col]);

        for (py::ssize_t row = col + 1; row < n; ++row) {
            const double candidate = std::abs(A[row * n + col]);

            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }

        if (best < eps) {
            throw std::runtime_error("Singular ridge system in C++ backend.");
        }

        if (pivot != col) {
            for (py::ssize_t j = 0; j < n; ++j) {
                std::swap(A[col * n + j], A[pivot * n + j]);
            }

            std::swap(b[col], b[pivot]);
        }

        const double diag = A[col * n + col];

        for (py::ssize_t j = col; j < n; ++j) {
            A[col * n + j] /= diag;
        }

        b[col] /= diag;

        for (py::ssize_t row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }

            const double factor = A[row * n + col];

            if (factor == 0.0) {
                continue;
            }

            for (py::ssize_t j = col; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }

            b[row] -= factor * b[col];
        }
    }

    return b;
}


py::array_t<double> ridge_solve(
    ArrayD Z,
    ArrayD y,
    double alpha
) {
    auto zb = Z.request();
    auto yb = y.request();

    require_2d(zb, "Z");
    require_1d(yb, "y");

    const py::ssize_t n = zb.shape[0];
    const py::ssize_t p = zb.shape[1];

    if (yb.shape[0] != n) {
        throw std::invalid_argument("Z and y length mismatch.");
    }

    if (alpha <= 0.0) {
        throw std::invalid_argument("alpha must be positive.");
    }

    auto z = Z.unchecked<2>();
    auto target = y.unchecked<1>();

    std::vector<double> A(static_cast<size_t>(p * p), 0.0);
    std::vector<double> rhs(static_cast<size_t>(p), 0.0);

    for (py::ssize_t j = 0; j < p; ++j) {
        for (py::ssize_t k = 0; k < p; ++k) {
            double acc = 0.0;

            for (py::ssize_t i = 0; i < n; ++i) {
                acc += z(i, j) * z(i, k);
            }

            A[static_cast<size_t>(j * p + k)] = acc;
        }

        A[static_cast<size_t>(j * p + j)] += alpha;
    }

    for (py::ssize_t j = 0; j < p; ++j) {
        double acc = 0.0;

        for (py::ssize_t i = 0; i < n; ++i) {
            acc += z(i, j) * target(i);
        }

        rhs[static_cast<size_t>(j)] = acc;
    }

    std::vector<double> beta = solve_linear_system(A, rhs, p);

    py::array_t<double> out({p});
    auto result = out.mutable_unchecked<1>();

    for (py::ssize_t j = 0; j < p; ++j) {
        result(j) = beta[static_cast<size_t>(j)];
    }

    return out;
}


py::array_t<double> linear_predict(
    ArrayD Z,
    ArrayD beta
) {
    auto zb = Z.request();
    auto bb = beta.request();

    require_2d(zb, "Z");
    require_1d(bb, "beta");

    const py::ssize_t n = zb.shape[0];
    const py::ssize_t p = zb.shape[1];

    if (bb.shape[0] != p) {
        throw std::invalid_argument("beta length must match Z.shape[1].");
    }

    auto z = Z.unchecked<2>();
    auto b = beta.unchecked<1>();

    py::array_t<double> out({n});
    auto pred = out.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < n; ++i) {
        double acc = 0.0;

        for (py::ssize_t j = 0; j < p; ++j) {
            acc += z(i, j) * b(j);
        }

        pred(i) = acc;
    }

    return out;
}

}  // namespace mq_cenn_native
