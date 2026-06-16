#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

bool is_finite_1d(ArrayD x);
bool is_finite_2d(ArrayD x);
py::tuple shape_1d(ArrayD x);
py::tuple shape_2d(ArrayD x);

py::array_t<double> rff_cosine_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    double scale
);

py::array_t<double> polynomial_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    int degree
);

py::array_t<double> ridge_solve(
    ArrayD Z,
    ArrayD y,
    double alpha
);

py::array_t<double> linear_predict(
    ArrayD Z,
    ArrayD beta
);

}  // namespace mq_cenn_native


PYBIND11_MODULE(_cpp_backend, m) {
    m.doc() = "Native C++ CPU backend for MQ-CeNN.";

    m.def(
        "is_finite_1d",
        &mq_cenn_native::is_finite_1d,
        "Check whether a 1D float64 array contains only finite values."
    );

    m.def(
        "is_finite_2d",
        &mq_cenn_native::is_finite_2d,
        "Check whether a 2D float64 array contains only finite values."
    );

    m.def(
        "shape_1d",
        &mq_cenn_native::shape_1d,
        "Return the shape of a 1D array."
    );

    m.def(
        "shape_2d",
        &mq_cenn_native::shape_2d,
        "Return the shape of a 2D array."
    );

    m.def(
        "rff_cosine_transform",
        &mq_cenn_native::rff_cosine_transform,
        "Compute cosine random-feature transform: scale * cos(XW + b)."
    );

    m.def(
        "polynomial_transform",
        &mq_cenn_native::polynomial_transform,
        "Compute bounded polynomial random-feature transform."
    );

    m.def(
        "ridge_solve",
        &mq_cenn_native::ridge_solve,
        "Solve ridge regression normal equations."
    );

    m.def(
        "linear_predict",
        &mq_cenn_native::linear_predict,
        "Compute linear predictions Z @ beta."
    );
}
