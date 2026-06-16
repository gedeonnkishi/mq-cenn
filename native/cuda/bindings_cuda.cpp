#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mq_cenn_cuda {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

bool cuda_runtime_available();

py::array_t<double> rff_cosine_transform_cuda(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    double scale
);

py::array_t<double> polynomial_transform_cuda(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    int degree
);

py::array_t<double> ridge_solve_cuda(
    ArrayD Z,
    ArrayD y,
    double alpha
);

py::array_t<double> linear_predict_cuda(
    ArrayD Z,
    ArrayD beta
);

}  // namespace mq_cenn_cuda


PYBIND11_MODULE(_cuda_backend, m) {
    m.doc() = "Native CUDA GPU backend for MQ-CeNN.";

    m.def(
        "cuda_runtime_available",
        &mq_cenn_cuda::cuda_runtime_available,
        "Return True if the CUDA runtime can see at least one CUDA device."
    );

    m.def(
        "rff_cosine_transform",
        &mq_cenn_cuda::rff_cosine_transform_cuda,
        "CUDA cosine random-feature transform: scale * cos(XW + b)."
    );

    m.def(
        "polynomial_transform",
        &mq_cenn_cuda::polynomial_transform_cuda,
        "CUDA bounded polynomial random-feature transform."
    );

    m.def(
        "ridge_solve",
        &mq_cenn_cuda::ridge_solve_cuda,
        "CUDA-assisted ridge regression solver."
    );

    m.def(
        "linear_predict",
        &mq_cenn_cuda::linear_predict_cuda,
        "CUDA linear prediction: Z @ beta."
    );
}
