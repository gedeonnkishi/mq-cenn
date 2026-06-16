#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

/**
 * Solve the ridge regression system:
 *
 *     beta = argmin ||Z beta - y||² + alpha ||beta||²
 *
 * Internally, this computes:
 *
 *     beta = (ZᵀZ + alpha I)^(-1) Zᵀy
 *
 * Parameters
 * ----------
 * Z:
 *     Feature matrix with shape (n_samples, n_features).
 *
 * y:
 *     Target vector with shape (n_samples,).
 *
 * alpha:
 *     Positive ridge regularization coefficient.
 *
 * Returns
 * -------
 * py::array_t<double>
 *     Coefficient vector beta with shape (n_features,).
 */
py::array_t<double> ridge_solve(
    ArrayD Z,
    ArrayD y,
    double alpha
);

/**
 * Compute linear predictions:
 *
 *     pred = Z @ beta
 *
 * Parameters
 * ----------
 * Z:
 *     Feature matrix with shape (n_samples, n_features).
 *
 * beta:
 *     Coefficient vector with shape (n_features,).
 *
 * Returns
 * -------
 * py::array_t<double>
 *     Prediction vector with shape (n_samples,).
 */
py::array_t<double> linear_predict(
    ArrayD Z,
    ArrayD beta
);

}  // namespace mq_cenn_native
