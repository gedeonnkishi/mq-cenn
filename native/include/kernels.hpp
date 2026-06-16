#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mq_cenn_native {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;

/**
 * Compute the cosine random-feature transform:
 *
 *     Z = scale * cos(XW + b)
 *
 * Parameters
 * ----------
 * X:
 *     Input matrix with shape (n_samples, n_features).
 *
 * W:
 *     Projection matrix with shape (n_features, n_random_features).
 *
 * b:
 *     Bias vector with shape (n_random_features,).
 *
 * scale:
 *     Output scaling factor.
 *
 * Returns
 * -------
 * py::array_t<double>
 *     Transformed feature matrix with shape
 *     (n_samples, n_random_features).
 */
py::array_t<double> rff_cosine_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    double scale
);

/**
 * Compute bounded polynomial random-feature transform.
 *
 * The operation is equivalent to:
 *
 *     z = tanh(XW + b)
 *     z = sign(z) * |z|^degree
 *     z = z / sqrt(n_random_features)
 *
 * Parameters
 * ----------
 * X:
 *     Input matrix with shape (n_samples, n_features).
 *
 * W:
 *     Projection matrix with shape (n_features, n_random_features).
 *
 * b:
 *     Bias vector with shape (n_random_features,).
 *
 * degree:
 *     Polynomial degree. Values lower than 1 are treated as 1.
 *
 * Returns
 * -------
 * py::array_t<double>
 *     Transformed feature matrix.
 */
py::array_t<double> polynomial_transform(
    ArrayD X,
    ArrayD W,
    ArrayD b,
    int degree
);

}  // namespace mq_cenn_native
