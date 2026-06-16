#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace mq_cenn_cuda {

using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;


static void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(status)
        );
    }
}


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


__global__ void gram_matrix_kernel(
    const double* Z,
    double* A,
    long long n,
    long long p
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = p * p;

    if (idx >= total) {
        return;
    }

    const long long row = idx / p;
    const long long col = idx % p;

    double acc = 0.0;

    for (long long i = 0; i < n; ++i) {
        acc += Z[i * p + row] * Z[i * p + col];
    }

    A[row * p + col] = acc;
}


__global__ void aty_kernel(
    const double* Z,
    const double* y,
    double* rhs,
    long long n,
    long long p
) {
    const long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= p) {
        return;
    }

    double acc = 0.0;

    for (long long i = 0; i < n; ++i) {
        acc += Z[i * p + j] * y[i];
    }

    rhs[j] = acc;
}


__global__ void add_ridge_kernel(
    double* A,
    long long p,
    double alpha
) {
    const long long j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= p) {
        return;
    }

    A[j * p + j] += alpha;
}


__global__ void linear_predict_kernel(
    const double* Z,
    const double* beta,
    double* pred,
    long long n,
    long long p
) {
    const long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    double acc = 0.0;

    for (long long j = 0; j < p; ++j) {
        acc += Z[i * p + j] * beta[j];
    }

    pred[i] = acc;
}


static std::vector<double> solve_linear_system_cpu(
    std::vector<double> A,
    std::vector<double> b,
    long long n
) {
    const double eps = 1e-14;

    for (long long col = 0; col < n; ++col) {
        long long pivot = col;
        double best = std::abs(A[static_cast<size_t>(col * n + col)]);

        for (long long row = col + 1; row < n; ++row) {
            const double candidate = std::abs(
                A[static_cast<size_t>(row * n + col)]
            );

            if (candidate > best) {
                best = candidate;
                pivot = row;
            }
        }

        if (best < eps) {
            throw std::runtime_error("Singular ridge system in CUDA backend.");
        }

        if (pivot != col) {
            for (long long j = 0; j < n; ++j) {
                std::swap(
                    A[static_cast<size_t>(col * n + j)],
                    A[static_cast<size_t>(pivot * n + j)]
                );
            }

            std::swap(
                b[static_cast<size_t>(col)],
                b[static_cast<size_t>(pivot)]
            );
        }

        const double diag = A[static_cast<size_t>(col * n + col)];

        for (long long j = col; j < n; ++j) {
            A[static_cast<size_t>(col * n + j)] /= diag;
        }

        b[static_cast<size_t>(col)] /= diag;

        for (long long row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }

            const double factor = A[static_cast<size_t>(row * n + col)];

            if (factor == 0.0) {
                continue;
            }

            for (long long j = col; j < n; ++j) {
                A[static_cast<size_t>(row * n + j)] -=
                    factor * A[static_cast<size_t>(col * n + j)];
            }

            b[static_cast<size_t>(row)] -=
                factor * b[static_cast<size_t>(col)];
        }
    }

    return b;
}


py::array_t<double> ridge_solve_cuda(
    ArrayD Z,
    ArrayD y,
    double alpha
) {
    auto zb = Z.request();
    auto yb = y.request();

    require_2d(zb, "Z");
    require_1d(yb, "y");

    const long long n = static_cast<long long>(zb.shape[0]);
    const long long p = static_cast<long long>(zb.shape[1]);

    if (static_cast<long long>(yb.shape[0]) != n) {
        throw std::invalid_argument("Z and y length mismatch.");
    }

    if (alpha <= 0.0) {
        throw std::invalid_argument("alpha must be positive.");
    }

    const auto* h_Z = static_cast<const double*>(zb.ptr);
    const auto* h_y = static_cast<const double*>(yb.ptr);

    double* d_Z = nullptr;
    double* d_y = nullptr;
    double* d_A = nullptr;
    double* d_rhs = nullptr;

    const size_t bytes_Z = static_cast<size_t>(n * p) * sizeof(double);
    const size_t bytes_y = static_cast<size_t>(n) * sizeof(double);
    const size_t bytes_A = static_cast<size_t>(p * p) * sizeof(double);
    const size_t bytes_rhs = static_cast<size_t>(p) * sizeof(double);

    check_cuda(cudaMalloc(&d_Z, bytes_Z), "cudaMalloc d_Z failed");
    check_cuda(cudaMalloc(&d_y, bytes_y), "cudaMalloc d_y failed");
    check_cuda(cudaMalloc(&d_A, bytes_A), "cudaMalloc d_A failed");
    check_cuda(cudaMalloc(&d_rhs, bytes_rhs), "cudaMalloc d_rhs failed");

    std::vector<double> h_A(static_cast<size_t>(p * p), 0.0);
    std::vector<double> h_rhs(static_cast<size_t>(p), 0.0);

    try {
        check_cuda(cudaMemcpy(d_Z, h_Z, bytes_Z, cudaMemcpyHostToDevice), "copy Z failed");
        check_cuda(cudaMemcpy(d_y, h_y, bytes_y, cudaMemcpyHostToDevice), "copy y failed");

        const int threads = 256;

        const long long total_A = p * p;
        const int blocks_A = static_cast<int>((total_A + threads - 1) / threads);

        gram_matrix_kernel<<<blocks_A, threads>>>(d_Z, d_A, n, p);

        check_cuda(cudaGetLastError(), "gram_matrix_kernel launch failed");

        const int blocks_p = static_cast<int>((p + threads - 1) / threads);

        add_ridge_kernel<<<blocks_p, threads>>>(d_A, p, alpha);
        check_cuda(cudaGetLastError(), "add_ridge_kernel launch failed");

        aty_kernel<<<blocks_p, threads>>>(d_Z, d_y, d_rhs, n, p);
        check_cuda(cudaGetLastError(), "aty_kernel launch failed");

        check_cuda(cudaDeviceSynchronize(), "ridge kernels sync failed");

        check_cuda(cudaMemcpy(h_A.data(), d_A, bytes_A, cudaMemcpyDeviceToHost), "copy A failed");
        check_cuda(cudaMemcpy(h_rhs.data(), d_rhs, bytes_rhs, cudaMemcpyDeviceToHost), "copy rhs failed");
    } catch (...) {
        cudaFree(d_Z);
        cudaFree(d_y);
        cudaFree(d_A);
        cudaFree(d_rhs);
        throw;
    }

    cudaFree(d_Z);
    cudaFree(d_y);
    cudaFree(d_A);
    cudaFree(d_rhs);

    std::vector<double> beta = solve_linear_system_cpu(h_A, h_rhs, p);

    py::array_t<double> out({p});
    auto ob = out.request();
    auto* h_beta = static_cast<double*>(ob.ptr);

    for (long long j = 0; j < p; ++j) {
        h_beta[j] = beta[static_cast<size_t>(j)];
    }

    return out;
}


py::array_t<double> linear_predict_cuda(
    ArrayD Z,
    ArrayD beta
) {
    auto zb = Z.request();
    auto bb = beta.request();

    require_2d(zb, "Z");
    require_1d(bb, "beta");

    const long long n = static_cast<long long>(zb.shape[0]);
    const long long p = static_cast<long long>(zb.shape[1]);

    if (static_cast<long long>(bb.shape[0]) != p) {
        throw std::invalid_argument("beta length must match Z.shape[1].");
    }

    const auto* h_Z = static_cast<const double*>(zb.ptr);
    const auto* h_beta = static_cast<const double*>(bb.ptr);

    py::array_t<double> out({zb.shape[0]});
    auto ob = out.request();
    auto* h_pred = static_cast<double*>(ob.ptr);

    double* d_Z = nullptr;
    double* d_beta = nullptr;
    double* d_pred = nullptr;

    const size_t bytes_Z = static_cast<size_t>(n * p) * sizeof(double);
    const size_t bytes_beta = static_cast<size_t>(p) * sizeof(double);
    const size_t bytes_pred = static_cast<size_t>(n) * sizeof(double);

    check_cuda(cudaMalloc(&d_Z, bytes_Z), "cudaMalloc d_Z failed");
    check_cuda(cudaMalloc(&d_beta, bytes_beta), "cudaMalloc d_beta failed");
    check_cuda(cudaMalloc(&d_pred, bytes_pred), "cudaMalloc d_pred failed");

    try {
        check_cuda(cudaMemcpy(d_Z, h_Z, bytes_Z, cudaMemcpyHostToDevice), "copy Z failed");
        check_cuda(cudaMemcpy(d_beta, h_beta, bytes_beta, cudaMemcpyHostToDevice), "copy beta failed");

        const int threads = 256;
        const int blocks = static_cast<int>((n + threads - 1) / threads);

        linear_predict_kernel<<<blocks, threads>>>(d_Z, d_beta, d_pred, n, p);

        check_cuda(cudaGetLastError(), "linear_predict_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "linear_predict_kernel sync failed");

        check_cuda(cudaMemcpy(h_pred, d_pred, bytes_pred, cudaMemcpyDeviceToHost), "copy pred failed");
    } catch (...) {
        cudaFree(d_Z);
        cudaFree(d_beta);
        cudaFree(d_pred);
        throw;
    }

    cudaFree(d_Z);
    cudaFree(d_beta);
    cudaFree(d_pred);

    return out;
}

}  // namespace mq_cenn_cuda
