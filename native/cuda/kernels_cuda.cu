#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

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


bool cuda_runtime_available() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);

    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    return count > 0;
}


__global__ void rff_cosine_kernel(
    const double* X,
    const double* W,
    const double* b,
    double* Z,
    long long n,
    long long d,
    long long m,
    double scale
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = n * m;

    if (idx >= total) {
        return;
    }

    const long long i = idx / m;
    const long long j = idx % m;

    double acc = b[j];

    for (long long k = 0; k < d; ++k) {
        acc += X[i * d + k] * W[k * m + j];
    }

    Z[i * m + j] = scale * cos(acc);
}


__global__ void polynomial_kernel(
    const double* X,
    const double* W,
    const double* b,
    double* Z,
    long long n,
    long long d,
    long long m,
    int degree
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = n * m;

    if (idx >= total) {
        return;
    }

    const long long i = idx / m;
    const long long j = idx % m;

    double acc = b[j];

    for (long long k = 0; k < d; ++k) {
        acc += X[i * d + k] * W[k * m + j];
    }

    double value = tanh(acc);

    if (degree < 1) {
        degree = 1;
    }

    if (degree > 1) {
        const double sign = value >= 0.0 ? 1.0 : -1.0;
        value = sign * pow(abs(value), static_cast<double>(degree));
    }

    Z[i * m + j] = value / sqrt(static_cast<double>(m));
}


py::array_t<double> rff_cosine_transform_cuda(
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

    const long long n = static_cast<long long>(xb.shape[0]);
    const long long d = static_cast<long long>(xb.shape[1]);
    const long long wd = static_cast<long long>(wb.shape[0]);
    const long long m = static_cast<long long>(wb.shape[1]);

    if (d != wd) {
        throw std::invalid_argument("X.shape[1] must match W.shape[0].");
    }

    if (static_cast<long long>(bb.shape[0]) != m) {
        throw std::invalid_argument("b.shape[0] must match W.shape[1].");
    }

    const auto* h_X = static_cast<const double*>(xb.ptr);
    const auto* h_W = static_cast<const double*>(wb.ptr);
    const auto* h_b = static_cast<const double*>(bb.ptr);

    py::array_t<double> Z({xb.shape[0], wb.shape[1]});
    auto zb = Z.request();
    auto* h_Z = static_cast<double*>(zb.ptr);

    double* d_X = nullptr;
    double* d_W = nullptr;
    double* d_b = nullptr;
    double* d_Z = nullptr;

    const size_t bytes_X = static_cast<size_t>(n * d) * sizeof(double);
    const size_t bytes_W = static_cast<size_t>(d * m) * sizeof(double);
    const size_t bytes_b = static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_Z = static_cast<size_t>(n * m) * sizeof(double);

    check_cuda(cudaMalloc(&d_X, bytes_X), "cudaMalloc d_X failed");
    check_cuda(cudaMalloc(&d_W, bytes_W), "cudaMalloc d_W failed");
    check_cuda(cudaMalloc(&d_b, bytes_b), "cudaMalloc d_b failed");
    check_cuda(cudaMalloc(&d_Z, bytes_Z), "cudaMalloc d_Z failed");

    try {
        check_cuda(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice), "copy X failed");
        check_cuda(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice), "copy W failed");
        check_cuda(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice), "copy b failed");

        const long long total = n * m;
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        rff_cosine_kernel<<<blocks, threads>>>(d_X, d_W, d_b, d_Z, n, d, m, scale);

        check_cuda(cudaGetLastError(), "rff_cosine_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "rff_cosine_kernel sync failed");

        check_cuda(cudaMemcpy(h_Z, d_Z, bytes_Z, cudaMemcpyDeviceToHost), "copy Z failed");
    } catch (...) {
        cudaFree(d_X);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_Z);
        throw;
    }

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_Z);

    return Z;
}


py::array_t<double> polynomial_transform_cuda(
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

    const long long n = static_cast<long long>(xb.shape[0]);
    const long long d = static_cast<long long>(xb.shape[1]);
    const long long wd = static_cast<long long>(wb.shape[0]);
    const long long m = static_cast<long long>(wb.shape[1]);

    if (d != wd) {
        throw std::invalid_argument("X.shape[1] must match W.shape[0].");
    }

    if (static_cast<long long>(bb.shape[0]) != m) {
        throw std::invalid_argument("b.shape[0] must match W.shape[1].");
    }

    const auto* h_X = static_cast<const double*>(xb.ptr);
    const auto* h_W = static_cast<const double*>(wb.ptr);
    const auto* h_b = static_cast<const double*>(bb.ptr);

    py::array_t<double> Z({xb.shape[0], wb.shape[1]});
    auto zb = Z.request();
    auto* h_Z = static_cast<double*>(zb.ptr);

    double* d_X = nullptr;
    double* d_W = nullptr;
    double* d_b = nullptr;
    double* d_Z = nullptr;

    const size_t bytes_X = static_cast<size_t>(n * d) * sizeof(double);
    const size_t bytes_W = static_cast<size_t>(d * m) * sizeof(double);
    const size_t bytes_b = static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_Z = static_cast<size_t>(n * m) * sizeof(double);

    check_cuda(cudaMalloc(&d_X, bytes_X), "cudaMalloc d_X failed");
    check_cuda(cudaMalloc(&d_W, bytes_W), "cudaMalloc d_W failed");
    check_cuda(cudaMalloc(&d_b, bytes_b), "cudaMalloc d_b failed");
    check_cuda(cudaMalloc(&d_Z, bytes_Z), "cudaMalloc d_Z failed");

    try {
        check_cuda(cudaMemcpy(d_X, h_X, bytes_X, cudaMemcpyHostToDevice), "copy X failed");
        check_cuda(cudaMemcpy(d_W, h_W, bytes_W, cudaMemcpyHostToDevice), "copy W failed");
        check_cuda(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice), "copy b failed");

        const long long total = n * m;
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        polynomial_kernel<<<blocks, threads>>>(d_X, d_W, d_b, d_Z, n, d, m, degree);

        check_cuda(cudaGetLastError(), "polynomial_kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "polynomial_kernel sync failed");

        check_cuda(cudaMemcpy(h_Z, d_Z, bytes_Z, cudaMemcpyDeviceToHost), "copy Z failed");
    } catch (...) {
        cudaFree(d_X);
        cudaFree(d_W);
        cudaFree(d_b);
        cudaFree(d_Z);
        throw;
    }

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_Z);

    return Z;
}

}  // namespace mq_cenn_cuda
