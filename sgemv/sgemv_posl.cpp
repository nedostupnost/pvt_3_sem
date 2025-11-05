#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <sys/time.h>

constexpr int n = 45000;
constexpr int m = 45000;

double wtime() {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 1e-6;
}

void sgemv_seq(const std::vector<float> &mat,
               const std::vector<float> &vec,
               std::vector<float> &res,
               int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const float* rowp = &mat[static_cast<size_t>(i) * cols];
        for (int j = 0; j < cols; ++j) {
            acc += rowp[j] * vec[j];
        }
        res[i] = acc;
    }
}

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);

    uint64_t bytes = (static_cast<uint64_t>(m) * n + m + n) * sizeof(float);
    uint64_t mem_mib = bytes >> 20;
    std::cout << "Memory used: " << mem_mib << " MiB\n";

    double t0 = wtime();

    std::vector<float> mat(static_cast<size_t>(m) * n);
    std::vector<float> vec(static_cast<size_t>(n));
    std::vector<float> res(static_cast<size_t>(m));

    for (int i = 0; i < m; ++i) {
        float val = static_cast<float>(i + 1);
        float* rowp = &mat[static_cast<size_t>(i) * n];
        for (int j = 0; j < n; ++j) {
            rowp[j] = val;
        }
    }

    for (int j = 0; j < n; ++j) {
        vec[j] = static_cast<float>(j + 1);
    }

    sgemv_seq(mat, vec, res, m, n);

    double elapsed = wtime() - t0;

    const double sum_j = static_cast<double>(n) * (n + 1) / 2.0;
    for (int i = 0; i < m; ++i) {
        double expected = (i + 1) * sum_j;
        if (std::fabs(res[i] - expected) > 1e-4) {
            std::cerr << "Validation failed: elem " << i
                      << " = " << res[i]
                      << " (expected " << expected
                      << "), diff = " << std::fabs(res[i] - expected) << '\n';
            break;
        }
    }

    double gflop = 2.0 * static_cast<double>(m) * static_cast<double>(n) * 1e-9;
    std::cout << "Elapsed time (serial): " << elapsed << " sec.\n";
    std::cout << "Performance: " << (gflop / elapsed) << " GFLOPS\n";

    return 0;
}
