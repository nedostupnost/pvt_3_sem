#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdint>
#include <sys/time.h>

constexpr int n = 45000;
constexpr int m = 45000;

double wtime()
{
    timeval t{};
    gettimeofday(&t, nullptr);
    return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1e-6;
}

void dgemv(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            sum += a[i * n + j] * b[j];
        }
        c[i] = sum;
    }
}

int main()
{
    std::cout << std::fixed << std::setprecision(6);

    uint64_t memory_used = static_cast<uint64_t>(((static_cast<double>(m) * n + m + n) * sizeof(float))) >> 20;
    std::cout << "Memory used: " << memory_used << " MiB\n";

    double t = wtime();

    std::vector<float> a(static_cast<size_t>(m) * n);
    std::vector<float> b(static_cast<size_t>(n));
    std::vector<float> c(static_cast<size_t>(m));

    for (int i = 0; i < m; ++i)
    {
        float val = static_cast<float>(i + 1);
        for (int j = 0; j < n; ++j)
        {
            a[i * n + j] = val;
        }
    }

    for (int j = 0; j < n; ++j)
    {
        b[j] = static_cast<float>(j + 1);
    }

    dgemv(a, b, c, m, n);

    t = wtime() - t;

    for (int i = 0; i < m; ++i)
    {
        double expected = (i + 1) * (n / 2.0 + std::pow(static_cast<double>(n), 2) / 2.0);
        if (std::fabs(c[i] - expected) > 1e-6)
        {
            std::cerr << "Validation failed: elem " << i
                      << " = " << c[i]
                      << " (expected " << expected
                      << "), difference = " << std::fabs(c[i] - expected)
                      << '\n';
            break;
        }
    }

    double gflop = 2.0 * m * n * 1e-9;
    std::cout << "Elapsed time (serial): " << t << " sec.\n";
    std::cout << "Performance: " << (gflop / t) << " GFLOPS\n";

    return 0;
}
