#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sys/time.h>

constexpr int n = 45000;
constexpr int m = 45000;

double wtime()
{
    timeval t{};
    gettimeofday(&t, nullptr);
    return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1e-6;
}

void get_chunk(int a, int b, int commsize, int rank, int &lb, int &ub)
{
    int N = b - a + 1;
    int q = N / commsize;
    if (N % commsize)
        q++;
    int r = commsize * q - N;

    int chunk = q;
    if (rank >= commsize - r)
        chunk = q - 1;

    lb = a;
    if (rank > 0)
    {
        if (rank <= commsize - r)
            lb += q * rank;
        else
            lb += q * (commsize - r) + (q - 1) * (rank - (commsize - r));
    }
    ub = lb + chunk - 1;
}

void sgemv(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c,
           int m_total, int n_total, int lb, int ub, MPI_Comm comm)
{
    int commsize, rank;
    MPI_Comm_size(comm, &commsize);
    MPI_Comm_rank(comm, &rank);

    int nrows = ub - lb + 1;

    for (int i = 0; i < nrows; ++i)
    {
        float sum = 0.0f;
        const float *row = &a[i * n_total];
        for (int j = 0; j < n_total; ++j)
            sum += row[j] * b[j];
        c[lb + i] = sum;
    }

    std::vector<int> rcounts(commsize), displs(commsize);
    for (int i = 0; i < commsize; ++i)
    {
        int l, u;
        get_chunk(0, m_total - 1, commsize, i, l, u);
        rcounts[i] = u - l + 1;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + rcounts[i - 1];
    }

    MPI_Allgatherv(MPI_IN_PLACE, nrows, MPI_FLOAT,
                   c.data(), rcounts.data(), displs.data(),
                   MPI_FLOAT, comm);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int commsize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t0 = wtime();

    int lb, ub;
    get_chunk(0, m - 1, commsize, rank, lb, ub);
    int nrows = ub - lb + 1;

    std::vector<float> A(static_cast<size_t>(nrows) * n);
    std::vector<float> B(static_cast<size_t>(n));
    std::vector<float> C(static_cast<size_t>(m));

    for (int i = 0; i < nrows; ++i)
    {
        float val = static_cast<float>(lb + i + 1);
        for (int j = 0; j < n; ++j)
            A[i * n + j] = val;
    }

    for (int j = 0; j < n; ++j)
        B[j] = static_cast<float>(j + 1);

    sgemv(A, B, C, m, n, lb, ub, MPI_COMM_WORLD);

    double elapsed = wtime() - t0;

    bool valid = true;
    if (rank == 0)
    {
        for (int i = 0; i < m; ++i)
        {
            double expected = (i + 1) * (n / 2.0 + std::pow(static_cast<double>(n), 2) / 2.0);
            if (std::fabs(C[i] - expected) > 1e-4)
            {
                std::cerr << "Validation failed at element " << i
                          << ": got " << C[i] << ", expected " << expected << "\n";
                valid = false;
                break;
            }
        }
    }

    if (rank == 0)
    {
        uint64_t mem = static_cast<uint64_t>((static_cast<double>(m) * n + m + n) * sizeof(float)) >> 20;
        double gflop = 2.0 * m * n * 1e-9;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "SGEMV: C[m] = A[m,n] * B[n]\n";
        std::cout << "Processes: " << commsize << "\n";
        std::cout << "Matrix size: " << m << " x " << n << "\n";
        std::cout << "Memory used per process: " << mem << " MiB\n";
        std::cout << "Elapsed time: " << elapsed << " s\n";
        std::cout << "Performance: " << (gflop / elapsed) << " GFLOPS\n";
        std::cout << (valid ? "Validation passed" : "Validation failed") << "\n";
    }

    MPI_Finalize();
    return 0;
}
