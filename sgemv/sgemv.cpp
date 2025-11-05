#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sys/time.h>

constexpr int n = 45000;
constexpr int m = 45000;

double wtime() {
    timeval t{};
    gettimeofday(&t, nullptr);
    return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) * 1e-6;
}

inline void split_range_blocked(int a, int b, int nprocs, int proc_id, int &lo, int &hi) {
    const int N = b - a + 1;
    const int q = N / nprocs;
    const int r = N % nprocs;
    if (proc_id < r) {
        lo = a + proc_id * (q + 1);
        hi = lo + (q + 1) - 1;
    } else {
        lo = a + r * (q + 1) + (proc_id - r) * q;
        hi = lo + q - 1;
    }
}

void sgemv_rows(const std::vector<float> &mat, const std::vector<float> &vec,
                std::vector<float> &out, int m_total, int n_total,
                int row_lo, int row_hi, MPI_Comm comm)
{
    int nprocs = 1, proc_id = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &proc_id);

    const int local_rows = row_hi - row_lo + 1;

    for (int i = 0; i < local_rows; ++i) {
        const float* rowp = &mat[static_cast<size_t>(i) * n_total];
        float acc = 0.0f;
        for (int j = 0; j < n_total; ++j) {
            acc += rowp[j] * vec[j];
        }
        out[row_lo + i] = acc;
    }

    std::vector<int> seg_counts(nprocs), seg_offsets(nprocs);
    for (int r = 0; r < nprocs; ++r) {
        int lo, hi;
        split_range_blocked(0, m_total - 1, nprocs, r, lo, hi);
        seg_counts[r]  = hi - lo + 1;
        seg_offsets[r] = lo;
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   out.data(), seg_counts.data(), seg_offsets.data(),
                   MPI_FLOAT, comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int nprocs = 1, proc_id = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

    double t0 = wtime();

    int row_lo = 0, row_hi = -1;
    split_range_blocked(0, m - 1, nprocs, proc_id, row_lo, row_hi);
    const int local_rows = row_hi - row_lo + 1;

    std::vector<float> mat(static_cast<size_t>(local_rows) * n);
    std::vector<float> vec(static_cast<size_t>(n));
    std::vector<float> out(static_cast<size_t>(m));

    for (int i = 0; i < local_rows; ++i) {
        float val = static_cast<float>(row_lo + i + 1);
        float* rowp = &mat[static_cast<size_t>(i) * n];
        for (int j = 0; j < n; ++j) rowp[j] = val;
    }

    for (int j = 0; j < n; ++j) vec[j] = static_cast<float>(j + 1);

    sgemv_rows(mat, vec, out, m, n, row_lo, row_hi, MPI_COMM_WORLD);

    double elapsed = wtime() - t0;

    bool valid = true;
    if (proc_id == 0) {
        const double sum_j = static_cast<double>(n) * (n + 1) / 2.0;
        for (int i = 0; i < m; ++i) {
            double expected = (i + 1) * sum_j;
            if (std::fabs(out[i] - expected) > 1e-3) {
                std::cerr << "Validation failed at element " << i
                          << ": got " << out[i] << ", expected " << expected << "\n";
                valid = false;
                break;
            }
        }
    }

    if (proc_id == 0) {
        uint64_t mem_local_bytes =
            (static_cast<uint64_t>(local_rows) * n + m + n) * sizeof(float);
        double mem_local_mib = static_cast<double>(mem_local_bytes) / (1024.0 * 1024.0);

        double gflop = 2.0 * static_cast<double>(m) * static_cast<double>(n) * 1e-9;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "SGEMV: C[m] = A[m,n] * B[n]\n";
        std::cout << "Processes: " << nprocs << "\n";
        std::cout << "Matrix size: " << m << " x " << n << "\n";
        std::cout << "Memory used per process: " << mem_local_mib << " MiB\n";
        std::cout << "Elapsed time: " << elapsed << " s\n";
        std::cout << "Performance: " << (gflop / elapsed) << " GFLOPS\n";
        std::cout << (valid ? "Validation passed" : "Validation failed") << "\n";
    }

    MPI_Finalize();
    return 0;
}
