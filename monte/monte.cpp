#include <mpi.h>
#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>

double f(double x, double y) {
    return exp(x - y);
}

double monte_carlo_integration(double x_min, double x_max, 
                               double y_min, double y_max,
                               long long n_total, int rank, int size) {
    double area = (x_max - x_min) * (y_max - y_min);

    long long n_local = n_total / size;
    long long remainder = n_total % size;

    if (rank < remainder) {
        n_local++;
    }

    std::random_device rd;
    std::mt19937_64 gen(rd() + rank * 12345);
    std::uniform_real_distribution<double> dist_x(x_min, x_max);
    std::uniform_real_distribution<double> dist_y(y_min, y_max);

    double local_sum = 0.0;
    for (long long i = 0; i < n_local; i++) {
        double x = dist_x(gen);
        double y = dist_y(gen);
        local_sum += f(x, y);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        return (area * global_sum) / n_total;
    }
    
    return 0.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double x_min = -1.0;
    double x_max = 0.0;
    double y_min = 0.0;
    double y_max = 1.0;
    

    long long n_points = 10000000;
    
    if (argc > 1) {
        n_points = std::atoll(argv[1]);
    }
    
    if (rank == 0) {
        std::cout << "=== Monte Carlo Integration ===" << std::endl;
        std::cout << "Function: f(x,y) = exp(x-y)" << std::endl;
        std::cout << "Domain: x ∈ [" << x_min << ", " << x_max << "], "
                  << "y ∈ [" << y_min << ", " << y_max << "]" << std::endl;
        std::cout << "Number of processes: " << size << std::endl;
        std::cout << "Number of points: " << n_points << std::endl;
        std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    double result = monte_carlo_integration(x_min, x_max, y_min, y_max, 
                                           n_points, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "=== Results ===" << std::endl;
        std::cout << "Integral value: " << std::setprecision(10) << std::fixed 
                  << result << std::endl;
        std::cout << "Execution time: " << std::setprecision(6) 
                  << (end_time - start_time) << " seconds" << std::endl;
        std::cout << "Points per second: " << std::scientific 
                  << n_points / (end_time - start_time) << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
