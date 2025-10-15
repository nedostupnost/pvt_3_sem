#include <mpi.h>
#include <iostream>
#include <cmath>
#include <iomanip>

double f(double x) {
    return sqrt(x * (3.0 - x) / (x + 1.0));
}

double midpoint_rule(double a, double b, long long n, int rank, int size) {
    double h = (b - a) / n;
    long long local_n = n / size;
    long long start = rank * local_n;
    long long end = (rank == size - 1) ? n : (rank + 1) * local_n;
    
    double local_sum = 0.0;
    for (long long i = start; i < end; i++) {
        double x_mid = a + (i + 0.5) * h;
        local_sum += f(x_mid);
    }
    
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    return global_sum * h;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double a = 1.0;
    double b = 1.2;
    double epsilon = 1e-6;
    
    // Начальное число разбиений
    long long n = 100;
    
    double I_n, I_2n;
    double error;
    int iterations = 0;
    
    double start_time = MPI_Wtime();

    I_n = midpoint_rule(a, b, n, rank, size);
    
    do {
        n *= 2;
        I_2n = midpoint_rule(a, b, n, rank, size);
        
        error = fabs(I_2n - I_n) / 3.0;
        
        I_n = I_2n;
        iterations++;
        
        if (rank == 0) {
            std::cout << "Iteration " << iterations 
                      << ": n = " << n 
                      << ", I = " << std::setprecision(10) << I_2n 
                      << ", error = " << std::scientific << error << std::endl;
        }
        
    } while (error > epsilon);
    
    double end_time = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Processes: " << size << std::endl;
        std::cout << "Final integral value: " << std::setprecision(10) << std::fixed << I_2n << std::endl;
        std::cout << "Final error estimate: " << std::scientific << error << std::endl;
        std::cout << "Number of iterations: " << iterations << std::endl;
        std::cout << "Final n: " << n << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(6) 
                  << (end_time - start_time) << " seconds" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
