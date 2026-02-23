#include <math.h>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Helper to print a matrix
void print_matrix(const char *name, int m, int n, double *A, int lda) {
  printf("%s:\n", name);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%8.4f ", A[i + j * lda]);
    }
    printf("\n");
  }
}

// Helper to extract the R factor (upper triangular part) from a QR result
void extract_R(int m, int n, double *A, int lda, double *R, int ldr) {
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j && i < m; i++) {
      R[i + j * ldr] = A[i + j * lda];
    }
    for (int i = j + 1; i < n; i++) {
      R[i + j * ldr] = 0.0;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 4) {
    if (rank == 0)
      printf("Requires exactly 4 MPI ranks.\n");
    MPI_Finalize();
    return 1;
  }

  // Problem dimensions (from command line or defaults)
  int m_global = (argc >= 3) ? atoi(argv[1]) : 16;
  int n = (argc >= 3) ? atoi(argv[2]) : 3;
  int m_local = m_global / size;

  // Allocate memory (Column-major storage for LAPACK)
  double *W_local = calloc(m_local * n, sizeof(double));
  double *tau = malloc(n * sizeof(double));
  double *R_local = calloc(n * n, sizeof(double));

  // Initialize local data (deterministic random values)
  srand48(rank * 1000 + 42);
  for (int j = 0; j < n; j++)
    for (int i = 0; i < m_local; i++)
      W_local[i + j * m_local] = drand48();

  // Retain a copy of original global W on rank 0 for verification later
  double *W_global = NULL;
  if (rank == 0)
    W_global = malloc(m_global * n * sizeof(double));
  for (int j = 0; j < n; j++) {
    MPI_Gather(&W_local[j * m_local], m_local, MPI_DOUBLE,
               rank == 0 ? &W_global[j * m_global] : NULL, m_local, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
  }

  // Synchronize and start timing
  MPI_Barrier(MPI_COMM_WORLD);
  double t_start = MPI_Wtime();

  // STEP 1: Local QR Factorizations
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m_local, n, W_local, m_local, tau);
  // Each proc extracts its local R
  extract_R(m_local, n, W_local, m_local, R_local, n);

  // STEP 2: Reduction Tree
  double *stacked_R = calloc(2 * n * n, sizeof(double));
  double *R_recv = malloc(n * n * sizeof(double));

  // First Level Reduction: Ranks 1->0 and 3->2
  if (rank == 1 || rank == 3) {
    // Rank 1 sends its R to rank 0 (rank - 1 = 0)
    // Rank 3 sends its R to rank 2 (rank - 1 = 2)
    MPI_Send(R_local, n * n, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
  } else if (rank == 0 || rank == 2) {
    // Rank 0 recieves the local R from rank 1
    // Rank 2 recieves the local R from rank 3

    MPI_Recv(R_recv, n * n, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    // Ranks 0 and 2 take their own R_local for their blocks, then append the
    // R_recv they got from ranks 1 and 3 respectively, stacking them to get
    // stacked_R.
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        stacked_R[i + j * (2 * n)] = R_local[i + j * n];      // Top block
        stacked_R[(i + n) + j * (2 * n)] = R_recv[i + j * n]; // Bottom block
      }
    }

    // Local QR on the 2n x n stacked matrix
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2 * n, n, stacked_R, 2 * n, tau);
    extract_R(2 * n, n, stacked_R, 2 * n, R_local,
              n); // Store result back in R_local
  }

  // Second Level Reduction: Rank 2->0
  if (rank == 2) {
    MPI_Send(R_local, n * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 0) {
    // Rank 0 is the final proc on the reduction tree.
    MPI_Recv(R_recv, n * n, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    for (int j = 0; j < n; j++) {
      for (int i = 0; i < n; i++) {
        stacked_R[i + j * (2 * n)] = R_local[i + j * n];
        stacked_R[(i + n) + j * (2 * n)] = R_recv[i + j * n];
      }
    }

    // Final QR
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, 2 * n, n, stacked_R, 2 * n, tau);
    extract_R(2 * n, n, stacked_R, 2 * n, R_local, n);
  }

  // Stop timing
  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();
  double elapsed = t_end - t_start;

  // Verification (Rank 0 only)
  if (rank == 0) {
    printf("--- TSQR Verification ---\n");
    print_matrix("Final R factor from TSQR", n, n, R_local, n);

    // Verify that W^T * W == R^T * R
    double *WtW = calloc(n * n, sizeof(double));
    double *RtR = calloc(n * n, sizeof(double));

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < m_global; k++)
          WtW[i + j * n] +=
              W_global[k + i * m_global] * W_global[k + j * m_global];
        for (int k = 0; k < n; k++)
          RtR[i + j * n] += R_local[k + i * n] * R_local[k + j * n];
      }
    }

    double max_err = 0.0;
    for (int i = 0; i < n * n; i++) {
      double err = fabs(WtW[i] - RtR[i]);
      if (err > max_err)
        max_err = err;
    }

    printf("\nMaximum difference between W^T*W and R^T*R: %e\n", max_err);
    if (max_err < 1e-10)
      printf("Result: SUCCESS. The factorization is correct.\n");
    else
      printf("Result: FAILURE.\n");

    printf("\nTIMING,%d,%d,%.6e\n", m_global, n, elapsed);

    free(W_global);
  }

  // Cleanup
  free(W_local);
  free(tau);
  free(R_local);
  free(stacked_R);
  free(R_recv);
  MPI_Finalize();
  return 0;
}
