// This program computes matrix multiplication using shared memory tiling
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include "fstream" 
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "math_func.h"
#include "help_func.h"     
#include "string"
#include <cmath>   
#define NUMDIMS 3

using namespace std;

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size
const int M = 1 << 10;
const int N = 1 << 11;
const int K = 1 << 12;
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMul(const double *a, const double *b, double *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];

  // Accumulate in temporary variable
  int tmp = 0;

  // Sweep tile across matrix
  for (int i = 0; i < K; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }

  // Write back results
  c[row * N + col] = tmp;
}

// Check result on the CPU
// MxN = MxK * KxN
void verify_result(vector<double> &a, vector<double> &b, vector<double> &c) {
  // For every row...
  for (int row = 0; row < M; row++) {
    // For every column...
    for (int col = 0; col < N; col++) {
      // For every element in the row-column pair
      double tmp = 0;
      for (int i = 0; i < K; i++) {
        // Accumulate the partial results
        tmp += a[row * K + i] * b[i * N + col];
      }

      // Check against the CPU result
      assert(tmp == c[row * N + col]);
    }
  }
}

int main() {
  // Size (in bytes) of matrix
  // MxN = MxK * KxN
  size_t bytes_a = M * K * sizeof(double);
  size_t bytes_b = K * N * sizeof(double);
  size_t bytes_c = M * N * sizeof(double);

  // Host vectors
  vector<double> h_a(M * K);
  vector<double> h_b(K * N);
  vector<double> h_c(M * N);


int rows = 128;
const char txtMatObs[] = "vectorInObsCM.txt";
const char txtMatRef[] = "vectorInRef.txt";


ifstream fpMatObs(txtMatObs);
ifstream fpMatRef(txtMatRef);
// Check if either text file failed to open
if ((!fpMatObs) || (!fpMatRef)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // end if
double *matObs = (double*) malloc(rows*NUMDIMS * sizeof(double));
double *matRef = (double*) malloc(rows*NUMDIMS * sizeof(double));

cout << "readin data" << endl;
for (int i = 0; i < rows*NUMDIMS; i++){

    fpMatObs >> h_a[i];//matObs[i];
    fpMatRef >> h_b[i];//matRef[i];
} // end for x
cout << "read data" << endl;

cout << "verify data" << endl;
for (int i = 0; i < rows*NUMDIMS; i++){

    cout << h_a[i];//matObs[i];
    // fpMatRef >> h_b[i];//matRef[i];
} // end for x


  // Allocate device memory
  double *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides M and N evenly)
  int BLOCKS_X = rows / THREADS;
  int BLOCKS_Y = M / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, 1);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);
 for (int i = 0; i < NUMDIMS * NUMDIMS; i ++){
    cout << "h_c[" << i << "] = " << h_c[i] << endl;
  } // for

  // Check result
  verify_result(h_a, h_b, h_c);

 

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}