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
using namespace std;

using std::cout;
using std::generate;
using std::vector;

// Pull out matrix and shared memory tile size
const int M = 1 << 10;
const int N = 1 << 11;
const int K = 1 << 12;
const int SHMEM_SIZE = 1 << 10;

__global__ void matrixMul(const int *a, const int *b, int *c) {
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
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c) {
  // For every row...
  for (int row = 0; row < M; row++) {
    // For every column...
    for (int col = 0; col < N; col++) {
      // For every element in the row-column pair
      int tmp = 0;
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
  size_t bytes_a = M * K * sizeof(int);
  size_t bytes_b = K * N * sizeof(int);
  size_t bytes_c = M * N * sizeof(int);

  // Host vectors
  vector<int> h_a(M * K);
  vector<int> h_b(K * N);
  vector<int> h_c(M * N);


int NUMDIMS = 3;
int rows = 128;
const char txtMatObs[] = "vectorInObsCM.txt";
const char txtMatRef[] = "vectorInRef.txt";

ifstream fpMatObs(txtMatObs);
ifstream fpMatRef(txtMatRef);
cout << "WTF?" <<endl;
// Check if either text file failed to open
if ((!fpMatObs) || (!fpMatRef)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // end if

double *matObs = (double *) malloc(rows * NUMDIMS * sizeof(double));
importMatrix(txtMatObs, matObs, rows, NUMDIMS);
double *matRef = (double *) malloc(rows * NUMDIMS * sizeof(double));
importMatrix(txtMatRef, matRef, rows, NUMDIMS);

for (int i = 0; i < rows*NUMDIMS; i ++){
    h_a[i] = matObs[i];
    h_b[i] = matRef[i];
    cout << h_a[i] << endl;
}


  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides M and N evenly)
  int BLOCKS_X = N / THREADS;
  int BLOCKS_Y = M / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes_c, cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c);

  for (int i = 0; i < NUMDIMS * rows; i ++){
    cout << "h_c[" << i << "] = " << h_c[i] << endl;
  } // for

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}