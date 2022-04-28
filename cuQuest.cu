// Title:   cuQuest.cu
// Author:  Leon Tannenbaum
// Project: StarTrack Speed
// Date of initial write: April 27th 2022
//


// Goal:  Proceed through the QUEST algorithm in C optimizing as much as possible without using the GPU.

// GENERAL COMMENTS:
// ALL MATRICES ARE IN ROW MAJOR FORM UNLESS OTHERWISE SPECIFIED
// OUTPUT QUATERNION IS VECTOR FIRST SCALAR LAST: [Q1 Q2 Q3 q4] where q4 is the scalar.

// == Import Header File Block ==================================================================== ⋁ Import Header File Block
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

using namespace std;
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Macros ========================================================================= ⋁ Preprocessor Macros
#define NUMDIMS 3
#define ROWS 4
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512
#define SZBLK 1
#define TILE_DIM 32
// ================================================================================================ ⋀ Preprocessor Macros

__global__ void transposeCoalesced(float *odata, const float *idata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}
// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

int rows = atoi(argv[1]);  // The number of vectors coming in
    

// Declare the constants
const char txtMatObs[] = "vectorInObs.txt";
const char txtMatRef[] = "vectorInRef.txt";
int i, j;


ifstream fpMatObs(txtMatObs);
ifstream fpMatRef(txtMatRef);

// Check if either text file failed to open
if ((!fpMatObs) || (!fpMatRef)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // end if

// -- Read in the data ---------------------------------------------- Read in the data
// Declare storage matrices
double *matObs = (double*) malloc(rows*NUMDIM * sizeof(double));
double *matRef = (double*) malloc(rows*NUMDIM * sizeof(double));

cout << "readin data" << endl;
for (i = 0; i < rows*NUMDIM; i++){

    fpMatObs >> matObs[i];
    fpMatRef >> matRef[i];
} // end for x
cout << "read data" << endl;

// mat in memory: for matrix
// (0,0) (0,1) (0, 2)
// (1,0) (1,1) (1, 2)

// is REALLY LIKE THIS IN MEMORY (x,y)
// (0,0) (0,1) (0,2) (1,0) (1,1) (1,2)
//   0     1     2     3     4     5 
// to access as rows: mat[x*c+y]
// where x is the row iter and y is column iter, 
// and c is the number of columns in the matrix

// to access as column vectors: mat[x*c+y]
// access pattern
// ------------------------------------------------------------------ Read in the data

// DECLARE ALL THE CUDA MEMORY:
double *cuMatObs, *cuMatRef, *cuB, *cuBT, *cuS, *cuZ, *cuScalarArr, *cuS2
// cuScalarArr will be a special Array holding all the important scalars, row major, like this in memory:
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]



// Free heap memory
free(matObs);
free(matRef);
/*
cudaFree(cuMatObs);
cudaFree(cuMatRef);
cudaFree(B);
cudaFree(S);
cudaFree(Z);
cudaFree(S2);
*/
} // int main()
// ================================================================================================ ⋀ Main Function
