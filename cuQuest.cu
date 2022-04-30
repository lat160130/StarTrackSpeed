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
#define NUMSCALARS 11
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512
#define SZBLK 64
#define TILE_DIM 32
// ================================================================================================ ⋀ Preprocessor Macros

// == CUDA Functions ============================================================================== ⋁ CUDA Functions

// GENERAL MATRIX MULTIPLICATION
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(double *C, double *A, double *B, long long wA, long long wB) {
    /**
     * Matrix multiplication (CUDA Kernel) on the device: C = A * B
     * wA is A's width and wB is B's width
     */
    
      // Block index
      int bx = blockIdx.x;
      int by = blockIdx.y;
    
      // Thread index
      int tx = threadIdx.x;
      int ty = threadIdx.y;
    
      // Index of the first sub-matrix of A processed by the block
      int aBegin = wA * BLOCK_SIZE * by;
    
      // Index of the last sub-matrix of A processed by the block
      int aEnd   = aBegin + wA - 1;
    
      // Step size used to iterate through the sub-matrices of A
      int aStep  = BLOCK_SIZE;
    
      // Index of the first sub-matrix of B processed by the block
      int bBegin = BLOCK_SIZE * bx;
    
      // Step size used to iterate through the sub-matrices of B
      int bStep  = BLOCK_SIZE * wB;
    
      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      double Csub = 0;
    
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int a = aBegin, b = bBegin;
           a <= aEnd;
           a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    
        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
    
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
    
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll
    
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          Csub += As[ty][k] * Bs[k][tx];
        }
    
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }
    
      // Write the block sub-matrix to device memory;
      // each thread writes one element
      int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
      C[c + wB * ty + tx] = Csub;
 } // __global__ void MatrixMulCUDA(double *C, double *A, double *B, int wA, int wB) 

// CREATE S FROM SUMMING B AND B TRANSPOSE
__global__ void matrixAddBandBT(double *C, double *B,  int r, int c){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
        // C = B + BT (B Transpose)
    if (tidx < r && tidy < c){
        C[tidx*c + tidy] = B[tidx*c + tidy] + B[tidy*c + tidx];
    } // if (tidx < r && tidy < c)

} // __global__ void matrixAdd2(double *C, double *A, double *B, int r, int c)

// CREATE Z VECTOR TO IMPLEMENT INTO K VECTOR
__global__ void createZVec(double *Z, double *A, double *B, long long rows){
    int tidx = threadIdx.x; // this will always equal to three.
    if (tidx < 3){
        Z[tidx] = 0;
        __syncthreads();

        for (int i = 0; i < rows; i++){
            int 
            Z[tidx] = Z[tidx] + ( (A[(tidx%3 + 1)*rows + i ] * B[i*NUMDIMS + ((tidx+2)%3)]) - (A[(tidx%3 + 2)*rows + i ] * B[i*NUMDIMS + ((tidx+1)%3)]) )

        } // end for i
        __syncthreads();

        Z[tidx] = pow(-1, tidx % 2) * Z[tidx];
        __syncthreads();

    } // end if tidx < 3
} // __global__ void createZVec(double *Z, double *A, double *B)

__global__ void initScalars(double *scalarVector, int numScal){
    int tidx = threadIdx.x;

    scalarVector[tidx] = 1;
    __syncthreads();
} // __global__ void initScalars(double *scalarVector, int numScal)

__global__ void matTrace(double *mat, int columns, double *scalarMat,  int arrPosn){
    // mat - matrix to find the trace of
     // columns - number of rows in mat (3x3) is coming in ALWAYS, so columns == 3
    // scalarMat, this matrix holds all the scalar values in the algorithm for simple access for gpu shared memory
    // arrPosn - position in the array that sigma will hold

    if (columns != 3){
        perror("Columns must equal 3!\n");
    } // end if columns

    int tidx = threadIdx.x;
    double sigma = 0;
    for (int i = 0; i < rows; i++){
        sigma = sigma + mat[i*columns + i];
    } // end for i
    scalarMat[arrPosn] = .5 * sigma;
} // __global__ void matTrace(double *mat, long long rows)

__global__ void genCofactor(double *mat, double *cofM){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    
} // __global__ void genCofactor(double *mat, double *cofM)
 // ================================================================================================ ⋀ CUDA Functions

// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

long long rows = atoi(argv[1]);  // The number of vectors coming in

// Input validate the number of rows
if ((rows % 2 !=0) || (rows < 32)){
    perror("Number of rows must be a factor 2 (2^n) and greater than 32.\n");
   return -1;
} // end if 

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
double *matObs = (double*) malloc(rows*NUMDIMS * sizeof(double));
double *matRef = (double*) malloc(rows*NUMDIMS * sizeof(double));

cout << "readin data" << endl;
for (i = 0; i < rows; i++){
    for (j = 0; j < NUMDIMS; j++){
    fpMatObs >> matObs[j*NUMDIMS + i]; // switched to column major for pretranspose
    fpMatRef >> matRef[i*NUMDIMS + j];
    } // end for j
} // end for i
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


// == Quest Algorithm ============================================================================= ⋁ QUEST Algorithm
// CREATE a_i - assume no bias with each vector sample
// double a_i = 1/rows;


// DECLARE ALL THE CUDA MEMORY:
double *cuMatObs, *cuMatRef, *cuB, *cuS, *cuZ, *cuCofactorS;//, *cuScalarArr, *cuS2
dim3 threads(SZBLK, SZBLK);
dim3 grid(rows/SZBLK, rows/SZBLK);
// cuScalarArr will be a special Array holding all the important scalars, row major, like this in memory:
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
int sizeMatInput = rows*NUMDIMS    * sizeof(double); // nx3 matrix
int sizeDCM      = NUMDIMS*NUMDIMS * sizeof(double); // 3x3 matrix


cudaMalloc((void**) &cuMatObs, sizeMatInput);
cudaMalloc((void**) &cuMatRef, sizeMatInput);
cudaMemcpy(cuMatObs, matObs, sizeMatInput, cudaMemcpyHostToDevice);
cudaMemcpy(cuMatRef, matRef, sizeMatInput, cudaMemcpyHostToDevice);



// -- CREATE B ------------------------------------------------------ B MATRIX
cudaMalloc((void**) &cuB, sizeDCM);
MatrixMulCUDA<SZBLK> <<<1, threads>>>(cuB, cuMatObs, cuMatRef, rows, rows);
// ------------------------------------------------------------------ B MATRIX

// -- CREATE S ------------------------------------------------------ S MATRIX
cudaMalloc((void**) &cuS, sizeDCM);
dim3 threads3x3(3,3);
matrixAddBandBT <<<1, threads3x3>>> (cuS, cuB,  NUMDIMS, NUMDIMS);
// ------------------------------------------------------------------ S MATRIX

// -- CREATE Z ------------------------------------------------------ Z MATRIX
cudaMalloc((void**), &cuZ, NUMDIMS*sizeof(double));
createZVec <<<1,3>>>(cuZ, cuMatObs, cuMatRef, rows);
// ------------------------------------------------------------------ Z MATRIX

// CREATE SCALARS
cudaMalloc((void**), &cuScalarArr, NUMSCALARS*sizeof(double));
initScalars <<<1,NUMSCALARS>>>(cuScalarArr, NUMSCALARS); // this initializes all values in the array to 1;

// -- CREATE SIGMA -------------------------------------------------- SIGMA
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
matTrace <<<1,1>>>(cuB, NUMDIMS, cuScalarArr,  0);
// ------------------------------------------------------------------ SIGMA

// -- CREATE KAPPA -------------------------------------------------- KAPPA
cudaMalloc((void**), &cuCofactorS, sizeDCM);

// ------------------------------------------------------------------ KAPPA

// ================================================================================================ ⋀ QUEST Algorithm
// Free heap memory
free(matObs);
free(matRef);

cudaFree(cuMatObs);
cudaFree(cuMatRef);
cudaFree(cuB);
cudaFree(cuS);
cudaFree(cuZ);
cudaFree(cuScalarArr);
//cudaFree(cuS2);

} // int main()
// ================================================================================================ ⋀ Main Function
