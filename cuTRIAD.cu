
// This is the TRIAD algorithm rewritten for CUDA C.
// This code is not fully optimized.

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
// ================================================================================================ ⋀ Preprocessor Macros

__global__ void printMat(double *mat){
    int ti = threadIdx.x;
    if (ti < 9){
        printf("R[%d] = %lf\n", ti, mat[ti]);
    } // end if (ti < 9)
} // __global__ void printMat(double *mat)

// NVIDIA introduction to parallel matMult
 template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(double *C, double *A, double *B, int wA, int wB) {
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
}



__global__ void vec12Build(double *mat, double *matV3){
    int ti = threadIdx.x;
    if(ti < 2){
    // printf("ti = %d\n", ti);
    // CREATE t2
        //double s[3]; m[3];
        
        // V1
        //s[0] = mat[ti*3+0];
        //s[1] = mat[ti*3+1];
        //s[2] = mat[ti*3+2];
        
        // V2
        //m[0] = mat[ti*3+6];
        //m[1] = mat[ti*3+7];
        //m[2] = mat[ti*3+8];
        // MAY NEED A CUDA SYNC THREADS
        // create the unnormalized cross product in matv3 memory
        matV3[ti*NUMDIMS + 0] = mat[ti*3+1] * mat[ti*3+8] - mat[ti*3+7] * mat[ti*3+2];
        matV3[ti*NUMDIMS + 1] = mat[ti*3+6] * mat[ti*3+2] - mat[ti*3+0] * mat[ti*3+8];
        matV3[ti*NUMDIMS + 2] = mat[ti*3+0] * mat[ti*3+7] - mat[ti*3+6] * mat[ti*3+1];
        __syncthreads();
        double norm[2];
        norm[ti] = sqrt(matV3[ti*NUMDIMS + 0] * matV3[ti*NUMDIMS + 0] + 
                        matV3[ti*NUMDIMS + 1] * matV3[ti*NUMDIMS + 1] +
                        matV3[ti*NUMDIMS + 2] * matV3[ti*NUMDIMS + 2]);

        // These first two rows hold t2. 0th row --> t2b, 1st row --> tri
        matV3[ti*NUMDIMS + 0] = matV3[ti*NUMDIMS + 0] / norm[ti];
        matV3[ti*NUMDIMS + 1] = matV3[ti*NUMDIMS + 1] / norm[ti];
        matV3[ti*NUMDIMS + 2] = matV3[ti*NUMDIMS + 2] / norm[ti];
        __syncthreads();
        // Create t3 = t2 x t1
        // V1 = t2
        // t2[0] = matV3[ti*3+0]
        // t2[1] = matV3[ti*3+1]
        // t2[2] = matV3[ti*3+2]

        // V2 = t1
        // t1[0] = mat[ti*3+0];
        // t1[1] = mat[ti*3+1];
        // t1[2] = mat[ti*3+2];
        matV3[ti*NUMDIMS + 6] = matV3[ti*3+1] * mat[ti*3+2]   - mat[ti*3+1]   * matV3[ti*3+2];
        matV3[ti*NUMDIMS + 7] = mat[ti*3+0]   * matV3[ti*3+2] - matV3[ti*3+0] * mat[ti*3+2];
        matV3[ti*NUMDIMS + 8] = matV3[ti*3+0] * mat[ti*3+1]   - mat[ti*3+0]   * matV3[ti*3+1];
        __syncthreads();
        for (int i = 0; i < NUMDIMS; i++){
            //printf("mat[%d] = %lf\n", ti*NUMDIMS+i, mat[ti*NUMDIMS+i]);
            printf("matV3[%d] = %lf\n", ti*6+i, matV3[ti*6+i]); // GRABS 
            printf("matV3[%d] = %lf\n", ti*6+i+3, matV3[ti*6+i+3]);

        } // end for int i
        

           // matV3[]
           __syncthreads();
        } // end if ti < 2
} // __global__ void vec12Build

__global__ void matBuild1(double *mat, double *bodyMat, double *inerMat){
    int ti = threadIdx.x;
    if(ti < 1){
        bodyMat[ti] = mat[ti];   bodyMat[ti+3] = mat[ti+1]; bodyMat[ti+6] = mat[ti+2];
        inerMat[ti] = mat[ti+3]; inerMat[ti+1] = mat[ti+4]; inerMat[ti+2] = mat[ti+5];
    } // if(ti < 1)

} // end __global__ void matBuild1

__global__ void matBuild23(double *mat, double *matV3, double *bodyMat, double *inerMat){
    int ti = threadIdx.x;
    if(ti < 2){
        for (int i = 0; i < NUMDIMS; i++){
            bodyMat[i*3  + 1 + ti] = matV3[ti*6 + i]; // we need the transpose of this for later in the matrix mult - storing in column major
            inerMat[ti*3 + i + 3]  = matV3[ti*6 + i+3];
        } // end for i
        
        __syncthreads();
    } // if(ti < 2)

    for(int i = 0; i < 9; i++)
        printf("B[%d] = %lf\n", i, bodyMat[i]);
        //printf("I[%d] = %lf\n", i, inerMat[i]);
} // __global__ void matBuild23
// == Main Function =============================================================================== ⋁ Main Function
int main(){
    int sizeMat = ROWS*NUMDIMS    * sizeof(double);
    int sizeR   = NUMDIMS*NUMDIMS * sizeof(double);

    double *mat = (double*) malloc(sizeMat);
    double *R   = (double*) malloc(sizeR); 
    const char vecIn[] = "vectorInput.txt";
    FILE *fpVecIn      = fopen(vecIn, "r");

    int i;
    printf("readin data\n");
    for (i = 0; i < ROWS*NUMDIMS; i++){
        fscanf(fpVecIn, "%lf", &mat[i]);
        // printf("mat[%d] = %lf\n", i, mat[i]);
} // end for x

// PRINT MATRIX TO CHECK INPUT
/*
for (i = 0; i < ROWS; i++){
   for (int j = 0; j < NUMDIMS; j++){
       printf("mat[%d][%d] = %lf ", i, j, mat[i*NUMDIMS+j] );
    } // end for i 
    printf("\n");
} // end for j
*/

// CUDA VARIABLES
double *cuMat, *cuMatT3, *cuBodyMat, *cuInerMat, *cuR;

// Declare GPU memory variables and copy to device
cudaMalloc((void**) &cuMat, sizeMat); // Malloc space for the 4 vectors
cudaMalloc((void**) &cuMatT3, sizeMat); // We need to store t2b, t2i, t3b,t3i
cudaMalloc((void**) &cuBodyMat, sizeR);
cudaMalloc((void**) &cuInerMat, sizeR);
cudaMalloc((void**) &cuR,       sizeR);

// checkCudaErrors(cudaMalloc((void**)  &cuMat,   sizeMat));
// checkCudaErrors(cudaMalloc((void** ) &cuMatT3, sizeMat));

cudaMemcpy(cuMat, mat, sizeMat, cudaMemcpyHostToDevice);
vec12Build<<<1,2>>>(cuMat, cuMatT3);
cudaError err = cudaGetLastError();
if (cudaSuccess != err) {
        cout << "Cuda kernel failed! " << cudaGetErrorString(err) << endl;
    } // end if cudaSuccess
matBuild1<<<1,1>>>(cuMat, cuBodyMat, cuInerMat);
matBuild23<<<1,2>>>(cuMat, cuMatT3, cuBodyMat, cuInerMat);
/*
    bodyMat[0] = mat[0]; bodyMat[1] = mat[1]; bodyMat[2] = mat[2];
    inerMat[0] = mat[3]; inerMat[0] = mat[4]; inerMat[0] = mat[5];
*/

/*
cudaStream_t stream;
checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
*/
// BAD PRACTICE, NEED TO ENSURE NUMDIMS >> SZBLK, a workaround for now.
dim3 threads(SZBLK,SZBLK);
dim3 grid(NUMDIMS/SZBLK, NUMDIMS/SZBLK);
MatrixMulCUDA<SZBLK>  <<<grid, threads>>>(cuR, cuBodyMat, cuInerMat, NUMDIMS, NUMDIMS);
err = cudaGetLastError();
if (cudaSuccess != err) {
        cout << "Cuda kernel (MatrixMulCUDA) failed! " << cudaGetErrorString(err) << endl;
    } // end if cudaSuccess

/*
printMat<<<1,NUMDIMS*NUMDIMS>>>(cuR);
err = cudaGetLastError();
if (cudaSuccess != err) {
        cout << "Cuda kernel (printMat) failed! " << cudaGetErrorString(err) << endl;
    } // end if cudaSuccess
*/

cudaMemcpy(R, cuR, sizeR, cudaMemcpyDeviceToHost);
err = cudaGetLastError();
if (cudaSuccess != err) {
        cout << "Cuda kernel (cudaMemcpy) failed! " << cudaGetErrorString(err) << endl;
    } // end if cudaSuccess


// Rotation matrix has now arrived at the host (CPU).
for (int i = 0; i < NUMDIMS; i++){
    for (int j = 0; j < NUMDIMS; j++){
        printf("R[%d, %d] = %lf ", i,j, R[i*NUMDIMS + j]);
    } // end for j
    printf("\n");
} // end for i







// Free CPU memory and GPU(CUDA) memory
free(mat);
free(R);
cudaFree(cuMat);
cudaFree(cuMatT3);
cudaFree(cuBodyMat);
cudaFree(cuInerMat);
cudaFree(cuR);
} // int main()
// ================================================================================================ ⋀ Main Function
