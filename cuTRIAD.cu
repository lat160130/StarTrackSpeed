// This is the TRIAD algorithm rewritten for CUDA C.
// This code is not fully optimized.

// == Import Header File Block ==================================================================== ⋁ Import Header File Block
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "math_func.h"
#include "help_func.h"

using namespace std;
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Macros ========================================================================= ⋁ Preprocessor Macros
#define NUMDIMS 3
#define ROWS 4
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512
// ================================================================================================ ⋀ Preprocessor Macros



__global__ void vec12Build(double *mat, double *matV3){
    int tx = threadIdx.x;
    printf("tx = %d\n", tx);

} // __global__ void vec12Build

// == Main Function =============================================================================== ⋁ Main Function
int main(){
    int sizeMat = ROWS*NUMDIMS * sizeof(double);
    double *mat = (double*) malloc(sizeMat);
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
double *cuMat, *cuMatT3;

// Declare GPU memory variables and copy to device
cudaMalloc((void**) &cuMat, sizeMat); // Malloc space for the 4 vectors
cudaMalloc((void**) &cuMatT3, sizeMat*.5); // Multiply by .5 because we only need space for 2 vectors not 4

// checkCudaErrors(cudaMalloc((void**)  &cuMat,   sizeMat));
// checkCudaErrors(cudaMalloc((void** ) &cuMatT3, sizeMat));

cudaMemcpy(cuMat, mat, sizeMat, cudaMemcpyHostToDevice);
vec12Build<<<1,2>>>(cuMat, cuMatT3);
cudaError err = cudaGetLastError();
if (cudaSuccess != err) {
        cout << "Cuda kernel failed! " << cudaGetErrorString(err) << endl;
    } // end if cudaSuccess








// Free memory and CUDA memory
free(mat);
cudaFree(cuMat);
cudaFree(cuMatT3);
} // int main()
// ================================================================================================ ⋀ Main Function
