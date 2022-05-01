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
#include "math_func.h"
#include <math.h>
#include "help_func.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "cublas_v2.h"
// #include "cublas.h"

using namespace std;
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Macros ========================================================================= ⋁ Preprocessor Macros
#define NUMDIMS 3
// ================================================================================================ ⋀ Preprocessor Macros

// == CUDA Functions ============================================================================== ⋁ CUDA Functions
__global__ void printMat(float *mat, int r, int c){
    int tidx = threadIdx.x;
    if (tidx < 9){
        printf("mat[%d] = %lf\n", tidx, mat[tidx]);
    }
} //
// ================================================================================================ ⋀ CUDA Functions

// == FUNCTIONS =================================================================================== ⋁ FUNCTIONS
// == 1. printMat ================================================================================= ⋁ printMatHeap
void printMatHeap(float *mat, int r, int c, string name){
    cout << name << endl;
    for (int y = 0; y < c; y++) {
        for (int x = 0; x < r; x++) {
            printf("(%d,%d) = %e\t", x,y, mat[y*c+x]);
        } // end for y
        cout << endl;
    } // end for x
} // void printMat
// ================================================================================================ ⋀ printMatHeap

// == 2. matMult ================================================================================== ⋁ matMult
void matMult(float *matA, float *matB, int rA, int cA, int rB, int cB, float *product, string name){
    for (int i = 0; i < rA*cB; i++)
        product[i] = 0;
    cout << name << endl;

    if (cA != rB){
        perror("ERROR, matrix dimension error.  Columns of A != Rows of B.\n");
        return;
    } // if (cA != rB)

    for (int x = 0; x < rA; x++) {
        for (int y = 0; y < cA; y++) {
            for (int k = 0; k < rB; k++){
                product[x*cA+y] += matA[x*cA+k] * matB[k*cB+y];
            } // end for k

        } // end for y
    } // end for x
    return;
} // void printMat
// ================================================================================================ ⋀ matMult
// ================================================================================================ ⋀ FUNCTIONS  


// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

long rows = atoi(argv[1]);  // The number of vectors coming in
const float a_i = 1;//a_i = 1/ (float) rows;
const float *p_a_i;
p_a_i = &a_i;
/*
// Input validate the number of rows
if ((rows % 2 !=0) || (rows < 32)){
    perror("Number of rows must be a factor 2 (2^n) and greater than 32.\n");
   return -1;
} // end if 
*/
// Declare the constants
const char txtMatObs[] = "vectorInObsCM.txt";
const char txtMatRef[] = "vectorInRefCM.txt";
int i, j;
FILE *fpMatObs, *fpMatRef;
fpMatObs = fopen(txtMatObs, "r");
fpMatRef = fopen(txtMatRef, "r");

//ifstream fpMatObs(txtMatObs);
//ifstream fpMatRef(txtMatRef);

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
// read in as column major for cuBLAS package


double a;

for (i = 0; i < NUMDIMS; i++){
    for (j = 0; j < rows; j++){
    //    fpMatObs >> a;
    //    fpMatObs >> matObs[j*rows + i];
    //    fpMatRef >> matRef[j*rows + i];
    //    cout << a << endl;
        fscanf(fpMatObs, "%lf", &matObs[i*rows + j]);
        fscanf(fpMatRef, "%lf", &matRef[i*rows + j]);  
    } // end for j
} // end for i
fclose(fpMatObs);
fclose(fpMatRef);
cout << "read data" << endl;

for (int k = 0; k < rows*NUMDIMS; k++){
    cout << "matobs[" << k << "] = " << matObs[k] << endl;
}
for (int k = 0; k < rows*NUMDIMS; k++){
    cout << "matref[" << k << "] = " << matRef[k] << endl;
}
//printMatHeap(matObs, rows, NUMDIMS, "matObs");
//printMatHeap(matRef, rows, NUMDIMS, "matRef");

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

// We don't do much with CUDA here other than treating what could be massive amounts of matrix multiplication and the long xProd sums,
// everything else will be done with the CPU, way faster since our resulting matrices are puny in comparison to wasting time with the GPU.


// DECLARE ALL THE CUDA & CUBLAS MEMORY/TYPES:
float *cuMatObs, *cuMatRef, *cuB, *cuS, *cuScalarArr;//, *cuX *cuCofactorS, *cuScalarArr, *cuS2;

cudaError_t cudaStat;
cublasStatus_t stat;
cublasHandle_t handle;
//cublasInit();

dim3 threads((long) rows, (long) rows);
dim3 threads3x3(3,3);

int sizeMatInput = rows*NUMDIMS    * sizeof(float); // nx3 matrix
int sizeDCM      = NUMDIMS*NUMDIMS * sizeof(float); // 3x3 matrix

printf("Going to the device\n");
/*
cudaStat=cudaMalloc((void**) &cuMatObs, sizeMatInput);
if (cudaStat != CUBLAS_STATUS_SUCCESS){
    printf ("cuMatObs: CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // if cudaStat != cublas success

cudaStat=cudaMalloc((void**) &cuMatRef, sizeMatInput);
if (cudaStat != CUBLAS_STATUS_SUCCESS){
    printf ("cuMatRef: CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // if cudaStat != cublas success

cudaStat=cudaMalloc((void**) &cuB, sizeDCM);
if (cudaStat != CUBLAS_STATUS_SUCCESS){
    printf ("cuB: CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // if stat != cublas success
*/
/*cudaMalloc((void**) &cuMatObs, sizeMatInput);
cudaMalloc((void**) &cuMatRef, sizeMatInput);
cudaMemcpy(cuMatObs, matObs, sizeMatInput, cudaMemcpyHostToDevice);
cudaMemcpy(cuMatRef, matRef, sizeMatInput, cudaMemcpyHostToDevice);*/
float *B = (float*) malloc(sizeDCM);
cublasSetMatrix(rows, NUMDIMS,    sizeMatInput, matObs, rows,    cuMatObs, rows);
cublasSetMatrix(rows, NUMDIMS,    sizeMatInput, matRef, rows,    cuMatRef, rows);
cublasSetMatrix(NUMDIMS, NUMDIMS, sizeDCM,      B,      NUMDIMS, cuB,      NUMDIMS);
// -- CREATE B ------------------------------------------------------ B MATRIX
//cudaMalloc((void**) &cuB, sizeDCM);
stat = cublasCreate(&handle);
cout << "cublasCreate stat = " << stat << endl;
    if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
 } // if stat != cublas success

stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N, NUMDIMS, NUMDIMS, rows, p_a_i, cuMatObs, rows, cuMatRef, rows, p_a_i, cuB, NUMDIMS);
cout << "cublasSgemm stat = " << stat << endl;
cudaDeviceSynchronize();
cublasDestroy(handle);
// printMat<<<1,1>>>(cuB, rows, NUMDIMS);
if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("matrix multiply failed");
    cudaFree(cuMatObs);
    cudaFree(cuMatRef);
    cudaFree(cuB);
    free(matObs);
    free(matRef);
    cublasDestroy(handle);
    return EXIT_FAILURE;
} // if stat != cublas success   


stat = cublasGetMatrix(NUMDIMS, NUMDIMS, sizeDCM, cuB, NUMDIMS, B, NUMDIMS);
cout << "cublas Get Matrix stat = " << stat << endl;
if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("data download failed\n");
    cudaFree(cuMatObs);
    cudaFree(cuMatRef);
    cudaFree(cuB);
    free(matObs);
    free(matRef);
    cublasDestroy(handle);
    free(B);    

    return EXIT_FAILURE;
} // if stat != cublas success

printMatHeap(B, NUMDIMS, NUMDIMS, "B matrix");
// ------------------------------------------------------------------ B MATRIX


// ==================================================================|
// = BEGINNING FROM HERE BADNESS BELOW BADNESS BELOW BADNESS BELOW ==|
// ==================================================================|




// ================================================================================================ ⋀ QUEST Algorithm
// Free heap memory
free(matObs);
free(matRef);
free(B);

cudaFree(cuMatObs);
cudaFree(cuMatRef);
cudaFree(cuB);

// cudaFree(cuS);
//cudaFree(cuZ);

/*
cudaFree(cuX);
cudaFree(cuScalarArr);
cudaFree(cuS2);
*/
return 0;
} // int main()
// ================================================================================================ ⋀ Main Function
