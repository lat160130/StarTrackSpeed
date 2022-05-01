// == Import Header File Block ==================================================================== ⋁ Import Header File Block
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "math_func.h"
#include "help_func.h"
#include "iostream"
#include "fstream"      
#include "string"
#include <cmath>  

#include <cuda_runtime.h>
#include "cublas_v2.h"
using namespace std;
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Directives ===================================================================== ⋁ Preprocessor Directives
#define NUMDIMS 3
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
// ================================================================================================ ⋀ Preprocessor Directives
// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

int rows = atoi(argv[1]);  // The number of vectors coming in
float a_i = 1 / (float) rows;
const char txtMatObs[] = "vectorInObsCM.txt";
const char txtMatRef[] = "vectorInRefCM.txt";


ifstream fpMatObs(txtMatObs);
ifstream fpMatRef(txtMatRef);
// Check if either text file failed to open
if ((!fpMatObs) || (!fpMatRef)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // end if
float *matObs = (float*) malloc(rows*NUMDIMS * sizeof(float));
float *matRef = (float*) malloc(rows*NUMDIMS * sizeof(float));
if (!matObs || !matRef) {
    printf ("host memory allocation failed");
    return EXIT_FAILURE;
 } // if (!matObs || !matRef)

cout << "readin data" << endl;
for (int i = 0; i < rows*NUMDIMS; i++){

    fpMatObs >> matObs[i];
    fpMatRef >> matRef[i];
} // end for x
cout << "read data" << endl;


// BEGIN CUDA AND CUBLAS
float *cuB, *cuMatObs, *cuMatRef;
cudaError_t cudaStat;
cublasStatus_t stat;
cublasHandle_t handle;

int sizeDCM = NUMDIMS * NUMDIMS * sizeof(float);
int sizeMat = rows    * NUMDIMS * sizeof(float);

cudaStat = cudaMalloc ((void**)&cuMatObs, sizeMat);
if (cudaStat != cudaSuccess) {
    cout << "cudaStat = " << cudaStat << endl;
    printf ("obs device memory allocation failed");
    return EXIT_FAILURE;
} // end if (cudaStat != cudaSuccess)

cudaStat = cudaMalloc ((void**)&cuMatRef, sizeMat);
if (cudaStat != cudaSuccess) {
    cout << "cudaStat = " << cudaStat << endl;
    printf ("ref device memory allocation failed");
    return EXIT_FAILURE;
} // end if (cudaStat != cudaSuccess)

cudaStat = cudaMalloc ((void**)&cuB, sizeDCM);
if (cudaStat != cudaSuccess) {
    cout << "cudaStat = " << cudaStat << endl;
    printf ("B: device memory allocation failed");
    return EXIT_FAILURE;
} // end if (cudaStat != cudaSuccess)

stat = cublasCreate(&handle);
if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // end if (stat != CUBLAS_STATUS_SUCCESS)
cudaDeviceSynchronize();
// copy matObs to GPU space
stat = cublasSetMatrix(rows, NUMDIMS, sizeof(*cuMatObs), matObs, rows, cuMatObs, rows);
if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "cuMatObs stat = " << stat << endl;
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // end if (stat != CUBLAS_STATUS_SUCCESS)

// copy matRef to GPU space
stat = cublasSetMatrix(rows, NUMDIMS, sizeof(*cuMatRef), matRef, rows, cuMatRef, rows);
if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "cuMatRef Stat = " << stat << endl;
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
} // end if (stat != CUBLAS_STATUS_SUCCESS)

stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N, NUMDIMS, NUMDIMS, rows, &a_i, cuMatObs, rows, cuMatRef, rows, 0, cuB, NUMDIMS);
if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "cublasSgemm Stat = " << stat << endl;
    printf ("CUBLAS mat mult failed\n");
    return EXIT_FAILURE;
} // end if (stat != CUBLAS_STATUS_SUCCESS)

float *B = (float *) malloc(sizeDCM);
if (!B) {
    printf ("host memory allocation failed");
    return EXIT_FAILURE;
 } // if (!matObs || !matRef)
stat = cublasGetMatrix(NUMDIMS, NUMDIMS, sizeof(float), cuB, NUMDIMS, B, NUMDIMS);
if (stat != CUBLAS_STATUS_SUCCESS) {
    cout << "cublasGetMatrix Stat = " << stat << endl;
    printf ("CUBLAS device to host movement failed\n");
    return EXIT_FAILURE;
} // end if (stat != CUBLAS_STATUS_SUCCESS)
free(matObs);
free(matRef);
} // int main(int argc, char *argv[])
