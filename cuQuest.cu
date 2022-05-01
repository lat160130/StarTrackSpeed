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
<<<<<<< HEAD
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

=======

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

__global__ void modifyZ(double *Z, double rows){
    int tidx = threadIdx.x;
    if (tidx < 3)
        Z[tidx] = Z[tidx] / rows;

} // modifyZ(double *Z, double rows)

// ================================================================================================ ⋀ CUDA Functions
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e

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

<<<<<<< HEAD
// We don't do much with CUDA here other than treating what could be massive amounts of matrix multiplication and the long xProd sums,
// everything else will be done with the CPU, way faster since our resulting matrices are puny in comparison to wasting time with the GPU.
=======
// We don't do much with CUDA here other than treating what could me massive amounts of matrix multiplication and the long xProd sums,
// everything else will be done with the CPU, way faster since our resulting matrices are puny in comparison to wasting time with the GPU.
// DECLARE ALL THE CUDA MEMORY:
double *cuMatObs, *cuMatRef, *cuB, *cuS, *cuZ;//, *cuX *cuCofactorS, *cuScalarArr, *cuS2;
dim3 threads(SZBLK, SZBLK);
dim3 threads3x3(3,3);

int sizeMatInput = rows*NUMDIMS    * sizeof(double); // nx3 matrix
int sizeDCM      = NUMDIMS*NUMDIMS * sizeof(double); // 3x3 matrix
int size3        = NUMDIMS         * sizeof(double); // 3x1 matrix
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e


// DECLARE ALL THE CUDA & CUBLAS MEMORY/TYPES:
float *cuMatObs, *cuMatRef, *cuB, *cuS, *cuScalarArr;//, *cuX *cuCofactorS, *cuScalarArr, *cuS2;

cudaError_t cudaStat;
cublasStatus_t stat;
cublasHandle_t handle;
//cublasInit();

<<<<<<< HEAD
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
=======
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e
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

<<<<<<< HEAD

// ==================================================================|
// = BEGINNING FROM HERE BADNESS BELOW BADNESS BELOW BADNESS BELOW ==|
// ==================================================================|

=======
// -- CREATE S ------------------------------------------------------ S MATRIX
cudaMalloc((void**) &cuS, sizeDCM);
matrixAddBandBT <<<1, threads3x3>>> (cuS, cuB,  NUMDIMS, NUMDIMS);

double *S = (double*) malloc(sizeDCM);
cudaMemcpy(S, cuS, sizeDCM, cudaMemcpyDeviceToHost);
// ------------------------------------------------------------------ S MATRIX

// -- CREATE Z ------------------------------------------------------ Z VECTOR
double *Z = (double*) malloc(size3);
cudaMalloc((void**), &cuZ, NUMDIMS*sizeof(double));
createZVec <<<1,3>>>(cuZ, cuMatObs, cuMatRef, rows);
cudaMemcpy(Z, cuZ, size3, cudaMemcpyDeviceToHost);
// divide Z[j] by the a_i modifier
for (j = 0; j < NUMDIM; j++)
    Z[j] =  Z[j] / (double) rows;
// ------------------------------------------------------------------ Z VECTOR

// -- CREATE SIGMA -------------------------------------------------- Sigma
double sigma = 0;
// sigma = .5 * trace(S);
for (i = 0; i < NUMDIM; i++){
        sigma = sigma + S[i*NUMDIM + i];
} // end for i
sigma = .5*sigma; 

printf("sigma = %lf\n", sigma);
cout << "sigma = " << sigma << endl;
// ------------------------------------------------------------------ Sigma



// CONVENIENT EXPRESSION OF THE CHARACTERISTIC EQUATION WITH THE 
// CAYLEY-HAMILTON THEOREM


// -- CREATE KAPPA -------------------------------------------------- Kappa
// Kappa = trace(adjoint(S));
// Since we need the trace of the adjoint, we don't need to take transpose.
double cofactorS[NUMDIM][NUMDIM];
// ROW 0
cofactorS[0][0] =  (S[4]*S[8] - S[5]*S[7]);
cofactorS[0][1] = -(S[3]*S[8] - S[5]*S[6]);
cofactorS[0][2] =  (S[3]*S[7] - S[4]*S[6]);
// ROW 1
cofactorS[1][0] = -(S[1]*S[8] - S[2]*S[7]);
cofactorS[1][1] =  (S[0]*S[8] - S[2]*S[6]);
cofactorS[1][2] = -(S[0]*S[7] - S[1]*S[6]);
// ROW 2
cofactorS[2][0] =  (S[1]*S[5] - S[2]*S[4]);
cofactorS[2][1] = -(S[0]*S[5] - S[2]*S[3]);
cofactorS[2][2] =  (S[0]*S[4] - S[1]*S[3]);

double kappa = cofactorS[0][0] + cofactorS[1][1] + cofactorS[2][2];

// -- CREATE DELTA -------------------------------------------------- Delta
// delta = det(S);
// double part1 = S[0] * (S[4]*S[8] - S[5]*S[7]);
// double part2 = S[1] * (S[3]*S[8] - S[5]*S[6]);
// double part3 = S[2] * (S[3]*S[7] - S[4]*S[6]);
double delta = (S[0] * (S[4]*S[8] - S[5]*S[7])) 
             - (S[1] * (S[3]*S[8] - S[5]*S[6]))
             + (S[2] * (S[3]*S[7] - S[4]*S[6]));
// ------------------------------------------------------------------ Delta

// -- CREATE a ------------------------------------------------------ a
double a = sigma*sigma - kappa; // (PROBABLY CAN GET RID OF IT)
// ------------------------------------------------------------------ a

// -- CREATE b ------------------------------------------------------ b
double b = dotN(Z, Z, NUMDIM) + sigma*sigma;
// ------------------------------------------------------------------ b

// -- CREATE c ------------------------------------------------------ c
// c = delta + Z'SZ;
double c =  ( Z[0]* (S[0]*Z[0] + S[1]*Z[1] + S[2]*Z[2]) )
          + ( Z[1]* (S[3]*Z[0] + S[4]*Z[1] + S[5]*Z[2]) )
          + ( Z[2]* (S[6]*Z[0] + S[7]*Z[1] + S[8]*Z[2]) );
c = delta + c;
// ------------------------------------------------------------------ c

// -- CREATE d ------------------------------------------------------ d
double *S2 = (double*) malloc(NUMDIM*NUMDIM * sizeof(double));
matMult(S, S, NUMDIM, NUMDIM, NUMDIM, NUMDIM, S2, "S^2");
printMatHeap(S2, 3, 3, "Printing S^2");
double d =  (Z[0]* (S2[0]*Z[0] + S2[1]*Z[1] + S2[2]*Z[2]))
          + (Z[1]* (S2[3]*Z[0] + S2[4]*Z[1] + S2[5]*Z[2]))
          + (Z[2]* (S2[6]*Z[0] + S2[7]*Z[1] + S2[8]*Z[2]));
// ------------------------------------------------------------------ d



printf("kappa = %lf\t delta = %lf\n", kappa, delta);
printf("a = %lf\tb = %lf\n", a, b);
printf("c = %lf\td = %lf\n", c, d);



// -- Newton's method for convergence of lambda --------------------- lambda
double tol = 1e-12;
int max_it = 100;
int iters = 0;
double error = tol + 1;

double l = 1, l2;
while (error > tol && iters < max_it){
    l2 = l - (((l*l*l*l) - (a+b)*l*l - c*l + (a*b + c*sigma - d)) / (4*l*l*l - 2*l*(a+b) -c));
    error = fabs(l2 - l);
    l = l2;
    iters++;
    cout << error << endl;
} // end while error > tol && it < max_it
cout << "root or final l = " << l << endl;
// ------------------------------------------------------------------ lambda

// -- Optimal Quaternion -------------------------------------------- Optimal Quaternion

double beta = l - sigma;
double alpha = l*l - sigma*sigma + kappa;
double gamma = alpha*(l + sigma) - delta;

// X = (alpha*I + beta*S + S^2)Z; --> 3x1 Matrix
double X[3]; 
X[0] = (Z[0]*(beta*S[0] + S2[0] + alpha)) + (Z[1]*(beta*S[1] + S2[1]))         + (Z[2] *(beta*S[2] + S2[2]));
X[1] = (Z[0]*(beta*S[3] + S2[3]        )) + (Z[1]*(beta*S[4] + S2[4] + alpha)) + (Z[2] *(beta*S[5] + S2[5]));
X[2] = (Z[0]*(beta*S[6] + S2[6]        )) + (Z[1]*(beta*S[7] + S2[7]))         + (Z[2] *(beta*S[8] + S2[8] + alpha));

double quat_denom = sqrt((gamma * gamma) + sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]) );


double Q_opt[4];
Q_opt[0] = X[0] / quat_denom;
Q_opt[1] = X[1] / quat_denom;
Q_opt[2] = X[2] / quat_denom;
Q_opt[3] = gamma / quat_denom;
// ------------------------------------------------------------------ Optimal Quaternion
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e



// ================================================================================================ ⋀ QUEST Algorithm
// Free heap memory
free(matObs);
free(matRef);
free(B);

cudaFree(cuMatObs);
cudaFree(cuMatRef);
cudaFree(cuB);
<<<<<<< HEAD

// cudaFree(cuS);
//cudaFree(cuZ);

=======
cudaFree(cuS);
cudaFree(cuZ);

>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e
/*
cudaFree(cuX);
cudaFree(cuScalarArr);
cudaFree(cuS2);
*/
<<<<<<< HEAD
return 0;
=======
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e
} // int main()
// ================================================================================================ ⋀ Main Function
