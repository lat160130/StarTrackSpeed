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

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

using namespace std;
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Macros ========================================================================= ⋁ Preprocessor Macros
#define NUMDIMS 3
#define ROWS 4
#define NUMSCALARS 11
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512
#define SZBLK 32
#define TILE_DIM 32
// ================================================================================================ ⋀ Preprocessor Macros

// == CUDA Functions ============================================================================== ⋁ CUDA Functions
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
static __inline__ void modify (cublasHandle_t handle, double *m, int ldm, int n, int p, int q, double alpha, double beta){
 cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
 cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
} // static __inline__ void modify (cublasHandle_t handle, double *m, int ldm, int n, int p, int q, double alpha, double beta){


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


// CREATE S FROM SUMMING B AND B TRANSPOSE
__global__ void matrixAddBandBT(double *C, double *B,  int r, int c){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
        // C = B + BT (B Transpose)
    if (tidx < r && tidy < c){
        C[tidx*c + tidy] = B[tidx*c + tidy] + B[tidy*c + tidx];
    } // if (tidx < r && tidy < c)

} // __global__ void matrixAdd2(double *C, double *A, double *B, int r, int c)

__global__ void modB(double *B, double a_i){
    int tidx = threadIdx.x;
    if (tidx < 9)
        B[tidx] = B[tidx] * a_i;
} // 

// CREATE Z VECTOR TO IMPLEMENT INTO K MATRIX
__global__ void createZVec(double *Z, double *A, double *B, long rows){
    int tidx = threadIdx.x; // this will always equal to three.
    
    if (tidx < 3){
        Z[tidx] = 0;
        __syncthreads();

        for (int i = 0; i < rows; i++){ 
            Z[tidx] = Z[tidx] + ( (A[((tidx+1)%3)*rows + i ] * B[i*NUMDIMS + ((tidx+2)%3)]) - (A[((tidx+2)%3)*rows + i ] * B[i*NUMDIMS + ((tidx+1)%3)]) );
            printf("Z[%d] = %lf\n", tidx, Z[tidx]);
            printf("\n");
        } // end for i
        __syncthreads();

        Z[tidx] = pow(-1, tidx % 2) * Z[tidx];
        __syncthreads();

    } // end if tidx < 3
} // __global__ void createZVec(double *Z, double *A, double *B)

__global__ void modifyZ(double *Z, double a_i){
    int tidx = threadIdx.x;
    if (tidx < 3)
        Z[tidx] = Z[tidx] * a_i;

} // modifyZ(double *Z, double rows)

// ================================================================================================ ⋀ CUDA Functions

// == FUNCTIONS =================================================================================== ⋁ FUNCTIONS
// == 1. printMat ================================================================================= ⋁ printMatHeap
void printMatHeap(double *mat, int r, int c, string name){
    cout << name << endl;
    for (int x = 0; x < r; x++) {
        for (int y = 0; y < c; y++) {
            printf("(%d,%d) = %lf\t", x,y, mat[x*c+y]);
        } // end for y
        cout << endl;
    } // end for x
} // void printMat
// ================================================================================================ ⋀ printMatHeap

// == 2. matMult ================================================================================== ⋁ matMult
void matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, double *product, string name){
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
double a_i = 1/ (double) rows;
/*
// Input validate the number of rows
if ((rows % 2 !=0) || (rows < 32)){
    perror("Number of rows must be a factor 2 (2^n) and greater than 32.\n");
   return -1;
} // end if 
*/
// Declare the constants
const char txtMatObs[] = "vectorInObsCM.txt";
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
// switched to column major for cuBLAS package
for (i = 0; i < rows; i++){
    for (j = 0; j < NUMDIMS; j++){
    fpMatObs >> matObs[i*NUMDIMS + j];
    fpMatRef >> matRef[i*NUMDIMS + j];
    } // end for j
} // end for i
cout << "read data" << endl;

//printMatHeap(matObs, NUMDIMS, rows, "matObs");
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

// We don't do much with CUDA here other than treating what could me massive amounts of matrix multiplication and the long xProd sums,
// everything else will be done with the CPU, way faster since our resulting matrices are puny in comparison to wasting time with the GPU.


// DECLARE ALL THE CUDA & CUBLAS MEMORY/TYPES:
double *cuMatObs, *cuMatRef, *cuB, *cuS;;//, *cuX *cuCofactorS, *cuScalarArr, *cuS2;
cudaError_t cudaStat;
cublasStatus_t stat;
cublasHandle_t handle;

int THREADS = 32;
int BLOCKS_X = 1, BLOCKS_Y = 1;
dim3 threads(THREADS, THREADS);
dim3 blocks(BLOCKS_X, BLOCKS_Y);


int sizeMatInput = rows*NUMDIMS    * sizeof(double); // nx3 matrix
int sizeDCM      = NUMDIMS*NUMDIMS * sizeof(double); // 3x3 matrix
printf("Going to the device\n");
cudaMalloc((void**) &cuMatObs, sizeMatInput);
cudaMalloc((void**) &cuMatRef, sizeMatInput);
cudaMemcpy(cuMatObs, matObs, sizeMatInput, cudaMemcpyHostToDevice);
cudaMemcpy(cuMatRef, matRef, sizeMatInput, cudaMemcpyHostToDevice);


// -- CREATE B ------------------------------------------------------ B MATRIX
cudaMalloc((void**) &cuB, sizeDCM);
// MatrixMulCUDA <16> <<<1, rows>>>(cuB, cuMatObs, cuMatRef, rows, 3);
// MatrixMulCUDA <<<1,threads3x3>>> (cuB, cuMatObs, cuMatRef, 3);
matrixMul<<<blocks, threads>>>(cuMatObs, cuMatRef, cuB);
double *B = (double*) malloc(sizeDCM);

//modB <<<1,NUMDIMS*NUMDIMS>>> (cuB, a_i);

cudaMemcpy(B, cuB, sizeDCM, cudaMemcpyDeviceToHost);
printMatHeap(B, NUMDIMS, NUMDIMS, "B matrix");
// ------------------------------------------------------------------ B MATRIX


/* ==================================================================|
// = BEGINNING FROM HERE BADNESS BELOW BADNESS BELOW BADNESS BELOW ==|
// ==================================================================|


// -- CREATE S ------------------------------------------------------ S MATRIX
cudaMalloc((void**) &cuS, sizeDCM);
matrixAddBandBT <<<1, threads3x3>>> (cuS, cuB,  NUMDIMS, NUMDIMS);
double *S = (double*) malloc(sizeDCM);
cudaMemcpy(S, cuS, sizeDCM, cudaMemcpyDeviceToHost);
printMatHeap(S, NUMDIMS, NUMDIMS, "S = B + B'");
// ------------------------------------------------------------------ S MATRIX
cout << "RETURNED TO THE HOST" << endl;
// -- CREATE Z ------------------------------------------------------ Z VECTOR
double Z[NUMDIMS]; //Z[0] - x dimension, Z[1] - y dimension, Z[2] - z dimension
// inialize to zero
for (int j = 0; j < NUMDIMS; j++)
    Z[j] = 0;

vector3 v3, v1, v2;

for (int i = 0; i < rows; i++){
    v1.x = matObs[i];         v1.y = matObs[rows + i];       v1.z = matObs[2*rows + i];
    v2.x = matRef[i*NUMDIMS]; v2.y = matRef[i*NUMDIMS + 1];  v2.z = matRef[i*NUMDIMS + 2];
    XProd(v1, v2, &v3);

    Z[0] = v3.x + Z[0];
    Z[1] = v3.y + Z[1];
    Z[2] = v3.z + Z[2];
} // end for i


for (int i = 0; i < NUMDIMS; i++)
    Z[i] = Z[i] * a_i;


for (int i = 0; i < NUMDIMS; i++)
    printf("Z[%d] = %lf\n", i, Z[i]);




/*
double *cuZ;
int size3        = NUMDIMS         * sizeof(double); // 3x1 matrix
double *Z = (double*) malloc(size3);
cudaMalloc((void**) &cuZ, size3);
createZVec <<<2,3>>> (cuZ, cuMatObs, cuMatRef, rows);
modifyZ<<<1,3>>>(cuZ,  a_i);
cudaMemcpy(Z, cuZ, size3, cudaMemcpyDeviceToHost);
for (int i = 0; i < NUMDIMS; i++)
    printf("Z[%d] = %lf\n", i, Z[i]);
*/

// ------------------------------------------------------------------ Z VECTOR


// -- CREATE SIGMA -------------------------------------------------- Sigma
double sigma = 0;
// sigma = .5 * trace(B);
sigma = B[0] + B[4] + B[8]; // trace
sigma = sigma * .5;

printf("sigma = %lf\n", sigma);
// ------------------------------------------------------------------ Sigma



// CONVENIENT EXPRESSION OF THE CHARACTERISTIC EQUATION WITH THE 
// CAYLEY-HAMILTON THEOREM


// -- CREATE KAPPA -------------------------------------------------- Kappa
// Kappa = trace(adjoint(S));
// Since we need the trace of the adjoint, we don't need to take transpose.
double cofactorS[NUMDIMS][NUMDIMS];
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
printf("kappa = %lf\n", kappa);
// -- CREATE DELTA -------------------------------------------------- Delta
// delta = det(S);
// double part1 = S[0] * (S[4]*S[8] - S[5]*S[7]);
// double part2 = S[1] * (S[3]*S[8] - S[5]*S[6]);
// double part3 = S[2] * (S[3]*S[7] - S[4]*S[6]);
double delta = (S[0] * (S[4]*S[8] - S[5]*S[7])) 
             - (S[1] * (S[3]*S[8] - S[5]*S[6]))
             + (S[2] * (S[3]*S[7] - S[4]*S[6]));
printf("delta = %lf\n", delta);
// ------------------------------------------------------------------ Delta

// -- CREATE a ------------------------------------------------------ a
double a = sigma*sigma - kappa; // (PROBABLY CAN GET RID OF IT)
// ------------------------------------------------------------------ a

// -- CREATE b ------------------------------------------------------ b
double b = dotN(Z, Z, NUMDIMS) + sigma*sigma;
// ------------------------------------------------------------------ b

// -- CREATE c ------------------------------------------------------ c
// c = delta + Z'SZ;
double c =  ( Z[0]* (S[0]*Z[0] + S[1]*Z[1] + S[2]*Z[2]) )
          + ( Z[1]* (S[3]*Z[0] + S[4]*Z[1] + S[5]*Z[2]) )
          + ( Z[2]* (S[6]*Z[0] + S[7]*Z[1] + S[8]*Z[2]) );
c = delta + c;
// ------------------------------------------------------------------ c

// -- CREATE d ------------------------------------------------------ d
double *S2 = (double*) malloc(NUMDIMS*NUMDIMS * sizeof(double));
matMult(S, S, NUMDIMS, NUMDIMS, NUMDIMS, NUMDIMS, S2, "S^2");
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
cout << "final l2 = " << l2 << endl;
// ------------------------------------------------------------------ lambda

// -- Optimal Quaternion -------------------------------------------- Optimal Quaternion

double beta = l - sigma;
double alpha = l*l - sigma*sigma + kappa;
double gamma = alpha*(l + sigma) - delta;

cout << "beta  = " << beta  << endl;
cout << "alpha = " << alpha << endl;
cout << "gamma = " << gamma << endl;

// X = (alpha*I + beta*S + S^2)Z; --> 3x1 Matrix
double X[3]; 
X[0] = (Z[0]*(beta*S[0] + S2[0] + alpha)) + (Z[1]*(beta*S[1] + S2[1]))         + (Z[2] *(beta*S[2] + S2[2]));
X[1] = (Z[0]*(beta*S[3] + S2[3]        )) + (Z[1]*(beta*S[4] + S2[4] + alpha)) + (Z[2] *(beta*S[5] + S2[5]));
X[2] = (Z[0]*(beta*S[6] + S2[6]        )) + (Z[1]*(beta*S[7] + S2[7]))         + (Z[2] *(beta*S[8] + S2[8] + alpha));

for (int i = 0; i < 3; i++){
    printf("X(%d) = %lf\n", i, X[i]);
}

double quat_denom = sqrt((gamma * gamma) + sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]) );
cout << "quat_denom = " << quat_denom << endl;

double Q_opt[4];
Q_opt[0] = X[0] / quat_denom;
Q_opt[1] = X[1] / quat_denom;
Q_opt[2] = X[2] / quat_denom;
Q_opt[3] = gamma / quat_denom;

for (int i = 0; i < 4; i++){
    printf("Q_opt(%d) = %lf\n", i, Q_opt[i]);
}
// ------------------------------------------------------------------ Optimal Quaternion



// ================================================================================================ ⋀ QUEST Algorithm
// Free heap memory
free(matObs);
free(matRef);

cudaFree(cuMatObs);
cudaFree(cuMatRef);
cudaFree(cuB);
cudaFree(cuS);
//cudaFree(cuZ);

/*
cudaFree(cuX);
cudaFree(cuScalarArr);
cudaFree(cuS2);
*/
} // int main()
// ================================================================================================ ⋀ Main Function
