// Title:   QUEST_CPU.c
// Author:  Leon Tannenbaum
// Project: StarTrack Speed
// Date of initial write: Feb 4th 2022
//


// Goal:  Proceed through the QUEST algorithm in C optimizing as much as possible without using the GPU.
// Input: AB- Acceleration   vector in body frame [3x1]
//        MB- Magnetic field vector in body frame [3x1]
//        GB- Gravity vector  in navigation frame [3x1]
//        MN- Magnetic vector in navigation frame [3x1]

// GENERAL COMMENTS:
// ALL MATRICES ARE IN ROW MAJOR FORM.
// OUTPUT QUATERNION IS VECTOR FIRST SCALAR LAST: [Q1 Q2 Q3 q4] where q4 is the scalar.

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
using namespace std;

// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Directives ===================================================================== ⋁ Preprocessor Directives
#define NUMDIM 3
// ================================================================================================ ⋀ Preprocessor Directives

// == Internal Program Function Prototypes ======================================================== ⋁ Function Prototypes
void printMatHeap (double *mat, int r, int col, string name);
//void printMatStack (double mat[][], int r, int c, string name);
void matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, double *product, string name);

// ================================================================================================ ⋀ Function Prototypes


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



// == Quest Algorithm ============================================================================= ⋁ QUEST Algorithm
// CREATE a_i - assume no bias with each vector sample
// double a_i = 1/rows;


// -- CREATE B ------------------------------------------------------ B MATRIX
double *B = (double*) malloc(NUMDIM*NUMDIM * sizeof(double));
double sum = 0;

for(int i=0; i<NUMDIM; ++i){ // rows of the first matrix === 
    for(int j=0; j<NUMDIM; ++j){ // columns of the 2nd matrix
        for(int k=0; k<rows; ++k) {
            sum += matObs[k*NUMDIM+i] * matRef[k*NUMDIM+j];
        } // end for k
        B[i*NUMDIM + j]= sum;
        sum = 0;
    } // end for j
} // end for i
// ------------------------------------------------------------------ B Matrix

// -- CREATE S ------------------------------------------------------ S Matrix
double *S = (double*) malloc(NUMDIM*NUMDIM * sizeof(double));
for (i = 0; i < NUMDIM; i++){
    for (j = 0; j < NUMDIM; j++){
        S[i*NUMDIM + j] = B[i*NUMDIM + j] + B[j*NUMDIM + i];
    } // end for j
} // end for i
// PRINT OUT S
printMatHeap(S, NUMDIM, NUMDIM, "S = B + B'");
// ------------------------------------------------------------------ S Matrix

// -- CREATE Z ------------------------------------------------------ Z Matrix
double Z[NUMDIM]; //Z[0] - x dimension, Z[1] - y dimension, Z[2] - z dimension
// inialize to zero
for (j = 0; j < NUMDIM; j++)
    Z[j] = 0;

for (i = 0; i < rows; i++){
    // traversal row major: mat[x*c+y]
    // vector3 on row i; x v.x = mat[i*c+0] v.y = mat[i*c+1], v.z = mat[i*c+2]
    // v1 = matObs, v2 = matRef
    Z[0] = Z[0] + ((matObs[i*NUMDIM+1]*matRef[i*NUMDIM+2]) - (matRef[i*NUMDIM+1]*matObs[i*NUMDIM+2]));
    Z[1] = Z[1] + ((matRef[i*NUMDIM+0]*matObs[i*NUMDIM+2]) - (matObs[i*NUMDIM+0]*matRef[i*NUMDIM+2]));
    Z[2] = Z[2] + ((matObs[i*NUMDIM+0]*matRef[i*NUMDIM+1]) - (matRef[i*NUMDIM+0]*matObs[i*NUMDIM+1]));    
} // end for i

for (j = 0; j < NUMDIM; j++)
    Z[j] =  Z[j] / (double) rows;

// cout << a_i << endl;
for (j = 0; j < NUMDIM; j++)
    printf("Z[%d] = %lf\n", j, Z[j]);

// ------------------------------------------------------------------ Z Matrix

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

// -- Initial Lambda ------------------------------------------------ Initial Lambda
/*
double *invertS = (double*) malloc(NUMDIM*NUMDIM * sizeof(double));
invertS[0] = 0;
invertS[0] = 0;
*/
// ------------------------------------------------------------------ Initial Lambda


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

double quat_denom = sqrt((gamma * gamma) + abs(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]) );


double Q_opt[4];
Q_opt[0] = X[0] / quat_denom;
Q_opt[1] = X[1] / quat_denom;
Q_opt[2] = X[2] / quat_denom;
Q_opt[3] = gamma / quat_denom;
// ------------------------------------------------------------------ Optimal Quaternion
// ================================================================================================ ⋀ QUEST Algorithm

// FREE Dynamically allocated memory.
free(matObs);
free(matRef);
free(B);
free(S);
free(S2);
// free(invertS);

} // int main()
// ================================================================================================ ⋀ Main Function

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

    double sum = 0;
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



/*
void printMatStack (double mat[][], int r, int c, string name){
    cout << name << endl;
    for (int x = 0; x < r; x++) {
        for (int y = 0; y < c; y++) {
            printf("(%d,%d) = %lf\t", x,y, mat[x][y]);
        } // end for y
        cout << endl;
    } // end for x

} // 

*/