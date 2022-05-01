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
#define NUMDIMS 3
// ================================================================================================ ⋀ Preprocessor Directives

// == Internal Program Function Prototypes ======================================================== ⋁ Function Prototypes
void printMatHeap (double *mat, int r, int col, string name);
//void printMatStack (double mat[][], int r, int c, string name);
void matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, double *product, string name);

// ================================================================================================ ⋀ Function Prototypes


// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

long rows = atoi(argv[1]);  // The number of vectors coming in
double a_i = 1/ (double) rows;
cout << "a_i = " << a_i << endl;

/*
// Input validate the number of rows
if ((rows % 2 !=0) || (rows < 32)){
    perror("Number of rows must be a factor 2 (2^n) and greater than 32.\n");
   // return (-1);
} // end if 
*/

// Declare the constants
const char txtMatObs[] = "vectorInObs.txt";
const char txtMatRef[] = "vectorInRef.txt";


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
for (int i = 0; i < rows*NUMDIMS; i++){

    fpMatObs >> matObs[i];
    fpMatRef >> matRef[i];
} // end for x
cout << "read data" << endl;

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


// -- CREATE B ------------------------------------------------------ B MATRIX
// HERE B IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
double *B = (double*) malloc(NUMDIMS*NUMDIMS * sizeof(double));
double sum = 0;

for(int i=0; i<NUMDIMS; ++i){ // rows of the first matrix === 
    for(int j=0; j<NUMDIMS; ++j){ // columns of the 2nd matrix
        for(int k=0; k<rows; ++k) {
            sum += matObs[k*NUMDIMS+i] * matRef[k*NUMDIMS+j];
        } // end for k
        B[i*NUMDIMS + j]= sum;
        sum = 0;
    } // end for j
} // end for i

for (int i = 0; i < NUMDIMS * NUMDIMS; i++){
    B[i] = B[i] / (double) rows;
} // end for i 


printMatHeap(B, NUMDIMS, NUMDIMS, "B matrix");
// HERE B IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
// ------------------------------------------------------------------ B Matrix

// -- CREATE S ------------------------------------------------------ S Matrix
// HERE B IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
double *S = (double*) malloc(NUMDIMS*NUMDIMS * sizeof(double));
for (int i = 0; i < NUMDIMS; i++){
    for (int j = 0; j < NUMDIMS; j++){
        S[i*NUMDIMS + j] = B[i*NUMDIMS + j] + B[j*NUMDIMS + i];
    } // end for j
} // end for i
// PRINT OUT S
printMatHeap(S, NUMDIMS, NUMDIMS, "S = B + B'");
// HERE B IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
// ------------------------------------------------------------------ S Matrix

// -- CREATE Z ------------------------------------------------------ Z Matrix
// HERE Z IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
double Z[NUMDIMS]; //Z[0] - x dimension, Z[1] - y dimension, Z[2] - z dimension
// inialize to zero
for (int j = 0; j < NUMDIMS; j++)
    Z[j] = 0;

for (int i = 0; i < rows; i++){
    // traversal row major: mat[x*c+y]
    // vector3 on row i; x v.x = mat[i*c+0] v.y = mat[i*c+1], v.z = mat[i*c+2]
    // v1 = matObs, v2 = matRef
    Z[0] = Z[0] + ((matObs[i*NUMDIMS+1]*matRef[i*NUMDIMS+2]) - (matRef[i*NUMDIMS+1]*matObs[i*NUMDIMS+2]));
    Z[1] = Z[1] + ((matRef[i*NUMDIMS+0]*matObs[i*NUMDIMS+2]) - (matObs[i*NUMDIMS+0]*matRef[i*NUMDIMS+2]));
    Z[2] = Z[2] + ((matObs[i*NUMDIMS+0]*matRef[i*NUMDIMS+1]) - (matRef[i*NUMDIMS+0]*matObs[i*NUMDIMS+1]));    
} // end for i

for (int j = 0; j < NUMDIMS; j++)
    Z[j] =  Z[j] * a_i;


for (int j = 0; j < NUMDIMS; j++)
    printf("Z[%d] = %lf\n", j, Z[j]);
// HERE Z IS CORRECT, VERIFIED BY MATLAB!!!!!!! ==============================
// ------------------------------------------------------------------ Z Matrix

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

// FREE Dynamically allocated memory.
free(matObs);
free(matRef);
free(B);
free(S);
free(S2);
// free(invertS);
// CLOSE IFSTREAM - this is taken care of by the destructor
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