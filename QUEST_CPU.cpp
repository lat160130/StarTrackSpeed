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

// == Import Header File Block ==================================================================== ⋁ Import Header File Block
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "math_func.h"
#include "help_func.h"
#include "iostream"
#include "fstream"      
#include "string"   
using namespace std;

// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Directives ===================================================================== ⋁ Preprocessor Directives
#define NUMDIM 3
// ================================================================================================ ⋀ Preprocessor Directives

// == Internal Program Function Prototypes ======================================================== ⋁ Function Prototypes
void printMatHeap (double *mat, int r, int col, string name);
//void printMatStack (double mat[][], int r, int c, string name);
double* matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, string name);

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



// printMatHeap(matObs, rows, NUMDIM, "MAT OBS");
// printMatHeap(matRef, rows, NUMDIM, "MAT REF");




// == Quest Algorithm ============================================================================= ⋁ QUEST Algorithm
// CREATE a_i - assume no bias with each vector sample
double a_i = 1/rows;


// CREATE B
// double B[NUMDIM][NUMDIM];
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

// double S[NUMDIM][NUMDIM];
double *S = (double*) malloc(NUMDIM*NUMDIM * sizeof(double));
// Create S
for (i = 0; i < NUMDIM; i++){
    for (j = 0; j < NUMDIM; j++){
        S[i*NUMDIM + j] = B[i*NUMDIM + j] + B[j*NUMDIM + i];
    } // end for j
} // end for i
// PRINT OUT S
printMatHeap(S, NUMDIM, NUMDIM, "S = B + B'");


/*
for (i = 0; i < NUMDIM; i++){
    for (j = 0; j < NUMDIM; j++){
        cout << S[i][j] << " ";
    } // end for j
    cout << endl;
} // end for i
*/



// ================================================================================================ ⋀ QUEST Algorithm

// FREE Dynamically allocated memory.
free(matObs);
free(matRef);

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

// == 2. matMult 
double* matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, string name){
    double *product = (double*) malloc(rA*cB * sizeof(double));
    for (int i = 0; i < rA*cB; i++)
        product[i] = 0;
    cout << name << endl;

    if (cA != rB){
        perror("ERROR, matrix dimension error.  Columns of A != Rows of B.\n");
        return product;
    } // if (cA != rB)

    double sum = 0;
    for (int x = 0; x < cA; x++) {
        for (int y = 0; y < rB; y++) {
            for (int k = 0; k < cA; k++){

                
            } // end for k

        } // end for y
    } // end for x
    return product;
} // void printMat
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