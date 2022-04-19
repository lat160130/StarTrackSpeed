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
void printMat(double *mat, int r, int col, string name);
// ================================================================================================ ⋀ Function Prototypes


// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

int rows = atoi(argv[1]);  // The number of vectors coming in
    

// Declare the constants
const char txtMatObs[] = "vectorInObs.txt";
const char txtMatRef[] = "vectorInRef.txt";
int x,y;


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
for (x = 0; x < rows*NUMDIM; x++){

    fpMatObs >> matObs[x];
    fpMatRef >> matRef[x];
} // end for x
cout << "read data" << endl;

for (x = 0; x < rows*NUMDIM; x++)
    cout << matObs[x] << endl;




// == Quest Algorithm ============================================================================= ⋁ QUEST Algorithm
// CREATE a_i - assume no bias with each vector sample
double a_i = 1/rows;

// CREATE S

double B[NUMDIM][NUMDIM];
for (x = 0; x < NUMDIM; x++){
    for (y = 0; y < NUMDIM; y++){
        for(int z = 0; z < )
        B[x][y] = mat[x*c+y]
    } // end for y
} // end for x




// ================================================================================================ ⋀ QUEST Algorithm

// FREE Dynamically allocated memory.
/*
for (x = 0; x < rows; x++){
    delete[] matObs[x];
    delete[] matRef[x];
} // end for x
*/
free(matObs);
free(matRef);

} // int main()
// ================================================================================================ ⋀ Main Function

// == FUNCTIONS =================================================================================== ⋁ FUNCTIONS
// == 1. printMat
void printMat(double *mat, int r, int c, string name){
    cout << name << endl;
    for (int x = 0; x < r; x++) {
        for (int y = 0; y < c; y++) {
            printf("(%d,%d) = %lf\t", x,y, mat[x*c+y]);
        } // end for y
        cout << endl;
    } // end for x
} // void printMat
// ================================================================================================ ⋀ FUNCTIONS  

// == 2. matMult 
double* matMult(double *matA, double *matB, int rA, int cA, int rB, int cB, string name){
    cout << name << endl;

    if (cA != rB){
        perror("ERROR, matrix dimension error.  Columns of A != Rows of B.\n");
        return -2;
    } // if (cA != rB)


    for (int x = 0; x < cA; x++) {
        for (int y = 0; y < rB; y++) {
            
            for (int k = 0; k < rB; k++){
                
            } // end for k
        } // end for y
        cout << endl;
    } // end for x
    return 
} // void printMat
// ================================================================================================ ⋀ FUNCTIONS  

