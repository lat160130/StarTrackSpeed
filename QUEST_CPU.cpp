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
void printMat(double *mat[3], int r, int col, char name[]);
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
double **matObs = new double*[rows];
double **matRef = new double*[rows];
for(x = 0; x < rows; x++){
    matObs[x] = new double[NUMDIM];
    matRef[x] = new double[NUMDIM];
} // end for x

// READ IN TEXT FILE CONTENT
for (x = 0; x < rows; x++) {
    for (y = 0; y < NUMDIM; y++) {
        fpMatObs >> matObs[x][y];
        fpMatRef >> matRef[x][y];
    } // end for x
} // end for y


// == Quest Algorithm ============================================================================= ⋁ QUEST Algorithm
// CREATE a_i - assume no bias with each vector sample
double a_i = 1/rows;

// CREATE S
double sMAT[NUMDIM][NUMDIM];
for (x = 0; x < NUMDIM; x++){
    for (y = 0; y < NUMDIM; y++){
        sMAT[x][y] = dotN(matObs[x], matRef[y], rows);
    } // end for y
} // end for x

printMat(*sMAT, NUMDIM, NUMDIM, "sMAT");


// ================================================================================================ ⋀ QUEST Algorithm

// FREE Dynamically allocated memory.
for (x = 0; x < rows; x++){
    delete[] matObs[x];
    delete[] matRef[x];
} // end for x
delete[] matObs;
delete[] matRef;

} // int main()
// ================================================================================================ ⋀ Main Function

// == FUNCTIONS =================================================================================== ⋁ FUNCTIONS
void printMat(double **mat, int r, int c, string name){
    cout << name << endl;
    for (int x = 0; x < r; x++) {
        for (int y = 0; y < c; y++) {
            printf("(%d,%d) = %lf\t", x,y, mat[x][y]);
        } // end for x
        cout << endl;
    } // end for y
} // void printMat
// ================================================================================================ ⋀ FUNCTIONS  
