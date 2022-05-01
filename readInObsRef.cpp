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
// == Main Function =============================================================================== ⋁ Main Function
int main(int argc, char *argv[]){

int rows = atoi(argv[1]);  // The number of vectors coming in

const char txtMatObs[] = "vectorInObsCM.txt";
const char txtMatRef[] = "vectorInRef.txt";


ifstream fpMatObs(txtMatObs);
ifstream fpMatRef(txtMatRef);
// Check if either text file failed to open
if ((!fpMatObs) || (!fpMatRef)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // end if
double *matObs = (double*) malloc(rows*NUMDIMS * sizeof(double));
double *matRef = (double*) malloc(rows*NUMDIMS * sizeof(double));

cout << "readin data" << endl;
for (int i = 0; i < rows*NUMDIMS; i++){

    fpMatObs >> matObs[i];
    fpMatRef >> matRef[i];
} // end for x
cout << "read data" << endl;

for (int i = 0; i < rows*NUMDIMS; i++)
    cout << matObs[i] << endl;

} // int main(int argc, char *argv[])
