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
#include "math_func.h"
#include "help_func.h"
// ================================================================================================ ⋀ Import Header File Block


// == Main Function =============================================================================== ⋁ Main Function
int main(){

// Declare the constants
const char txtMatObs[] = "vectorInObs.txt";
const char txtMatRef[] = "vectorInRef.txt";

FILE *fpMatObs = fopen(txtMatObs, "r");
FILE *fpMatRef = fopen(txtMatRef, "r");

// Check if either text file failed to open
if ((fpMatObs == NULL) || (fpMatRef == NULL)){
    perror("Text file opening failed: vectorInObs.txt or vectorInRef.txt failed to open.");
    return 1;
} // if




// Read in the two vector table inputs


// == Quest Algorithm ============================================================================= QUEST Algorithm
// ================================================================================================ QUEST Algorithm


} // int main()
// ================================================================================================ ⋀ Main Function


