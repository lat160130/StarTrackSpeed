// Title: TRIAD_CPU.c
// Author: Leon Tannenbaum
// Project: StarTrack Speed
// Date of initial write: Feb 11th 2022
//
//
// Goal: Proceed through the Triad Algorithm.

// == Import Header File Block ==================================================================== ⋁ Import Header File Block
# include <stdio.h>        // import standard input/output functions
# include "math_func.h"    // import all written math functions (e.g., cross product)
// # include "math_strucs.h"  // import all important math structures that require a struct to make code more math readable (vector3)
# include "help_func.h"    // import all helper functions - print a vector (useful for debugging)
// ================================================================================================ ⋀ Import Header File Block

// == Preprocessor Macros ========================================================================= ⋁ Preprocessor Macros
# define DIMS 3
// ================================================================================================ ⋀ Preprocessor Macros


// == Main Function =============================================================================== ⋁ Main Function
int main() {

//for(int i = 0; i++; i < DIMS){
//    printf("sunbody[%d] = %f", i, sunBody[i]);
//} // for

vector3 sunBody, sunIne, magBody, magIne;
char vecIn[] = "vectorInput.txt";
int returnCodeImport = -1;

returnCodeImport = importSunAndMagVecs(vecIn, sunBody, sunIne, magBody, magIne);

printVecVals(sunBody);

return 0;
} // int main()
// ================================================================================================ ⋀ Main Function


