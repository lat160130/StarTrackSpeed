// Title: TRIAD_CPU.c
// Author: Leon Tannenbaum
// Project: StarTrack Speed
// Date of initial write: Feb 11th 2022
//
//
// Goal: Proceed through the Triad Algorithm.

// == Import Header File Block ==================================================================== ⋁ Import Header File Block
# include <stdio.h>        // import standard input/output functions
// # include "math_func.h"    // import all written math functions (e.g., cross product)
# include "math_strucs.h"  // import all important math structures that require a struct to make code more math readable (vector3)
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
sunBody.x = 0;
sunBody.y = 0;
sunBody.z = 0;

sunIne.x = 0;
sunIne.y = 0;
sunIne.z = 0;

magBody.x = 0;
magBody.y = 0;
magBody.z = 0;

magIne.x = 0;
magIne.y = 0;
magIne.z = 0;


char vecIn[] = "vectorInput.txt";
int returnCodeImport = -1;

returnCodeImport = importSunAndMagVecs(vecIn, &sunBody, &sunIne, &magBody, &magIne);


printf("ReturnCodeImport = %d\n", returnCodeImport);
printVecVals(sunBody);
printVecVals(sunIne);
printVecVals(magBody);
printVecVals(magIne);

return 0;
} // int main()
// ================================================================================================ ⋀ Main Function


