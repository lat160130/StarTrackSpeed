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


// Declare and assign each vector component a value.
// Initialize sun/magnet Body/Inertial frame vector components to 0.
// The sun vectors and the magnet vectors come from measurement tools on a spacecraft, a magnetometer for example.
vector3 sunBody, sunIne, magBody, magIne, t2b, t2i, t3b, t3i, test;
double normCross = 0;
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


const char vecIn[] = "vectorInput.txt";
int returnCodeImport = -1;

returnCodeImport = importSunAndMagVecs(vecIn, &sunBody, &sunIne, &magBody, &magIne);


// Check to see if each vector worked.
printf("ReturnCodeImport = %d\n", returnCodeImport);
// Checking validity of cross product
XProd(sunBody, sunIne, &test);
// printVecVals(test);

// == Triad Algorithm ============================================================================= ⋁ Triad Algorithm
// Let us assume that the sun vector has more accurate components than the magnetic field vector.
// t1b = sunBody, t1i = sunIne

XProd(sunBody, magBody, &test); // this has sB x mB
normCross = norm3L2(test); // the norm of sB x mB

// evaluate cross product / norm of the cross product
t2b.x = test.x / normCross;
t2b.y = test.y / normCross;
t2b.z = test.z / normCross;

XProd(sunIne, magIne, &test);
normCross = norm3L2(test);
t2i.x = test.x / normCross;
t2i.y = test.y / normCross;
t2i.z = test.z / normCross;

XProd(t2b, sunBody, &t3b);
XProd(t2i, sunIne,  &t3i);

printVecVals(sunBody);
printVecVals(t2b);
printVecVals(t3b);
printVecVals(sunIne);
printVecVals(t2i);
printVecVals(t3i);

// Organize here and create the rotation matrix matRot
// R = [t1b t2b t3b][t1i t2i t3i]^T each vector is a column vector pre-transpose
// double dotCheck = dot3(sunBody.x, t2b.x, t3b.x, sunIne.x, t2i.x, t3i.x);
double matRot[DIMS][DIMS];
// Column 0 of matRot
matRot[0][0] = dot3(sunBody.x, t2b.x, t3b.x, sunIne.x, t2i.x, t3i.x);
matRot[1][0] = dot3(sunBody.y, t2b.y, t3b.y, sunIne.x, t2i.x, t3i.x);
matRot[2][0] = dot3(sunBody.z, t2b.z, t3b.z, sunIne.x, t2i.x, t3i.x);

// column 1 of matRot
matRot[0][1] = dot3(sunBody.x, t2b.x, t3b.x, sunIne.y, t2i.y, t3i.y);
matRot[1][1] = dot3(sunBody.y, t2b.y, t3b.y, sunIne.y, t2i.y, t3i.y);
matRot[2][1] = dot3(sunBody.z, t2b.z, t3b.z, sunIne.y, t2i.y, t3i.y);

// column 2 of matRot
matRot[0][2] = dot3(sunBody.x, t2b.x, t3b.x, sunIne.z, t2i.z, t3i.z);
matRot[1][2] = dot3(sunBody.y, t2b.y, t3b.y, sunIne.z, t2i.z, t3i.z);
matRot[2][2] = dot3(sunBody.z, t2b.z, t3b.z, sunIne.z, t2i.z, t3i.z);


    for(int i = 0; i < DIMS; i++){
        for(int j = 0; j < DIMS; j++){
            printf("%g\t", matRot[i][j]);
        } // for - j - cols
        printf("\n");
    } // for - i - rows

// ADD CONVERSION TO QUATERNION HERE
// ================================================================================================ ⋀ Triad Algorithm

return 0;
} // int main()
// ================================================================================================ ⋀ Main Function


