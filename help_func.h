// Author: Leon Tannenbaum
// Date of initial write: February 14th 2022

// This header file will provide helper functions to smooth integration for c functions.
// C doesn't have the capabilities to make life easy for input and output the functions.
//
// 


// == Include Block =============================================================================== ⋁ Include Block
#include <stdio.h>
#include <string.h>
#include "math_strucs.h"
// ================================================================================================ ⋀ Include Block

// == Constants =================================================================================== ⋁ Constants
# define DIMS 3    // of dimensions
# define NUMVECS 4 // number of vectors, 2 sun, 2 magnetic
// ================================================================================================ ⋀ Constants


// == List of Functions =========================================================================== ⋁ List of functions
// 1. importSunAndMagVecs(char *vectorTxtFile, vector3 sunBody, vector3 sunIne, vector3 magBody, vector3 magIne)
// 2. printVecVals(vector3 inVector)
// ================================================================================================ ⋀ Include Block


// == 1. importSunAndMagVecs ====================================================================== ⋁ 1. importSunAndMagVecs
int importSunAndMagVecs(char *vectorTxtFile, vector3 sunBody, vector3 sunIne, vector3 magBody, vector3 magIne){
    FILE *fpSunMagVecs;
    double sunMagMat[NUMVECS][DIMS];

    fpSunMagVecs = fopen(vectorTxtFile, "r"); // read only

    if (!fpSunMagVecs) {// check if opens
        printf("Text file could not be found.");
        return -1;
    } // if (!fpSunMagVecs)

    // fscanf(fpSunMagVecs, "%f %f %f", &sunBody.x, &sunBody.y, &sunBody.z)
    
    // == Read in four lines from vectorTextfile =========================
    for(int i = 0; i < NUMVECS; i++){
    fscanf(fpSunMagVecs, "%lf %lf %lf", &sunMagMat[i][0], &sunMagMat[i][1], &sunMagMat[i][2]);
    } // while (!feof (fpSunMagVecs)){
    // ===================================================================

    // Create sunBody vector
    sunBody.x = sunMagMat[0][0]; 
    sunBody.y = sunMagMat[0][1];
    sunBody.z = sunMagMat[0][2];

    // Create sunIne vector
    sunIne.x = sunMagMat[1][0]; 
    sunIne.y = sunMagMat[1][1];
    sunIne.z = sunMagMat[1][2];

    // Create magBody vector
    magBody.x = sunMagMat[2][0]; 
    magBody.y = sunMagMat[2][1];
    magBody.z = sunMagMat[2][2];

    // Create magIne vector
    magIne.x = sunMagMat[3][0]; 
    magIne.y = sunMagMat[3][1];
    magIne.z = sunMagMat[3][2];

 
    
    float a = sunMagMat[0][0];
    sunBody.x = a;

    return 0;
} // importSunAndMagVecs(char *vectorTxtFile){


// ================================================================================================ ⋀ 1. importSunAndMagVecs

// == 2. printVecVals ============================================================================= ⋁ 2. printVecVals
void printVecVals(vector3 inVector){
printf("vec.x = %lf,\t vec.y = %lf,\t vec.z = %lf\n", inVector.x, inVector.y, inVector.z);
return;
} // void printVecVals(vector3 inVector){
// ================================================================================================ ⋀ 2. printVecVals

