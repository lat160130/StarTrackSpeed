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


// == List of Functions =========================================================================== ⋁ List of functions
// 1. importSunAndMagVecs(char *vectorTxtFile)
// ================================================================================================ ⋀ Include Block


// == 1. importSunAndMagVecs ====================================================================== ⋁ importSunAndMagVecs
importSunAndMagVecs(char *vectorTxtFile, vector3 *sunBody, vector3 *sunIne, vector3 *magBody, vector3 *magIne){
    FILE *fpSunMagVecs;

    fpSunMagVecs = fopen(vectorTxtFile, "r"); // read only

    if (!fpSunMagVecs) {// check if opens
        printf("Text file could not be found.")
        return -1;
    }

    fscanf(fpSunMagVecs, "%f %f %f", &sunBody.x, &sunBody.y, &sunBody.z)
    
} // importSunAndMagVecs(char *vectorTxtFile){
// ================================================================================================ ⋀ importSunAndMagVecs