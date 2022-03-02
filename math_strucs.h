// Author: Leon Tannenbaum
// Date of initial write: February 7th 2022
// Goal: This header file will hold math structures such as vectors and matrices to be easy for
// other programs to use
//
//

// == Include Block =============================================================================== Include Block
#include <stdio.h>
#include <string.h>
// ================================================================================================ Include Block


/********************************************************************/
/* Struct Vector3 creates a 3d vector.  Each of the three           */
/* components are double precision.                                 */                  
/********************************************************************/
typedef struct Vectors {
    double x;
    double y;
    double z;
} vector3;
// struct Vectors is the same as vector3

