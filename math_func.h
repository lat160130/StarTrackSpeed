// Author: Leon Tannenbaum
// Date of initial write: February 7th 2022
// Goal: This header file will hold math functions needed to make c code easier to write.
// 
//

// Note: This code contains copying of mathmatics functions written by David Tannenbaum.
//       He has allowed its use for this research project on the condition that he is alerted
//       everytime the licensing changes.

#include <stdio.h>   /* Standard I/O library header file */
#include <stdlib.h>  /* Standard library header file */
#include <math.h>    /* Standard math library header file */
#include "math_strucs.h"

// == List of Functions =========================================================================== ⋁ List of functions
// 1. void XProd(vector3 V1, vector3 V2, vector3 *V3)
// 2. double norm3L2(vector3 v1)
// ================================================================================================ ⋀ Include Block


// Prototype list
// vector3 xProd(vector3 V1, vector3 V2);


/********************************************************************/
/* Function XProd returns the cross products of two vectors.        */
/* The input vectors and the resultant vector are all in 3d.        */
/********************************************************************/
// == 1. XProd ==================================================================================== ⋁ XProd
void XProd(vector3 V1, vector3 V2, vector3 *V3) {
  // v3 = v1 x v2

  
  V3->x = V1.y*V2.z - V2.y*V1.z;
  V3->y = V2.x*V1.z - V1.x*V2.z;
  V3->z = V1.x*V2.y - V2.x*V1.y;
 
  return;
} /* Function XProd  */
// ================================================================================================ ⋀ XProd

// == 2. norm3L2 ================================================================================== ⋁ norm3L2
double norm3L2(vector3 V1){
  double output = pow(V1.x, 2) + pow(V1.y, 2) + pow(V1.z, 2);
  return sqrt(output);

} // Function norm3L2
// ================================================================================================ ⋀ norm3L2

// == 3. dot3 ===================================================================================== ⋁ dot3
double dot3(double x, double y, double z, double i, double j, double k){
return ((x*i) + (y*j) + (z*k));

} // Function dot3
// ================================================================================================ ⋀ dot3