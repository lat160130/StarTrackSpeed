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
// 3. double dot3(double x, double y, double z, double i, double j, double k)
// 4. double dotN(double arr1[], double arr2[], int n);
// 4. double standDevN
// ================================================================================================ ⋀ Include Block


// Prototype list
vector3 xProd(vector3 V1, vector3 V2);
double norm3L2(vector3 V1);
double dot3(double x, double y, double z, double i, double j, double k);
double dotN(double arr1[], double arr2[], int n);

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
  double out = (x*i) + (y*j) + (z*k);
return out;

} // Function dot3
// ================================================================================================ ⋀ dot3

// == 4. dotN ===================================================================================== ⋁ dotN
double dotN(double arr1[], double arr2[], int n){
  int sum = 0;
  for (int i = 0; i < n; i++){
      sum = sum + (arr1[i] * arr2[i]);
  } // 
  return sum;
} // Function dotN
// ================================================================================================ ⋀ dotN

/*
// == 4. standDevN ================================================================================ ⋁ standDevN
double standDevN(double *array, int arrLen){
  
} // Function standDevN
// ================================================================================================ ⋀ standDevN

// == 4. rot2Quat ================================================================================= ⋁ rot2Quat
void (double *matrix, quat q){

} // Function rot2Quat
// ================================================================================================ ⋀ rot2Quat


*/