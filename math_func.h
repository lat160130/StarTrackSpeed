// Author: Leon Tannenbaum
// Date of initial write: February 7th 2022
// Goal: This header file will hold math functions needed to make c code easier to write.
// 
//

// Note: This code contains copying of mathmatics functions written by David Tannenbaum.
//       He has allowed its use for this research project on the condition that he is alerted
//       if the licensing changes.


#include <stdio.h>   /* Standard I/O library header file */
#include <stdlib.h>  /* Standard library header file */
#include <math.h>    /* Standard math library header file */
#include "math_strucs.h"



/********************************************************************/
/* Function XProd returns the cross products of two vectors.        */
/* The input vectors and the resultant vector are all in 3d.        */
/********************************************************************/
vector3 XProd(vector3 V1, vector3 V2)
{
  vector V3;
 
  V3.x = V1.y*V2.z - V2.y*V1.z;
  V3.y = V2.x*V1.z - V1.x*V2.z;
  V3.z = V1.x*V2.y - V2.x*V1.y;
 
  return V3;
} /* Function XProd */
 