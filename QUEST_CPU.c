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


char vecIn[] = "vectorInput.txt";
int returnCodeImport = -1;

returnCodeImport = importSunAndMagVecs(vecIn, &sunBody, &sunIne, &magBody, &magIne);



} // int main()
// ================================================================================================ ⋀ Main Function


