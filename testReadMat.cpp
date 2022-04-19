#include <stdio.h>
#include <stdlib.h>
// #include "help_func.h"
#include <malloc.h>

#define NUMDIM 3

int main(int argc, char *argv[]) {

int rows = atoi(argv[1]); // the number of rows we want
printf("rows = %d\n", rows);
int i, j;

double** mat=malloc(100*sizeof(double*));
for(i = 0; i < 100; ++i);
    mat[i] = malloc(100*sizeof(double));


int sRows = sizeof(mat) / sizeof (mat[0]);
int sCol  = sizeof(mat[0]) / sizeof(mat[0][0]);

printf("sRows = %d\t sCol = %d\n", sRows, sCol);

FILE *file = fopen("vectorInput.txt", "r");
double in0, in1, in2;
for (i = 0; i < rows; i++){
        fscanf(file, "%lf %lf %lf", &in0, &in1, &in2);
        mat[i][0] = in0; mat[i][1] = in1; mat[i][2] = in2;
} // end for i



for (i = 0; i < rows; i++){
    for (j = 0; j < NUMDIM; j++){
        printf("%lf\t", mat[i][j]);
    } // end for j
    printf("\n");
} // end for i


// FREE BLOCK
for(i = 0; i < rows; ++i);
    free(mat[i]);
free(mat);


rewind(file);
return -1;
} // end int main
