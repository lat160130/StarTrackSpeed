#include "iostream"
#include "fstream"      
#include "string"   
using namespace std;


#define NUMDIM 3

int main(int argc, char *argv[]) {
int rows = atoi(argv[1]);
ifstream in("vectorInput.txt");

int x,y;
double **mat = new double*[rows];
for(x = 0; x < rows; x++)
    mat[x] = new double[NUMDIM];


if (!in){
    cout << "Cannot open vectorInput.txt" << endl;
    return -1; // failure to open
} // if in

for (x = 0; x < rows; x++) {
    for (y = 0; y < NUMDIM; y++) {
        in >> mat[x][y];
    } // end for x
} // end for y

for (x = 0; x < rows; x++) {
    for (y = 0; y < NUMDIM; y++) {
        printf("(%d,%d) = %lf\t", x,y, mat[x][y]);
    } // end for x
    cout << endl;
} // end for y

for (x = 0; x < rows; x++)
    delete[] mat[x];
delete[] mat;


return -1;
} // end int main


