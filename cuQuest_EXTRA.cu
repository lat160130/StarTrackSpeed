/*
<<<<<<< HEAD
// GENERAL MATRIX MULTIPLICATION
__global__ void MatrixMulCUDA(double* d_M, double* d_N, double* d_P, long width)
int row = blockIdx.y*width+threadIdx.y;
int col = blockIdx.x*width+threadIdx.x;
if(row<width && col <width) {
    float product_val = 0
    for(int k=0;k<width;k++) {
       product_val += d_M[row*width+k]*d_N[k*width+col];
    }
    d_p[row*width+col] = product_val;
 }

{

*/
/*
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(double *C, double *A, double *B, long wA, long wB) {
    /**
     * Matrix multiplication (CUDA Kernel) on the device: C = A * B
     * wA is A's width and wB is B's width
     */
    
      // Block index
      int bx = blockIdx.x;
      int by = blockIdx.y;
    
      // Thread index
      int tx = threadIdx.x;
      int ty = threadIdx.y;
    
      // Index of the first sub-matrix of A processed by the block
      int aBegin = wA * BLOCK_SIZE * by;
    
      // Index of the last sub-matrix of A processed by the block
      int aEnd   = aBegin + wA - 1;
    
      // Step size used to iterate through the sub-matrices of A
      int aStep  = BLOCK_SIZE;
    
      // Index of the first sub-matrix of B processed by the block
      int bBegin = BLOCK_SIZE * bx;
    
      // Step size used to iterate through the sub-matrices of B
      int bStep  = BLOCK_SIZE * wB;
    
      // Csub is used to store the element of the block sub-matrix
      // that is computed by the thread
      double Csub = 0;
    
      // Loop over all the sub-matrices of A and B
      // required to compute the block sub-matrix
      for (int a = aBegin, b = bBegin;
           a <= aEnd;
           a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    
        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
    
        // Synchronize to make sure the matrices are loaded
        __syncthreads();
    
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
    #pragma unroll
    
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          Csub += As[ty][k] * Bs[k][tx];
        }
    
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
      }
    
      // Write the block sub-matrix to device memory;
      // each thread writes one element
      int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
      C[c + wB * ty + tx] = Csub;
 } // __global__ void MatrixMulCUDA(double *C, double *A, double *B, int wA, int wB) 
// CREATE Z VECTOR TO IMPLEMENT INTO K VECTOR
__global__ void createZVec(double *Z, double *A, double *B, long rows){
    int tidx = threadIdx.x; // this will always equal to three.
    
    if (tidx < 3){
        Z[tidx] = 0;
        __syncthreads();

        for (int i = 0; i < rows; i++){ 
            Z[tidx] = Z[tidx] + ( (A[(tidx%3 + 1)*rows + i ] * B[i*NUMDIMS + ((tidx+2)%3)]) - (A[(tidx%3 + 2)*rows + i ] * B[i*NUMDIMS + ((tidx+1)%3)]) );
        } // end for i
        __syncthreads();

        Z[tidx] = pow(-1, tidx % 2) * Z[tidx];
        __syncthreads();

    } // end if tidx < 3
} // __global__ void createZVec(double *Z, double *A, double *B)

__global__ void modifyZ(double *Z, double rows){
    int tidx = threadIdx.x;
    if (tidx < 3)
        Z[tidx] = Z[tidx] / rows;

} // modifyZ(double *Z, double rows)
=======
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e
__global__ void initScalars(double *scalarVector, int numScal){
    int tidx = threadIdx.x;

    scalarVector[tidx] = 1;
    __syncthreads();
} // __global__ void initScalars(double *scalarVector, int numScal)

__global__ void matTrace(double *mat, int columns, double *scalarMat,  int arrPosn){
    // mat - matrix to find the trace of
     // columns - number of rows in mat (3x3) is coming in ALWAYS, so columns == 3
    // scalarMat, this matrix holds all the scalar values in the algorithm for simple access for gpu shared memory
    // arrPosn - position in the array that sigma will hold

    if (columns != 3){
        perror("Columns must equal 3!\n");
    } // end if columns

    int tidx = threadIdx.x;
    double sigma = 0;
    for (int i = 0; i < rows; i++){
        sigma = sigma + mat[i*columns + i];
    } // end for i
    scalarMat[arrPosn] = .5 * sigma;
} // __global__ void matTrace(double *mat, long long rows)

__global__ void genCofactor(double *mat, double *cofM, int col){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    if (tidx < 3 || tidy < 3){
        // int a = ((tidx+1)%3) ;
        // int b = ((tidy+1)%3) ;
        // int c = ((tidx+2)%3) ;
        // int d = ((tidy+2)%3) ;
        
        // This construct already has the negative multiplication by nature of the pointer math.
        cofM[tidx][tidy] = mat[((tidx+1)%3) * col + ((tidy+1)%3)] * mat[((tidx+2)%3) * col + ((tidy+2)%3)]
                         - mat[((tidx+1)%3) * col + ((tidy+2)%3)] * mat[((tidx+2)%3) * col + ((tidy+1)%3)];
    } // if (tidx < 3 || tidy < 3)
} // __global__ void genCofactor(double *mat, double *cofM)
 
__global__ void createDelta(double *S, double *scalarVec, int arrPosn){
     if (threadIdx.x != 1)
        perror("This function must unfortuneatly be single threaded...\n");

    scalarVec[arrPosn] = (S[0] * (S[4]*S[8] - S[5]*S[7])) 
                       - (S[1] * (S[3]*S[8] - S[5]*S[6]))
                       + (S[2] * (S[3]*S[7] - S[4]*S[6]));
} //  __global__ void createDelta(double *S, double *scalarVec, int arrPosn)

__global__ void generate_a_b_c_d(double *scalarVec, double *S, double *S2, double *Z, int arrPosn){
    // cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
    int tidx = threadIdx.x;

    // Create a -3
    if (tidx == 0){
        scalarVec[arrPosn + tidx] = scalarVec[0] * scalarVec[0] - scalarVec[1];
    } // end if (tidx == 0)

    // Create b - 4
    if (tidx == 1){
        scalarVec[arrPosn + tidx] = scalarVec[0] * scalarVec[0] + (Z[0]*Z[0] + Z[1]*Z[1] +Z[2]*Z[2]);
    } // end if (tidx == 1)

    // Create c - 5
    if (tidx == 2){
        scalarVec[arrPosn + tidx] = scalarVec[3] + 

            ( Z[0]* (S[0]*Z[0] + S[1]*Z[1] + S[2]*Z[2]) )
          + ( Z[1]* (S[3]*Z[0] + S[4]*Z[1] + S[5]*Z[2]) )
          + ( Z[2]* (S[6]*Z[0] + S[7]*Z[1] + S[8]*Z[2]) );
    } // end if (tidx == 2)

    if (tidx == 3){
        scalarVec[arrPosn + tidx] = 

            (Z[0]* (S2[0]*Z[0] + S2[1]*Z[1] + S2[2]*Z[2]))
          + (Z[1]* (S2[3]*Z[0] + S2[4]*Z[1] + S2[5]*Z[2]))
          + (Z[2]* (S2[6]*Z[0] + S2[7]*Z[1] + S2[8]*Z[2]));
    } // end  if (tidx == 3)
    __syncthreads();

} // __global__ void generate_a_b_c_d(double *scalarVec, double *S, double *Z, int arrPosn)

__global__ void genOptimalLambda(double *scalarVec, int arrPosn){
    // cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
    int tidx = threadIdx.x; // tidx will only be equal to zero;
    if (tidx != 0)
        perror("TIDX MUST BE EQUAL TO ZERO!!!!\n");

    // -- Newton's method for convergence of lambda --------------------- lambda
    double tol = 1e-12;
    int max_it = 100;
    int iters = 0;
    double error = tol + 1;

    double l2;
    double l = scalarVec[arrPosn];
    while (error > tol && iters < max_it){
        l2 = l - (((l*l*l*l) - (a+b)*l*l - c*l + (a*b + c*sigma - d)) / (4*l*l*l - 2*l*(a+b) -c));
        error = fabs(l2 - l);
        l = l2;
        iters++;
        
} // end while error > tol && it < max_it
    scalarVec[arrPosn] = l;
// ------------------------------------------------------------------ lambda

} // 

__global__ void genBetaAlphaGamma(double *scalarVec, int arrPosn){
    // arrPosn will always == 8
    if (arrPosn != 8)
        perror('arrPosn must = 8\n');
    // cuScalarArr = [sigma0, kappa1, delta2, a3, b4, c5, d6, lambda7, beta8, alpha9, gamma10]
    int tidx = threadIdx.x; // tidx will only be equal to zero;
    if (tidx != 0)
        perror("TIDX MUST BE EQUAL TO ZERO!!!!\n");

    scalarVec[arrPosn]     = scalarVec[7]           - scalarVec[0]; // beta = lambda - sigma;
    scalarVec[arrPosn + 1] = pow(scalarVec[7],2)    - pow(scalarVec[0],2) + scalarVec[1]; // alpha = lambda^2 - sigma^2 + kappa;
    scalarVec[arrPosn + 2] = scalarVec[arrPosn + 1] * (scalarVec[7] * scalarVec[0]) - scalarVec[2]; // gamma = alpha * (lambda + sigma) - delta; 
} // 

__global__ void createXVector(double *X, double *scalarVec, double *S, double *S2){
    int tidx = threadIdx.x; // this will equal to three

} // __global__ void createXVector(double *X, double *scalarVec, double *S, double *S2)



// CREATE SCALARS
cudaMalloc((void**), &cuScalarArr, NUMSCALARS*sizeof(double));
initScalars <<<1,NUMSCALARS>>>(cuScalarArr, NUMSCALARS); // this initializes all values in the array to 1;

// -- CREATE SIGMA -------------------------------------------------- SIGMA
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
matTrace <<<1,1>>>(cuB, NUMDIMS, cuScalarArr,  0);
// ------------------------------------------------------------------ SIGMA

// -- CREATE KAPPA -------------------------------------------------- KAPPA
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
cudaMalloc((void**), &cuCofactorS, sizeDCM);
genCofactor <<<1, threads3x3>>> (cuS, cuCofactorS, NUMDIMS);
matTrace <<<1,1>>> (cuCofactorS, NUMDIMS, cuScalarArr, 1);
// ------------------------------------------------------------------ KAPPA

// -- CREATE DELTA -------------------------------------------------- DELTA
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
createDelta <<<1,1>>> (cuS, cuScalarArr, 2);
// ------------------------------------------------------------------ DELTA

// -- CREATE a,b,c,d ------------------------------------------------ a,b,c,d
// cuScalarArr = [sigma, kappa, delta, a, b, c, d, lambda, beta, alpha, gamma]
// this will take scalar arr posn, 3,4,5,6, so we will enter only, 3 and populate 4,5,6 in the function
// have 4 threads going at the same time doing different things
generate_a_b_c_d <<<1,4>>> (cuScalarArr, cuS, cuS2, cuZ, 3);
// ------------------------------------------------------------------ a,b,c,d

// -- CREATE LAMBDA ------------------------------------------------- LAMBDA
genOptimalLambda <<<1,1>>> (cuScalarArr, 7); // VERY INEFFICIENT
// ------------------------------------------------------------------ LAMBDA

// -- CREATE ALPHA BETA GAMMA --------------------------------------- ALPHA BETA GAMMA
genBetaAlphaGamma <<<1, 1>>> (cuScalarArr, 8); // VERY INEFFICIENT
// ------------------------------------------------------------------ ALPHA BETA GAMMA

// -- CREATE X ------------------------------------------------------ X VECTOR
cudaMalloc((void**), &cuX, NUMDIMS*sizeof(double));
// ------------------------------------------------------------------ X VECTOR


<<<<<<< HEAD
=======
*/
>>>>>>> ce8fc7912c9fe3fd2e697911130e3995af22d28e
