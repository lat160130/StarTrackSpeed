/*
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


*/