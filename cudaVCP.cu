#include <stdio.h>

#define TV1 1
#define TV2 2
const size_t N = 4096;    // number of 3D vectors
const int blksize = 192;  // choose as multiple of 3 and 32, and less than 1024
typedef float mytype;

using namespace std;

//pairwise vector cross product
template <typename T>
__global__ void vcp(const T * __restrict__ vec1, const T * __restrict__ vec2, T * __restrict__ res, const size_t n){

  __shared__ T sv1[blksize];
  __shared__ T sv2[blksize];
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  while (idx < 3*n){ // grid-stride loop
    // load shared memory using coalesced pattern to global memory
    sv1[threadIdx.x] = vec1[idx];
    sv2[threadIdx.x] = vec2[idx];
    // compute modulo/offset indexing for thread loads of shared data from vec1, vec2
    int my_mod = threadIdx.x%3;   // costly, but possibly hidden by global load latency
    int off1 = my_mod+1;
    if (off1 > 2) off1 -= 3;
    int off2 = my_mod+2;
    if (off2 > 2) off2 -= 3;
    __syncthreads();
    // each thread loads its computation elements from shared memory
    T t1 = sv1[threadIdx.x-my_mod+off1];
    T t2 = sv2[threadIdx.x-my_mod+off2];
    T t3 = sv1[threadIdx.x-my_mod+off2];
    T t4 = sv2[threadIdx.x-my_mod+off1];
    // compute result, and store using coalesced pattern, to global memory
    res[idx] = t1*t2-t3*t4;
    idx += gridDim.x*blockDim.x;}  // for grid-stride loop
}

int main(){

  mytype *h_v1, *h_v2, *d_v1, *d_v2, *h_res, *d_res;
  h_v1  = (mytype *)malloc(N*3*sizeof(mytype));
  h_v2  = (mytype *)malloc(N*3*sizeof(mytype));
  h_res = (mytype *)malloc(N*3*sizeof(mytype));
  cudaMalloc(&d_v1,  N*3*sizeof(mytype));
  cudaMalloc(&d_v2,  N*3*sizeof(mytype));
  cudaMalloc(&d_res, N*3*sizeof(mytype));
  for (int i = 0; i<N; i++){
    h_v1[3*i]    = 1;
    h_v1[3*i+1]  = 1;
    h_v1[3*i+2]  = 1;

    h_v2[3*i]    = 0;
    h_v2[3*i+1]  = 2;
    h_v2[3*i+2]  = 0;
    
    h_res[3*i]   = 0;
    h_res[3*i+1] = 0;
    h_res[3*i+2] = 0;}

  // value printing
  for (int i = 0; i < 3; i++){
    printf("h_v1  = %f\n", h_v1);
    printf("h_v2  = %f\n", h_v2);
    printf("h_res = %f\n", h_res);
  } // 


  cudaMemcpy(d_v1, h_v1, N*3*sizeof(mytype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v2, h_v2, N*3*sizeof(mytype), cudaMemcpyHostToDevice);
  vcp<<<(N*3+blksize-1)/blksize, blksize>>>(d_v1, d_v2, d_res, N);
  cudaMemcpy(h_res, d_res, N*3*sizeof(mytype), cudaMemcpyDeviceToHost);

  // value printing
  for (int i = 0; i < 3; i++){
    printf("h_v1  = %lf\n", h_v1);
    printf("h_v2  = %lf\n", h_v2);
    printf("h_res = %lf\n", h_res);
  } // 

  // verification
  for (int i = 0; i < N; i++) if ((h_res[3*i] != 0) || (h_res[3*i+1] != 0) || (h_res[3*i+2] != TV1*TV2)) { printf("mismatch at %d, was: %f, %f, %f, should be: %f, %f, %f\n", i, h_res[3*i], h_res[3*i+1], h_res[3*i+2], (float)0, (float)0, (float)(TV1*TV2)); return -1;}
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));
  return 0;

}