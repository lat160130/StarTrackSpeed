// Compiling and running this program:
//   nvcc -std=c++11 device-prop-test.cu && ./a.out
#include <chrono>
#include <iostream>
using namespace std;

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t status = call;                              \
    if(status != cudaSuccess) {                             \
      printf("FAIL: call='%s'. Reason:%s\n", #call,         \
             cudaGetErrorString(status));                   \
      return -1;                                            \
    }                                                       \
  } while (0)

int main(int argc, char** argv) {
  int devId;
  CUDA_CHECK(cudaGetDevice(&devId));
  cudaGetDevice(&devId);

  auto start = chrono::high_resolution_clock::now();
  cudaDeviceProp prop;
  for(int i = 0; i < 25; ++i) {
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));
  }
  cudaGetDeviceProperties(&prop, devId);
  auto end = chrono::high_resolution_clock::now();
  cout
    << "cudaGetDeviceProperties -> "
    << chrono::duration_cast<chrono::microseconds>(end - start).count() / 25.0
    << "us" << endl;

  int smemSize, numProcs;
  start = chrono::high_resolution_clock::now();
  for(int i = 0; i < 25; ++i) {
    CUDA_CHECK(cudaDeviceGetAttribute(&smemSize,
                                      cudaDevAttrMaxSharedMemoryPerBlock,
                                      devId));
    CUDA_CHECK(cudaDeviceGetAttribute(&numProcs,
                                      cudaDevAttrMultiProcessorCount,
                                      devId));
  }

  cudaDeviceGetAttribute(&numProcs,
    cudaDevAttrMultiProcessorCount,
    devId);
  end = chrono::high_resolution_clock::now();
  cout
    << "cudaDeviceGetAttribute -> "
    << chrono::duration_cast<chrono::microseconds>(end - start).count() / 25.0
    << "us" << endl;


cout << "device id = " << devId << endl;
cout << "cuDP.name      = " << prop.name << endl;
cout << "cuDP.totGloMem = " << prop.totalGlobalMem << endl;
cout << "cuDP.sMpBlock  = " << prop.sharedMemPerBlock << endl;
cout << "cuDP.regpBlock = " << prop.regsPerBlock << endl;
cout << "cuDP.warpSize  = " << prop.warpSize << endl;
cout << "cuDP.memPitch  = " << prop.memPitch << endl;
cout << "cuDP.maxTpB    = " << prop.maxThreadsPerBlock << endl;
cout << "cuDP.maxGS     = " << prop.maxGridSize << endl;
cout << "cuDP.clockR    = " << prop.clockRate << endl;
cout << "cuDP.totConstM = " << prop.totalConstMem << endl;
cout << "cuDP.major     = " << prop.major << endl;
cout << "cuDP.minor     = " << prop.minor << endl;
cout << "cuDP.pciBus    = " << prop.pciBusID << endl;
cout << "cuDP.maxTpP    = " << prop.maxThreadsPerMultiProcessor << endl;
  return 0;
}