// This is the TRIAD algorithm rewritten for CUDA C.
// This code is not fully optimized.


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"
#include <stdio.h>
#include <stdlib.h>
#define N (2048 * 8)
#define THREADS_PER_BLOCK 512

// __global__ void dot3 (double )

