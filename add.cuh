#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

static void HandleError(cudaError_t err, const char *file, int line);
cudaError_t initCuda();//初始化cuda
cudaError_t convWithCuda(float* src, float* dst, float* filter, int imageOutSize, int imageInSize, int filterSize);
__global__ void conv2MexCuda(float* src, float* dst, float* filter, int imageOutSize, int imageInSize, int filterSize);//cuda 卷积函数
// Helper function for using CUDA to add vectors in parallel.