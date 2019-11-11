#include "add.cuh"
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

cudaError_t initCuda()
{
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return cudaStatus;
	}
	printf("Cuda init success!\n");
	return cudaStatus;
}
static void HandleError(cudaError_t err,const char *file,int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
	}
}
cudaError_t convWithCuda(float* src, float* dst, float* filter, int imageOutSize, int imageInSize, int filterSize)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_c = 0;
	cudaError_t cudaStatus;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, filterSize * filterSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_a, imageInSize *imageInSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, imageOutSize *imageOutSize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, src, imageInSize *imageInSize * sizeof(float),cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!a");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_c, filter, filterSize *  filterSize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!c");
		goto Error;
	}
	dim3 grid(1);
	dim3 block(imageOutSize, imageOutSize);
	// Launch a kernel on the GPU with one thread for each element.
	conv2MexCuda << <grid,block >> > (dev_a, dev_b, dev_c,imageOutSize,imageInSize,filterSize);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	HANDLE_ERROR(cudaStatus =cudaDeviceSynchronize());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(dst, dev_b, imageOutSize *imageOutSize * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!b");
		goto Error;
	}
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	return cudaStatus;
}
__global__ void conv2MexCuda(float* src, float* dst, float* filter, int imageOutSize, int imageInSize, int filterSize)
{
	int row = threadIdx.x;
	if (row < 0 || row > imageOutSize - 1)
		return;
	int col = threadIdx.y;
	if (col < 0 || col > imageOutSize - 1)
		return;
	int dstIndex = col * imageOutSize + row;
	int fSize = filterSize * filterSize;
	dst[dstIndex] = 0;
#pragma unroll
	for (int fy = 0; fy < filterSize; fy++) {
#pragma unroll
		for (int fx = 0; fx < filterSize; fx++) {
			float filterItem = filter[--fSize];
			float imageItem = src[row + fx + (fy + col)*imageInSize];
			dst[dstIndex] += filterItem*imageItem;
		}
	}
}
