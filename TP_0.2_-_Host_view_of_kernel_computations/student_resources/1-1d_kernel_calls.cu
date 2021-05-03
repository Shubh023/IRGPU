#include <iostream>
#include <cstdlib>
#include <cmath>
#include "kernels.h"

#define cudaCheckError() {                                                                       \
  cudaError_t e=cudaGetLastError();                                                        \
  if(e!=cudaSuccess) {                                                                     \
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));        \
      exit(EXIT_FAILURE);                                                                  \
  }                                                                                        \
}

// Check that all values of array (which contains `length` float elements) are
// close to `expectedValue`
bool checkHostArray(float *array, float expectedValue, size_t length){
  float maxError = 0.0f;
  for (int i = 0; i < length; i++)
    maxError = fmax(maxError, fabs(array[i]-expectedValue));
  std::cout << "Max error: " << maxError << std::endl;
  return (maxError < 0.0001f);
}

int main(void)
{
  // 1<<20
  int N = 1<<20;  //< Number of elements in arrays (1M, you may want to lower this to begin)
  float *d_x;  //< Pointer to the 1D buffer we will manipulate 
 
  // Initialize grid and block sizes for later kernel launches.
  // Use as many threads as possible.
  //@@ Choose some values here, stick to 1D
  int threadsPerBlock = N;  // FIXME
  int blocksPerGrid = 1;  // FIXME

  // Array allocation on device
  //@@ Use cudaMalloc to perform the allocation.
  cudaMalloc(&d_x, N * sizeof(float)); // FIXME
  cudaCheckError();
 
  // Initialize the x and y arrays on the device
  const float firstValue = 1.f;
  //@@ Call the fill1D kernel to fill d_x with `firstValue`, see kernels.h for the API
  fill1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue, N * sizeof(float));  // FIXME
  // Wait for GPU to finish and check for errors
  cudaDeviceSynchronize();
  cudaCheckError();
  
  // Check for errors on device
  //@@ Call the check1D kernel to control device memory content, see kernels.h for API
  check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue, N * sizeof(float));  // FIXME
  // Wait for GPU to finish and check for errors
  //@@ call CUDA device synchronisation function
  //@@ ???
  cudaDeviceSynchronize();
 
  cudaCheckError();

  // Copy back the buffer to the host for inspection:
  //@@ Allocate a buffer on the host
  float *h_x = (float*) std::malloc(N * sizeof(float));  //FIXME
  //@@ Copy the buffer content from device to host
  //@@ use cudaMemcpy
  cudaMemcpy(&d_x, &h_x, N * sizeof(float), cudaMemcpyHostToDevice);  // FIXME
  cudaCheckError();

  // Check for errors (all values should be close to `firstValue`)
  std::cout << "First control..." << std::endl;
  bool noerror = checkHostArray(h_x, firstValue, N);
  
  // Now increment the array values by some other value
  const float otherValue = 10.f;
  //@@ Call the inc1D kernel to add `otherValue` to all values of our buffer, see kernels.h for API
  inc1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, otherValue, N * sizeof(float));
  // Wait for GPU to finish
  //@@ call CUDA device synchronisation function
  //@@ ???
  cudaDeviceSynchronize();

  cudaCheckError();

  // Check for errors on device
  //@@ Call the check1D kernel to control device memory content, see kernels.h for API
  check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue + otherValue, N * sizeof(float));  // FIXME
  // Wait for GPU to finish and check for errors
  //@@ call CUDA device synchronisation function
  //@@ ???
  cudaDeviceSynchronize();

  cudaCheckError();

  // Copy back the buffer to the host for inspection:
  //@@ Copy the buffer content from device to host (reuse previous buffer)
  cudaMemcpy(&d_x, &h_x, N * sizeof(float), cudaMemcpyHostToDevice);  // FIXME
  cudaCheckError();

  // Check for errors (all values should be close to `firstValue+otherValue`)
  std::cout << "Second control..." << std::endl;
  noerror &= checkHostArray(h_x, firstValue+otherValue, N);

  // Free memory
  //@@ free d_h using CUDA primitives 
  cudaFree(d_x);
  cudaCheckError();
  std::free(h_x);

  if (noerror) {
    printf("Test completed successfully.\n");
    return 0;
  } else {
    printf("WARNING there were some errors.\n");
    return 1;
  }
}
