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

/*
Getting GPU Data.
There is 1 device supporting CUDA
Device 0 name: NVIDIA GeForce RTX 3060 Ti
 Computational Capabilities: 8.6
 Maximum global memory size: 7979
 Maximum constant memory size: 64
 Maximum shared memory size per block: 48
 Maximum block dimensions: 1024 x 1024 x 64
 Maximum grid dimensions: 2147483647 x 65535 x 65535
 Warp size: 32
End of GPU data gathering.

*/

int main(void)
{
  int N = 1<<20;;  //< Number of elements in arrays (1M, you may want to lower this to begin)
  float *d_x;  //< Pointer to the 1D buffer we will manipulate 
  printf("%d\n", N);

  // Initialize grid and block sizes for later kernel launches.
  // Use as many threads as possible.
  //@@ Choose some values here, stick to 1D
  int threadsPerBlock = 256;  // FIXME
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // FIXME

  // Array allocation on device
  //@@ Use cudaMalloc to perform the allocation.
  cudaMalloc(&d_x, N * sizeof(float)); // FIXME
  cudaCheckError();
 
  // Initialize the x and y arrays on the device
  const float firstValue = 1.f;
  //@@ Call the fill1D kernel to fill d_x with `firstValue`, see kernels.h for the API
  fill1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue, N);  // FIXME
  // Wait for GPU to finish and check for errors
  cudaDeviceSynchronize();
  cudaCheckError();
  
  // Check for errors on device
  //@@ Call the check1D kernel to control device memory content, see kernels.h for API
  check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue, N);  // FIXME
  // Wait for GPU to finish and check for errors
  //@@ call CUDA device synchronisation function
  cudaDeviceSynchronize(); //@@ ???
   
  cudaCheckError();

  // Copy back the buffer to the host for inspection:
  //@@ Allocate a buffer on the host
  float *h_x = (float*) std::malloc(N * sizeof(float));  //FIXME
  //@@ Copy the buffer content from device to host
  //@@ use cudaMemcpy
  cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);  // FIXME
  cudaCheckError();

  // Check for errors (all values should be close to `firstValue`)
  std::cout << "First control..." << std::endl;
  bool noerror = checkHostArray(h_x, firstValue, N);
  
  // Now increment the array values by some other value
  const float otherValue = 10.f;
  //@@ Call the inc1D kernel to add `otherValue` to all values of our buffer, see kernels.h for API
  inc1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, otherValue, N);
  // Wait for GPU to finish
  //@@ call CUDA device synchronisation function
  cudaDeviceSynchronize(); //@@ ???

  cudaCheckError();

  // Check for errors on device
  //@@ Call the check1D kernel to control device memory content, see kernels.h for API
  check1D<<<blocksPerGrid, threadsPerBlock>>>(d_x, firstValue + otherValue, N);  // FIXME
  // Wait for GPU to finish and check for errors
  //@@ call CUDA device synchronisation function
  //@@ ???
  cudaDeviceSynchronize();

  cudaCheckError();

  // Copy back the buffer to the host for inspection:
  //@@ Copy the buffer content from device to host (reuse previous buffer)
  cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);  // FIXME
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
