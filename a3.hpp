/*  Pratik
 *  Agarwal
 *  pagarwal
 */

#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <iostream>
#ifndef A3_HPP
#define A3_HPP
#define numThreads 1024

__global__ void Kernel(float *d_x , float *d_y , int N, float h)
{
    //__shared__ float *sharedVar;
    float sum=0;
    __shared__ testData[];
    int I_dx = threadIdx.x+blockIdx.x*blockDim.x;
    testData[I_dx] = d_x[I_dx];
    __syncthreads();
    if( I_dx < N)
    {
        for(int i = 0;i < N;i++)
        {
            sum = sum+((1/sqrt(2*3.14))* exp(-(((d_x[I_dx]-testData[i])/h)*(d_x[I_dx]-testData[i])/2)));
        }
        d_y[I_dx] = sum/(N * h);
    }
}

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {

    int numBlocks = ((numThreads + n - 1)/numThreads);

    float *d_x = NULL;
    float *d_y = NULL;
    cudaMalloc(&d_x,sizeof(float)* n);
    cudaMalloc(&d_y,sizeof(float)* n);
    cudaMemcpy(d_x, x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    dim3 d_numThreads(numThreads);
    dim3 d_numBlocks(numBlocks);

    Kernel<<<d_numBlocks , d_numThreads>>>(d_x, d_y, n, h);

    cudaMemcpy(y.data(), d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
    for (int z=0;z<n;z++)
      print("%f\n", y[z]);
    cudaFree(d_x);
    cudaFree(d_y);

} // gaussian_kde

#endif // A3_HPP
