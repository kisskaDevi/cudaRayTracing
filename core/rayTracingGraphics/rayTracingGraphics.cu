#include "rayTracingGraphics.h"
#include "base.h"
#include "bloom.h"

void rayTracingGraphics::submit(hitableList* list, size_t xThreads, size_t yThreads)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1);
    dim3 threads(xThreads, yThreads);

    Image::clear << <blocks, threads >> > (swapChain, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    base::render << <blocks, threads >> > (colorImage, width, height, cam, randState, 20, 50, 16, list);
    bloom::render << <blocks, threads >> > (bloomImage, width, height, cam, randState, 20, 50, 16, list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Image::combine << <blocks, threads >> > (swapChain, colorImage, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    Image::combine << <blocks, threads >> > (swapChain, bloomImage, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    Image::normalize << <blocks, threads >> > (swapChain, width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}