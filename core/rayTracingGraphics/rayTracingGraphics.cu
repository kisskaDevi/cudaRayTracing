#include "rayTracingGraphics.h"
#include "base.h"
#include "bloom.h"

void rayTracingGraphics::submit(hitableList* list, size_t xThreads, size_t yThreads)
{
    dim3 blocks(static_cast<unsigned int>(width / xThreads + 1), static_cast<unsigned int>(height / yThreads + 1));
    dim3 threads(static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads));

    base::render << <blocks, threads >> > (colorImage.get(), width, height, cam, randState, 8, 10, 8, list);
    bloom::render << <blocks, threads >> > (bloomImage.get(), width, height, cam, randState, 8, 10, 8, list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Image::clear << <blocks, threads >> > (swapChain.get(), width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    Image::combine << <blocks, threads >> > (swapChain.get(), colorImage.get(), width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    Image::combine << <blocks, threads >> > (swapChain.get(), bloomImage.get(), width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    Image::normalize << <blocks, threads >> > (swapChain.get(), width, height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}