#include "rayTracingGraphics.h"
#include "base.h"
#include "bloom.h"

void rayTracingGraphics::submit(hitableList* list, size_t xThreads, size_t yThreads)
{
    dim3 blocks(width / xThreads + 1, height / yThreads + 1);
    dim3 threads(xThreads, yThreads);

    Image::clear(swapChain, width * height);
    base::render << <blocks, threads >> > (colorImage, width, height, cam, randState, 8, 50, 8, list);
    bloom::render << <blocks, threads >> > (bloomImage, width, height, cam, randState, 8, 50, 8, list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Image::combine(swapChain, colorImage, width * height);
    Image::combine(swapChain, bloomImage, width * height);
    Image::normalize(swapChain, width * height);
}