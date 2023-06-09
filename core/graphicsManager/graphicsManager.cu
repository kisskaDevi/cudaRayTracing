#include "graphicsManager.h"
#include "operations.h"

#include <string>
#include <time.h>
#include <cstdlib>

__global__ void initRandomState(unsigned long long seed, int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < max_x && j < max_y) {
		int pixel_index = j * max_x + i;
		curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
	}
}

void graphicsManager::destroy()
{
	checkCudaErrors(cudaFree(randState));
}

void graphicsManager::createInstance(rayTracingGraphics* graphics)
{
	checkCudaErrors(cudaMalloc((void**)&randState, graphics->getWidth() * graphics->getHeight() * sizeof(curandState)));

	this->graphics = graphics;
	this->graphics->setRandState(randState);
}

void graphicsManager::drawFrame(hitableList* list) {
	dim3 blocks(static_cast<unsigned int>(graphics->getWidth() / xThreads + 1), static_cast<unsigned int>(graphics->getHeight() / yThreads + 1));
	dim3 threads(static_cast<unsigned int>(xThreads), static_cast<unsigned int>(yThreads));

	srand(static_cast<unsigned int>(time(0)));
	initRandomState<<<blocks, threads>>>(rand(), static_cast<int>(graphics->getWidth()), static_cast<int>(graphics->getHeight()), randState);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	graphics->submit(list, xThreads, yThreads);
}