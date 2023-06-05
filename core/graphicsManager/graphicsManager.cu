#include "graphicsManager.h"
#include "operations.h"

#include <string>
#include <time.h>
#include <cstdlib>

__global__ void createRandomState(unsigned long long seed, int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
}

void graphicsManager::destroy()
{
	checkCudaErrors(cudaFree(randState));
}

void graphicsManager::createInstance(size_t width, size_t height, rayTracingGraphics* graphics)
{
	this->width = width;
	this->height = height;

	checkCudaErrors(cudaMalloc((void**)&randState, width * height * sizeof(curandState)));

	this->graphics = graphics;
	this->graphics->setRandState(randState);
}

void graphicsManager::drawFrame(hitableList* list) {
	for (size_t i = 0; i < 10; i++)
	{
		dim3 blocks(width / xThreads + 1, height / yThreads + 1);
		dim3 threads(xThreads, yThreads);

		srand(time(0));
		createRandomState<<<blocks, threads>>>(rand(), width, height, randState);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		graphics->submit(list, xThreads, yThreads);

		Image::outPPM(graphics->getSwapChain(), width, height, "image" + std::to_string(i + 1) + ".ppm");
		std::cout << (i + 1) * 10 << " %" << std::endl;
	}
}