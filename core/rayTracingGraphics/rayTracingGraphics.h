#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "vec4.h"
#include "camera.h"
#include "operations.h"
#include "hitableList.h"

class rayTracingGraphics {
private:
	vec4* bloomImage;
	vec4* colorImage;
	vec4* swapChain;

	size_t width;
	size_t height;

	camera* cam;
	curandState* randState;

public:
	rayTracingGraphics(size_t width, size_t height, camera* cam) {
		create(width, height, cam);
	}

	void create(size_t width, size_t height, camera* cam) {
		this->width = width;
		this->height = height;
		this->cam = cam;

		colorImage = Buffer::create(sizeof(vec4) * width * height);
		bloomImage = Buffer::create(sizeof(vec4) * width * height);
		swapChain = Buffer::create(sizeof(vec4) * width * height);
	}

	void destroy() {
		checkCudaErrors(cudaFree(colorImage));
		checkCudaErrors(cudaFree(bloomImage));
		checkCudaErrors(cudaFree(swapChain));
	}

	void setRandState(curandState* randState) {
		this->randState = randState;
	}

	void submit(hitableList* list, size_t xThreads, size_t yThreads);

	inline vec4* getSwapChain() {
		return swapChain;
	}
};

#endif // !RAYTRACINGGRAPHICS

