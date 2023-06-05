#ifndef GRAPHICSMANAGER_H
#define GRAPHICSMANAGER_H

#include "hitableList.h"
#include <curand_kernel.h>
#include "ray.h"
#include "operations.h"
#include "rayTracingGraphics.h"

class graphicsManager
{
private:
	size_t width;
	size_t height;
	size_t xThreads{ 4 };
	size_t yThreads{ 4 };

	curandState* randState{ nullptr };

	rayTracingGraphics* graphics{ nullptr };

public:
	graphicsManager(){}
	void destroy();

	void createInstance(size_t width, size_t height, rayTracingGraphics* graphics);

	void drawFrame(hitableList* list);
};

#endif
