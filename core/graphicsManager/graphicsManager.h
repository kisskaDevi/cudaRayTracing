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
	size_t xThreads{ 8 };
	size_t yThreads{ 4 };

	curandState* randState{ nullptr };

	rayTracingGraphics* graphics{ nullptr };

public:
	graphicsManager(){}
	graphicsManager(rayTracingGraphics* graphics) {
		createInstance(graphics);
	}
	void destroy();

	void createInstance(rayTracingGraphics* graphics);

	void drawFrame(hitableList* list);
};

#endif
