#ifndef RAYTRACINGGRAPHICS
#define RAYTRACINGGRAPHICS

#include "vec4.h"
#include "camera.h"
#include "hitableList.h"
#include "buffer.h"

class rayTracingGraphics {
private:
	buffer<vec4> bloomImage;
	buffer<vec4> colorImage;
	buffer<vec4> swapChain;

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

		colorImage = buffer<vec4>(width * height);
		bloomImage = buffer<vec4>(width * height);
		swapChain  = buffer<vec4>(width * height);
	}

	void destroy() {
		colorImage.destroy();
		bloomImage.destroy();
		swapChain.destroy();
	}

	void setRandState(curandState* randState) {
		this->randState = randState;
	}

	void submit(hitableList* list, size_t xThreads, size_t yThreads);

	inline vec4* getSwapChain() {
		return swapChain.get();
	}
	size_t getWidth() const {
		return width;
	}
	size_t getHeight() const {
		return height;
	}
};

#endif // !RAYTRACINGGRAPHICS

