#ifndef BASEH
#define BASEH

#include "camera.h"
#include "hitableList.h"
#include "buffer.h"

namespace base {

	__global__ void render(vec4* frameBuffer, size_t width, size_t height, camera* cam, curandState* randState, size_t hitCount, size_t raysCount, size_t samplesCount, hitableList* list);
}

#endif // !BASEH

