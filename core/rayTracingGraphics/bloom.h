#ifndef BLOOMH
#define BLOOMH

#include "camera.h"
#include "hitableList.h"

namespace bloom {

	__global__ void render(vec4* frameBuffer, size_t width, size_t height, camera* cam, curandState* randState, size_t hitCount, size_t raysCount, size_t samplesCount, hitableList* list);
}

#endif // !BLOOMH