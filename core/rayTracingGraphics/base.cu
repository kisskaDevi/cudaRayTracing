#include "base.h"
#include "material.h"

namespace base {

    __device__ vec4 color(ray r, size_t maxIterations, hitableList* list, curandState* local_rand_state) {
        vec4 color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
        hitRecord rec;
        for (; maxIterations > 0; maxIterations--) {
            if (r.getDirection().length2() != 0.0f && list->hit(r, 0.001f, FLT_MAX, rec)) {
                color = min(rec.mat->getAlbedo(), color);
                r = ray(rec.point, rec.mat->scatter(r, rec.normal, local_rand_state));
            }
            else {
                break;
            }
        }
        return  rec.mat && rec.mat->lightFound()
                ? vec4(color.x(), color.y(), color.z(), 1.0f)
                : vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __global__ void render(vec4* frameBuffer, size_t width, size_t height, camera* cam, curandState* randState, size_t hitCount, size_t raysCount, size_t samplesCount, hitableList* list)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        float u = 2.0f * float(i) / float(width) - 1.0f;
        float v = 2.0f * float(j) / float(height) - 1.0f;

        if ((i < width) && (j < height)) {
            int pixel_index = j * width + i;
            curandState local_rand_state = randState[pixel_index];

            for (size_t sampleIndex = 0; sampleIndex < samplesCount; sampleIndex++) {
                ray camRay = cam->getPixelRay(u, v, &local_rand_state);
                for (size_t rayIndex = 0; rayIndex < raysCount; rayIndex++) {
                    frameBuffer[pixel_index] += color(camRay, hitCount, list, &local_rand_state);
                }
            }
        }
    }
}