#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

class camera {
private:
    ray viewRay;
    vec4 horizontal;
    vec4 vertical;

    float matrixScale{ 0.04f };
    float matrixOffset{ 0.05f };
    float focus{ 0.049f };
    float apertura{ 0.005f };

public:
    __host__ __device__ camera(
        const ray viewRay,
        float aspect,
        float matrixScale,
        float matrixOffset,
        float focus,
        float apertura);

    __host__ __device__ camera(const ray viewRay, float aspect);

    __device__ ray getPixelRay(float u, float v, curandState* local_rand_state);

    static camera* create(const ray& viewRay, float aspect);
    static void destroy(camera* cam);
};
#endif
