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

    __host__ __device__ vec4 getHorizontal(const vec4& d) {
        float D = std::sqrt(d.x() * d.x() + d.y() * d.y());
        float x = d.y();
        float y = -d.x();
        return vec4(x, y, 0.0f, 0.0f) / D;
    }

    __host__ __device__ vec4 getVertical(const vec4& d) {
        float z = std::sqrt(d.x() * d.x() + d.y() * d.y());
        float x = -d.z() * d.x() / z;
        float y = -d.z() * d.y() / z;
        return vec4(x, y, z, 0.0f) / d.length();
    }

public:
    __host__ __device__ camera(
        const ray& viewRay,
        float aspect,
        float matrixScale,
        float matrixOffset,
        float focus,
        float apertura) :
        viewRay(viewRay),
        matrixScale(matrixScale),
        matrixOffset(matrixOffset),
        focus(focus),
        apertura(apertura)
    {
        horizontal = aspect * getHorizontal(viewRay.getDirection());
        vertical = getVertical(viewRay.getDirection());
    }

    __host__ __device__ camera(const ray& viewRay, float aspect) : viewRay(viewRay)
    {
        horizontal = aspect * getHorizontal(viewRay.getDirection());
        vertical = getVertical(viewRay.getDirection());
    }

    __device__ ray getPixelRay(float u, float v, curandState* local_rand_state) {
        const float t = focus / (matrixOffset - focus);
        u = matrixScale * t * u + apertura * float(curand_uniform(local_rand_state));
        v = matrixScale * t * v + apertura * float(curand_uniform(local_rand_state));
        return ray(viewRay.point(matrixOffset), t * matrixOffset * viewRay.getDirection() - (u * horizontal + v * vertical));
    }

};
#endif
