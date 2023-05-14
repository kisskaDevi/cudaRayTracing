#ifndef CAMERAH
#define CAMERAH

#include "ray.h"

class camera {
private:
    ray viewRay;
    vec4 horizontal;
    vec4 vertical;
    int maxX;
    int maxY;

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
    __host__ __device__ camera(const ray& viewRay, int maxX, int maxY) : viewRay(viewRay), maxX(maxX), maxY(maxX) {
        horizontal = float(maxX) / float(maxY) * getHorizontal(viewRay.getDirection());
        vertical = getVertical(viewRay.getDirection());
    }

    __host__ __device__ ray getPixelRay(float u, float v) {
        return ray(viewRay.getOrigin(), viewRay.getDirection() + u * horizontal + v * vertical);
    }

};

#endif
