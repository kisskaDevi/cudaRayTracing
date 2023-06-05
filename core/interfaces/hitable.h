#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hitRecord
{
    float t{0};
    vec4 point{0.0f, 0.0f, 0.0f, 1.0f };
    vec4 normal{0.0f, 0.0f, 0.0f, 0.0f };
    material* mat{nullptr};
};

class hitable {
public:
    hitable* next{ nullptr };
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const = 0;
    __host__ __device__ virtual void destroy() = 0;
};

#endif