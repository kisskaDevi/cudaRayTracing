#ifndef MATERIALH
#define MATERIALH

#include "ray.h"

struct properties
{
    float refractiveIndex{ 1.0f };
    float refractProb{ 0.0f };
    float fuzz{ 0.0f };
    float angle{ 0.0f };
};

class material {
public:
    __device__ virtual vec4 scatter(const ray& r, const vec4& normal, const properties& props, curandState* local_rand_state) const = 0;
    __device__ virtual bool lightFound() const = 0;
};

#endif
