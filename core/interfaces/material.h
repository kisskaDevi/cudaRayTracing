#ifndef MATERIALH
#define MATERIALH

#include "ray.h"

class material {
public:
    __device__ virtual vec4 scatter(const ray& r, const vec4& normal, curandState* local_rand_state) const = 0;
    __device__ virtual bool lightFound() const = 0;
};

#endif
