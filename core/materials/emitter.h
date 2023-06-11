#ifndef EMITTERH
#define EMITTERH

#include "material.h"
#include "operations.h"

class emitter : public material {
private:
public:
    __host__ __device__ emitter(){}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __device__ bool lightFound() const override {
        return true;
    }
};

#endif