#ifndef EMITTERH
#define EMITTERH

#include "material.h"
#include "operations.h"

class emitter : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
public:
    __host__ __device__ emitter(const vec4& a) : albedo(a) {}
    __device__ vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        return vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return true;
    }
};

#endif