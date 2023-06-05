#ifndef LAMBERTIANH
#define LAMBERTIANH

#include "material.h"

class lambertian : public material {
private:
    vec4 albedo{ 0.0f, 0.0f, 0.0f, 0.0f };
    float angle{ pi };
public:
    __host__ __device__ lambertian(const vec4& a) : albedo(a) {}
    __device__ virtual vec4 scatter(const ray& r, const vec4& norm, curandState* local_rand_state) const override {
        vec4 scattered = random_in_unit_sphere(norm, angle, local_rand_state);
        return (dot(norm, scattered) > 0.0f ? 1.0f : -1.0f) * scattered;
    }
    __device__ virtual vec4 getAlbedo() const override {
        return albedo;
    }
    __device__ bool lightFound() const override {
        return false;
    }
};

#endif