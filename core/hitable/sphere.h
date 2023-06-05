#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "operations.h"

class sphere : public hitable {
private:
    vec4 center{ 0.0f,0.0f, 0.0f, 1.0f };
    float radius{ 0.0f };
    material* matptr{ nullptr };

public:
    __host__ __device__ sphere() {}
    __host__ __device__ ~sphere() {}
    __host__ __device__ void destroy() override {
        if (matptr) {
            delete matptr;
        }
    }
    __host__ __device__ sphere(vec4 cen, float r, material* matptr) : center(cen), radius(r), matptr(matptr) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
};

#endif