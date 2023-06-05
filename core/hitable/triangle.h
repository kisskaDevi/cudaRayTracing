#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"

namespace {
    __device__ float det3(const vec4& a, const vec4& b, const vec4& c) {
        return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
            (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
    }
}

class triangle : public hitable {
private:
    vec4 v0{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4 v1{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4 v2{ 0.0f, 0.0f, 0.0f, 1.0f };
    vec4 n0{ 0.0f, 0.0f, 0.0f, 0.0f };
    vec4 n1{ 0.0f, 0.0f, 0.0f, 0.0f };
    vec4 n2{ 0.0f, 0.0f, 0.0f, 0.0f };
    material* matptr{ nullptr };

public:
    __host__ __device__ triangle() {}
    __host__ __device__ ~triangle() {}
    __host__ __device__ void destroy() {
        if (matptr) {
            delete matptr;
        }
    }
    __host__ __device__ triangle(const vec4& v0, const vec4& v1, const vec4& v2, material* matptr) : v0(v0), v1(v1), v2(v2), matptr(matptr){};
    __host__ __device__ triangle(const vec4& v0, const vec4& v1, const vec4& v2, const vec4& n0, const vec4& n1, const vec4& n2, material* matptr)
        : v0(v0), v1(v1), v2(v2), n0(n0), n1(n1), n2(n2), matptr(matptr) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
};

#endif