#ifndef TRIANGLE
#define TRIANGLE

#include "hitable.h"
#include "buffer.h"

namespace {
    __device__ float det3(const vec4& a, const vec4& b, const vec4& c) {
        return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
            (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
    }
}

struct alignas(64) vertex {
    vec4 point{0.0f, 0.0f, 0.0f, 1.0f};
    vec4 normal{ 0.0f, 0.0f, 0.0f, 0.0f };
    vec4 color{ 0.0f, 0.0f, 0.0f, 0.0f };
    __host__ __device__ vertex() {}
    __host__ __device__ vertex(vec4 point, vec4 normal, vec4 color):
        point(point), normal(normal), color(color)
    {}
};

class alignas(64) triangle : public hitable {
private:
    size_t index0, index1, index2;
    vertex* vertexBuffer{ nullptr };
    material* matptr{ nullptr };

public:
    __host__ __device__ triangle() {}
    __host__ __device__ ~triangle() {}
    __host__ __device__ void destroy() {
        if (matptr) {
            delete matptr;
        }
    }
    __host__ __device__ triangle(const size_t& i0, const size_t& i1, const size_t& i2, vertex* vertexBuffer, material* matptr)
        : index0(i0), index1(i1), index2(i2), vertexBuffer(vertexBuffer), matptr(matptr) {};
    __device__ virtual bool hit(const ray& r, float tMin, float tMax, hitRecord& rec) const override;
};

#endif