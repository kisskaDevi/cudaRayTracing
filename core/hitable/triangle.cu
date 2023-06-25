#include "triangle.h"

namespace {
    __device__ float det3(const vec4& a, const vec4& b, const vec4& c) {
        return a.x() * b.y() * c.z() + b.x() * c.y() * a.z() + c.x() * a.y() * b.z() -
            (a.x() * c.y() * b.z() + b.x() * a.y() * c.z() + c.x() * b.y() * a.z());
    }
}

__device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    bool result = false;

    const vec4 a = vertexBuffer[index1].point - r.getOrigin();
    const vec4 b = vertexBuffer[index1].point - vertexBuffer[index2].point;
    const vec4 c = vertexBuffer[index1].point - vertexBuffer[index0].point;
    const vec4 d = r.getDirection();

    float det = det3(d, b, c);
    if (det != 0.0f) {
        det = 1.0f / det;
        const float t = det3(a, b, c) * det;
        const float u = det3(d, a, c) * det;
        const float v = det3(d, b, a) * det;

        result = (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) && (t < tMax&& t > tMin);
        if (result) {
            rec.t = t;
            rec.point = r.point(rec.t);
            rec.normal = normal(v * vertexBuffer[index0].normal + u * vertexBuffer[index2].normal + (1 - u - v) * vertexBuffer[index1].normal);
            rec.color = v * vertexBuffer[index0].color + u * vertexBuffer[index2].color + (1 - u - v) * vertexBuffer[index1].color;

            const properties& props0 = vertexBuffer[index0].props;
            const properties& props1 = vertexBuffer[index1].props;
            const properties& props2 = vertexBuffer[index2].props;

            rec.props = {
                v * props0.refractiveIndex + u * props1.refractiveIndex + (1 - u - v) * props2.refractiveIndex,
                v * props0.refractProb + u * props1.refractProb + (1 - u - v) * props2.refractProb,
                v * props0.fuzz + u * props1.fuzz + (1 - u - v) * props2.fuzz,
                v * props0.angle + u * props1.angle + (1 - u - v) * props2.angle
            };

            rec.mat = matptr;
        }
    }

    return result;
}

__global__ void createTriangle(triangle** tr, const size_t i0, const size_t i1, const size_t i2, vertex* vertexBuffer, material* matptr) {
    *tr = new triangle(i0, i1, i2, vertexBuffer, matptr);
}