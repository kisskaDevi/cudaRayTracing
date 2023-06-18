#include "triangle.h"

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
            rec.mat = matptr;
        }
    }

    return result;
}

__global__ void createTriangle(triangle** tr, const size_t i0, const size_t i1, const size_t i2, vertex* vertexBuffer, material* matptr) {
    *tr = new triangle(i0, i1, i2, vertexBuffer, matptr);
}