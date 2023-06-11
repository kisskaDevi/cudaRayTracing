#include "triangle.h"

__device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    bool result = false;

    const vec4 a = r.getOrigin() - v1.point;
    const vec4 b = v2.point - v1.point;
    const vec4 c = v0.point - v1.point;
    const vec4 d = r.getDirection();

    const float det = det3(d, -b, -c);
    if (det != 0.0f) {
        const float t = det3(-a, -b, -c) / det;
        const float u = det3(d, -a, -c) / det;
        const float v = det3(d, -b, -a) / det;

        result = (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) && (t < tMax&& t > tMin);
        if (result) {
            rec.t = t;
            rec.point = r.point(rec.t);
            rec.normal = normal(v * v0.normal + u * v2.normal + (1 - u - v) * v1.normal);
            rec.color = v * v0.color + u * v2.color + (1 - u - v) * v1.color;
            rec.mat = matptr;
        }
    }

    return result;
}