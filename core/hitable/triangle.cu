#include "triangle.h"

__device__ bool triangle::hit(const ray& r, float tMin, float tMax, hitRecord& rec) const {
    bool result = false;

    const vec4 a = r.getOrigin() - v1;
    const vec4 b = v2 - v1;
    const vec4 c = v0 - v1;
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
            rec.normal = normal(v * n0 + u * n2 + (1 - u - v) * n1);
            rec.mat = matptr;
        }
    }

    return result;
}